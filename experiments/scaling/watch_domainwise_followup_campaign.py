#!/usr/bin/env python3
"""Read-only process and artifact watchdog for formal Phase-C/D campaigns.

The formal matrix is global, while processes are host-local.  This watchdog
therefore evaluates artifacts from shared storage and obtains an identity-
checked ``/proc``/NVIDIA snapshot on both physical hosts.  It never launches,
signals, repairs, or removes anything.  In particular, the structural verifier
used while training deliberately does not load checkpoint tensors; the
controller performs a strict checkpoint-loading verification exactly once at
each phase boundary.

``--host-probe`` is the small read-only entry point invoked over SSH.  It is
kept in this file so the local and remote hosts execute precisely the same
identity rules.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from run_domainwise_phase_slot import load_json, safe_component
from supervise_domainwise_phase_host import invoked_scaling_scripts, invoked_script_argument, validate_global_matrix
from verify_domainwise_campaign import atomic_write_json
from verify_domainwise_followup_campaign import verify_phase_campaign


SCHEMA = "domainwise_followup_campaign_watchdog_v1"
HOST_SCHEMA = "domainwise_followup_host_probe_v1"
SUPERVISOR_SCRIPT = "supervise_domainwise_phase_host.py"
WORKER_SCRIPT = "run_domainwise_phase_slot.py"
GUARD_SCRIPT = "guard_domainwise_phase_thermal.py"
TRAINER_SCRIPT = "train_qwen3_pretrain.py"
DATALOADER_PROCESS_NAME = "pt_data_worker"
KNOWN_SCALING_SCRIPTS = {
    SUPERVISOR_SCRIPT,
    WORKER_SCRIPT,
    GUARD_SCRIPT,
    TRAINER_SCRIPT,
    "supervise_domainwise_phase_b_local.py",
    "run_domainwise_phase_b_slot.py",
    "guard_local_secondary_slots_thermal.py",
    "guard_local_primary_slots_thermal.py",
}
OOM_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)
ERROR_PATTERNS = (
    "traceback (most recent call last)",
    "non-finite",
    "nonfinite",
)
NONFINITE_RE = re.compile(r"(?:^|[^a-z0-9_])(?:nan|inf)(?:[^a-z0-9_]|$)", re.I)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def command_value(command: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"expected exactly one {flag}")
    return command[positions[0] + 1]


def contains_exact_script(command: list[str], expected: Path) -> bool:
    expected = expected.resolve()
    token = invoked_script_argument(command)
    if token is None:
        return False
    try:
        candidate = Path(token)
        return candidate.name == expected.name and candidate.resolve() == expected
    except (OSError, RuntimeError, ValueError):
        return False


def proc_info(entry: Path) -> dict[str, Any]:
    if entry.stat().st_uid != os.getuid():
        raise PermissionError(entry)
    raw = (entry / "stat").read_text(encoding="utf-8")
    open_paren = raw.find("(")
    close = raw.rfind(")")
    if open_paren < 0 or close <= open_paren:
        raise ValueError("unparseable stat")
    tail = raw[close + 2 :].split()
    if len(tail) < 20:
        raise ValueError("short stat")
    command = [
        token.decode("utf-8", errors="replace")
        for token in (entry / "cmdline").read_bytes().split(b"\0")
        if token
    ]
    return {
        "pid": int(entry.name),
        "state": tail[0],
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
        "comm": raw[open_paren + 1 : close],
        "cmdline": command,
    }


def all_owned_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            info = proc_info(entry)
        except (FileNotFoundError, OSError, PermissionError, ProcessLookupError, ValueError):
            continue
        if info["state"] not in {"Z", "X", "x"}:
            rows.append(info)
    return rows


def _is_exact_main_trainer(
    candidate: dict[str, Any], worker: dict[str, Any]
) -> bool:
    """Bind one trainer invocation to the exact worker process generation."""
    return bool(
        candidate.get("comm") != DATALOADER_PROCESS_NAME
        and int(candidate["ppid"]) == int(worker["pid"])
        and int(candidate["pgid"]) == int(worker["pid"])
        and int(candidate["sid"]) == int(worker["sid"])
        and int(candidate["start_ticks"]) >= int(worker["start_ticks"])
    )


def _is_exact_dataloader_descendant(
    candidate: dict[str, Any],
    main_trainer: dict[str, Any],
    worker: dict[str, Any],
    candidates_by_pid: dict[int, dict[str, Any]],
) -> bool:
    """Recognize fork-inherited DataLoader argv without weakening ownership.

    PyTorch fork workers retain the trainer's complete ``cmdline``. Every link
    must therefore remain a ``pt_data_worker`` with the same invocation,
    worker PGID/SID, monotonic start time, and an unbroken ancestry chain to
    the one direct trainer child. Anything else remains unclassified so the
    caller can fail closed.
    """
    main_pid = int(main_trainer["pid"])
    worker_pid = int(worker["pid"])
    worker_sid = int(worker["sid"])
    expected_identity = (
        main_trainer.get("run_id"),
        main_trainer.get("attempt_dir"),
        main_trainer.get("cmdline"),
    )
    current = candidate
    visited: set[int] = set()
    while int(current["pid"]) != main_pid:
        current_pid = int(current["pid"])
        if current_pid in visited:
            return False
        visited.add(current_pid)
        if (
            current.get("comm") != DATALOADER_PROCESS_NAME
            or int(current["pgid"]) != worker_pid
            or int(current["sid"]) != worker_sid
            or (
                current.get("run_id"),
                current.get("attempt_dir"),
                current.get("cmdline"),
            )
            != expected_identity
        ):
            return False
        parent = candidates_by_pid.get(int(current["ppid"]))
        if (
            parent is None
            or int(parent["start_ticks"]) > int(current["start_ticks"])
        ):
            return False
        current = parent
    return True


def classify_trainer_candidates(
    candidates: list[dict[str, Any]],
    workers_by_pid: dict[int, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    list[str],
]:
    """Separate one main trainer per worker from inherited DataLoader forks."""
    candidates_by_pid = {int(item["pid"]): item for item in candidates}
    trainers_by_worker: dict[int, list[dict[str, Any]]] = {
        pid: [] for pid in workers_by_pid
    }
    main_trainers: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    issues: list[str] = []

    for candidate in candidates:
        worker = workers_by_pid.get(int(candidate["pgid"]))
        if worker is not None and _is_exact_main_trainer(candidate, worker):
            main_trainers.append(candidate)
            trainers_by_worker[int(worker["pid"])].append(candidate)
        else:
            remaining.append(candidate)

    for worker_pid, worker_trainers in trainers_by_worker.items():
        if len(worker_trainers) > 1:
            issues.append(
                "multiple trainers share worker process group "
                f"pgid={worker_pid}: "
                f"{sorted(int(item['pid']) for item in worker_trainers)}"
            )

    dataloader_descendants: list[dict[str, Any]] = []
    for candidate in remaining:
        pgid = int(candidate["pgid"])
        worker = workers_by_pid.get(pgid)
        if worker is None:
            issues.append(f"orphan trainer pid={candidate['pid']} pgid={pgid}")
            continue
        worker_trainers = trainers_by_worker[pgid]
        main_trainer = worker_trainers[0] if len(worker_trainers) == 1 else None
        if main_trainer is not None and _is_exact_dataloader_descendant(
            candidate, main_trainer, worker, candidates_by_pid
        ):
            dataloader_descendants.append(
                {**candidate, "main_trainer_pid": int(main_trainer["pid"])}
            )
            continue
        issues.append(
            "trainer is not the direct child/session member of its exact "
            "worker generation or an exact DataLoader descendant "
            f"pid={candidate['pid']} pgid={pgid}"
        )

    return main_trainers, dataloader_descendants, trainers_by_worker, issues


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory failed: {result.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 6)]
        if len(fields) != 7:
            raise ValueError(f"cannot parse GPU inventory: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "temperature_c": float(fields[5]),
                "utilization_pct": float(fields[6]),
            }
        )
    return rows


def compute_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute query failed: {result.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(f"cannot parse compute process: {line!r}")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": int(fields[3]),
            }
        )
    return rows


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                last = value
    return last


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def thermal_event_timestamp(event: dict[str, Any]) -> float | None:
    value = event.get("time")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def worker_generation_key(
    value: dict[str, Any], *, slot_field: str
) -> tuple[Any, ...] | None:
    slot = value.get(slot_field)
    if not isinstance(slot, str) or not slot:
        return None
    fields: list[Any] = [slot]
    for name in ("pid", "pgid", "sid", "start_ticks"):
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool):
            return None
        fields.append(item)
    return tuple(fields)


def guard_generation_key(value: dict[str, Any]) -> tuple[int, int, int, int] | None:
    fields: list[int] = []
    for name in ("pid", "pgid", "sid", "start_ticks"):
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool):
            return None
        fields.append(item)
    return fields[0], fields[1], fields[2], fields[3]


def intersect_intervals(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            intersections.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return intersections


def verified_thermal_pause_state(
    event_path: Path,
    workers: list[dict[str, Any]],
    guards: list[dict[str, Any]],
    *,
    phase: str,
    host_id: str,
    since: float,
    until: float,
) -> tuple[float, bool]:
    """Return common closed pause time and exact ownership of a current pause."""
    worker_keys = [
        worker_generation_key(worker, slot_field="worker_slot")
        for worker in workers
    ]
    if (
        not workers
        or any(key is None for key in worker_keys)
        or len(set(worker_keys)) != len(workers)
        or until < since
    ):
        return 0.0, False
    live_keys = {key for key in worker_keys if key is not None}
    live_guard_keys = {
        key
        for guard in guards
        if (key := guard_generation_key(guard)) is not None
    }
    intervals = {key: [] for key in live_keys}
    stopped_at: dict[tuple[Any, ...], tuple[float, tuple[int, int, int, int]]] = {}
    resume_intent_at: dict[
        tuple[Any, ...], tuple[float, tuple[int, int, int, int]]
    ] = {}
    events = sorted(
        read_jsonl(event_path),
        key=lambda event: thermal_event_timestamp(event) or float("-inf"),
    )
    for event in events:
        identity = event.get("identity")
        if not isinstance(identity, dict):
            continue
        if identity.get("phase") != phase or identity.get("host_id") != host_id:
            continue
        key = worker_generation_key(identity, slot_field="slot")
        if key not in live_keys:
            continue
        event_guard = event.get("guard_identity")
        if not isinstance(event_guard, dict):
            continue
        event_guard_key = guard_generation_key(event_guard)
        if event_guard_key is None:
            continue
        timestamp = thermal_event_timestamp(event)
        if timestamp is None:
            continue
        if event.get("event") == "sigstop_sent":
            stopped_at[key] = (timestamp, event_guard_key)
            resume_intent_at.pop(key, None)
        elif event.get("event") == "sigcont_intent" and key in stopped_at:
            _, stop_guard_key = stopped_at[key]
            if event_guard_key == stop_guard_key:
                resume_intent_at[key] = (timestamp, event_guard_key)
        elif event.get("event") == "sigcont_sent" and key in stopped_at:
            start, stop_guard_key = stopped_at[key]
            if event_guard_key != stop_guard_key:
                continue
            stopped_at.pop(key)
            resume_intent_at.pop(key, None)
            if timestamp >= start:
                clipped_start = max(start, since)
                clipped_end = min(timestamp, until)
                if clipped_end > clipped_start:
                    intervals[key].append((clipped_start, clipped_end))

    # A durable resume intent followed by observation of this exact worker
    # generation in a running/sleeping state closes the otherwise ambiguous
    # crash window between SIGCONT and ``sigcont_sent``.  Credit ends at the
    # immutable intent timestamp, never at observation time: otherwise each
    # watchdog poll would extend the pause and permanently conceal a real hang
    # after resume.  Do not infer a close from intent alone when still stopped.
    states = {
        worker_generation_key(worker, slot_field="worker_slot"): worker.get("state")
        for worker in workers
    }
    for key, (start, stop_guard_key) in list(stopped_at.items()):
        intent = resume_intent_at.get(key)
        if (
            intent is None
            or intent[1] != stop_guard_key
            or intent[0] < start
            or intent[0] > until
            or states.get(key) in {"T", "t"}
        ):
            continue
        clipped_start = max(start, since)
        clipped_end = min(intent[0], until)
        if clipped_end > clipped_start:
            intervals[key].append((clipped_start, clipped_end))
        stopped_at.pop(key)
        resume_intent_at.pop(key, None)

    common = intervals[next(iter(live_keys))]
    for key in live_keys:
        if not common:
            break
        common = intersect_intervals(common, intervals[key])
    paused_seconds = sum(end - start for start, end in common)
    current_pause_owned = bool(
        live_keys
        and all(states[key] in {"T", "t"} for key in live_keys)
        and all(
            key in stopped_at and stopped_at[key][1] in live_guard_keys
            for key in live_keys
        )
    )
    return paused_seconds, current_pause_owned


def newest_file_age(paths: Iterable[Path], *, current_time: float) -> float | None:
    """Return the age of the newest durable heartbeat without reading files."""
    newest: float | None = None
    for path in paths:
        try:
            age = max(0.0, current_time - path.stat().st_mtime)
        except OSError:
            continue
        newest = age if newest is None else min(newest, age)
    return newest


def newest_generation_file_age(
    paths: Iterable[Path],
    *,
    current_time: float,
    generation_started_at: float,
) -> float | None:
    """Return progress age without borrowing artifacts from an older worker.

    Worker status/console paths are reused across supervisor restarts.  Their
    mtimes are useful for the current generation only when they are no older
    than the exact ``/proc`` start time of that worker.  Attempt paths supplied
    by a live trainer are unique, but applying the same boundary to every path
    keeps the rule fail-closed and uniform.
    """
    eligible: list[Path] = []
    for path in paths:
        try:
            if path.stat().st_mtime < generation_started_at:
                continue
        except OSError:
            continue
        eligible.append(path)
    return newest_file_age(eligible, current_time=current_time)


def classify_worker_runtime_progress(
    worker: dict[str, Any],
    trainers: list[dict[str, Any]],
    *,
    runtime_ages_seconds: list[float],
    nonthermal_progress_age_seconds: float | None,
    thermal_heartbeat_age_seconds: float | None,
    verified_thermal_pause_seconds: float,
    current_pause_owned: bool,
    stale_seconds: float,
) -> dict[str, Any]:
    """Classify one exact worker generation and its live attempt, if any."""
    compute_tree = [worker, *trainers]
    all_compute_stopped = all(
        proc.get("state") in {"T", "t"} for proc in compute_tree
    )
    thermal_pause_verified = bool(
        all_compute_stopped
        and current_pause_owned
        and thermal_heartbeat_age_seconds is not None
        and thermal_heartbeat_age_seconds <= stale_seconds
    )
    youngest_runtime_age = (
        min(runtime_ages_seconds) if runtime_ages_seconds else None
    )
    active_age = (
        max(
            0.0,
            nonthermal_progress_age_seconds
            - max(0.0, verified_thermal_pause_seconds),
        )
        if nonthermal_progress_age_seconds is not None
        else None
    )
    outside_startup_grace = bool(
        youngest_runtime_age is not None and youngest_runtime_age > stale_seconds
    )
    return {
        "runtime_youngest_age_seconds": youngest_runtime_age,
        "thermal_pause_verified": thermal_pause_verified,
        "nonthermal_active_progress_age_seconds": active_age,
        "progress_stalled": bool(
            outside_startup_grace
            and (active_age is None or active_age > stale_seconds)
            and not thermal_pause_verified
        ),
    }


def classify_owned_runtime_progress(
    roles: dict[str, list[dict[str, Any]]],
    *,
    runtime_ages_seconds: list[float],
    nonthermal_progress_age_seconds: float | None,
    thermal_heartbeat_age_seconds: float | None,
    verified_thermal_pause_seconds: float = 0.0,
    current_pause_owned: bool = False,
    stale_seconds: float,
) -> dict[str, Any]:
    """Classify a live host without treating thermal samples as work progress.

    A supervisor, worker, or guard can hang before a trainer is born, and can
    also remain after every artifact is complete.  Conversely, a thermal guard
    deliberately stops whole worker groups; its fresh samples prove that this
    is a managed pause and must not be mistaken for a hang.
    """
    runtime = [proc for values in roles.values() for proc in values]
    compute_tree = [
        *roles.get("workers", []),
        *roles.get("trainers", []),
    ]
    all_compute_stopped = bool(compute_tree) and all(
        proc.get("state") in {"T", "t"} for proc in compute_tree
    )
    thermal_pause_verified = bool(
        all_compute_stopped
        and roles.get("guards")
        and current_pause_owned
        and thermal_heartbeat_age_seconds is not None
        and thermal_heartbeat_age_seconds <= stale_seconds
    )
    youngest_runtime_age = (
        min(runtime_ages_seconds) if runtime_ages_seconds else None
    )
    outside_startup_grace = bool(
        runtime
        and youngest_runtime_age is not None
        and youngest_runtime_age > stale_seconds
    )
    nonthermal_active_progress_age_seconds = (
        max(
            0.0,
            nonthermal_progress_age_seconds
            - max(0.0, verified_thermal_pause_seconds),
        )
        if nonthermal_progress_age_seconds is not None
        else None
    )
    progress_is_stale = (
        nonthermal_active_progress_age_seconds is None
        or nonthermal_active_progress_age_seconds > stale_seconds
    )
    return {
        "runtime_present": bool(runtime),
        "runtime_youngest_age_seconds": youngest_runtime_age,
        "thermal_pause_verified": thermal_pause_verified,
        "nonthermal_active_progress_age_seconds": nonthermal_active_progress_age_seconds,
        "progress_stalled": bool(
            outside_startup_grace
            and progress_is_stale
            and not thermal_pause_verified
        ),
    }


def validate_host_completion(
    path: Path,
    *,
    matrix: dict[str, Any],
    matrix_path: Path,
    phase: str,
    host_id: str,
    expected_run_ids: set[str],
) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, []
    issues: list[str] = []
    try:
        value = load_json(path)
        if not isinstance(value, dict):
            raise ValueError("marker is not an object")
        expected = {
            "schema_version": "domainwise_phase_host_completion_v1",
            "status": "complete",
            "phase": phase,
            "host_id": host_id,
            "matrix_json": str(matrix_path),
            "matrix_generation_id": matrix["metadata"]["generation_id"],
            "matrix_semantic_fingerprint": matrix["metadata"]["semantic_fingerprint"]["value"],
            "rows": len(expected_run_ids),
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                issues.append(f"host completion {key} mismatch")
        completed = value.get("completed_runs")
        if not isinstance(completed, list):
            issues.append("host completion completed_runs is not a list")
        else:
            actual = [str(item.get("run_id")) for item in completed if isinstance(item, dict)]
            if len(actual) != len(expected_run_ids) or set(actual) != expected_run_ids:
                issues.append("host completion run set mismatch")
    except Exception as exc:
        issues.append(f"host completion unreadable: {type(exc).__name__}: {exc}")
    return not issues, issues


def deferred_admission_wait(
    value: dict[str, Any],
    *,
    supervisor: dict[str, Any],
    pid_record: dict[str, Any],
    matrix_path: Path,
    matrix_generation_id: str,
    phase: str,
    host_id: str,
    slot_gpus: dict[str, int],
    probes: dict[int, dict[str, Any]],
    completed_slots: set[str],
    roles: dict[str, list[dict[str, Any]]],
    current_time: float,
    stale_seconds: float,
    predecessor_waits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate generation-bound admission; never hide an active worker stall."""
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "domainwise_gpu_admission_state_v1"
        or pid_record.get("admission_policy") != "defer_unavailable_gpus_v1"
        or value.get("phase") != phase
        or value.get("host_id") != host_id
        or canonical(value.get("matrix_json", "")) != matrix_path
        or value.get("matrix_generation_id") != matrix_generation_id
        or value.get("generation_nonce") != pid_record.get("generation_nonce")
        or any(int(value.get(key, -1)) != int(supervisor[key]) for key in ("pid", "start_ticks"))
        or not isinstance(value.get("slots"), dict)
        or set(value["slots"]) != set(slot_gpus)
    ):
        raise ValueError("deferred admission record identity/coverage mismatch")
    timestamp = datetime.fromisoformat(str(value.get("updated_at", "")))
    if timestamp.tzinfo is None:
        raise ValueError("deferred admission timestamp has no timezone")
    age = current_time - timestamp.timestamp()
    if age < -5 or age > stale_seconds:
        raise ValueError("deferred admission heartbeat is stale or future-dated")
    blocked: list[str] = []
    remaining: list[str] = []
    for slot, gpu in slot_gpus.items():
        item = value["slots"][slot]
        if not isinstance(item, dict) or int(item.get("gpu", -1)) != gpu:
            raise ValueError(f"deferred slot GPU mismatch: {slot}")
        status = item.get("status")
        if status not in {"pending_foreign", "pending_unavailable", "pending_predecessor", "ready", "running", "complete", "failed"}:
            raise ValueError(f"unknown deferred admission status: {slot}")
        if slot in completed_slots:
            continue
        remaining.append(slot)
        probe = probes[gpu]
        if status == "pending_predecessor" and (predecessor_waits or {}).get(slot, {}).get("ready") is False:
            blocked.append(slot)
        elif status == "pending_foreign" and probe["status"] == "ok" and probe["compute"]:
            blocked.append(slot)
        elif status == "pending_unavailable" and (
            probe["status"] != "ok"
            or int(probe.get("inventory", {}).get("memory_used_mib", 0)) > 256
        ):
            blocked.append(slot)
    waiting = bool(
        remaining
        and set(blocked) == set(remaining)
        and len(roles.get("supervisors", [])) == 1
        and len(roles.get("guards", [])) == 1
        and not roles.get("workers")
        and not roles.get("trainers")
    )
    return {"verified_wait": waiting, "age_seconds": max(0.0, age), "blocked_slots": blocked}


def host_probe(
    matrix_path: Path,
    phase: str,
    host_id: str,
    *,
    stale_seconds: float,
    local_execution_plan: Path | None = None,
) -> dict[str, Any]:
    """Return one host's authoritative runtime snapshot without writing."""
    matrix_path = canonical(matrix_path)
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError("matrix root is not an object")
    host_rows, slots, devices, root = validate_global_matrix(matrix, phase, host_id)
    execution_plan = None
    if local_execution_plan is not None:
        from domainwise_local_execution import load_local_execution_plan

        execution_plan = load_local_execution_plan(matrix_path, local_execution_plan)
        if phase != "c" or host_id != "remote":
            raise ValueError("local execution placement only applies to logical remote Phase C")
    script_dir = Path(__file__).resolve().parent
    expected_scripts = {
        "supervisor": script_dir / SUPERVISOR_SCRIPT,
        "worker": script_dir / WORKER_SCRIPT,
        "guard": script_dir / GUARD_SCRIPT,
        "trainer": script_dir.parents[1] / "cli" / "pretrain" / TRAINER_SCRIPT,
    }
    rows_by_run_dir = {
        canonical(row["out_dir"]): row
        for row in host_rows
    }
    process_rows = all_owned_processes()
    peer_pids: set[int] = set()
    predecessor_waits: dict[str, dict[str, Any]] = {}
    if execution_plan is not None:
        from domainwise_local_execution import local_predecessor_ready, peer_owned_processes_allowed

        peer_pids = peer_owned_processes_allowed(matrix_path, execution_plan, process_rows)
        predecessor_waits = {
            slot: local_predecessor_ready(matrix, slot, process_rows=process_rows)
            for slot in slots
        }
    scaling_processes = [
        item
        for item in process_rows
        if invoked_scaling_scripts(item["cmdline"], KNOWN_SCALING_SCRIPTS)
    ]
    roles: dict[str, list[dict[str, Any]]] = {
        "supervisors": [],
        "workers": [],
        "guards": [],
        "trainers": [],
    }
    trainer_candidates: list[dict[str, Any]] = []
    issues: list[str] = []
    warnings: list[str] = []

    for proc in process_rows:
        command = proc["cmdline"]
        role: str | None = None
        if contains_exact_script(command, expected_scripts["supervisor"]):
            role = "supervisors"
        elif contains_exact_script(command, expected_scripts["worker"]):
            role = "workers"
        elif contains_exact_script(command, expected_scripts["guard"]):
            role = "guards"
        elif contains_exact_script(command, expected_scripts["trainer"]):
            role = "trainers"
        if role is None:
            continue
        try:
            if role == "trainers":
                attempt = canonical(command_value(command, "--out-dir"))
                row = rows_by_run_dir.get(attempt.parent)
                if row is None or not attempt.name.startswith("attempt_"):
                    continue
                proc = {**proc, "run_id": row["run_id"], "attempt_dir": str(attempt)}
            else:
                if canonical(command_value(command, "--matrix-json")) != matrix_path:
                    continue
                if command_value(command, "--host-id") != host_id:
                    continue
                if execution_plan is not None and canonical(command_value(command, "--local-execution-plan")) != canonical(local_execution_plan):
                    raise ValueError("local execution plan CLI mismatch")
                if role in {"supervisors", "guards"} and command_value(command, "--phase") != phase:
                    continue
                if role == "workers":
                    slot = command_value(command, "--worker-slot")
                    if slot not in slots:
                        issues.append(f"worker pid={proc['pid']} declares unknown slot {slot}")
                    proc = {**proc, "worker_slot": slot}
            if role == "trainers":
                trainer_candidates.append(proc)
            else:
                roles[role].append(proc)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            issues.append(
                f"cannot validate {role[:-1]} pid={proc['pid']}: {type(exc).__name__}: {exc}"
            )

    supervisors = roles["supervisors"]
    supervisor_pids = {int(item["pid"]) for item in supervisors}
    if len(supervisors) > 1:
        issues.append(f"multiple matching supervisors are live: {sorted(supervisor_pids)}")
    for proc in supervisors:
        if (
            int(proc["ppid"]) != 1
            or int(proc["pgid"]) != int(proc["pid"])
            or int(proc["sid"]) != int(proc["pid"])
        ):
            issues.append(f"supervisor identity/session mismatch pid={proc['pid']}")

    workers_by_pid = {int(item["pid"]): item for item in roles["workers"]}
    seen_slots: dict[str, int] = {}
    for proc in roles["workers"]:
        slot = str(proc.get("worker_slot"))
        if slot in seen_slots:
            issues.append(f"duplicate live worker slot {slot}: {seen_slots[slot]},{proc['pid']}")
        seen_slots[slot] = int(proc["pid"])
        if (
            int(proc["pgid"]) != int(proc["pid"])
            or int(proc["sid"]) != int(proc["pid"])
            or int(proc["ppid"]) not in supervisor_pids
        ):
            issues.append(f"orphan/mis-grouped worker pid={proc['pid']} slot={slot}")

    for proc in roles["guards"]:
        if int(proc["pgid"]) != int(proc["pid"]) or int(proc["sid"]) != int(proc["pid"]):
            issues.append(f"thermal guard is not a session/group leader pid={proc['pid']}")
        if supervisor_pids and int(proc["ppid"]) not in supervisor_pids:
            issues.append(f"thermal guard parent mismatch pid={proc['pid']}")

    (
        main_trainers,
        trainer_dataloader_descendants,
        trainers_by_worker,
        trainer_issues,
    ) = classify_trainer_candidates(trainer_candidates, workers_by_pid)
    roles["trainers"].extend(main_trainers)
    issues.extend(trainer_issues)
    for proc in roles["trainers"]:
        if proc["state"] in {"T", "t"} and not roles["guards"]:
            issues.append(f"stopped trainer has no matching thermal guard pid={proc['pid']}")
    for proc in roles["workers"]:
        if proc["state"] in {"T", "t"} and not roles["guards"]:
            issues.append(f"stopped worker has no matching thermal guard pid={proc['pid']}")

    per_gpu_admission = bool(
        phase == "c" and host_id == "remote"
        and all(int(device["concurrent_slots"]) == 1 for device in devices.values())
    )
    gpu_probes: dict[int, dict[str, Any]] = {}
    gpu_query_errors: dict[int, str] = {}
    admission_ready_gpus: list[int] = []
    if per_gpu_admission:
        from domainwise_gpu_admission import probe_gpus

        gpu_probes = probe_gpus(sorted(devices))
        inventory = [value["inventory"] for value in gpu_probes.values() if value["status"] == "ok"]
        raw_compute = [item for value in gpu_probes.values() if value["status"] == "ok" for item in value["compute"]]
        for gpu, value in gpu_probes.items():
            if value["status"] != "ok":
                gpu_query_errors[gpu] = str(value.get("error"))
            elif (
                not value["compute"]
                and int(value["inventory"]["memory_used_mib"]) <= 256
                and int(value["inventory"]["memory_total_mib"]) >= int(devices[gpu]["memory_gib"]) * 1024 - 1536
            ):
                admission_ready_gpus.append(gpu)
    else:
        inventory = gpu_inventory()
        raw_compute = compute_processes()
    inventory_by_index = {int(item["index"]): item for item in inventory}
    selected_uuids = {
        str(inventory_by_index[gpu]["uuid"])
        for gpu in devices
        if gpu in inventory_by_index
    }
    if len(selected_uuids) != len(devices) and not per_gpu_admission:
        issues.append("host GPU inventory does not cover the frozen devices")
    if gpu_query_errors:
        warnings.append(f"unavailable frozen GPUs remain deferred, not idle: {gpu_query_errors}")
        active_slots = {str(item.get("worker_slot")) for item in roles["workers"]}
        active_run_ids = {str(item["run_id"]) for item in roles["trainers"]}
        active_gpus = {
            int(row["assigned_gpu"]) for row in host_rows
            if row["worker_slot"] in active_slots or row["run_id"] in active_run_ids
        }
        if active_gpus & set(gpu_query_errors):
            # A failed observation is not evidence that the exact runtime died.
            # Its independent guard retains the bounded fail-closed protection;
            # the controller must reobserve, not clean on one NVML timeout.
            raise RuntimeError(f"active campaign GPU telemetry unavailable: {sorted(active_gpus & set(gpu_query_errors))}")
    compute = [row for row in raw_compute if row["gpu_uuid"] in selected_uuids]
    compute_pids = {int(item["pid"]) for item in compute}
    trainer_pids = {int(item["pid"]) for item in roles["trainers"]}
    foreign_compute = [item for item in compute if int(item["pid"]) not in trainer_pids]
    dataloader_compute_pids = sorted(
        int(item["pid"])
        for item in trainer_dataloader_descendants
        if int(item["pid"]) in compute_pids
    )
    if dataloader_compute_pids:
        issues.append(
            "trainer DataLoader descendants unexpectedly own selected-GPU compute: "
            f"{dataloader_compute_pids}"
        )
    if foreign_compute:
        warnings.append(f"selected GPUs contain non-campaign compute: {foreign_compute}")
    for proc in roles["trainers"]:
        if int(proc["pid"]) not in compute_pids:
            # A trainer can be in CPU-side loading for a short period.
            warnings.append(f"trainer pid={proc['pid']} is not yet visible to nvidia-smi")

    pid_path = root / "supervisors" / host_id / "supervisor.pid.json"
    pid_record: dict[str, Any] | None = None
    if pid_path.is_file():
        try:
            value = load_json(pid_path)
            if not isinstance(value, dict):
                raise ValueError("PID record is not an object")
            pid_record = value
            matching = [item for item in supervisors if int(item["pid"]) == int(value["pid"])]
            if len(matching) != 1:
                issues.append("supervisor PID record has no unique live generation")
            else:
                proc = matching[0]
                for key in ("pid", "ppid", "pgid", "sid", "start_ticks"):
                    if int(value[key]) != int(proc[key]):
                        issues.append(f"supervisor PID record {key} mismatch")
                if (
                    value.get("phase") != phase
                    or value.get("host_id") != host_id
                    or canonical(value.get("matrix_json")) != matrix_path
                    or value.get("matrix_generation_id") != matrix["metadata"]["generation_id"]
                    or value.get("matrix_semantic_fingerprint")
                    != matrix["metadata"]["semantic_fingerprint"]["value"]
                ):
                    issues.append("supervisor PID record semantic identity mismatch")
        except Exception as exc:
            issues.append(f"invalid supervisor PID record: {type(exc).__name__}: {exc}")
    elif supervisors:
        warnings.append("supervisor is live before/without its PID record")

    marker_path = root / "host_completion" / f"{host_id}.json"
    marker_valid, marker_issues = validate_host_completion(
        marker_path,
        matrix=matrix,
        matrix_path=matrix_path,
        phase=phase,
        host_id=host_id,
        expected_run_ids={str(row["run_id"]) for row in host_rows},
    )
    issues.extend(marker_issues)

    status_path = root / "supervisors" / host_id / "status.jsonl"
    last_status = read_last_jsonl(status_path)
    if (
        last_status
        and last_status.get("event") in {
            "supervisor_failed",
            "worker_restart_exhausted",
            "worker_group_cleanup_failed",
            "final_worker_group_cleanup_failed",
            "thermal_guard_unexpected_exit",
            "host_completion_revalidation_failed",
        }
    ):
        warnings.append(f"latest supervisor status is terminal/problematic: {last_status}")

    retry_states: list[dict[str, Any]] = []
    exhausted: list[str] = []
    malformed_retry: list[str] = []
    oom_failures = 0
    for row in host_rows:
        run_dir = canonical(row["out_dir"])
        retry_path = run_dir / "retry_state.json"
        if not retry_path.is_file():
            continue
        try:
            value = load_json(retry_path)
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != "domainwise_phase_retry_state_v1"
                or value.get("run_id") != row["run_id"]
                or int(value.get("effective_batch_records", -1))
                != int(row["effective_batch_records"])
                or int(value.get("current_microbatch", 0))
                * int(value.get("current_grad_accum", 0))
                != int(row["effective_batch_records"])
            ):
                raise ValueError("retry ledger identity/effective batch mismatch")
            state = {
                "run_id": row["run_id"],
                "status": value.get("status"),
                "total_attempts_started": int(value.get("total_attempts_started", 0)),
                "oom_failures": int(value.get("oom_failures", 0)),
                "fallback_level": int(value.get("fallback_level", 0)),
            }
            retry_states.append(state)
            oom_failures += state["oom_failures"]
            if state["status"] == "attempts_exhausted":
                exhausted.append(str(row["run_id"]))
        except Exception as exc:
            malformed_retry.append(f"{row['run_id']}: {type(exc).__name__}: {exc}")
    if exhausted:
        issues.append(f"run retry ledgers are exhausted: {exhausted}")
    if malformed_retry:
        issues.append(f"malformed retry ledgers: {malformed_retry}")

    newest_log_age: float | None = None
    log_findings: list[dict[str, Any]] = []
    current_time = time.time()
    log_paths = list((root / "workers" / host_id).glob("*/worker_console.log"))
    progress_paths: list[Path] = [status_path, marker_path]
    progress_paths.extend(
        (root / "workers" / host_id).glob("*/status.jsonl")
    )
    progress_paths.extend(log_paths)
    for row in host_rows:
        run_dir = canonical(row["out_dir"])
        attempt_logs = list(run_dir.glob("attempt_*/console.log"))
        log_paths.extend(attempt_logs)
        progress_paths.extend(attempt_logs)
        progress_paths.extend(run_dir.glob("attempt_*/train_log.jsonl"))
        progress_paths.extend(run_dir.glob("attempt_*/summary.json"))
        progress_paths.extend(run_dir.glob("attempt_*/attempt_contract.json"))
        progress_paths.extend([run_dir / "retry_state.json", run_dir / "complete.json"])
    for path in log_paths:
        try:
            age = max(0.0, current_time - path.stat().st_mtime)
            newest_log_age = age if newest_log_age is None else min(newest_log_age, age)
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 512 * 1024))
                tail = handle.read().decode("utf-8", errors="replace").lower()
            kinds: list[str] = []
            if any(pattern in tail for pattern in OOM_PATTERNS):
                kinds.append("oom")
            if any(pattern in tail for pattern in ERROR_PATTERNS) or NONFINITE_RE.search(tail):
                kinds.append("error_or_nonfinite")
            if kinds:
                log_findings.append({"path": str(path), "kinds": kinds, "age_seconds": age})
        except OSError:
            continue
    nonthermal_progress_age = newest_file_age(
        progress_paths, current_time=current_time
    )
    thermal_event_path = root / "thermal" / host_id / "events.jsonl"
    thermal_heartbeat_age = newest_file_age(
        [thermal_event_path],
        current_time=current_time,
    )
    verified_pause_seconds, current_pause_owned = verified_thermal_pause_state(
        thermal_event_path,
        roles["workers"],
        roles["guards"],
        phase=phase,
        host_id=host_id,
        since=(
            current_time - nonthermal_progress_age
            if nonthermal_progress_age is not None
            else current_time
        ),
        until=current_time,
    )
    runtime_age_by_pid: dict[int, float] = {}
    worker_started_at_by_pid: dict[int, float] = {}
    try:
        uptime_seconds = float(
            Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
        )
        ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
        for values in roles.values():
            for proc in values:
                age = max(
                    0.0,
                    uptime_seconds - float(proc["start_ticks"]) / ticks_per_second,
                )
                runtime_age_by_pid[int(proc["pid"])] = age
        runtime_ages = list(runtime_age_by_pid.values())
        worker_started_at_by_pid = {
            int(worker["pid"]): current_time - runtime_age_by_pid[int(worker["pid"])]
            for worker in roles["workers"]
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        runtime_ages = []
        if any(roles.values()):
            issues.append(
                f"cannot determine owned runtime age: {type(exc).__name__}: {exc}"
            )
    progress = classify_owned_runtime_progress(
        roles,
        runtime_ages_seconds=runtime_ages,
        nonthermal_progress_age_seconds=nonthermal_progress_age,
        thermal_heartbeat_age_seconds=thermal_heartbeat_age,
        verified_thermal_pause_seconds=verified_pause_seconds,
        current_pause_owned=current_pause_owned,
        stale_seconds=stale_seconds,
    )
    admission_state: dict[str, Any] | None = None
    if pid_record and pid_record.get("admission_policy") == "defer_unavailable_gpus_v1":
        admission_path = root / "supervisors" / host_id / "admission.json"
        try:
            if not per_gpu_admission or len(supervisors) != 1:
                raise ValueError("deferred admission requires one remote single-slot supervisor")
            if "--defer-unavailable-gpus" not in supervisors[0]["cmdline"]:
                raise ValueError("supervisor CLI does not match admission policy")
            slot_gpus = {str(row["worker_slot"]): int(row["assigned_gpu"]) for row in host_rows}
            completed_slots = {
                slot for slot in slots
                if all((canonical(row["out_dir"]) / "complete.json").is_file() for row in host_rows if row["worker_slot"] == slot)
            }
            admission_state = deferred_admission_wait(
                load_json(admission_path), supervisor=supervisors[0], pid_record=pid_record,
                matrix_path=matrix_path, matrix_generation_id=matrix["metadata"]["generation_id"],
                phase=phase, host_id=host_id, slot_gpus=slot_gpus, probes=gpu_probes,
                completed_slots=completed_slots, roles=roles, current_time=current_time,
                stale_seconds=stale_seconds,
                predecessor_waits=predecessor_waits,
            )
            if admission_state["verified_wait"]:
                progress["progress_stalled"] = False
        except FileNotFoundError:
            if runtime_ages and min(runtime_ages) > stale_seconds:
                issues.append("deferred supervisor has no admission record after startup grace")
            else:
                warnings.append("deferred supervisor is publishing initial admission state")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            issues.append(f"invalid deferred admission state: {type(exc).__name__}: {exc}")
    progress_stalled = bool(progress["progress_stalled"])
    if progress_stalled:
        stale_age = progress["nonthermal_active_progress_age_seconds"]
        age_text = "no durable progress heartbeat"
        if stale_age is not None:
            age_text = f"durable progress is stale by {stale_age:.0f}s"
        issues.append(
            "owned runtime progress stalled: " + age_text
        )

    # The host-wide clock above detects pre-worker/supervisor-wide silence.
    # It must not be the only clock once workers exist: fresh output from slot
    # B must never conceal a hung slot A.  Bind each clock to an exact live
    # worker process generation and, when present, the trainer attempt whose
    # PGID is that worker PID.
    worker_progress: list[dict[str, Any]] = []
    for worker in roles["workers"]:
        worker_pid = int(worker["pid"])
        slot = str(worker["worker_slot"])
        worker_trainers = trainers_by_worker.get(worker_pid, [])
        generation_started_at = worker_started_at_by_pid.get(worker_pid)
        worker_paths = [
            root / "workers" / host_id / slot / "status.jsonl",
            root / "workers" / host_id / slot / "worker_console.log",
        ]
        active_attempt: str | None = None
        active_run_id: str | None = None
        if len(worker_trainers) == 1:
            trainer = worker_trainers[0]
            attempt = canonical(str(trainer["attempt_dir"]))
            run_dir = attempt.parent
            active_attempt = str(attempt)
            active_run_id = str(trainer["run_id"])
            worker_paths.extend(
                [
                    attempt / "attempt_contract.json",
                    attempt / "scaling_config.json",
                    attempt / "console.log",
                    attempt / "train_log.jsonl",
                    attempt / "summary.json",
                    run_dir / "retry_state.json",
                    run_dir / "complete.json",
                ]
            )
        worker_progress_age = (
            newest_generation_file_age(
                worker_paths,
                current_time=current_time,
                generation_started_at=generation_started_at,
            )
            if generation_started_at is not None
            else None
        )
        worker_pause_seconds, worker_pause_owned = verified_thermal_pause_state(
            thermal_event_path,
            [worker],
            roles["guards"],
            phase=phase,
            host_id=host_id,
            since=(
                current_time - worker_progress_age
                if worker_progress_age is not None
                else current_time
            ),
            until=current_time,
        )
        worker_runtime_ages = [
            age
            for proc in [worker, *worker_trainers]
            if (age := runtime_age_by_pid.get(int(proc["pid"]))) is not None
        ]
        worker_state = classify_worker_runtime_progress(
            worker,
            worker_trainers,
            runtime_ages_seconds=worker_runtime_ages,
            nonthermal_progress_age_seconds=worker_progress_age,
            thermal_heartbeat_age_seconds=thermal_heartbeat_age,
            verified_thermal_pause_seconds=worker_pause_seconds,
            current_pause_owned=worker_pause_owned,
            stale_seconds=stale_seconds,
        )
        worker_record = {
            "slot": slot,
            "worker_pid": worker_pid,
            "worker_start_ticks": int(worker["start_ticks"]),
            "trainer_pids": [int(proc["pid"]) for proc in worker_trainers],
            "active_run_id": active_run_id,
            "active_attempt_dir": active_attempt,
            "nonthermal_progress_age_seconds": worker_progress_age,
            "verified_thermal_pause_seconds": worker_pause_seconds,
            "current_pause_owned": worker_pause_owned,
            **worker_state,
        }
        worker_progress.append(worker_record)
        if worker_state["progress_stalled"]:
            active_age = worker_state["nonthermal_active_progress_age_seconds"]
            age_text = "no generation-bound durable progress heartbeat"
            if active_age is not None:
                age_text = f"active durable progress is stale by {active_age:.0f}s"
            attempt_text = (
                f" run={active_run_id} attempt={active_attempt}"
                if active_attempt is not None
                else " no-live-trainer"
            )
            issues.append(
                f"worker progress stalled slot={slot} pid={worker_pid}"
                f"{attempt_text}: {age_text}"
            )

    return {
        "schema_version": HOST_SCHEMA,
        "observed_at": now(),
        "phase": phase,
        "host_id": host_id,
        "physical_host_id": "local" if execution_plan is not None else host_id,
        "local_execution_plan_json": str(canonical(local_execution_plan)) if execution_plan is not None else None,
        "local_execution_admission_supported": execution_plan is not None,
        "predecessor_waits": predecessor_waits,
        "cohosted_peer_pids": sorted(peer_pids),
        "matrix_json": str(matrix_path),
        "matrix_generation_id": matrix["metadata"]["generation_id"],
        "matrix_semantic_fingerprint": matrix["metadata"]["semantic_fingerprint"]["value"],
        "expected_rows": len(host_rows),
        "expected_slots": slots,
        "processes": roles,
        "role_counts": {key: len(value) for key, value in roles.items()},
        "trainer_dataloader_descendants": trainer_dataloader_descendants,
        "selected_gpu_compute": compute,
        "foreign_selected_gpu_compute": foreign_compute,
        "gpu_query_errors": gpu_query_errors,
        "admission_ready_gpus": admission_ready_gpus,
        "deferred_admission_supported": per_gpu_admission,
        "admission_state": admission_state,
        "gpu_inventory": [inventory_by_index[gpu] for gpu in devices if gpu in inventory_by_index],
        "supervisor_pid_record": pid_record,
        "last_supervisor_status": last_status,
        "host_completion": {
            "path": str(marker_path),
            "exists": marker_path.is_file(),
            "valid": marker_valid,
        },
        "retry_states": retry_states,
        "oom_failures": oom_failures,
        "log_findings": log_findings,
        "newest_log_age_seconds": newest_log_age,
        "nonthermal_progress_age_seconds": nonthermal_progress_age,
        "verified_thermal_pause_seconds": verified_pause_seconds,
        "current_pause_owned": current_pause_owned,
        "nonthermal_active_progress_age_seconds": progress[
            "nonthermal_active_progress_age_seconds"
        ],
        "thermal_heartbeat_age_seconds": thermal_heartbeat_age,
        "runtime_youngest_age_seconds": progress[
            "runtime_youngest_age_seconds"
        ],
        "thermal_pause_verified": progress["thermal_pause_verified"],
        "progress_stalled": progress_stalled,
        "worker_progress": worker_progress,
        "other_scaling_processes": [
            item
            for item in scaling_processes
            if int(item["pid"])
            not in {
                int(proc["pid"])
                for values in roles.values()
                for proc in values
            }
            | {int(proc["pid"]) for proc in trainer_dataloader_descendants}
            | peer_pids
        ],
        "issues": issues,
        "warnings": warnings,
    }


def ssh_host_probe(
    *,
    script: Path,
    python: Path,
    matrix_path: Path,
    phase: str,
    host_id: str,
    stale_seconds: float,
    remote_host: str,
    remote_port: int,
    remote_identity: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = [
        str(python),
        str(script),
        "--host-probe",
        "--matrix-json",
        str(matrix_path),
        "--phase",
        phase,
        "--host-id",
        host_id,
        "--stale-seconds",
        str(stale_seconds),
    ]
    import shlex

    result = subprocess.run(
        [
            "ssh",
            "-i",
            str(remote_identity),
            "-p",
            str(remote_port),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(timeout_seconds))}",
            remote_host,
            shlex.join(command),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds + 5,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"remote host probe failed rc={result.returncode}: {result.stderr.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("remote host probe produced no JSON")
    value = json.loads(lines[-1])
    if not isinstance(value, dict) or value.get("schema_version") != HOST_SCHEMA:
        raise ValueError("remote host probe returned an unexpected schema")
    return value


def structural_verification(
    matrix_path: Path, proposal_path: Path
) -> dict[str, Any]:
    result = verify_phase_campaign(
        matrix_path,
        proposal_path,
        load_checkpoint_tensors=False,
    )
    # The verifier deliberately labels a no-tensor pass as an issue so it can
    # never be mistaken for formal acceptance.  For a watchdog this is an
    # expected limitation, retained explicitly below.
    expected_limitation = "checkpoint tensor validation is disabled"
    raw_issues = list(result.get("matrix_issues", []))
    effective_issues = [issue for issue in raw_issues if issue != expected_limitation]
    return {
        **result,
        "status": (
            "invalid"
            if effective_issues or int(result.get("counts", {}).get("rejected", 0))
            else "incomplete"
            if int(result.get("counts", {}).get("pending", 0))
            else "structurally_complete_requires_checkpoint_load"
        ),
        "complete": False,
        "matrix_issues": effective_issues,
        "checkpoint_validation_deferred": True,
    }


def evaluate_campaign(
    *,
    matrix_path: Path,
    proposal_path: Path,
    phase: str,
    remote_host: str,
    remote_port: int,
    remote_identity: Path,
    remote_python: Path,
    remote_script: Path,
    stale_seconds: float,
    no_remote: bool = False,
    local_execution_plan: Path | None = None,
) -> dict[str, Any]:
    matrix_path = canonical(matrix_path)
    proposal_path = canonical(proposal_path)
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict) or matrix.get("metadata", {}).get("phase") != phase:
        raise ValueError("watchdog phase differs from formal matrix")
    verification = structural_verification(matrix_path, proposal_path)
    hosts: dict[str, Any] = {}
    probe_errors: dict[str, str] = {}
    try:
        hosts["local"] = host_probe(
            matrix_path, phase, "local", stale_seconds=stale_seconds
        )
    except Exception as exc:
        probe_errors["local"] = f"{type(exc).__name__}: {exc}"
    if local_execution_plan is not None:
        try:
            hosts["remote"] = host_probe(
                matrix_path, phase, "remote", stale_seconds=stale_seconds,
                local_execution_plan=local_execution_plan,
            )
        except Exception as exc:
            probe_errors["remote"] = f"local execution probe failed: {type(exc).__name__}: {exc}"
    elif not no_remote:
        try:
            hosts["remote"] = ssh_host_probe(
                script=remote_script,
                python=remote_python,
                matrix_path=matrix_path,
                phase=phase,
                host_id="remote",
                stale_seconds=stale_seconds,
                remote_host=remote_host,
                remote_port=remote_port,
                remote_identity=remote_identity,
                timeout_seconds=15.0,
            )
        except Exception as exc:
            probe_errors["remote"] = f"{type(exc).__name__}: {exc}"
    alerts: list[dict[str, Any]] = []
    for issue in verification.get("matrix_issues", []):
        alerts.append({"severity": "error", "kind": "matrix", "message": issue})
    for run in verification.get("runs", []):
        if run.get("status") == "rejected":
            alerts.append(
                {
                    "severity": "error",
                    "kind": "rejected_artifact",
                    "run_id": run.get("run_id"),
                    "message": "; ".join(str(item) for item in run.get("reasons", [])),
                }
            )
    for host_id, error in probe_errors.items():
        alerts.append(
            {"severity": "warning", "kind": "host_probe", "host_id": host_id, "message": error}
        )
    for host_id, snapshot in hosts.items():
        for issue in snapshot.get("issues", []):
            alerts.append(
                {"severity": "error", "kind": "runtime_identity", "host_id": host_id, "message": issue}
            )
        for warning in snapshot.get("warnings", []):
            alerts.append(
                {"severity": "warning", "kind": "runtime_warning", "host_id": host_id, "message": warning}
            )
        for finding in snapshot.get("log_findings", []):
            alerts.append(
                {
                    "severity": "warning",
                    "kind": "log_finding",
                    "host_id": host_id,
                    "message": json.dumps(finding, ensure_ascii=False),
                }
            )
    fatal = any(item["severity"] == "error" for item in alerts)
    running = any(
        sum(int(value) for value in snapshot.get("role_counts", {}).values()) > 0
        for snapshot in hosts.values()
    )
    accepted = int(verification.get("counts", {}).get("accepted", 0))
    expected = 60 if phase == "c" else 25
    artifacts_ready = (
        verification.get("status") == "structurally_complete_requires_checkpoint_load"
        and accepted == expected
    )
    if fatal:
        status = "invalid"
    elif probe_errors:
        status = "observation_incomplete"
    elif artifacts_ready:
        status = "artifacts_ready_runtime_active" if running else "ready_for_strict_verification"
    elif running:
        status = "running"
    else:
        status = "incomplete_idle"
    return {
        "schema_version": SCHEMA,
        "observed_at": now(),
        "status": status,
        "phase": phase,
        "matrix_json": str(matrix_path),
        "matrix_generation_id": matrix["metadata"]["generation_id"],
        "matrix_semantic_fingerprint": matrix["metadata"]["semantic_fingerprint"]["value"],
        "structural_verification": verification,
        "hosts": hosts,
        "host_probe_errors": probe_errors,
        "artifacts_ready_for_strict_verification": artifacts_ready,
        "runtime_active": running,
        "alerts": alerts,
    }


def alert_key(alert: dict[str, Any]) -> str:
    return json.dumps(
        {key: alert.get(key) for key in ("severity", "kind", "host_id", "run_id", "message")},
        sort_keys=True,
        ensure_ascii=False,
    )


def append_new_alerts(path: Path, alerts: Iterable[dict[str, Any]], old_keys: set[str]) -> None:
    new = [alert for alert in alerts if alert_key(alert) not in old_keys]
    if not new:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for alert in new:
            handle.write(json.dumps({"observed_at": now(), **alert}, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    script = Path(__file__).resolve()
    pipeline = script.parents[2]
    workspace = pipeline.parents[2]
    run_root = workspace / "local_runtime" / "runs" / "scaling"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--phase", choices=("c", "d"), required=True)
    parser.add_argument("--proposal-json", type=Path, default=workspace / "scaling_data/reports/domainwise_scaling_matrix_proposal_v1.json")
    parser.add_argument("--host-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--host-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--alerts-jsonl", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=900.0)
    parser.add_argument("--remote-host", default="user@compute-host")
    parser.add_argument("--remote-port", type=int, default=10090)
    parser.add_argument("--remote-identity", type=Path, default=Path("~/.ssh/id_rsa"))
    parser.add_argument("--remote-python", type=Path, default=workspace / "local_runtime/envs/holomol-py311-cu121/bin/python")
    parser.add_argument("--remote-script", type=Path, default=script)
    parser.add_argument("--no-remote", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.stale_seconds <= 0:
        raise ValueError("watch intervals must be positive")
    matrix_path = canonical(args.matrix_json)
    if args.host_probe:
        if not args.host_id:
            parser.error("--host-probe requires --host-id")
        safe_component(args.host_id, "--host-id")
        print(
            json.dumps(
                host_probe(matrix_path, args.phase, args.host_id, stale_seconds=args.stale_seconds),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0

    matrix = load_json(matrix_path)
    root = canonical(matrix["metadata"]["output_base"])
    output = canonical(args.output_json or root / "watchdog_current.json")
    alerts_path = canonical(args.alerts_jsonl or root / "watchdog_alerts.jsonl")
    lock_path = canonical(args.lock_file or Path(f"/dev/shm/holomol-{os.getuid()}/scaling/domainwise_followup_{args.phase}_watchdog.lock"))
    pid_path = canonical(args.pid_file or root / "watchdog.pid.json")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another watchdog owns {lock_path}") from exc
    atomic_write_json(pid_path, {"pid": os.getpid(), "phase": args.phase, "matrix": str(matrix_path), "started_at": now()})
    old_keys: set[str] = set()
    try:
        while True:
            report = evaluate_campaign(
                matrix_path=matrix_path,
                proposal_path=canonical(args.proposal_json),
                phase=args.phase,
                remote_host=args.remote_host,
                remote_port=args.remote_port,
                remote_identity=canonical(args.remote_identity),
                remote_python=canonical(args.remote_python),
                remote_script=canonical(args.remote_script),
                stale_seconds=args.stale_seconds,
                no_remote=args.no_remote,
            )
            append_new_alerts(alerts_path, report["alerts"], old_keys)
            old_keys = {alert_key(item) for item in report["alerts"]}
            atomic_write_json(output, report)
            print(json.dumps({"status": report["status"], "phase": args.phase, "counts": report["structural_verification"]["counts"], "alerts": len(report["alerts"])}, ensure_ascii=False, allow_nan=False), flush=True)
            if args.once:
                return 1 if report["status"] == "invalid" else 0
            time.sleep(args.interval_seconds)
    finally:
        try:
            if pid_path.is_file() and load_json(pid_path).get("pid") == os.getpid():
                pid_path.unlink()
        except Exception:
            pass
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
