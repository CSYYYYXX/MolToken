#!/usr/bin/env python3
"""Generation-aware, non-intervening watchdog for domain-wise scaling runs.

The watchdog reads status/artifact files, /proc and nvidia-smi locally, and uses
BatchMode SSH for the remote host probe.  It never signals or launches training
processes.  Its only writes are an atomic current-status report and a deduplicated
append-only alert stream.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from verify_domainwise_campaign import atomic_write_json, verify_campaign


SUPERVISOR_SCRIPT = "supervise_domainwise_phase_b_local.py"
WORKER_SCRIPT = "run_domainwise_phase_b_slot.py"
TRAINER_SCRIPT = "train_qwen3_pretrain.py"
PROCESS_SCRIPTS = (SUPERVISOR_SCRIPT, WORKER_SCRIPT, TRAINER_SCRIPT)
OOM_RE = ("cuda out of memory", "outofmemoryerror", "cublas_status_alloc_failed")
TRACE_RE = ("traceback (most recent call last)", "cuda error", "segmentation fault")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_json_line(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = parse_json_line(line)
            if value is not None:
                events.append(value)
    return events


def proc_info(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    if root.stat().st_uid != os.getuid():
        raise PermissionError(f"pid {pid} has a different owner")
    raw = (root / "stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    tail = raw[close + 2 :].split()
    if close < 0 or len(tail) < 20:
        raise RuntimeError(f"cannot parse /proc/{pid}/stat")
    command = [
        item.decode("utf-8", errors="replace")
        for item in (root / "cmdline").read_bytes().split(b"\0")
        if item
    ]
    return {
        "pid": pid,
        "state": tail[0],
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
        "cmdline": command,
    }


def nvidia_snapshot() -> list[dict[str, Any]]:
    fields = (
        "index",
        "uuid",
        "name",
        "temperature.gpu",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "power.draw",
        "clocks_event_reasons.sw_thermal_slowdown",
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    rows: list[dict[str, Any]] = []
    for raw in csv.reader(result.stdout.splitlines()):
        if not raw:
            continue
        values = [item.strip() for item in raw]
        if len(values) != len(fields):
            raise RuntimeError(f"unexpected nvidia-smi row: {raw}")
        row = dict(zip(fields, values))
        row["index"] = int(row["index"])
        rows.append(row)
    return rows


def disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": usage.free / (1024**3),
        "free_fraction": usage.free / max(usage.total, 1),
    }


def collect_host_probe(campaign_root: Path) -> dict[str, Any]:
    processes: dict[str, dict[str, Any]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            info = proc_info(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, RuntimeError):
            continue
        names = {Path(token).name for token in info["cmdline"]}
        if any(script in names for script in PROCESS_SCRIPTS):
            processes[str(info["pid"])] = info
    result: dict[str, Any] = {
        "time": now_iso(),
        "hostname": os.uname().nodename,
        "processes": processes,
        "disks": {},
    }
    for label, path in (("campaign", campaign_root), ("tmpfs", Path("/dev/shm"))):
        try:
            result["disks"][label] = disk_snapshot(path)
        except Exception as exc:
            result["disks"][label] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    try:
        result["gpus"] = nvidia_snapshot()
    except Exception as exc:
        result["gpus"] = []
        result["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return result


def remote_host_probe(
    *,
    campaign_root: Path,
    remote_host: str,
    remote_port: int,
    remote_identity: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = [
        "ssh",
        "-i",
        str(remote_identity),
        "-p",
        str(remote_port),
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout_seconds))}",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=1",
        remote_host,
        "python3",
        str(Path(__file__).resolve()),
        "--host-probe",
        "--campaign-root",
        str(campaign_root),
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds + 15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH probe rc={result.returncode}: {result.stderr.strip()}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("SSH probe returned no JSON")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise RuntimeError("SSH probe result is not an object")
    return value


def arg_value(command: list[Any], flag: str) -> str | None:
    tokens = [str(item) for item in command]
    positions = [index for index, token in enumerate(tokens) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(tokens):
        return None
    return tokens[positions[0] + 1]


def command_contains_script(command: list[Any], script: str) -> bool:
    return script in {Path(str(item)).name for item in command}


def alert(kind: str, key: str, message: str, *, severity: str = "error", **context: Any) -> dict[str, Any]:
    return {
        "key": f"{kind}:{key}",
        "kind": kind,
        "severity": severity,
        "message": message,
        "context": context,
    }


def thermal_owned_generations(
    campaign_root: Path, host_id: str = "local"
) -> dict[str, set[tuple[Any, ...]]]:
    """Return exact worker generations currently owned by a thermal guard.

    The local host has independent primary- and secondary-slot guards.  The
    remote primary guard writes into the same shared campaign directory, so it
    must be included when evaluating the remote /proc snapshot.
    """
    owned: dict[str, set[tuple[Any, ...]]] = {}
    events: list[dict[str, Any]] = []
    for path in sorted(campaign_root.glob(f"thermal_guard_{host_id}*_events.jsonl")):
        events.extend(read_jsonl(path))
    events.sort(key=lambda item: event_timestamp(item) or float("-inf"))
    for event in events:
        identity = event.get("identity")
        if not isinstance(identity, dict) and isinstance(event.get("recorded"), dict):
            identity = event["recorded"]
        if not isinstance(identity, dict):
            continue
        key = process_generation_key(identity, slot_field="slot")
        if key is None:
            continue
        slot = str(key[0])
        if event.get("event") in {"sigstop_sent", "sigstop_primary_sent"}:
            owned.setdefault(slot, set()).add(key)
        elif event.get("event") in {
            "sigcont_sent",
            "sigcont_primary_sent",
            "paused_identity_disappeared",
            "resume_skipped_identity_changed",
        }:
            owned.setdefault(slot, set()).discard(key)
            if not owned[slot]:
                owned.pop(slot)
    return owned


def thermal_owned_slots(campaign_root: Path, host_id: str = "local") -> set[str]:
    return set(thermal_owned_generations(campaign_root, host_id))


def event_timestamp(event: dict[str, Any]) -> float | None:
    value = event.get("timestamp", event.get("time"))
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def process_generation_key(value: dict[str, Any], *, slot_field: str) -> tuple[Any, ...] | None:
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


def verified_thermal_pause_seconds(
    campaign_root: Path,
    host_id: str,
    slot: str,
    process: dict[str, Any],
    *,
    since: float,
    until: float,
) -> float:
    """Count only canonical STOP/CONT time for this exact worker generation."""
    process_key = process_generation_key({**process, "slot": slot}, slot_field="slot")
    if process_key is None or until <= since:
        return 0.0
    events: list[dict[str, Any]] = []
    for path in sorted(campaign_root.glob(f"thermal_guard_{host_id}*_events.jsonl")):
        events.extend(read_jsonl(path))
    events.sort(key=lambda item: event_timestamp(item) or float("-inf"))
    stopped_at: float | None = None
    intervals: list[tuple[float, float]] = []
    for event in events:
        identity = event.get("identity")
        if not isinstance(identity, dict):
            continue
        if process_generation_key(identity, slot_field="slot") != process_key:
            continue
        timestamp = event_timestamp(event)
        if timestamp is None:
            continue
        kind = event.get("event")
        if kind in {"sigstop_sent", "sigstop_primary_sent"}:
            stopped_at = timestamp
        elif kind in {"sigcont_sent", "sigcont_primary_sent"} and stopped_at is not None:
            if timestamp >= stopped_at:
                intervals.append((stopped_at, timestamp))
            stopped_at = None
    return sum(
        max(0.0, min(end, until) - max(start, since))
        for start, end in intervals
    )


def collect_probe_with_thermal_bracket(
    campaign_root: Path,
    host_id: str,
    collect_probe: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, set[tuple[Any, ...]]]]:
    """Pair a process snapshot with guard ownership observed around it.

    Guard STOP/CONT cycles are deliberately much shorter than the watchdog
    interval.  Reading ownership only after a process snapshot can pair a
    stale ``T`` state with a later resume event.  Bracketing makes that sample
    coherent without signalling or otherwise changing either process.
    """

    owned_before = thermal_owned_generations(campaign_root, host_id)
    probe = collect_probe()
    owned_after = thermal_owned_generations(campaign_root, host_id)
    merged = {
        slot: owned_before.get(slot, set()) | owned_after.get(slot, set())
        for slot in set(owned_before) | set(owned_after)
    }
    return probe, merged


def active_worker_events(
    campaign_root: Path,
    slot: str,
    supervisor_start_time: str,
) -> list[dict[str, Any]]:
    events = read_jsonl(campaign_root / "workers" / slot / "status.jsonl")
    eligible = [event for event in events if str(event.get("time", "")) >= supervisor_start_time]
    starts = [index for index, event in enumerate(eligible) if event.get("event") == "worker_start"]
    return eligible[starts[-1] :] if starts else eligible


def latest_active_run(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    active: dict[str, Any] | None = None
    for event in events:
        if event.get("event") == "run_start":
            active = event
        elif event.get("event") in {"run_complete", "run_failed"}:
            if active is not None and event.get("run_id") == active.get("run_id"):
                active = None
    return active


def read_tail(path: Path, max_bytes: int = 1 << 20) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def latest_training_events(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the newest progress record and newest actual training record."""
    latest_progress: dict[str, Any] | None = None
    latest_train: dict[str, Any] | None = None
    progress_events = {
        "train",
        "eval",
        "final_eval_start",
        "final_eval_heartbeat",
        "final_eval",
    }
    for line in reversed(read_tail(path).splitlines()):
        candidate = parse_json_line(line)
        if candidate is None:
            continue
        event = candidate.get("event")
        if latest_progress is None and event in progress_events:
            latest_progress = candidate
        if latest_train is None and event == "train":
            latest_train = candidate
        if latest_progress is not None and latest_train is not None:
            break
    return latest_progress, latest_train


def training_budget_complete(
    event: dict[str, Any] | None,
    row: dict[str, Any] | None,
) -> bool:
    """Recognize only the formal target-token stop contract used by Phase B."""
    if not isinstance(event, dict) or not isinstance(row, dict):
        return False
    target = row.get("target_input_tokens_with_eos")
    logged_target = event.get("target_input_tokens")
    actual = event.get("cumulative_input_tokens")
    context = row.get("context_length")
    values = (target, logged_target, actual, context)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return False
    return bool(
        event.get("event") == "train"
        and target > 0
        and context > 0
        and logged_target == target
        and target <= actual < target + context
    )


def contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_nonfinite(item) for item in value)
    return False


def scan_active_logs(attempt_dir: Path, host_id: str, slot: str, run_id: str) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    console_path = attempt_dir / "console.log"
    if console_path.is_file():
        text = read_tail(console_path).lower()
        if any(pattern in text for pattern in OOM_RE):
            alerts.append(alert("oom", f"{host_id}:{slot}:{attempt_dir}", "active attempt contains an OOM", attempt_dir=str(attempt_dir)))
        if any(pattern in text for pattern in TRACE_RE):
            alerts.append(alert("traceback", f"{host_id}:{slot}:{attempt_dir}", "active attempt contains a traceback/CUDA failure", attempt_dir=str(attempt_dir)))
    train_log = attempt_dir / "train_log.jsonl"
    if train_log.is_file():
        malformed = False
        nonfinite = False
        tail = read_tail(train_log)
        ends_with_newline = tail.endswith("\n")
        lines = tail.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                # The first line can be partial because this is a byte tail; the
                # final line can be partial while the trainer is appending it.
                if index != 0 and not (index == len(lines) - 1 and not ends_with_newline):
                    malformed = True
                continue
            nonfinite = nonfinite or contains_nonfinite(event)
        if malformed:
            alerts.append(alert("malformed_train_log", f"{host_id}:{slot}:{attempt_dir}", "active train log contains malformed JSON", attempt_dir=str(attempt_dir)))
        if nonfinite:
            alerts.append(alert("nonfinite_train_log", f"{host_id}:{slot}:{attempt_dir}", "active train log contains NaN/Inf", attempt_dir=str(attempt_dir)))
    return alerts


def evaluate_host(
    *,
    host_id: str,
    matrix_path: Path,
    campaign_root: Path,
    probe: dict[str, Any],
    stale_seconds: float,
    hot_temperature_c: float,
    min_campaign_free_gib: float,
    min_tmpfs_free_gib: float,
    owned_thermal_generations: dict[str, set[tuple[Any, ...]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    rows = matrix["rows"]
    rows_by_run_id = {
        str(row["run_id"]): row
        for row in rows
        if isinstance(row, dict) and "run_id" in row
    }
    expected_slots = sorted({str(row["worker_slot"]) for row in rows})
    status_path = campaign_root / f"supervisor_{host_id}_status.jsonl"
    supervisor_events = read_jsonl(status_path)
    start_indices = [i for i, event in enumerate(supervisor_events) if event.get("event") == "supervisor_start"]
    alerts: list[dict[str, Any]] = []
    host: dict[str, Any] = {
        "host_id": host_id,
        "probe_time": probe.get("time"),
        "hostname": probe.get("hostname"),
        "matrix": str(matrix_path),
        "expected_rows": len(rows),
        "expected_slots": expected_slots,
        "workers": {},
        "gpus": probe.get("gpus", []),
        "disks": probe.get("disks", {}),
    }
    if not start_indices:
        alerts.append(alert("supervisor_missing_generation", host_id, "no supervisor_start event exists"))
        return host, alerts
    generation_events = supervisor_events[start_indices[-1] :]
    start_event = generation_events[0]
    supervisor_pid = int(start_event.get("pid", -1))
    host["generation"] = start_event
    done_events = [event for event in generation_events if event.get("event") == "supervisor_done"]
    generation_done = done_events[-1] if done_events else None
    host["generation_done"] = generation_done
    processes = probe.get("processes", {})
    supervisor_proc = processes.get(str(supervisor_pid))
    host["supervisor_process"] = supervisor_proc
    if not isinstance(supervisor_proc, dict):
        successful_done = (
            isinstance(generation_done, dict)
            and not generation_done.get("failures")
            and all(int(code) == 0 for code in generation_done.get("returncodes", {}).values())
        )
        if not successful_done:
            alerts.append(alert("supervisor_not_running", host_id, "latest-generation supervisor PID is not running", pid=supervisor_pid))
    else:
        command = supervisor_proc.get("cmdline", [])
        if (
            not command_contains_script(command, SUPERVISOR_SCRIPT)
            or arg_value(command, "--matrix-json") != str(matrix_path)
            or int(supervisor_proc.get("ppid", -1)) != 1
        ):
            alerts.append(alert("supervisor_identity", host_id, "supervisor PID command/matrix/PPID mismatch", pid=supervisor_pid))
        if str(supervisor_proc.get("state", "?")) in {"T", "t"}:
            alerts.append(alert("supervisor_stopped", host_id, "latest-generation supervisor process is stopped", pid=supervisor_pid))

    spawn_by_slot = {
        str(event["worker_slot"]): int(event["pid"])
        for event in generation_events
        if event.get("event") == "worker_spawn" and "worker_slot" in event and "pid" in event
    }
    exit_by_slot = {
        str(event["worker_slot"]): event
        for event in generation_events
        if event.get("event") == "worker_exit" and "worker_slot" in event
    }
    for event in generation_events:
        if event.get("event") == "worker_exit" and int(event.get("returncode", -1)) != 0:
            slot = str(event.get("worker_slot"))
            alerts.append(alert("worker_exit", f"{host_id}:{slot}:{event.get('returncode')}", "worker exited nonzero", slot=slot, returncode=event.get("returncode")))
        if event.get("event") == "supervisor_done" and event.get("failures"):
            alerts.append(alert("supervisor_failed", host_id, "supervisor finished with failed workers", failures=event.get("failures")))

    owned = owned_thermal_generations or {}
    wall_now = time.time()
    for slot in expected_slots:
        worker: dict[str, Any] = {"slot": slot, "status": "unknown"}
        host["workers"][slot] = worker
        pid = spawn_by_slot.get(slot)
        if pid is None:
            worker["status"] = "not_spawned"
            alerts.append(alert("worker_not_spawned", f"{host_id}:{slot}", "expected worker was not spawned", slot=slot))
            continue
        worker["pid"] = pid
        proc = processes.get(str(pid))
        worker["process"] = proc
        if not isinstance(proc, dict):
            exit_event = exit_by_slot.get(slot)
            worker["exit_event"] = exit_event
            if isinstance(exit_event, dict) and int(exit_event.get("returncode", -1)) == 0:
                worker["status"] = "finished"
            else:
                worker["status"] = "not_running"
                alerts.append(alert("worker_not_running", f"{host_id}:{slot}", "expected worker PID is not running", slot=slot, pid=pid))
            continue
        command = proc.get("cmdline", [])
        identity_ok = (
            command_contains_script(command, WORKER_SCRIPT)
            and arg_value(command, "--matrix-json") == str(matrix_path)
            and arg_value(command, "--worker-slot") == slot
            and int(proc.get("ppid", -1)) == supervisor_pid
            and int(proc.get("pgid", -1)) == pid
            and int(proc.get("sid", -1)) == pid
        )
        if not identity_ok:
            worker["status"] = "identity_mismatch"
            alerts.append(alert("worker_identity", f"{host_id}:{slot}", "worker identity/PGID/matrix mismatch", slot=slot, pid=pid))
            continue
        state = str(proc.get("state", "?"))
        stopped = state in {"T", "t"}
        proc_key = process_generation_key({**proc, "slot": slot}, slot_field="slot")
        thermally_owned = proc_key is not None and proc_key in owned.get(slot, set())
        if stopped and not thermally_owned:
            alerts.append(alert("unexpected_stop", f"{host_id}:{slot}:{pid}", "worker is stopped without thermal-guard ownership", slot=slot, pid=pid))
        worker["status"] = "paused_thermal" if stopped and thermally_owned else ("stopped_unowned" if stopped else "running")

        worker_events = active_worker_events(campaign_root, slot, str(start_event.get("time", "")))
        worker["event_counts"] = {
            name: sum(event.get("event") == name for event in worker_events)
            for name in ("run_start", "run_complete", "run_failed", "worker_done")
        }
        for event in worker_events:
            if event.get("event") == "run_failed":
                alerts.append(
                    alert(
                        "run_failed",
                        f"{host_id}:{slot}:{event.get('run_id')}:{event.get('attempt_dir')}",
                        "worker reported a failed run attempt",
                        slot=slot,
                        run_id=event.get("run_id"),
                        returncode=event.get("returncode"),
                        attempt_dir=event.get("attempt_dir"),
                    )
                )
            if event.get("event") == "worker_done" and int(event.get("failures", 0)):
                alerts.append(alert("worker_done_failures", f"{host_id}:{slot}", "worker_done contains failures", slot=slot, failures=event.get("failures")))
        active = latest_active_run(worker_events)
        worker["active_run"] = active
        if active is not None:
            attempt_dir = Path(str(active.get("attempt_dir")))
            train_log = attempt_dir / "train_log.jsonl"
            log_mtime = train_log.stat().st_mtime if train_log.is_file() else None
            age = wall_now - log_mtime if log_mtime is not None else None
            paused_seconds = (
                verified_thermal_pause_seconds(
                    campaign_root,
                    host_id,
                    slot,
                    proc,
                    since=log_mtime,
                    until=wall_now,
                )
                if log_mtime is not None
                else 0.0
            )
            active_age = max(0.0, age - paused_seconds) if age is not None else None
            worker["train_log_age_seconds"] = age
            worker["verified_thermal_pause_seconds"] = paused_seconds
            worker["train_log_active_age_seconds"] = active_age
            latest_progress_event = None
            latest_train_event = None
            if train_log.is_file():
                latest_progress_event, latest_train_event = latest_training_events(train_log)
            worker["latest_progress_event"] = latest_progress_event
            worker["latest_train_event"] = latest_train_event
            run_row = rows_by_run_id.get(str(active.get("run_id")))
            budget_complete = training_budget_complete(latest_train_event, run_row)
            worker["training_budget_complete"] = budget_complete
            worker["phase"] = "finalizing" if budget_complete else "training"
            direct_trainers = [
                item
                for item in processes.values()
                if int(item.get("ppid", -1)) == pid
                and int(item.get("pgid", -1)) == pid
                and command_contains_script(item.get("cmdline", []), TRAINER_SCRIPT)
            ]
            worker["direct_trainers"] = direct_trainers
            if age is None:
                run_started = datetime.fromisoformat(str(active["time"])).timestamp()
                start_age = wall_now - run_started
                start_paused = verified_thermal_pause_seconds(
                    campaign_root,
                    host_id,
                    slot,
                    proc,
                    since=run_started,
                    until=wall_now,
                )
                worker["train_log_missing_active_age_seconds"] = max(
                    0.0, start_age - start_paused
                )
                if (
                    not stopped
                    and worker["train_log_missing_active_age_seconds"] > stale_seconds
                ):
                    alerts.append(alert("train_log_missing", f"{host_id}:{slot}:{attempt_dir}", "active run has no train log", slot=slot, attempt_dir=str(attempt_dir)))
            elif active_age is not None and active_age > stale_seconds and not stopped:
                if budget_complete:
                    alerts.append(
                        alert(
                            "finalization_stalled",
                            f"{host_id}:{slot}:{attempt_dir}",
                            "finalizing run has no durable validation progress",
                            slot=slot,
                            age_seconds=age,
                            active_age_seconds=active_age,
                            attempt_dir=str(attempt_dir),
                        )
                    )
                else:
                    alerts.append(
                        alert(
                            "train_log_stale",
                            f"{host_id}:{slot}:{attempt_dir}",
                            "active train log is stale",
                            slot=slot,
                            age_seconds=age,
                            active_age_seconds=active_age,
                            attempt_dir=str(attempt_dir),
                        )
                    )
            if len(direct_trainers) != 1 and not stopped:
                alerts.append(alert("trainer_count", f"{host_id}:{slot}:{attempt_dir}", "active run does not have exactly one direct trainer", slot=slot, count=len(direct_trainers)))
            alerts.extend(scan_active_logs(attempt_dir, host_id, slot, str(active.get("run_id"))))

    if probe.get("gpu_error"):
        alerts.append(alert("gpu_probe", host_id, "GPU telemetry probe failed", error=probe["gpu_error"]))
    for gpu in probe.get("gpus", []):
        try:
            temperature = float(gpu["temperature.gpu"])
        except (KeyError, TypeError, ValueError):
            alerts.append(alert("gpu_temperature_invalid", f"{host_id}:{gpu.get('index')}", "GPU temperature is unavailable", gpu=gpu))
            continue
        if temperature >= hot_temperature_c:
            alerts.append(alert("gpu_hot", f"{host_id}:{gpu['index']}", "GPU temperature reached watchdog threshold", severity="warning", gpu_index=gpu["index"], temperature_c=temperature))
    for label, minimum in (("campaign", min_campaign_free_gib), ("tmpfs", min_tmpfs_free_gib)):
        disk = probe.get("disks", {}).get(label, {})
        if "error" in disk:
            alerts.append(alert("disk_probe", f"{host_id}:{label}", "disk probe failed", disk=disk))
        elif float(disk.get("free_gib", -1)) < minimum:
            alerts.append(alert("disk_low", f"{host_id}:{label}", "disk free space is below threshold", label=label, free_gib=disk.get("free_gib"), threshold_gib=minimum))
    return host, alerts


def append_new_alerts(path: Path, alerts: list[dict[str, Any]], previous_keys: set[str]) -> int:
    new = [item for item in alerts if str(item["key"]) not in previous_keys]
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in new:
            handle.write(json.dumps({"time": now_iso(), **item}, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return len(new)


def write_status_and_alerts(
    status_path: Path,
    alerts_path: Path,
    status: dict[str, Any],
) -> int:
    previous_keys: set[str] = set()
    if status_path.is_file():
        try:
            old = json.loads(status_path.read_text(encoding="utf-8"))
            previous_keys = {str(item["key"]) for item in old.get("alerts", []) if isinstance(item, dict) and "key" in item}
        except (OSError, ValueError, TypeError):
            previous_keys = set()
    written = append_new_alerts(alerts_path, status["alerts"], previous_keys)
    atomic_write_json(status_path, status)
    return written


def acquire_instance_lock(lock_path: Path, pid_path: Path) -> tuple[Any, Any]:
    """Acquire a nonblocking campaign singleton lock and atomically publish PID."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another watchdog holds {lock_path}")
    identity = {"pid": os.getpid(), "started_at": now_iso(), "lock_file": str(lock_path)}
    try:
        atomic_write_json(pid_path, identity)
    except Exception:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise

    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        try:
            if pid_path.is_file():
                try:
                    current = json.loads(pid_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    current = {}
                if current.get("pid") == os.getpid():
                    pid_path.unlink(missing_ok=True)
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    return handle, cleanup


def main() -> int:
    script = Path(__file__).resolve()
    workspace = script.parents[5]
    default_campaign = workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", type=Path, default=default_campaign)
    ap.add_argument("--local-matrix", type=Path, default=None)
    ap.add_argument("--remote-matrix", type=Path, default=None)
    ap.add_argument("--status-json", type=Path, default=None)
    ap.add_argument("--alerts-jsonl", type=Path, default=None)
    ap.add_argument("--lock-file", type=Path, default=None)
    ap.add_argument("--pid-file", type=Path, default=None)
    ap.add_argument("--interval-seconds", type=float, default=60.0)
    ap.add_argument("--stale-seconds", type=float, default=600.0)
    ap.add_argument("--hot-temperature-c", type=float, default=88.0)
    ap.add_argument("--min-campaign-free-gib", type=float, default=20.0)
    ap.add_argument("--min-tmpfs-free-gib", type=float, default=10.0)
    ap.add_argument("--remote-host", default="user@compute-host")
    ap.add_argument("--remote-port", type=int, default=10090)
    ap.add_argument("--remote-identity", type=Path, default=Path("~/.ssh/id_rsa"))
    ap.add_argument("--remote-timeout-seconds", type=float, default=10.0)
    ap.add_argument("--no-remote", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--host-probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    if args.host_probe:
        print(json.dumps(collect_host_probe(campaign_root), ensure_ascii=False, allow_nan=False))
        return 0
    if args.interval_seconds <= 0 or args.stale_seconds <= 0:
        raise ValueError("interval and stale thresholds must be positive")

    local_matrix = (args.local_matrix or campaign_root / "phase_b_matrix_local.json").expanduser().resolve()
    remote_matrix = (args.remote_matrix or campaign_root / "phase_b_matrix_remote.json").expanduser().resolve()
    status_path = (args.status_json or campaign_root / "campaign_watchdog_current.json").expanduser().resolve()
    alerts_path = (args.alerts_jsonl or campaign_root / "campaign_watchdog_alerts.jsonl").expanduser().resolve()
    lock_path = (args.lock_file or campaign_root / "campaign_watchdog.lock").expanduser().resolve()
    pid_path = (args.pid_file or campaign_root / "campaign_watchdog.pid").expanduser().resolve()
    _lock_handle, cleanup_lock = acquire_instance_lock(lock_path, pid_path)
    atexit.register(cleanup_lock)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda _signum, _frame: stop.set())

    while not stop.is_set():
        sample_time = now_iso()
        all_alerts: list[dict[str, Any]] = []
        hosts: dict[str, Any] = {}
        try:
            # Thermal guards can legitimately STOP/CONT a worker while the
            # process/GPU probe is being assembled.  A single ownership read
            # after that probe can therefore pair an old ``T`` process
            # snapshot with a newer ``SIGCONT`` event and report a false
            # unowned stop.  Bracket the process snapshot and accept ownership
            # observed on either side; an actually unowned persistent stop is
            # still absent from both reads and is reported normally.
            local_probe, local_thermal_owned = collect_probe_with_thermal_bracket(
                campaign_root,
                "local",
                lambda: collect_host_probe(campaign_root),
            )
            local_status, local_alerts = evaluate_host(
                host_id="local",
                matrix_path=local_matrix,
                campaign_root=campaign_root,
                probe=local_probe,
                stale_seconds=args.stale_seconds,
                hot_temperature_c=args.hot_temperature_c,
                min_campaign_free_gib=args.min_campaign_free_gib,
                min_tmpfs_free_gib=args.min_tmpfs_free_gib,
                owned_thermal_generations=local_thermal_owned,
            )
            hosts["local"] = local_status
            all_alerts.extend(local_alerts)
        except Exception as exc:
            all_alerts.append(alert("local_probe_failed", "local", "local campaign probe failed", error=f"{type(exc).__name__}: {exc}"))

        matrices = [local_matrix]
        if not args.no_remote:
            matrices.append(remote_matrix)
            try:
                remote_probe, remote_thermal_owned = collect_probe_with_thermal_bracket(
                    campaign_root,
                    "remote",
                    lambda: remote_host_probe(
                        campaign_root=campaign_root,
                        remote_host=args.remote_host,
                        remote_port=args.remote_port,
                        remote_identity=args.remote_identity.expanduser().resolve(),
                        timeout_seconds=args.remote_timeout_seconds,
                    ),
                )
                remote_status, remote_alerts = evaluate_host(
                    host_id="remote",
                    matrix_path=remote_matrix,
                    campaign_root=campaign_root,
                    probe=remote_probe,
                    stale_seconds=args.stale_seconds,
                    hot_temperature_c=args.hot_temperature_c,
                    min_campaign_free_gib=args.min_campaign_free_gib,
                    min_tmpfs_free_gib=args.min_tmpfs_free_gib,
                    owned_thermal_generations=remote_thermal_owned,
                )
                hosts["remote"] = remote_status
                all_alerts.extend(remote_alerts)
            except Exception as exc:
                all_alerts.append(alert("remote_probe_failed", "remote", "remote SSH campaign probe failed", error=f"{type(exc).__name__}: {exc}"))

        try:
            completion = verify_campaign(
                matrices,
                expected_total_rows=60 if not args.no_remote else 0,
                load_checkpoints=False,
            )
            for run in completion["runs"]:
                if run["status"] == "rejected":
                    all_alerts.append(alert("completion_rejected", str(run["run_id"]), "completed artifact failed strict verification", run_id=run["run_id"], reasons=run["reasons"]))
            for issue in completion["matrix_issues"]:
                all_alerts.append(alert("matrix_contract", issue, "campaign matrix contract is invalid", issue=issue))
        except Exception as exc:
            completion = {"status": "probe_failed", "complete": False, "error": f"{type(exc).__name__}: {exc}"}
            all_alerts.append(alert("completion_probe_failed", "campaign", "completion verification failed", error=completion["error"]))

        if completion.get("complete") is True:
            terminal_absence = {
                "supervisor_not_running",
                "worker_not_running",
                "worker_not_spawned",
                "trainer_count",
                "train_log_missing",
                "train_log_stale",
            }
            all_alerts = [item for item in all_alerts if item.get("kind") not in terminal_absence]
        deduped = {str(item["key"]): item for item in all_alerts}
        alerts = [deduped[key] for key in sorted(deduped)]
        severity = "healthy"
        if any(item["severity"] == "error" for item in alerts):
            severity = "error"
        elif alerts:
            severity = "warning"
        status = {
            "schema_version": "domainwise_campaign_watchdog_v1",
            "time": sample_time,
            "health": severity,
            "hosts": hosts,
            "completion": completion,
            "alerts": alerts,
        }
        write_status_and_alerts(status_path, alerts_path, status)
        if args.once:
            break
        stop.wait(args.interval_seconds)
    cleanup_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
