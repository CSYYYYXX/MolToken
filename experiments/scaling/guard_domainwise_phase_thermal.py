#!/usr/bin/env python3
"""Identity-safe thermal shedding for Phase-C/D generic workers.

On the local 48 GiB cards, slot1 is shed first; if selected cards remain at
the thermal wall, slot0 is shed too.  On the remote 24 GiB host, the sole slot
is guarded directly.  The guard records and later resumes only process groups
whose exact PID generation, matrix, phase, host, slot and GPU mapping were
revalidated immediately before signalling.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from domainwise_gpu_admission import probe_gpus, query_gpu_csv
from run_domainwise_phase_slot import load_json, matrix_phase, safe_component
from supervise_domainwise_phase_host import validate_global_matrix


WORKER_SCRIPT_NAME = "run_domainwise_phase_slot.py"
SUPERVISOR_SCRIPT_NAME = "supervise_domainwise_phase_host.py"
GPU_FIELDS = (
    "index",
    "uuid",
    "temperature.gpu",
    "utilization.gpu",
    "power.draw",
    "clocks_event_reasons.sw_thermal_slowdown",
)


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    pgid: int
    sid: int
    state: str
    start_ticks: int
    cmdline: tuple[str, ...]


@dataclass(frozen=True)
class WorkerIdentity:
    phase: str
    host_id: str
    slot: str
    gpu: int
    pid: int
    ppid: int
    pgid: int
    sid: int
    state: str
    start_ticks: int


@dataclass(frozen=True)
class GuardPolicy:
    name: str
    gpu: int
    slot: str
    pause_c: float
    resume_c: float
    requires_paused_policy: str | None = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_event(
    path: Path, event: str, *, durable: bool = False, **values: Any
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"time": now(), "event": event, **values},
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        if durable:
            os.fsync(handle.fileno())


def read_proc(pid: int) -> ProcInfo:
    proc = Path("/proc") / str(int(pid))
    if proc.stat().st_uid != os.getuid():
        raise PermissionError(f"pid={pid} is not owned by uid={os.getuid()}")
    raw = (proc / "stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise ValueError(f"cannot parse /proc/{pid}/stat")
    tail = raw[close + 2 :].split()
    if len(tail) < 20:
        raise ValueError(f"short /proc/{pid}/stat")
    cmdline = tuple(
        token.decode("utf-8", errors="replace")
        for token in (proc / "cmdline").read_bytes().split(b"\0")
        if token
    )
    return ProcInfo(
        pid=int(pid),
        state=tail[0],
        ppid=int(tail[1]),
        pgid=int(tail[2]),
        sid=int(tail[3]),
        start_ticks=int(tail[19]),
        cmdline=cmdline,
    )


def self_generation_identity() -> dict[str, int]:
    raw = Path("/proc/self/stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    tail = raw[close + 2 :].split()
    if close < 0 or len(tail) < 20:
        raise ValueError("cannot read guard /proc/self/stat identity")
    return {
        "pid": os.getpid(),
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
    }


def one_arg(cmdline: tuple[str, ...], flag: str) -> str:
    positions = [index for index, token in enumerate(cmdline) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(cmdline):
        raise ValueError(f"expected exactly one {flag}")
    return cmdline[positions[0] + 1]


def contains_script(cmdline: tuple[str, ...], script: Path) -> bool:
    expected = script.resolve()
    for token in cmdline:
        try:
            candidate = Path(token)
            if candidate.name == expected.name and candidate.resolve() == expected:
                return True
        except (OSError, RuntimeError):
            continue
    return False


def candidate_pids(slot: str) -> Iterable[int]:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = tuple(
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            )
            if WORKER_SCRIPT_NAME not in {Path(token).name for token in tokens}:
                continue
            if one_arg(tokens, "--worker-slot") == slot:
                yield int(entry.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            continue


def validate_worker(
    pid: int,
    *,
    phase: str,
    host_id: str,
    slot: str,
    gpu: int,
    matrix_path: Path,
    worker_script: Path,
    supervisor_script: Path,
) -> WorkerIdentity:
    proc = read_proc(pid)
    if proc.state in {"Z", "X", "x"} or proc.pid != proc.pgid or proc.pid != proc.sid:
        raise ValueError("worker is not a live process-group/session leader")
    if not contains_script(proc.cmdline, worker_script):
        raise ValueError("worker script mismatch")
    if (
        one_arg(proc.cmdline, "--worker-slot") != slot
        or one_arg(proc.cmdline, "--host-id") != host_id
        or Path(one_arg(proc.cmdline, "--matrix-json")).resolve() != matrix_path
    ):
        raise ValueError("worker identity arguments mismatch")
    parent = read_proc(proc.ppid)
    if not contains_script(parent.cmdline, supervisor_script):
        raise ValueError("worker parent is not the generic supervisor")
    if (
        one_arg(parent.cmdline, "--phase") != phase
        or one_arg(parent.cmdline, "--host-id") != host_id
        or Path(one_arg(parent.cmdline, "--matrix-json")).resolve() != matrix_path
    ):
        raise ValueError("supervisor identity arguments mismatch")
    return WorkerIdentity(
        phase=phase,
        host_id=host_id,
        slot=slot,
        gpu=gpu,
        pid=proc.pid,
        ppid=proc.ppid,
        pgid=proc.pgid,
        sid=proc.sid,
        state=proc.state,
        start_ticks=proc.start_ticks,
    )


def discover_worker(**kwargs: Any) -> tuple[WorkerIdentity | None, str]:
    pids = sorted(set(candidate_pids(str(kwargs["slot"]))))
    if not pids:
        return None, "not_found"
    valid: list[WorkerIdentity] = []
    errors: list[str] = []
    for pid in pids:
        try:
            valid.append(validate_worker(pid, **kwargs))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError) as exc:
            errors.append(f"pid={pid}:{type(exc).__name__}:{exc}")
    if len(pids) == 1 and len(valid) == 1:
        return valid[0], "valid"
    if len(valid) > 1:
        return None, "ambiguous=" + ",".join(str(item.pid) for item in valid)
    return None, "invalid=" + "|".join(errors)


def same_generation(left: WorkerIdentity, right: WorkerIdentity) -> bool:
    return (
        left.phase,
        left.host_id,
        left.slot,
        left.gpu,
        left.pid,
        left.pgid,
        left.sid,
        left.start_ticks,
    ) == (
        right.phase,
        right.host_id,
        right.slot,
        right.gpu,
        right.pid,
        right.pgid,
        right.sid,
        right.start_ticks,
    )


def validate_for_resume(
    identity: WorkerIdentity, matrix_path: Path, worker_script: Path
) -> ProcInfo:
    proc = read_proc(identity.pid)
    if (
        proc.state in {"Z", "X", "x"}
        or proc.start_ticks != identity.start_ticks
        or proc.pgid != identity.pgid
        or proc.sid != identity.sid
        or proc.pid != proc.pgid
        or proc.pid != proc.sid
        or not contains_script(proc.cmdline, worker_script)
        or one_arg(proc.cmdline, "--worker-slot") != identity.slot
        or one_arg(proc.cmdline, "--host-id") != identity.host_id
        or Path(one_arg(proc.cmdline, "--matrix-json")).resolve() != matrix_path
    ):
        raise ValueError("recorded worker generation/identity changed")
    return proc


def csv_query(selector: str, fields: tuple[str, ...]) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{selector}={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi {selector} query failed: {result.stderr.strip()}")
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]
    if any(len(row) != len(fields) for row in rows):
        raise RuntimeError(f"unexpected nvidia-smi fields for {selector}: {rows}")
    return rows


def gpu_telemetry() -> dict[int, dict[str, str]]:
    return {
        int(row[0]): dict(zip(GPU_FIELDS, row))
        for row in csv_query("gpu", GPU_FIELDS)
    }


def compute_gpu_by_pid() -> dict[int, str]:
    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    return {int(row[1]): row[0] for row in csv_query("compute-apps", fields)}


def scoped_compute_gpu_by_pid(gpu: int, expected_uuid: str) -> dict[int, str]:
    """Recheck one admitted physical device; an error is never an empty map."""
    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    result: dict[int, str] = {}
    for row in query_gpu_csv(gpu, "compute-apps", fields):
        try:
            pid = int(row[1])
            used_memory = int(row[3])
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid GPU{gpu} compute row") from exc
        if row[0] != expected_uuid or pid <= 0 or used_memory < 0 or pid in result:
            raise RuntimeError(f"GPU{gpu} compute identity is invalid or duplicated")
        result[pid] = row[0]
    return result


def deferred_gpu_sample(
    gpu_ids: Iterable[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, str], dict[int, str]]:
    """Keep per-device query failures visible without poisoning healthy GPUs."""
    probes = probe_gpus(gpu_ids)
    telemetry: dict[int, dict[str, Any]] = {}
    compute: dict[int, str] = {}
    errors: dict[int, str] = {}
    for gpu, probe in probes.items():
        if probe["status"] != "ok":
            errors[gpu] = str(probe.get("error") or "GPU probe unavailable")
            continue
        inventory = probe["inventory"]
        telemetry[gpu] = {
            "index": str(gpu),
            "uuid": inventory["uuid"],
            "temperature.gpu": inventory["temperature_c"],
            "utilization.gpu": inventory["utilization_gpu"],
            # These optional display fields are deliberately not used as
            # evidence: the minimal admission probe does not request them.
            "power.draw": inventory.get("power_draw_w"),
            "clocks_event_reasons.sw_thermal_slowdown": inventory.get(
                "sw_thermal_slowdown"
            ),
        }
        for process in probe["compute"]:
            pid, uuid = int(process["pid"]), str(process["gpu_uuid"])
            if pid in compute and compute[pid] != uuid:
                # Preserve cross-device evidence for group validation without
                # letting an unrelated multi-GPU foreign process kill guard.
                compute[pid] = "multiple_observed_gpu_uuids"
            else:
                compute[pid] = uuid
    return telemetry, compute, errors


def validate_deferred_layout(
    host_id: str, host_devices: dict[int, dict[str, Any]]
) -> None:
    if host_id != "remote" or not host_devices or any(
        int(device["concurrent_slots"]) != 1 for device in host_devices.values()
    ):
        raise ValueError("deferred GPU thermal protection requires remote single-slot GPUs")


def validate_group_compute(
    identity: WorkerIdentity, expected_uuid: str, gpu_by_pid: dict[int, str]
) -> tuple[bool, list[int]]:
    group: list[tuple[int, str]] = []
    for pid, uuid in gpu_by_pid.items():
        try:
            if os.getpgid(pid) == identity.pgid:
                group.append((pid, uuid))
        except (ProcessLookupError, PermissionError):
            continue
    return bool(group) and all(uuid == expected_uuid for _, uuid in group), sorted(
        pid for pid, _ in group
    )


def build_policies(
    host_id: str,
    host_rows: list[dict[str, Any]],
    host_devices: dict[int, dict[str, Any]],
) -> list[GuardPolicy]:
    slots = {str(row["worker_slot"]) for row in host_rows}
    policies: list[GuardPolicy] = []
    if host_id == "local":
        for gpu, device in sorted(host_devices.items()):
            if int(device["concurrent_slots"]) >= 2:
                secondary = f"{host_id}_gpu{gpu}_slot1"
                if secondary in slots:
                    policies.append(
                        GuardPolicy(
                            name=f"gpu{gpu}_secondary",
                            gpu=gpu,
                            slot=secondary,
                            pause_c=87.0,
                            resume_c=82.0,
                        )
                    )
        # Emergency slot0 protection applies to every local physical card.
        # Dual-slot cards still shed slot1 first whenever that worker exists.
        for gpu in sorted(host_devices):
            primary = f"{host_id}_gpu{gpu}_slot0"
            secondary_name = f"gpu{gpu}_secondary"
            if gpu in host_devices and primary in slots:
                policies.append(
                    GuardPolicy(
                        name=f"gpu{gpu}_primary",
                        gpu=gpu,
                        slot=primary,
                        pause_c=90.0,
                        resume_c=84.0,
                        requires_paused_policy=(
                            secondary_name
                            if int(host_devices[gpu]["concurrent_slots"]) >= 2
                            else None
                        ),
                    )
                )
    else:
        for gpu in sorted(host_devices):
            primary = f"{host_id}_gpu{gpu}_slot0"
            if primary in slots:
                policies.append(
                    GuardPolicy(
                        name=f"gpu{gpu}_primary",
                        gpu=gpu,
                        slot=primary,
                        pause_c=89.0,
                        resume_c=83.0,
                    )
                )
    if not policies:
        raise ValueError(f"no thermal policies apply to host {host_id!r}")
    return policies


def main() -> int:
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--phase", choices=("c", "d"), required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--event-jsonl", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--consecutive-hot-samples", type=int, default=1)
    parser.add_argument("--consecutive-cool-samples", type=int, default=1)
    parser.add_argument("--max-consecutive-telemetry-errors", type=int, default=3)
    parser.add_argument("--defer-unavailable-gpus", action="store_true")
    parser.add_argument("--local-execution-plan", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    safe_component(args.host_id, "--host-id")
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    if args.consecutive_hot_samples <= 0 or args.consecutive_cool_samples <= 0:
        raise ValueError("thermal sample counts must be positive")
    if args.max_consecutive_telemetry_errors <= 0:
        raise ValueError("--max-consecutive-telemetry-errors must be positive")
    matrix_path = args.matrix_json.expanduser().resolve()
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError("matrix root must be an object")
    actual_phase, _ = matrix_phase(matrix)
    if actual_phase != args.phase:
        raise ValueError("thermal guard phase differs from matrix")
    host_rows, _, host_devices, _ = validate_global_matrix(
        matrix, args.phase, args.host_id
    )
    if args.defer_unavailable_gpus:
        validate_deferred_layout(args.host_id, host_devices)
    if args.local_execution_plan is not None:
        from domainwise_local_execution import load_local_execution_plan

        load_local_execution_plan(matrix_path, args.local_execution_plan)
        if args.phase != "c" or args.host_id != "remote" or not args.defer_unavailable_gpus:
            raise ValueError("local execution placement requires deferred logical remote Phase C")
    policies = build_policies(args.host_id, host_rows, host_devices)
    event_path = args.event_jsonl.expanduser().resolve()
    pid_path = args.pid_file.expanduser().resolve()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = pid_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another thermal guard owns {pid_path}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    worker_script = script_path.with_name(WORKER_SCRIPT_NAME)
    supervisor_script = script_path.with_name(SUPERVISOR_SCRIPT_NAME)
    guard_identity = self_generation_identity()
    stop = threading.Event()
    received_signal: int | None = None
    paused: dict[str, WorkerIdentity] = {}
    hot = {policy.name: 0 for policy in policies}
    cool = {policy.name: 0 for policy in policies}

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        stop.set()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_stop)

    def discovery(policy: GuardPolicy) -> tuple[WorkerIdentity | None, str]:
        return discover_worker(
            phase=args.phase,
            host_id=args.host_id,
            slot=policy.slot,
            gpu=policy.gpu,
            matrix_path=matrix_path,
            worker_script=worker_script,
            supervisor_script=supervisor_script,
        )

    def resume(policy: GuardPolicy, identity: WorkerIdentity, reason: str) -> bool:
        try:
            validate_for_resume(identity, matrix_path, worker_script)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError) as exc:
            append_event(
                event_path,
                "resume_skipped_identity_changed",
                policy=policy.name,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
                identity=asdict(identity),
            )
            return False
        if args.dry_run:
            append_event(
                event_path,
                "would_sigcont",
                policy=policy.name,
                reason=reason,
                identity=asdict(identity),
            )
        else:
            # Write-ahead ownership is required here for the same reason as
            # ``sigstop_intent``: a process death after killpg() but before
            # the durable ``sigcont_sent`` record must not leave the watchdog
            # believing that this exact worker generation is still paused.
            append_event(
                event_path,
                "sigcont_intent",
                durable=True,
                policy=policy.name,
                reason=reason,
                identity=asdict(identity),
                guard_identity=guard_identity,
            )
            os.killpg(identity.pgid, signal.SIGCONT)
            append_event(
                event_path,
                "sigcont_sent",
                durable=True,
                policy=policy.name,
                reason=reason,
                identity=asdict(identity),
                guard_identity=guard_identity,
            )
        return True

    def recorded_generation_is_alive(identity: WorkerIdentity) -> bool:
        try:
            proc = read_proc(identity.pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            return False
        return (
            proc.state not in {"Z", "X", "x"}
            and proc.start_ticks == identity.start_ticks
            and proc.pgid == identity.pgid
            and proc.sid == identity.sid
        )

    append_event(
        event_path,
        "guard_start",
        pid=os.getpid(),
        guard_identity=guard_identity,
        phase=args.phase,
        host_id=args.host_id,
        matrix=str(matrix_path),
        policies=[asdict(policy) for policy in policies],
        dry_run=args.dry_run,
        defer_unavailable_gpus=args.defer_unavailable_gpus,
    )
    samples = 0
    telemetry_errors = 0
    per_gpu_errors = {policy.gpu: 0 for policy in policies}
    try:
        while not stop.is_set():
            samples += 1
            required_gpus: set[int] = set()
            sample_errors: dict[int, str] = {}
            if args.defer_unavailable_gpus:
                # Discover every frozen slot each cycle. A late worker becomes
                # required immediately; a missing card is harmless only while
                # no exact worker/owned stopped generation needs protection.
                for policy in policies:
                    identity = paused.get(policy.name)
                    if identity is not None and not recorded_generation_is_alive(identity):
                        append_event(
                            event_path,
                            "stale_owned_generation_dropped",
                            policy=policy.name,
                            reason="generation_not_alive",
                            identity=asdict(identity),
                        )
                        paused.pop(policy.name, None)
                    worker, worker_status = discovery(policy)
                    if (worker is not None and worker_status == "valid") or policy.name in paused:
                        required_gpus.add(policy.gpu)
                    elif worker is None and worker_status == "not_found":
                        append_event(
                            event_path, "awaiting_worker", policy=policy.name,
                            gpu=policy.gpu, slot=policy.slot,
                            discovery_status=worker_status, guard_identity=guard_identity,
                        )
                    else:
                        # An ambiguous or invalid process is not evidence of
                        # an empty deferred slot. Do not signal it, and retain
                        # the protection-failure budget while it is unresolved.
                        required_gpus.add(policy.gpu)
                        append_event(
                            event_path, "worker_discovery_error", policy=policy.name,
                            gpu=policy.gpu, slot=policy.slot,
                            discovery_status=worker_status, guard_identity=guard_identity,
                        )
            try:
                if args.defer_unavailable_gpus:
                    telemetry, gpu_by_pid, sample_errors = deferred_gpu_sample(per_gpu_errors)
                else:
                    telemetry = gpu_telemetry()
                    gpu_by_pid = compute_gpu_by_pid()
            except Exception as exc:
                telemetry_errors += 1
                append_event(event_path, "telemetry_error", error=f"{type(exc).__name__}: {exc}")
                if (
                    args.once
                    or telemetry_errors >= args.max_consecutive_telemetry_errors
                ):
                    return 1
                stop.wait(args.interval_seconds)
                continue
            missing_gpus = sorted({policy.gpu for policy in policies} - set(telemetry))
            if args.defer_unavailable_gpus:
                # Per-card counters below distinguish an unavailable deferred
                # card from loss of protection for an already active card.
                missing_gpus = []
            if missing_gpus:
                telemetry_errors += 1
                append_event(
                    event_path,
                    "assigned_gpu_telemetry_missing",
                    gpus=missing_gpus,
                    consecutive_errors=telemetry_errors,
                )
                if telemetry_errors >= args.max_consecutive_telemetry_errors:
                    return 1
            else:
                telemetry_errors = 0
            for policy in policies:
                values = telemetry.get(policy.gpu)
                if values is None:
                    append_event(event_path, "gpu_missing", policy=policy.name, gpu=policy.gpu)
                    continue
                temperature = float(values["temperature.gpu"])
                hot[policy.name] = hot[policy.name] + 1 if temperature >= policy.pause_c else 0
                cool[policy.name] = cool[policy.name] + 1 if temperature <= policy.resume_c else 0
                identity = paused.get(policy.name)
                decision = "observe"
                # Reconcile ownership before applying any temperature branch.
                # A stopped worker may die while the card is still hot and the
                # supervisor may then spawn a replacement for the same slot.
                # Keeping the vanished generation in ``paused`` would suppress
                # protection of that replacement (and could incorrectly satisfy
                # a primary policy's secondary-shed prerequisite).  Drop only
                # our stale bookkeeping here; never SIGCONT an unvalidated PID.
                if identity is not None and not recorded_generation_is_alive(identity):
                    append_event(
                        event_path,
                        "stale_owned_generation_dropped",
                        policy=policy.name,
                        reason="generation_not_alive",
                        identity=asdict(identity),
                    )
                    paused.pop(policy.name, None)
                    identity = None
                    decision = "stale_owned_generation_dropped"
                if identity is not None:
                    if cool[policy.name] >= args.consecutive_cool_samples:
                        if resume(policy, identity, "cooled"):
                            paused.pop(policy.name, None)
                            hot[policy.name] = 0
                            cool[policy.name] = 0
                            decision = "resumed"
                        elif not recorded_generation_is_alive(identity):
                            # The old generation vanished or the PID was
                            # reused. Drop ownership without ever signalling
                            # the replacement, then re-arm this policy.
                            paused.pop(policy.name, None)
                            decision = "stale_owned_generation_dropped"
                    else:
                        decision = "paused"
                secondary_cleared = True
                if policy.requires_paused_policy is not None:
                    secondary_cleared = policy.requires_paused_policy in paused
                    if not secondary_cleared:
                        secondary_policy = next(
                            item
                            for item in policies
                            if item.name == policy.requires_paused_policy
                        )
                        secondary_worker, _ = discovery(secondary_policy)
                        # If slot1 has completed or is not currently spawned,
                        # or is already stopped by another still-valid owner,
                        # slot0 must still receive emergency protection.  The
                        # latter does not transfer ownership: this guard must
                        # never SIGCONT a STOP it did not send.
                        secondary_cleared = (
                            secondary_worker is None
                            or secondary_worker.state in {"T", "t"}
                        )
                elif hot[policy.name] >= args.consecutive_hot_samples:
                    secondary_cleared = True
                if (
                    identity is None
                    and hot[policy.name] >= args.consecutive_hot_samples
                    and secondary_cleared
                ):
                    candidate, status = discovery(policy)
                    if candidate is None or status != "valid" or candidate.state in {"T", "t"}:
                        decision = "pause_skipped_fail_closed"
                    else:
                        if args.defer_unavailable_gpus:
                            # The worker may have appeared after this cycle's
                            # initial discovery and before the hot-card branch.
                            required_gpus.add(policy.gpu)
                        compute_ok, compute_pids = validate_group_compute(
                            candidate, values["uuid"], gpu_by_pid
                        )
                        current, current_status = discovery(policy)
                        fresh_query_ok = True
                        if args.defer_unavailable_gpus:
                            try:
                                fresh_map = scoped_compute_gpu_by_pid(
                                    policy.gpu, values["uuid"]
                                )
                            except RuntimeError as exc:
                                sample_errors[policy.gpu] = str(exc)
                                required_gpus.add(policy.gpu)
                                fresh_map = {}
                                fresh_query_ok = False
                                append_event(
                                    event_path, "pause_compute_revalidation_error",
                                    policy=policy.name, gpu=policy.gpu, error=str(exc),
                                    guard_identity=guard_identity,
                                )
                        else:
                            fresh_map = compute_gpu_by_pid()
                        fresh_ok = False
                        fresh_pids: list[int] = []
                        if fresh_query_ok and current is not None and same_generation(candidate, current):
                            fresh_ok, fresh_pids = validate_group_compute(
                                current, values["uuid"], fresh_map
                            )
                        if (
                            not compute_ok
                            or current_status != "valid"
                            or current is None
                            or not same_generation(candidate, current)
                            or not fresh_ok
                        ):
                            decision = "pause_revalidation_failed"
                        elif args.dry_run:
                            decision = "would_pause"
                        else:
                            append_event(
                                event_path,
                                "sigstop_intent",
                                durable=True,
                                policy=policy.name,
                                temperature_c=temperature,
                                compute_pids=fresh_pids,
                                identity=asdict(current),
                                guard_identity=guard_identity,
                            )
                            os.killpg(current.pgid, signal.SIGSTOP)
                            # Record in-memory ownership immediately after the
                            # signal so a failed durable event write is still
                            # recovered by the finally block.
                            paused[policy.name] = current
                            append_event(
                                event_path,
                                "sigstop_sent",
                                durable=True,
                                policy=policy.name,
                                temperature_c=temperature,
                                compute_pids=fresh_pids,
                                identity=asdict(current),
                                guard_identity=guard_identity,
                            )
                            decision = "paused_now"
                append_event(
                    event_path,
                    "sample",
                    sample=samples,
                    policy=policy.name,
                    gpu=policy.gpu,
                    slot=policy.slot,
                    temperature_c=temperature,
                    utilization_pct=values["utilization.gpu"],
                    power_draw_w=values["power.draw"],
                    sw_thermal_slowdown=values[
                        "clocks_event_reasons.sw_thermal_slowdown"
                    ],
                    hot_count=hot[policy.name],
                    cool_count=cool[policy.name],
                    guard_paused=policy.name in paused,
                    decision=decision,
                )
            if args.defer_unavailable_gpus:
                for gpu in sorted(per_gpu_errors):
                    required = gpu in required_gpus
                    failed = gpu in sample_errors
                    per_gpu_errors[gpu] = per_gpu_errors[gpu] + 1 if required and failed else 0
                    append_event(
                        event_path, "gpu_probe", gpu=gpu, required=required,
                        status="unavailable" if failed else "ok",
                        error=sample_errors.get(gpu),
                        consecutive_errors=per_gpu_errors[gpu],
                        guard_identity=guard_identity,
                    )
                if any(
                    count >= args.max_consecutive_telemetry_errors
                    for count in per_gpu_errors.values()
                ) or (args.once and any(gpu in sample_errors for gpu in required_gpus)):
                    return 1
            if args.once:
                break
            stop.wait(args.interval_seconds)
    finally:
        # Never release a pre-existing/unmanaged STOP.  Resume only exact
        # generations this guard itself recorded after sending SIGSTOP.
        for policy in reversed(policies):
            identity = paused.get(policy.name)
            if identity is not None:
                try:
                    resumed = resume(policy, identity, "guard_exit")
                    if resumed or not recorded_generation_is_alive(identity):
                        paused.pop(policy.name, None)
                except Exception as exc:
                    append_event(
                        event_path,
                        "exit_resume_error",
                        policy=policy.name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        append_event(
            event_path,
            "guard_exit",
            pid=os.getpid(),
            samples=samples,
            signal=received_signal,
        )
        try:
            lock_handle.seek(0)
            if lock_handle.read().strip() == str(os.getpid()):
                pid_path.unlink(missing_ok=True)
        finally:
            lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
