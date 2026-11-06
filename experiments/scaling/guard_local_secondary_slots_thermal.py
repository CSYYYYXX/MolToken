#!/usr/bin/env python3
"""Thermally pause/resume only local GPU4-7 Phase-B secondary worker groups.

Safety is fail-closed.  A signal is sent only after the host-qualified worker
slot, matrix, parent supervisor, PID generation, session/PGID, and live NVIDIA
compute-process-to-physical-GPU mapping all agree.  The guard never kills jobs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


GUARDED_GPUS = (4, 5, 6, 7)
WORKER_SCRIPT_NAME = "run_domainwise_phase_b_slot.py"
SUPERVISOR_SCRIPT_NAME = "supervise_domainwise_phase_b_local.py"

GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "temperature.gpu",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "power.draw",
    "clocks.current.graphics",
    "clocks_event_reasons.active",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
)

CSV_FIELDS = (
    "timestamp",
    "sample",
    "gpu_index",
    "gpu_uuid",
    "temperature_c",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_pct",
    "power_draw_w",
    "graphics_clock_mhz",
    "throttle_active_mask",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hot_consecutive",
    "cool_consecutive",
    "secondary_slot",
    "worker_status",
    "worker_pid",
    "worker_pgid",
    "worker_start_ticks",
    "worker_state",
    "group_compute_pids",
    "guard_paused",
    "decision",
    "dry_run",
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
    slot: str
    gpu_index: int
    pid: int
    pgid: int
    sid: int
    ppid: int
    start_ticks: int
    state: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_proc(pid: int) -> ProcInfo:
    proc = Path("/proc") / str(int(pid))
    if proc.stat().st_uid != os.getuid():
        raise PermissionError(f"pid={pid} is not owned by uid={os.getuid()}")
    raw_stat = (proc / "stat").read_text(encoding="utf-8")
    close = raw_stat.rfind(")")
    if close < 0:
        raise RuntimeError(f"cannot parse /proc/{pid}/stat")
    tail = raw_stat[close + 2 :].split()
    if len(tail) < 20:
        raise RuntimeError(f"short /proc/{pid}/stat")
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


def one_arg(cmdline: tuple[str, ...], flag: str) -> str:
    positions = [index for index, token in enumerate(cmdline) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(cmdline):
        raise ValueError(f"expected exactly one {flag} argument")
    return cmdline[positions[0] + 1]


def contains_script(cmdline: tuple[str, ...], expected: Path) -> bool:
    expected = expected.resolve()
    for token in cmdline:
        try:
            candidate = Path(token)
            if candidate.name == expected.name and candidate.resolve() == expected:
                return True
        except (OSError, RuntimeError):
            continue
    return False


def iter_worker_candidate_pids(slot: str) -> Iterable[int]:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        tokens = tuple(
            token.decode("utf-8", errors="replace")
            for token in raw.split(b"\0")
            if token
        )
        if WORKER_SCRIPT_NAME not in {Path(token).name for token in tokens}:
            continue
        try:
            if one_arg(tokens, "--worker-slot") == slot:
                yield int(entry.name)
        except ValueError:
            continue


def structurally_validate_worker(
    pid: int,
    *,
    slot: str,
    gpu_index: int,
    matrix_path: Path,
    worker_script: Path,
    supervisor_script: Path,
) -> WorkerIdentity:
    proc = read_proc(pid)
    if proc.state in {"Z", "X", "x"}:
        raise RuntimeError(f"worker pid={pid} has terminal state={proc.state}")
    if proc.pid <= 1 or proc.pgid != proc.pid or proc.sid != proc.pid:
        raise RuntimeError(
            f"worker must lead its own process group/session: pid={proc.pid} pgid={proc.pgid} sid={proc.sid}"
        )
    if not contains_script(proc.cmdline, worker_script):
        raise RuntimeError(f"worker script mismatch for pid={pid}")
    if one_arg(proc.cmdline, "--worker-slot") != slot:
        raise RuntimeError(f"worker slot mismatch for pid={pid}")
    if Path(one_arg(proc.cmdline, "--matrix-json")).resolve() != matrix_path:
        raise RuntimeError(f"worker matrix mismatch for pid={pid}")

    parent = read_proc(proc.ppid)
    if not contains_script(parent.cmdline, supervisor_script):
        raise RuntimeError(f"worker parent is not the expected supervisor: pid={pid} ppid={proc.ppid}")
    if Path(one_arg(parent.cmdline, "--matrix-json")).resolve() != matrix_path:
        raise RuntimeError(f"supervisor matrix mismatch for pid={pid}")
    return WorkerIdentity(
        slot=slot,
        gpu_index=gpu_index,
        pid=proc.pid,
        pgid=proc.pgid,
        sid=proc.sid,
        ppid=proc.ppid,
        start_ticks=proc.start_ticks,
        state=proc.state,
    )


def discover_worker(
    *,
    slot: str,
    gpu_index: int,
    matrix_path: Path,
    worker_script: Path,
    supervisor_script: Path,
) -> tuple[WorkerIdentity | None, str]:
    candidate_pids = sorted(set(iter_worker_candidate_pids(slot)))
    if not candidate_pids:
        return None, "not_found"
    valid: list[WorkerIdentity] = []
    errors: list[str] = []
    for pid in candidate_pids:
        try:
            valid.append(
                structurally_validate_worker(
                    pid,
                    slot=slot,
                    gpu_index=gpu_index,
                    matrix_path=matrix_path,
                    worker_script=worker_script,
                    supervisor_script=supervisor_script,
                )
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, RuntimeError, ValueError) as exc:
            errors.append(f"pid={pid}:{type(exc).__name__}:{exc}")
    if len(valid) == 1 and len(candidate_pids) == 1:
        return valid[0], "valid"
    if len(valid) > 1:
        return None, "ambiguous_valid_candidates=" + ",".join(str(item.pid) for item in valid)
    return None, "invalid_candidates=" + "|".join(errors)


def run_csv_query(fields: tuple[str, ...], *, compute: bool = False) -> list[list[str]]:
    selector = "--query-compute-apps" if compute else "--query-gpu"
    result = subprocess.run(
        ["nvidia-smi", f"{selector}={','.join(fields)}", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi query failed: {result.stderr.strip()}")
    rows = [[cell.strip() for cell in row] for row in csv.reader(result.stdout.splitlines()) if row]
    if any(len(row) != len(fields) for row in rows):
        raise RuntimeError(f"unexpected nvidia-smi column count for fields={fields}: {rows}")
    return rows


def gpu_telemetry() -> dict[int, dict[str, str]]:
    rows = run_csv_query(GPU_QUERY_FIELDS)
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        values = dict(zip(GPU_QUERY_FIELDS, row))
        result[int(values["index"])] = values
    return result


def compute_gpu_by_pid() -> dict[int, str]:
    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    return {int(row[1]): row[0] for row in run_csv_query(fields, compute=True)}


def group_compute_validation(
    identity: WorkerIdentity,
    *,
    expected_gpu_uuid: str,
    gpu_by_pid: dict[int, str],
) -> tuple[bool, list[int], str]:
    group_rows: list[tuple[int, str]] = []
    for pid, gpu_uuid in gpu_by_pid.items():
        try:
            if os.getpgid(pid) == identity.pgid:
                group_rows.append((pid, gpu_uuid))
        except (ProcessLookupError, PermissionError):
            continue
    if not group_rows:
        return False, [], "no_live_compute_process_in_worker_group"
    wrong = [(pid, uuid) for pid, uuid in group_rows if uuid != expected_gpu_uuid]
    if wrong:
        return False, [pid for pid, _ in group_rows], f"group_compute_gpu_mismatch={wrong}"
    return True, sorted(pid for pid, _ in group_rows), "valid"


def same_generation(left: WorkerIdentity, right: WorkerIdentity) -> bool:
    return (
        left.slot == right.slot
        and left.gpu_index == right.gpu_index
        and left.pid == right.pid
        and left.pgid == right.pgid
        and left.sid == right.sid
        and left.start_ticks == right.start_ticks
    )


def validate_recorded_identity_for_cont(
    identity: WorkerIdentity,
    *,
    matrix_path: Path,
    worker_script: Path,
) -> ProcInfo:
    """Revalidate an exact worker generation before releasing its process group.

    This intentionally does not require the supervisor to remain alive.  If the
    supervisor exits while its worker is stopped, Linux may reparent the worker;
    requiring the old parent here could strand a group in SIGSTOP on guard exit.
    PID start time plus the original session/PGID and exact worker arguments
    still fail closed against PID/PGID reuse.
    """

    proc = read_proc(identity.pid)
    if proc.state in {"Z", "X", "x"}:
        raise RuntimeError(f"recorded worker pid={proc.pid} has terminal state={proc.state}")
    if (
        proc.pid != identity.pid
        or proc.start_ticks != identity.start_ticks
        or proc.pgid != identity.pgid
        or proc.sid != identity.sid
        or proc.pid != proc.pgid
        or proc.pid != proc.sid
    ):
        raise RuntimeError(
            "recorded worker generation changed: "
            f"pid={proc.pid} start={proc.start_ticks} pgid={proc.pgid} sid={proc.sid}"
        )
    if not contains_script(proc.cmdline, worker_script):
        raise RuntimeError(f"recorded worker script mismatch for pid={proc.pid}")
    if one_arg(proc.cmdline, "--worker-slot") != identity.slot:
        raise RuntimeError(f"recorded worker slot mismatch for pid={proc.pid}")
    if Path(one_arg(proc.cmdline, "--matrix-json")).resolve() != matrix_path:
        raise RuntimeError(f"recorded worker matrix mismatch for pid={proc.pid}")
    return proc


def append_event(path: Path, event: str, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now_iso(), "event": event, **values}, ensure_ascii=False) + "\n")
        handle.flush()


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def default_paths() -> tuple[Path, Path]:
    workspace = Path(__file__).resolve().parents[5]
    campaign = workspace / "local_runtime" / "runs" / "scaling" / "domainwise_scaling_fddfs_mf80000_v1"
    return campaign / "phase_b_matrix_local.json", campaign


def validate_matrix(matrix_path: Path) -> None:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("metadata", {}).get("status") != "runnable_phase_b":
        raise ValueError("matrix is not runnable_phase_b")
    if matrix["metadata"].get("host_id") != "local":
        raise ValueError("thermal guard accepts only the host-qualified local matrix")
    for gpu in GUARDED_GPUS:
        slot = f"local_gpu{gpu}_slot1"
        rows = [row for row in matrix.get("rows", []) if row.get("worker_slot") == slot]
        if not rows:
            raise ValueError(f"matrix has no secondary worker slot {slot}")
        if any(
            int(row.get("assigned_gpu", -1)) != gpu
            or int(row.get("gpu_local_slot", -1)) != 1
            or row.get("assigned_host") != "local"
            for row in rows
        ):
            raise ValueError(f"matrix mapping is inconsistent for {slot}")


def main() -> int:
    default_matrix, campaign = default_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, default=default_matrix)
    ap.add_argument("--output-csv", type=Path, default=campaign / "thermal_guard_local.csv")
    ap.add_argument("--event-jsonl", type=Path, default=campaign / "thermal_guard_local_events.jsonl")
    ap.add_argument("--pid-file", type=Path, default=campaign / "thermal_guard_local.pid")
    ap.add_argument("--interval-seconds", type=float, default=10.0)
    ap.add_argument("--pause-temperature-c", type=float, default=88.0)
    ap.add_argument("--resume-temperature-c", type=float, default=82.0)
    ap.add_argument("--consecutive-samples", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="Discover and log decisions but never send signals.")
    ap.add_argument("--once", action="store_true", help="Take exactly one sample and exit.")
    args = ap.parse_args()
    if args.interval_seconds <= 0 or args.consecutive_samples <= 0:
        raise ValueError("interval and consecutive sample count must be positive")
    if args.resume_temperature_c >= args.pause_temperature_c:
        raise ValueError("resume temperature must be below pause temperature")

    matrix_path = args.matrix_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    event_jsonl = args.event_jsonl.expanduser().resolve()
    pid_file = args.pid_file.expanduser().resolve()
    validate_matrix(matrix_path)
    worker_script = Path(__file__).resolve().with_name(WORKER_SCRIPT_NAME)
    supervisor_script = Path(__file__).resolve().with_name(SUPERVISOR_SCRIPT_NAME)
    if not worker_script.is_file() or not supervisor_script.is_file():
        raise FileNotFoundError("expected Phase-B worker/supervisor scripts are unavailable")

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = pid_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another thermal guard owns {pid_file}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    stop_event = threading.Event()
    paused: dict[int, WorkerIdentity] = {}
    unmanaged_stops_reported: set[tuple[int, int]] = set()
    hot_counts = {gpu: 0 for gpu in GUARDED_GPUS}
    cool_counts = {gpu: 0 for gpu in GUARDED_GPUS}
    received_signal: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_stop)

    def rediscover(gpu: int) -> tuple[WorkerIdentity | None, str]:
        return discover_worker(
            slot=f"local_gpu{gpu}_slot1",
            gpu_index=gpu,
            matrix_path=matrix_path,
            worker_script=worker_script,
            supervisor_script=supervisor_script,
        )

    def resume_identity(gpu: int, identity: WorkerIdentity, reason: str) -> bool:
        try:
            validate_recorded_identity_for_cont(
                identity,
                matrix_path=matrix_path,
                worker_script=worker_script,
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, RuntimeError, ValueError) as exc:
            append_event(
                event_jsonl,
                "resume_skipped_identity_changed",
                gpu=gpu,
                reason=reason,
                validation=f"{type(exc).__name__}: {exc}",
                recorded=asdict(identity),
            )
            return False
        if args.dry_run:
            append_event(event_jsonl, "would_sigcont", gpu=gpu, reason=reason, identity=asdict(identity))
            return True
        # Identity was re-read immediately above; PGID==PID and generation are
        # unchanged.  SIGCONT is the only signal emitted by this path.
        os.killpg(identity.pgid, signal.SIGCONT)
        append_event(event_jsonl, "sigcont_sent", gpu=gpu, reason=reason, identity=asdict(identity))
        return True

    append_event(
        event_jsonl,
        "guard_start",
        pid=os.getpid(),
        matrix=str(matrix_path),
        gpus=list(GUARDED_GPUS),
        interval_seconds=args.interval_seconds,
        pause_temperature_c=args.pause_temperature_c,
        resume_temperature_c=args.resume_temperature_c,
        consecutive_samples=args.consecutive_samples,
        dry_run=args.dry_run,
    )
    sample = 0
    exit_code = 0
    try:
        while not stop_event.is_set():
            sample += 1
            timestamp = now_iso()
            try:
                telemetry = gpu_telemetry()
                gpu_by_pid = compute_gpu_by_pid()
            except Exception as exc:
                append_event(event_jsonl, "nvidia_smi_error", error=f"{type(exc).__name__}: {exc}")
                if args.once:
                    exit_code = 1
                    break
                stop_event.wait(args.interval_seconds)
                continue

            rows: list[dict[str, Any]] = []
            for gpu in GUARDED_GPUS:
                if stop_event.is_set():
                    break
                decision = "observe"
                values = telemetry.get(gpu)
                if values is None:
                    append_event(event_jsonl, "gpu_missing", gpu=gpu)
                    continue
                temperature = float(values["temperature.gpu"])
                hot_counts[gpu] = hot_counts[gpu] + 1 if temperature >= args.pause_temperature_c else 0
                cool_counts[gpu] = cool_counts[gpu] + 1 if temperature <= args.resume_temperature_c else 0
                identity, worker_status = rediscover(gpu)
                group_compute_pids: list[int] = []
                if identity is not None:
                    compute_ok, group_compute_pids, compute_status = group_compute_validation(
                        identity,
                        expected_gpu_uuid=values["uuid"],
                        gpu_by_pid=gpu_by_pid,
                    )
                    if not compute_ok:
                        worker_status = compute_status

                recorded = paused.get(gpu)
                if recorded is not None:
                    if identity is None or not same_generation(recorded, identity):
                        append_event(
                            event_jsonl,
                            "paused_identity_disappeared",
                            gpu=gpu,
                            discovery=worker_status,
                            recorded=asdict(recorded),
                        )
                        paused.pop(gpu, None)
                        hot_counts[gpu] = 1 if temperature >= args.pause_temperature_c else 0
                        cool_counts[gpu] = 1 if temperature <= args.resume_temperature_c else 0
                        decision = "paused_identity_disappeared"
                    elif cool_counts[gpu] >= args.consecutive_samples:
                        decision = "resume_secondary"
                        if resume_identity(gpu, recorded, "cooled"):
                            paused.pop(gpu, None)
                            hot_counts[gpu] = 0
                            cool_counts[gpu] = 0
                elif identity is not None and identity.state in {"T", "t"}:
                    # SIGSTOP is not reference counted.  Never claim or later
                    # SIGCONT a process group that some other operator stopped.
                    decision = "preexisting_stop_unmanaged"
                    worker_status = "preexisting_stop_unmanaged"
                    hot_counts[gpu] = 0
                    cool_counts[gpu] = 0
                    generation_key = (identity.pid, identity.start_ticks)
                    if generation_key not in unmanaged_stops_reported:
                        unmanaged_stops_reported.add(generation_key)
                        append_event(
                            event_jsonl,
                            "preexisting_stop_unmanaged",
                            gpu=gpu,
                            identity=asdict(identity),
                        )
                elif hot_counts[gpu] >= args.consecutive_samples:
                    if identity is None or worker_status != "valid":
                        decision = "pause_skipped_fail_closed"
                        append_event(
                            event_jsonl,
                            "pause_skipped",
                            gpu=gpu,
                            temperature_c=temperature,
                            worker_status=worker_status,
                        )
                    else:
                        # Re-query both process identity and GPU ownership at the
                        # signal boundary; stale nvidia-smi/proc snapshots never
                        # authorize SIGSTOP.
                        current, current_status = rediscover(gpu)
                        fresh_gpu_by_pid = compute_gpu_by_pid()
                        valid_compute = False
                        fresh_compute_pids: list[int] = []
                        fresh_compute_status = "identity_changed"
                        if current is not None and same_generation(identity, current):
                            valid_compute, fresh_compute_pids, fresh_compute_status = group_compute_validation(
                                current,
                                expected_gpu_uuid=values["uuid"],
                                gpu_by_pid=fresh_gpu_by_pid,
                            )
                        if current is None or not same_generation(identity, current) or not valid_compute:
                            decision = "pause_skipped_revalidation_failed"
                            append_event(
                                event_jsonl,
                                "pause_revalidation_failed",
                                gpu=gpu,
                                discovery=current_status,
                                compute_status=fresh_compute_status,
                            )
                        elif stop_event.is_set():
                            decision = "pause_skipped_guard_exiting"
                        elif args.dry_run:
                            decision = "would_pause_secondary"
                            append_event(
                                event_jsonl,
                                "would_sigstop",
                                gpu=gpu,
                                temperature_c=temperature,
                                compute_pids=fresh_compute_pids,
                                identity=asdict(current),
                            )
                        else:
                            append_event(
                                event_jsonl,
                                "sigstop_intent",
                                gpu=gpu,
                                temperature_c=temperature,
                                compute_pids=fresh_compute_pids,
                                identity=asdict(current),
                            )
                            os.killpg(current.pgid, signal.SIGSTOP)
                            paused[gpu] = current
                            decision = "paused_secondary"
                            append_event(
                                event_jsonl,
                                "sigstop_sent",
                                gpu=gpu,
                                temperature_c=temperature,
                                compute_pids=fresh_compute_pids,
                                identity=asdict(current),
                            )

                rows.append(
                    {
                        "timestamp": timestamp,
                        "sample": sample,
                        "gpu_index": gpu,
                        "gpu_uuid": values["uuid"],
                        "temperature_c": values["temperature.gpu"],
                        "memory_used_mib": values["memory.used"],
                        "memory_total_mib": values["memory.total"],
                        "utilization_pct": values["utilization.gpu"],
                        "power_draw_w": values["power.draw"],
                        "graphics_clock_mhz": values["clocks.current.graphics"],
                        "throttle_active_mask": values["clocks_event_reasons.active"],
                        "sw_thermal_slowdown": values["clocks_event_reasons.sw_thermal_slowdown"],
                        "hw_thermal_slowdown": values["clocks_event_reasons.hw_thermal_slowdown"],
                        "hot_consecutive": hot_counts[gpu],
                        "cool_consecutive": cool_counts[gpu],
                        "secondary_slot": f"local_gpu{gpu}_slot1",
                        "worker_status": worker_status,
                        "worker_pid": identity.pid if identity else "",
                        "worker_pgid": identity.pgid if identity else "",
                        "worker_start_ticks": identity.start_ticks if identity else "",
                        "worker_state": identity.state if identity else "",
                        "group_compute_pids": ";".join(str(pid) for pid in group_compute_pids),
                        "guard_paused": gpu in paused,
                        "decision": decision,
                        "dry_run": args.dry_run,
                    }
                )
            append_csv(output_csv, rows)
            if args.once:
                break
            stop_event.wait(args.interval_seconds)
    except BaseException as exc:
        exit_code = 1
        try:
            append_event(event_jsonl, "guard_exception", error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
    finally:
        # Every process group actually paused by this guard is resumed on every
        # catchable exit path.  Identity validation prevents CONT reaching a
        # recycled or unrelated PGID.
        for gpu, identity in list(paused.items()):
            try:
                resume_identity(gpu, identity, "guard_exit")
            except Exception as exc:
                try:
                    append_event(
                        event_jsonl,
                        "exit_resume_error",
                        gpu=gpu,
                        error=f"{type(exc).__name__}: {exc}",
                        identity=asdict(identity),
                    )
                except Exception:
                    pass
            finally:
                paused.pop(gpu, None)
        try:
            append_event(
                event_jsonl,
                "guard_exit",
                pid=os.getpid(),
                samples=sample,
                exit_code=exit_code,
                signal=received_signal,
            )
        except Exception:
            pass
        try:
            lock_handle.seek(0)
            recorded = lock_handle.read().strip()
            if recorded == str(os.getpid()):
                pid_file.unlink(missing_ok=True)
        finally:
            lock_handle.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
