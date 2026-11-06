#!/usr/bin/env python3
"""Emergency thermal guard for host-qualified Phase-B primary worker groups.

This complements ``guard_local_secondary_slots_thermal.py``.  The secondary
guard sheds slot1 first.  If a 48 GB card is still at the thermal wall with
slot1 stopped, this guard briefly freezes slot0 and releases it after the card
has cooled.  CUDA memory and optimizer state stay resident; no job is killed.

Signals are fail-closed and target only a fully revalidated worker process
group from the frozen, host-qualified matrix.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from guard_local_secondary_slots_thermal import (
    SUPERVISOR_SCRIPT_NAME,
    WORKER_SCRIPT_NAME,
    WorkerIdentity,
    append_event,
    compute_gpu_by_pid,
    discover_worker,
    gpu_telemetry,
    group_compute_validation,
    now_iso,
    same_generation,
    validate_recorded_identity_for_cont,
)


DEFAULT_GPUS = (4, 5)


def default_paths() -> tuple[Path, Path]:
    workspace = Path(__file__).resolve().parents[5]
    campaign = workspace / "local_runtime" / "runs" / "scaling" / "domainwise_scaling_fddfs_mf80000_v1"
    return campaign / "phase_b_matrix_local.json", campaign


def parse_gpus(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError("--gpus must be a non-empty comma-separated list of unique nonnegative indices")
    return values


def validate_matrix(matrix_path: Path, gpus: tuple[int, ...]) -> str:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    metadata = matrix.get("metadata", {})
    host_id = str(metadata.get("host_id", ""))
    if metadata.get("status") != "runnable_phase_b" or not host_id:
        raise ValueError("primary thermal guard accepts only a host-qualified Phase-B matrix")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in host_id):
        raise ValueError(f"unsafe matrix host_id: {host_id!r}")
    for gpu in gpus:
        slot = f"{host_id}_gpu{gpu}_slot0"
        rows = [row for row in matrix.get("rows", []) if row.get("worker_slot") == slot]
        if not rows:
            raise ValueError(f"matrix has no primary worker slot {slot}")
        if any(
            int(row.get("assigned_gpu", -1)) != gpu
            or int(row.get("gpu_local_slot", -1)) != 0
            or row.get("assigned_host") != host_id
            for row in rows
        ):
            raise ValueError(f"matrix mapping is inconsistent for {slot}")
    return host_id


def main() -> int:
    default_matrix, campaign = default_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, default=default_matrix)
    ap.add_argument("--gpus", default=",".join(str(gpu) for gpu in DEFAULT_GPUS))
    ap.add_argument("--event-jsonl", type=Path, default=campaign / "thermal_guard_local_primary_events.jsonl")
    ap.add_argument("--pid-file", type=Path, default=campaign / "thermal_guard_local_primary.pid")
    ap.add_argument("--interval-seconds", type=float, default=5.0)
    ap.add_argument("--pause-temperature-c", type=float, default=90.0)
    ap.add_argument("--resume-temperature-c", type=float, default=84.0)
    ap.add_argument("--consecutive-hot-samples", type=int, default=1)
    ap.add_argument("--consecutive-cool-samples", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    if args.consecutive_hot_samples <= 0 or args.consecutive_cool_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.resume_temperature_c >= args.pause_temperature_c:
        raise ValueError("resume temperature must be below pause temperature")

    gpus = parse_gpus(args.gpus)
    matrix_path = args.matrix_json.expanduser().resolve()
    event_jsonl = args.event_jsonl.expanduser().resolve()
    pid_file = args.pid_file.expanduser().resolve()
    host_id = validate_matrix(matrix_path, gpus)
    worker_script = Path(__file__).resolve().with_name(WORKER_SCRIPT_NAME)
    supervisor_script = Path(__file__).resolve().with_name(SUPERVISOR_SCRIPT_NAME)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = pid_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another primary thermal guard owns {pid_file}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    stop_event = threading.Event()
    paused: dict[int, WorkerIdentity] = {}
    hot_counts = {gpu: 0 for gpu in gpus}
    cool_counts = {gpu: 0 for gpu in gpus}
    unmanaged_reported: set[tuple[int, int]] = set()
    received_signal: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_stop)

    def rediscover(gpu: int) -> tuple[WorkerIdentity | None, str]:
        return discover_worker(
            slot=f"{host_id}_gpu{gpu}_slot0",
            gpu_index=gpu,
            matrix_path=matrix_path,
            worker_script=worker_script,
            supervisor_script=supervisor_script,
        )

    def resume(gpu: int, identity: WorkerIdentity, reason: str) -> bool:
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
                error=f"{type(exc).__name__}: {exc}",
                identity=asdict(identity),
            )
            return False
        if args.dry_run:
            append_event(event_jsonl, "would_sigcont_primary", gpu=gpu, reason=reason, identity=asdict(identity))
            return True
        os.killpg(identity.pgid, signal.SIGCONT)
        append_event(event_jsonl, "sigcont_primary_sent", gpu=gpu, reason=reason, identity=asdict(identity))
        return True

    append_event(
        event_jsonl,
        "primary_guard_start",
        pid=os.getpid(),
        matrix=str(matrix_path),
        host_id=host_id,
        gpus=list(gpus),
        interval_seconds=args.interval_seconds,
        pause_temperature_c=args.pause_temperature_c,
        resume_temperature_c=args.resume_temperature_c,
        consecutive_hot_samples=args.consecutive_hot_samples,
        consecutive_cool_samples=args.consecutive_cool_samples,
        dry_run=args.dry_run,
    )
    samples = 0
    try:
        while not stop_event.is_set():
            samples += 1
            try:
                telemetry = gpu_telemetry()
                gpu_by_pid = compute_gpu_by_pid()
            except Exception as exc:
                append_event(event_jsonl, "telemetry_error", error=f"{type(exc).__name__}: {exc}")
                if args.once:
                    return 1
                stop_event.wait(args.interval_seconds)
                continue

            for gpu in gpus:
                values = telemetry.get(gpu)
                if values is None:
                    append_event(event_jsonl, "gpu_missing", gpu=gpu)
                    continue
                temperature = float(values["temperature.gpu"])
                hot_counts[gpu] = hot_counts[gpu] + 1 if temperature >= args.pause_temperature_c else 0
                cool_counts[gpu] = cool_counts[gpu] + 1 if temperature <= args.resume_temperature_c else 0
                recorded = paused.get(gpu)
                decision = "observe"

                if recorded is not None:
                    if cool_counts[gpu] >= args.consecutive_cool_samples:
                        if resume(gpu, recorded, "cooled"):
                            paused.pop(gpu, None)
                            hot_counts[gpu] = 0
                            cool_counts[gpu] = 0
                            decision = "resumed_primary"
                    else:
                        decision = "primary_paused"
                elif hot_counts[gpu] >= args.consecutive_hot_samples:
                    identity, worker_status = rediscover(gpu)
                    if identity is None or worker_status != "valid":
                        decision = "pause_skipped_fail_closed"
                        append_event(
                            event_jsonl,
                            "pause_primary_skipped",
                            gpu=gpu,
                            temperature_c=temperature,
                            worker_status=worker_status,
                        )
                    elif identity.state in {"T", "t"}:
                        decision = "preexisting_stop_unmanaged"
                        key = (identity.pid, identity.start_ticks)
                        if key not in unmanaged_reported:
                            unmanaged_reported.add(key)
                            append_event(event_jsonl, decision, gpu=gpu, identity=asdict(identity))
                    else:
                        compute_ok, compute_pids, compute_status = group_compute_validation(
                            identity,
                            expected_gpu_uuid=values["uuid"],
                            gpu_by_pid=gpu_by_pid,
                        )
                        current, current_status = rediscover(gpu)
                        fresh_gpu_by_pid = compute_gpu_by_pid()
                        fresh_ok = False
                        fresh_pids: list[int] = []
                        fresh_status = "identity_changed"
                        if current is not None and same_generation(identity, current):
                            fresh_ok, fresh_pids, fresh_status = group_compute_validation(
                                current,
                                expected_gpu_uuid=values["uuid"],
                                gpu_by_pid=fresh_gpu_by_pid,
                            )
                        if not compute_ok or current is None or not same_generation(identity, current) or not fresh_ok:
                            decision = "pause_revalidation_failed"
                            append_event(
                                event_jsonl,
                                decision,
                                gpu=gpu,
                                initial_compute_status=compute_status,
                                current_status=current_status,
                                fresh_compute_status=fresh_status,
                            )
                        elif args.dry_run:
                            decision = "would_pause_primary"
                            append_event(
                                event_jsonl,
                                decision,
                                gpu=gpu,
                                temperature_c=temperature,
                                compute_pids=fresh_pids,
                                identity=asdict(current),
                            )
                        else:
                            append_event(
                                event_jsonl,
                                "sigstop_primary_intent",
                                gpu=gpu,
                                temperature_c=temperature,
                                compute_pids=fresh_pids,
                                identity=asdict(current),
                            )
                            os.killpg(current.pgid, signal.SIGSTOP)
                            paused[gpu] = current
                            decision = "paused_primary"
                            append_event(
                                event_jsonl,
                                "sigstop_primary_sent",
                                gpu=gpu,
                                temperature_c=temperature,
                                compute_pids=fresh_pids,
                                identity=asdict(current),
                            )

                append_event(
                    event_jsonl,
                    "primary_guard_sample",
                    sample=samples,
                    gpu=gpu,
                    temperature_c=temperature,
                    utilization_pct=values["utilization.gpu"],
                    power_draw_w=values["power.draw"],
                    sw_thermal_slowdown=values["clocks_event_reasons.sw_thermal_slowdown"],
                    guard_paused=gpu in paused,
                    hot_consecutive=hot_counts[gpu],
                    cool_consecutive=cool_counts[gpu],
                    decision=decision,
                )
            if args.once:
                break
            stop_event.wait(args.interval_seconds)
    finally:
        for gpu, identity in list(paused.items()):
            try:
                resume(gpu, identity, "guard_exit")
            except Exception as exc:
                append_event(
                    event_jsonl,
                    "exit_resume_error",
                    gpu=gpu,
                    error=f"{type(exc).__name__}: {exc}",
                    identity=asdict(identity),
                )
            finally:
                paused.pop(gpu, None)
        append_event(
            event_jsonl,
            "primary_guard_exit",
            pid=os.getpid(),
            samples=samples,
            signal=received_signal,
            time=now_iso(),
        )
        try:
            lock_handle.seek(0)
            if lock_handle.read().strip() == str(os.getpid()):
                pid_file.unlink(missing_ok=True)
        finally:
            lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
