#!/usr/bin/env python3
"""Lease otherwise-idle GPU memory while a campaign controller stays in one step.

This is a resource-reservation helper, not a training process.  It deliberately
does no recurring CUDA work and releases its allocation as soon as the bound
controller identity or state no longer matches.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


GIB = 1024**3


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def process_start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def identity_alive(pid: int, start_ticks: int) -> bool:
    return process_start_ticks(pid) == start_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--active-gib", type=float, required=True)
    parser.add_argument("--idle-gib", type=float, required=True)
    parser.add_argument("--safety-gib", type=float, default=3.0)
    parser.add_argument("--controller-pid-json", type=Path, required=True)
    parser.add_argument("--controller-state-json", type=Path, required=True)
    parser.add_argument("--expected-step", required=True)
    parser.add_argument("--active-supervisor-pid", type=int, required=True)
    parser.add_argument("--active-supervisor-start-ticks", type=int, required=True)
    parser.add_argument("--identity-json", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.active_gib, args.idle_gib) < 0:
        raise SystemExit("lease sizes must be non-negative")
    if args.safety_gib < 1:
        raise SystemExit("--safety-gib must be at least 1 GiB")

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock_file.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"lease lock already held: {args.lock_file}")

    controller_identity = read_json(args.controller_pid_json)
    controller_pid = int(controller_identity["pid"])
    controller_start_ticks = int(controller_identity["start_ticks"])
    campaign_generation_id = str(controller_identity["campaign_generation_id"])
    if not identity_alive(controller_pid, controller_start_ticks):
        raise SystemExit("bound controller identity is not alive")

    # CUDA visibility must be fixed before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch  # noqa: PLC0415

    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    tensor: Any = None
    allocated_gib = 0.0
    mode = "starting"
    started_at = now()

    def controller_matches() -> bool:
        if not identity_alive(controller_pid, controller_start_ticks):
            return False
        live_identity = read_json(args.controller_pid_json)
        if (
            int(live_identity.get("pid", -1)) != controller_pid
            or int(live_identity.get("start_ticks", -1)) != controller_start_ticks
            or str(live_identity.get("campaign_generation_id"))
            != campaign_generation_id
        ):
            return False
        state = read_json(args.controller_state_json)
        return (
            state.get("step") == args.expected_step
            and not bool(state.get("terminal"))
            and str(state.get("campaign_generation_id")) == campaign_generation_id
        )

    def publish(event: str, error: str | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": "holomol_gpu_memory_lease_v1",
            "event": event,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "start_ticks": process_start_ticks(os.getpid()),
            "physical_gpu": args.physical_gpu,
            "mode": mode,
            "allocated_gib": round(allocated_gib, 3),
            "active_gib": args.active_gib,
            "idle_gib": args.idle_gib,
            "safety_gib": args.safety_gib,
            "controller_pid": controller_pid,
            "controller_start_ticks": controller_start_ticks,
            "campaign_generation_id": campaign_generation_id,
            "expected_step": args.expected_step,
            "active_supervisor_pid": args.active_supervisor_pid,
            "active_supervisor_start_ticks": args.active_supervisor_start_ticks,
            "started_at": started_at,
            "heartbeat_at": now(),
        }
        if error is not None:
            payload["error"] = error
        write_json_atomic(args.identity_json, payload)

    def allocate(requested_gib: float) -> tuple[Any, float]:
        if requested_gib <= 0:
            return None, 0.0
        torch.cuda.set_device(0)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
        cap_bytes = max(0, free_bytes - int(args.safety_gib * GIB))
        wanted_bytes = min(int(requested_gib * GIB), cap_bytes)
        # Step down on fragmentation instead of endangering a real job.
        while wanted_bytes >= GIB:
            try:
                value = torch.empty(wanted_bytes, dtype=torch.uint8, device="cuda:0")
                return value, wanted_bytes / GIB
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                wanted_bytes -= GIB
        return None, 0.0

    consecutive_state_errors = 0
    try:
        while not stopping:
            try:
                if not controller_matches():
                    mode = "released_controller_changed"
                    break
                consecutive_state_errors = 0
            except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, ValueError):
                consecutive_state_errors += 1
                if consecutive_state_errors >= 3:
                    mode = "released_state_unreadable"
                    break
                time.sleep(args.interval_seconds)
                continue

            supervisor_alive = identity_alive(
                args.active_supervisor_pid,
                args.active_supervisor_start_ticks,
            )
            desired_mode = "active_training" if supervisor_alive else "idle_reserved"
            desired_gib = args.active_gib if supervisor_alive else args.idle_gib
            if desired_mode != mode:
                tensor = None
                allocated_gib = 0.0
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                tensor, allocated_gib = allocate(desired_gib)
                mode = desired_mode
                publish("allocated")
            else:
                publish("heartbeat")
            time.sleep(args.interval_seconds)
    except Exception as exc:
        mode = "released_error"
        publish("error", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        tensor = None
        allocated_gib = 0.0
        if "torch" in locals() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        publish("released")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
