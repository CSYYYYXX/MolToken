#!/usr/bin/env python3
"""Append lightweight physical-GPU telemetry while local scaling jobs run.

The monitor is deliberately observational: sustained high temperature produces
warnings, but it never signals or kills a training process.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


NVIDIA_FIELDS = (
    "index",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "pstate",
    "clocks_event_reasons.active",
    "clocks_event_reasons.gpu_idle",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)

CSV_FIELDS = (
    "timestamp",
    "sample",
    "gpu_index",
    "gpu_name",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_gpu_pct",
    "temperature_gpu_c",
    "power_draw_w",
    "power_limit_w",
    "clock_graphics_mhz",
    "clock_sm_mhz",
    "clock_memory_mhz",
    "pstate",
    "throttle_active_mask",
    "throttle_gpu_idle",
    "throttle_sw_power_cap",
    "throttle_hw_slowdown",
    "throttle_hw_thermal_slowdown",
    "throttle_sw_thermal_slowdown",
    "consecutive_hot_samples",
    "warning",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_number(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return None


def query_nvidia_smi() -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(completed.stdout.splitlines())
        if row
    ]
    for row in rows:
        if len(row) != len(NVIDIA_FIELDS):
            raise RuntimeError(
                f"nvidia-smi returned {len(row)} columns, expected {len(NVIDIA_FIELDS)}: {row}"
            )
    return rows


def telemetry_row(
    raw: list[str],
    *,
    timestamp: str,
    sample: int,
    consecutive_hot_samples: int,
    warning: str,
) -> dict[str, Any]:
    values = dict(zip(NVIDIA_FIELDS, raw))
    return {
        "timestamp": timestamp,
        "sample": sample,
        "gpu_index": values["index"],
        "gpu_name": values["name"],
        "memory_used_mib": values["memory.used"],
        "memory_total_mib": values["memory.total"],
        "utilization_gpu_pct": values["utilization.gpu"],
        "temperature_gpu_c": values["temperature.gpu"],
        "power_draw_w": values["power.draw"],
        "power_limit_w": values["power.limit"],
        "clock_graphics_mhz": values["clocks.current.graphics"],
        "clock_sm_mhz": values["clocks.current.sm"],
        "clock_memory_mhz": values["clocks.current.memory"],
        "pstate": values["pstate"],
        "throttle_active_mask": values["clocks_event_reasons.active"],
        "throttle_gpu_idle": values["clocks_event_reasons.gpu_idle"],
        "throttle_sw_power_cap": values["clocks_event_reasons.sw_power_cap"],
        "throttle_hw_slowdown": values["clocks_event_reasons.hw_slowdown"],
        "throttle_hw_thermal_slowdown": values[
            "clocks_event_reasons.hw_thermal_slowdown"
        ],
        "throttle_sw_thermal_slowdown": values[
            "clocks_event_reasons.sw_thermal_slowdown"
        ],
        "consecutive_hot_samples": consecutive_hot_samples,
        "warning": warning,
    }


def append_warning(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_pid_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            previous_pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            previous_pid = -1
        if pid_is_running(previous_pid):
            raise RuntimeError(f"GPU monitor already running with pid={previous_pid}: {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(f"{os.getpid()}\n", encoding="utf-8")
    os.replace(temporary, path)


def release_pid_file(path: Path) -> None:
    try:
        recorded_pid = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return
    if recorded_pid == os.getpid():
        path.unlink(missing_ok=True)


def default_output_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[5]
    return workspace_root / "local_runtime" / "logs" / "scaling" / "gpu_monitor.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-csv", type=Path, default=default_output_path())
    ap.add_argument("--warning-log", type=Path, default=None)
    ap.add_argument("--pid-file", type=Path, default=None)
    ap.add_argument("--interval-seconds", type=float, default=30.0)
    ap.add_argument("--temperature-threshold-c", type=float, default=88.0)
    ap.add_argument("--consecutive-hot-samples", type=int, default=2)
    ap.add_argument("--once", action="store_true", help="Take one sample and exit (for self-test).")
    args = ap.parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    if args.consecutive_hot_samples <= 0:
        raise ValueError("--consecutive-hot-samples must be positive")

    output_path = args.output_csv.expanduser().resolve()
    warning_path = (
        args.warning_log.expanduser().resolve()
        if args.warning_log is not None
        else output_path.with_name(output_path.stem + ".warnings.log")
    )
    pid_path = args.pid_file.expanduser().resolve() if args.pid_file is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pid_path is not None:
        claim_pid_file(pid_path)

    stop_event = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        print(f"{now_iso()} INFO received signal={signum}; stopping GPU monitor", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    hot_counts: dict[str, int] = {}
    sample_number = 0
    try:
        print(
            f"{now_iso()} INFO GPU monitor started pid={os.getpid()} output={output_path} "
            f"interval_sec={args.interval_seconds:g} threshold_c={args.temperature_threshold_c:g}",
            flush=True,
        )
        while not stop_event.is_set():
            sample_number += 1
            timestamp = now_iso()
            try:
                raw_rows = query_nvidia_smi()
            except Exception as exc:
                message = f"{timestamp} WARN nvidia-smi sample failed: {type(exc).__name__}: {exc}"
                append_warning(warning_path, message)
                print(message, file=sys.stderr, flush=True)
                if args.once:
                    return 1
            else:
                rows: list[dict[str, Any]] = []
                for raw in raw_rows:
                    gpu_index = raw[0]
                    temperature = parse_number(raw[5])
                    if temperature is not None and temperature >= args.temperature_threshold_c:
                        hot_counts[gpu_index] = hot_counts.get(gpu_index, 0) + 1
                    else:
                        hot_counts[gpu_index] = 0
                    hot_count = hot_counts[gpu_index]
                    warning = ""
                    if hot_count >= args.consecutive_hot_samples:
                        warning = "WARN_SUSTAINED_HIGH_TEMPERATURE"
                        message = (
                            f"{timestamp} WARN gpu={gpu_index} temperature_c={raw[5]} "
                            f"threshold_c={args.temperature_threshold_c:g} "
                            f"consecutive_samples={hot_count} utilization_pct={raw[4]} "
                            f"power_draw_w={raw[6]} throttle_active={raw[12]}"
                        )
                        append_warning(warning_path, message)
                        print(message, file=sys.stderr, flush=True)
                    rows.append(
                        telemetry_row(
                            raw,
                            timestamp=timestamp,
                            sample=sample_number,
                            consecutive_hot_samples=hot_count,
                            warning=warning,
                        )
                    )

                needs_header = not output_path.exists() or output_path.stat().st_size == 0
                with output_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                    if needs_header:
                        writer.writeheader()
                    writer.writerows(rows)
                    handle.flush()
            if args.once:
                break
            stop_event.wait(args.interval_seconds)
    finally:
        if pid_path is not None:
            release_pid_file(pid_path)
    print(f"{now_iso()} INFO GPU monitor stopped samples={sample_number}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
