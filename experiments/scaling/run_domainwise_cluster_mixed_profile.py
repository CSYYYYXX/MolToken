#!/usr/bin/env python3
"""Profile the actual mixture of completion points assigned to each H200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_domainwise_cluster_completion import (
    atomic_json, load_json, parse_device_indices, validate_manifest,
    visible_cuda_device_count,
)
from run_domainwise_cluster_profile import run_wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("python"))
    parser.add_argument("--pipeline-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device-indices", required=True)
    parser.add_argument("--global-gpu-offset", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.python = args.python.absolute()
    args.pipeline_root = args.pipeline_root.absolute()
    devices = parse_device_indices(args.device_indices)
    if args.steps < 2 or max(devices) >= visible_cuda_device_count(args.python):
        raise ValueError("profile requires two optimizer steps and valid allocated CUDA indices")
    plan = load_json(args.plan)
    rows = [row for row in plan["rows"] if args.global_gpu_offset <= row["global_gpu_index"] < args.global_gpu_offset + len(devices)]
    if not rows:
        raise ValueError("no runs assigned to this allocation")
    assigned_devices = [devices[row["global_gpu_index"] - args.global_gpu_offset] for row in rows]
    args.manifest = Path(plan["domain_split_manifest"])
    manifest = validate_manifest(args.manifest, rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out_dir / "placement.json", {
        "plan": str(args.plan), "global_gpu_offset": args.global_gpu_offset,
        "runs": [{"run_id": row["run_id"], "device": device} for row, device in zip(rows, assigned_devices)],
    })
    wave = run_wave(args, manifest, rows, assigned_devices, 1, args.out_dir / "mixed")
    observed_peaks = {device: 0 for device in devices}
    for line in (args.out_dir / "mixed" / "gpu_telemetry.jsonl").read_text().splitlines():
        for sample in json.loads(line)["gpu_csv"]:
            fields = sample.split(",")
            device = int(fields[0])
            if device in observed_peaks:
                observed_peaks[device] = max(observed_peaks[device], int(fields[2]))
    per_device = []
    for device in devices:
        results = [result for result in wave["results"] if result["device"] == device]
        total_mib = results[0]["memory"]["device_total_mib"]
        sum_reserved = sum(result["memory"]["peak_reserved_mib"] for result in results)
        per_device.append({
            "device": device, "runs": len(results), "total_mib": total_mib,
            "sum_process_peak_reserved_mib": sum_reserved,
            "sum_peak_percent": sum_reserved / total_mib * 100,
            "observed_nvidia_smi_peak_used_mib": observed_peaks[device],
            "sum_asynchronous_last_update_rates_not_steady_tps": sum(result["input_tokens_per_sec"] for result in results),
            "input_tokens_per_sec_including_startup": sum(
                result["summary"]["actual_input_tokens"] for result in results
            ) / wave["wall_time_sec"],
        })
    report = {
        "status": "complete", "per_device": per_device, "wave": wave,
        "measurement_notes": {
            "sum_process_peaks": "Peaks occur at different times; this is not observed device utilization and excludes CUDA contexts.",
            "throughput": "Asynchronous last-update rates are not steady aggregate throughput. The wall-time rate includes process startup and teardown.",
        },
    }
    atomic_json(args.out_dir / "result.json", report)
    print({"status": "complete", "per_device": per_device}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
