#!/usr/bin/env python3
"""Measure concurrent independent trainers on scheduler-assigned CUDA devices."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from run_domainwise_cluster_completion import (
    atomic_json,
    load_json,
    parse_device_indices,
    single_process_environment,
    training_command,
    validate_manifest,
    validate_plan,
    visible_cuda_device_count,
)


def representatives(rows: list[dict]) -> list[dict]:
    """Cover the largest model in each domain and microbatch/GA combination."""
    groups: dict[tuple, dict] = {}
    for row in rows:
        key = (row["domain"], row["per_device_train_batch_size"], row["gradient_accumulation_steps"])
        previous = groups.get(key)
        if previous is None or row["parameter_count"] > previous["parameter_count"]:
            groups[key] = row
    return [groups[key] for key in sorted(groups)]


def run_wave(args, manifest, rows, devices, concurrency, wave_dir):
    wave_dir.mkdir(parents=True, exist_ok=False)
    children = []
    telemetry_path = wave_dir / "gpu_telemetry.jsonl"
    started = time.monotonic()
    try:
        for device, row in zip(devices, rows):
            for replica in range(concurrency):
                output = wave_dir / f"cuda{device}_{row['run_id']}_replica{replica}"
                output.mkdir()
                base = training_command(
                    python=args.python,
                    pipeline_root=args.pipeline_root,
                    manifest=manifest,
                    manifest_path=args.manifest,
                    row=row,
                    attempt_dir=output,
                    num_workers=args.num_workers,
                    profile_steps=args.steps,
                )
                base[base.index("--log-every") + 1] = "1"
                command = [
                    str(args.python),
                    str(args.pipeline_root / "experiments/scaling/run_profile_smoke_train_wrapper.py"),
                    "--memory-report", str(output / "cuda_memory.json"), "--", *base[2:],
                ]
                environment = single_process_environment(
                    dict(os.environ), device_index=device, pipeline_root=args.pipeline_root
                )
                atomic_json(output / "profile_config.json", {
                    "run_id": row["run_id"], "device": device,
                    "concurrency": concurrency, "replica": replica,
                    "microbatch": row["per_device_train_batch_size"],
                    "gradient_accumulation_steps": row["gradient_accumulation_steps"],
                    "optimizer_steps": args.steps, "command": command,
                })
                log = (output / "console.log").open("w", buffering=1)
                process = subprocess.Popen(command, env=environment, cwd=args.pipeline_root,
                                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                children.append((process, log, output, device, row))
                print(f"started K={concurrency} cuda={device} {row['run_id']} replica={replica} pid={process.pid}", flush=True)
        with telemetry_path.open("w", buffering=1) as telemetry:
            while any(process.poll() is None for process, *_ in children):
                sample = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    check=True, text=True, capture_output=True, timeout=15,
                )
                telemetry.write(json.dumps({
                    "elapsed_sec": time.monotonic() - started,
                    "gpu_csv": sample.stdout.strip().splitlines(),
                    "active_processes": [p.pid for p, *_ in children if p.poll() is None],
                }) + "\n")
                time.sleep(2)
    finally:
        # Let already launched experiments finish; never terminate a live worker.
        for process, log, *_ in children:
            process.wait()
            log.close()

    failed = [(str(output), process.returncode) for process, _, output, *_ in children if process.returncode]
    if failed:
        atomic_json(wave_dir / "failure.json", {"failed_processes": failed})
        raise RuntimeError(f"profile subprocesses failed; see their console.log files: {failed}")

    results = []
    for _, _, output, device, row in children:
        summary = load_json(output / "summary.json")
        memory = load_json(output / "cuda_memory.json")
        logs = [json.loads(line) for line in (output / "train_log.jsonl").read_text().splitlines() if line.strip()]
        updates = [entry for entry in logs if entry.get("event") == "train"]
        if len(updates) != args.steps or any(not math.isfinite(float(entry["loss"])) for entry in updates):
            raise RuntimeError(f"missing optimizer updates or non-finite loss: {output}")
        if any(int(entry["optimizer_micro_steps"]) != int(row["gradient_accumulation_steps"]) for entry in updates):
            raise RuntimeError(f"profile did not finish complete gradient accumulation windows: {output}")
        if int(memory["visible_cuda_device"]) != device:
            raise RuntimeError(f"profile ran on the wrong allocated device: {output}")
        # The trainer resets CUDA peaks after each log, so include every update.
        memory["peak_measurement"] = "maximum_of_per_update_logs_and_final_cuda_stats"
        for field in ("allocated", "reserved"):
            memory[f"peak_{field}_mib"] = max(
                memory[f"peak_{field}_mib"],
                *(entry["cuda_memory_gb_rank0"][f"max_{field}"] * 1024 for entry in updates),
            )
        results.append({
            "run_id": row["run_id"], "device": device,
            "output": str(output), "memory": memory,
            "input_tokens_per_sec": updates[-1]["input_tokens_per_sec_global_approx"],
            "updates": updates, "summary": summary,
        })
    result = {"concurrency": concurrency, "wall_time_sec": time.monotonic() - started, "results": results}
    atomic_json(wave_dir / "result.json", result)
    print(f"completed K={concurrency} {wave_dir.name} processes={len(results)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device-indices", default="0")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.pipeline_root = args.pipeline_root.absolute()
    # Do not resolve a venv Python symlink into its base interpreter.
    args.python = args.python.absolute()
    if args.steps < 2:
        raise ValueError("profile needs at least two optimizer steps to include Adam state")
    levels = [int(item) for item in args.concurrency.split(",")]
    if not levels or any(level < 1 for level in levels):
        raise ValueError("concurrency levels must be positive")
    devices = parse_device_indices(args.device_indices)
    if max(devices) >= visible_cuda_device_count(args.python):
        raise ValueError("device index is outside scheduler-assigned CUDA visibility")
    plan = load_json(args.plan)
    rows = representatives(validate_plan(plan))
    args.manifest = Path(plan["domain_split_manifest"])
    manifest = validate_manifest(args.manifest, rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out_dir / "profile_plan.json", {
        "devices": devices, "concurrency": levels, "steps": args.steps,
        "representative_run_ids": [row["run_id"] for row in rows],
    })
    completed = []
    for level in levels:
        for offset in range(0, len(rows), len(devices)):
            wave = run_wave(args, manifest, rows[offset:offset + len(devices)], devices,
                            level, args.out_dir / f"k{level}_wave{offset // len(devices)}")
            completed.append(wave)
            atomic_json(args.out_dir / "progress.json", {"completed_waves": completed})
    atomic_json(args.out_dir / "result.json", {"status": "complete", "waves": completed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
