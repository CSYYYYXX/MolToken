#!/usr/bin/env python3
"""Run one complete optimizer update in every frozen Phase-B worker slot."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_domainwise_phase_b_slot import training_command


def replace_arg(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def main() -> int:
    script_path = Path(__file__).resolve()
    pipeline_root = script_path.parents[2]
    workspace = pipeline_root.parents[2]
    default_matrix = workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1/phase_b_matrix_local.json"
    default_python = workspace / "local_runtime/envs/holomol-py311-cu121/bin/python"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, default=default_matrix)
    ap.add_argument("--python", type=Path, default=default_python)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument(
        "--selection",
        choices=("largest", "first"),
        default="largest",
        help="Select the largest model per slot or the first formal run per slot.",
    )
    args = ap.parse_args()
    matrix_path = args.matrix_json.expanduser().resolve()
    python = args.python.expanduser().absolute()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(matrix["metadata"]["domain_split_manifest"]).read_text(encoding="utf-8"))
    by_slot: dict[str, dict] = {}
    # Largest is the stronger generic memory test.  First reproduces the exact
    # 12-way workload with which the formal supervisor will start.
    for row in matrix["rows"]:
        slot = row["worker_slot"]
        if slot not in by_slot or (
            args.selection == "largest"
            and int(row["parameter_count"]) > int(by_slot[slot]["parameter_count"])
        ):
            by_slot[slot] = row
    expected_slots = sum(
        int(count) for count in matrix["metadata"]["gpu_layout"].values()
    )
    if len(by_slot) != expected_slots:
        raise ValueError(
            f"expected {expected_slots} worker slots from matrix layout, got {len(by_slot)}"
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1" / f"smoke_12slot_{stamp}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    processes: dict[str, tuple[subprocess.Popen[bytes], object, Path, dict]] = {}
    base_environment = os.environ.copy()
    base_environment.update(
        {
            "PYTHONPATH": str(pipeline_root),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    for slot, source in sorted(by_slot.items()):
        row = dict(source)
        # 95% of the expected update tokens lands safely inside micro-step GA
        # for these large batches: it exercises sample-aligned trimming on the
        # last microbatch while still completing exactly one full GA window.
        row["target_input_tokens_with_eos"] = int(
            math.floor(float(row["expected_input_tokens_per_optimizer_step"]) * 0.95)
        )
        row["scheduler_total_steps"] = 1
        attempt_dir = output_root / slot
        attempt_dir.mkdir()
        command = training_command(python, pipeline_root, manifest, row, attempt_dir, 2, 2e-4)
        replace_arg(command, "--final-eval-batches", "2")
        environment = base_environment.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(row["assigned_gpu"])
        log_handle = (attempt_dir / "console.log").open("wb", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=pipeline_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes[slot] = (process, log_handle, attempt_dir, row)
        print(json.dumps({"event": "spawn", "slot": slot, "gpu": row["assigned_gpu"], "pid": process.pid}), flush=True)

    failures: dict[str, str] = {}
    results: dict[str, dict] = {}
    for slot, (process, log_handle, attempt_dir, row) in processes.items():
        code = process.wait()
        log_handle.close()
        summary_path = attempt_dir / "summary.json"
        if code != 0 or not summary_path.is_file():
            failures[slot] = f"returncode={code} log={attempt_dir / 'console.log'}"
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        target = int(row["target_input_tokens_with_eos"])
        checks = {
            "global_step_one": int(summary.get("global_step", -1)) == 1,
            "full_gradient_window": int(summary.get("final_update_micro_steps", -1))
            == int(row["gradient_accumulation_steps"]),
            "target_reached": int(summary.get("actual_input_tokens", -1)) >= target,
            "overshoot_bounded": 0 <= int(summary.get("input_token_overshoot", -1)) < int(row["context_length"]),
            "partial_validation_two_batches": int((summary.get("final_validation") or {}).get("batches", -1)) == 2,
        }
        results[slot] = {
            "gpu": row["assigned_gpu"],
            "model_id": row["model_id"],
            "domain": row["domain"],
            "target": target,
            "actual": summary.get("actual_input_tokens"),
            "peak": summary.get("cuda_memory_gb_rank0"),
            "checks": checks,
        }
        if not all(checks.values()):
            failures[slot] = f"summary checks failed: {checks}"
    report = {"output_root": str(output_root), "results": results, "failures": failures}
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
