#!/usr/bin/env python3
"""Run an explicitly profiled recovery slice without touching existing queues.

Each allocated GPU initializes one trainer at a time. After its first logged
training update, another row on that GPU may start; existing trainers continue
concurrently. No profiling, worker allocation, signals, or queue edits occur.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import threading
import time

from run_domainwise_cluster_completion import (
    append_jsonl, load_json, parse_device_indices, run_one, validate_manifest,
    visible_cuda_device_count,
)
from run_domainwise_supplement_queue import formal_context_ready


def validate_recovery_rows(plan, devices):
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("recovery plan must contain nonempty rows")
    if "row_count" in plan and plan["row_count"] != len(rows):
        raise ValueError("recovery plan row_count differs from rows")
    seen_ids, seen_paths = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("recovery row must be an object")
        run_id = row["run_id"]
        if not isinstance(run_id, str) or not run_id or "/" in run_id or run_id in seen_ids:
            raise ValueError(f"unsafe or duplicate recovery run ID: {run_id!r}")
        seen_ids.add(run_id)
        device = row["recovery_device_index"]
        if isinstance(device, bool) or not isinstance(device, int) or device not in devices:
            raise ValueError(f"recovery device is outside selected allocation: {run_id}: {device}")
        attempt = int(row["attempt_number"])
        run_dir = Path(row["run_dir"]).expanduser().resolve()
        attempt_dir = Path(row["attempt_dir"]).expanduser().resolve()
        if (attempt < 1 or run_dir.name != run_id or row["attempt_name"] != f"attempt_{attempt:03d}"
                or attempt_dir != run_dir / row["attempt_name"] or attempt_dir in seen_paths):
            raise ValueError(f"invalid recovery attempt routing: {run_id}")
        seen_paths.add(attempt_dir)
        if attempt_dir.exists() or (run_dir / "complete.json").exists():
            raise FileExistsError(f"recovery must use a fresh incomplete run attempt: {attempt_dir}")
        if (row.get("random_initialization") is not True
                or row.get("init_checkpoint") is not None
                or row.get("restart_from_step") != 0
                or row.get("world_size") != 1):
            raise ValueError(f"fresh random-initialization contract changed: {run_id}")
        if row.get("source_status") not in {
            "not_started", "interrupted_restart_required", "adaptive_supplement",
        }:
            raise ValueError(f"unsupported recovery source_status: {run_id}")
        seed = row["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"invalid recovery seed: {run_id}")
        for field in (
            "parameter_count", "target_input_tokens_with_eos", "target_flops_6nd",
            "scheduler_total_steps", "per_device_train_batch_size",
            "gradient_accumulation_steps", "effective_batch_records", "vocab_size",
            "context_length", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
        ):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] <= 0:
                raise ValueError(f"invalid positive integer {field}: {run_id}")
        if (row["per_device_train_batch_size"] * row["gradient_accumulation_steps"]
                != row["effective_batch_records"]):
            raise ValueError(f"recovery effective batch is inconsistent: {run_id}")
        n = row["parameter_count"]
        if abs(6 * n * row["target_input_tokens_with_eos"] - row["target_flops_6nd"]) > 3 * n:
            raise ValueError(f"recovery token/FLOPs budget is inconsistent: {run_id}")
        warmup_ratio = float(row["warmup_ratio"])
        if (not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1
                or row["warmup_steps"] != max(1, round(row["scheduler_total_steps"] * warmup_ratio))):
            raise ValueError(f"recovery warmup schedule is inconsistent: {run_id}")
    return rows


def run_recovery(rows, *, common, poll_seconds=1.0):
    """Reuse the real runner; serialize initialization, not entire training runs."""
    pending = {}
    for row in rows:
        pending.setdefault(row["recovery_device_index"], []).append(row)
    starting = {}
    futures = {}
    checked = set()
    inspecting_run_id = None

    def event(name, **details):
        append_jsonl(common["status_path"], {"event": name, **details}, common["status_lock"])

    with ThreadPoolExecutor(max_workers=len(rows)) as executor:
        try:
            while any(pending.values()) or len(checked) < len(futures):
                # Surface any failure before admitting another row on any GPU.
                for future, row in futures.items():
                    if future.done() and future not in checked:
                        inspecting_run_id = row["run_id"]
                        future.result()
                        checked.add(future)
                        event("recovery_complete", run_id=row["run_id"],
                              attempt=row["attempt_name"], device_index=row["recovery_device_index"])
                for device, remaining in pending.items():
                    if not remaining:
                        continue
                    current = starting.get(device)
                    if current is not None:
                        future, previous_row = current
                        inspecting_run_id = previous_row["run_id"]
                        if future.done():
                            future.result()
                            evidence = "completed_before_readiness_poll"
                        else:
                            evidence = formal_context_ready(
                                attempt_dir=Path(previous_row["attempt_dir"]).expanduser().resolve())
                            if evidence is None:
                                continue
                        event("recovery_initialized", run_id=previous_row["run_id"],
                              device_index=device, evidence=evidence)
                    row = remaining.pop(0)
                    inspecting_run_id = row["run_id"]
                    event("recovery_launch", run_id=row["run_id"],
                          attempt=row["attempt_name"], device_index=device)
                    future = executor.submit(run_one, row=row, device_index=device, **common)
                    futures[future] = row
                    starting[device] = (future, row)
                if any(pending.values()) or len(checked) < len(futures):
                    time.sleep(poll_seconds)
        except BaseException as error:
            event("recovery_failed", run_id=inspecting_run_id,
                  error_type=type(error).__name__, error=str(error),
                  not_started_run_ids=[row["run_id"] for group in pending.values() for row in group],
                  policy="no further launches; already-started trainers finish without signals")
            # The executor waits for existing trainers; retain the original
            # exception and nonzero exit instead of silently skipping failures.
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--device-indices", required=True)
    parser.add_argument("--status-jsonl", type=Path, required=True)
    parser.add_argument("--num-workers-per-trainer", type=int, default=2)
    parser.add_argument("--pipeline-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.num_workers_per_trainer < 0:
        parser.error("--num-workers-per-trainer must be nonnegative")
    plan_path = args.plan_json.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    python = args.python.expanduser().absolute()  # Preserve the venv interpreter path.
    if not python.is_file():
        raise FileNotFoundError(python)
    if not pipeline_root.is_dir():
        raise FileNotFoundError(pipeline_root)
    devices = parse_device_indices(args.device_indices)
    if max(devices) >= visible_cuda_device_count(python):
        raise ValueError("selected recovery device is outside scheduler-assigned CUDA visibility")
    plan = load_json(plan_path)
    rows = validate_recovery_rows(plan, set(devices))
    manifest_path = Path(plan["domain_split_manifest"]).expanduser().resolve()
    manifest = validate_manifest(manifest_path, rows)
    common = dict(
        python=python, pipeline_root=pipeline_root, manifest=manifest,
        manifest_path=manifest_path, num_workers=args.num_workers_per_trainer,
        plan_path=plan_path, status_path=args.status_jsonl.expanduser().resolve(),
        status_lock=threading.Lock(),
    )
    append_jsonl(common["status_path"], {
        "event": "recovery_start", "plan_json": str(plan_path),
        "devices": devices, "run_ids": [row["run_id"] for row in rows],
        "profile_policy": "explicit worker profile is completed separately before this entry point",
    }, common["status_lock"])
    run_recovery(rows, common=common)
    append_jsonl(common["status_path"], {"event": "recovery_all_complete"}, common["status_lock"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
