#!/usr/bin/env python3
"""Build a fresh-attempt cluster plan after changing the H200 lane layout.

The input is the semantic 39-point cluster completion plan.  Existing
incomplete attempt directories are retained and the corresponding runs get a
new attempt number, always starting from random initialization at step zero.
Runs with a valid completion marker are omitted.  Optional per-run batch
overrides may change microbatch and gradient accumulation only when their
product remains the original effective record batch.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_domainwise_cluster_completion_plan import atomic_json, atomic_text, now
    from .run_domainwise_cluster_completion import load_json, partition_lpt
except ImportError:  # direct script execution
    from build_domainwise_cluster_completion_plan import atomic_json, atomic_text, now
    from run_domainwise_cluster_completion import load_json, partition_lpt


SCHEMA_VERSION = "domainwise_cluster_relaunch_plan_v2"
OVERRIDE_SCHEMA_VERSION = "domainwise_cluster_batch_overrides_v1"
PLACEMENT_SCHEMA_VERSION = "domainwise_cluster_lane_placement_v1"
ATTEMPT_PATTERN = re.compile(r"^attempt_([0-9]{3,})$")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value):
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def load_batch_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path.expanduser().resolve())
    if not isinstance(payload, dict) or payload.get("schema_version") != OVERRIDE_SCHEMA_VERSION:
        raise ValueError(
            f"batch override file must use {OVERRIDE_SCHEMA_VERSION}"
        )
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("batch override file must contain a runs object")
    for run_id, override in runs.items():
        if not isinstance(run_id, str) or not run_id or not isinstance(override, dict):
            raise ValueError("each batch override must be keyed by a non-empty run_id")
        allowed = {
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "reason",
        }
        unexpected = set(override) - allowed
        if unexpected:
            raise ValueError(f"unsupported batch override fields for {run_id}: {sorted(unexpected)}")
        _positive_int(
            override.get("per_device_train_batch_size"),
            f"{run_id}.per_device_train_batch_size",
        )
        _positive_int(
            override.get("gradient_accumulation_steps"),
            f"{run_id}.gradient_accumulation_steps",
        )
        if "reason" in override and not isinstance(override["reason"], str):
            raise ValueError(f"{run_id}.reason must be a string")
    return runs


def existing_attempt_numbers(run_dir: Path) -> list[int]:
    if not run_dir.exists():
        return []
    numbers: list[int] = []
    for child in run_dir.iterdir():
        match = ATTEMPT_PATTERN.fullmatch(child.name)
        if child.is_dir() and match is not None:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def completed_attempt(run_dir: Path, run_id: str) -> Path | None:
    marker_path = run_dir / "complete.json"
    if not marker_path.is_file():
        return None
    marker = load_json(marker_path)
    if not isinstance(marker, dict) or marker.get("status") != "complete":
        raise ValueError(f"malformed completion marker: {marker_path}")
    if marker.get("run_id") != run_id:
        raise ValueError(f"completion marker run_id mismatch: {marker_path}")
    attempt_dir = Path(str(marker.get("attempt_dir", ""))).expanduser().resolve()
    if attempt_dir.parent != run_dir.resolve() or not (attempt_dir / "summary.json").is_file():
        raise ValueError(f"completion marker points to an incomplete attempt: {marker_path}")
    return attempt_dir


def add_placement(
    rows: list[dict[str, Any]], *, global_gpu_count: int, lanes_per_gpu: int
) -> None:
    """Annotate rows with the same interleaved LPT placement used by the runner."""

    partitions = partition_lpt(rows, global_gpu_count * lanes_per_gpu)
    for global_gpu_index in range(global_gpu_count):
        for lane_within_gpu in range(lanes_per_gpu):
            global_lane_index = lane_within_gpu * global_gpu_count + global_gpu_index
            for lane_sequence_index, row in enumerate(partitions[global_lane_index]):
                row["global_gpu_index"] = global_gpu_index
                row["lane_within_gpu"] = lane_within_gpu
                row["global_lane_index"] = global_lane_index
                row["lane_sequence_index"] = lane_sequence_index


def build_relaunch_plan(
    source_plan: dict[str, Any],
    *,
    source_plan_path: Path,
    batch_overrides: dict[str, dict[str, Any]],
    batch_overrides_path: Path | None,
    global_gpu_count: int,
    lanes_per_gpu: int,
) -> dict[str, Any]:
    source_rows = source_plan.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 39:
        raise ValueError("source completion plan must contain the original 39 rows")
    if global_gpu_count <= 0 or lanes_per_gpu <= 0:
        raise ValueError("global_gpu_count and lanes_per_gpu must be positive")
    if global_gpu_count * lanes_per_gpu < len(source_rows):
        raise ValueError("the requested lane layout cannot place all 39 runs concurrently")

    source_ids = {str(row.get("run_id", "")) for row in source_rows}
    unknown_overrides = set(batch_overrides) - source_ids
    if unknown_overrides:
        raise ValueError(f"batch overrides contain unknown run IDs: {sorted(unknown_overrides)}")

    rows: list[dict[str, Any]] = []
    completed_run_ids: list[str] = []
    for source in source_rows:
        if not isinstance(source, dict):
            raise ValueError("source plan row must be an object")
        row = dict(source)
        run_id = str(row["run_id"])
        run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        complete_attempt = completed_attempt(run_dir, run_id)
        if complete_attempt is not None:
            completed_run_ids.append(run_id)
            continue

        baseline_attempt = int(source["attempt_number"])
        previous_attempts = existing_attempt_numbers(run_dir)
        next_attempt = max([baseline_attempt - 1, *previous_attempts]) + 1
        attempt_dir = run_dir / f"attempt_{next_attempt:03d}"

        baseline_micro = int(source.get(
            "baseline_per_device_train_batch_size",
            source["per_device_train_batch_size"],
        ))
        baseline_accum = int(source.get(
            "baseline_gradient_accumulation_steps",
            source["gradient_accumulation_steps"],
        ))
        effective_batch = int(source["effective_batch_records"])
        if baseline_micro * baseline_accum != effective_batch:
            raise ValueError(f"source effective batch is inconsistent for {run_id}")
        override = batch_overrides.get(run_id)
        if override is None:
            micro = int(source["per_device_train_batch_size"])
            accum = int(source["gradient_accumulation_steps"])
            reason = "unchanged_for_mixed_k5_smoke"
        else:
            micro = _positive_int(
                override["per_device_train_batch_size"],
                f"{run_id}.per_device_train_batch_size",
            )
            accum = _positive_int(
                override["gradient_accumulation_steps"],
                f"{run_id}.gradient_accumulation_steps",
            )
            reason = str(override.get("reason", "mixed_k5_memory_tuning"))
        if micro * accum != effective_batch:
            raise ValueError(
                f"batch override changes effective batch for {run_id}: "
                f"{micro} x {accum} != {effective_batch}"
            )

        row.update(
            {
                "attempt_number": next_attempt,
                "attempt_name": attempt_dir.name,
                "attempt_dir": str(attempt_dir),
                "restart_from_step": 0,
                "init_checkpoint": None,
                "random_initialization": True,
                "previous_attempt_numbers": previous_attempts,
                "relaunch_reason": (
                    "lane_layout_changed_after_incomplete_cluster_attempt"
                    if previous_attempts
                    else "not_started_in_previous_cluster_layout"
                ),
                "baseline_per_device_train_batch_size": baseline_micro,
                "baseline_gradient_accumulation_steps": baseline_accum,
                "per_device_train_batch_size": micro,
                "gradient_accumulation_steps": accum,
                "batch_override_applied": override is not None,
                "batch_adjustment_reason": reason,
            }
        )
        rows.append(row)

    add_placement(
        rows,
        global_gpu_count=global_gpu_count,
        lanes_per_gpu=lanes_per_gpu,
    )
    rows.sort(key=lambda row: row["run_id"])
    attempt_counts = Counter(str(row["attempt_name"]) for row in rows)
    per_gpu_counts = Counter(int(row["global_gpu_index"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "source_plan": str(source_plan_path.expanduser().resolve()),
        "source_schema_version": source_plan.get("schema_version"),
        "source_row_count": 39,
        "status": "ready_for_mixed_k5_smoke_then_training",
        "phase": source_plan.get("phase"),
        "phase_root": source_plan["phase_root"],
        "domain_split_manifest": source_plan["domain_split_manifest"],
        "training_control": source_plan.get("training_control", {}),
        "execution_policy": {
            **source_plan.get("execution_policy", {}),
            "relaunch_incomplete_attempts_from_step_zero": True,
            "batch_adjustment_must_preserve_effective_records": True,
        },
        "placement": {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "algorithm": "target_flops_lpt_with_lanes_interleaved_across_gpus",
            "global_gpu_count": global_gpu_count,
            "lanes_per_gpu": lanes_per_gpu,
            "total_lanes": global_gpu_count * lanes_per_gpu,
            "per_gpu_run_counts": {
                str(index): per_gpu_counts.get(index, 0)
                for index in range(global_gpu_count)
            },
        },
        "batch_overrides": (
            str(batch_overrides_path.expanduser().resolve())
            if batch_overrides_path is not None
            else None
        ),
        "completed_before_relaunch": sorted(completed_run_ids),
        "completed_before_relaunch_count": len(completed_run_ids),
        "rows": rows,
        "row_count": len(rows),
        "attempt_counts": dict(sorted(attempt_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--batch-overrides-json", type=Path)
    parser.add_argument("--global-gpu-count", type=int, default=8)
    parser.add_argument("--lanes-per-gpu", type=int, default=5)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--run-id-file", type=Path, required=True)
    args = parser.parse_args()

    source_plan_path = args.source_plan.expanduser().resolve()
    overrides = load_batch_overrides(args.batch_overrides_json)
    plan = build_relaunch_plan(
        load_json(source_plan_path),
        source_plan_path=source_plan_path,
        batch_overrides=overrides,
        batch_overrides_path=args.batch_overrides_json,
        global_gpu_count=args.global_gpu_count,
        lanes_per_gpu=args.lanes_per_gpu,
    )
    output = args.out_json.expanduser().resolve()
    run_ids = args.run_id_file.expanduser().resolve()
    atomic_json(output, plan)
    atomic_text(run_ids, "".join(f"{row['run_id']}\n" for row in plan["rows"]))
    print(
        json.dumps(
            {
                "status": "written",
                "plan": str(output),
                "run_id_file": str(run_ids),
                "rows": plan["row_count"],
                "completed_before_relaunch": plan["completed_before_relaunch_count"],
                "attempt_counts": plan["attempt_counts"],
                "per_gpu_run_counts": plan["placement"]["per_gpu_run_counts"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
