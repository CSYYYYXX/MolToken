#!/usr/bin/env python3
"""Run a v2 relaunch plan on one scheduler allocation.

The plan carries a fixed global 8-GPU/K5 placement.  This runner selects the
global GPU indices owned by its allocation and maps each run to the matching
allocation-local CUDA ordinal through ``LOCAL_RANK``.  It neither sets nor
rewrites ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

try:
    from .build_domainwise_cluster_relaunch_plan import SCHEMA_VERSION
    from .run_domainwise_cluster_completion import (
        append_jsonl,
        load_json,
        now,
        parse_device_indices,
        read_run_ids,
        run_lane,
        validate_manifest,
        visible_cuda_device_count,
    )
except ImportError:  # direct script execution
    from build_domainwise_cluster_relaunch_plan import SCHEMA_VERSION
    from run_domainwise_cluster_completion import (
        append_jsonl,
        load_json,
        now,
        parse_device_indices,
        read_run_ids,
        run_lane,
        validate_manifest,
        visible_cuda_device_count,
    )


def validate_relaunch_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported relaunch plan schema: {plan.get('schema_version')!r}")
    rows = plan.get("rows")
    if not isinstance(rows, list) or int(plan.get("row_count", -1)) != len(rows):
        raise ValueError("plan rows are missing or row_count is inconsistent")
    completed = plan.get("completed_before_relaunch")
    if not isinstance(completed, list) or len(rows) + len(completed) != 39:
        raise ValueError("relaunch and completed run counts must cover the original 39 runs")
    if len(set(str(run_id) for run_id in completed)) != len(completed):
        raise ValueError("completed_before_relaunch contains duplicates")

    placement = plan.get("placement")
    if not isinstance(placement, dict):
        raise ValueError("relaunch plan has no placement object")
    global_gpu_count = int(placement.get("global_gpu_count", 0))
    lanes_per_gpu = int(placement.get("lanes_per_gpu", 0))
    if global_gpu_count <= 0 or lanes_per_gpu <= 0:
        raise ValueError("plan placement dimensions must be positive")

    phase_root = Path(str(plan["phase_root"])).expanduser().resolve()
    seen_ids: set[str] = set()
    seen_positions: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("plan row must be an object")
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in seen_ids or run_id in completed:
            raise ValueError(f"empty, duplicate, or already-complete run_id: {run_id!r}")
        seen_ids.add(run_id)

        source_status = row.get("source_status")
        if source_status == "not_started":
            baseline_attempt = 1
        elif source_status == "interrupted_restart_required":
            baseline_attempt = 2
        else:
            raise ValueError(f"unsupported source status for {run_id}")
        previous = row.get("previous_attempt_numbers")
        if not isinstance(previous, list) or any(
            isinstance(number, bool) or not isinstance(number, int) or number <= 0
            for number in previous
        ):
            raise ValueError(f"invalid previous attempt lineage for {run_id}")
        attempt = int(row["attempt_number"])
        expected_attempt = max([baseline_attempt - 1, *previous]) + 1
        if (
            attempt != expected_attempt
            or row.get("attempt_name") != f"attempt_{attempt:03d}"
            or int(row["restart_from_step"]) != 0
            or row.get("init_checkpoint") is not None
            or row.get("random_initialization") is not True
            or int(row["world_size"]) != 1
        ):
            raise ValueError(f"fresh-attempt contract changed for {run_id}")

        effective = int(row["effective_batch_records"])
        baseline_micro = int(row["baseline_per_device_train_batch_size"])
        baseline_accum = int(row["baseline_gradient_accumulation_steps"])
        micro = int(row["per_device_train_batch_size"])
        accum = int(row["gradient_accumulation_steps"])
        if min(effective, baseline_micro, baseline_accum, micro, accum) <= 0:
            raise ValueError(f"non-positive batch control for {run_id}")
        if baseline_micro * baseline_accum != effective or micro * accum != effective:
            raise ValueError(f"effective record batch changed for {run_id}")

        steps = int(row["scheduler_total_steps"])
        if int(row["warmup_steps"]) != max(1, round(steps * float(row["warmup_ratio"]))):
            raise ValueError(f"warmup schedule changed for {run_id}")
        parameters = int(row["parameter_count"])
        target_tokens = int(row["target_input_tokens_with_eos"])
        target_flops = int(row["target_flops_6nd"])
        if min(parameters, target_tokens, target_flops) <= 0:
            raise ValueError(f"non-positive scientific budget for {run_id}")
        if abs(6 * parameters * target_tokens - target_flops) > 3 * parameters:
            raise ValueError(f"token and 6ND targets disagree for {run_id}")

        gpu = int(row["global_gpu_index"])
        lane = int(row["lane_within_gpu"])
        global_lane = int(row["global_lane_index"])
        sequence = int(row["lane_sequence_index"])
        if not 0 <= gpu < global_gpu_count or not 0 <= lane < lanes_per_gpu:
            raise ValueError(f"placement is outside the declared layout for {run_id}")
        if global_lane != lane * global_gpu_count + gpu or sequence < 0:
            raise ValueError(f"interleaved lane identity is inconsistent for {run_id}")
        position = (global_lane, sequence)
        if position in seen_positions:
            raise ValueError(f"duplicate lane position in relaunch plan: {position}")
        seen_positions.add(position)

        run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        attempt_dir = Path(str(row["attempt_dir"])).expanduser().resolve()
        if run_dir != phase_root / "runs" / run_id:
            raise ValueError(f"run output routing changed for {run_id}")
        if attempt_dir != run_dir / f"attempt_{attempt:03d}":
            raise ValueError(f"attempt output routing changed for {run_id}")
    return rows


def selected_rows(
    plan: dict[str, Any], run_ids: Iterable[str]
) -> list[dict[str, Any]]:
    rows = validate_relaunch_plan(plan)
    by_id = {str(row["run_id"]): row for row in rows}
    requested = list(run_ids)
    unknown = [run_id for run_id in requested if run_id not in by_id]
    if unknown:
        raise ValueError(f"whitelist contains run IDs outside the relaunch plan: {unknown}")
    return [by_id[run_id] for run_id in requested]


def per_gpu_plan(
    rows: Iterable[dict[str, Any]], *, global_gpu_count: int
) -> dict[int, list[dict[str, Any]]]:
    """Return rows grouped by global GPU in deterministic lane order."""

    grouped = {index: [] for index in range(global_gpu_count)}
    for row in rows:
        gpu = int(row["global_gpu_index"])
        if gpu not in grouped:
            raise ValueError(f"row has global GPU outside 0..{global_gpu_count - 1}")
        grouped[gpu].append(row)
    for gpu_rows in grouped.values():
        gpu_rows.sort(
            key=lambda row: (
                int(row["lane_within_gpu"]),
                int(row["lane_sequence_index"]),
            )
        )
    return grouped


def main() -> int:
    script = Path(__file__).resolve()
    pipeline_default = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--run-id-file", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=pipeline_default)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            os.environ.get(
                "HOLOMOL_SCALING_PYTHON",
                "python",
            )
        ),
    )
    parser.add_argument("--device-indices", required=True)
    parser.add_argument("--global-gpu-offset", type=int, required=True)
    parser.add_argument("--global-gpu-count", type=int, required=True)
    parser.add_argument("--lanes-per-gpu", type=int, required=True)
    parser.add_argument("--num-workers-per-trainer", type=int, default=2)
    parser.add_argument(
        "--status-jsonl",
        type=Path,
        default=Path("./logs/domainwise_cluster_status_v2.jsonl"),
    )
    args = parser.parse_args()
    if args.global_gpu_count <= 0 or args.global_gpu_offset < 0:
        raise ValueError("global GPU count/offset must be positive/non-negative")
    if args.lanes_per_gpu <= 0:
        raise ValueError("--lanes-per-gpu must be positive")
    if args.num_workers_per_trainer < 0:
        raise ValueError("--num-workers-per-trainer must be non-negative")

    plan_path = args.plan_json.expanduser().resolve()
    whitelist_path = args.run_id_file.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    python = args.python.expanduser().absolute()
    status_path = args.status_jsonl.expanduser().resolve()
    if not pipeline_root.is_dir():
        raise FileNotFoundError(pipeline_root)
    if not python.is_file():
        raise FileNotFoundError(python)
    device_indices = parse_device_indices(args.device_indices)
    if args.global_gpu_offset + len(device_indices) > args.global_gpu_count:
        raise ValueError("allocation GPU slice exceeds --global-gpu-count")
    visible_devices = visible_cuda_device_count(python)
    if visible_devices <= max(device_indices):
        raise RuntimeError(
            f"allocation exposes {visible_devices} CUDA devices, cannot use {device_indices}"
        )

    plan = load_json(plan_path)
    placement = plan.get("placement", {})
    if (
        int(placement.get("global_gpu_count", -1)) != args.global_gpu_count
        or int(placement.get("lanes_per_gpu", -1)) != args.lanes_per_gpu
    ):
        raise ValueError("runner GPU/lane layout differs from the relaunch plan")
    rows = selected_rows(plan, read_run_ids(whitelist_path))
    manifest_path = Path(str(plan["domain_split_manifest"])).expanduser().resolve()
    manifest = validate_manifest(manifest_path, rows)
    grouped = per_gpu_plan(rows, global_gpu_count=args.global_gpu_count)

    status_lock = threading.Lock()
    common = {
        "python": python,
        "pipeline_root": pipeline_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "num_workers": args.num_workers_per_trainer,
        "plan_path": plan_path,
        "status_path": status_path,
        "status_lock": status_lock,
    }
    lane_jobs: list[tuple[int, int, list[dict[str, Any]]]] = []
    for local_position, device_index in enumerate(device_indices):
        global_gpu_index = args.global_gpu_offset + local_position
        by_lane = {lane: [] for lane in range(args.lanes_per_gpu)}
        for row in grouped[global_gpu_index]:
            by_lane[int(row["lane_within_gpu"])].append(row)
        for lane, lane_rows in by_lane.items():
            lane_rows.sort(key=lambda row: int(row["lane_sequence_index"]))
            global_lane = lane * args.global_gpu_count + global_gpu_index
            lane_jobs.append((device_index, global_lane, lane_rows))

    append_jsonl(
        status_path,
        {
            "event": "allocation_start_v2",
            "device_indices": device_indices,
            "global_gpu_offset": args.global_gpu_offset,
            "global_gpu_count": args.global_gpu_count,
            "lanes_per_gpu": args.lanes_per_gpu,
            "owned_rows": sum(len(lane_rows) for _, _, lane_rows in lane_jobs),
            "started_at": now(),
        },
        status_lock,
    )
    stop = threading.Event()
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=len(lane_jobs)) as executor:
        futures = {
            executor.submit(
                run_lane,
                rows=lane_rows,
                device_index=device_index,
                stop=stop,
                common=common,
            ): global_lane
            for device_index, global_lane, lane_rows in lane_jobs
        }
        for future in as_completed(futures):
            error = future.exception()
            if error is not None and first_error is None:
                first_error = error
                stop.set()
                append_jsonl(
                    status_path,
                    {
                        "event": "allocation_failure_v2",
                        "global_lane": futures[future],
                        "error": f"{type(error).__name__}: {error}",
                    },
                    status_lock,
                )
    if first_error is not None:
        raise RuntimeError("one or more v2 cluster lanes failed") from first_error
    append_jsonl(
        status_path,
        {
            "event": "allocation_complete_v2",
            "global_gpu_offset": args.global_gpu_offset,
            "device_indices": device_indices,
        },
        status_lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
