#!/usr/bin/env python3
"""Split the frozen Phase-B rows into disjoint local/remote host matrices."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .finalize_domainwise_phase_b_matrix import BATCH_PROFILES
    from .scaling_paths import DEFAULT_RUN_ROOT
except ImportError:  # direct script execution
    from finalize_domainwise_phase_b_matrix import BATCH_PROFILES
    from scaling_paths import DEFAULT_RUN_ROOT


HOST_LAYOUTS: dict[str, dict[int, int]] = {
    "local": {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2},
    "remote": {0: 1, 1: 1, 2: 1, 3: 1},
}


def profile_for_row(
    domain: str,
    parameter_count: int,
    concurrent_slots: int,
) -> tuple[str, int, int]:
    """Choose a measured profile without changing records per optimizer update."""
    if concurrent_slots > 1 or parameter_count >= 46_000_000:
        tier = "large_or_shared_48gb"
    elif parameter_count <= 6_100_000:
        tier = "single_24gb_le_6m"
    elif parameter_count < 32_000_000:
        tier = "single_24gb_gt_6m_to_31m"
    else:
        tier = "single_24gb_32m_to_41m"
    micro, grad_accum = BATCH_PROFILES[tier][domain]
    expected = int(BATCH_PROFILES["effective_batch_records"][domain])
    if int(micro) * int(grad_accum) != expected:
        raise AssertionError(f"batch profile changes effective batch for {domain}/{tier}")
    return tier, int(micro), int(grad_accum)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    experiment_name = "domainwise_scaling_fddfs_mf80000_v1"
    campaign_root = DEFAULT_RUN_ROOT / experiment_name
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-matrix",
        type=Path,
        default=campaign_root / "phase_b_matrix.json",
    )
    ap.add_argument(
        "--local-output",
        type=Path,
        default=campaign_root / "phase_b_matrix_local.json",
    )
    ap.add_argument(
        "--remote-output",
        type=Path,
        default=campaign_root / "phase_b_matrix_remote.json",
    )
    args = ap.parse_args()
    source_path = args.source_matrix.expanduser().resolve()
    output_paths = {
        "local": args.local_output.expanduser().resolve(),
        "remote": args.remote_output.expanduser().resolve(),
    }
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = source.get("rows", [])
    if source["metadata"].get("status") != "runnable_phase_b":
        raise ValueError("source matrix is not runnable_phase_b")
    if len(source_rows) != 60 or len({row["run_id"] for row in source_rows}) != 60:
        raise ValueError("expected exactly 60 unique frozen Phase-B rows")

    # LPT ties are deterministic.  Prefer single-slot devices for the largest
    # rows so the dual-slot 48 GB cards receive more splittable work and keep
    # both resident processes useful for longer.
    host_order = {host_id: index for index, host_id in enumerate(HOST_LAYOUTS)}
    devices = sorted(
        [
            (host_id, gpu, slots)
            for host_id, layout in HOST_LAYOUTS.items()
            for gpu, slots in layout.items()
        ],
        key=lambda device: (device[2], host_order[device[0]], device[1]),
    )
    device_order = {device[:2]: index for index, device in enumerate(devices)}
    physical_load = {(host, gpu): 0.0 for host, gpu, _ in devices}
    slot_load = {
        (host, gpu): [0.0 for _ in range(slots)]
        for host, gpu, slots in devices
    }
    assigned: list[dict[str, Any]] = []
    for source_row in sorted(
        source_rows,
        key=lambda row: (
            -float(row["target_flops_after_integer_token_rounding"]),
            str(row["run_id"]),
        ),
    ):
        host_id, gpu, concurrent_slots = min(
            devices,
            key=lambda device: (
                physical_load[device[:2]],
                device_order[device[:2]],
            ),
        )
        local_slot = min(
            range(concurrent_slots),
            key=lambda slot: (slot_load[(host_id, gpu)][slot], slot),
        )
        row = copy.deepcopy(source_row)
        row.update(
            {
                "assigned_host": host_id,
                "assigned_gpu": gpu,
                "gpu_local_slot": local_slot,
                "worker_slot": f"{host_id}_gpu{gpu}_slot{local_slot}",
            }
        )
        tier, micro, grad_accum = profile_for_row(
            str(row["domain"]),
            int(row["parameter_count"]),
            concurrent_slots,
        )
        effective_records = micro * grad_accum
        mean_tokens = float(row["train_input_tokens_per_epoch"]) / int(row["train_records"])
        expected_tokens_per_step = mean_tokens * effective_records
        scheduler_steps = int(
            math.ceil(int(row["target_input_tokens_with_eos"]) / expected_tokens_per_step)
        )
        row.update(
            {
                "batch_profile_tier": tier,
                "per_device_train_batch_size": micro,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_records": effective_records,
                "expected_input_tokens_per_optimizer_step": expected_tokens_per_step,
                "max_steps": scheduler_steps,
                "scheduler_total_steps": scheduler_steps,
                "warmup_steps": max(
                    1,
                    int(round(scheduler_steps * float(row["warmup_ratio"]))),
                ),
            }
        )
        flops = float(row["target_flops_after_integer_token_rounding"])
        physical_load[(host_id, gpu)] += flops
        slot_load[(host_id, gpu)][local_slot] += flops
        assigned.append(row)

    partition_id = "phase_b_fixed_local8_remote4_v1"
    created_at = datetime.now(timezone.utc).isoformat()
    global_ids = {row["run_id"] for row in assigned}
    if len(assigned) != 60 or len(global_ids) != 60:
        raise AssertionError("partition lost or duplicated a Phase-B row")

    outputs: dict[str, dict[str, Any]] = {}
    for host_id, layout in HOST_LAYOUTS.items():
        rows = [row for row in assigned if row["assigned_host"] == host_id]
        rows.sort(
            key=lambda row: (
                int(row["assigned_gpu"]),
                int(row["gpu_local_slot"]),
                int(row["budget_index"]),
                str(row["domain"]),
                int(row["model_slot"]),
            )
        )
        physical = {
            str(gpu): physical_load[(host_id, gpu)] for gpu in layout
        }
        workers = {
            f"{host_id}_gpu{gpu}_slot{slot}": slot_load[(host_id, gpu)][slot]
            for gpu, count in layout.items()
            for slot in range(count)
        }
        outputs[host_id] = {
            "metadata": {
                **source["metadata"],
                "matrix_version": "domainwise_scaling_phase_b_fixed_host_partition_v1",
                "created_at_utc": created_at,
                "partition_id": partition_id,
                "partition_source_matrix": str(source_path),
                "host_id": host_id,
                "partition_hosts": list(HOST_LAYOUTS),
                "global_rows": 60,
                "rows": len(rows),
                "gpu_layout": {str(gpu): count for gpu, count in layout.items()},
                "physical_gpu_target_flops": physical,
                "global_physical_gpu_target_flops": {
                    f"{host}:{gpu}": load
                    for (host, gpu), load in physical_load.items()
                },
                "worker_slot_target_flops": workers,
            },
            "rows": rows,
        }
        atomic_json(output_paths[host_id], outputs[host_id])

    local_ids = {row["run_id"] for row in outputs["local"]["rows"]}
    remote_ids = {row["run_id"] for row in outputs["remote"]["rows"]}
    if local_ids & remote_ids or local_ids | remote_ids != global_ids:
        raise AssertionError("host matrices are not an exact disjoint partition")
    print(
        json.dumps(
            {
                "partition_id": partition_id,
                "source_rows": len(source_rows),
                "local": {
                    "matrix": str(output_paths["local"]),
                    "rows": len(outputs["local"]["rows"]),
                    "slots": sum(HOST_LAYOUTS["local"].values()),
                },
                "remote": {
                    "matrix": str(output_paths["remote"]),
                    "rows": len(outputs["remote"]["rows"]),
                    "slots": sum(HOST_LAYOUTS["remote"].values()),
                },
                "physical_gpu_target_flops": {
                    f"{host}:{gpu}": load
                    for (host, gpu), load in physical_load.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
