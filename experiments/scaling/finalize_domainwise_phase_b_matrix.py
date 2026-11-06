#!/usr/bin/env python3
"""Freeze the reviewed Phase-B proposal into a runnable physical-host matrix."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_RUN_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_RUN_ROOT


BATCH_PROFILES = {
    "effective_batch_records": {
        "small_molecule": 12_288,
        "protein": 2_560,
        "dna": 2_560,
        "rna": 6_144,
    },
    "large_or_shared_48gb": {
        "small_molecule": (1536, 8),
        "protein": (320, 8),
        "dna": (320, 8),
        "rna": (768, 8),
    },
    "single_24gb_le_6m": {
        "small_molecule": (6144, 2),
        "protein": (1280, 2),
        "dna": (1280, 2),
        "rna": (3072, 2),
    },
    "single_24gb_gt_6m_to_31m": {
        "small_molecule": (3072, 4),
        "protein": (640, 4),
        "dna": (640, 4),
        "rna": (1536, 4),
    },
    "single_24gb_32m_to_41m": {
        "small_molecule": (2048, 6),
        "protein": (512, 5),
        "dna": (320, 8),
        "rna": (1024, 6),
    },
}


def batch_profile(domain: str, parameter_count: int, assigned_gpu: int) -> tuple[str, int, int]:
    if assigned_gpu >= 4 or parameter_count >= 46_000_000:
        tier = "large_or_shared_48gb"
    # The documented family labels are rounded (0p006b is 5,979,648 exact).
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


def parse_gpu_layout(raw: str) -> dict[int, int]:
    """Parse GPU:concurrent-slot pairs, e.g. 0:1,...,4:2."""
    layout: dict[int, int] = {}
    for item in raw.split(","):
        gpu_text, slots_text = item.strip().split(":", 1)
        gpu, slots = int(gpu_text), int(slots_text)
        if gpu < 0 or slots <= 0 or gpu in layout:
            raise ValueError(f"invalid --gpu-layout component: {item!r}")
        layout[gpu] = slots
    if not layout:
        raise ValueError("--gpu-layout is empty")
    return layout


def assign_physical_gpus(rows: list[dict[str, Any]], layout: dict[int, int]) -> None:
    """LPT-balance FLOPs per physical GPU, then balance its local slots."""
    gpu_load = {gpu: 0.0 for gpu in layout}
    slot_load = {
        gpu: [0.0 for _ in range(num_slots)] for gpu, num_slots in layout.items()
    }
    for row in sorted(rows, key=lambda item: float(item["target_flops_6nd"]), reverse=True):
        gpu = min(gpu_load, key=lambda key: (gpu_load[key], key))
        local_slot = min(
            range(layout[gpu]), key=lambda key: (slot_load[gpu][key], key)
        )
        flops = float(row["target_flops_6nd"])
        row["assigned_gpu"] = gpu
        row["gpu_local_slot"] = local_slot
        row["worker_slot"] = f"gpu{gpu}_slot{local_slot}"
        gpu_load[gpu] += flops
        slot_load[gpu][local_slot] += flops


def main() -> None:
    default_proposal = DEFAULT_DATA_ROOT / "reports" / "domainwise_scaling_matrix_proposal_v1.json"
    default_manifest = DEFAULT_DATA_ROOT / "processed" / "domain_split_manifests_v1" / "manifest.json"
    experiment_name = "domainwise_scaling_fddfs_mf80000_v1"
    default_output = DEFAULT_RUN_ROOT / experiment_name / "phase_b_matrix.json"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposal-json", type=Path, default=default_proposal)
    ap.add_argument("--domain-split-manifest", type=Path, default=default_manifest)
    ap.add_argument("--output-json", type=Path, default=default_output)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", default=experiment_name)
    ap.add_argument(
        "--gpu-layout",
        default="0:1,1:1,2:1,3:1,4:2,5:2,6:2,7:2",
        help="Physical GPU to concurrent worker-slot count.",
    )
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    args = ap.parse_args()
    proposal_path = args.proposal_json.expanduser().resolve()
    manifest_path = args.domain_split_manifest.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not args.experiment_name or "/" in args.experiment_name:
        raise ValueError("--experiment-name must be one safe directory name")

    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if proposal["metadata"].get("status") != "design_only_not_runnable":
        raise ValueError("unexpected proposal status")
    if manifest.get("schema_version") != "domain_split_raw_indices_v1" or manifest.get("status") != "complete":
        raise ValueError("domain split manifest is not complete domain_split_raw_indices_v1")

    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for source in proposal["rows"]:
        if source["execution_phase"] != "B_bracketing":
            continue
        domain = str(source["domain"])
        if domain not in BATCH_PROFILES["effective_batch_records"]:
            raise ValueError(f"no Phase-B batch profile for {domain}")
        view = manifest["views"][domain]["train"]
        if int(view["records"]) != int(source["train_records"]):
            raise ValueError(f"manifest/matrix train record mismatch for {domain}")
        if int(view["effective_input_tokens_with_eos"]) != int(
            source["train_input_tokens_per_epoch"]
        ):
            raise ValueError(f"manifest/matrix train token mismatch for {domain}")
        run_id = (
            f"{domain}_{source['budget_label']}_{source['model_id']}_s{int(source['seed'])}"
        )
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate run id: {run_id}")
        seen_run_ids.add(run_id)
        row = dict(source)
        row.update(
            {
                "matrix_status": "runnable_phase_b",
                "run_id": run_id,
                "run_name": run_id,
                "out_dir": str(run_root / args.experiment_name / "runs" / run_id),
                "gpu_count": 1,
                "warmup_ratio": args.warmup_ratio,
                "token_stop_authority": "target_input_tokens_with_eos",
                "domain_split_manifest": str(manifest_path),
                "train_split": "train",
                "validation_split": "validation",
                "bpe_unfold_policy": "none",
                "world_size": 1,
            }
        )
        # Old design-only placeholders must not survive under ambiguous names.
        row.pop("measured_input_tokens_per_step", None)
        rows.append(row)

    if len(rows) != 60:
        raise AssertionError(f"expected 60 Phase-B rows, got {len(rows)}")
    layout = parse_gpu_layout(args.gpu_layout)
    assign_physical_gpus(rows, layout)
    for row in rows:
        domain = str(row["domain"])
        tier, micro, grad_accum = batch_profile(
            domain,
            int(row["parameter_count"]),
            int(row["assigned_gpu"]),
        )
        effective_records = micro * grad_accum
        mean_tokens = float(row["train_input_tokens_per_epoch"]) / int(row["train_records"])
        expected_tokens_per_step = mean_tokens * effective_records
        target_tokens = int(row["target_input_tokens_with_eos"])
        max_steps = int(math.ceil(target_tokens / expected_tokens_per_step))
        row.update(
            {
                "batch_profile_tier": tier,
                "per_device_train_batch_size": micro,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_records": effective_records,
                "expected_input_tokens_per_optimizer_step": expected_tokens_per_step,
                "max_steps": max_steps,
                "scheduler_total_steps": max_steps,
                "warmup_steps": max(1, int(round(max_steps * args.warmup_ratio))),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["assigned_gpu"]),
            int(row["gpu_local_slot"]),
            int(row["budget_index"]),
            str(row["domain"]),
            int(row["model_slot"]),
        )
    )
    physical_loads = {
        str(gpu): sum(
            float(row["target_flops_6nd"])
            for row in rows
            if int(row["assigned_gpu"]) == gpu
        )
        for gpu in layout
    }
    slot_loads = {
        f"gpu{gpu}_slot{slot}": sum(
            float(row["target_flops_6nd"])
            for row in rows
            if int(row["assigned_gpu"]) == gpu and int(row["gpu_local_slot"]) == slot
        )
        for gpu, count in layout.items()
        for slot in range(count)
    }
    output = {
        "metadata": {
            "matrix_version": "domainwise_scaling_phase_b_runnable_v1",
            "status": "runnable_phase_b",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "proposal_json": str(proposal_path),
            "domain_split_manifest": str(manifest_path),
            "run_root": str(run_root),
            "experiment_name": args.experiment_name,
            "rows": len(rows),
            "gpu_layout": {str(key): value for key, value in layout.items()},
            "physical_gpu_target_flops": physical_loads,
            "worker_slot_target_flops": slot_loads,
            "batch_profiles": BATCH_PROFILES,
            "invariants": [
                "one process and one randomly initialized model per run",
                "target input tokens including EOS are the authoritative stop condition",
                "padding tokens are excluded and BPE unfolding is disabled",
                "validation uses the complete frozen validation view only after training",
                "test view is not read during Phase B",
            ],
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output_path)
    print(
        json.dumps(
            {
                "matrix": str(output_path),
                "rows": len(rows),
                "worker_slots": sum(layout.values()),
                "physical_gpu_target_flops": physical_loads,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
