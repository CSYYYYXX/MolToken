#!/usr/bin/env python3
"""Rebuild the 39-point domain-wise completion plan from the handoff inventory.

The September 6 inventory is the portable semantic record for the remaining
Phase-C points.  This builder deliberately does not import the physical-host
matrix, placement, retry ledger, or batch-profile evidence.  It keeps the
scientific controls recorded in the inventory and relocates outputs and the
split manifest to explicit cluster paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

try:
    from .build_domainwise_scaling_matrix import enrich_model_specs
except ImportError:  # direct script execution
    from build_domainwise_scaling_matrix import enrich_model_specs


SCHEMA_VERSION = "domainwise_cluster_completion_plan_v1"
VOCAB_SIZE = 17_207
CONTEXT_LENGTH = 512
WARMUP_RATIO = 0.03
SELECTED_STATUSES = {
    "not_started": 1,
    "interrupted_restart_required": 2,
}
EXPECTED_CAMPAIGN_STATUS_COUNTS = {
    "complete": 106,
    "not_started": 31,
    "interrupted_restart_required": 8,
}
EXPECTED_SELECTED_DOMAIN_COUNTS = {
    "small_molecule": 13,
    "protein": 12,
    "dna": 3,
    "rna": 11,
}
GPFS_PROJECT_ROOT = Path(
    "./storage"
)
DEFAULT_PHASE_ROOT = (
    GPFS_PROJECT_ROOT
    / "molscaling"
    / "runs"
    / "domainwise_scaling_fddfs_mf80000_v1_followup"
    / "phase_c"
)
DEFAULT_MANIFEST = (
    GPFS_PROJECT_ROOT
    / "molscaling"
    / "data"
    / "processed"
    / "domain_split_manifests_v1"
    / "manifest.json"
)
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def absolute_path(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    return expanded.resolve()


def _integer(row: dict[str, str], key: str) -> int:
    text = row.get(key, "")
    if not text or not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}")
    value = int(text)
    if value <= 0:
        raise ValueError(f"{row.get('run_id', '<unknown>')} has non-positive {key}")
    return value


def _nonnegative_integer(row: dict[str, str], key: str) -> int:
    text = row.get(key, "")
    if not text or not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}")
    return int(text)


def _decimal(row: dict[str, str], key: str) -> Decimal:
    text = row.get(key, "")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(
            f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}"
        ) from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}")
    return value


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"inventory is empty: {path}")
    return rows


def _status_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _model_specs() -> dict[str, dict[str, Any]]:
    return {
        str(spec["model_id"]): spec
        for spec in enrich_model_specs(VOCAB_SIZE)
    }


def build_plan(
    inventory_rows: list[dict[str, str]],
    *,
    inventory_path: Path,
    phase_root: Path,
    domain_split_manifest: Path,
) -> dict[str, Any]:
    """Build one portable plan while preserving the inventory's run semantics."""

    if len(inventory_rows) != 145:
        raise ValueError(f"expected the 145-point inventory, found {len(inventory_rows)} rows")
    counts = _status_counts(inventory_rows)
    if counts != EXPECTED_CAMPAIGN_STATUS_COUNTS:
        raise ValueError(
            "inventory status counts differ from the September 6 handoff: "
            f"{counts}"
        )

    phase_root = absolute_path(phase_root, "--phase-root")
    manifest_path = absolute_path(
        domain_split_manifest, "--domain-split-manifest"
    )
    specs = _model_specs()
    selected = [row for row in inventory_rows if row.get("status") in SELECTED_STATUSES]
    selected_domain_counts: dict[str, int] = {}
    plan_rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for source in selected:
        run_id = source.get("run_id", "")
        if SAFE_RUN_ID.fullmatch(run_id) is None or run_id in seen_run_ids:
            raise ValueError(f"unsafe or duplicate selected run_id: {run_id!r}")
        seen_run_ids.add(run_id)
        if source.get("phase") != "C":
            raise ValueError(f"remaining point is not Phase C: {run_id}")
        domain = source.get("domain", "")
        budget = source.get("budget", "")
        model_id = source.get("model_id", "")
        seed = _integer(source, "seed")
        if run_id != f"{domain}_{budget}_{model_id}_s{seed}":
            raise ValueError(f"run_id does not encode its controls: {run_id}")
        if model_id not in specs:
            raise ValueError(f"unknown model_id in inventory: {model_id}")
        spec = specs[model_id]
        parameter_count = _integer(source, "parameter_count")
        non_embedding = _integer(source, "non_embedding_parameter_count")
        if parameter_count != int(spec["parameter_count"]):
            raise ValueError(f"parameter_count differs from model family for {run_id}")
        if non_embedding != int(spec["non_embedding_parameter_count"]):
            raise ValueError(
                f"non_embedding_parameter_count differs from model family for {run_id}"
            )

        target_tokens = _integer(source, "target_input_tokens_with_eos")
        target_flops_decimal = _decimal(source, "target_flops_6nd")
        if target_flops_decimal != target_flops_decimal.to_integral_value():
            raise ValueError(f"target_flops_6nd is not integral for {run_id}")
        target_flops = int(target_flops_decimal)
        rounded_flops = 6 * parameter_count * target_tokens
        if abs(rounded_flops - target_flops) > 3 * parameter_count:
            raise ValueError(f"token budget is not the nearest 6ND budget for {run_id}")

        microbatch = _integer(source, "matrix_microbatch")
        grad_accum = _integer(source, "matrix_grad_accum")
        effective_batch = _integer(source, "effective_batch_records")
        if microbatch * grad_accum != effective_batch:
            raise ValueError(f"microbatch x grad_accum changes effective batch for {run_id}")
        scheduler_steps = _integer(source, "scheduler_total_steps")
        warmup_steps = _integer(source, "warmup_steps")
        expected_warmup = max(1, int(round(scheduler_steps * WARMUP_RATIO)))
        if warmup_steps != expected_warmup:
            raise ValueError(f"warmup steps differ from the frozen 3% schedule for {run_id}")

        attempt_number = SELECTED_STATUSES[source["status"]]
        run_dir = phase_root / "runs" / run_id
        attempt_dir = run_dir / f"attempt_{attempt_number:03d}"
        target_epoch_fraction = float(_decimal(source, "target_epoch_fraction"))
        if not math.isfinite(target_epoch_fraction):
            raise ValueError(f"non-finite target_epoch_fraction for {run_id}")

        plan_rows.append(
            {
                "phase": "C",
                "domain": domain,
                "budget": budget,
                "run_id": run_id,
                "model_id": model_id,
                "parameter_count": parameter_count,
                "non_embedding_parameter_count": non_embedding,
                "target_input_tokens_with_eos": target_tokens,
                "target_flops_6nd": target_flops,
                "target_epoch_fraction": target_epoch_fraction,
                "per_device_train_batch_size": microbatch,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_records": effective_batch,
                "scheduler_total_steps": scheduler_steps,
                "warmup_ratio": WARMUP_RATIO,
                "warmup_steps": warmup_steps,
                "seed": seed,
                "vocab_size": VOCAB_SIZE,
                "context_length": CONTEXT_LENGTH,
                "hidden_size": int(spec["hidden_size"]),
                "intermediate_size": int(spec["intermediate_size"]),
                "num_hidden_layers": int(spec["num_hidden_layers"]),
                "num_attention_heads": int(spec["num_attention_heads"]),
                "num_key_value_heads": int(spec["num_key_value_heads"]),
                "head_dim": int(spec["head_dim"]),
                "world_size": 1,
                "random_initialization": True,
                "init_checkpoint": None,
                "restart_from_step": 0,
                "source_status": source["status"],
                "attempt_number": attempt_number,
                "attempt_name": attempt_dir.name,
                "run_dir": str(run_dir),
                "attempt_dir": str(attempt_dir),
                "source_batch_profile": source.get("batch_profile", ""),
                "source_placement": {
                    "logical_host": source.get("logical_host", ""),
                    "logical_gpu": _nonnegative_integer(source, "logical_gpu"),
                    "worker_slot": source.get("worker_slot", ""),
                },
            }
        )
        selected_domain_counts[domain] = selected_domain_counts.get(domain, 0) + 1

    if len(plan_rows) != 39 or selected_domain_counts != EXPECTED_SELECTED_DOMAIN_COUNTS:
        raise ValueError(
            "remaining-point domain counts differ from the September 6 handoff: "
            f"{selected_domain_counts}"
        )
    plan_rows.sort(key=lambda row: row["run_id"])
    attempt_counts = {
        "attempt_001": sum(row["attempt_number"] == 1 for row in plan_rows),
        "attempt_002": sum(row["attempt_number"] == 2 for row in plan_rows),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "source_inventory": str(inventory_path.expanduser().resolve()),
        "status": "ready_for_cluster_smoke_then_training",
        "phase": "C",
        "phase_root": str(phase_root),
        "domain_split_manifest": str(manifest_path),
        "rows": plan_rows,
        "row_count": len(plan_rows),
        "source_status_counts": counts,
        "selected_domain_counts": selected_domain_counts,
        "attempt_counts": attempt_counts,
        "execution_policy": {
            "independent_single_process_runs": True,
            "world_size_per_run": 1,
            "restart_from_step_zero": True,
            "not_started_attempt": "attempt_001",
            "interrupted_restart_attempt": "attempt_002",
            "reuse_interrupted_checkpoint": False,
            "device_selection": "allocation_local_cuda_index_without_setting_CUDA_VISIBLE_DEVICES",
            "multiple_lanes_per_h200_require_worker_smoke": True,
        },
        "training_control": {
            "learning_rate": 2e-4,
            "minimum_learning_rate_ratio": 0.1,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "maximum_gradient_norm": 1.0,
            "optimizer": "adamw-fused",
            "precision": "bf16",
            "model_implementation": "packed-xformers",
            "attention_backend": "xformers_memory_efficient_block_diagonal_causal",
            "gradient_checkpointing": False,
            "torch_compile": False,
            "deepspeed": False,
            "save_every_steps": 0,
            "full_validation_at_end": True,
        },
    }


def main() -> int:
    script = Path(__file__).resolve()
    default_inventory = script.parents[4] / "doc" / "results" / "domainwise145_run_inventory_20260906.csv"
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--inventory-csv", type=Path, default=default_inventory)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--domain-split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--run-id-file", type=Path, required=True)
    args = parser.parse_args()

    inventory_path = args.inventory_csv.expanduser().resolve()
    plan = build_plan(
        read_inventory(inventory_path),
        inventory_path=inventory_path,
        phase_root=args.phase_root,
        domain_split_manifest=args.domain_split_manifest,
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
                "attempt_counts": plan["attempt_counts"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
