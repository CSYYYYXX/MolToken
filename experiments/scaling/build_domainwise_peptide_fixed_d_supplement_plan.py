#!/usr/bin/env python3
"""Build an explicit round of peptide fixed-input-token supplements.

These measurements extend equal-data peptide slices; they are not IsoFLOP
curves.  The builder reads only the frozen 25-row validation table, the small
domain split manifest, and optional existing plans.  It writes one isolated
supplement plan plus its run-id list and never edits a queue/watcher list or
launches training. The default remains the original three round-004 points;
later rounds require an explicit candidate JSON list and use a separate root.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    from .build_domainwise_bracketing_supplement_plan import (
        APPROVAL_SOURCE,
        MAXIMUM_EPOCH_FRACTION,
        PROFILE_OPTIMIZER_STEPS,
        PURPOSE,
        SAFE_RUN_ID,
        SCHEMA_VERSION,
        _integral_decimal,
        _manifest_domain_metadata,
        _model_specs,
        _positive_float,
        _positive_int,
        _scheduler_steps,
        load_json,
        read_csv,
    )
    from .build_domainwise_cluster_completion_plan import (
        CONTEXT_LENGTH,
        DEFAULT_MANIFEST,
        GPFS_PROJECT_ROOT,
        VOCAB_SIZE,
        WARMUP_RATIO,
        absolute_path,
        atomic_json,
        atomic_text,
        now,
    )
except ImportError:  # direct script execution
    from build_domainwise_bracketing_supplement_plan import (
        APPROVAL_SOURCE,
        MAXIMUM_EPOCH_FRACTION,
        PROFILE_OPTIMIZER_STEPS,
        PURPOSE,
        SAFE_RUN_ID,
        SCHEMA_VERSION,
        _integral_decimal,
        _manifest_domain_metadata,
        _model_specs,
        _positive_float,
        _positive_int,
        _scheduler_steps,
        load_json,
        read_csv,
    )
    from build_domainwise_cluster_completion_plan import (
        CONTEXT_LENGTH,
        DEFAULT_MANIFEST,
        GPFS_PROJECT_ROOT,
        VOCAB_SIZE,
        WARMUP_RATIO,
        absolute_path,
        atomic_json,
        atomic_text,
        now,
    )


SCIENTIFIC_PROTOCOL = "fixed_input_tokens_not_isoflop"
SUPPLEMENT_ROUND = 4
PEPTIDE_TRAIN_INPUT_TOKENS = 17_052_305
PEPTIDE_EFFECTIVE_BATCH_RECORDS = 6_144
EXPECTED_SOURCE_MODELS = (
    "0p006b",
    "0p009b",
    "0p014b",
    "0p022b",
    "0p041b",
)
SOURCE_BUDGET_TO_EPOCH = {
    "E0p25": 0.25,
    "E0p5": 0.5,
    "E1p0": 1.0,
    "E2p0": 2.0,
    "E4p0": 4.0,
}
PEPTIDE_BATCHES = {
    "0p031b": (1_536, 4, "peptide_7m_to_31m_1536x4"),
    "0p047b": (768, 8, "peptide_46m_plus_768x8"),
    "0p053b": (768, 8, "peptide_46m_plus_768x8"),
    "0p059b": (768, 8, "peptide_46m_plus_768x8"),
    "0p065b": (768, 8, "peptide_46m_plus_768x8"),
}
SUPPLEMENTS: tuple[dict[str, Any], ...] = (
    {
        "budget": "E2p0",
        "model_id": "0p031b",
        "target_epoch_fraction": 2.0,
        "reason": "measure the 30.7M model between the observed 22.1M and 41.0M equal-data points",
    },
    {
        "budget": "E4p0",
        "model_id": "0p031b",
        "target_epoch_fraction": 4.0,
        "reason": "add measured lower-N support around the observed 41.0M four-epoch minimum",
    },
    {
        "budget": "E4p0",
        "model_id": "0p047b",
        "target_epoch_fraction": 4.0,
        "reason": "add measured higher-N support around the observed 41.0M four-epoch minimum",
    },
)

DEFAULT_PHASE_ROOT = (
    GPFS_PROJECT_ROOT
    / "molscaling"
    / "runs"
    / "domainwise_scaling_fddfs_mf80000_v1_followup"
    / "phase_supplement"
    / "round_004"
)


def default_completed_csv() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "doc"
        / "results"
        / "domainwise145_tables_20260906"
        / "completed_runs.csv"
    )


def _is_round004_path(path: Path) -> bool:
    return any(
        ancestor.name == "round_004" or ancestor == DEFAULT_PHASE_ROOT.resolve()
        for ancestor in (path, *path.parents)
    )


def validate_source_d25(
    rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Validate and retain the frozen 5x5 peptide validation evidence."""

    peptide = [dict(row) for row in rows if row["domain"] == "peptide"]
    if len(peptide) != 25:
        raise ValueError(f"expected the frozen 25 peptide rows, found {len(peptide)}")

    specs = _model_specs()
    seen_ids: set[str] = set()
    seen_cells: set[tuple[str, str, int]] = set()
    validation_views: set[tuple[int, int, int]] = set()
    for row in peptide:
        run_id = row.get("run_id", "")
        model_id = row.get("model_id", "")
        budget = row.get("budget", "")
        if not run_id or run_id in seen_ids:
            raise ValueError(f"empty or duplicate D25 run_id: {run_id!r}")
        seen_ids.add(run_id)
        if (
            row.get("phase") != "D"
            or row.get("status") != "complete"
            or row.get("metric_split") != "validation"
            or row.get("validation_complete_view") != "true"
        ):
            raise ValueError(f"D25 row lacks complete validation evidence: {run_id}")
        if budget not in SOURCE_BUDGET_TO_EPOCH or model_id not in EXPECTED_SOURCE_MODELS:
            raise ValueError(f"unexpected D25 scientific cell: {run_id}")
        seed = _positive_int(row, "seed")
        if seed != 42:
            raise ValueError(f"D25 seed is not 42: {run_id}")
        cell = (budget, model_id, seed)
        if cell in seen_cells:
            raise ValueError(f"duplicate D25 scientific cell: {cell}")
        seen_cells.add(cell)

        epoch = SOURCE_BUDGET_TO_EPOCH[budget]
        if not math.isclose(
            _positive_float(row, "target_epoch_fraction"),
            epoch,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"D25 budget/epoch mismatch: {run_id}")
        train_tokens = _positive_int(row, "train_input_tokens_per_epoch")
        if train_tokens != PEPTIDE_TRAIN_INPUT_TOKENS:
            raise ValueError(f"D25 train token epoch changed: {run_id}")
        target_tokens = _positive_int(row, "target_input_tokens_with_eos")
        if target_tokens != round(epoch * train_tokens):
            raise ValueError(f"D25 fixed-input-token target mismatch: {run_id}")
        parameters = _positive_int(row, "parameter_count")
        if model_id not in specs or parameters != int(specs[model_id]["parameter_count"]):
            raise ValueError(f"D25 model registry mismatch: {run_id}")
        if _integral_decimal(row, "target_flops_6nd") != 6 * parameters * target_tokens:
            raise ValueError(f"D25 6ND mismatch: {run_id}")
        actual_tokens = _positive_int(row, "actual_input_tokens")
        if not target_tokens <= actual_tokens < target_tokens + CONTEXT_LENGTH:
            raise ValueError(f"D25 token-stop overshoot mismatch: {run_id}")
        if not math.isclose(
            _positive_float(row, "actual_epoch_fraction"),
            actual_tokens / train_tokens,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"D25 actual epoch fraction mismatch: {run_id}")
        _positive_float(row, "validation_loss")
        _positive_float(row, "validation_ppl")
        view = (
            _positive_int(row, "validation_samples"),
            _positive_int(row, "validation_loss_tokens"),
            _positive_int(row, "validation_input_tokens"),
        )
        validation_views.add(view)

    expected_cells = {
        (budget, model_id, 42)
        for budget in SOURCE_BUDGET_TO_EPOCH
        for model_id in EXPECTED_SOURCE_MODELS
    }
    if seen_cells != expected_cells:
        raise ValueError("D25 is not the complete five-model by five-epoch grid")
    if len(validation_views) != 1:
        raise ValueError("D25 rows disagree on the complete validation view")
    samples, loss_tokens, input_tokens = next(iter(validation_views))
    return {
        "status": "validated_complete_d25",
        "row_count": 25,
        "metric_split": "validation",
        "test_accessed": False,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "model_ids": list(EXPECTED_SOURCE_MODELS),
        "token_epochs": list(SOURCE_BUDGET_TO_EPOCH.values()),
        "train_input_tokens_per_epoch": PEPTIDE_TRAIN_INPUT_TOKENS,
        "validation_view": {
            "samples": samples,
            "loss_tokens": loss_tokens,
            "input_tokens": input_tokens,
        },
        "csv_columns": list(rows[0]),
        # Preserve the source CSV rows and their order verbatim as provenance.
        "rows": peptide,
    }


def validate_manifest_against_source(
    manifest: dict[str, Any], source: dict[str, Any]
) -> tuple[int, int]:
    train_records, train_tokens = _manifest_domain_metadata(manifest, "peptide")
    if train_tokens != PEPTIDE_TRAIN_INPUT_TOKENS:
        raise ValueError(
            f"peptide manifest U changed: {train_tokens} != {PEPTIDE_TRAIN_INPUT_TOKENS}"
        )
    validation = manifest.get("views", {}).get("peptide", {}).get("validation")
    if not isinstance(validation, dict):
        raise ValueError("split manifest has no peptide/validation view")
    expected = source["validation_view"]
    validation_records = int(validation["records"])
    validation_input_tokens = int(validation["effective_input_tokens_with_eos"])
    observed = {
        "samples": validation_records,
        "loss_tokens": validation_input_tokens - validation_records,
        "input_tokens": validation_input_tokens,
    }
    if observed != expected:
        raise ValueError(
            f"manifest and D25 validation view disagree: {observed} != {expected}"
        )
    return train_records, train_tokens


def existing_plan_identities(
    plans: Sequence[tuple[Path, dict[str, Any]]]
) -> dict[str, set[Any]]:
    identities: dict[str, set[Any]] = {
        "run_ids": set(),
        "coordinates": set(),
        "paths": set(),
    }
    for plan_path, plan in plans:
        rows = plan.get("rows") if isinstance(plan, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"existing plan has no rows list: {plan_path}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"existing plan row is not an object: {plan_path}")
            identities["run_ids"].add(str(row["run_id"]))
            identities["coordinates"].add(
                (
                    str(row["domain"]),
                    str(row["budget"]),
                    str(row["model_id"]),
                    int(row["seed"]),
                )
            )
            identities["paths"].update(
                {
                    absolute_path(Path(str(row["run_dir"])), "existing run_dir"),
                    absolute_path(
                        Path(str(row["attempt_dir"])), "existing attempt_dir"
                    ),
                }
            )
    return identities


def build_plan(
    completed_rows: Sequence[dict[str, str]],
    *,
    completed_csv_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    phase_root: Path,
    existing_plans: Sequence[tuple[Path, dict[str, Any]]] = (),
    supplements: Sequence[dict[str, Any]] | None = None,
    round_number: int = SUPPLEMENT_ROUND,
) -> dict[str, Any]:
    """Append explicit scientific cells, retaining the frozen D25 as provenance.

    Callers must supply all existing plans through ``existing_plans`` so even
    differently named runs, scientific coordinates, and output paths remain
    protected. This function never writes or submits a plan.
    """
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("round_number must be a positive integer")
    if round_number != SUPPLEMENT_ROUND and supplements is None:
        raise ValueError("non-default rounds require explicit supplements")
    candidates = SUPPLEMENTS if supplements is None else supplements
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("supplements must be a non-empty list of candidate objects")
    completed_csv_path = absolute_path(completed_csv_path, "--completed-csv")
    manifest_path = absolute_path(manifest_path, "--domain-split-manifest")
    phase_root = absolute_path(phase_root, "--phase-root")
    if round_number != SUPPLEMENT_ROUND and _is_round004_path(phase_root):
        raise ValueError("non-default rounds must not reuse the round_004 phase root")
    source = validate_source_d25(completed_rows)
    train_records, train_tokens = validate_manifest_against_source(manifest, source)
    specs = _model_specs()
    existing = existing_plan_identities(existing_plans)
    source_coordinates = {
        ("peptide", row["budget"], row["model_id"], int(row["seed"]))
        for row in source["rows"]
    }

    plan_rows: list[dict[str, Any]] = []
    new_ids: set[str] = set()
    new_coordinates: set[tuple[str, str, str, int]] = set()
    new_run_dirs: set[Path] = set()
    new_attempt_dirs: set[Path] = set()
    for priority_rank, candidate in enumerate(candidates, start=1):
        required_keys = {"budget", "model_id", "target_epoch_fraction", "reason"}
        if (
            not isinstance(candidate, dict)
            or not required_keys <= set(candidate)
            or set(candidate) - required_keys - {"evidence"}
        ):
            raise ValueError(
                f"candidate {priority_rank} requires {sorted(required_keys)} "
                "and permits only an optional evidence object"
            )
        if "evidence" in candidate and not isinstance(candidate["evidence"], dict):
            raise ValueError(f"candidate {priority_rank} evidence must be an object")
        budget = candidate["budget"]
        model_id = candidate["model_id"]
        raw_epoch = candidate["target_epoch_fraction"]
        if not isinstance(budget, str) or not isinstance(model_id, str):
            raise ValueError(f"candidate {priority_rank} budget/model_id must be strings")
        if (
            isinstance(raw_epoch, bool)
            or not isinstance(raw_epoch, (int, float))
            or not math.isfinite(raw_epoch)
        ):
            raise ValueError(f"candidate {priority_rank} epoch must be a finite number")
        epoch = float(raw_epoch)
        if (
            SOURCE_BUDGET_TO_EPOCH.get(budget) != epoch
            or not 0 < epoch <= MAXIMUM_EPOCH_FRACTION
        ):
            raise ValueError(f"candidate budget/epoch mismatch: {budget}/{epoch}")
        if not isinstance(candidate["reason"], str) or not candidate["reason"].strip():
            raise ValueError(f"candidate {priority_rank} requires a non-empty reason")
        spec = specs.get(model_id)
        if spec is None:
            raise ValueError(f"candidate model is absent from the registry: {model_id}")
        parameters = int(spec["parameter_count"])
        if ("peptide", budget, model_id, 42) in source_coordinates:
            raise ValueError(
                f"candidate duplicates the frozen D25: {(budget, model_id, 42)}"
            )
        if model_id not in PEPTIDE_BATCHES:
            raise ValueError(f"candidate has no peptide batch profile: {model_id}")
        target_tokens = round(epoch * train_tokens)
        target_flops = 6 * parameters * target_tokens
        microbatch, grad_accum, batch_profile = PEPTIDE_BATCHES[model_id]
        if microbatch * grad_accum != PEPTIDE_EFFECTIVE_BATCH_RECORDS:
            raise ValueError(f"peptide effective batch changed: {model_id}")
        scheduler_steps = _scheduler_steps(
            target_tokens,
            train_records,
            train_tokens,
            PEPTIDE_EFFECTIVE_BATCH_RECORDS,
        )
        warmup_steps = max(1, round(scheduler_steps * WARMUP_RATIO))
        seed = 42
        run_id = f"peptide_{budget}_{model_id}_s{seed}"
        if SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ValueError(f"unsafe peptide supplement run_id: {run_id!r}")
        run_dir = phase_root / "runs" / run_id
        attempt_dir = run_dir / "attempt_001"
        coordinate = ("peptide", budget, model_id, seed)
        if (
            run_id in existing["run_ids"]
            or coordinate in existing["coordinates"]
            or run_dir in existing["paths"]
            or attempt_dir in existing["paths"]
        ):
            raise ValueError(f"candidate conflicts with an existing plan: {run_id}")
        if run_dir.exists() or attempt_dir.exists():
            raise ValueError(f"candidate output directory already exists: {run_dir}")
        if (
            run_id in new_ids
            or coordinate in new_coordinates
            or run_dir in new_run_dirs
            or attempt_dir in new_attempt_dirs
        ):
            raise ValueError(f"duplicate generated peptide supplement: {run_id}")
        new_ids.add(run_id)
        new_coordinates.add(coordinate)
        new_run_dirs.add(run_dir)
        new_attempt_dirs.add(attempt_dir)

        source_slice = [
            {
                "run_id": row["run_id"],
                "model_id": row["model_id"],
                "parameter_count": _positive_int(row, "parameter_count"),
                "target_input_tokens_with_eos": _positive_int(
                    row, "target_input_tokens_with_eos"
                ),
                "validation_loss": _positive_float(row, "validation_loss"),
            }
            for row in source["rows"]
            if row["budget"] == budget
        ]
        plan_rows.append(
            {
                "phase": "D",
                "execution_phase": "D_peptide_surface_supplement",
                "purpose": PURPOSE,
                "scientific_protocol": SCIENTIFIC_PROTOCOL,
                "round": round_number,
                "priority_tier": "A",
                "priority_rank": priority_rank,
                "domain": "peptide",
                "budget": budget,
                "run_id": run_id,
                "model_id": model_id,
                "parameter_count": parameters,
                "non_embedding_parameter_count": int(
                    spec["non_embedding_parameter_count"]
                ),
                "target_input_tokens_with_eos": target_tokens,
                "target_flops_6nd": target_flops,
                "target_flops_after_integer_token_rounding": target_flops,
                "target_flops_relative_rounding_error": 0.0,
                "target_epoch_fraction": epoch,
                "maximum_epoch_fraction": MAXIMUM_EPOCH_FRACTION,
                "four_epoch_compliant": epoch <= MAXIMUM_EPOCH_FRACTION,
                "train_records": train_records,
                "train_input_tokens_per_epoch": train_tokens,
                "per_device_train_batch_size": microbatch,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_records": PEPTIDE_EFFECTIVE_BATCH_RECORDS,
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
                "source_status": "adaptive_supplement",
                "attempt_number": 1,
                "attempt_name": "attempt_001",
                "run_dir": str(run_dir),
                "attempt_dir": str(attempt_dir),
                "source_batch_profile": batch_profile,
                "batch_candidate_only": True,
                "need_profile": True,
                "profile_optimizer_steps": PROFILE_OPTIMIZER_STEPS,
                "profile_status": "required_before_formal_training",
                "claim_policy": "worker_empty_slot",
                "supplement_reason": candidate["reason"],
                "source_equal_data_validation_slice": source_slice,
                "approval_source": APPROVAL_SOURCE,
            }
        )
        if "evidence" in candidate:
            # This is traceable scientific motivation, never an acceptance gate.
            plan_rows[-1]["supplement_evidence"] = copy.deepcopy(candidate["evidence"])

    existing_sources = [str(path.resolve()) for path, _ in existing_plans]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "status": "ready_for_50_step_profile",
        "phase": "supplement",
        "purpose": PURPOSE,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "is_isoflop_experiment": False,
        "round": round_number,
        "approval_source": APPROVAL_SOURCE,
        "phase_root": str(phase_root),
        "domain_split_manifest": str(manifest_path),
        "source_completed_csv": str(completed_csv_path),
        "source_existing_plans": existing_sources,
        "source_peptide_d25_validation": source,
        "rows": plan_rows,
        "row_count": len(plan_rows),
        "priority_tier_counts": dict(
            sorted(Counter(row["priority_tier"] for row in plan_rows).items())
        ),
        "execution_policy": {
            "independent_single_process_runs": True,
            "world_size_per_run": 1,
            "random_initialization": True,
            "restart_from_step_zero": True,
            "worker_empty_slot_claim": True,
            "gpu_or_lane_binding_in_plan": False,
            "all_batch_candidates_require_profile": True,
            "profile_optimizer_steps": PROFILE_OPTIMIZER_STEPS,
            "formal_training_requires_successful_profile": True,
            "maximum_epoch_fraction": MAXIMUM_EPOCH_FRACTION,
        },
        "analysis_policy": {
            "metric": "final_validation.loss",
            "metric_split": "validation",
            "test_data_accessed": False,
            "comparison": "equal_input_token_slice",
            "fit_as_isoflop_curve": False,
            "retain_all_original_d25_observations": True,
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
            "save_every_steps": 0,
            "full_validation_at_end": True,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--completed-csv", type=Path, default=default_completed_csv())
    parser.add_argument("--domain-split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase-root", type=Path,
                        help="isolated output root; defaults to the selected round_NNN")
    parser.add_argument("--supplements-json", type=Path,
                        help="explicit candidate JSON list; required for non-default rounds")
    parser.add_argument("--round-number", type=int, default=SUPPLEMENT_ROUND)
    parser.add_argument("--existing-plan-json", type=Path, action="append", default=[])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--run-id-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.round_number < 1:
        parser.error("--round-number must be positive")
    if args.phase_root is None:
        args.phase_root = DEFAULT_PHASE_ROOT.with_name(f"round_{args.round_number:03d}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    completed_path = args.completed_csv.expanduser().resolve()
    manifest_path = args.domain_split_manifest.expanduser().resolve()
    phase_root = args.phase_root.expanduser().resolve()
    output_path = args.out_json.expanduser().resolve()
    run_id_path = args.run_id_file.expanduser().resolve()
    existing_paths = [path.expanduser().resolve() for path in args.existing_plan_json]
    protected_inputs = {completed_path, manifest_path, *existing_paths}
    supplements_path = (
        args.supplements_json.expanduser().resolve()
        if args.supplements_json is not None else None
    )
    if supplements_path is not None:
        protected_inputs.add(supplements_path)
    if (
        output_path == run_id_path
        or output_path in protected_inputs
        or run_id_path in protected_inputs
    ):
        raise ValueError(
            "outputs must not overwrite an input CSV, manifest, existing plan, or candidates JSON"
        )
    if args.round_number != SUPPLEMENT_ROUND and any(
        _is_round004_path(output)
        for output in (output_path, run_id_path)
    ):
        raise ValueError("non-default rounds must not write into round_004")
    supplements = load_json(supplements_path) if supplements_path is not None else None
    if supplements_path is not None and not isinstance(supplements, list):
        raise ValueError("--supplements-json must contain a JSON list")
    rows = read_csv(completed_path)
    existing_plans = [(path, load_json(path)) for path in existing_paths]
    plan = build_plan(
        rows,
        completed_csv_path=completed_path,
        manifest=load_json(manifest_path),
        manifest_path=manifest_path,
        phase_root=phase_root,
        existing_plans=existing_plans,
        supplements=supplements,
        round_number=args.round_number,
    )
    if supplements_path is not None:
        plan["source_supplements_json"] = str(supplements_path)
    atomic_json(output_path, plan)
    atomic_text(run_id_path, "".join(f"{row['run_id']}\n" for row in plan["rows"]))
    print(
        json.dumps(
            {
                "status": plan["status"],
                "scientific_protocol": SCIENTIFIC_PROTOCOL,
                "plan": str(output_path),
                "run_id_file": str(run_id_path),
                "rows": plan["row_count"],
                "queue_or_watcher_list_modified": False,
                "training_started": False,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
