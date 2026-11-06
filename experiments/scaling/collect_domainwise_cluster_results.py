#!/usr/bin/env python3
"""Merge validated cluster completions into a live validation-only snapshot.

Baseline CSV rows are retained as reported physical-machine evidence; their
historical checkpoints are not reopened.  A planned cluster run is promoted
from pending to completed only when ``run_dir/complete.json`` selects an
attempt and the existing cluster ``validate_completion`` routine accepts that
attempt's summary, run_config, full validation coverage, and final checkpoint
files.  Model weights are never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .run_domainwise_cluster_completion import (
        COMPLETION_SCHEMA,
        load_json,
        validate_completion,
    )
except ImportError:  # direct script execution
    from run_domainwise_cluster_completion import (
        COMPLETION_SCHEMA,
        load_json,
        validate_completion,
    )


SCHEMA_VERSION = "domainwise_live_cluster_results_v1"
PLOT_COMPLETED_REQUIRED = {
    "phase",
    "domain",
    "budget",
    "run_id",
    "model_id",
    "parameter_count",
    "target_flops_6nd",
    "target_epoch_fraction",
    "validation_loss",
    "hidden_size",
    "metric_split",
    "validation_complete_view",
}
PLOT_PENDING_REQUIRED = {
    "phase",
    "domain",
    "budget",
    "run_id",
    "model_id",
    "parameter_count",
    "target_flops_6nd",
    "target_epoch_fraction",
    "status",
}
SCIENTIFIC_TEXT_FIELDS = (
    "phase",
    "domain",
    "budget",
    "model_id",
)
SCIENTIFIC_INTEGER_FIELDS = (
    "parameter_count",
    "non_embedding_parameter_count",
    "target_input_tokens_with_eos",
    "target_flops_6nd",
    "effective_batch_records",
    "scheduler_total_steps",
    "warmup_steps",
    "seed",
    "vocab_size",
    "context_length",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "world_size",
)


@dataclass(frozen=True)
class PlanCandidate:
    row: dict[str, Any]
    plan_path: Path
    manifest_path: Path
    plan_purpose: str | None


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _resolve_path(value: Any, *, relative_to: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields:
        raise ValueError(f"CSV has no header: {path}")
    return fields, rows


def _require_columns(fields: Iterable[str], required: set[str], label: str) -> None:
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{label} lacks required columns: {sorted(missing)}")


def _unique_rows(
    completed: list[dict[str, str]], pending: list[dict[str, str]]
) -> None:
    seen: set[str] = set()
    for row in [*completed, *pending]:
        run_id = row.get("run_id", "")
        if not run_id or run_id in seen:
            raise ValueError(f"empty or duplicate baseline run_id: {run_id!r}")
        seen.add(run_id)
    for row in completed:
        if row.get("metric_split") != "validation" or row.get(
            "validation_complete_view"
        ) != "true":
            raise ValueError(f"baseline row is not complete validation evidence: {row['run_id']}")
        loss = float(row["validation_loss"])
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError(f"baseline row has invalid validation loss: {row['run_id']}")


def read_id_files(paths: Sequence[Path]) -> set[str]:
    run_ids: set[str] = set()
    for path in paths:
        for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines():
            run_id = line.strip()
            if run_id and not run_id.startswith("#"):
                run_ids.add(run_id)
    return run_ids


def read_plan_list_files(paths: Sequence[Path]) -> list[Path]:
    plans: list[Path] = []
    for list_path in paths:
        resolved_list = list_path.expanduser().resolve()
        for line in resolved_list.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                plans.append(_resolve_path(text, relative_to=resolved_list.parent))
    return plans


def load_plan_candidates(
    plan_paths: Sequence[Path],
) -> tuple[dict[str, list[PlanCandidate]], list[dict[str, Any]]]:
    candidates: dict[str, list[PlanCandidate]] = defaultdict(list)
    plan_provenance: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for supplied_path in plan_paths:
        plan_path = supplied_path.expanduser().resolve()
        if plan_path in seen_paths:
            continue
        seen_paths.add(plan_path)
        plan = load_json(plan_path)
        if not isinstance(plan, dict) or not isinstance(plan.get("rows"), list):
            raise ValueError(f"plan has no rows list: {plan_path}")
        manifest_path = _resolve_path(
            plan.get("domain_split_manifest", ""), relative_to=plan_path.parent
        )
        purpose = plan.get("purpose")
        if purpose is not None and not isinstance(purpose, str):
            raise ValueError(f"plan purpose must be a string: {plan_path}")
        plan_provenance.append(
            {
                "path": str(plan_path),
                "schema_version": plan.get("schema_version"),
                "purpose": purpose,
                "row_count": len(plan["rows"]),
                "domain_split_manifest": str(manifest_path),
            }
        )
        for row in plan["rows"]:
            if not isinstance(row, dict):
                raise ValueError(f"plan row is not an object: {plan_path}")
            run_id = str(row.get("run_id", ""))
            if not run_id:
                raise ValueError(f"plan row has no run_id: {plan_path}")
            if "run_dir" not in row or "attempt_dir" not in row:
                raise ValueError(f"plan row lacks output paths: {run_id}")
            candidates[run_id].append(
                PlanCandidate(
                    row=dict(row),
                    plan_path=plan_path,
                    manifest_path=manifest_path,
                    plan_purpose=purpose,
                )
            )
    if not candidates:
        raise ValueError("no plan rows were supplied")
    return dict(candidates), plan_provenance


def _same_scientific_run(run_id: str, candidates: Sequence[PlanCandidate]) -> None:
    first = candidates[0].row
    first_run_dir = _resolve_path(first["run_dir"], relative_to=candidates[0].plan_path.parent)
    for candidate in candidates[1:]:
        row = candidate.row
        run_dir = _resolve_path(row["run_dir"], relative_to=candidate.plan_path.parent)
        if run_dir != first_run_dir:
            raise ValueError(f"plan run_dir disagreement for {run_id}")
        for field in SCIENTIFIC_TEXT_FIELDS:
            if str(row.get(field)) != str(first.get(field)):
                raise ValueError(f"plan {field} disagreement for {run_id}")
        for field in SCIENTIFIC_INTEGER_FIELDS:
            if int(row[field]) != int(first[field]):
                raise ValueError(f"plan {field} disagreement for {run_id}")
        if not math.isclose(
            float(row["target_epoch_fraction"]),
            float(first["target_epoch_fraction"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"plan target_epoch_fraction disagreement for {run_id}")
        if not math.isclose(
            float(row["warmup_ratio"]),
            float(first["warmup_ratio"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"plan warmup_ratio disagreement for {run_id}")


def _match_baseline_pending(plan_row: dict[str, Any], baseline: dict[str, str]) -> None:
    run_id = str(plan_row["run_id"])
    for field in ("phase", "domain", "budget", "model_id"):
        if str(plan_row[field]) != baseline[field]:
            raise ValueError(f"plan differs from baseline pending {field} for {run_id}")
    integer_pairs = (
        ("parameter_count", "parameter_count"),
        ("target_input_tokens_with_eos", "target_input_tokens_with_eos"),
        ("target_flops_6nd", "target_flops_6nd"),
        ("effective_batch_records", "effective_batch_records"),
        ("scheduler_total_steps", "scheduler_total_steps"),
        ("warmup_steps", "warmup_steps"),
        ("seed", "seed"),
    )
    for plan_field, csv_field in integer_pairs:
        baseline_value = Decimal(baseline[csv_field])
        if (
            baseline_value != baseline_value.to_integral_value()
            or int(plan_row[plan_field]) != int(baseline_value)
        ):
            raise ValueError(f"plan differs from baseline pending {plan_field} for {run_id}")
    matrix_micro = int(
        plan_row.get(
            "baseline_per_device_train_batch_size",
            plan_row["per_device_train_batch_size"],
        )
    )
    matrix_accum = int(
        plan_row.get(
            "baseline_gradient_accumulation_steps",
            plan_row["gradient_accumulation_steps"],
        )
    )
    if matrix_micro != int(baseline["matrix_microbatch"]) or matrix_accum != int(
        baseline["matrix_grad_accum"]
    ):
        raise ValueError(f"plan baseline batch differs for {run_id}")
    if not math.isclose(
        float(plan_row["target_epoch_fraction"]),
        float(baseline["target_epoch_fraction"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"plan target_epoch_fraction differs for {run_id}")


def _is_declared_supplement(
    run_id: str,
    candidates: Sequence[PlanCandidate],
    explicit_supplements: set[str],
) -> bool:
    if run_id in explicit_supplements:
        return True
    return any(
        candidate.plan_purpose == "adaptive_bracketing_supplement"
        # Recovery changes lifecycle status, not the row's scientific purpose.
        or candidate.row.get("purpose") == "adaptive_bracketing_supplement"
        or candidate.row.get("source_status") == "adaptive_supplement"
        for candidate in candidates
    )


def _marker_candidate(
    run_id: str, candidates: Sequence[PlanCandidate]
) -> tuple[PlanCandidate | None, dict[str, Any] | None]:
    _same_scientific_run(run_id, candidates)
    run_dir = _resolve_path(
        candidates[0].row["run_dir"], relative_to=candidates[0].plan_path.parent
    )
    marker_path = run_dir / "complete.json"
    if not marker_path.is_file():
        return None, None
    marker = load_json(marker_path)
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != COMPLETION_SCHEMA
        or marker.get("status") != "complete"
        or marker.get("run_id") != run_id
    ):
        raise ValueError(f"invalid cluster completion marker: {marker_path}")
    marker_attempt = _resolve_path(marker.get("attempt_dir", ""), relative_to=run_dir)
    matching = [
        candidate
        for candidate in candidates
        if _resolve_path(
            candidate.row["attempt_dir"], relative_to=candidate.plan_path.parent
        )
        == marker_attempt
    ]
    if not matching:
        raise ValueError(f"completion attempt is absent from supplied plans: {run_id}")
    expected_summary = marker_attempt / "summary.json"
    if _resolve_path(marker.get("summary", ""), relative_to=run_dir) != expected_summary:
        raise ValueError(f"completion summary identity mismatch for {run_id}")
    if len(matching) > 1:
        attempt_record_path = marker_attempt / "cluster_attempt.json"
        if attempt_record_path.is_file():
            attempt_record = load_json(attempt_record_path)
            recorded_plan = attempt_record.get("plan_json") if isinstance(attempt_record, dict) else None
            if recorded_plan:
                plan_path = _resolve_path(recorded_plan, relative_to=marker_attempt)
                recorded = [candidate for candidate in matching if candidate.plan_path == plan_path]
                if recorded:
                    matching = recorded
        if len(matching) > 1:
            run_config = load_json(marker_attempt / "run_config.json")
            args = run_config.get("args", {}) if isinstance(run_config, dict) else {}
            by_batch = [
                candidate
                for candidate in matching
                if int(candidate.row["per_device_train_batch_size"])
                == int(args.get("batch_size", -1))
                and int(candidate.row["gradient_accumulation_steps"])
                == int(args.get("grad_accum", -1))
            ]
            if by_batch:
                matching = by_batch
    first = matching[0]
    if any(candidate.row != first.row for candidate in matching[1:]):
        raise ValueError(f"completion matches conflicting plan rows for {run_id}")
    return first, marker


def _latest_candidate(candidates: Sequence[PlanCandidate]) -> PlanCandidate:
    return max(
        candidates,
        key=lambda candidate: (
            int(candidate.row.get("attempt_number", 0)),
            str(candidate.plan_path),
        ),
    )


def _project(fields: Sequence[str], values: dict[str, Any]) -> dict[str, Any]:
    return {field: values.get(field, "") for field in fields}


def _pending_row(
    candidate: PlanCandidate, fields: Sequence[str]
) -> dict[str, Any]:
    row = candidate.row
    source = row.get("source_placement")
    source = source if isinstance(source, dict) else {}
    values = {
        **row,
        "matrix_microbatch": row.get(
            "baseline_per_device_train_batch_size",
            row["per_device_train_batch_size"],
        ),
        "matrix_grad_accum": row.get(
            "baseline_gradient_accumulation_steps",
            row["gradient_accumulation_steps"],
        ),
        "logical_host": "cluster",
        "logical_gpu": row.get("global_gpu_index", source.get("logical_gpu", "")),
        "worker_slot": (
            f"global_gpu{row['global_gpu_index']}_lane{row['lane_within_gpu']}"
            if "global_gpu_index" in row and "lane_within_gpu" in row
            else ""
        ),
        "batch_profile": row.get("batch_adjustment_reason", row.get("source_batch_profile", "")),
        "status": "adaptive_supplement_pending",
        "out_dir_relative_to_workspace": row["run_dir"],
        "source_matrix_relative_to_workspace": str(candidate.plan_path),
    }
    return _project(fields, values)


def _training_provenance(
    candidate: PlanCandidate,
    marker: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    row = candidate.row
    attempt_dir = _resolve_path(row["attempt_dir"], relative_to=candidate.plan_path.parent)
    attempt_record_path = attempt_dir / "cluster_attempt.json"
    attempt_record = load_json(attempt_record_path) if attempt_record_path.is_file() else None
    runtime = run_config.get("runtime_versions")
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    effective = run_config.get("effective_config")
    if isinstance(effective, dict) and "transformers_version" in effective:
        runtime.setdefault("transformers", effective["transformers_version"])
    source_host = None
    if isinstance(attempt_record, dict):
        source_host = attempt_record.get("source_host")
    if source_host is None and isinstance(run_config.get("runtime"), dict):
        source_host = run_config["runtime"].get("hostname")
    command = attempt_record.get("command") if isinstance(attempt_record, dict) else None
    return {
        "run_id": row["run_id"],
        "evidence_level": "cluster_complete_marker_plus_validate_completion",
        "plan_json": str(candidate.plan_path),
        "plan_schema_version": load_json(candidate.plan_path).get("schema_version"),
        "plan_purpose": candidate.plan_purpose,
        "complete_marker": str(
            _resolve_path(row["run_dir"], relative_to=candidate.plan_path.parent)
            / "complete.json"
        ),
        "attempt_dir": str(attempt_dir),
        "attempt_number": row.get("attempt_number"),
        "cluster_attempt_record": str(attempt_record_path) if attempt_record is not None else None,
        "source_host": source_host,
        "source_host_provenance": (
            "recorded_training_artifact" if source_host is not None else "unknown_not_recorded"
        ),
        "source_placement_from_inventory": row.get("source_placement"),
        "cluster_global_gpu_index": row.get("global_gpu_index"),
        "cluster_lane_within_gpu": row.get("lane_within_gpu"),
        "allocation_local_device_index": (
            attempt_record.get("device_index_within_scheduler_allocation")
            if isinstance(attempt_record, dict)
            else None
        ),
        "python_executable": (
            command[0] if isinstance(command, list) and command else None
        ),
        "training_runtime_versions": runtime,
        "training_runtime_versions_provenance": (
            "run_config_recorded_fields_only" if runtime else "unknown_not_recorded"
        ),
        "completed_at": marker.get("completed_at"),
    }


def _completed_row(
    candidate: PlanCandidate,
    marker: dict[str, Any],
    summary: dict[str, Any],
    run_config: dict[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    row = candidate.row
    args = run_config["args"]
    validation = summary["final_validation"]
    actual_flops = int(summary["actual_flops_6nd"])
    nominal_flops = int(row["target_flops_6nd"])
    source = row.get("source_placement")
    source = source if isinstance(source, dict) else {}
    attempt_dir = _resolve_path(row["attempt_dir"], relative_to=candidate.plan_path.parent)
    values = {
        **row,
        "matrix_microbatch": row.get(
            "baseline_per_device_train_batch_size",
            row["per_device_train_batch_size"],
        ),
        "matrix_grad_accum": row.get(
            "baseline_gradient_accumulation_steps",
            row["gradient_accumulation_steps"],
        ),
        "logical_host": "cluster",
        "logical_gpu": row.get("global_gpu_index", source.get("logical_gpu", "")),
        "worker_slot": (
            f"global_gpu{row['global_gpu_index']}_lane{row['lane_within_gpu']}"
            if "global_gpu_index" in row and "lane_within_gpu" in row
            else ""
        ),
        "batch_profile": row.get("batch_adjustment_reason", row.get("source_batch_profile", "")),
        "status": "complete",
        "completed_attempt": row.get("attempt_name", attempt_dir.name),
        "actual_microbatch": args["batch_size"],
        "actual_grad_accum": args["grad_accum"],
        "actual_input_tokens": summary["actual_input_tokens"],
        "actual_flops_6nd": actual_flops,
        "actual_epoch_fraction": summary["actual_epoch_fraction"],
        "validation_loss": validation["loss"],
        "validation_samples": validation["samples"],
        "validation_loss_tokens": validation["tokens"],
        "training_elapsed_sec": summary["elapsed_sec"],
        "out_dir_relative_to_workspace": row["run_dir"],
        "source_matrix_relative_to_workspace": str(candidate.plan_path),
        "completed_at": marker.get("completed_at", ""),
        "actual_optimizer_steps": summary["global_step"],
        "input_token_overshoot": int(summary["actual_input_tokens"])
        - int(row["target_input_tokens_with_eos"]),
        "actual_flops_relative_error_vs_nominal": abs(actual_flops - nominal_flops)
        / nominal_flops,
        "validation_input_tokens": validation["input_tokens"],
        "validation_ppl": validation["ppl"],
        "train_input_tokens_per_epoch": summary["train_input_tokens_per_epoch"],
        "metric_split": "validation",
        "validation_complete_view": "true",
        "summary_relative_to_workspace": str(attempt_dir / "summary.json"),
        "checkpoint_relative_to_workspace": str(attempt_dir / "final"),
        "strict_phase_evidence": "cluster_complete_marker_plus_validate_completion",
    }
    return _project(fields, values)


def collect_results(
    *,
    baseline_completed_csv: Path,
    baseline_pending_csv: Path,
    plan_jsons: Sequence[Path],
    supplement_run_ids: Iterable[str] = (),
    environment_jsons: Sequence[Path] = (),
) -> dict[str, Any]:
    """Return a serializable live snapshot without writing output files."""

    completed_path = baseline_completed_csv.expanduser().resolve()
    pending_path = baseline_pending_csv.expanduser().resolve()
    completed_fields, baseline_completed = read_csv(completed_path)
    pending_fields, baseline_pending = read_csv(pending_path)
    _require_columns(completed_fields, PLOT_COMPLETED_REQUIRED, "baseline completed CSV")
    _require_columns(pending_fields, PLOT_PENDING_REQUIRED, "baseline pending CSV")
    _unique_rows(baseline_completed, baseline_pending)

    candidates_by_id, plan_provenance = load_plan_candidates(plan_jsons)
    baseline_completed_ids = {row["run_id"] for row in baseline_completed}
    baseline_pending_by_id = {row["run_id"]: row for row in baseline_pending}
    explicit_supplements = set(supplement_run_ids)
    if explicit_supplements & (baseline_completed_ids | set(baseline_pending_by_id)):
        raise ValueError("supplement IDs must not duplicate baseline run IDs")
    missing_explicit = explicit_supplements - set(candidates_by_id)
    if missing_explicit:
        raise ValueError(f"supplement IDs are absent from plans: {sorted(missing_explicit)}")

    supplement_ids: set[str] = set()
    for run_id, candidates in candidates_by_id.items():
        if run_id in baseline_completed_ids:
            raise ValueError(f"plan attempts to replace baseline completed run: {run_id}")
        if run_id in baseline_pending_by_id:
            _same_scientific_run(run_id, candidates)
            for candidate in candidates:
                _match_baseline_pending(candidate.row, baseline_pending_by_id[run_id])
        elif _is_declared_supplement(run_id, candidates, explicit_supplements):
            supplement_ids.add(run_id)
        else:
            raise ValueError(
                f"non-baseline plan row is not declared adaptive supplement: {run_id}"
            )

    live_completed = [dict(row) for row in baseline_completed]
    new_completed_ids: list[str] = []
    run_provenance: list[dict[str, Any]] = []
    promoted: set[str] = set()
    manifest_cache: dict[Path, dict[str, Any]] = {}
    for run_id in sorted(set(baseline_pending_by_id) | supplement_ids):
        candidates = candidates_by_id.get(run_id)
        if not candidates:
            continue
        candidate, marker = _marker_candidate(run_id, candidates)
        if candidate is None or marker is None:
            continue
        if candidate.manifest_path not in manifest_cache:
            manifest = load_json(candidate.manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError(f"domain split manifest is not an object: {candidate.manifest_path}")
            manifest_cache[candidate.manifest_path] = manifest
        manifest = manifest_cache[candidate.manifest_path]
        attempt_dir = _resolve_path(
            candidate.row["attempt_dir"], relative_to=candidate.plan_path.parent
        )
        summary = validate_completion(
            attempt_dir=attempt_dir,
            row=candidate.row,
            manifest=manifest,
        )
        run_config = load_json(attempt_dir / "run_config.json")
        live_completed.append(
            _completed_row(
                candidate,
                marker,
                summary,
                run_config,
                completed_fields,
            )
        )
        promoted.add(run_id)
        new_completed_ids.append(run_id)
        run_provenance.append(
            _training_provenance(candidate, marker, run_config)
        )

    live_pending = [
        dict(row) for row in baseline_pending if row["run_id"] not in promoted
    ]
    for run_id in sorted(supplement_ids - promoted):
        live_pending.append(
            _pending_row(_latest_candidate(candidates_by_id[run_id]), pending_fields)
        )
    if len({row["run_id"] for row in live_completed}) != len(live_completed):
        raise ValueError("live completed rows contain duplicate run IDs")
    if len({row["run_id"] for row in live_pending}) != len(live_pending):
        raise ValueError("live pending rows contain duplicate run IDs")
    if {row["run_id"] for row in live_completed} & {
        row["run_id"] for row in live_pending
    }:
        raise ValueError("a run appears in both live completed and pending tables")

    environment_sources = []
    for path in environment_jsons:
        resolved = path.expanduser().resolve()
        environment_sources.append({"path": str(resolved), "record": load_json(resolved)})
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now(),
        "metric_split": "validation",
        "test_accessed": False,
        "weights_loaded": False,
        "hash_gate_used": False,
        "baseline": {
            "evidence_level": "reported_baseline_csv",
            "completed_csv": str(completed_path),
            "pending_csv": str(pending_path),
            "completed_rows": len(baseline_completed),
            "pending_rows": len(baseline_pending),
            "historical_checkpoints_opened": False,
            "logical_host_counts": dict(
                Counter(row.get("logical_host", "unknown") for row in baseline_completed)
            ),
        },
        "plans": plan_provenance,
        "environment_records": environment_sources,
        "new_completion_evidence_level": "cluster_complete_marker_plus_validate_completion",
        "new_completed_run_ids": new_completed_ids,
        "new_completed_count": len(new_completed_ids),
        "supplement_run_ids": sorted(supplement_ids),
        "live_completed_count": len(live_completed),
        "live_pending_count": len(live_pending),
        "run_provenance": run_provenance,
        "collector_runtime": {
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "note": "collector runtime only; not inferred as training runtime",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_fields": completed_fields,
        "pending_fields": pending_fields,
        "completed_rows": live_completed,
        "pending_rows": live_pending,
        "metadata": metadata,
    }


def _atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_tables(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Atomically publish live CSVs and provenance metadata."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported collector payload schema")
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / "completed_runs.csv"
    pending_path = output / "pending_runs.csv"
    metadata_path = output / "collection_metadata.json"
    baseline_paths = {
        Path(payload["metadata"]["baseline"]["completed_csv"]).resolve(),
        Path(payload["metadata"]["baseline"]["pending_csv"]).resolve(),
    }
    if completed_path in baseline_paths or pending_path in baseline_paths:
        raise ValueError("output directory would overwrite a baseline CSV")
    if metadata_path.is_file():
        previous = load_json(metadata_path)
        if previous.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"existing metadata has another schema: {metadata_path}")
        previous_ids = set(previous.get("new_completed_run_ids", []))
        current_ids = set(payload["metadata"]["new_completed_run_ids"])
        if not previous_ids <= current_ids:
            lost = sorted(previous_ids - current_ids)
            raise ValueError(f"previously collected completions disappeared: {lost}")
        previous_supplements = set(previous.get("supplement_run_ids", []))
        current_supplements = set(payload["metadata"]["supplement_run_ids"])
        if not previous_supplements <= current_supplements:
            lost = sorted(previous_supplements - current_supplements)
            raise ValueError(f"previously tracked supplements disappeared: {lost}")
    _atomic_csv(completed_path, payload["completed_fields"], payload["completed_rows"])
    _atomic_csv(pending_path, payload["pending_fields"], payload["pending_rows"])
    _atomic_json(metadata_path, payload["metadata"])
    return {
        "completed_csv": str(completed_path),
        "pending_csv": str(pending_path),
        "metadata_json": str(metadata_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-completed-csv", type=Path, required=True)
    parser.add_argument("--baseline-pending-csv", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, action="append", default=[])
    parser.add_argument("--plan-list-file", type=Path, action="append", default=[])
    parser.add_argument("--supplement-run-id-file", type=Path, action="append", default=[])
    parser.add_argument("--environment-json", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan_paths = [
        *(path.expanduser().resolve() for path in args.plan_json),
        *read_plan_list_files(args.plan_list_file),
    ]
    payload = collect_results(
        baseline_completed_csv=args.baseline_completed_csv,
        baseline_pending_csv=args.baseline_pending_csv,
        plan_jsons=plan_paths,
        supplement_run_ids=read_id_files(args.supplement_run_id_file),
        environment_jsons=args.environment_json,
    )
    outputs = write_tables(payload, args.output_dir)
    print(
        json.dumps(
            {
                "status": "written",
                "live_completed": payload["metadata"]["live_completed_count"],
                "live_pending": payload["metadata"]["live_pending_count"],
                "new_completed": payload["metadata"]["new_completed_count"],
                **outputs,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
