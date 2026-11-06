#!/usr/bin/env python3
"""Strict Phase-C/D artifact verifier and final B+C+D 145-cell verifier.

The verifier is read-only except for an optional atomically-written report.  It
does not hash released data repeatedly: the frozen, path-independent manifest
semantic contract is the acceptance authority.  Missing completions are a
normal ``incomplete`` result; malformed or scientifically drifting artifacts
are ``invalid``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from finalize_domainwise_phase_matrix import (
    EFFECTIVE_BATCH_RECORDS,
    FROZEN_TRAINING_CONTROL,
    MATRIX_SCHEMA,
    PHASES,
    PROFILE_EVIDENCE_SCHEMA,
    PROPOSAL_STATUS,
    REVIEW_SCHEMA,
    manifest_semantic_contract,
    matrix_semantic_fingerprint,
    rebuild_phase_c_rows,
    required_profile_evidence,
    validate_hardware_layout,
    validate_manifest,
    validate_phase_c_review,
    validate_phase_shape,
    validate_profile_evidence,
    validate_proposal_manifest_contract,
    validate_proposal_row,
    validate_row_against_manifest,
)
from verify_domainwise_campaign import (
    atomic_write_json,
    command_has_once,
    command_value,
    finite_number,
    load_json,
    validate_checkpoint,
    verify_campaign as verify_phase_b_campaign,
)
from run_domainwise_phase_slot import (
    RECORD_CONSUMPTION_AUDIT_SCHEMA,
    build_record_consumption_audit,
)


ATTEMPT_CONTRACT_SCHEMA = "domainwise_phase_attempt_contract_v1"
SCALING_CONFIG_SCHEMA = "domainwise_phase_scaling_config_v1"
RETRY_STATE_SCHEMA = "domainwise_phase_retry_state_v1"
PREJOURNAL_ORPHAN_SCHEMA = "domainwise_prejournal_attempt_orphan_v1"
PACKED_MODEL_IMPLEMENTATION = "packed-xformers"
PACKED_ATTENTION_BACKEND = "xformers_memory_efficient_block_diagonal_causal"
ATTEMPT_RE = re.compile(r"^attempt_([0-9]{3,})$")
OOM_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)
PHASE_RUNTIME_FIELDS = {
    "matrix_status",
    "gpu_count",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "measured_input_tokens_per_step",
    "max_steps",
    "warmup_steps",
}
MODEL_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)


def add_if(reasons: list[str], condition: bool, message: str) -> None:
    if condition:
        reasons.append(message)


def resolved(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def execution_placement_reasons(
    matrix_path: Path,
    row: dict[str, Any],
    contract: dict[str, Any],
    *,
    scaling: dict[str, Any] | None = None,
    marker: dict[str, Any] | None = None,
) -> list[str]:
    """Validate explicit physical relocation without changing logical lineage.

    A published canonical plan makes placement mandatory for its remote C rows.
    Earlier local/B/D artifacts keep their original contract.  This only reads
    small JSON placement artifacts; checkpoints and test data are not involved.
    """
    matrix_path = resolved(matrix_path)
    canonical_plan = matrix_path.parent / "local_execution_plan.json"
    required = (
        row.get("assigned_host") == "remote"
        and str(row.get("execution_phase", "")).startswith("C_")
        and (canonical_plan.exists() or canonical_plan.is_symlink())
    )
    artifacts = {"attempt contract": contract}
    if scaling is not None:
        artifacts["scaling config"] = scaling
    if marker is not None:
        artifacts["completion marker"] = marker
    claim_keys = {
        "execution_placement", "actual_execution_host", "physical_host_id",
        "execution_plan_json",
    }
    if not required and not any(claim_keys.intersection(value) for value in artifacts.values()):
        return []
    placement = contract.get("execution_placement")
    if not isinstance(placement, dict):
        return ["attempt contract execution_placement is required and must be an object"]
    plan_value = placement.get("plan_json")
    if not isinstance(plan_value, str) or not plan_value:
        return ["execution_placement plan_json must be a non-empty path"]
    plan_path = resolved(plan_value)
    if required and plan_path != canonical_plan.resolve():
        return ["execution_placement does not use the canonical local execution plan"]
    try:
        from domainwise_local_execution import execution_placement_for_row

        expected = execution_placement_for_row(matrix_path, plan_path, row)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        return [f"execution_placement plan validation failed: {exc}"]
    reasons: list[str] = []
    for label, artifact in artifacts.items():
        # Markers predate this optional redundant field.  Contract + scaling
        # are the authoritative, required pair for every relocated attempt.
        if label != "completion marker" or "execution_placement" in artifact:
            add_if(reasons, artifact.get("execution_placement") != expected,
                   f"{label} execution_placement mismatch")
        for key, expected_value in (
            ("actual_execution_host", expected["actual_execution_host"]),
            ("physical_host_id", expected["actual_execution_host"]),
            ("execution_plan_json", str(plan_path)),
        ):
            if key in artifact:
                add_if(reasons, artifact[key] != expected_value,
                       f"{label} {key} disagrees with execution_placement")
    return reasons


def canonical_json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def command_uses_pretrain_entrypoint(command: list[Any]) -> bool:
    if len(command) < 2:
        return False
    parts = Path(str(command[1])).parts
    return tuple(parts[-3:]) == (
        "cli",
        "pretrain",
        "train_qwen3_pretrain.py",
    )


def phase_spec(phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unsupported phase {phase!r}")
    return PHASES[phase]


def scientific_cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    execution = str(row.get("execution_phase"))
    base: tuple[Any, ...] = (
        execution,
        str(row.get("domain")),
        str(row.get("budget_label")),
    )
    if execution == "D_peptide_surface":
        return (*base, str(row.get("model_id")), int(row.get("seed", -1)))
    return (*base, int(row.get("model_slot", -1)), int(row.get("seed", -1)))


def derived_run_id(row: dict[str, Any]) -> str:
    return (
        f"{row['domain']}_{row['budget_label']}_{row['model_id']}_s{int(row['seed'])}"
    )


def scientific_projection(
    row: dict[str, Any], reference: dict[str, Any] | None = None
) -> dict[str, Any]:
    fields = reference.keys() if reference is not None else row.keys()
    return {
        key: row.get(key)
        for key in fields
        if key not in PHASE_RUNTIME_FIELDS
    }


def next_oom_batch(effective_batch_records: int, current_microbatch: int) -> tuple[int, int] | None:
    effective = int(effective_batch_records)
    current = int(current_microbatch)
    if effective <= 0 or current <= 0 or effective % current != 0:
        raise ValueError("microbatch must be a positive exact divisor of effective batch")
    for candidate in range(current // 2, 0, -1):
        if effective % candidate == 0:
            return candidate, effective // candidate
    return None


def expected_runtime_profile(row: dict[str, Any], level: int) -> dict[str, Any]:
    if level < 0:
        raise ValueError("negative fallback level")
    effective = int(row["effective_batch_records"])
    micro = int(row["per_device_train_batch_size"])
    grad_accum = int(row["gradient_accumulation_steps"])
    for _ in range(level):
        fallback = next_oom_batch(effective, micro)
        if fallback is None:
            raise ValueError("fallback level exceeds exact-divisor sequence")
        micro, grad_accum = fallback
    profile_id = str(row["batch_profile_id"])
    if level:
        profile_id += f"_oom_fallback_{micro}x{grad_accum}"
    return {
        "source": "matrix" if level == 0 else "oom_fallback",
        "level": level,
        "batch_profile_id": profile_id,
        "per_device_train_batch_size": micro,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_records": effective,
    }


def validate_runtime_profile(row: dict[str, Any], profile: Any) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if not isinstance(profile, dict):
        return None, ["runtime_profile is not an object"]
    try:
        expected = expected_runtime_profile(row, int(profile["level"]))
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"invalid runtime fallback level: {type(exc).__name__}: {exc}"]
    add_if(reasons, profile != expected, f"runtime_profile differs from approved profile: expected={expected}")
    return expected, reasons


def strict_manifest_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest root is not an object")
    return manifest, manifest_semantic_contract(manifest)


def manifest_token_paths(manifest: dict[str, Any]) -> list[str]:
    paths = {
        str(resolved(shard["token_file"]))
        for domain_views in manifest["views"].values()
        for view in domain_views.values()
        for shard in view["shards"]
    }
    if not paths:
        raise ValueError("manifest contains no token files")
    return sorted(paths)


def proposal_rows(proposal: dict[str, Any], execution_phase: str) -> list[dict[str, Any]]:
    rows = proposal.get("rows")
    if not isinstance(rows, list):
        raise ValueError("proposal has no rows list")
    return [row for row in rows if isinstance(row, dict) and row.get("execution_phase") == execution_phase]


def expected_phase_scientific_rows(
    phase: str,
    proposal: dict[str, Any],
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    spec = phase_spec(phase)
    source = proposal_rows(proposal, str(spec["execution_phase"]))
    review: dict[str, Any] | None = None
    if phase == "c":
        gate = matrix["metadata"].get("review_gate")
        if not isinstance(gate, dict):
            raise ValueError("Phase-C matrix has no review_gate")
        if (
            gate.get("required") is not True
            or gate.get("satisfied") is not True
            or gate.get("selection_source") != "approved_phase_b_validation_review"
            or gate.get("test_data_accessed") is not False
        ):
            raise ValueError("Phase-C review gate is not strictly satisfied without test access")
        review_path = resolved(gate.get("review_json"))
        review = load_json(review_path)
        if not isinstance(review, dict) or review.get("schema_version") != REVIEW_SCHEMA:
            raise ValueError("Phase-C review JSON/schema is invalid")
        if gate.get("review_semantic_fingerprint") != canonical_json_fingerprint(review):
            raise ValueError("Phase-C review content differs from the matrix-frozen fingerprint")
        selections = validate_phase_c_review(review, proposal, source)
        source = rebuild_phase_c_rows(proposal, source, selections)
    else:
        gate = matrix["metadata"].get("review_gate")
        if (
            not isinstance(gate, dict)
            or gate.get("required") is not False
            or gate.get("satisfied") is not True
            or gate.get("selection_source") != "proposal_v1"
            or gate.get("review_json") is not None
            or gate.get("review_semantic_fingerprint") is not None
            or gate.get("test_data_accessed") is not None
        ):
            raise ValueError("Phase-D review gate metadata is invalid")
    return source, review


def validate_matrix_and_rows(
    matrix_path: Path,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict) or not isinstance(matrix.get("metadata"), dict):
        raise ValueError("matrix root/metadata is invalid")
    metadata = matrix["metadata"]
    phase = str(metadata.get("phase", ""))
    spec = phase_spec(phase)
    expected_status = str(spec["formal_status"])
    expected_rows_count = int(spec["expected_rows"])
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ValueError("matrix has no rows list")
    add_if(issues, metadata.get("schema_version") != MATRIX_SCHEMA, "matrix schema_version mismatch")
    add_if(issues, metadata.get("matrix_version") != f"domainwise_{spec['phase_name']}_matrix_v1", "matrix_version mismatch")
    add_if(issues, metadata.get("status") != expected_status, "matrix is not formal runnable status")
    add_if(issues, metadata.get("is_runnable") is not True, "matrix is_runnable is not true")
    add_if(issues, metadata.get("phase_name") != spec["phase_name"], "phase_name mismatch")
    add_if(issues, metadata.get("execution_phase") != spec["execution_phase"], "execution_phase mismatch")
    add_if(issues, metadata.get("proposal_status") != PROPOSAL_STATUS, "proposal status mismatch")
    add_if(
        issues,
        metadata.get("training_control") != FROZEN_TRAINING_CONTROL,
        "matrix training control differs from the frozen optimizer contract",
    )
    add_if(
        issues,
        metadata.get("effective_batch_records") != EFFECTIVE_BATCH_RECORDS,
        "matrix effective-record batches differ from the frozen domain contract",
    )
    add_if(issues, len(rows) != expected_rows_count, f"row count is not exactly {expected_rows_count}")
    try:
        add_if(issues, int(metadata.get("rows", -1)) != len(rows), "metadata row count mismatch")
    except (TypeError, ValueError):
        issues.append("metadata rows is not an integer")
    add_if(
        issues,
        metadata.get("proposal_matrix_version") != proposal.get("metadata", {}).get("matrix_version"),
        "proposal matrix version mismatch",
    )
    try:
        fingerprint = matrix_semantic_fingerprint(matrix)
        expected_identity = {
            "algorithm": "sha256_canonical_json_excluding_timestamps_and_identity",
            "value": fingerprint,
            "scope": "matrix_semantics_not_dataset_payload_files",
        }
        add_if(
            issues,
            metadata.get("semantic_fingerprint") != expected_identity,
            "matrix semantic fingerprint mismatch",
        )
        add_if(
            issues,
            metadata.get("generation_id")
            != f"{spec['phase_name']}_{fingerprint[:20]}",
            "matrix generation_id mismatch",
        )
    except Exception as exc:
        issues.append(f"matrix identity validation failed: {type(exc).__name__}: {exc}")
    try:
        output_base = resolved(metadata["output_base"])
        expected_base = (
            resolved(metadata["run_root"])
            / str(metadata["experiment_name"])
            / str(metadata["phase_name"])
        ).resolve()
        add_if(issues, output_base != expected_base, "output_base differs from run_root/experiment/phase")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        issues.append(f"invalid output-base metadata: {type(exc).__name__}: {exc}")
        output_base = matrix_path.parent.resolve()

    try:
        manifest_path = resolved(metadata["domain_split_manifest"])
        manifest, actual_semantics = strict_manifest_contract(manifest_path)
        add_if(
            issues,
            metadata.get("domain_split_semantic_contract") != actual_semantics,
            "matrix frozen manifest semantics differ from source manifest",
        )
        stats = validate_manifest(manifest)
        validate_proposal_manifest_contract(proposal, manifest)
    except Exception as exc:
        issues.append(f"manifest semantic validation failed: {type(exc).__name__}: {exc}")
        manifest = {}
        actual_semantics = {}
        stats = {}

    try:
        expected_scientific, _review = expected_phase_scientific_rows(phase, proposal, matrix)
        for expected_row in expected_scientific:
            validate_proposal_row(expected_row, proposal)
            validate_row_against_manifest(expected_row, manifest, stats)
    except Exception as exc:
        issues.append(f"scientific source/review validation failed: {type(exc).__name__}: {exc}")
        expected_scientific = []
    expected_by_cell = {scientific_cell_key(row): row for row in expected_scientific}
    actual_by_cell: dict[tuple[Any, ...], dict[str, Any]] = {}
    run_ids: set[str] = set()
    out_dirs: set[str] = set()

    try:
        validate_phase_shape(phase, rows)
    except Exception as exc:
        issues.append(f"phase shape validation failed: {type(exc).__name__}: {exc}")
    try:
        recorded_layout = metadata.get("hardware_layout", {})
        if not isinstance(recorded_layout, dict) or not isinstance(recorded_layout.get("devices"), list):
            raise ValueError("hardware_layout/devices is invalid")
        input_layout = {
            "schema_version": recorded_layout.get("schema_version"),
            "layout_id": recorded_layout.get("layout_id"),
            "devices": [
                {
                    key: device.get(key)
                    for key in ("host_id", "gpu_id", "memory_gib", "concurrent_slots")
                }
                for device in recorded_layout["devices"]
                if isinstance(device, dict)
            ],
        }
        layout_devices = validate_hardware_layout(input_layout)
        add_if(
            issues,
            recorded_layout.get("devices") != layout_devices,
            "recorded hardware devices differ from normalized eligible layout",
        )
    except Exception as exc:
        issues.append(f"hardware layout validation failed: {type(exc).__name__}: {exc}")
        layout_devices = []
    devices = {
        (str(device["host_id"]), int(device["gpu_id"])): device
        for device in layout_devices
    }
    try:
        requirements, contract_id = required_profile_evidence(rows)
        gate = metadata.get("batch_profile_evidence_gate")
        if not isinstance(gate, dict):
            raise ValueError("matrix has no batch_profile_evidence_gate")
        if (
            gate.get("required") is not True
            or gate.get("satisfied") is not True
            or gate.get("schema_version") != PROFILE_EVIDENCE_SCHEMA
            or gate.get("requirements") != requirements
            or gate.get("profile_contract_id") != contract_id
        ):
            raise ValueError("batch-profile evidence gate differs from assigned profiles")
        evidence_path = resolved(gate.get("report_json"))
        evidence = load_json(evidence_path)
        if not isinstance(evidence, dict):
            raise ValueError("batch-profile evidence report is not an object")
        if gate.get("report_semantic_fingerprint") != canonical_json_fingerprint(evidence):
            raise ValueError("batch-profile evidence differs from the matrix-frozen fingerprint")
        validate_profile_evidence(
            evidence,
            phase=phase,
            layout_id=str(metadata["hardware_layout"]["layout_id"]),
            requirements=requirements,
            contract_id=contract_id,
        )
    except Exception as exc:
        issues.append(f"batch-profile evidence validation failed: {type(exc).__name__}: {exc}")
    semantic_views = actual_semantics.get("views", {}) if isinstance(actual_semantics, dict) else {}
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{prefix}: row is not an object")
            continue
        try:
            cell = scientific_cell_key(row)
            add_if(issues, cell in actual_by_cell, f"{prefix}: duplicate scientific cell {cell}")
            actual_by_cell[cell] = row
            expected = expected_by_cell.get(cell)
            add_if(issues, expected is None, f"{prefix}: scientific cell absent from reviewed proposal {cell}")
            if expected is not None:
                add_if(
                    issues,
                    scientific_projection(row, expected) != scientific_projection(expected),
                    f"{prefix}: scientific fields drift from reviewed proposal cell {cell}",
                )
            run_id = str(row["run_id"])
            add_if(issues, run_id != derived_run_id(row), f"{prefix}: run_id is not deterministically derived")
            add_if(issues, run_id in run_ids, f"{prefix}: duplicate run_id {run_id}")
            run_ids.add(run_id)
            out_dir = resolved(row["out_dir"])
            add_if(issues, out_dir != output_base / "runs" / run_id, f"{prefix}: out_dir mismatch")
            add_if(issues, str(out_dir) in out_dirs, f"{prefix}: duplicate out_dir")
            out_dirs.add(str(out_dir))
            add_if(issues, row.get("matrix_status") != expected_status, f"{prefix}: matrix_status mismatch")
            add_if(issues, int(row.get("world_size", -1)) != 1, f"{prefix}: world_size != 1")
            add_if(issues, int(row.get("gpu_count", -1)) != 1, f"{prefix}: gpu_count != 1")
            add_if(issues, row.get("independent_random_initialization") is not True, f"{prefix}: initialization is not independent")
            add_if(issues, row.get("token_stop_authority") != "target_input_tokens_with_eos", f"{prefix}: token authority mismatch")
            add_if(issues, row.get("bpe_unfold_policy") != "none", f"{prefix}: BPE unfolding is enabled")
            add_if(
                issues,
                row.get("model_implementation") != PACKED_MODEL_IMPLEMENTATION,
                f"{prefix}: model implementation is not packed-xformers",
            )
            add_if(
                issues,
                row.get("attention_backend") != PACKED_ATTENTION_BACKEND,
                f"{prefix}: actual attention backend is not xFormers block-diagonal causal",
            )
            add_if(issues, row.get("train_split") != "train", f"{prefix}: train split mismatch")
            add_if(issues, row.get("validation_split") != "validation", f"{prefix}: validation split mismatch")
            add_if(issues, row.get("test_split") != "test_reserved_until_analysis_freeze", f"{prefix}: test is not reserved")
            add_if(issues, resolved(row["domain_split_manifest"]) != resolved(metadata["domain_split_manifest"]), f"{prefix}: manifest path mismatch")

            parameters = int(row["parameter_count"])
            target_tokens = int(row["target_input_tokens_with_eos"])
            rounded_flops = 6 * parameters * target_tokens
            add_if(issues, int(row["target_flops_after_integer_token_rounding"]) != rounded_flops, f"{prefix}: FLOPs != 6*N*D")
            train_epoch_tokens = int(row["train_input_tokens_per_epoch"])
            epoch_fraction = target_tokens / train_epoch_tokens
            declared_epoch_fraction = float(row["target_epoch_fraction"])
            if phase == "d":
                add_if(
                    issues,
                    target_tokens != max(1, round(train_epoch_tokens * declared_epoch_fraction)),
                    f"{prefix}: peptide target does not implement declared epoch fraction",
                )
            else:
                add_if(issues, not math.isclose(declared_epoch_fraction, epoch_fraction, rel_tol=0.0, abs_tol=1e-15), f"{prefix}: epoch fraction mismatch")
            add_if(issues, declared_epoch_fraction > 4.0 or row.get("four_epoch_compliant") is not True, f"{prefix}: violates four-epoch cap")

            domain = str(row["domain"])
            views = semantic_views[domain]
            add_if(issues, int(row["domain_id"]) != int(views["train"]["domain_id"]), f"{prefix}: domain_id mismatch")
            add_if(issues, int(row["train_records"]) != int(views["train"]["records"]), f"{prefix}: train records mismatch")
            add_if(issues, int(row["train_input_tokens_per_epoch"]) != int(views["train"]["effective_input_tokens_with_eos"]), f"{prefix}: train tokens mismatch")
            add_if(issues, int(row["validation_loss_tokens"]) != int(views["validation"]["supervised_causal_loss_tokens"]), f"{prefix}: validation loss tokens mismatch")
            add_if(issues, int(row["test_loss_tokens"]) != int(views["test"]["supervised_causal_loss_tokens"]), f"{prefix}: reserved test token metadata mismatch")
            add_if(issues, int(row["context_length"]) != int(actual_semantics["dataset"]["context_length"]), f"{prefix}: context length mismatch")

            host = str(row["assigned_host"])
            gpu = int(row["assigned_gpu"])
            local_slot = int(row["gpu_local_slot"])
            device = devices.get((host, gpu))
            add_if(issues, device is None, f"{prefix}: assigned hardware is absent from layout")
            add_if(issues, str(row["worker_slot"]) != f"{host}_gpu{gpu}_slot{local_slot}", f"{prefix}: worker_slot mismatch")
            if device is not None:
                add_if(issues, not 0 <= local_slot < int(device["concurrent_slots"]), f"{prefix}: local slot exceeds device concurrency")
                add_if(issues, row.get("assigned_hardware_class") != device["hardware_class"], f"{prefix}: hardware class mismatch")
                add_if(issues, int(row["assigned_device_memory_gib"]) != int(device["memory_gib"]), f"{prefix}: device memory mismatch")
                add_if(issues, int(row["assigned_device_concurrent_slots"]) != int(device["concurrent_slots"]), f"{prefix}: device concurrency mismatch")

            micro = int(row["per_device_train_batch_size"])
            grad_accum = int(row["gradient_accumulation_steps"])
            effective = int(row["effective_batch_records"])
            add_if(issues, micro <= 0 or grad_accum <= 0 or micro * grad_accum != effective, f"{prefix}: initial batch does not preserve effective records")
            add_if(issues, int(metadata.get("effective_batch_records", {}).get(domain, -1)) != effective, f"{prefix}: metadata effective batch mismatch")
            eligibility = row.get("hardware_eligibility")
            profiles = eligibility.get("profiles") if isinstance(eligibility, dict) else None
            selected = [profile for profile in profiles or [] if isinstance(profile, dict) and profile.get("batch_profile_id") == row.get("batch_profile_id")]
            add_if(issues, len(selected) != 1, f"{prefix}: selected batch profile is not unique in eligibility")
            if len(selected) == 1:
                profile = selected[0]
                for key, value in (
                    ("hardware_class", row["assigned_hardware_class"]),
                    ("per_device_train_batch_size", micro),
                    ("gradient_accumulation_steps", grad_accum),
                    ("effective_batch_records", effective),
                ):
                    add_if(issues, profile.get(key) != value, f"{prefix}: selected profile {key} mismatch")
            expected_tokens_per_step = int(row["train_input_tokens_per_epoch"]) / int(row["train_records"]) * effective
            add_if(issues, not math.isclose(float(row["expected_input_tokens_per_optimizer_step"]), expected_tokens_per_step, rel_tol=1e-12, abs_tol=1e-9), f"{prefix}: expected tokens/update mismatch")
            scheduler_steps = math.ceil(target_tokens / expected_tokens_per_step)
            add_if(issues, int(row["max_steps"]) != scheduler_steps or int(row["scheduler_total_steps"]) != scheduler_steps, f"{prefix}: scheduler horizon mismatch")
            expected_warmup = max(1, round(scheduler_steps * float(row["warmup_ratio"])))
            add_if(issues, int(row["warmup_steps"]) != expected_warmup, f"{prefix}: warmup steps mismatch")
        except (KeyError, TypeError, ValueError, OSError, ZeroDivisionError) as exc:
            issues.append(f"{prefix}: invalid field: {type(exc).__name__}: {exc}")
    add_if(issues, set(actual_by_cell) != set(expected_by_cell), "matrix scientific cell set differs from reviewed proposal")
    return matrix, manifest, rows, issues


def read_console_tail(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace").lower()


def validate_retry_ledger(
    run_dir: Path,
    row: dict[str, Any],
    completed_attempt: Path,
    completed_profile: dict[str, Any],
    matrix_path: Path,
    effective_manifest_path: Path,
    effective_manifest: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    try:
        state = load_json(run_dir / "retry_state.json")
    except Exception as exc:
        return [f"invalid retry_state: {type(exc).__name__}: {exc}"]
    if not isinstance(state, dict):
        return ["retry_state is not an object"]
    try:
        frozen_matrix = load_json(matrix_path)
        matrix_generation = frozen_matrix["metadata"]["generation_id"]
        matrix_fingerprint = frozen_matrix["metadata"]["semantic_fingerprint"]["value"]
        completed_number = int(ATTEMPT_RE.fullmatch(completed_attempt.name).group(1))
        add_if(reasons, state.get("schema_version") != RETRY_STATE_SCHEMA, "retry_state schema mismatch")
        add_if(reasons, state.get("run_id") != row["run_id"], "retry_state run_id mismatch")
        add_if(reasons, state.get("status") != "complete", "retry_state is not complete")
        add_if(reasons, int(state.get("completed_attempt", -1)) != completed_number, "retry_state completed_attempt mismatch")
        add_if(reasons, int(state.get("effective_batch_records", -1)) != int(row["effective_batch_records"]), "retry_state effective batch mismatch")
        history = state.get("history")
        if not isinstance(history, list):
            return reasons + ["retry_state history is not a list"]
        prejournal = state.get("prejournal_orphans", [])
        if not isinstance(prejournal, list):
            return reasons + ["retry_state prejournal_orphans is not a list"]
        prejournal_numbers: set[int] = set()
        for index, item in enumerate(prejournal):
            if not isinstance(item, dict):
                reasons.append(f"prejournal_orphans[{index}] is not an object")
                continue
            number = int(item.get("attempt_number", -1))
            expected_dir = run_dir / f"attempt_{number:03d}"
            add_if(
                reasons,
                item.get("schema_version") != PREJOURNAL_ORPHAN_SCHEMA,
                f"pre-journal orphan {number} schema mismatch",
            )
            add_if(
                reasons,
                number <= 0 or number in prejournal_numbers,
                f"pre-journal orphan {number} is duplicate/invalid",
            )
            prejournal_numbers.add(number)
            add_if(
                reasons,
                resolved(item.get("attempt_dir")) != expected_dir,
                f"pre-journal orphan {number} path mismatch",
            )
            add_if(
                reasons,
                item.get("reason") != "worker_crash_before_attempt_journal",
                f"pre-journal orphan {number} reason mismatch",
            )
            try:
                entries = sorted(
                    str(path.relative_to(expected_dir))
                    for path in expected_dir.rglob("*")
                    if path.is_file()
                )
                directories = [
                    path for path in expected_dir.rglob("*") if path.is_dir()
                ]
            except OSError as exc:
                reasons.append(f"pre-journal orphan {number} is unreadable: {exc}")
                entries, directories = [], []
            add_if(
                reasons,
                not expected_dir.is_dir(),
                f"pre-journal orphan {number} directory is missing",
            )
            add_if(
                reasons,
                bool(directories),
                f"pre-journal orphan {number} contains subdirectories",
            )
            allowed = {
                "attempt_contract.json",
                "attempt_contract.json.tmp",
                "scaling_config.json",
                "scaling_config.json.tmp",
            }
            add_if(
                reasons,
                any(name not in allowed for name in entries),
                f"pre-journal orphan {number} contains runtime artifacts",
            )
            add_if(
                reasons,
                item.get("artifacts_present") != entries,
                f"pre-journal orphan {number} artifact inventory mismatch",
            )
        level = 0
        starts = 0
        oom_failures = 0
        active_attempt: int | None = None
        completed_events = 0
        seen_attempt_numbers: set[int] = set()
        last_started_number = 0
        for index, event in enumerate(history):
            if not isinstance(event, dict):
                reasons.append(f"retry history[{index}] is not an object")
                continue
            kind = event.get("event")
            if kind == "attempt_start":
                starts += 1
                attempt_number = int(event["attempt_number"])
                add_if(reasons, active_attempt is not None, f"attempt {attempt_number} starts before prior attempt ends")
                add_if(reasons, attempt_number in seen_attempt_numbers, f"attempt {attempt_number} starts more than once")
                add_if(reasons, attempt_number <= last_started_number, f"attempt {attempt_number} is not strictly ordered")
                seen_attempt_numbers.add(attempt_number)
                last_started_number = attempt_number
                active_attempt = attempt_number
                expected = expected_runtime_profile(row, level)
                add_if(reasons, int(event.get("microbatch", -1)) != expected["per_device_train_batch_size"], f"attempt {attempt_number} retry microbatch mismatch")
                add_if(reasons, int(event.get("grad_accum", -1)) != expected["gradient_accumulation_steps"], f"attempt {attempt_number} retry grad_accum mismatch")
                attempt_dir = run_dir / f"attempt_{attempt_number:03d}"
                contract = load_json(attempt_dir / "attempt_contract.json")
                add_if(reasons, not isinstance(contract, dict), f"attempt {attempt_number} contract is not an object")
                if isinstance(contract, dict):
                    reasons.extend(
                        f"attempt {attempt_number} {reason}"
                        for reason in execution_placement_reasons(matrix_path, row, contract)
                    )
                    add_if(reasons, contract.get("schema_version") != ATTEMPT_CONTRACT_SCHEMA, f"attempt {attempt_number} contract schema mismatch")
                    add_if(reasons, contract.get("run_id") != row["run_id"], f"attempt {attempt_number} contract run mismatch")
                    add_if(reasons, int(contract.get("attempt_number", -1)) != attempt_number, f"attempt {attempt_number} contract number mismatch")
                    add_if(reasons, resolved(contract.get("matrix_json")) != matrix_path, f"attempt {attempt_number} contract matrix mismatch")
                    add_if(reasons, contract.get("matrix_generation_id") != matrix_generation, f"attempt {attempt_number} contract matrix generation mismatch")
                    add_if(reasons, contract.get("matrix_semantic_fingerprint") != matrix_fingerprint, f"attempt {attempt_number} contract matrix fingerprint mismatch")
                    add_if(reasons, contract.get("matrix_status") != row["matrix_status"], f"attempt {attempt_number} contract status mismatch")
                    add_if(reasons, contract.get("phase") != str(row["execution_phase"])[0].lower(), f"attempt {attempt_number} contract phase mismatch")
                    add_if(reasons, contract.get("host_id") != row["assigned_host"], f"attempt {attempt_number} contract host mismatch")
                    add_if(reasons, contract.get("worker_slot") != row["worker_slot"], f"attempt {attempt_number} contract slot mismatch")
                    add_if(reasons, int(contract.get("physical_gpu", -1)) != int(row["assigned_gpu"]), f"attempt {attempt_number} contract GPU mismatch")
                    add_if(reasons, contract.get("model_implementation") != PACKED_MODEL_IMPLEMENTATION, f"attempt {attempt_number} model implementation mismatch")
                    add_if(reasons, contract.get("attention_backend") != PACKED_ATTENTION_BACKEND, f"attempt {attempt_number} attention backend mismatch")
                    add_if(reasons, contract.get("runtime_profile") != expected, f"attempt {attempt_number} contract runtime profile mismatch")
                    add_if(reasons, contract.get("batch_profile") != expected, f"attempt {attempt_number} contract batch profile mismatch")
                    add_if(reasons, resolved(contract.get("domain_split_manifest")) != effective_manifest_path, f"attempt {attempt_number} contract manifest mismatch")
                    command = contract.get("command")
                    if not isinstance(command, list):
                        reasons.append(f"attempt {attempt_number} command is not a list")
                    else:
                        add_if(reasons, not command_uses_pretrain_entrypoint(command), f"attempt {attempt_number} command uses another training entrypoint")
                        expected_flags = {
                            "--token-files": str(
                                Path(effective_manifest["dataset"]["compact_dir"])
                                / "shard_*.bin"
                            ),
                            "--val-token-files": str(
                                Path(effective_manifest["dataset"]["compact_dir"])
                                / "shard_*.bin"
                            ),
                            "--domain-split-manifest": str(effective_manifest_path),
                            "--domain-name": str(row["domain"]),
                            "--model-implementation": PACKED_MODEL_IMPLEMENTATION,
                            "--vocab-size": str(row["vocab_size"]),
                            "--hidden-size": str(row["hidden_size"]),
                            "--intermediate-size": str(row["intermediate_size"]),
                            "--num-hidden-layers": str(row["num_hidden_layers"]),
                            "--num-attention-heads": str(row["num_attention_heads"]),
                            "--num-key-value-heads": str(row["num_key_value_heads"]),
                            "--head-dim": str(row["head_dim"]),
                            "--batch-size": str(expected["per_device_train_batch_size"]),
                            "--grad-accum": str(expected["gradient_accumulation_steps"]),
                            "--max-steps": str(row["scheduler_total_steps"]),
                            "--target-input-tokens": str(row["target_input_tokens_with_eos"]),
                            "--seed": str(row["seed"]),
                        }
                        for flag, value in expected_flags.items():
                            add_if(reasons, command_value(command, flag) != value, f"attempt {attempt_number} command {flag} mismatch")
                        for flag in ("--online-truncate-long-samples", "--online-shuffle", "--final-eval", "--bf16", "--no-gradient-checkpointing"):
                            add_if(reasons, not command_has_once(command, flag), f"attempt {attempt_number} command missing unique {flag}")
                        for forbidden in ("--init-checkpoint", "--resume-global-step", "--gradient-checkpointing", "--dry-run"):
                            add_if(reasons, any(str(token) == forbidden for token in command), f"attempt {attempt_number} command unexpectedly contains {forbidden}")
                        add_if(reasons, any(str(token).startswith("--test") for token in command), f"attempt {attempt_number} command accesses a test option")
            elif kind == "attempt_failed":
                attempt_number = int(event["attempt_number"])
                add_if(reasons, active_attempt != attempt_number, f"attempt {attempt_number} failure has no matching start")
                active_attempt = None
                expected = expected_runtime_profile(row, level)
                add_if(reasons, int(event.get("microbatch", -1)) != expected["per_device_train_batch_size"], f"attempt {attempt_number} failure microbatch mismatch")
                add_if(reasons, int(event.get("grad_accum", -1)) != expected["gradient_accumulation_steps"], f"attempt {attempt_number} failure grad_accum mismatch")
                add_if(reasons, int(event.get("returncode", 0)) == 0, f"attempt {attempt_number} failure has rc0")
                if event.get("failure_class") == "cuda_oom":
                    oom_failures += 1
                    add_if(reasons, int(event.get("returncode", 0)) == 0, f"attempt {attempt_number} labels rc0 as OOM")
                    console = run_dir / f"attempt_{attempt_number:03d}" / "console.log"
                    try:
                        oom_text = read_console_tail(console)
                    except OSError as exc:
                        reasons.append(f"attempt {attempt_number} OOM console unavailable: {exc}")
                        oom_text = ""
                    add_if(reasons, not any(pattern in oom_text for pattern in OOM_PATTERNS), f"attempt {attempt_number} OOM has no console evidence")
                    fallback = next_oom_batch(int(row["effective_batch_records"]), int(expected["per_device_train_batch_size"]))
                    if fallback is not None:
                        add_if(reasons, int(event.get("next_microbatch", -1)) != fallback[0], f"attempt {attempt_number} next microbatch is not approved fallback")
                        add_if(reasons, int(event.get("next_grad_accum", -1)) != fallback[1], f"attempt {attempt_number} next grad_accum is not approved fallback")
                        level += 1
                else:
                    add_if(reasons, "next_microbatch" in event or "next_grad_accum" in event, f"attempt {attempt_number} non-OOM changes batch")
            elif kind == "attempt_interrupted":
                attempt_number = int(event["attempt_number"])
                add_if(
                    reasons,
                    active_attempt != attempt_number,
                    f"attempt {attempt_number} interruption has no matching start",
                )
                active_attempt = None
                expected = expected_runtime_profile(row, level)
                add_if(
                    reasons,
                    event.get("reason") != "worker_restart_recovery",
                    f"attempt {attempt_number} interruption reason mismatch",
                )
                add_if(
                    reasons,
                    event.get("batch_profile_preserved") is not True,
                    f"attempt {attempt_number} interruption changes batch semantics",
                )
                add_if(
                    reasons,
                    int(event.get("microbatch", -1))
                    != expected["per_device_train_batch_size"],
                    f"attempt {attempt_number} interrupted microbatch mismatch",
                )
                add_if(
                    reasons,
                    int(event.get("grad_accum", -1))
                    != expected["gradient_accumulation_steps"],
                    f"attempt {attempt_number} interrupted grad_accum mismatch",
                )
                add_if(
                    reasons,
                    "next_microbatch" in event or "next_grad_accum" in event,
                    f"attempt {attempt_number} interruption applies an OOM fallback",
                )
            elif kind == "attempt_complete":
                attempt_number = int(event["attempt_number"])
                add_if(reasons, active_attempt != attempt_number, f"attempt {attempt_number} completion has no matching start")
                active_attempt = None
                completed_events += 1
                add_if(reasons, attempt_number != completed_number, "history completion attempt differs from marker")
            else:
                reasons.append(f"unknown retry history event: {kind!r}")
        add_if(reasons, active_attempt is not None, "retry history ends with an active attempt")
        disk_numbers = {
            int(match.group(1))
            for path in run_dir.glob("attempt_*")
            if path.is_dir()
            and (match := ATTEMPT_RE.fullmatch(path.name)) is not None
        }
        add_if(
            reasons,
            bool(prejournal_numbers & seen_attempt_numbers),
            "pre-journal orphan overlaps a started attempt",
        )
        add_if(
            reasons,
            disk_numbers != prejournal_numbers | seen_attempt_numbers,
            "attempt directories are not fully accounted by retry ledger",
        )
        add_if(reasons, completed_events != 1, "retry history must contain exactly one completion")
        add_if(reasons, completed_number != last_started_number, "marked completion is not the final started attempt")
        add_if(reasons, int(state.get("total_attempts_started", -1)) != starts, "retry_state attempt count mismatch")
        add_if(reasons, int(state.get("oom_failures", -1)) != oom_failures, "retry_state OOM count mismatch")
        add_if(reasons, int(state.get("fallback_level", -1)) != level, "retry_state fallback level mismatch")
        add_if(reasons, completed_profile != expected_runtime_profile(row, level), "completed runtime profile differs from retry history")
        add_if(reasons, int(state.get("current_microbatch", -1)) != completed_profile["per_device_train_batch_size"], "retry_state current microbatch mismatch")
        add_if(reasons, int(state.get("current_grad_accum", -1)) != completed_profile["gradient_accumulation_steps"], "retry_state current grad_accum mismatch")
    except Exception as exc:
        reasons.append(f"retry ledger validation raised {type(exc).__name__}: {exc}")
    return reasons


def validate_completed_row(
    matrix_path: Path,
    matrix: dict[str, Any],
    row: dict[str, Any],
    *,
    load_checkpoint_tensors: bool,
    manifest_cache: dict[Path, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    run_id = str(row["run_id"])
    run_dir = resolved(row["out_dir"])
    marker_path = run_dir / "complete.json"
    base = {
        "run_id": run_id,
        "phase": str(matrix["metadata"]["phase"]),
        "scientific_cell": list(scientific_cell_key(row)),
        "run_dir": str(run_dir),
    }
    if not marker_path.is_file():
        return {**base, "status": "pending", "reasons": []}
    reasons: list[str] = []
    record_consumption_audit: dict[str, Any] | None = None
    try:
        marker = load_json(marker_path)
        if not isinstance(marker, dict):
            raise ValueError("marker is not an object")
        attempt = resolved(marker["attempt_dir"])
        match = ATTEMPT_RE.fullmatch(attempt.name)
        add_if(reasons, attempt.parent != run_dir or match is None, "marked attempt is not a direct canonical child")
        add_if(reasons, marker.get("status") != "complete", "marker status mismatch")
        add_if(reasons, marker.get("phase") != matrix["metadata"]["phase"], "marker phase mismatch")
        add_if(reasons, marker.get("run_id") != run_id, "marker run_id mismatch")
        add_if(reasons, resolved(marker.get("summary")) != attempt / "summary.json", "marker summary path mismatch")
        contract = load_json(attempt / "attempt_contract.json")
        scaling = load_json(attempt / "scaling_config.json")
        run_config = load_json(attempt / "run_config.json")
        summary = load_json(attempt / "summary.json")
        model_config = load_json(attempt / "final" / "config.json")
        if not all(isinstance(value, dict) for value in (contract, scaling, run_config, summary, model_config)):
            raise ValueError("one or more accepted artifacts are not JSON objects")
        reasons.extend(execution_placement_reasons(
            matrix_path, row, contract, scaling=scaling, marker=marker,
        ))
        add_if(reasons, contract.get("schema_version") != ATTEMPT_CONTRACT_SCHEMA, "attempt contract schema mismatch")
        expected_identity = {
            "matrix_status": row["matrix_status"],
            "phase": matrix["metadata"]["phase"],
            "run_id": run_id,
            "host_id": row["assigned_host"],
            "worker_slot": row["worker_slot"],
            "physical_gpu": int(row["assigned_gpu"]),
            "model_implementation": PACKED_MODEL_IMPLEMENTATION,
            "attention_backend": PACKED_ATTENTION_BACKEND,
        }
        for key, value in expected_identity.items():
            add_if(reasons, contract.get(key) != value, f"attempt contract {key} mismatch")
        add_if(reasons, resolved(contract.get("matrix_json")) != matrix_path, "attempt contract matrix path mismatch")
        add_if(reasons, contract.get("matrix_generation_id") != matrix["metadata"].get("generation_id"), "attempt contract matrix generation mismatch")
        add_if(
            reasons,
            contract.get("matrix_semantic_fingerprint")
            != matrix["metadata"].get("semantic_fingerprint", {}).get("value"),
            "attempt contract matrix fingerprint mismatch",
        )
        add_if(reasons, int(contract.get("attempt_number", -1)) != int(match.group(1)), "attempt contract number mismatch")
        profile = contract.get("runtime_profile")
        expected_profile, profile_reasons = validate_runtime_profile(row, profile)
        reasons.extend(profile_reasons)
        if expected_profile is None:
            expected_profile = {}
        add_if(reasons, contract.get("batch_profile") != profile, "batch_profile differs from runtime_profile")
        add_if(reasons, marker.get("runtime_profile") != profile, "marker runtime_profile mismatch")
        add_if(reasons, marker.get("batch_profile") != profile, "marker batch_profile mismatch")

        manifest_path = resolved(contract.get("domain_split_manifest"))
        if manifest_path not in manifest_cache:
            manifest_cache[manifest_path] = strict_manifest_contract(manifest_path)
        effective_manifest, effective_semantics = manifest_cache[manifest_path]
        add_if(reasons, effective_semantics != matrix["metadata"]["domain_split_semantic_contract"], "attempt manifest semantic contract mismatch")

        add_if(reasons, scaling.get("schema_version") != SCALING_CONFIG_SCHEMA, "scaling config schema mismatch")
        add_if(reasons, scaling.get("matrix_row") != row, "scaling config frozen row mismatch")
        add_if(reasons, scaling.get("runtime_profile") != profile, "scaling config runtime profile mismatch")
        add_if(reasons, scaling.get("command") != contract.get("command"), "scaling/attempt command mismatch")
        add_if(reasons, resolved(scaling.get("matrix_json")) != matrix_path, "scaling config matrix path mismatch")
        add_if(reasons, scaling.get("matrix_generation_id") != matrix["metadata"].get("generation_id"), "scaling config matrix generation mismatch")
        add_if(
            reasons,
            scaling.get("matrix_semantic_fingerprint")
            != matrix["metadata"].get("semantic_fingerprint", {}).get("value"),
            "scaling config matrix fingerprint mismatch",
        )
        add_if(reasons, scaling.get("phase") != matrix["metadata"]["phase"], "scaling config phase mismatch")
        add_if(reasons, scaling.get("host_id") != row["assigned_host"], "scaling config host mismatch")
        add_if(reasons, scaling.get("worker_slot") != row["worker_slot"], "scaling config worker slot mismatch")
        add_if(reasons, int(scaling.get("physical_gpu", -1)) != int(row["assigned_gpu"]), "scaling config physical GPU mismatch")
        add_if(reasons, scaling.get("model_implementation") != PACKED_MODEL_IMPLEMENTATION, "scaling config model implementation mismatch")
        add_if(reasons, scaling.get("attention_backend") != PACKED_ATTENTION_BACKEND, "scaling config attention backend mismatch")
        add_if(reasons, resolved(scaling.get("effective_domain_split_manifest")) != manifest_path, "scaling effective manifest mismatch")
        add_if(
            reasons,
            resolved(scaling.get("frozen_domain_split_manifest"))
            != resolved(matrix["metadata"]["domain_split_manifest"]),
            "scaling frozen manifest mismatch",
        )
        add_if(reasons, int(scaling.get("visible_cuda_device", -1)) != 0, "visible CUDA device is not zero")

        command = contract.get("command")
        if not isinstance(command, list):
            reasons.append("attempt command is not a list")
            command = []
        add_if(reasons, not command_uses_pretrain_entrypoint(command), "attempt command uses another training entrypoint")
        command_expected = {
            "--token-files": str(
                Path(effective_manifest["dataset"]["compact_dir"])
                / "shard_*.bin"
            ),
            "--val-token-files": str(
                Path(effective_manifest["dataset"]["compact_dir"])
                / "shard_*.bin"
            ),
            "--out-dir": str(attempt),
            "--domain-split-manifest": str(manifest_path),
            "--domain-name": str(row["domain"]),
            "--vocab-size": str(row["vocab_size"]),
            "--dataset-mode": "online-pad",
            "--model-implementation": "packed-xformers",
            "--compact-index-dir": str(resolved(effective_manifest["dataset"]["index_dir"])),
            "--online-pad-length": str(row["context_length"]),
            "--token-offset": "3",
            "--val-fraction": "0",
            "--bpe-unfold-policy": "none",
            "--bpe-unfold-prob": "0",
            "--hidden-size": str(row["hidden_size"]),
            "--intermediate-size": str(row["intermediate_size"]),
            "--num-hidden-layers": str(row["num_hidden_layers"]),
            "--num-attention-heads": str(row["num_attention_heads"]),
            "--num-key-value-heads": str(row["num_key_value_heads"]),
            "--head-dim": str(row["head_dim"]),
            "--max-position-embeddings": str(row["context_length"]),
            "--batch-size": str(expected_profile.get("per_device_train_batch_size")),
            "--grad-accum": str(expected_profile.get("gradient_accumulation_steps")),
            "--max-steps": str(row["scheduler_total_steps"]),
            "--target-input-tokens": str(row["target_input_tokens_with_eos"]),
            "--lr": str(FROZEN_TRAINING_CONTROL["learning_rate"]),
            "--min-lr-ratio": str(FROZEN_TRAINING_CONTROL["minimum_learning_rate_ratio"]),
            "--weight-decay": str(FROZEN_TRAINING_CONTROL["weight_decay"]),
            "--adam-beta1": str(FROZEN_TRAINING_CONTROL["adam_beta1"]),
            "--adam-beta2": str(FROZEN_TRAINING_CONTROL["adam_beta2"]),
            "--adam-eps": str(FROZEN_TRAINING_CONTROL["adam_epsilon"]),
            "--max-grad-norm": str(FROZEN_TRAINING_CONTROL["maximum_gradient_norm"]),
            "--warmup-ratio": str(row["warmup_ratio"]),
            "--eval-every": "0",
            "--eval-batches": "0",
            "--final-eval-batches": "0",
            "--save-every": "0",
            "--seed": str(row["seed"]),
            "--optimizer": "adamw-fused",
            "--attn-implementation": "sdpa",
            "--sdpa-kernel": "auto",
        }
        for flag, value in command_expected.items():
            add_if(reasons, command_value(command, flag) != value, f"command {flag} mismatch")
        for flag in ("--online-truncate-long-samples", "--online-shuffle", "--final-eval", "--bf16", "--no-gradient-checkpointing"):
            add_if(reasons, not command_has_once(command, flag), f"command missing unique {flag}")
        for forbidden in (
            "--torch-compile",
            "--deepspeed",
            "--init-checkpoint",
            "--resume-from",
            "--resume-global-step",
            "--gradient-checkpointing",
            "--dry-run",
        ):
            add_if(reasons, any(str(token) == forbidden for token in command), f"command unexpectedly contains {forbidden}")
        add_if(reasons, any(str(token).startswith("--test") for token in command), "training command accesses a test option")

        args = run_config.get("args")
        config = run_config.get("effective_config")
        if not isinstance(args, dict) or not isinstance(config, dict):
            reasons.append("run_config args/effective_config missing")
            args, config = {}, {}
        expected_args = {
            "token_files": [
                str(
                    Path(effective_manifest["dataset"]["compact_dir"])
                    / "shard_*.bin"
                )
            ],
            "val_token_files": [
                str(
                    Path(effective_manifest["dataset"]["compact_dir"])
                    / "shard_*.bin"
                )
            ],
            "vocab_size": int(row["vocab_size"]),
            "dataset_mode": "online-pad",
            "model_implementation": "packed-xformers",
            "domain_split_manifest": str(manifest_path),
            "domain_name": str(row["domain"]),
            "online_pad_length": int(row["context_length"]),
            "online_truncate_long_samples": True,
            "online_shuffle": True,
            "token_offset": 3,
            "val_fraction": 0.0,
            "bpe_unfold_policy": "none",
            "bpe_unfold_prob": 0.0,
            "hidden_size": int(row["hidden_size"]),
            "intermediate_size": int(row["intermediate_size"]),
            "num_hidden_layers": int(row["num_hidden_layers"]),
            "num_attention_heads": int(row["num_attention_heads"]),
            "num_key_value_heads": int(row["num_key_value_heads"]),
            "head_dim": int(row["head_dim"]),
            "max_position_embeddings": int(row["context_length"]),
            "batch_size": int(expected_profile.get("per_device_train_batch_size", -1)),
            "grad_accum": int(expected_profile.get("gradient_accumulation_steps", -1)),
            "target_input_tokens": int(row["target_input_tokens_with_eos"]),
            "max_steps": int(row["scheduler_total_steps"]),
            "seed": int(row["seed"]),
            "warmup_ratio": float(row["warmup_ratio"]),
            "lr": FROZEN_TRAINING_CONTROL["learning_rate"],
            "min_lr_ratio": FROZEN_TRAINING_CONTROL["minimum_learning_rate_ratio"],
            "weight_decay": FROZEN_TRAINING_CONTROL["weight_decay"],
            "adam_beta1": FROZEN_TRAINING_CONTROL["adam_beta1"],
            "adam_beta2": FROZEN_TRAINING_CONTROL["adam_beta2"],
            "adam_eps": FROZEN_TRAINING_CONTROL["adam_epsilon"],
            "max_grad_norm": FROZEN_TRAINING_CONTROL["maximum_gradient_norm"],
            "final_eval": True,
            "final_eval_batches": 0,
            "save_every": 0,
            "optimizer": "adamw-fused",
            "bf16": True,
            "gradient_checkpointing": False,
            "attn_implementation": "sdpa",
            "sdpa_kernel": "auto",
            "torch_compile": False,
            "deepspeed": False,
            "deepspeed_zero_stage": 0,
            "resume_global_step": 0,
            "init_checkpoint": None,
        }
        for key, value in expected_args.items():
            add_if(reasons, args.get(key) != value, f"run_config args.{key} mismatch")
        add_if(
            reasons,
            resolved(args.get("compact_index_dir"))
            != resolved(effective_manifest["dataset"]["index_dir"]),
            "run_config compact index path mismatch",
        )
        domain = str(row["domain"])
        expected_token_files = manifest_token_paths(effective_manifest)
        add_if(reasons, run_config.get("token_files") != expected_token_files, "run_config training token files differ from manifest")
        add_if(reasons, run_config.get("val_token_files") != expected_token_files, "run_config validation token files differ from manifest")
        add_if(reasons, int(run_config.get("num_token_files", -1)) != len(expected_token_files), "run_config training token-file count mismatch")
        add_if(reasons, int(run_config.get("num_val_token_files", -1)) != len(expected_token_files), "run_config validation token-file count mismatch")
        train_view = effective_semantics["views"][domain]["train"]
        validation_view = effective_semantics["views"][domain]["validation"]
        add_if(reasons, int(run_config.get("train_items", -1)) != int(train_view["records"]), "run_config train records mismatch")
        add_if(reasons, int(run_config.get("val_items", -1)) != int(validation_view["records"]), "run_config validation records mismatch")
        add_if(reasons, int(run_config.get("train_tokens_visible", -1)) != int(train_view["effective_input_tokens_with_eos"]), "run_config train tokens mismatch")
        add_if(reasons, int(run_config.get("val_tokens_visible", -1)) != int(validation_view["effective_input_tokens_with_eos"]), "run_config validation tokens mismatch")
        add_if(reasons, int(run_config.get("world_size", -1)) != 1, "run_config world_size mismatch")
        add_if(reasons, run_config.get("training_engine") != "single_process", "run_config training engine mismatch")
        add_if(reasons, run_config.get("train_sampler") != "DistributedAffineShuffleSampler", "run_config train sampler mismatch")
        add_if(reasons, run_config.get("optimizer") != "adamw-fused", "run_config optimizer mismatch")
        add_if(reasons, run_config.get("dataset_mode") != "online-pad", "run_config dataset mode mismatch")
        add_if(reasons, run_config.get("model_implementation") != "packed-xformers", "run_config model implementation mismatch")
        add_if(
            reasons,
            run_config.get("collate_mode")
            != "online_padded_samples_with_optional_bpe_unfold",
            "run_config collate mode mismatch",
        )
        add_if(
            reasons,
            run_config.get("attention_backend") != PACKED_ATTENTION_BACKEND,
            "run_config actual attention backend mismatch",
        )
        # This remains only the Qwen config tag.  It must never be accepted as
        # the actual backend for the custom packed implementation.
        add_if(reasons, run_config.get("attention_implementation") != "sdpa", "run_config Qwen attention config tag mismatch")
        add_if(reasons, any("test" in str(key).lower() for key in run_config), "run_config contains a test-derived top-level field")
        add_if(reasons, any("test" in str(key).lower() for key in args), "run_config args contain a test option")
        view_config = run_config.get("domain_split_view")
        if not isinstance(view_config, dict):
            reasons.append("run_config domain_split_view missing")
        else:
            add_if(reasons, resolved(view_config.get("manifest")) != manifest_path, "run_config view manifest mismatch")
            add_if(reasons, view_config.get("domain") != domain, "run_config view domain mismatch")
            add_if(reasons, view_config.get("train_split") != "train", "run_config train split mismatch")
            add_if(reasons, view_config.get("validation_split") != "validation", "run_config validation split mismatch")
        add_if(reasons, not str(run_config.get("init_source", "")).startswith("random initialization"), "run_config is not random initialization")
        add_if(reasons, int(run_config.get("parameter_count", -1)) != int(row["parameter_count"]), "run_config parameter count mismatch")
        add_if(reasons, int(run_config.get("trainable_parameter_count", -1)) != int(row["parameter_count"]), "run_config trainable parameter count mismatch")

        expected_config = {key: int(row[key]) for key in MODEL_CONFIG_FIELDS}
        expected_config.update({"max_position_embeddings": int(row["context_length"]), "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "tie_word_embeddings": True})
        for key, value in expected_config.items():
            add_if(reasons, config.get(key) != value, f"run_config effective_config.{key} mismatch")
            add_if(reasons, model_config.get(key) != value, f"final model config {key} mismatch")

        target = int(row["target_input_tokens_with_eos"])
        actual = int(summary.get("actual_input_tokens", -1))
        parameters = int(row["parameter_count"])
        add_if(reasons, int(summary.get("target_input_tokens", -1)) != target, "summary target tokens mismatch")
        add_if(reasons, actual < target or actual - target >= int(row["context_length"]), "summary token stop/overshoot mismatch")
        add_if(reasons, int(summary.get("input_token_overshoot", -1)) != actual - target, "summary input-token overshoot mismatch")
        add_if(
            reasons,
            not math.isclose(
                float(summary.get("input_token_relative_error", math.nan)),
                (actual - target) / target,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "summary input-token relative error mismatch",
        )
        add_if(reasons, int(summary.get("parameter_count", -1)) != parameters, "summary parameters mismatch")
        add_if(reasons, summary.get("model_implementation") != PACKED_MODEL_IMPLEMENTATION, "summary model implementation mismatch")
        add_if(reasons, summary.get("attention_backend") != PACKED_ATTENTION_BACKEND, "summary actual attention backend mismatch")
        add_if(reasons, int(summary.get("target_flops_6nd", -1)) != 6 * parameters * target, "summary target FLOPs mismatch")
        add_if(reasons, int(summary.get("actual_flops_6nd", -1)) != 6 * parameters * actual, "summary actual FLOPs mismatch")
        add_if(reasons, int(summary.get("scheduler_total_steps", -1)) != int(row["scheduler_total_steps"]), "summary scheduler mismatch")
        add_if(reasons, int(summary.get("warmup_steps", -1)) != int(row["warmup_steps"]), "summary warmup mismatch")
        add_if(reasons, int(summary.get("train_input_tokens_per_epoch", -1)) != int(train_view["effective_input_tokens_with_eos"]), "summary train epoch tokens mismatch")
        add_if(reasons, summary.get("termination_reason") != "target_input_tokens", "summary termination mismatch")
        # ``scheduler_total_steps`` is a frozen LR horizon, not a hard stop in
        # target-token mode.  Variable record lengths can require extra updates.
        add_if(reasons, int(summary.get("global_step", -1)) <= 0, "summary global step invalid")
        add_if(reasons, not finite_number(summary.get("mean_train_loss")) or float(summary.get("mean_train_loss", 0)) <= 0, "summary mean loss invalid")
        total_samples = int(summary.get("total_train_samples", -1))
        total_loss_tokens = int(summary.get("total_train_loss_tokens", -1))
        add_if(reasons, total_samples <= 0, "summary train sample count invalid")
        add_if(reasons, total_loss_tokens <= 0, "summary train loss-token count invalid")
        add_if(reasons, total_loss_tokens != actual - total_samples, "summary input/loss/sample accounting mismatch")
        expected_record_audit = build_record_consumption_audit(
            total_samples,
            int(train_view["records"]),
        )
        marker_record_audit = marker.get("record_consumption_audit")
        add_if(
            reasons,
            marker_record_audit != expected_record_audit,
            "marker record-consumption audit mismatch",
        )
        if isinstance(marker_record_audit, dict):
            add_if(
                reasons,
                marker_record_audit.get("schema_version")
                != RECORD_CONSUMPTION_AUDIT_SCHEMA,
                "marker record-consumption audit schema mismatch",
            )
            record_consumption_audit = marker_record_audit
        final_micro_steps = int(summary.get("final_update_micro_steps", -1))
        add_if(
            reasons,
            not 1 <= final_micro_steps <= int(expected_profile.get("gradient_accumulation_steps", 0)),
            "summary final accumulation-window size is invalid",
        )
        add_if(reasons, int(summary.get("final_update_loss_tokens", -1)) <= 0, "summary final update loss-token count invalid")
        add_if(reasons, not math.isclose(float(summary.get("actual_epoch_fraction", math.nan)), actual / int(train_view["effective_input_tokens_with_eos"]), rel_tol=1e-12, abs_tol=1e-12), "summary actual epoch fraction mismatch")
        add_if(reasons, resolved(summary.get("final_checkpoint")) != attempt / "final", "summary final checkpoint mismatch")
        for key in ("final_logged_train_loss", "trailing_train_loss"):
            value = summary.get(key)
            add_if(reasons, value is not None and (not finite_number(value) or float(value) <= 0), f"summary {key} invalid")
        validation = summary.get("final_validation")
        if not isinstance(validation, dict):
            reasons.append("summary final_validation missing")
        else:
            add_if(reasons, validation.get("complete_view") is not True, "validation is not complete view")
            add_if(reasons, not finite_number(validation.get("loss")) or float(validation.get("loss", 0)) <= 0, "validation loss invalid")
            add_if(reasons, not finite_number(validation.get("ppl")) or float(validation.get("ppl", 0)) <= 0, "validation ppl invalid")
            add_if(reasons, int(validation.get("samples", -1)) != int(validation_view["records"]), "validation sample count mismatch")
            add_if(reasons, int(validation.get("tokens", -1)) != int(validation_view["supervised_causal_loss_tokens"]), "validation loss-token count mismatch")
            add_if(reasons, int(validation.get("input_tokens", -1)) != int(validation_view["effective_input_tokens_with_eos"]), "validation input-token count mismatch")
            add_if(reasons, int(validation.get("expected_input_tokens", -1)) != int(validation_view["effective_input_tokens_with_eos"]), "validation expected tokens mismatch")
            add_if(reasons, not math.isclose(float(validation.get("input_token_coverage", math.nan)), 1.0, rel_tol=0.0, abs_tol=1e-12), "validation coverage mismatch")
        add_if(reasons, any("test" in str(key).lower() for key in summary), "summary contains a test-derived top-level field")

        reasons.extend(
            validate_retry_ledger(
                run_dir,
                row,
                attempt,
                expected_profile,
                matrix_path,
                manifest_path,
                effective_manifest,
            )
        )
        reasons.extend(validate_checkpoint(attempt / "final" / "pytorch_model.bin", load_checkpoint=load_checkpoint_tensors))
    except Exception as exc:
        reasons.append(f"artifact validation raised {type(exc).__name__}: {exc}")
        attempt = None
        profile = None
    return {
        **base,
        "status": "accepted" if not reasons else "rejected",
        "attempt_dir": str(attempt) if attempt is not None else None,
        "runtime_profile": profile,
        "record_consumption_audit": record_consumption_audit,
        "checkpoint_tensor_validation": bool(load_checkpoint_tensors),
        "reasons": reasons,
    }


def verify_phase_campaign(
    matrix_path: Path,
    proposal_path: Path,
    *,
    load_checkpoint_tensors: bool = True,
) -> dict[str, Any]:
    matrix_path = matrix_path.expanduser().resolve()
    proposal_path = proposal_path.expanduser().resolve()
    matrix_issues: list[str] = []
    try:
        proposal = load_json(proposal_path)
        if not isinstance(proposal, dict) or proposal.get("metadata", {}).get("status") != PROPOSAL_STATUS:
            raise ValueError("proposal is not the frozen design-only proposal")
        matrix, _manifest, rows, matrix_issues = validate_matrix_and_rows(matrix_path, proposal)
    except Exception as exc:
        return {
            "schema_version": "domainwise_phase_campaign_verification_v1",
            "status": "invalid",
            "complete": False,
            "phase": None,
            "matrix_path": str(matrix_path),
            "proposal_path": str(proposal_path),
            "matrix_issues": [f"matrix validation raised {type(exc).__name__}: {exc}"],
            "counts": {"rows": 0, "accepted": 0, "pending": 0, "rejected": 0},
            "runs": [],
        }
    if not load_checkpoint_tensors:
        matrix_issues.append("checkpoint tensor validation is disabled")
    cache: dict[Path, tuple[dict[str, Any], dict[str, Any]]] = {}
    runs: list[dict[str, Any]] = []
    for row in rows:
        try:
            runs.append(
                validate_completed_row(
                    matrix_path,
                    matrix,
                    row,
                    load_checkpoint_tensors=load_checkpoint_tensors,
                    manifest_cache=cache,
                )
            )
        except Exception as exc:
            runs.append(
                {
                    "run_id": str(row.get("run_id", "<missing>"))
                    if isinstance(row, dict)
                    else "<invalid-row>",
                    "phase": matrix["metadata"].get("phase"),
                    "status": "rejected",
                    "checkpoint_tensor_validation": bool(load_checkpoint_tensors),
                    "reasons": [
                        f"artifact validation raised {type(exc).__name__}: {exc}"
                    ],
                }
            )
    counts = {
        status: sum(run["status"] == status for run in runs)
        for status in ("accepted", "pending", "rejected")
    }
    status = (
        "invalid"
        if matrix_issues or counts["rejected"]
        else "incomplete"
        if counts["pending"]
        else "complete"
    )
    return {
        "schema_version": "domainwise_phase_campaign_verification_v1",
        "status": status,
        "complete": status == "complete",
        "phase": matrix["metadata"]["phase"],
        "matrix_path": str(matrix_path),
        "matrix_generation_id": matrix["metadata"].get("generation_id"),
        "matrix_semantic_fingerprint": matrix["metadata"].get(
            "semantic_fingerprint", {}
        ).get("value"),
        "proposal_path": str(proposal_path),
        "checkpoint_tensor_validation": bool(load_checkpoint_tensors),
        "manifest_semantic_contract_validated": not any("manifest" in issue for issue in matrix_issues),
        "matrix_issues": matrix_issues,
        "counts": {"rows": len(runs), **counts},
        "runs": runs,
    }


def validate_b_rows_against_proposal(
    b_matrix_paths: Iterable[Path],
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    for path in b_matrix_paths:
        try:
            matrix = load_json(path)
            if not isinstance(matrix, dict) or not isinstance(matrix.get("rows"), list):
                raise ValueError("matrix root/rows is invalid")
            rows.extend(matrix["rows"])
        except Exception as exc:
            issues.append(
                f"invalid Phase-B matrix {path}: {type(exc).__name__}: {exc}"
            )
    expected_rows = proposal_rows(proposal, "B_bracketing")
    expected = {scientific_cell_key(row): row for row in expected_rows}
    actual = {scientific_cell_key(row): row for row in rows if isinstance(row, dict)}
    add_if(issues, len(rows) != 60 or len(actual) != 60, "Phase-B matrices do not contain 60 unique scientific cells")
    add_if(issues, set(actual) != set(expected), "Phase-B scientific cells differ from proposal")
    for cell, row in actual.items():
        source = expected.get(cell)
        if source is None:
            continue
        add_if(issues, scientific_projection(row, source) != scientific_projection(source), f"Phase-B scientific drift at {cell}")
        try:
            add_if(issues, str(row["run_id"]) != derived_run_id(row), f"Phase-B run_id mismatch at {cell}")
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"Phase-B run identity invalid at {cell}: {exc}")
    return rows, issues


def verify_union_145(
    b_matrix_paths: list[Path],
    phase_results: list[dict[str, Any]],
    proposal_path: Path,
) -> dict[str, Any]:
    proposal = load_json(proposal_path)
    issues: list[str] = []
    add_if(
        issues,
        not isinstance(proposal, dict)
        or proposal.get("metadata", {}).get("status") != PROPOSAL_STATUS,
        "union proposal is not the frozen design-only proposal",
    )
    proposal_design_rows = proposal.get("rows", []) if isinstance(proposal, dict) else []
    if not isinstance(proposal_design_rows, list):
        proposal_design_rows = []
        issues.append("union proposal rows are invalid")
    expected_proposal_counts = {
        "B_bracketing": 60,
        "C_completion": 60,
        "D_peptide_surface": 25,
    }
    for execution_phase, expected_count in expected_proposal_counts.items():
        add_if(
            issues,
            sum(
                isinstance(row, dict)
                and row.get("execution_phase") == execution_phase
                for row in proposal_design_rows
            )
            != expected_count,
            f"proposal does not contain exactly {expected_count} {execution_phase} cells",
        )
    for index, row in enumerate(proposal_design_rows):
        if not isinstance(row, dict):
            issues.append(f"proposal row[{index}] is not an object")
            continue
        try:
            validate_proposal_row(row, proposal)
        except Exception as exc:
            issues.append(
                f"proposal row[{index}] is scientifically invalid: "
                f"{type(exc).__name__}: {exc}"
            )
    b_verification = verify_phase_b_campaign(
        b_matrix_paths,
        expected_total_rows=60,
        load_checkpoints=True,
    )
    b_rows, b_issues = validate_b_rows_against_proposal(b_matrix_paths, proposal)
    issues.extend(b_issues)
    by_phase = {str(result.get("phase")): result for result in phase_results}
    add_if(
        issues,
        len(phase_results) != 2 or len(by_phase) != 2 or set(by_phase) != {"c", "d"},
        "union requires exactly one Phase-C and one Phase-D report",
    )
    add_if(issues, b_verification.get("status") == "invalid", "Phase-B strict verification is invalid")
    add_if(
        issues,
        b_verification.get("status") not in {"complete", "incomplete"},
        "Phase-B strict verification has an unknown status",
    )
    add_if(
        issues,
        bool(b_verification.get("complete"))
        != (b_verification.get("status") == "complete"),
        "Phase-B completion flag contradicts its status",
    )
    add_if(
        issues,
        b_verification.get("checkpoint_load_validation") is not True,
        "Phase-B checkpoint tensors were not validated",
    )
    try:
        b_counts = b_verification.get("counts", {})
        add_if(issues, int(b_counts.get("rows", -1)) != 60, "Phase-B verification does not cover exactly 60 rows")
        add_if(
            issues,
            sum(int(b_counts.get(key, -60)) for key in ("accepted", "pending", "rejected")) != 60,
            "Phase-B verification status counts are inconsistent",
        )
    except (TypeError, ValueError):
        issues.append("Phase-B verification counts are invalid")
    for phase in ("c", "d"):
        result = by_phase.get(phase)
        add_if(issues, result is None or result.get("status") == "invalid", f"Phase-{phase.upper()} strict verification is invalid")
        add_if(
            issues,
            result is not None and result.get("status") not in {"complete", "incomplete"},
            f"Phase-{phase.upper()} strict verification has an unknown status",
        )
        add_if(
            issues,
            result is not None
            and bool(result.get("complete")) != (result.get("status") == "complete"),
            f"Phase-{phase.upper()} completion flag contradicts its status",
        )
        add_if(
            issues,
            result is not None
            and int(result.get("counts", {}).get("rows", -1))
            != int(phase_spec(phase)["expected_rows"]),
            f"Phase-{phase.upper()} verification does not cover its exact row count",
        )
        if result is not None:
            counts = result.get("counts", {})
            expected_count = int(phase_spec(phase)["expected_rows"])
            try:
                counted = sum(
                    int(counts.get(key, -expected_count))
                    for key in ("accepted", "pending", "rejected")
                )
            except (TypeError, ValueError):
                counted = -1
            add_if(
                issues,
                counted != expected_count,
                f"Phase-{phase.upper()} verification status counts are inconsistent",
            )
        add_if(
            issues,
            result is not None and result.get("checkpoint_tensor_validation") is not True,
            f"Phase-{phase.upper()} checkpoint tensors were not validated",
        )
    rows = list(b_rows)
    for phase in ("c", "d"):
        result = by_phase.get(phase)
        if result is None:
            continue
        try:
            matrix, _manifest, phase_rows, phase_issues = validate_matrix_and_rows(
                Path(str(result["matrix_path"])).expanduser().resolve(), proposal
            )
            issues.extend(
                f"Phase-{phase.upper()} matrix: {issue}" for issue in phase_issues
            )
            add_if(
                issues,
                matrix.get("metadata", {}).get("phase") != phase,
                f"Phase-{phase.upper()} report points at a different phase matrix",
            )
            add_if(
                issues,
                result.get("matrix_generation_id")
                != matrix.get("metadata", {}).get("generation_id"),
                f"Phase-{phase.upper()} report refers to another matrix generation",
            )
            add_if(
                issues,
                result.get("matrix_semantic_fingerprint")
                != matrix.get("metadata", {}).get("semantic_fingerprint", {}).get("value"),
                f"Phase-{phase.upper()} report refers to another matrix fingerprint",
            )
            rows.extend(phase_rows)
        except Exception as exc:
            issues.append(
                f"Phase-{phase.upper()} matrix revalidation failed: "
                f"{type(exc).__name__}: {exc}"
            )
    actual_cells = [scientific_cell_key(row) for row in rows if isinstance(row, dict)]
    expected_cells = [scientific_cell_key(row) for row in proposal.get("rows", []) if isinstance(row, dict)]
    add_if(issues, len(rows) != 145, f"union has {len(rows)} rows instead of 145")
    add_if(issues, len(set(actual_cells)) != 145, "union scientific cells are duplicated")
    add_if(issues, set(actual_cells) != set(expected_cells), "union scientific cell set differs from proposal145")
    add_if(issues, len(expected_cells) != 145 or len(set(expected_cells)) != 145, "proposal itself is not 145 unique scientific cells")
    phase_semantics: list[Any] = []
    for phase in ("c", "d"):
        result = by_phase.get(phase)
        if result is None:
            continue
        try:
            phase_matrix = load_json(Path(str(result["matrix_path"])))
            phase_semantics.append(phase_matrix["metadata"]["domain_split_semantic_contract"])
        except Exception as exc:
            issues.append(f"Phase-{phase.upper()} semantic contract unavailable: {type(exc).__name__}: {exc}")
    add_if(issues, len(phase_semantics) == 2 and phase_semantics[0] != phase_semantics[1], "Phase-C and Phase-D manifest semantic contracts differ")
    b_semantics: list[Any] = []
    for path in b_matrix_paths:
        try:
            b_matrix = load_json(path)
            b_manifest_path = resolved(b_matrix["metadata"]["domain_split_manifest"])
            _manifest, semantics = strict_manifest_contract(b_manifest_path)
            b_semantics.append(semantics)
        except Exception as exc:
            issues.append(f"Phase-B manifest semantics unavailable for {path}: {type(exc).__name__}: {exc}")
    add_if(
        issues,
        bool(b_semantics)
        and (
            any(item != b_semantics[0] for item in b_semantics[1:])
            or any(item != b_semantics[0] for item in phase_semantics)
        ),
        "B/C/D manifest semantic contracts differ",
    )
    run_ids: list[str] = []
    out_dirs: list[str] = []
    for row in rows:
        try:
            run_ids.append(str(row["run_id"]))
            out_dirs.append(str(resolved(row["out_dir"])))
            add_if(issues, str(row["run_id"]) != derived_run_id(row), f"union run_id is not derived: {row.get('run_id')}")
        except Exception as exc:
            issues.append(f"union row identity invalid: {type(exc).__name__}: {exc}")
    add_if(issues, len(run_ids) != 145 or len(set(run_ids)) != 145, "union run_ids are not 145 unique values")
    add_if(issues, len(out_dirs) != 145 or len(set(out_dirs)) != 145, "union output directories are not 145 unique values")
    proposal_c = {
        scientific_cell_key(row): row
        for row in proposal.get("rows", [])
        if isinstance(row, dict) and row.get("execution_phase") == "C_completion"
    }
    actual_c = {
        scientific_cell_key(row): row
        for row in rows
        if isinstance(row, dict) and row.get("execution_phase") == "C_completion"
    }
    reviewed_c_replacements = [
        {
            "scientific_cell": list(cell),
            "proposal_model_id": proposal_c[cell].get("model_id"),
            "actual_model_id": actual_c[cell].get("model_id"),
        }
        for cell in sorted(set(proposal_c) & set(actual_c), key=str)
        if proposal_c[cell].get("model_id") != actual_c[cell].get("model_id")
    ]
    accepted = int(b_verification.get("counts", {}).get("accepted", 0)) + sum(
        int(result.get("counts", {}).get("accepted", 0)) for result in by_phase.values()
    )
    pending = int(b_verification.get("counts", {}).get("pending", 0)) + sum(
        int(result.get("counts", {}).get("pending", 0)) for result in by_phase.values()
    )
    rejected = int(b_verification.get("counts", {}).get("rejected", 0)) + sum(
        int(result.get("counts", {}).get("rejected", 0)) for result in by_phase.values()
    )
    if issues or rejected:
        status = "invalid"
    elif accepted == 145 and pending == 0:
        status = "complete"
    else:
        status = "incomplete"
    return {
        "schema_version": "domainwise_union_145_verification_v1",
        "status": status,
        "complete": status == "complete",
        "proposal_path": str(proposal_path),
        "counts": {"rows": len(rows), "accepted": accepted, "pending": pending, "rejected": rejected},
        "design_contract": {
            "phase_b_cells": len(b_rows),
            "phase_c_cells": len(actual_c),
            "phase_d_cells": sum(
                isinstance(row, dict)
                and row.get("execution_phase") == "D_peptide_surface"
                for row in rows
            ),
            "unique_scientific_cells": len(set(actual_cells)),
            "unique_run_ids": len(set(run_ids)),
            "reviewed_phase_c_model_replacements": reviewed_c_replacements,
        },
        "issues": issues,
        "phase_b": b_verification,
        "phase_c": by_phase.get("c"),
        "phase_d": by_phase.get("d"),
    }


def main() -> int:
    script = Path(__file__).resolve()
    workspace = script.parents[5]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-matrix", type=Path, action="append", required=True)
    parser.add_argument("--b-matrix", type=Path, action="append", default=None)
    parser.add_argument(
        "--proposal-json",
        type=Path,
        default=workspace / "scaling_data/reports/domainwise_scaling_matrix_proposal_v1.json",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--union145", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    proposal_path = args.proposal_json.expanduser().resolve()
    phase_results = [
        verify_phase_campaign(path, proposal_path, load_checkpoint_tensors=True)
        for path in args.phase_matrix
    ]
    if args.union145:
        if not args.b_matrix:
            raise ValueError("--union145 requires both Phase-B --b-matrix paths")
        result = verify_union_145(
            [path.expanduser().resolve() for path in args.b_matrix],
            phase_results,
            proposal_path,
        )
    else:
        status = "invalid" if any(item["status"] == "invalid" for item in phase_results) else "incomplete" if any(not item["complete"] for item in phase_results) else "complete"
        result = {
            "schema_version": "domainwise_followup_verification_bundle_v1",
            "status": status,
            "complete": status == "complete",
            "phases": phase_results,
        }
    if args.output_json is not None:
        atomic_write_json(args.output_json.expanduser().resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "complete": result["complete"],
                "phases": [
                    {"phase": item.get("phase"), "status": item["status"], "counts": item["counts"]}
                    for item in phase_results
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 1 if args.require_complete and not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
