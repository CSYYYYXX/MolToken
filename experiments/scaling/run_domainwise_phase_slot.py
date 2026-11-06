#!/usr/bin/env python3
"""Run one Phase-C/D worker slot from a single global frozen matrix.

The retry ledger is stored under each run directory, so restarting either the
worker or its supervisor cannot reset the total-attempt cap.  CUDA OOM failures
halve the record microbatch (to the next exact divisor) and increase gradient
accumulation to preserve the frozen effective record batch.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_STATUSES = {
    "c": "runnable_phase_c",
    "d": "runnable_phase_d",
}
SUPPORTED_EXECUTION_PHASES = {
    "c": "C_completion",
    "d": "D_peptide_surface",
}
RETRY_STATE_SCHEMA = "domainwise_phase_retry_state_v1"
ATTEMPT_CONTRACT_SCHEMA = "domainwise_phase_attempt_contract_v1"
PREJOURNAL_ORPHAN_SCHEMA = "domainwise_prejournal_attempt_orphan_v1"
RETRYABLE_EXIT = 75
ATTEMPTS_EXHAUSTED_EXIT = 76
RECORD_CONSUMPTION_AUDIT_SCHEMA = "domainwise_record_consumption_audit_v1"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
FROZEN_TRAINING_CONTROL = {
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
    "actual_attention_backend": "xformers_memory_efficient_block_diagonal_causal",
    "qwen_config_attention_tag": "sdpa_unused_by_packed_backend",
    "sdpa_kernel_setting": "auto_unused_by_packed_backend",
    "gradient_checkpointing": False,
    "torch_compile": False,
    "deepspeed": False,
    "random_initialization": True,
    "save_every_steps": 0,
    "full_validation_at_end": True,
}

OOM_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)

_DISTRIBUTED_LAUNCH_ENV_KEYS = {
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
}


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_constant,
    )


def safe_component(value: str, label: str) -> str:
    if SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def single_process_environment(source: Any) -> dict[str, str]:
    """Return an environment that cannot inherit an enclosing DDP launch.

    Each formal scaling row is intentionally one process on one physical GPU.
    A controller started from ``torchrun`` must therefore not leak rendezvous or
    rank variables into the trainer and silently turn a row into a DDP job.
    """

    environment = {str(key): str(value) for key, value in source.items()}
    for key in _DISTRIBUTED_LAUNCH_ENV_KEYS:
        environment.pop(key, None)
    for key in tuple(environment):
        if key.startswith("TORCHELASTIC_"):
            environment.pop(key, None)
    return environment


def build_record_consumption_audit(
    total_occurrences: int,
    dataset_records: int,
) -> dict[str, Any]:
    """Derive exact record coverage for the frozen single-process sampler.

    ``DistributedAffineShuffleSampler`` is a bijection over the complete
    dataset in every epoch and the formal loader keeps its final short batch.
    Training can enter a new epoch only after consuming the preceding one in
    full.  Therefore the actual occurrence count is sufficient to derive
    exact unique, repeated, and omitted record counts without a large bitmap.
    """

    occurrences = int(total_occurrences)
    records = int(dataset_records)
    if occurrences <= 0 or records <= 0:
        raise ValueError("record-consumption counts must be positive")
    full_epochs, partial_records = divmod(occurrences, records)
    unique_records = min(occurrences, records)
    return {
        "schema_version": RECORD_CONSUMPTION_AUDIT_SCHEMA,
        "status": "verified",
        "verification_method": "actual_occurrences_plus_affine_epoch_bijection_v1",
        "sampler": "DistributedAffineShuffleSampler",
        "world_size": 1,
        "drop_last": False,
        "dataset_records": records,
        "actual_record_occurrences": occurrences,
        "full_epochs_completed": full_epochs,
        "partial_epoch_records": partial_records,
        "unique_records": unique_records,
        "repeated_records": occurrences - unique_records,
        "omitted_records": records - unique_records,
    }


def summary_record_consumption_audit(
    summary: dict[str, Any],
    row: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return build_record_consumption_audit(
        int(summary["total_train_samples"]),
        int(manifest["views"][str(row["domain"])]["train"]["records"]),
    )


def matrix_semantic_fingerprint(matrix: dict[str, Any]) -> str:
    semantic = copy.deepcopy(matrix)
    metadata = semantic.get("metadata", {})
    for key in ("created_at_utc", "generation_id", "semantic_fingerprint"):
        metadata.pop(key, None)
    encoded = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_fingerprint(value: Any) -> str:
    """Fingerprint a frozen external JSON artifact without trusting its path."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def append_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"time": now(), **value}, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        handle.flush()


def matrix_phase(matrix: dict[str, Any]) -> tuple[str, str]:
    metadata = matrix.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("matrix has no metadata object")
    phase = str(metadata.get("phase", ""))
    if phase not in SUPPORTED_STATUSES:
        raise ValueError(f"matrix phase must be c or d, got {phase!r}")
    status = SUPPORTED_STATUSES[phase]
    if metadata.get("status") != status or metadata.get("is_runnable") is not True:
        raise ValueError(f"matrix is not a formal {status} matrix")
    expected_execution = SUPPORTED_EXECUTION_PHASES[phase]
    if metadata.get("execution_phase") != expected_execution:
        raise ValueError(
            f"matrix execution_phase must be {expected_execution!r}, got "
            f"{metadata.get('execution_phase')!r}"
        )
    rows = matrix.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("formal matrix must contain a non-empty rows list")
    if int(metadata.get("rows", -1)) != len(rows):
        raise ValueError("matrix metadata row count does not match rows")
    if metadata.get("training_control") != FROZEN_TRAINING_CONTROL:
        raise ValueError("matrix training_control differs from the frozen optimizer contract")
    expected_rows = 60 if phase == "c" else 25
    if len(rows) != expected_rows:
        raise ValueError(f"formal Phase-{phase.upper()} must contain {expected_rows} rows")
    evidence_gate = metadata.get("batch_profile_evidence_gate")
    if (
        not isinstance(evidence_gate, dict)
        or evidence_gate.get("required") is not True
        or evidence_gate.get("satisfied") is not True
        or not evidence_gate.get("report_json")
        or not evidence_gate.get("report_semantic_fingerprint")
    ):
        raise ValueError("formal matrix batch-profile evidence gate is not satisfied")
    evidence_path = Path(str(evidence_gate["report_json"])).expanduser().resolve()
    evidence = load_json(evidence_path)
    if (
        not isinstance(evidence, dict)
        or canonical_json_fingerprint(evidence)
        != evidence_gate["report_semantic_fingerprint"]
        or evidence.get("schema_version") != evidence_gate.get("schema_version")
        or evidence.get("status") != "approved_for_formal_freeze"
        or evidence.get("phase") != phase
        or evidence.get("profile_contract_id")
        != evidence_gate.get("profile_contract_id")
    ):
        raise ValueError("batch-profile evidence content differs from the frozen matrix")
    review_gate = metadata.get("review_gate")
    if phase == "c" and (
        not isinstance(review_gate, dict)
        or review_gate.get("required") is not True
        or review_gate.get("satisfied") is not True
        or not review_gate.get("review_json")
    ):
        raise ValueError("formal Phase-C validation review gate is not satisfied")
    if phase == "c":
        review_path = Path(str(review_gate["review_json"])).expanduser().resolve()
        review = load_json(review_path)
        if (
            not isinstance(review, dict)
            or canonical_json_fingerprint(review)
            != review_gate.get("review_semantic_fingerprint")
            or review.get("schema_version") != "domainwise_phase_b_validation_review_v1"
            or review.get("status") != "approved_for_phase_c_freeze"
            or review.get("test_data_accessed") is not False
        ):
            raise ValueError("Phase-C review content differs from the frozen matrix")
    fingerprint = metadata.get("semantic_fingerprint")
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("algorithm")
        != "sha256_canonical_json_excluding_timestamps_and_identity"
        or fingerprint.get("value") != matrix_semantic_fingerprint(matrix)
        or metadata.get("generation_id")
        != f"{metadata.get('phase_name')}_{str(fingerprint.get('value'))[:20]}"
    ):
        raise ValueError("matrix semantic identity is missing or inconsistent")
    return phase, status


def phase_root(matrix: dict[str, Any]) -> Path:
    metadata = matrix["metadata"]
    expected = (
        Path(metadata["run_root"])
        / str(metadata["experiment_name"])
        / str(metadata["phase_name"])
    ).expanduser().resolve()
    recorded = Path(metadata["output_base"]).expanduser().resolve()
    if recorded != expected:
        raise ValueError(f"matrix output_base does not match run_root/experiment/phase: {recorded}")
    return expected


def manifest_semantic_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the path-independent manifest projection frozen in the matrix."""
    if (
        manifest.get("schema_version") != "domain_split_raw_indices_v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("domain split manifest is not complete")
    dataset = manifest.get("dataset")
    views = manifest.get("views")
    domains = ("small_molecule", "protein", "dna", "rna", "peptide")
    if not isinstance(dataset, dict) or not isinstance(views, dict) or set(views) != set(domains):
        raise ValueError("manifest must contain the frozen five-domain dataset")
    contract_views: dict[str, Any] = {}
    seen_domain_ids: set[int] = set()
    for domain in domains:
        domain_views = views[domain]
        if not isinstance(domain_views, dict) or set(domain_views) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"manifest {domain} must have exactly train/validation/test")
        contract_views[domain] = {}
        domain_ids: set[int] = set()
        for split in ("train", "validation", "test"):
            view = domain_views[split]
            if not isinstance(view, dict) or not isinstance(view.get("shards"), list):
                raise ValueError(f"invalid manifest view {domain}/{split}")
            domain_id = int(view["domain_id"])
            records = int(view["records"])
            input_tokens = int(view["effective_input_tokens_with_eos"])
            if records <= 0 or input_tokens < records:
                raise ValueError(f"invalid manifest totals for {domain}/{split}")
            domain_ids.add(domain_id)
            shards: list[dict[str, Any]] = []
            seen_shard_ids: set[int] = set()
            for shard in view["shards"]:
                shard_id = int(shard["shard_id"])
                shard_records = int(shard["records"])
                shard_tokens = int(shard["effective_input_tokens_with_eos"])
                if shard_id in seen_shard_ids:
                    raise ValueError(f"duplicate shard id in {domain}/{split}: {shard_id}")
                seen_shard_ids.add(shard_id)
                if (
                    str(shard["domain"]) != domain
                    or str(shard["split"]) != split
                    or int(shard["domain_id"]) != domain_id
                    or shard_records < 0
                    or shard_tokens < shard_records
                ):
                    raise ValueError(f"invalid shard identity/totals for {domain}/{split}/{shard_id}")
                shards.append(
                    {
                        "domain": domain,
                        "domain_id": domain_id,
                        "split": split,
                        "shard_id": shard_id,
                        "token_file_name": Path(str(shard["token_file"])).name,
                        "raw_indices_file_name": Path(
                            str(shard["raw_indices_file"])
                        ).name,
                        "records": shard_records,
                        "effective_input_tokens_with_eos": shard_tokens,
                        "supervised_causal_loss_tokens": shard_tokens - shard_records,
                    }
                )
            shards.sort(key=lambda item: item["shard_id"])
            if sum(item["records"] for item in shards) != records:
                raise ValueError(f"manifest shard record total mismatch for {domain}/{split}")
            if sum(item["effective_input_tokens_with_eos"] for item in shards) != input_tokens:
                raise ValueError(f"manifest shard token total mismatch for {domain}/{split}")
            contract_views[domain][split] = {
                "domain_id": domain_id,
                "records": records,
                "effective_input_tokens_with_eos": input_tokens,
                "supervised_causal_loss_tokens": input_tokens - records,
                "shards": shards,
            }
        if len(domain_ids) != 1:
            raise ValueError(f"domain id changes across splits for {domain}")
        domain_id = next(iter(domain_ids))
        if domain_id in seen_domain_ids:
            raise ValueError(f"duplicate domain id: {domain_id}")
        seen_domain_ids.add(domain_id)
    return {
        "schema_version": str(manifest["schema_version"]),
        "status": str(manifest["status"]),
        "dataset": {
            "context_length": int(dataset["context_length"]),
            "num_shards": int(dataset["num_shards"]),
        },
        "split_assignment": manifest.get("split_assignment"),
        "views": contract_views,
    }


def validate_manifest_for_matrix(
    matrix: dict[str, Any],
    manifest_path: Path,
    *,
    check_files: bool,
) -> dict[str, Any]:
    """Validate a host-local manifest against the matrix's frozen semantics."""
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("domain split manifest root must be an object")
    actual_contract = manifest_semantic_contract(manifest)
    expected_contract = matrix.get("metadata", {}).get("domain_split_semantic_contract")
    if not isinstance(expected_contract, dict):
        raise ValueError("matrix has no frozen domain_split_semantic_contract")
    if actual_contract != expected_contract:
        raise ValueError("host-local manifest differs from the matrix-frozen semantics")

    frozen_manifest = Path(
        str(matrix["metadata"]["domain_split_manifest"])
    ).expanduser().resolve()
    if str(manifest_path).startswith("/dev/shm/"):
        ready_path = manifest_path.parent.parent / "READY.json"
        ready = load_json(ready_path)
        if not isinstance(ready, dict) or ready.get("status") != "ready":
            raise ValueError(f"tmpfs stage is not ready: {ready_path}")
        if Path(str(ready.get("staged_manifest", ""))).expanduser().resolve() != manifest_path:
            raise ValueError("tmpfs READY staged_manifest does not identify the selected manifest")
        source = ready.get("source_manifest", ready.get("staged_from_manifest"))
        if source is None or Path(str(source)).expanduser().resolve() != frozen_manifest:
            raise ValueError("tmpfs READY source manifest differs from the matrix-frozen manifest")

    if check_files:
        dataset = manifest["dataset"]
        compact_dir = Path(str(dataset["compact_dir"])).expanduser().resolve()
        index_dir = Path(str(dataset["index_dir"])).expanduser().resolve()
        if not compact_dir.is_dir() or not index_dir.is_dir():
            raise FileNotFoundError(
                f"manifest dataset paths are unavailable: {compact_dir} / {index_dir}"
            )
        for domain_views in manifest["views"].values():
            for view in domain_views.values():
                for shard in view["shards"]:
                    token_file = Path(str(shard["token_file"])).expanduser().resolve()
                    indices_file = Path(str(shard["raw_indices_file"])).expanduser().resolve()
                    if not token_file.is_file() or token_file.stat().st_size <= 0:
                        raise FileNotFoundError(f"manifest token file unavailable: {token_file}")
                    if not indices_file.is_file():
                        raise FileNotFoundError(f"manifest raw-index file unavailable: {indices_file}")
                    expected_bytes = int(shard["records"]) * 4
                    if indices_file.stat().st_size != expected_bytes:
                        raise ValueError(
                            f"raw-index byte count mismatch: {indices_file} "
                            f"!= {expected_bytes}"
                        )
    return manifest


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def validate_runtime_profile(row: dict[str, Any], profile: dict[str, Any]) -> tuple[int, int]:
    """Return micro/GA only for the matrix profile or its exact OOM sequence."""
    effective = int(row["effective_batch_records"])
    expected_micro = int(row["per_device_train_batch_size"])
    expected_grad_accum = int(row["gradient_accumulation_steps"])
    level = int(profile["level"])
    if level < 0:
        raise ValueError("negative OOM fallback level")
    for _ in range(level):
        fallback = next_oom_batch(effective, expected_micro)
        if fallback is None:
            raise ValueError("OOM fallback level exceeds the exact-divisor sequence")
        expected_micro, expected_grad_accum = fallback
    source = str(profile["source"])
    if source != ("matrix" if level == 0 else "oom_fallback"):
        raise ValueError("runtime profile source/level mismatch")
    micro = int(profile["per_device_train_batch_size"])
    grad_accum = int(profile["gradient_accumulation_steps"])
    if (micro, grad_accum) != (expected_micro, expected_grad_accum):
        raise ValueError("runtime profile is not an approved OOM fallback")
    if int(profile["effective_batch_records"]) != effective:
        raise ValueError("runtime profile changes the frozen effective batch")
    return micro, grad_accum


def _attempt_contract(attempt_dir: Path) -> dict[str, Any] | None:
    path = attempt_dir / "attempt_contract.json"
    if not path.is_file():
        return None
    try:
        value = load_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != ATTEMPT_CONTRACT_SCHEMA:
        return None
    return value


def load_complete_summary(
    attempt_dir: Path,
    run_dir: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    matrix_path: Path,
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Return a summary only when the attempt satisfies the frozen contract."""
    try:
        attempt = attempt_dir.expanduser().resolve()
        run = run_dir.expanduser().resolve()
        matrix_resolved = matrix_path.expanduser().resolve()
        manifest_resolved = manifest_path.expanduser().resolve()
        frozen_matrix = load_json(matrix_resolved)
        matrix_generation = frozen_matrix["metadata"]["generation_id"]
        matrix_fingerprint = frozen_matrix["metadata"]["semantic_fingerprint"]["value"]
        # An accepted attempt is an immediate child, never a nested path.
        if not attempt.is_dir() or attempt.parent != run or not re.fullmatch(
            r"attempt_[0-9]{3,}", attempt.name
        ):
            return None
        contract = _attempt_contract(attempt)
        if contract is None:
            return None
        expected_identity = {
            "matrix_status": row["matrix_status"],
            "phase": str(row["execution_phase"])[0].lower(),
            "run_id": row["run_id"],
            "host_id": row["assigned_host"],
            "worker_slot": row["worker_slot"],
            "physical_gpu": int(row["assigned_gpu"]),
            "model_implementation": "packed-xformers",
            "attention_backend": "xformers_memory_efficient_block_diagonal_causal",
        }
        if any(contract.get(key) != value for key, value in expected_identity.items()):
            return None
        if Path(str(contract.get("matrix_json", ""))).expanduser().resolve() != matrix_resolved:
            return None
        if (
            contract.get("matrix_generation_id") != matrix_generation
            or contract.get("matrix_semantic_fingerprint") != matrix_fingerprint
        ):
            return None
        if (
            Path(str(contract.get("domain_split_manifest", ""))).expanduser().resolve()
            != manifest_resolved
        ):
            return None
        batch = contract.get("runtime_profile", contract.get("batch_profile"))
        if not isinstance(batch, dict):
            return None
        micro, grad_accum = validate_runtime_profile(row, batch)
        effective = int(row["effective_batch_records"])
        if micro * grad_accum != effective:
            return None

        summary = load_json(attempt / "summary.json")
        run_config = load_json(attempt / "run_config.json")
        scaling_config = load_json(attempt / "scaling_config.json")
        if not all(isinstance(value, dict) for value in (summary, run_config, scaling_config)):
            return None
        if scaling_config.get("schema_version") != "domainwise_phase_scaling_config_v1":
            return None
        canonical_plan = matrix_resolved.parent / "local_execution_plan.json"
        placement_required = (
            row.get("assigned_host") == "remote"
            and frozen_matrix["metadata"].get("phase") == "c"
            and (canonical_plan.exists() or canonical_plan.is_symlink())
        )
        if placement_required or "execution_placement" in contract or "execution_placement" in scaling_config:
            from domainwise_local_execution import execution_placement_for_row

            placement = contract.get("execution_placement")
            if not isinstance(placement, dict):
                return None
            placement_path = canonical_plan if placement_required else Path(placement["plan_json"])
            expected_placement = execution_placement_for_row(matrix_resolved, placement_path, row)
            if placement != expected_placement or scaling_config.get("execution_placement") != expected_placement:
                return None
        if scaling_config.get("matrix_row") != row:
            return None
        if scaling_config.get("runtime_profile") != batch:
            return None
        if scaling_config.get("command") != contract.get("command"):
            return None
        if Path(str(scaling_config.get("matrix_json", ""))).expanduser().resolve() != matrix_resolved:
            return None
        if (
            scaling_config.get("matrix_generation_id") != matrix_generation
            or scaling_config.get("matrix_semantic_fingerprint") != matrix_fingerprint
        ):
            return None
        if (
            Path(str(scaling_config.get("effective_domain_split_manifest", "")))
            .expanduser()
            .resolve()
            != manifest_resolved
        ):
            return None

        args = run_config.get("args")
        config = run_config.get("effective_config")
        if not isinstance(args, dict) or not isinstance(config, dict):
            return None
        expected_args = {
            "vocab_size": int(row["vocab_size"]),
            "dataset_mode": "online-pad",
            "model_implementation": "packed-xformers",
            "domain_split_manifest": str(manifest_resolved),
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
            "batch_size": micro,
            "grad_accum": grad_accum,
            "target_input_tokens": int(row["target_input_tokens_with_eos"]),
            "max_steps": int(row["scheduler_total_steps"]),
            "seed": int(row["seed"]),
            "warmup_ratio": float(row["warmup_ratio"]),
            "lr": FROZEN_TRAINING_CONTROL["learning_rate"],
            "min_lr_ratio": FROZEN_TRAINING_CONTROL[
                "minimum_learning_rate_ratio"
            ],
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
        if any(args.get(key) != value for key, value in expected_args.items()):
            return None
        if Path(str(args.get("compact_index_dir", ""))).expanduser().resolve() != Path(
            str(manifest["dataset"]["index_dir"])
        ).expanduser().resolve():
            return None
        domain = str(row["domain"])
        train_view = manifest["views"][domain]["train"]
        validation_view = manifest["views"][domain]["validation"]
        train_records = int(train_view["records"])
        train_input = int(train_view["effective_input_tokens_with_eos"])
        validation_records = int(validation_view["records"])
        validation_input = int(validation_view["effective_input_tokens_with_eos"])
        if (
            int(run_config.get("train_items", -1)) != train_records
            or int(run_config.get("val_items", -1)) != validation_records
            or int(run_config.get("train_tokens_visible", -1)) != train_input
            or int(run_config.get("val_tokens_visible", -1)) != validation_input
            or int(run_config.get("world_size", -1)) != 1
            or run_config.get("training_engine") != "single_process"
            or run_config.get("train_sampler") != "DistributedAffineShuffleSampler"
            or run_config.get("optimizer") != "adamw-fused"
            or run_config.get("dataset_mode") != "online-pad"
            or run_config.get("model_implementation") != "packed-xformers"
            or run_config.get("collate_mode")
            != "online_padded_samples_with_optional_bpe_unfold"
            or run_config.get("attention_implementation") != "sdpa"
            or run_config.get("attention_backend")
            != "xformers_memory_efficient_block_diagonal_causal"
        ):
            return None
        view_config = run_config.get("domain_split_view")
        if not isinstance(view_config, dict) or (
            Path(str(view_config.get("manifest", ""))).expanduser().resolve()
            != manifest_resolved
            or view_config.get("domain") != domain
            or view_config.get("train_split") != "train"
            or view_config.get("validation_split") != "validation"
        ):
            return None
        init_source = str(run_config.get("init_source", ""))
        if not init_source.startswith("random initialization") or "packed-xformers" not in init_source:
            return None
        if (
            int(run_config.get("parameter_count", -1)) != int(row["parameter_count"])
            or int(run_config.get("trainable_parameter_count", -1))
            != int(row["parameter_count"])
        ):
            return None
        expected_config = {
            "vocab_size": int(row["vocab_size"]),
            "max_position_embeddings": int(row["context_length"]),
            "hidden_size": int(row["hidden_size"]),
            "intermediate_size": int(row["intermediate_size"]),
            "num_hidden_layers": int(row["num_hidden_layers"]),
            "num_attention_heads": int(row["num_attention_heads"]),
            "num_key_value_heads": int(row["num_key_value_heads"]),
            "head_dim": int(row["head_dim"]),
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "tie_word_embeddings": True,
        }
        if any(config.get(key) != value for key, value in expected_config.items()):
            return None

        target = int(row["target_input_tokens_with_eos"])
        parameters = int(row["parameter_count"])
        actual = int(summary["actual_input_tokens"])
        if int(summary["target_input_tokens"]) != target or not 0 <= actual - target < int(
            row["context_length"]
        ):
            return None
        if (
            int(summary["parameter_count"]) != parameters
            or summary.get("model_implementation") != "packed-xformers"
            or summary.get("attention_backend")
            != "xformers_memory_efficient_block_diagonal_causal"
            or int(summary["target_flops_6nd"]) != 6 * parameters * target
            or int(summary["actual_flops_6nd"]) != 6 * parameters * actual
            or int(summary["scheduler_total_steps"]) != int(row["scheduler_total_steps"])
            or int(summary["warmup_steps"]) != int(row["warmup_steps"])
            or int(summary["train_input_tokens_per_epoch"]) != train_input
            or summary.get("termination_reason") != "target_input_tokens"
        ):
            return None
        global_step = int(summary["global_step"])
        # In target-token mode max_steps is the frozen scheduler horizon, not a
        # hard training stop.  Shuffled variable-length records can require a
        # few more optimizer updates than the mean-token estimate.
        if global_step <= 0:
            return None
        if not math.isclose(
            float(summary["actual_epoch_fraction"]),
            actual / train_input,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return None
        if not _finite_positive(summary.get("mean_train_loss")):
            return None
        total_samples = int(summary.get("total_train_samples", -1))
        total_loss_tokens = int(summary.get("total_train_loss_tokens", -1))
        if (
            total_samples <= 0
            or total_loss_tokens <= 0
            or total_loss_tokens != actual - total_samples
        ):
            return None
        for key in ("final_logged_train_loss", "trailing_train_loss"):
            value = summary.get(key)
            if value is not None and not _finite_positive(value):
                return None

        validation = summary.get("final_validation")
        if not isinstance(validation, dict) or validation.get("complete_view") is not True:
            return None
        if not _finite_positive(validation.get("loss")) or not _finite_positive(
            validation.get("ppl")
        ):
            return None
        if (
            int(validation["samples"]) != validation_records
            or int(validation["tokens"]) != int(row["validation_loss_tokens"])
            or int(validation["input_tokens"]) != validation_input
            or int(validation["expected_input_tokens"]) != validation_input
            or not math.isclose(
                float(validation["input_token_coverage"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return None

        final_dir = attempt / "final"
        if Path(str(summary["final_checkpoint"])).expanduser().resolve() != final_dir:
            return None
        final_config = load_json(final_dir / "config.json")
        if not isinstance(final_config, dict) or any(
            final_config.get(key) != value for key, value in expected_config.items()
        ):
            return None
        checkpoint = final_dir / "pytorch_model.bin"
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
            return None
        return summary
    except (KeyError, OSError, TypeError, ValueError, OverflowError):
        return None


def completed_attempt(
    run_dir: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    matrix_path: Path,
    manifest_path: Path,
) -> tuple[Path, dict[str, Any]] | None:
    def repair_retry_ledger(attempt: Path) -> bool:
        state_path = run / "retry_state.json"
        contract = _attempt_contract(attempt)
        if not state_path.is_file() or contract is None:
            return False
        try:
            state = load_retry_state(run, row)
            attempt_number = int(contract["attempt_number"])
            runtime_profile = contract.get(
                "runtime_profile", contract.get("batch_profile")
            )
            starts = [
                item
                for item in state["history"]
                if isinstance(item, dict)
                and item.get("event") == "attempt_start"
                and int(item.get("attempt_number", -1)) == attempt_number
            ]
            if len(starts) != 1 or int(state["total_attempts_started"]) < attempt_number:
                return False
            if (
                int(starts[0].get("microbatch", -1))
                != int(runtime_profile["per_device_train_batch_size"])
                or int(starts[0].get("grad_accum", -1))
                != int(runtime_profile["gradient_accumulation_steps"])
            ):
                return False
            completion_events = [
                item
                for item in state["history"]
                if isinstance(item, dict)
                and item.get("event") == "attempt_complete"
                and int(item.get("attempt_number", -1)) == attempt_number
            ]
            if not completion_events:
                state["history"].append(
                    {
                        "event": "attempt_complete",
                        "time": now(),
                        "attempt_number": attempt_number,
                        "recovered_from_valid_artifacts": True,
                    }
                )
            state["status"] = "complete"
            state["completed_attempt"] = attempt_number
            atomic_json(state_path, state)
            return True
        except (KeyError, OSError, TypeError, ValueError):
            return False

    run = run_dir.expanduser().resolve()
    marker = run / "complete.json"
    if marker.is_file():
        try:
            marked = load_json(marker)
            attempt = Path(str(marked["attempt_dir"])).expanduser().resolve()
            contract = _attempt_contract(attempt)
            if (
                marked.get("status") == "complete"
                and marked.get("phase") == str(row["execution_phase"])[0].lower()
                and marked.get("run_id") == row.get("run_id")
                and attempt.parent == run
                and Path(str(marked["summary"])).expanduser().resolve()
                == attempt / "summary.json"
                and contract is not None
                and marked.get("runtime_profile")
                == contract.get("runtime_profile", contract.get("batch_profile"))
            ):
                summary = load_complete_summary(
                    attempt, run, row, manifest, matrix_path, manifest_path
                )
                if (
                    summary is not None
                    and marked.get("record_consumption_audit")
                    == summary_record_consumption_audit(summary, row, manifest)
                    and repair_retry_ledger(attempt)
                ):
                    return attempt, summary
        except (KeyError, OSError, ValueError):
            pass
    for attempt in sorted(run.glob("attempt_*")):
        summary = load_complete_summary(
            attempt, run, row, manifest, matrix_path, manifest_path
        )
        if summary is not None:
            contract = _attempt_contract(attempt)
            assert contract is not None
            if not repair_retry_ledger(attempt):
                continue
            runtime_profile = contract.get(
                "runtime_profile", contract.get("batch_profile")
            )
            # Recover the narrow crash window after trainer completion but
            # before the worker wrote its strict marker.
            atomic_json(
                marker,
                {
                    "status": "complete",
                    "phase": str(row["execution_phase"])[0].lower(),
                    "run_id": row["run_id"],
                    "attempt_dir": str(attempt),
                    "summary": str(attempt / "summary.json"),
                    "batch_profile": runtime_profile,
                    "runtime_profile": runtime_profile,
                    "record_consumption_audit": summary_record_consumption_audit(
                        summary, row, manifest
                    ),
                    "completed_at": now(),
                    "recovered_from_valid_orphan": True,
                },
            )
            return attempt, summary
    return None


def existing_attempt_numbers(run_dir: Path) -> list[int]:
    numbers: list[int] = []
    for path in run_dir.glob("attempt_*"):
        if not path.is_dir():
            continue
        try:
            numbers.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return sorted(numbers)


def next_attempt_dir(run_dir: Path, total_attempts_started: int = 0) -> Path:
    next_number = max(
        max(existing_attempt_numbers(run_dir), default=0),
        int(total_attempts_started),
    ) + 1
    return run_dir / f"attempt_{next_number:03d}"


def load_retry_state(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "retry_state.json"
    if path.is_file():
        state = load_json(path)
        if not isinstance(state, dict):
            raise ValueError(f"retry state is not an object for {row['run_id']}")
        if state.get("schema_version") != RETRY_STATE_SCHEMA or state.get("run_id") != row["run_id"]:
            raise ValueError(f"retry state does not belong to run {row['run_id']}")
    else:
        state = {
            "schema_version": RETRY_STATE_SCHEMA,
            "run_id": row["run_id"],
            "status": "pending",
            # A directory is created before the durable attempt_start record.
            # It is not a launched trainer and must not consume retry budget.
            "total_attempts_started": 0,
            "oom_failures": 0,
            "fallback_level": 0,
            "current_microbatch": int(row["per_device_train_batch_size"]),
            "current_grad_accum": int(row["gradient_accumulation_steps"]),
            "effective_batch_records": int(row["effective_batch_records"]),
            "history": [],
            "prejournal_orphans": [],
        }
    micro = int(state["current_microbatch"])
    grad_accum = int(state["current_grad_accum"])
    effective = int(state["effective_batch_records"])
    if effective != int(row["effective_batch_records"]) or micro * grad_accum != effective:
        raise ValueError(f"retry state changes the frozen effective batch for {row['run_id']}")
    if micro > int(row["per_device_train_batch_size"]):
        raise ValueError(f"retry state increases the frozen microbatch for {row['run_id']}")
    level = int(state.get("fallback_level", 0))
    validate_runtime_profile(
        row,
        {
            "source": "matrix" if level == 0 else "oom_fallback",
            "level": level,
            "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_records": effective,
        },
    )
    if int(state.get("oom_failures", 0)) < level:
        raise ValueError(f"retry state fallback level exceeds OOM failures for {row['run_id']}")
    if not isinstance(state.get("history"), list):
        raise ValueError(f"retry state history is not a list for {row['run_id']}")
    if int(state.get("total_attempts_started", -1)) < 0:
        raise ValueError(f"negative retry count for {row['run_id']}")
    return state


def reconcile_retry_state_after_restart(
    run_dir: Path,
    row: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Close a crashed trainer attempt and audit pre-journal directories.

    The run lock proves that the prior worker no longer owns the row. A
    durable ``attempt_start`` without a terminal event is therefore recorded
    as interrupted, preserving the exact batch profile and consuming exactly
    one attempt (but never applying an OOM fallback). Conversely, an
    ``attempt_*`` directory absent from the journal was created before the
    trainer launch point. It is retained and audited, but is not counted as a
    started attempt.
    """

    run = run_dir.expanduser().resolve()
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError(f"retry state history is not a list for {row['run_id']}")
    active: dict[str, Any] | None = None
    started_numbers: set[int] = set()
    for index, event in enumerate(history):
        if not isinstance(event, dict):
            raise ValueError(f"retry history[{index}] is not an object")
        kind = event.get("event")
        if kind == "attempt_start":
            number = int(event.get("attempt_number", -1))
            if number <= 0 or number in started_numbers or active is not None:
                raise ValueError("retry history has overlapping/duplicate attempt starts")
            started_numbers.add(number)
            active = event
        elif kind in {"attempt_failed", "attempt_complete", "attempt_interrupted"}:
            number = int(event.get("attempt_number", -1))
            if active is None or int(active.get("attempt_number", -1)) != number:
                raise ValueError("retry history terminal event has no active attempt")
            active = None
        else:
            raise ValueError(f"unknown retry history event: {kind!r}")

    existing_numbers = set(existing_attempt_numbers(run))
    if not started_numbers.issubset(existing_numbers):
        raise ValueError("retry history refers to a missing attempt directory")
    orphan_numbers = sorted(existing_numbers - started_numbers)
    prejournal = state.setdefault("prejournal_orphans", [])
    if not isinstance(prejournal, list):
        raise ValueError("prejournal_orphans is not a list")
    recorded: dict[int, dict[str, Any]] = {}
    for item in prejournal:
        if not isinstance(item, dict) or item.get("schema_version") != PREJOURNAL_ORPHAN_SCHEMA:
            raise ValueError("invalid pre-journal orphan record")
        number = int(item.get("attempt_number", -1))
        if number <= 0 or number in recorded:
            raise ValueError("duplicate/invalid pre-journal orphan number")
        recorded[number] = item
    if set(recorded) - set(orphan_numbers):
        raise ValueError("recorded pre-journal orphan now overlaps history or is missing")

    changed = False
    allowed_files = {
        "attempt_contract.json",
        "attempt_contract.json.tmp",
        "scaling_config.json",
        "scaling_config.json.tmp",
    }
    for number in orphan_numbers:
        attempt = run / f"attempt_{number:03d}"
        entries = sorted(
            str(path.relative_to(attempt))
            for path in attempt.rglob("*")
            if path.is_file()
        )
        unexpected = [name for name in entries if name not in allowed_files]
        unexpected_dirs = [
            str(path.relative_to(attempt))
            for path in attempt.rglob("*")
            if path.is_dir()
        ]
        if unexpected or unexpected_dirs:
            raise ValueError(
                "pre-journal attempt contains trainer/runtime artifacts: "
                f"attempt={number} files={unexpected} dirs={unexpected_dirs}"
            )
        if number in recorded:
            item = recorded[number]
            if (
                Path(str(item.get("attempt_dir"))).expanduser().resolve()
                != attempt.resolve()
                or item.get("reason") != "worker_crash_before_attempt_journal"
                or item.get("artifacts_present") != entries
            ):
                raise ValueError("pre-journal orphan record no longer matches disk")
        else:
            prejournal.append(
                {
                    "schema_version": PREJOURNAL_ORPHAN_SCHEMA,
                    "attempt_number": number,
                    "attempt_dir": str(attempt.resolve()),
                    "reason": "worker_crash_before_attempt_journal",
                    "artifacts_present": entries,
                    "observed_at": now(),
                }
            )
            changed = True

    started_count = len(started_numbers)
    recorded_total = int(state.get("total_attempts_started", -1))
    if recorded_total != started_count:
        # Older code derived this count from directory cardinality. Repair
        # only that precisely recognizable pre-journal crash signature.
        if recorded_total != len(existing_numbers) or recorded_total < started_count:
            raise ValueError("retry attempt count differs from durable starts")
        state.setdefault("recovery_events", []).append(
            {
                "event": "prejournal_count_reconciled",
                "time": now(),
                "previous_total_attempts_started": recorded_total,
                "durable_attempt_starts": started_count,
            }
        )
        state["total_attempts_started"] = started_count
        changed = True

    if active is not None:
        number = int(active["attempt_number"])
        state["history"].append(
            {
                "event": "attempt_interrupted",
                "time": now(),
                "attempt_number": number,
                "reason": "worker_restart_recovery",
                "microbatch": int(active["microbatch"]),
                "grad_accum": int(active["grad_accum"]),
                "batch_profile_preserved": True,
            }
        )
        state["status"] = "failed_retryable"
        changed = True
    elif any(
        isinstance(event, dict) and event.get("event") == "attempt_complete"
        for event in history
    ):
        # completed_attempt() is called before reconciliation. Reaching this
        # branch means the ledger claims success but the strict artifacts did
        # not validate, so another launch would conceal corruption.
        raise ValueError("retry ledger claims completion without valid artifacts")
    elif state.get("status") == "running":
        raise ValueError("retry state says running but history has no active attempt")

    if changed:
        atomic_json(run / "retry_state.json", state)
    return state


def next_oom_batch(effective_batch_records: int, current_microbatch: int) -> tuple[int, int] | None:
    """Choose the largest exact divisor no larger than half the current microbatch."""
    effective = int(effective_batch_records)
    current = int(current_microbatch)
    if effective <= 0 or current <= 0 or effective % current != 0:
        raise ValueError("current microbatch must be a positive divisor of effective batch")
    upper = current // 2
    for candidate in range(upper, 0, -1):
        if effective % candidate == 0:
            return candidate, effective // candidate
    return None


def classify_failure(returncode: int, console_path: Path) -> str:
    try:
        with console_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 2 * 1024 * 1024))
            tail = handle.read().decode("utf-8", errors="replace").lower()
    except OSError:
        tail = ""
    if returncode != 0 and any(pattern in tail for pattern in OOM_PATTERNS):
        return "cuda_oom"
    if returncode < 0:
        return "signal"
    if returncode == 0:
        return "invalid_completion_artifact"
    return "non_oom_error"


def training_command(
    python: Path,
    pipeline_root: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    attempt_dir: Path,
    microbatch: int,
    grad_accum: int,
    num_workers: int,
    manifest_path: Path,
) -> list[str]:
    dataset = manifest["dataset"]
    token_glob = str(Path(dataset["compact_dir"]) / "shard_*.bin")
    return [
        str(python),
        str(pipeline_root / "cli" / "pretrain" / "train_qwen3_pretrain.py"),
        "--token-files",
        token_glob,
        "--val-token-files",
        token_glob,
        "--out-dir",
        str(attempt_dir),
        "--vocab-size",
        str(row["vocab_size"]),
        "--dataset-mode",
        "online-pad",
        "--model-implementation",
        "packed-xformers",
        "--compact-index-dir",
        str(dataset["index_dir"]),
        "--domain-split-manifest",
        str(manifest_path),
        "--domain-name",
        str(row["domain"]),
        "--online-pad-length",
        str(row["context_length"]),
        "--online-truncate-long-samples",
        "--online-shuffle",
        "--token-offset",
        "3",
        "--val-fraction",
        "0",
        "--bpe-unfold-policy",
        "none",
        "--bpe-unfold-prob",
        "0",
        "--hidden-size",
        str(row["hidden_size"]),
        "--intermediate-size",
        str(row["intermediate_size"]),
        "--num-hidden-layers",
        str(row["num_hidden_layers"]),
        "--num-attention-heads",
        str(row["num_attention_heads"]),
        "--num-key-value-heads",
        str(row["num_key_value_heads"]),
        "--head-dim",
        str(row["head_dim"]),
        "--max-position-embeddings",
        str(row["context_length"]),
        "--batch-size",
        str(microbatch),
        "--grad-accum",
        str(grad_accum),
        "--max-steps",
        str(row["scheduler_total_steps"]),
        "--target-input-tokens",
        str(row["target_input_tokens_with_eos"]),
        "--lr",
        str(FROZEN_TRAINING_CONTROL["learning_rate"]),
        "--min-lr-ratio",
        str(FROZEN_TRAINING_CONTROL["minimum_learning_rate_ratio"]),
        "--weight-decay",
        str(FROZEN_TRAINING_CONTROL["weight_decay"]),
        "--adam-beta1",
        str(FROZEN_TRAINING_CONTROL["adam_beta1"]),
        "--adam-beta2",
        str(FROZEN_TRAINING_CONTROL["adam_beta2"]),
        "--adam-eps",
        str(FROZEN_TRAINING_CONTROL["adam_epsilon"]),
        "--max-grad-norm",
        str(FROZEN_TRAINING_CONTROL["maximum_gradient_norm"]),
        "--warmup-ratio",
        str(row["warmup_ratio"]),
        "--log-every",
        "20",
        "--eval-every",
        "0",
        "--eval-batches",
        "0",
        "--final-eval",
        "--final-eval-batches",
        "0",
        "--save-every",
        "0",
        "--num-workers",
        str(num_workers),
        "--seed",
        str(row["seed"]),
        "--optimizer",
        "adamw-fused",
        "--attn-implementation",
        "sdpa",
        "--sdpa-kernel",
        "auto",
        "--bf16",
        "--no-gradient-checkpointing",
    ]


def main() -> int:
    script_path = Path(__file__).resolve()
    pipeline_root_default = script_path.parents[2]
    workspace_default = pipeline_root_default.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--worker-slot", required=True)
    parser.add_argument("--local-execution-plan", type=Path, default=None)
    parser.add_argument("--domain-split-manifest", type=Path, default=None)
    parser.add_argument("--pipeline-root", type=Path, default=pipeline_root_default)
    parser.add_argument(
        "--python",
        type=Path,
        default=workspace_default
        / "local_runtime"
        / "envs"
        / "holomol-py311-cu121"
        / "bin"
        / "python",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-total-attempts-per-run", type=int, default=4)
    parser.add_argument("--max-attempts-this-invocation", type=int, default=2)
    args = parser.parse_args()
    if args.max_total_attempts_per_run <= 0 or args.max_attempts_this_invocation <= 0:
        raise ValueError("attempt limits must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    safe_component(args.host_id, "--host-id")
    safe_component(args.worker_slot, "--worker-slot")

    matrix_path = args.matrix_json.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    python = args.python.expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError("matrix root must be an object")
    phase, required_status = matrix_phase(matrix)
    root = phase_root(matrix)
    all_rows = matrix["rows"]
    run_ids = [str(row.get("run_id", "")) for row in all_rows]
    out_dirs = [str(Path(str(row.get("out_dir", ""))).expanduser().resolve()) for row in all_rows]
    if any(not run_id for run_id in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("global matrix run_id values must be non-empty and unique")
    if len(out_dirs) != len(set(out_dirs)):
        raise ValueError("global matrix out_dir values must be unique")
    rows = [
        row
        for row in all_rows
        if str(row.get("assigned_host")) == args.host_id
        and str(row.get("worker_slot")) == args.worker_slot
    ]
    if not rows:
        raise ValueError(
            f"global matrix has no rows for host={args.host_id} slot={args.worker_slot}"
        )
    if not args.worker_slot.startswith(f"{args.host_id}_"):
        raise ValueError("worker slot is not qualified by --host-id")
    gpu_ids = {int(row["assigned_gpu"]) for row in rows}
    if len(gpu_ids) != 1:
        raise ValueError(f"worker slot spans physical GPUs: {gpu_ids}")
    gpu_id = next(iter(gpu_ids))
    execution_plan = None
    if args.local_execution_plan is not None:
        from domainwise_local_execution import load_local_execution_plan

        execution_plan = load_local_execution_plan(matrix_path, args.local_execution_plan)
        if phase != "c" or args.host_id != "remote":
            raise ValueError("local execution placement only applies to logical remote Phase C")
    for row in rows:
        if row.get("matrix_status") != required_status:
            raise ValueError(f"row is not {required_status}: {row.get('run_id')}")
        if row.get("execution_phase") != SUPPORTED_EXECUTION_PHASES[phase]:
            raise ValueError(f"row execution phase mismatch: {row.get('run_id')}")
        if (
            int(row.get("world_size", 0)) != 1
            or row.get("bpe_unfold_policy") != "none"
            or row.get("model_implementation") != "packed-xformers"
            or row.get("attention_backend")
            != "xformers_memory_efficient_block_diagonal_causal"
        ):
            raise ValueError(f"row violates single-process/no-unfold freeze: {row['run_id']}")
        run_dir = Path(row["out_dir"]).expanduser().resolve()
        if not run_dir.is_relative_to((root / "runs").resolve()):
            raise ValueError(f"run out_dir escapes phase root: {run_dir}")

    manifest_path = (
        args.domain_split_manifest
        if args.domain_split_manifest is not None
        else Path(matrix["metadata"]["domain_split_manifest"])
    ).expanduser().resolve()
    manifest = validate_manifest_for_matrix(matrix, manifest_path, check_files=True)

    worker_root = root / "workers" / args.host_id / args.worker_slot
    status_path = worker_root / "status.jsonl"
    lock_root = Path(f"/dev/shm/holomol-{os.getuid()}/scaling")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"domainwise_phase_{phase}_{args.host_id}_{args.worker_slot}.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"worker slot is already active: {args.worker_slot}") from exc
    lock_handle.write(f"pid={os.getpid()} time={now()} matrix={matrix_path}\n")
    lock_handle.flush()
    append_status(
        status_path,
        {
            "event": "worker_start",
            "phase": phase,
            "host_id": args.host_id,
            "worker_slot": args.worker_slot,
            "gpu": gpu_id,
            "rows": len(rows),
            "matrix": str(matrix_path),
        },
    )
    environment = single_process_environment(os.environ)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTHONPATH": str(pipeline_root)
            + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "2"),
            "MKL_NUM_THREADS": environment.get("MKL_NUM_THREADS", "2"),
            "PYTORCH_CUDA_ALLOC_CONF": environment.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
        }
    )

    exhausted_runs: list[str] = []
    retryable_runs: list[str] = []
    run_lock_handles: list[Any] = []
    for row in rows:
        run_dir = Path(row["out_dir"]).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        run_lock = (run_dir / ".run.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            run_lock.close()
            raise RuntimeError(f"run is already owned by another worker: {row['run_id']}") from exc
        run_lock.seek(0)
        run_lock.truncate()
        run_lock.write(
            f"pid={os.getpid()} phase={phase} host={args.host_id} "
            f"slot={args.worker_slot} time={now()}\n"
        )
        run_lock.flush()
        run_lock_handles.append(run_lock)
        done = completed_attempt(run_dir, row, manifest, matrix_path, manifest_path)
        if done is not None:
            append_status(
                status_path,
                {
                    "event": "skip_complete",
                    "run_id": row["run_id"],
                    "attempt_dir": str(done[0]),
                },
            )
            continue
        state = load_retry_state(run_dir, row)
        state = reconcile_retry_state_after_restart(run_dir, row, state)
        attempts_this_invocation = 0
        success = False
        while (
            int(state["total_attempts_started"]) < args.max_total_attempts_per_run
            and attempts_this_invocation < args.max_attempts_this_invocation
        ):
            attempt_dir = next_attempt_dir(
                run_dir, int(state["total_attempts_started"])
            )
            attempt_dir.mkdir(parents=True, exist_ok=False)
            micro = int(state["current_microbatch"])
            grad_accum = int(state["current_grad_accum"])
            attempt_number = int(attempt_dir.name.removeprefix("attempt_"))
            command = training_command(
                python,
                pipeline_root,
                manifest,
                row,
                attempt_dir,
                micro,
                grad_accum,
                args.num_workers,
                manifest_path,
            )
            fallback_level = int(state.get("fallback_level", 0))
            batch_contract = {
                "source": "matrix" if fallback_level == 0 else "oom_fallback",
                "level": fallback_level,
                "batch_profile_id": str(row["batch_profile_id"])
                + ("" if micro == int(row["per_device_train_batch_size"]) else f"_oom_fallback_{micro}x{grad_accum}"),
                "per_device_train_batch_size": micro,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_records": int(row["effective_batch_records"]),
            }
            contract = {
                "schema_version": ATTEMPT_CONTRACT_SCHEMA,
                "matrix_json": str(matrix_path),
                "matrix_status": required_status,
                "matrix_generation_id": matrix["metadata"]["generation_id"],
                "matrix_semantic_fingerprint": matrix["metadata"][
                    "semantic_fingerprint"
                ]["value"],
                "phase": phase,
                "run_id": row["run_id"],
                "host_id": args.host_id,
                "worker_slot": args.worker_slot,
                "physical_gpu": gpu_id,
                "model_implementation": row["model_implementation"],
                "attention_backend": row["attention_backend"],
                "attempt_number": attempt_number,
                "attempt_started_at": now(),
                "domain_split_manifest": str(manifest_path),
                "batch_profile": batch_contract,
                "runtime_profile": batch_contract,
                "command": command,
            }
            placement_fields: dict[str, Any] = {}
            if execution_plan is not None:
                from domainwise_local_execution import execution_placement_for_row

                placement_fields["execution_placement"] = execution_placement_for_row(
                    matrix_path, args.local_execution_plan, row
                )
                contract.update(placement_fields)
            atomic_json(attempt_dir / "attempt_contract.json", contract)
            atomic_json(
                attempt_dir / "scaling_config.json",
                {
                    "schema_version": "domainwise_phase_scaling_config_v1",
                    "matrix_json": str(matrix_path),
                    "matrix_generation_id": matrix["metadata"]["generation_id"],
                    "matrix_semantic_fingerprint": matrix["metadata"][
                        "semantic_fingerprint"
                    ]["value"],
                    "matrix_row": row,
                    "phase": phase,
                    "host_id": args.host_id,
                    "worker_slot": args.worker_slot,
                    "physical_gpu": gpu_id,
                    "model_implementation": row["model_implementation"],
                    "attention_backend": row["attention_backend"],
                    "visible_cuda_device": 0,
                    "frozen_domain_split_manifest": str(
                        Path(matrix["metadata"]["domain_split_manifest"])
                        .expanduser()
                        .resolve()
                    ),
                    "effective_domain_split_manifest": str(manifest_path),
                    "runtime_profile": batch_contract,
                    "attempt_started_at": contract["attempt_started_at"],
                    "command": command,
                    **placement_fields,
                },
            )
            state["total_attempts_started"] = int(state["total_attempts_started"]) + 1
            state["status"] = "running"
            state["history"].append(
                {
                    "event": "attempt_start",
                    "time": now(),
                    "attempt_number": attempt_number,
                    "microbatch": micro,
                    "grad_accum": grad_accum,
                }
            )
            atomic_json(run_dir / "retry_state.json", state)
            attempts_this_invocation += 1
            append_status(
                status_path,
                {
                    "event": "run_start",
                    "run_id": row["run_id"],
                    "attempt_dir": str(attempt_dir),
                    "attempt_number": attempt_number,
                    "total_attempts_started": state["total_attempts_started"],
                    "microbatch": micro,
                    "grad_accum": grad_accum,
                    "effective_batch_records": state["effective_batch_records"],
                },
            )
            console_path = attempt_dir / "console.log"
            with console_path.open("ab", buffering=0) as console:
                result = subprocess.run(
                    command,
                    cwd=pipeline_root,
                    env=environment,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            summary = (
                load_complete_summary(
                    attempt_dir,
                    run_dir,
                    row,
                    manifest,
                    matrix_path,
                    manifest_path,
                )
                if result.returncode == 0
                else None
            )
            if summary is not None:
                marker = {
                    "status": "complete",
                    "phase": phase,
                    "run_id": row["run_id"],
                    "attempt_dir": str(attempt_dir),
                    "summary": str(attempt_dir / "summary.json"),
                    "batch_profile": batch_contract,
                    "runtime_profile": batch_contract,
                    "record_consumption_audit": summary_record_consumption_audit(
                        summary, row, manifest
                    ),
                    "completed_at": now(),
                }
                atomic_json(run_dir / "complete.json", marker)
                state["status"] = "complete"
                state["completed_attempt"] = attempt_number
                state["history"].append(
                    {
                        "event": "attempt_complete",
                        "time": now(),
                        "attempt_number": attempt_number,
                    }
                )
                atomic_json(run_dir / "retry_state.json", state)
                append_status(
                    status_path,
                    {
                        "event": "run_complete",
                        "run_id": row["run_id"],
                        "attempt_dir": str(attempt_dir),
                        "global_step": summary["global_step"],
                        "actual_input_tokens": summary["actual_input_tokens"],
                        "final_validation_loss": summary["final_validation"]["loss"],
                        "total_attempts_started": state["total_attempts_started"],
                    },
                )
                success = True
                break

            failure_class = classify_failure(result.returncode, console_path)
            history = {
                "event": "attempt_failed",
                "time": now(),
                "attempt_number": attempt_number,
                "returncode": result.returncode,
                "failure_class": failure_class,
                "microbatch": micro,
                "grad_accum": grad_accum,
            }
            if failure_class == "cuda_oom":
                state["oom_failures"] = int(state.get("oom_failures", 0)) + 1
                fallback = next_oom_batch(int(state["effective_batch_records"]), micro)
                if fallback is not None:
                    state["current_microbatch"], state["current_grad_accum"] = fallback
                    state["fallback_level"] = int(state.get("fallback_level", 0)) + 1
                    history["next_microbatch"] = fallback[0]
                    history["next_grad_accum"] = fallback[1]
            state["status"] = "failed_retryable"
            state["history"].append(history)
            atomic_json(run_dir / "retry_state.json", state)
            append_status(
                status_path,
                {
                    "event": "run_failed",
                    "run_id": row["run_id"],
                    "attempt_dir": str(attempt_dir),
                    "attempt_number": attempt_number,
                    "returncode": result.returncode,
                    "failure_class": failure_class,
                    "total_attempts_started": state["total_attempts_started"],
                    "next_microbatch": state["current_microbatch"],
                    "next_grad_accum": state["current_grad_accum"],
                },
            )

        if success:
            continue
        if int(state["total_attempts_started"]) >= args.max_total_attempts_per_run:
            state["status"] = "attempts_exhausted"
            atomic_json(run_dir / "retry_state.json", state)
            exhausted_runs.append(str(row["run_id"]))
            append_status(
                status_path,
                {
                    "event": "run_exhausted",
                    "run_id": row["run_id"],
                    "total_attempts_started": state["total_attempts_started"],
                },
            )
        else:
            retryable_runs.append(str(row["run_id"]))

    append_status(
        status_path,
        {
            "event": "worker_done",
            "phase": phase,
            "worker_slot": args.worker_slot,
            "gpu": gpu_id,
            "exhausted_runs": exhausted_runs,
            "retryable_runs": retryable_runs,
        },
    )
    if retryable_runs:
        return RETRYABLE_EXIT
    if exhausted_runs:
        return ATTEMPTS_EXHAUSTED_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
