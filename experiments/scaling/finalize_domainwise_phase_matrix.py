#!/usr/bin/env python3
"""Freeze reviewed domain-wise Phase C or Phase D rows for later launch.

This intentionally lives beside, rather than replacing, the live Phase-B
finalizer.  A Phase-C formal freeze is fail-closed until a validation-only
Phase-B review explicitly approves all twelve completion curves.  Passing
``--design-dry-run`` produces a non-runnable preview without that review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_RUN_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_RUN_ROOT


PROPOSAL_STATUS = "design_only_not_runnable"
MANIFEST_SCHEMA = "domain_split_raw_indices_v1"
REVIEW_SCHEMA = "domainwise_phase_b_validation_review_v1"
PROFILE_EVIDENCE_SCHEMA = "domainwise_batch_profile_evidence_v1"
HARDWARE_LAYOUT_SCHEMA = "domainwise_hardware_layout_v1"
MATRIX_SCHEMA = "domainwise_phase_matrix_v1"
DEFAULT_EXPERIMENT_NAME = "domainwise_scaling_fddfs_mf80000_v1_followup"

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

PHASES: dict[str, dict[str, Any]] = {
    "c": {
        "phase_name": "phase_c",
        "execution_phase": "C_completion",
        "expected_rows": 60,
        "formal_status": "runnable_phase_c",
        "review_required": True,
    },
    "d": {
        "phase_name": "phase_d",
        "execution_phase": "D_peptide_surface",
        "expected_rows": 25,
        "formal_status": "runnable_phase_d",
        "review_required": False,
    },
}

STANDARD_DOMAINS = ("small_molecule", "protein", "dna", "rna")
EXPECTED_B_BUDGETS = ("C0", "C2", "C5")
EXPECTED_C_BUDGETS = ("C1", "C3", "C4")
EXPECTED_D_MODELS = ("0p006b", "0p009b", "0p014b", "0p022b", "0p041b")
EXPECTED_D_EPOCHS = (0.25, 0.5, 1.0, 2.0, 4.0)
TARGET_MULTIPLIERS = (0.65, 0.82, 1.0, 1.25, 1.6)
C_PHASE_B_NEIGHBORS = {
    "C1": {"C0", "C2"},
    "C3": {"C2", "C5"},
    "C4": {"C2", "C5"},
}

# These are scientific effective-record batches, held fixed within a domain
# regardless of model or hardware.  Peptide has 20.407 input tokens/record in
# the frozen train view, so 6,144 records give about 125,376 input tokens/update.
EFFECTIVE_BATCH_RECORDS = {
    "small_molecule": 12_288,
    "protein": 2_560,
    "dna": 2_560,
    "rna": 6_144,
    "peptide": 6_144,
}

# Profiles are conservative values already exercised for the standard-domain
# model family.  The peptide profiles deliberately mirror the short-sequence
# RNA record batch, but remain separately named so a peptide smoke report can
# be attached before launch.
SINGLE_24GB_BY_BAND: dict[str, dict[str, tuple[int, int]]] = {
    "le_6m": {
        "small_molecule": (6_144, 2),
        "protein": (1_280, 2),
        "dna": (1_280, 2),
        "rna": (3_072, 2),
        "peptide": (3_072, 2),
    },
    "7m_to_31m": {
        "small_molecule": (3_072, 4),
        "protein": (640, 4),
        "dna": (640, 4),
        "rna": (1_536, 4),
        "peptide": (1_536, 4),
    },
    "32m_to_41m": {
        "small_molecule": (2_048, 6),
        "protein": (512, 5),
        "dna": (320, 8),
        "rna": (1_024, 6),
        "peptide": (1_024, 6),
    },
    "46m_plus": {
        "small_molecule": (1_536, 8),
        "protein": (320, 8),
        "dna": (320, 8),
        "rna": (768, 8),
        "peptide": (768, 8),
    },
}

SHARED_48GB: dict[str, tuple[int, int]] = {
    "small_molecule": (3_072, 4),
    "protein": (512, 5),
    "dna": (512, 5),
    "rna": (1_024, 6),
    "peptide": (768, 8),
}

# Standard-domain v2 profiles use the 48 GiB cards more fully while preserving
# each domain's effective record batch.  They require new matching, concurrent
# smoke evidence before formal freeze.  Peptide stays on v1 so completed Phase-D
# matrices and their smoke contracts retain their original semantics.
SHARED_48GB_V2_DOMAINS = frozenset(STANDARD_DOMAINS)
SHARED_48GB_V2_MINIMUM_HEADROOM_MIB = 4_096

# Small-molecule v3 candidates split at the exact 17M configuration.  A larger
# model at 3072 x 4 completed a dual probe but did not retain the required 4 GiB
# device headroom.  Both bands still need matching concurrent smoke evidence;
# these values alone do not authorize a formal run.
SHARED_48GB_SMALL_MOLECULE_V3_CUTOFF = 17_000_448
SHARED_48GB_SMALL_MOLECULE_V3 = {
    "le_17m": (3_072, 4),
    "gt_17m": (2_048, 6),
}

HARDWARE_CLASS_RULES: dict[str, dict[str, Any]] = {
    "single_24gb": {
        "minimum_memory_gib": 24,
        "maximum_concurrent_slots": 1,
        "profile_family": "single_24gb_size_banded",
    },
    "single_48gb": {
        "minimum_memory_gib": 48,
        "maximum_concurrent_slots": 1,
        # Keep the 24 GB size-banded profiles rather than inventing a new,
        # unmeasured single-process 48 GB batch.
        "profile_family": "single_24gb_size_banded",
    },
    "shared_48gb_dual": {
        "minimum_memory_gib": 48,
        "maximum_concurrent_slots": 2,
        "profile_family": "shared_48gb_dual",
    },
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def matrix_semantic_fingerprint(matrix: dict[str, Any]) -> str:
    semantic = copy.deepcopy(matrix)
    metadata = semantic.get("metadata", {})
    for key in ("created_at_utc", "generation_id", "semantic_fingerprint"):
        metadata.pop(key, None)
    payload = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def attach_matrix_identity(matrix: dict[str, Any]) -> None:
    fingerprint = matrix_semantic_fingerprint(matrix)
    phase_name = str(matrix["metadata"]["phase_name"])
    matrix["metadata"]["semantic_fingerprint"] = {
        "algorithm": "sha256_canonical_json_excluding_timestamps_and_identity",
        "value": fingerprint,
        "scope": "matrix_semantics_not_dataset_payload_files",
    }
    matrix["metadata"]["generation_id"] = f"{phase_name}_{fingerprint[:20]}"


def write_matrix(path: Path, matrix: dict[str, Any], *, design_dry_run: bool) -> str:
    """Write a preview, or immutably install/reuse an identical formal matrix."""
    if design_dry_run or not path.exists():
        atomic_json(path, matrix)
        return "written"
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"existing formal matrix is not an object: {path}")
    existing_fingerprint = matrix_semantic_fingerprint(existing)
    recorded_fingerprint = existing.get("metadata", {}).get(
        "semantic_fingerprint", {}
    )
    if (
        not isinstance(recorded_fingerprint, dict)
        or recorded_fingerprint.get("value") != existing_fingerprint
        or existing.get("metadata", {}).get("generation_id")
        != f"{existing.get('metadata', {}).get('phase_name')}_{existing_fingerprint[:20]}"
    ):
        raise ValueError(f"existing formal matrix identity is inconsistent: {path}")
    if existing_fingerprint != matrix_semantic_fingerprint(matrix):
        raise FileExistsError(
            f"refusing to overwrite a different formal matrix generation: {path}"
        )
    return "reused_identical_formal_matrix"


def parameter_band(parameter_count: int) -> str:
    count = int(parameter_count)
    if count <= 6_100_000:
        return "le_6m"
    if count < 32_000_000:
        return "7m_to_31m"
    if count <= 41_100_000:
        return "32m_to_41m"
    return "46m_plus"


def hardware_class(memory_gib: int, concurrent_slots: int) -> str:
    memory = int(memory_gib)
    slots = int(concurrent_slots)
    if memory >= 48 and slots == 2:
        return "shared_48gb_dual"
    if memory >= 48 and slots == 1:
        return "single_48gb"
    if memory >= 24 and slots == 1:
        return "single_24gb"
    raise ValueError(
        "unsupported hardware shape; supported shapes are 24GB x1, 48GB x1, "
        f"or 48GB x2, got memory_gib={memory} concurrent_slots={slots}"
    )


def batch_profile(
    domain: str,
    parameter_count: int,
    hardware_class_name: str,
) -> dict[str, Any]:
    if domain not in EFFECTIVE_BATCH_RECORDS:
        raise ValueError(f"no effective batch is frozen for domain {domain!r}")
    if hardware_class_name not in HARDWARE_CLASS_RULES:
        raise ValueError(f"unknown hardware class: {hardware_class_name}")
    band = parameter_band(parameter_count)
    family = str(HARDWARE_CLASS_RULES[hardware_class_name]["profile_family"])
    if family == "shared_48gb_dual":
        if domain == "small_molecule":
            band = (
                "le_17m" if int(parameter_count) <= SHARED_48GB_SMALL_MOLECULE_V3_CUTOFF
                else "gt_17m"
            )
            micro, grad_accum = SHARED_48GB_SMALL_MOLECULE_V3[band]
            profile_id = f"{hardware_class_name}_{domain}_{band}_v3"
        else:
            micro, grad_accum = SHARED_48GB[domain]
            version = 2 if domain in SHARED_48GB_V2_DOMAINS else 1
            profile_id = f"{hardware_class_name}_{domain}_all_models_v{version}"
    else:
        micro, grad_accum = SINGLE_24GB_BY_BAND[band][domain]
        profile_id = f"{hardware_class_name}_{domain}_{band}_v1"
    effective = int(micro) * int(grad_accum)
    expected = int(EFFECTIVE_BATCH_RECORDS[domain])
    if effective != expected:
        raise AssertionError(
            f"profile changes effective batch for {domain}/{hardware_class_name}/{band}: "
            f"{micro}*{grad_accum} != {expected}"
        )
    return {
        "batch_profile_id": profile_id,
        "hardware_class": hardware_class_name,
        "parameter_band": band,
        "per_device_train_batch_size": int(micro),
        "gradient_accumulation_steps": int(grad_accum),
        "effective_batch_records": effective,
    }


def default_hardware_layout() -> dict[str, Any]:
    return {
        "schema_version": HARDWARE_LAYOUT_SCHEMA,
        "layout_id": "local8_remote4_12gpu_16slot_v1",
        "devices": [
            *[
                {
                    "host_id": "local",
                    "gpu_id": gpu,
                    "memory_gib": 24,
                    "concurrent_slots": 1,
                }
                for gpu in range(4)
            ],
            *[
                {
                    "host_id": "local",
                    "gpu_id": gpu,
                    "memory_gib": 48,
                    "concurrent_slots": 2,
                }
                for gpu in range(4, 8)
            ],
            *[
                {
                    "host_id": "remote",
                    "gpu_id": gpu,
                    "memory_gib": 24,
                    "concurrent_slots": 1,
                }
                for gpu in range(4)
            ],
        ],
    }


def validate_hardware_layout(layout: dict[str, Any]) -> list[dict[str, Any]]:
    if layout.get("schema_version") != HARDWARE_LAYOUT_SCHEMA:
        raise ValueError(f"unsupported hardware layout schema: {layout.get('schema_version')!r}")
    unexpected_layout_keys = set(layout) - {"schema_version", "layout_id", "devices"}
    if unexpected_layout_keys:
        raise ValueError(
            "hardware eligibility is owned by the finalizer; unexpected layout keys: "
            f"{sorted(unexpected_layout_keys)}"
        )
    raw_devices = layout.get("devices")
    if not isinstance(raw_devices, list) or not raw_devices:
        raise ValueError("hardware layout must contain a non-empty devices list")
    devices: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_devices:
        if not isinstance(raw, dict):
            raise ValueError("each hardware device must be an object")
        unexpected_device_keys = set(raw) - {
            "host_id",
            "gpu_id",
            "memory_gib",
            "concurrent_slots",
        }
        if unexpected_device_keys:
            raise ValueError(
                "hardware eligibility is owned by the finalizer; unexpected device keys: "
                f"{sorted(unexpected_device_keys)}"
            )
        host_id = str(raw.get("host_id", ""))
        if not host_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for ch in host_id
        ):
            raise ValueError(f"unsafe hardware host_id: {host_id!r}")
        gpu_id = int(raw["gpu_id"])
        memory_gib = int(raw["memory_gib"])
        concurrent_slots = int(raw["concurrent_slots"])
        if gpu_id < 0:
            raise ValueError(f"negative gpu_id for {host_id}: {gpu_id}")
        key = (host_id, gpu_id)
        if key in seen:
            raise ValueError(f"duplicate hardware device: {host_id}/gpu{gpu_id}")
        seen.add(key)
        class_name = hardware_class(memory_gib, concurrent_slots)
        devices.append(
            {
                "host_id": host_id,
                "gpu_id": gpu_id,
                "memory_gib": memory_gib,
                "concurrent_slots": concurrent_slots,
                "hardware_class": class_name,
            }
        )
    return devices


def _integer(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    return result


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    """Validate semantic view totals without re-hashing the released payloads."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise ValueError(f"manifest must be complete {MANIFEST_SCHEMA}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or _integer(dataset.get("context_length"), "context_length") < 2:
        raise ValueError("manifest dataset/context_length is invalid")
    views = manifest.get("views")
    if not isinstance(views, dict):
        raise ValueError("manifest has no views object")
    expected_domains = {*STANDARD_DOMAINS, "peptide"}
    if set(views) != expected_domains:
        raise ValueError(
            f"manifest domains differ from the frozen five domains: {sorted(views)}"
        )

    stats: dict[str, dict[str, dict[str, int]]] = {}
    seen_domain_ids: set[int] = set()
    for domain in (*STANDARD_DOMAINS, "peptide"):
        domain_views = views[domain]
        if not isinstance(domain_views, dict) or set(domain_views) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"manifest {domain} must have exactly train/validation/test")
        stats[domain] = {}
        domain_ids: set[int] = set()
        for split in ("train", "validation", "test"):
            view = domain_views[split]
            if not isinstance(view, dict):
                raise ValueError(f"manifest view is not an object: {domain}/{split}")
            records = _integer(view.get("records"), f"{domain}/{split} records")
            input_tokens = _integer(
                view.get("effective_input_tokens_with_eos"),
                f"{domain}/{split} effective_input_tokens_with_eos",
            )
            domain_id = _integer(view.get("domain_id"), f"{domain}/{split} domain_id")
            shards = view.get("shards")
            if records <= 0 or input_tokens < records:
                raise ValueError(f"invalid manifest totals for {domain}/{split}")
            if not isinstance(shards, list) or not shards:
                raise ValueError(f"manifest view has no shards: {domain}/{split}")
            shard_records = 0
            shard_tokens = 0
            seen_shards: set[int] = set()
            for shard in shards:
                if not isinstance(shard, dict):
                    raise ValueError(f"manifest shard is not an object: {domain}/{split}")
                if str(shard.get("domain")) != domain or str(shard.get("split")) != split:
                    raise ValueError(f"manifest shard label mismatch: {domain}/{split}")
                if _integer(shard.get("domain_id"), "shard domain_id") != domain_id:
                    raise ValueError(f"manifest shard domain_id mismatch: {domain}/{split}")
                shard_id = _integer(shard.get("shard_id"), "shard_id")
                if shard_id in seen_shards:
                    raise ValueError(f"duplicate shard_id in {domain}/{split}: {shard_id}")
                seen_shards.add(shard_id)
                shard_records += _integer(shard.get("records"), "shard records")
                shard_tokens += _integer(
                    shard.get("effective_input_tokens_with_eos"), "shard input tokens"
                )
            if shard_records != records or shard_tokens != input_tokens:
                raise ValueError(
                    f"manifest shard totals mismatch for {domain}/{split}: "
                    f"records {shard_records}/{records}, tokens {shard_tokens}/{input_tokens}"
                )
            domain_ids.add(domain_id)
            stats[domain][split] = {
                "domain_id": domain_id,
                "records": records,
                "input_tokens": input_tokens,
                # Every frozen record contributes exactly one EOS input token.
                "loss_tokens": input_tokens - records,
            }
        if len(domain_ids) != 1:
            raise ValueError(f"domain_id changes across splits for {domain}")
        only_domain_id = next(iter(domain_ids))
        if only_domain_id in seen_domain_ids:
            raise ValueError(f"duplicate domain_id across domains: {only_domain_id}")
        seen_domain_ids.add(only_domain_id)
    return stats


def manifest_semantic_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the path-independent data contract a remote host must reproduce.

    Host-local tmpfs manifests legitimately rewrite payload and raw-index paths.
    Counts, split assignment, domain ids, shard ids, and token accounting may not
    change.  Embedding this projection in the frozen matrix lets a second host
    verify its local manifest without relying on the finalizer host's paths.
    """
    validate_manifest(manifest)
    dataset = manifest["dataset"]
    views: dict[str, Any] = {}
    for domain in (*STANDARD_DOMAINS, "peptide"):
        views[domain] = {}
        for split in ("train", "validation", "test"):
            view = manifest["views"][domain][split]
            shards = sorted(
                (
                    {
                        "domain": str(shard["domain"]),
                        "domain_id": int(shard["domain_id"]),
                        "split": str(shard["split"]),
                        "shard_id": int(shard["shard_id"]),
                        "token_file_name": Path(str(shard["token_file"])).name,
                        "raw_indices_file_name": Path(
                            str(shard["raw_indices_file"])
                        ).name,
                        "records": int(shard["records"]),
                        "effective_input_tokens_with_eos": int(
                            shard["effective_input_tokens_with_eos"]
                        ),
                        "supervised_causal_loss_tokens": int(
                            shard["effective_input_tokens_with_eos"]
                        )
                        - int(shard["records"]),
                    }
                    for shard in view["shards"]
                ),
                key=lambda shard: shard["shard_id"],
            )
            views[domain][split] = {
                "domain_id": int(view["domain_id"]),
                "records": int(view["records"]),
                "effective_input_tokens_with_eos": int(
                    view["effective_input_tokens_with_eos"]
                ),
                "supervised_causal_loss_tokens": int(
                    view["effective_input_tokens_with_eos"]
                )
                - int(view["records"]),
                "shards": shards,
            }
    return {
        "schema_version": str(manifest["schema_version"]),
        "status": str(manifest["status"]),
        "dataset": {
            "context_length": int(dataset["context_length"]),
            "num_shards": int(dataset["num_shards"]),
        },
        "split_assignment": copy.deepcopy(manifest["split_assignment"]),
        "views": views,
    }


def validate_row_against_manifest(
    row: dict[str, Any],
    manifest: dict[str, Any],
    stats: dict[str, dict[str, dict[str, int]]],
) -> None:
    domain = str(row["domain"])
    if domain not in stats:
        raise ValueError(f"matrix row uses unknown domain: {domain}")
    if int(row["context_length"]) != int(manifest["dataset"]["context_length"]):
        raise ValueError(f"context length mismatch for {domain}")
    if int(row["domain_id"]) != stats[domain]["train"]["domain_id"]:
        raise ValueError(f"domain_id mismatch for {domain}")
    if int(row["train_records"]) != stats[domain]["train"]["records"]:
        raise ValueError(f"train record mismatch for {domain}")
    if int(row["train_input_tokens_per_epoch"]) != stats[domain]["train"]["input_tokens"]:
        raise ValueError(f"train input-token mismatch for {domain}")
    if int(row["validation_loss_tokens"]) != stats[domain]["validation"]["loss_tokens"]:
        raise ValueError(f"validation loss-token mismatch for {domain}")
    if int(row["test_loss_tokens"]) != stats[domain]["test"]["loss_tokens"]:
        raise ValueError(f"test loss-token mismatch for {domain}")


def validate_proposal_manifest_contract(
    proposal: dict[str, Any], manifest: dict[str, Any]
) -> None:
    metadata = proposal.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("proposal has no metadata object")
    if metadata.get("split_assignment") != manifest.get("split_assignment"):
        raise ValueError("proposal and manifest use different split assignments")
    accounting = metadata.get("token_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("proposal has no token_accounting contract")
    if int(accounting.get("context_length", -1)) != int(
        manifest["dataset"]["context_length"]
    ):
        raise ValueError("proposal and manifest context lengths differ")
    if int(accounting.get("eos_tokens_per_record", -1)) != 1:
        raise ValueError("domain-wise matrix requires exactly one EOS input token per record")
    if accounting.get("padding_tokens_counted") is not False:
        raise ValueError("domain-wise FLOPs accounting must exclude padding tokens")
    if accounting.get("bpe_unfolding_assumed") != "none":
        raise ValueError("domain-wise matrix requires BPE unfolding to be disabled")


def validate_proposal_row(row: dict[str, Any], proposal: dict[str, Any]) -> None:
    specs = {str(spec["model_id"]): spec for spec in proposal.get("model_specs", [])}
    model_id = str(row["model_id"])
    if model_id not in specs:
        raise ValueError(f"proposal row references unknown model_id: {model_id}")
    spec = specs[model_id]
    integer_fields = (
        "parameter_count",
        "non_embedding_parameter_count",
        "legacy_document_parameter_count",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    for field in integer_fields:
        if int(row[field]) != int(spec[field]):
            raise ValueError(f"proposal row/model_specs mismatch for {model_id}/{field}")
    if int(row["vocab_size"]) != int(proposal["metadata"]["vocab_size"]):
        raise ValueError(f"proposal row vocab mismatch for {model_id}")
    if int(row.get("seed", -1)) != 42 or row.get("independent_random_initialization") is not True:
        raise ValueError("primary Phase-C/D rows must use independent seed-42 initialization")
    params = int(row["parameter_count"])
    target_tokens = int(row["target_input_tokens_with_eos"])
    train_tokens = int(row["train_input_tokens_per_epoch"])
    if params <= 0 or target_tokens <= 0 or train_tokens <= 0:
        raise ValueError(f"non-positive N/D value in proposal row {model_id}")
    exact_rounded_flops = 6 * params * target_tokens
    if not math.isclose(
        float(row["target_flops_after_integer_token_rounding"]),
        float(exact_rounded_flops),
        rel_tol=1e-15,
        abs_tol=1.0,
    ):
        raise ValueError(f"rounded 6ND mismatch in proposal row {model_id}")
    target_flops = float(row["target_flops_6nd"])
    expected_error = (exact_rounded_flops - target_flops) / target_flops
    if not math.isclose(
        float(row["target_flops_relative_rounding_error"]),
        expected_error,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"FLOPs rounding error mismatch in proposal row {model_id}")
    epoch_fraction = target_tokens / train_tokens
    if row.get("execution_phase") == "C_completion":
        if not math.isclose(
            float(row["target_epoch_fraction"]),
            epoch_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"epoch fraction mismatch in proposal row {model_id}")
        if bool(row["four_epoch_compliant"]) != (epoch_fraction <= 4.0):
            raise ValueError(f"four_epoch_compliant mismatch in proposal row {model_id}")
        expected_tokens = max(1, int(round(target_flops / (6.0 * params))))
        if target_tokens != expected_tokens:
            raise ValueError(f"Phase-C token target does not implement fixed 6ND for {model_id}")
    elif row.get("execution_phase") == "D_peptide_surface":
        declared_fraction = float(row["target_epoch_fraction"])
        if bool(row["four_epoch_compliant"]) != (declared_fraction <= 4.0):
            raise ValueError(f"four_epoch_compliant mismatch in proposal row {model_id}")
        expected_tokens = max(1, int(round(train_tokens * declared_fraction)))
        if target_tokens != expected_tokens:
            raise ValueError(f"Phase-D token target does not implement the epoch fraction for {model_id}")


def c_curve_key(domain: str, budget_label: str) -> str:
    return f"{domain}/{budget_label}"


def expected_b_curve_keys() -> set[str]:
    return {
        c_curve_key(domain, budget)
        for domain in STANDARD_DOMAINS
        for budget in EXPECTED_B_BUDGETS
    }


def expected_c_curves(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = c_curve_key(str(row["domain"]), str(row["budget_label"]))
        grouped.setdefault(key, []).append(row)
    expected = {
        c_curve_key(domain, budget)
        for domain in STANDARD_DOMAINS
        for budget in EXPECTED_C_BUDGETS
    }
    if set(grouped) != expected:
        raise ValueError(
            f"Phase-C curves differ from expected 4x3 grid: "
            f"missing={sorted(expected-set(grouped))} extra={sorted(set(grouped)-expected)}"
        )
    if any(len(curve_rows) != 5 for curve_rows in grouped.values()):
        raise ValueError("every Phase-C curve must contain exactly five proposal rows")
    return grouped


def review_template(proposal: dict[str, Any], c_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = expected_c_curves(c_rows)
    return {
        "schema_version": REVIEW_SCHEMA,
        "status": "pending_validation_review",
        "proposal_matrix_version": proposal["metadata"].get("matrix_version"),
        "test_data_accessed": False,
        "phase_b_primary_runs_validated": 0,
        "phase_b_curves_reviewed": 0,
        "all_phase_b_curves_bracketed": None,
        "phase_b_curves": [
            {
                "domain": key.split("/", 1)[0],
                "budget_label": key.split("/", 1)[1],
                "valid_run_count": 0,
                "bracketing_status": "pending_validation_review",
                "placement_direction": "pending_validation_review",
                "decision": "pending_validation_review",
            }
            for key in sorted(expected_b_curve_keys())
        ],
        "phase_c_curves": [
            {
                "domain": key.split("/", 1)[0],
                "budget_label": key.split("/", 1)[1],
                "decision": "pending_validation_review",
                "responds_to_phase_b_curves": [],
                "model_ids": [
                    str(row["model_id"])
                    for row in sorted(grouped[key], key=lambda item: int(item["model_slot"]))
                ],
            }
            for key in sorted(grouped)
        ],
    }


def validate_phase_c_review(
    review: dict[str, Any],
    proposal: dict[str, Any],
    c_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    if review.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError(f"unsupported Phase-B review schema: {review.get('schema_version')!r}")
    if review.get("status") != "approved_for_phase_c_freeze":
        raise ValueError("Phase-B review is not approved_for_phase_c_freeze")
    if review.get("proposal_matrix_version") != proposal["metadata"].get("matrix_version"):
        raise ValueError("Phase-B review was made against a different proposal version")
    if review.get("test_data_accessed") is not False:
        raise ValueError("Phase-C review gate requires test_data_accessed=false")
    producer = review.get("producer")
    if (
        not isinstance(producer, dict)
        or producer.get("schema_version")
        != "domainwise_phase_b_validation_review_producer_v1"
        or producer.get("script") != "review_phase_b_validation_bracketing.py"
        or producer.get("method_version") not in {
            "log10n_quadratic_bracketing_v1",
            "log10n_quadratic_bracketing_placement_v2",
        }
    ):
        raise ValueError("Phase-C formal review must come from the frozen validation reviewer")
    schema_evidence = review.get("schema_evidence")
    if (
        not isinstance(schema_evidence, dict)
        or schema_evidence.get("target_review_schema") != REVIEW_SCHEMA
        or int(schema_evidence.get("expected_primary_runs", -1)) != 60
        or int(schema_evidence.get("expected_phase_b_curves", -1)) != 12
        or schema_evidence.get("validation_metric") != "summary.final_validation.loss"
        or schema_evidence.get("finalizer_compatibility_check") != "passed"
    ):
        raise ValueError("Phase-C review lacks passed numerical/schema evidence")
    if review.get("approval_blockers") not in (None, []):
        raise ValueError("Phase-C review still contains approval blockers")
    if int(review.get("phase_b_primary_runs_validated", -1)) != 60:
        raise ValueError("Phase-C review gate requires exactly 60 validated primary Phase-B runs")
    if int(review.get("phase_b_curves_reviewed", -1)) != 12:
        raise ValueError("Phase-C review gate requires all twelve Phase-B curves")

    expected_b = expected_b_curve_keys()
    raw_b_curves = review.get("phase_b_curves")
    if not isinstance(raw_b_curves, list):
        raise ValueError("Phase-B review has no phase_b_curves list")
    placement_by_b_curve: dict[str, str] = {}
    bracketing_by_b_curve: dict[str, str] = {}
    allowed_statuses = {"bracketed", "unbracketed_lower_n", "unbracketed_higher_n"}
    allowed_directions = {"keep", "lower_n", "higher_n"}
    placement_v2 = producer["method_version"] == "log10n_quadratic_bracketing_placement_v2"
    if placement_v2:
        # Delayed import avoids the reviewer's existing finalizer dependency.
        from review_phase_b_validation_bracketing import validate_placement_replay

        allowed_statuses.add("ambiguous")
        analysis_config = review.get("analysis_config")
        if not isinstance(analysis_config, dict):
            raise ValueError("v2 Phase-B review lacks its numerical analysis_config")
        review_model_counts = {
            str(spec["model_id"]): int(spec["parameter_count"])
            for spec in proposal.get("model_specs", [])
        }
    for item in raw_b_curves:
        if not isinstance(item, dict):
            raise ValueError("Phase-B review curve must be an object")
        key = c_curve_key(str(item.get("domain")), str(item.get("budget_label")))
        if key in placement_by_b_curve:
            raise ValueError(f"duplicate Phase-B review curve: {key}")
        if int(item.get("valid_run_count", -1)) < 5:
            raise ValueError(f"Phase-B curve {key} has fewer than five valid runs")
        status = str(item.get("bracketing_status"))
        direction = str(item.get("placement_direction"))
        decision = item.get("decision")
        if placement_v2:
            direction = validate_placement_replay(item, analysis_config, review_model_counts)
            decision = item["placement_resolution"]["rule"]
        if status not in allowed_statuses:
            raise ValueError(f"Phase-B curve {key} has invalid bracketing_status={status!r}")
        if direction not in allowed_directions:
            raise ValueError(f"Phase-B curve {key} has invalid placement_direction={direction!r}")
        if not isinstance(decision, str) or not decision.strip() or decision.startswith("pending"):
            raise ValueError(f"Phase-B curve {key} has no resolved decision")
        if status == "unbracketed_lower_n" and direction != "lower_n":
            raise ValueError(f"unbracketed lower-N curve {key} must direct C toward lower N")
        if status == "unbracketed_higher_n" and direction != "higher_n":
            raise ValueError(f"unbracketed higher-N curve {key} must direct C toward higher N")
        placement_by_b_curve[key] = direction
        bracketing_by_b_curve[key] = status
    if set(placement_by_b_curve) != expected_b:
        raise ValueError(
            f"review Phase-B curve set is incomplete: "
            f"missing={sorted(expected_b-set(placement_by_b_curve))} "
            f"extra={sorted(set(placement_by_b_curve)-expected_b)}"
        )
    actual_all_bracketed = all(
        status == "bracketed" for status in bracketing_by_b_curve.values()
    )
    if review.get("all_phase_b_curves_bracketed") is not actual_all_bracketed:
        raise ValueError(
            "all_phase_b_curves_bracketed must truthfully summarize the per-curve statuses"
        )

    grouped = expected_c_curves(c_rows)
    expected_keys = set(grouped)
    specs = {str(spec["model_id"]): spec for spec in proposal.get("model_specs", [])}
    raw_curves = review.get("phase_c_curves")
    if not isinstance(raw_curves, list):
        raise ValueError("Phase-B review has no phase_c_curves list")
    selections: dict[str, list[str]] = {}
    covered_directional_b_curves: set[str] = set()
    for item in raw_curves:
        if not isinstance(item, dict):
            raise ValueError("Phase-C review curve must be an object")
        key = c_curve_key(str(item.get("domain")), str(item.get("budget_label")))
        if key in selections:
            raise ValueError(f"duplicate Phase-C review curve: {key}")
        model_ids = item.get("model_ids")
        if not isinstance(model_ids, list) or len(model_ids) != 5:
            raise ValueError(f"review curve {key} must explicitly select five model_ids")
        model_ids = [str(model_id) for model_id in model_ids]
        if len(set(model_ids)) != 5:
            raise ValueError(f"review curve {key} repeats a model_id")
        if any(model_id not in specs for model_id in model_ids):
            unknown = sorted(model_id for model_id in model_ids if model_id not in specs)
            raise ValueError(f"review curve {key} selects unknown models: {unknown}")
        new_counts = [int(specs[model_id]["parameter_count"]) for model_id in model_ids]
        if new_counts != sorted(new_counts):
            raise ValueError(f"review model_ids must be ordered by increasing N for {key}")
        proposal_rows = sorted(grouped[key], key=lambda row: int(row["model_slot"]))
        proposal_ids = [str(row["model_id"]) for row in proposal_rows]
        proposal_counts = [int(row["parameter_count"]) for row in proposal_rows]
        decision = str(item.get("decision"))
        allowed_c_decisions = {
            "keep_proposal",
            "adjust_lower_n",
            "adjust_higher_n",
            "adjust_both",
        }
        if decision not in allowed_c_decisions:
            raise ValueError(f"review curve {key} has invalid decision={decision!r}")
        references = item.get("responds_to_phase_b_curves")
        if not isinstance(references, list) or len(set(map(str, references))) != len(references):
            raise ValueError(f"review curve {key} has invalid responds_to_phase_b_curves")
        domain, c_budget = key.split("/", 1)
        required_directions: set[str] = set()
        for raw_reference in references:
            reference = str(raw_reference)
            if reference not in expected_b:
                raise ValueError(f"review curve {key} references unknown Phase-B curve {reference}")
            ref_domain, ref_budget = reference.split("/", 1)
            if ref_domain != domain or ref_budget not in C_PHASE_B_NEIGHBORS[c_budget]:
                raise ValueError(f"review curve {key} references non-neighbor evidence {reference}")
            direction = placement_by_b_curve[reference]
            if direction != "keep":
                required_directions.add(direction)
                covered_directional_b_curves.add(reference)
        lower_moved = min(new_counts) < min(proposal_counts)
        higher_moved = max(new_counts) > max(proposal_counts)
        if decision == "keep_proposal":
            if model_ids != proposal_ids:
                raise ValueError(f"review curve {key} says keep_proposal but changes model_ids")
            if required_directions:
                raise ValueError(f"review curve {key} ignores directional Phase-B evidence")
        else:
            if not required_directions:
                raise ValueError(f"review curve {key} adjusts models without directional Phase-B evidence")
            if decision in {"adjust_lower_n", "adjust_both"} and not lower_moved:
                raise ValueError(f"review curve {key} does not actually extend toward lower N")
            if decision in {"adjust_higher_n", "adjust_both"} and not higher_moved:
                raise ValueError(f"review curve {key} does not actually extend toward higher N")
            if "lower_n" in required_directions and decision not in {
                "adjust_lower_n",
                "adjust_both",
            }:
                raise ValueError(f"review curve {key} fails lower-N Phase-B direction")
            if "higher_n" in required_directions and decision not in {
                "adjust_higher_n",
                "adjust_both",
            }:
                raise ValueError(f"review curve {key} fails higher-N Phase-B direction")
        selections[key] = model_ids
    if set(selections) != expected_keys:
        raise ValueError(
            f"review Phase-C curve set is incomplete: "
            f"missing={sorted(expected_keys-set(selections))} "
            f"extra={sorted(set(selections)-expected_keys)}"
        )
    required_directional_b_curves = {
        key for key, direction in placement_by_b_curve.items() if direction != "keep"
    }
    if covered_directional_b_curves != required_directional_b_curves:
        raise ValueError(
            "directional Phase-B decisions are not all applied to a neighboring Phase-C curve: "
            f"uncovered={sorted(required_directional_b_curves-covered_directional_b_curves)}"
        )
    return selections


def rebuild_phase_c_rows(
    proposal: dict[str, Any],
    c_rows: list[dict[str, Any]],
    selections: dict[str, list[str]],
) -> list[dict[str, Any]]:
    specs = {str(spec["model_id"]): spec for spec in proposal.get("model_specs", [])}
    if not specs:
        raise ValueError("proposal has no model_specs for reviewed Phase-C selection")
    grouped = expected_c_curves(c_rows)
    rebuilt: list[dict[str, Any]] = []
    for key in sorted(grouped):
        source_rows = sorted(grouped[key], key=lambda item: int(item["model_slot"]))
        base = source_rows[0]
        target_flops = float(base["target_flops_6nd"])
        train_tokens = int(base["train_input_tokens_per_epoch"])
        prior = float(base["prior_parameter_count"])
        selected_specs: list[dict[str, Any]] = []
        for model_id in selections[key]:
            if model_id not in specs:
                raise ValueError(f"review selects unknown model {model_id!r} for {key}")
            selected_specs.append(specs[model_id])
        counts = [int(spec["parameter_count"]) for spec in selected_specs]
        if counts != sorted(counts):
            raise ValueError(f"review model_ids must be ordered by increasing N for {key}")
        minimum = target_flops / (24.0 * train_tokens)
        if any(count < minimum for count in counts):
            raise ValueError(f"review selection violates the four-epoch N floor for {key}")
        for slot, (multiplier, spec) in enumerate(zip(TARGET_MULTIPLIERS, selected_specs)):
            row = copy.deepcopy(base)
            params = int(spec["parameter_count"])
            target_tokens = max(1, int(round(target_flops / (6.0 * params))))
            rounded_flops = 6 * params * target_tokens
            row.update(
                {
                    "model_slot": slot,
                    "target_multiplier_vs_prior": multiplier,
                    "model_id": str(spec["model_id"]),
                    "parameter_count": params,
                    "non_embedding_parameter_count": int(spec["non_embedding_parameter_count"]),
                    "legacy_document_parameter_count": int(
                        spec["legacy_document_parameter_count"]
                    ),
                    "minimum_parameter_count_four_epoch": minimum,
                    "actual_multiplier_vs_prior": params / prior,
                    "target_input_tokens_with_eos": target_tokens,
                    "target_flops_after_integer_token_rounding": rounded_flops,
                    "target_flops_relative_rounding_error": (
                        rounded_flops - target_flops
                    )
                    / target_flops,
                    "target_epoch_fraction": target_tokens / train_tokens,
                    "four_epoch_compliant": target_tokens / train_tokens <= 4.0,
                    "hidden_size": int(spec["hidden_size"]),
                    "intermediate_size": int(spec["intermediate_size"]),
                    "num_hidden_layers": int(spec["num_hidden_layers"]),
                    "num_attention_heads": int(spec["num_attention_heads"]),
                    "num_key_value_heads": int(spec["num_key_value_heads"]),
                    "head_dim": int(spec["head_dim"]),
                }
            )
            rebuilt.append(row)
    return rebuilt


def validate_phase_shape(phase: str, rows: list[dict[str, Any]]) -> None:
    phase_spec = PHASES[phase]
    expected_rows = int(phase_spec["expected_rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} Phase-{phase.upper()} rows, got {len(rows)}")
    if any(row.get("execution_phase") != phase_spec["execution_phase"] for row in rows):
        raise ValueError(f"Phase-{phase.upper()} rows contain another execution phase")
    if phase == "c":
        grouped = expected_c_curves(rows)
        for key, curve_rows in grouped.items():
            domain, budget = key.split("/", 1)
            expected_budget_index = int(budget.removeprefix("C"))
            if domain not in STANDARD_DOMAINS or expected_budget_index not in {1, 3, 4}:
                raise ValueError(f"unexpected Phase-C curve: {key}")
            if any(int(row["budget_index"]) != expected_budget_index for row in curve_rows):
                raise ValueError(f"Phase-C budget index mismatch for {key}")
            if len({float(row["target_flops_6nd"]) for row in curve_rows}) != 1:
                raise ValueError(f"Phase-C target FLOPs vary within {key}")
    else:
        if {str(row["domain"]) for row in rows} != {"peptide"}:
            raise ValueError("Phase D must contain peptide rows only")
        cross = {(str(row["model_id"]), float(row["target_epoch_fraction"])) for row in rows}
        expected = {(model, epoch) for model in EXPECTED_D_MODELS for epoch in EXPECTED_D_EPOCHS}
        if cross != expected:
            raise ValueError("Phase D is not the frozen five-model x five-epoch cross")
    if not all(bool(row.get("four_epoch_compliant")) for row in rows):
        raise ValueError(f"Phase-{phase.upper()} contains a row above four epochs")


def eligible_profiles_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    mean_tokens = float(row["train_input_tokens_per_epoch"]) / int(row["train_records"])
    profiles: list[dict[str, Any]] = []
    for class_name, rule in HARDWARE_CLASS_RULES.items():
        profile = batch_profile(str(row["domain"]), int(row["parameter_count"]), class_name)
        profile.update(
            {
                "minimum_memory_gib": int(rule["minimum_memory_gib"]),
                "maximum_concurrent_slots": int(rule["maximum_concurrent_slots"]),
                "expected_input_tokens_per_optimizer_step": mean_tokens
                * int(profile["effective_batch_records"]),
            }
        )
        profiles.append(profile)
    return profiles


def assign_rows_to_hardware(
    rows: list[dict[str, Any]],
    devices: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    physical_load = {
        (str(device["host_id"]), int(device["gpu_id"])): 0.0 for device in devices
    }
    slot_load = {
        (str(device["host_id"]), int(device["gpu_id"])): [
            0.0 for _ in range(int(device["concurrent_slots"]))
        ]
        for device in devices
    }
    device_order = {
        (str(device["host_id"]), int(device["gpu_id"])): index
        for index, device in enumerate(devices)
    }
    # When the phase has enough work, prime every declared worker slot once
    # before ordinary LPT balancing.  This makes the 48 GiB secondary slots a
    # real part of the campaign instead of allowing physical-device balancing
    # to leave them unused (Phase D has 25 rows for 16 slots).
    unused_slots = {
        (str(device["host_id"]), int(device["gpu_id"]), local_slot)
        for device in devices
        for local_slot in range(int(device["concurrent_slots"]))
    }
    for row in sorted(
        rows,
        key=lambda item: (
            -float(item["target_flops_after_integer_token_rounding"]),
            str(item["domain"]),
            str(item["budget_label"]),
            str(item["model_id"]),
        ),
    ):
        profiles = eligible_profiles_for_row(row)
        profile_by_class = {str(profile["hardware_class"]): profile for profile in profiles}
        eligible_devices = [
            device for device in devices if str(device["hardware_class"]) in profile_by_class
        ]
        if not eligible_devices:
            raise ValueError(f"no eligible hardware device for row {row.get('run_id')}")
        unused_eligible = [
            (device, local_slot)
            for device in eligible_devices
            for local_slot in range(int(device["concurrent_slots"]))
            if (
                str(device["host_id"]),
                int(device["gpu_id"]),
                local_slot,
            )
            in unused_slots
        ]
        if unused_eligible:
            device, local_slot = min(
                unused_eligible,
                key=lambda item: (
                    physical_load[(str(item[0]["host_id"]), int(item[0]["gpu_id"]))],
                    device_order[(str(item[0]["host_id"]), int(item[0]["gpu_id"]))],
                    item[1],
                ),
            )
        else:
            device = min(
                eligible_devices,
                key=lambda item: (
                    physical_load[(str(item["host_id"]), int(item["gpu_id"]))],
                    device_order[(str(item["host_id"]), int(item["gpu_id"]))],
                ),
            )
            device_key = (str(device["host_id"]), int(device["gpu_id"]))
            local_slot = min(
                range(int(device["concurrent_slots"])),
                key=lambda slot: (slot_load[device_key][slot], slot),
            )
        key = (str(device["host_id"]), int(device["gpu_id"]))
        unused_slots.discard((key[0], key[1], local_slot))
        selected = profile_by_class[str(device["hardware_class"])]
        expected_tokens = float(selected["expected_input_tokens_per_optimizer_step"])
        target_tokens = int(row["target_input_tokens_with_eos"])
        scheduler_steps = int(math.ceil(target_tokens / expected_tokens))
        row.update(
            {
                "hardware_eligibility": {
                    "minimum_memory_gib": min(
                        int(profile["minimum_memory_gib"]) for profile in profiles
                    ),
                    "eligible_hardware_classes": [
                        str(profile["hardware_class"]) for profile in profiles
                    ],
                    "profiles": profiles,
                },
                "assigned_host": str(device["host_id"]),
                "assigned_gpu": int(device["gpu_id"]),
                "gpu_local_slot": local_slot,
                "worker_slot": (
                    f"{device['host_id']}_gpu{int(device['gpu_id'])}_slot{local_slot}"
                ),
                "assigned_hardware_class": str(device["hardware_class"]),
                "assigned_device_memory_gib": int(device["memory_gib"]),
                "assigned_device_concurrent_slots": int(device["concurrent_slots"]),
                "batch_profile_id": str(selected["batch_profile_id"]),
                "per_device_train_batch_size": int(
                    selected["per_device_train_batch_size"]
                ),
                "gradient_accumulation_steps": int(
                    selected["gradient_accumulation_steps"]
                ),
                "effective_batch_records": int(selected["effective_batch_records"]),
                "expected_input_tokens_per_optimizer_step": expected_tokens,
                "max_steps": scheduler_steps,
                "scheduler_total_steps": scheduler_steps,
            }
        )
        flops = float(row["target_flops_after_integer_token_rounding"])
        physical_load[key] += flops
        slot_load[key][local_slot] += flops
    return (
        {f"{host}:gpu{gpu}": load for (host, gpu), load in physical_load.items()},
        {
            f"{host}_gpu{gpu}_slot{slot}": load
            for (host, gpu), loads in slot_load.items()
            for slot, load in enumerate(loads)
        },
    )


def required_profile_evidence(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["assigned_host"]),
            str(row["domain"]),
            str(row["assigned_hardware_class"]),
            str(row["batch_profile_id"]),
            int(row["per_device_train_batch_size"]),
            int(row["gradient_accumulation_steps"]),
            int(row["effective_batch_records"]),
        )
        current = grouped.get(key)
        requirement = {
            "host_id": key[0],
            "domain": key[1],
            "hardware_class": key[2],
            "batch_profile_id": key[3],
            "per_device_train_batch_size": key[4],
            "gradient_accumulation_steps": key[5],
            "effective_batch_records": key[6],
            "minimum_parameter_count_tested": int(row["parameter_count"]),
            "minimum_device_memory_gib": int(row["assigned_device_memory_gib"]),
            "minimum_concurrent_processes": (
                2 if str(row["assigned_hardware_class"]) == "shared_48gb_dual" else 1
            ),
        }
        v2_shared_profile = (
            key[1] in SHARED_48GB_V2_DOMAINS
            and key[3] == f"shared_48gb_dual_{key[1]}_all_models_v2"
        )
        v3_small_profile = (
            key[1] == "small_molecule"
            and key[3] in {
                f"shared_48gb_dual_small_molecule_{band}_v3"
                for band in SHARED_48GB_SMALL_MOLECULE_V3
            }
        )
        if key[2] == "shared_48gb_dual" and (v2_shared_profile or v3_small_profile):
            requirement["minimum_device_headroom_mib"] = (
                SHARED_48GB_V2_MINIMUM_HEADROOM_MIB
            )
        if current is None:
            grouped[key] = requirement
        else:
            current["minimum_parameter_count_tested"] = max(
                int(current["minimum_parameter_count_tested"]),
                int(requirement["minimum_parameter_count_tested"]),
            )
    requirements = sorted(
        grouped.values(),
        key=lambda item: (
            item["domain"],
            item["host_id"],
            item["hardware_class"],
            item["batch_profile_id"],
        ),
    )
    payload = json.dumps(
        requirements,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return requirements, hashlib.sha256(payload).hexdigest()


def validate_profile_evidence(
    report: dict[str, Any],
    *,
    phase: str,
    layout_id: str,
    requirements: list[dict[str, Any]],
    contract_id: str,
) -> None:
    """Fail closed unless every actually assigned batch profile has real evidence."""
    if report.get("schema_version") != PROFILE_EVIDENCE_SCHEMA:
        raise ValueError(f"unsupported batch-profile evidence schema: {report.get('schema_version')!r}")
    if report.get("status") != "approved_for_formal_freeze":
        raise ValueError("batch-profile evidence is not approved for formal freeze")
    if report.get("phase") != phase or report.get("hardware_layout_id") != layout_id:
        raise ValueError("batch-profile evidence phase/layout does not match the matrix")
    if report.get("profile_contract_id") != contract_id:
        raise ValueError("batch-profile evidence contract id does not match assigned profiles")
    if report.get("test_data_accessed") is not False:
        raise ValueError("batch-profile evidence must not access the reserved test split")
    producer = report.get("producer")
    if not isinstance(producer, dict) or not producer.get("script") or not producer.get(
        "method_version"
    ):
        raise ValueError("batch-profile evidence has no producer/method identity")
    entries = report.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("batch-profile evidence has no entries")
    indexed: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("batch-profile evidence entry is not an object")
        key = (
            str(entry.get("host_id")),
            str(entry.get("domain")),
            str(entry.get("hardware_class")),
            str(entry.get("batch_profile_id")),
            int(entry.get("per_device_train_batch_size", -1)),
            int(entry.get("gradient_accumulation_steps", -1)),
            int(entry.get("effective_batch_records", -1)),
        )
        if key[4] <= 0 or key[5] <= 0 or key[4] * key[5] != key[6]:
            raise ValueError(f"evidence entry changes effective batch: {key}")
        if entry.get("evidence_kind") not in {
            "phase_b_completed_run",
            "optimizer_step_smoke",
        }:
            raise ValueError(f"unsupported evidence kind: {entry.get('evidence_kind')!r}")
        if phase == "d" and entry.get("evidence_kind") != "optimizer_step_smoke":
            raise ValueError("Phase-D peptide profiles require an optimizer-step smoke")
        if (
            int(entry.get("complete_optimizer_steps", 0)) < 1
            or int(entry.get("backward_passes", 0))
            < int(entry.get("gradient_accumulation_steps", -1))
            or int(entry.get("maximum_optimizer_micro_steps", 0))
            < int(entry.get("gradient_accumulation_steps", -1))
            or entry.get("full_gradient_window_verified") is not True
            or entry.get("finite_loss") is not True
            or entry.get("oom_observed") is not False
        ):
            raise ValueError(f"evidence did not complete a finite optimizer step: {key}")
        peak_allocated = float(entry.get("peak_allocated_mib", math.nan))
        peak_reserved = float(entry.get("peak_reserved_mib", math.nan))
        device_total = float(entry.get("device_total_mib", math.nan))
        if not (
            math.isfinite(peak_allocated)
            and math.isfinite(peak_reserved)
            and math.isfinite(device_total)
            and 0 < peak_allocated <= peak_reserved < device_total
        ):
            raise ValueError(f"invalid peak/device memory evidence: {key}")
        artifacts = entry.get("source_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"evidence has no source artifacts: {key}")
        for artifact in artifacts:
            artifact_path = Path(str(artifact)).expanduser().resolve()
            if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
                raise FileNotFoundError(f"evidence source artifact is unavailable: {artifact_path}")
        indexed.setdefault(key, []).append(entry)

    for requirement in requirements:
        key = (
            requirement["host_id"],
            requirement["domain"],
            requirement["hardware_class"],
            requirement["batch_profile_id"],
            requirement["per_device_train_batch_size"],
            requirement["gradient_accumulation_steps"],
            requirement["effective_batch_records"],
        )
        candidates = indexed.get(key, [])
        covered = False
        for entry in candidates:
            if int(entry.get("parameter_count_tested", 0)) < int(
                requirement["minimum_parameter_count_tested"]
            ):
                continue
            if float(entry.get("device_total_mib", 0)) < (
                int(requirement["minimum_device_memory_gib"]) * 1024 - 1536
            ):
                continue
            required_concurrency = int(requirement["minimum_concurrent_processes"])
            if int(entry.get("concurrent_processes", 0)) < required_concurrency:
                continue
            if required_concurrency >= 2 and entry.get("simultaneous_execution_verified") is not True:
                continue
            if required_concurrency >= 2:
                aggregate_reserved = float(
                    entry.get("aggregate_peak_reserved_mib", math.nan)
                )
                if not (
                    math.isfinite(aggregate_reserved)
                    and 0 < aggregate_reserved < float(entry["device_total_mib"])
                ):
                    continue
                if float(entry["device_total_mib"]) - aggregate_reserved < float(
                    requirement.get("minimum_device_headroom_mib", 0)
                ):
                    continue
            covered = True
            break
        if not covered:
            raise ValueError(f"batch-profile evidence does not cover requirement: {requirement}")


def build_phase_matrix(
    *,
    phase: str,
    proposal: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    run_root: Path,
    experiment_name: str,
    hardware_layout: dict[str, Any],
    design_dry_run: bool,
    review: dict[str, Any] | None,
    review_path: Path | None,
    profile_evidence: dict[str, Any] | None,
    profile_evidence_path: Path | None,
    warmup_ratio: float,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    phase_spec = PHASES[phase]
    if proposal.get("metadata", {}).get("status") != PROPOSAL_STATUS:
        raise ValueError("unexpected proposal status")
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError("warmup_ratio must be in [0, 1)")
    if (
        not experiment_name
        or experiment_name in {".", ".."}
        or Path(experiment_name).name != experiment_name
        or "/" in experiment_name
        or "\\" in experiment_name
    ):
        raise ValueError("experiment_name must be one safe directory name")
    stats = validate_manifest(manifest)
    validate_proposal_manifest_contract(proposal, manifest)
    source_rows = [
        copy.deepcopy(row)
        for row in proposal.get("rows", [])
        if row.get("execution_phase") == phase_spec["execution_phase"]
    ]
    validate_phase_shape(phase, source_rows)
    for row in source_rows:
        validate_proposal_row(row, proposal)
        validate_row_against_manifest(row, manifest, stats)

    review_satisfied = not bool(phase_spec["review_required"])
    selection_source = "proposal_v1"
    if phase == "c":
        if review is None:
            if not design_dry_run:
                raise ValueError(
                    "formal Phase-C freeze requires --review-json; use --design-dry-run "
                    "only for a non-runnable preview"
                )
        else:
            selections = validate_phase_c_review(review, proposal, source_rows)
            source_rows = rebuild_phase_c_rows(proposal, source_rows, selections)
            for row in source_rows:
                validate_proposal_row(row, proposal)
                validate_row_against_manifest(row, manifest, stats)
            validate_phase_shape(phase, source_rows)
            review_satisfied = True
            selection_source = "approved_phase_b_validation_review"

    devices = validate_hardware_layout(hardware_layout)
    phase_name = str(phase_spec["phase_name"])
    formal_status = str(phase_spec["formal_status"])
    output_status = "design_dry_run_not_runnable" if design_dry_run else formal_status
    output_base = run_root / experiment_name / phase_name
    run_base = output_base / ("design_dry_run/runs" if design_dry_run else "runs")
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for source in source_rows:
        row = copy.deepcopy(source)
        run_id = (
            f"{row['domain']}_{row['budget_label']}_{row['model_id']}_s{int(row['seed'])}"
        )
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id after review selection: {run_id}")
        seen_run_ids.add(run_id)
        for old_placeholder in (
            "measured_input_tokens_per_step",
            "assigned_gpu",
            "gpu_local_slot",
            "worker_slot",
        ):
            row.pop(old_placeholder, None)
        row.update(
            {
                "matrix_status": output_status,
                "run_id": run_id,
                "run_name": run_id,
                "out_dir": str(run_base / run_id),
                "gpu_count": 1,
                "world_size": 1,
                "warmup_ratio": float(warmup_ratio),
                "token_stop_authority": "target_input_tokens_with_eos",
                "domain_split_manifest": str(manifest_path),
                "train_split": "train",
                "validation_split": "validation",
                "test_split": "test_reserved_until_analysis_freeze",
                "bpe_unfold_policy": "none",
                "model_implementation": "packed-xformers",
                "attention_backend": (
                    "xformers_memory_efficient_block_diagonal_causal"
                ),
            }
        )
        rows.append(row)
    physical_load, slot_load = assign_rows_to_hardware(rows, devices)
    for row in rows:
        row["warmup_steps"] = max(
            1,
            int(round(int(row["scheduler_total_steps"]) * float(warmup_ratio))),
        )
    rows.sort(
        key=lambda row: (
            str(row["assigned_host"]),
            int(row["assigned_gpu"]),
            int(row["gpu_local_slot"]),
            str(row["domain"]),
            -1 if row.get("budget_index") is None else int(row["budget_index"]),
            -1 if row.get("model_slot") is None else int(row["model_slot"]),
            str(row["model_id"]),
        )
    )
    validate_phase_shape(phase, rows)

    profile_requirements, profile_contract_id = required_profile_evidence(rows)
    profile_evidence_satisfied = False
    profile_evidence_fingerprint = None
    if not design_dry_run:
        if profile_evidence is None:
            raise ValueError(
                f"formal Phase-{phase.upper()} freeze requires --smoke-report-json "
                "covering every assigned batch/hardware profile"
            )
        validate_profile_evidence(
            profile_evidence,
            phase=phase,
            layout_id=str(hardware_layout.get("layout_id")),
            requirements=profile_requirements,
            contract_id=profile_contract_id,
        )
        profile_evidence_satisfied = True
        profile_evidence_fingerprint = hashlib.sha256(
            json.dumps(
                profile_evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    peptide_mean = (
        stats["peptide"]["train"]["input_tokens"]
        / stats["peptide"]["train"]["records"]
    )
    return {
        "metadata": {
            "schema_version": MATRIX_SCHEMA,
            "matrix_version": f"domainwise_{phase_name}_matrix_v1",
            "status": output_status,
            "is_runnable": not design_dry_run,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "phase_name": phase_name,
            "execution_phase": phase_spec["execution_phase"],
            "proposal_matrix_version": proposal["metadata"].get("matrix_version"),
            "proposal_status": proposal["metadata"].get("status"),
            "domain_split_manifest": str(manifest_path),
            "domain_split_semantic_contract": manifest_semantic_contract(manifest),
            "run_root": str(run_root),
            "experiment_name": experiment_name,
            "output_base": str(output_base),
            "rows": len(rows),
            "review_gate": {
                "required": bool(phase_spec["review_required"]),
                "satisfied": bool(review_satisfied),
                "review_json": str(review_path) if review_path is not None else None,
                "selection_source": selection_source,
                "test_data_accessed": review.get("test_data_accessed") if review else None,
                "review_semantic_fingerprint": (
                    hashlib.sha256(
                        json.dumps(
                            review,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    if review is not None
                    else None
                ),
            },
            "hardware_layout": {
                "schema_version": hardware_layout["schema_version"],
                "layout_id": hardware_layout.get("layout_id"),
                "devices": devices,
            },
            "physical_device_target_flops": physical_load,
            "worker_slot_target_flops": slot_load,
            "effective_batch_records": EFFECTIVE_BATCH_RECORDS,
            "training_control": FROZEN_TRAINING_CONTROL,
            "batch_profile_evidence_gate": {
                "required": True,
                "satisfied": profile_evidence_satisfied,
                "report_json": (
                    str(profile_evidence_path)
                    if profile_evidence_path is not None
                    else None
                ),
                "schema_version": PROFILE_EVIDENCE_SCHEMA,
                "profile_contract_id": profile_contract_id,
                "report_semantic_fingerprint": profile_evidence_fingerprint,
                "requirements": profile_requirements,
            },
            "peptide_batch_design": {
                "target_input_tokens_per_optimizer_step_approx": 125_000,
                "effective_batch_records": EFFECTIVE_BATCH_RECORDS["peptide"],
                "frozen_train_mean_input_tokens_per_record": peptide_mean,
                "expected_input_tokens_per_optimizer_step": peptide_mean
                * EFFECTIVE_BATCH_RECORDS["peptide"],
                "profile_policy": (
                    "24GB single: 3072x2 (<=6M), 1536x4 (7-31M), "
                    "1024x6 (32-41M); 48GB dual: 768x8"
                ),
                "launch_gate": "a real peptide full-optimizer-step smoke must pass",
            },
            "invariants": [
                "one process and one independently initialized model per run",
                "the frozen effective record batch does not change across hardware profiles",
                "target non-padding input tokens including EOS are the authoritative stop condition",
                "the final short record-aligned batch is retained with overshoot below context length",
                "BPE unfolding is disabled",
                "training reads train and final training-time evaluation reads validation only",
                "test remains reserved until the analysis configuration is frozen",
                "file hashes are provenance only; semantic manifest fields are acceptance gates",
            ],
        },
        "rows": rows,
    }


def main() -> None:
    default_proposal = (
        DEFAULT_DATA_ROOT / "reports" / "domainwise_scaling_matrix_proposal_v1.json"
    )
    default_manifest = (
        DEFAULT_DATA_ROOT / "processed" / "domain_split_manifests_v1" / "manifest.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--proposal-json", type=Path, default=default_proposal)
    parser.add_argument("--domain-split-manifest", type=Path, default=default_manifest)
    parser.add_argument("--review-json", type=Path, default=None)
    parser.add_argument(
        "--smoke-report-json",
        type=Path,
        default=None,
        help=(
            "Approved domainwise_batch_profile_evidence_v1 report. Required for "
            "every formal Phase-C/D freeze; previews expose the required contracts."
        ),
    )
    parser.add_argument(
        "--review-template-output",
        type=Path,
        default=None,
        help="For Phase-C planning, write a pending review contract beside the preview.",
    )
    parser.add_argument("--hardware-layout-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--design-dry-run",
        action="store_true",
        help="Write an explicitly non-runnable matrix preview; permits Phase C without review.",
    )
    args = parser.parse_args()

    phase_spec = PHASES[args.phase]
    proposal_path = args.proposal_json.expanduser().resolve()
    manifest_path = args.domain_split_manifest.expanduser().resolve()
    review_path = args.review_json.expanduser().resolve() if args.review_json else None
    profile_evidence_path = (
        args.smoke_report_json.expanduser().resolve()
        if args.smoke_report_json
        else None
    )
    layout_path = (
        args.hardware_layout_json.expanduser().resolve()
        if args.hardware_layout_json
        else None
    )
    run_root = args.run_root.expanduser().resolve()
    output_path = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else run_root
        / args.experiment_name
        / str(phase_spec["phase_name"])
        / ("design_dry_run_matrix.json" if args.design_dry_run else "matrix.json")
    )
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path is not None
        else None
    )
    profile_evidence = (
        json.loads(profile_evidence_path.read_text(encoding="utf-8"))
        if profile_evidence_path is not None
        else None
    )
    hardware_layout = (
        json.loads(layout_path.read_text(encoding="utf-8"))
        if layout_path is not None
        else default_hardware_layout()
    )
    matrix = build_phase_matrix(
        phase=args.phase,
        proposal=proposal,
        manifest=manifest,
        manifest_path=manifest_path,
        run_root=run_root,
        experiment_name=args.experiment_name,
        hardware_layout=hardware_layout,
        design_dry_run=bool(args.design_dry_run),
        review=review,
        review_path=review_path,
        profile_evidence=profile_evidence,
        profile_evidence_path=profile_evidence_path,
        warmup_ratio=float(args.warmup_ratio),
    )
    attach_matrix_identity(matrix)
    write_disposition = write_matrix(
        output_path, matrix, design_dry_run=bool(args.design_dry_run)
    )

    template_path = None
    if args.review_template_output is not None:
        if args.phase != "c":
            raise ValueError("--review-template-output is valid only for --phase c")
        c_rows = [
            row
            for row in proposal["rows"]
            if row.get("execution_phase") == PHASES["c"]["execution_phase"]
        ]
        template_path = args.review_template_output.expanduser().resolve()
        atomic_json(template_path, review_template(proposal, c_rows))

    print(
        json.dumps(
            {
                "matrix": str(output_path),
                "status": matrix["metadata"]["status"],
                "rows": len(matrix["rows"]),
                "phase": args.phase,
                "review_gate": matrix["metadata"]["review_gate"],
                "batch_profile_evidence_gate": matrix["metadata"][
                    "batch_profile_evidence_gate"
                ],
                "review_template": str(template_path) if template_path else None,
                "generation_id": matrix["metadata"]["generation_id"],
                "write_disposition": write_disposition,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
