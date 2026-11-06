#!/usr/bin/env python3
"""Build an isolated, non-runnable RNA59 row for the batch-tuning driver.

This is a hardware diagnostic, not a Phase-C placement decision or formal
matrix. It reads only existing preview/proposal JSON metadata, never dataset
payloads or evaluation metrics, and refuses an already-existing output root.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from finalize_domainwise_phase_matrix import (
    assign_rows_to_hardware,
    attach_matrix_identity,
    atomic_json,
    required_profile_evidence,
    validate_hardware_layout,
)
from run_domainwise_batch_profile_smoke import validate_design_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preview", type=Path, required=True)
    parser.add_argument("--proposal-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    source_path = args.source_preview.expanduser().resolve()
    proposal_path = args.proposal_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"diagnostic output root already exists: {output_root}")
    if args.physical_gpu < 0:
        raise ValueError("negative physical GPU")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_design_matrix(source, "c")
    if output_root.is_relative_to(Path(source["metadata"]["output_base"]).resolve()):
        raise ValueError("diagnostic must live outside the formal phase directory")
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    if (
        proposal["metadata"]["matrix_version"]
        != source["metadata"]["proposal_matrix_version"]
    ):
        raise ValueError("source preview and frozen proposal versions differ")
    matches = [spec for spec in proposal["model_specs"] if spec["model_id"] == "0p059b"]
    if len(matches) != 1:
        raise ValueError("proposal must contain exactly one frozen 0p059b model")
    spec = matches[0]
    expected = {
        "parameter_count": 59_162_624,
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen RNA59 parameter count/shape differs from the approved probe")
    candidates = [row for row in source["rows"] if row["domain"] == "rna" and row["budget_label"] == "C4"]
    if not candidates:
        raise ValueError("source preview has no RNA C4 representative")
    base = max(candidates, key=lambda row: int(row["parameter_count"]))
    row = copy.deepcopy(base)
    row.update(spec)
    parameters = int(spec["parameter_count"])
    flops = float(row["target_flops_6nd"])
    target_tokens = max(1, round(flops / (6 * parameters)))
    train_tokens = int(row["train_input_tokens_per_epoch"])
    run_id = "diagnostic_rna_C4_0p059b_s42"
    row.update(
        run_id=run_id,
        run_name=run_id,
        out_dir=str(output_root / "not_formal" / run_id),
        matrix_status="design_dry_run_not_runnable",
        target_input_tokens_with_eos=target_tokens,
        target_flops_after_integer_token_rounding=6 * parameters * target_tokens,
        target_flops_relative_rounding_error=(6 * parameters * target_tokens - flops) / flops,
        target_epoch_fraction=target_tokens / train_tokens,
        minimum_parameter_count_four_epoch=flops / (24 * train_tokens),
        actual_multiplier_vs_prior=parameters / float(row["prior_parameter_count"]),
        four_epoch_compliant=target_tokens / train_tokens <= 4,
        diagnostic_only=True,
    )
    if not row["four_epoch_compliant"]:
        raise ValueError("RNA59 diagnostic row exceeds the frozen four-epoch cap")
    layout = {
        "schema_version": "domainwise_hardware_layout_v1",
        "layout_id": f"diagnostic_rna59_local_gpu{args.physical_gpu}_dual",
        "devices": [{"host_id": "local", "gpu_id": args.physical_gpu, "memory_gib": 48, "concurrent_slots": 2}],
    }
    devices = validate_hardware_layout(layout)
    physical_load, slot_load = assign_rows_to_hardware([row], devices)
    row["warmup_steps"] = max(1, round(row["scheduler_total_steps"] * row["warmup_ratio"]))
    requirements, contract_id = required_profile_evidence([row])
    diagnostic = {"metadata": copy.deepcopy(source["metadata"]), "rows": [row]}
    metadata = diagnostic["metadata"]
    metadata.update(
        phase_name="phase_c_diagnostic",
        matrix_version="domainwise_rna59_batch_diagnostic_v1",
        diagnostic_only=True,
        rows=1,
        experiment_name=output_root.name,
        run_root=str(output_root.parent),
        output_base=str(output_root),
        hardware_layout={**layout, "devices": devices},
        physical_device_target_flops=physical_load,
        worker_slot_target_flops=slot_load,
        review_gate={"required": True, "satisfied": False, "review_json": None, "selection_source": "diagnostic_frozen_model_shape_only_not_phase_c_approval", "test_data_accessed": False},
        batch_profile_evidence_gate={"required": True, "satisfied": False, "report_json": None, "schema_version": "domainwise_batch_profile_evidence_v1", "profile_contract_id": contract_id, "report_semantic_fingerprint": None, "requirements": requirements},
        diagnostic_provenance={
            "source_preview": str(source_path),
            "source_preview_generation_id": source["metadata"]["generation_id"],
            "source_row_run_id": base["run_id"],
            "frozen_proposal": str(proposal_path),
            "frozen_model_spec": spec,
            "purpose": "real_train_only_batch_memory_throughput_thermal_probe",
            "formal_campaign_points_added": 0,
            "validation_evaluation_enabled": False,
            "test_data_accessed": False,
        },
    )
    if row["effective_batch_records"] != 6_144:
        raise ValueError("diagnostic changed frozen RNA effective batch")
    if not math.isclose(row["expected_input_tokens_per_optimizer_step"], train_tokens / row["train_records"] * 6_144):
        raise ValueError("diagnostic token accounting differs from frozen RNA train view")
    attach_matrix_identity(diagnostic)
    validate_design_matrix(diagnostic, "c")
    path = output_root / "diagnostic_design.json"
    atomic_json(path, diagnostic)
    print(json.dumps({"path": str(path), "generation_id": metadata["generation_id"], "rows": 1, "diagnostic_only": True, "is_runnable": metadata["is_runnable"], "model_id": row["model_id"], "parameter_count": row["parameter_count"], "profile_contract_id": contract_id}))


if __name__ == "__main__":
    main()
