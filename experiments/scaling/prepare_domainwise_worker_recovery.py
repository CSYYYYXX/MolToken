#!/usr/bin/env python3
"""Prepare the explicitly stopped 2026-09-06 2g_b runs; launch nothing.

Keep the original attempts and queue plans intact. The R008 recovery uses a
separate root because the surviving queue still owns its original reservation.
The read-only results plan list must replace R008 with this recovery plan.
"""
from copy import deepcopy
import argparse
from pathlib import Path

from run_domainwise_cluster_completion import atomic_json, load_json, now, validate_manifest
from collect_domainwise_cluster_results import SCIENTIFIC_INTEGER_FIELDS, SCIENTIFIC_TEXT_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new-point-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    phase_c_path = base / "phase_c/cluster_relaunch_plan_v2_mem70.json"
    round8_path = base / "phase_supplement/round_008/plan.json"
    phase_c, round8 = load_json(phase_c_path), load_json(round8_path)
    sources = {r["run_id"]: (r, phase_c_path) for r in phase_c["rows"]}
    sources.update({r["run_id"]: (r, round8_path) for r in round8["rows"]})
    selected = [
        ("protein_C3_0p017b_s42", 2, 0, 1280, 2),
        ("rna_C3_0p041b_s42", 2, 0, 1536, 4),
        ("small_molecule_C3_0p041b_s42", 1, 1, 6144, 2),
    ]
    rows = []
    for run_id, previous_number, device, batch, accum in selected:
        original, source = sources[run_id]
        old_attempt = Path(original["attempt_dir"])
        if int(original["attempt_number"]) != previous_number:
            raise ValueError(f"unexpected interrupted attempt: {run_id}")
        if not (old_attempt / "run_config.json").is_file():
            raise FileNotFoundError(old_attempt / "run_config.json")
        if (Path(original["run_dir"]) / "complete.json").exists():
            raise ValueError(f"run completed; do not restart: {run_id}")
        row = deepcopy(original)
        if run_id == "small_molecule_C3_0p041b_s42":
            row["run_dir"] = str(base / "phase_supplement/round_008_recovery/runs" / run_id)
        row.update(
            attempt_number=previous_number + 1,
            attempt_name=f"attempt_{previous_number + 1:03d}",
            interrupted_from_attempt_dir=str(old_attempt),
            recovery_source_plan=str(source),
            recovery_device_index=device,
            global_gpu_index=6 + device,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=accum,
            batch_override_applied=True,
            batch_adjustment_reason="recovery_memory_candidate_same_effective_batch_requires_new_50_step_profile",
            recovery_reason="worker_stopped_20260906T064100Z_restart_from_zero",
        )
        row["attempt_dir"] = str(Path(row["run_dir"]) / row["attempt_name"])
        row["previous_attempt_numbers"] = sorted(set(
            original.get("previous_attempt_numbers", []) + [previous_number]))
        if Path(row["attempt_dir"]).exists():
            raise FileExistsError(row["attempt_dir"])
        for field in (*SCIENTIFIC_TEXT_FIELDS, *SCIENTIFIC_INTEGER_FIELDS):
            if row[field] != original[field]:
                raise ValueError(f"recovery changes {field}: {run_id}")
        if batch * accum != int(row["effective_batch_records"]):
            raise ValueError(f"effective batch changed: {run_id}")
        if row["init_checkpoint"] is not None or row["restart_from_step"] != 0:
            raise ValueError(f"recovery must start from zero: {run_id}")
        rows.append(row)
    plan = {
        "schema_version": "domainwise_explicit_worker_recovery_v1",
        "created_at": now(), "row_count": len(rows), "rows": rows,
        "domain_split_manifest": phase_c["domain_split_manifest"],
        "stopped_worker_process": "ws-788c577781500897-worker-mdbbb",
        "confirmed_stopped_at": "2026-09-06T06:41:00Z",
        "recovery_policy": "fresh initialization; preserve interrupted attempts; no resumed weights",
        "replaces_results_plan": str(round8_path),
    }
    new_point = load_json(args.new_point_plan)
    if new_point["domain_split_manifest"] != plan["domain_split_manifest"]:
        raise ValueError("new point and recovery manifests differ")
    [extra] = deepcopy(new_point["rows"])
    if extra["run_id"] != "rna_C1_0p041b_s42":
        raise ValueError("unexpected additional experiment")
    extra.update(recovery_device_index=1, global_gpu_index=7,
                 per_device_train_batch_size=1536, gradient_accumulation_steps=4,
                 batch_adjustment_reason="recovery_memory_candidate_same_effective_batch_requires_new_50_step_profile")
    if extra["effective_batch_records"] != 1536 * 4:
        raise ValueError("new point effective batch changed")
    bundle = {**plan, "row_count": 4, "rows": [*rows, extra],
              "profile_required": "all four exact batch shapes, 50 optimizer steps, before formal start"}
    validate_manifest(Path(plan["domain_split_manifest"]), bundle["rows"])
    output.mkdir(parents=True)
    atomic_json(output / "recovery_plan.json", plan)
    atomic_json(output / "execution_plan.json", bundle)
    print({"status": "prepared_not_launched", "output": str(output),
           "recovery_runs": [r["run_id"] for r in rows], "additional_run": extra["run_id"]}, flush=True)


if __name__ == "__main__":
    main()
