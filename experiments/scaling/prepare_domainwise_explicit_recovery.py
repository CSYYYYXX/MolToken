#!/usr/bin/env python3
"""Prepare a named slice of confirmed interrupted runs; never launch trainers.

The caller confirms that the old worker is gone. Keep its attempts, copy the
scientific configuration, and allocate fresh attempt paths. An independent
joint memory profile is required before running this plan. Explicit microbatch
overrides may only change the microbatch/accumulation factorization, not the
effective batch, token budget, or scientific coordinates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from build_domainwise_cluster_relaunch_plan import existing_attempt_numbers
from run_domainwise_cluster_completion import atomic_json, load_json, now, validate_manifest
from run_domainwise_worker_recovery import validate_recovery_rows


def prepare_plan(selection):
    rows, source_paths = [], []
    manifest_path = None
    for item in selection["runs"]:
        source_path = Path(item["source_plan"]).resolve()
        source = load_json(source_path)
        current_manifest = Path(source["domain_split_manifest"]).resolve()
        if manifest_path is None:
            manifest_path = current_manifest
        elif current_manifest != manifest_path:
            raise ValueError("recovery sources use different domain split manifests")
        matches = [r for r in source["rows"] if r["run_id"] == item["run_id"]]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one source row: {item['run_id']}")
        row = deepcopy(matches[0])
        run_dir = Path(row["run_dir"]).resolve()
        previous = Path(row["attempt_dir"]).resolve()
        if (run_dir / "complete.json").exists() or (previous / "summary.json").exists():
            raise ValueError(f"completed output requires review, not restart: {previous}")
        config = load_json(previous / "run_config.json")
        if (config["args"]["batch_size"] != row["per_device_train_batch_size"]
                or config["args"]["grad_accum"] != row["gradient_accumulation_steps"]):
            raise ValueError(f"source plan does not match actual interrupted batch: {previous}")
        numbers = existing_attempt_numbers(run_dir)
        if not numbers or max(numbers) != row["attempt_number"]:
            raise ValueError(f"a newer or untracked attempt needs review: {run_dir}")
        number = max(numbers) + 1
        attempt = run_dir / f"attempt_{number:03d}"
        row.update(
            attempt_number=number, attempt_name=attempt.name, attempt_dir=str(attempt),
            previous_attempt_numbers=numbers,
            interrupted_from_attempt_dir=str(previous), recovery_source_plan=str(source_path),
            recovery_device_index=item["device_index"], global_gpu_index=item["device_index"],
            source_status="interrupted_restart_required",
            recovery_reason=selection["interruption_evidence"],
        )
        if "per_device_train_batch_size" in item or "gradient_accumulation_steps" in item:
            row.update(
                interrupted_microbatch=row["per_device_train_batch_size"],
                interrupted_gradient_accumulation=row["gradient_accumulation_steps"],
                per_device_train_batch_size=item["per_device_train_batch_size"],
                gradient_accumulation_steps=item["gradient_accumulation_steps"],
                batch_override_applied=True,
                batch_adjustment_reason=item["batch_adjustment_reason"],
            )
        rows.append(row)
        if str(source_path) not in source_paths:
            source_paths.append(str(source_path))
    plan = {
        "schema_version": "domainwise_explicit_worker_recovery_v1",
        "created_at": now(), "row_count": len(rows), "rows": rows,
        "domain_split_manifest": str(manifest_path), "source_plans": source_paths,
        "lost_worker_process": selection["lost_worker_process"],
        "confirmed_interrupted_at": selection["confirmed_interrupted_at"],
        "interruption_evidence": selection["interruption_evidence"],
        "recovery_policy": "fresh initialization; preserve all previous attempts; unchanged scientific configuration and effective batch; explicit microbatch changes require a new profile",
        "profile_required": "50 optimizer steps for this concurrent mixture before formal launch",
    }
    validate_recovery_rows(plan, {item["device_index"] for item in selection["runs"]})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if args.out_json.exists():
        raise FileExistsError(args.out_json)
    plan = prepare_plan(load_json(args.selection_json))
    validate_manifest(Path(plan["domain_split_manifest"]), plan["rows"])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_json, plan)
    print({"status": "prepared_not_launched", "out_json": str(args.out_json),
           "runs": [r["run_id"] for r in plan["rows"]]}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
