#!/usr/bin/env python3
"""Build/watch the user-approved peptide IsoFLOP pilot; reuse the existing trainer.

The 30 historical fixed-D observations remain separate. This file does not
allocate GPUs or submit training; the existing atomic supplement queue does.
Run both subcommands on an allocated worker.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from build_domainwise_bracketing_supplement_plan import (
    _manifest_domain_metadata, _model_specs, _scheduler_steps,
)
from build_domainwise_cluster_completion_plan import (
    CONTEXT_LENGTH, VOCAB_SIZE, WARMUP_RATIO, atomic_json, now,
)
from run_domainwise_cluster_completion import validate_completion, validate_manifest
from run_domainwise_supplement_queue import validate_supplement_plan
from analyze_domainwise_adaptive_curves import analyze_adaptive_curve

PROTOCOL = "peptide_isoflop_b1024_v1"
BUDGETS = {"C0": 600_000_000_000_000, "C1": 1_200_000_000_000_000,
           "C2": 2_400_000_000_000_000}
MODELS = ("0p006b", "0p009b", "0p014b", "0p022b", "0p041b")
EFFECTIVE_BATCH = 1024


def make_plan(manifest, manifest_path, root):
    records, epoch_tokens = _manifest_domain_metadata(manifest, "peptide")
    specs = _model_specs()
    rows = []
    for budget, flops in BUDGETS.items():
        for model_id in MODELS:
            spec = specs[model_id]
            n = int(spec["parameter_count"])
            tokens = round(flops / (6 * n))
            if tokens > 4 * epoch_tokens:
                raise ValueError(f"four-token-epoch limit exceeded: {budget}/{model_id}")
            steps = _scheduler_steps(tokens, records, epoch_tokens, EFFECTIVE_BATCH)
            run_id = f"peptide_Iso{budget}_{model_id}_s42"
            run_dir = root / "runs" / run_id
            rows.append({
                "phase": "supplement", "purpose": "adaptive_bracketing_supplement",
                "scientific_protocol": PROTOCOL, "priority_tier": "C",
                "priority_rank": len(rows) + 1, "domain": "peptide", "budget": budget,
                "run_id": run_id, "model_id": model_id,
                **{k: int(spec[k]) for k in (
                    "parameter_count", "non_embedding_parameter_count", "hidden_size",
                    "intermediate_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim")},
                "target_input_tokens_with_eos": tokens, "target_flops_6nd": flops,
                "target_epoch_fraction": tokens / epoch_tokens,
                "maximum_epoch_fraction": 4, "train_records": records,
                "train_input_tokens_per_epoch": epoch_tokens,
                "per_device_train_batch_size": EFFECTIVE_BATCH,
                "gradient_accumulation_steps": 1, "effective_batch_records": EFFECTIVE_BATCH,
                "scheduler_total_steps": steps, "warmup_ratio": WARMUP_RATIO,
                "warmup_steps": max(1, round(steps * WARMUP_RATIO)),
                "seed": 42, "vocab_size": VOCAB_SIZE, "context_length": CONTEXT_LENGTH,
                "world_size": 1, "random_initialization": True, "init_checkpoint": None,
                "restart_from_step": 0, "source_status": "adaptive_supplement",
                "attempt_number": 1, "attempt_name": "attempt_001",
                "run_dir": str(run_dir), "attempt_dir": str(run_dir / "attempt_001"),
                "source_batch_profile": "peptide_isoflop_1024x1_candidate",
                "need_profile": True, "profile_optimizer_steps": 50,
                "profile_status": "required_before_formal_training",
                "claim_policy": "worker_empty_slot", "approval_source": "user_20260906",
            })
    return {
        "schema_version": "domainwise_bracketing_supplement_plan_v1",
        "created_at": now(), "status": "ready_for_50_step_profile",
        "phase": "supplement", "purpose": "adaptive_bracketing_supplement",
        "scientific_protocol": PROTOCOL, "phase_root": str(root),
        "domain_split_manifest": str(manifest_path), "row_count": len(rows), "rows": rows,
        "approval_source": "user_20260906", "test_accessed": False,
        "analysis_note": "Separate from fixed-D/batch6144 history; full validation only; no early stopping.",
    }


def reuse_audit(plan, historical):
    old = [r for r in historical if r["domain"] == "peptide"]
    if not old:
        raise ValueError("historical table contains no peptide observations")
    for r in old:
        if r["status"] != "complete" or r["validation_complete_view"] != "true" or r["metric_split"] != "validation":
            raise ValueError(f"historical point lacks full validation: {r['run_id']}")
    comparisons = []
    for row in plan["rows"]:
        same_n = [r for r in old if int(r["parameter_count"]) == row["parameter_count"]]
        if not same_n:
            raise ValueError(f"historical model missing: {row['model_id']}")
        nearest = min(same_n, key=lambda r: abs(int(r["target_input_tokens_with_eos"]) - row["target_input_tokens_with_eos"]))
        same_tokens = int(nearest["target_input_tokens_with_eos"]) == row["target_input_tokens_with_eos"]
        same_batch = int(nearest["effective_batch_records"]) == row["effective_batch_records"]
        same_seed = int(nearest["seed"]) == row["seed"]
        comparisons.append({
            "new_run_id": row["run_id"], "nearest_historical_run_id": nearest["run_id"],
            "historical_validation_loss": float(nearest["validation_loss"]),
            "historical_target_tokens": int(nearest["target_input_tokens_with_eos"]),
            "new_target_tokens": row["target_input_tokens_with_eos"],
            "relative_compute_gap": abs(int(nearest["target_input_tokens_with_eos"]) / row["target_input_tokens_with_eos"] - 1),
            "same_target_tokens": same_tokens, "same_effective_batch": same_batch,
            "direct_candidate": same_tokens and same_batch and same_seed,
        })
    return {"historical_rows": len(old), "comparisons": comparisons,
            "direct_candidates": sum(c["direct_candidate"] for c in comparisons),
            "preserved_historical_observations": old,
            "note": "A direct candidate still requires matching split/model/schedule provenance before reuse. No weights are reused."}


def collect(plan, manifest):
    completed, pending = [], []
    for row in plan["rows"]:
        marker = Path(row["run_dir"]) / "complete.json"
        if not marker.is_file():
            pending.append(row)
            continue
        record = json.loads(marker.read_text())
        if (record["status"] != "complete" or record["run_id"] != row["run_id"]
                or Path(record["attempt_dir"]) != Path(row["attempt_dir"])):
            raise ValueError(f"completion marker routing differs: {marker}")
        summary = validate_completion(attempt_dir=Path(row["attempt_dir"]), row=row, manifest=manifest)
        completed.append({**row, "validation_loss": summary["final_validation"]["loss"],
                          "actual_input_tokens": summary["actual_input_tokens"],
                          "actual_flops_6nd": summary["actual_flops_6nd"],
                          "actual_optimizer_steps": summary["global_step"],
                          "actual_epoch_fraction": summary["actual_epoch_fraction"],
                          "final_validation": summary["final_validation"]})
    return completed, pending


def curves_for(completed, pending):
    from plot_domainwise_completed_snapshot import classify
    curves = []
    for budget, flops in BUDGETS.items():
        points = sorted([r for r in completed if r["budget"] == budget], key=lambda r: r["parameter_count"])
        missing = sorted([r for r in pending if r["budget"] == budget], key=lambda r: r["parameter_count"])
        analysis = analyze_adaptive_curve(points) if len(points) >= 5 else None
        curves.append({"domain": "peptide", "budget": budget, "target_flops": flops,
                       "points": points, "pending": missing, "analysis": analysis,
                       "visual_category": classify(analysis) if analysis else "incomplete"})
    return curves


def publish(plan, completed, pending, output):
    from plot_domainwise_comparison import comparison_panel
    from plot_domainwise_completed_snapshot import DOMAIN_ZH, plt, save_figure
    # Extend the existing plot labels only in this dedicated process. Do not
    # edit the running four-domain/fixed-D plotter or mix scientific protocols.
    DOMAIN_ZH["peptide"] = "多肽 IsoFLOP"
    curves = curves_for(completed, pending)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": ["Droid Sans Fallback", "DejaVu Sans"], "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(11.5, 6.4))
        comparison_panel(ax, curves)
        if not completed:
            ax.set_yticks([])
            ax.text(.5, .5, "等待完整 validation", transform=ax.transAxes, ha="center")
        fig.subplots_adjust(left=.1, right=.98, bottom=.2, top=.77)
        fig.text(.1, .045, "固定 FLOPs · 有效 batch 1024 · seed 42 · 至多 4 token-epoch\n不足五点不拟合；全部观测参与拟合；历史等数据量结果不混入。", fontsize=10)
        save_figure(fig, output, "peptide_isoflop")
    fields = ("run_id", "budget", "model_id", "parameter_count", "target_flops_6nd",
              "actual_flops_6nd", "actual_input_tokens", "actual_epoch_fraction",
              "actual_optimizer_steps", "validation_loss")
    temporary = output / "completed_runs.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
    temporary.replace(output / "completed_runs.csv")
    report = {"updated_at": now(), "scientific_protocol": PROTOCOL,
              "completed": len(completed), "pending": len(pending),
              "accepted_curves": sum(c["visual_category"] == "accepted" for c in curves),
              "curves": curves, "test_accessed": False, "worker_release_authorized": False}
    atomic_json(output / "snapshot.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--historical-csv", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    watch = sub.add_parser("watch")
    watch.add_argument("--plan-json", type=Path, required=True)
    watch.add_argument("--output-dir", type=Path, required=True)
    watch.add_argument("--interval-seconds", type=int, default=60)
    watch.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        root, manifest_path = args.output_root.resolve(), args.manifest.resolve()
        if root.exists():
            raise FileExistsError(root)
        manifest = json.loads(manifest_path.read_text())
        plan = make_plan(manifest, manifest_path, root)
        validate_supplement_plan(plan, root / "plan.json")
        validate_manifest(manifest_path, plan["rows"])
        with args.historical_csv.open(newline="") as stream:
            audit = reuse_audit(plan, list(csv.DictReader(stream)))
        if audit["direct_candidates"]:
            raise ValueError("Historical direct candidates need provenance review before submitting duplicates")
        root.mkdir(parents=True)
        atomic_json(root / "reuse_audit.json", audit)
        atomic_json(root / "plan.json", plan)
        print(json.dumps({"status": "prepared_not_launched", "rows": len(plan["rows"]),
                          "historical_rows": audit["historical_rows"], "direct_candidates": audit["direct_candidates"]}), flush=True)
        return
    if args.interval_seconds < 30:
        parser.error("monitor interval must be at least 30 seconds")
    plan = json.loads(args.plan_json.read_text())
    if plan["scientific_protocol"] != PROTOCOL:
        raise ValueError("not the peptide IsoFLOP protocol")
    manifest = json.loads(Path(plan["domain_split_manifest"]).read_text())
    previous = None
    while True:
        completed, pending = collect(plan, manifest)
        signature = [(r["run_id"], r["validation_loss"]) for r in completed]
        if signature != previous:
            report = publish(plan, completed, pending, args.output_dir)
            print(json.dumps({"event": "updated", **{k: report[k] for k in ("updated_at", "completed", "pending", "accepted_curves")}}), flush=True)
            previous = signature
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
