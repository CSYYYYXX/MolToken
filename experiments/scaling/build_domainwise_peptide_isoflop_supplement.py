#!/usr/bin/env python3
"""Worker-side, single-point supplements within the existing peptide protocol.

Uses only registered model shapes. Does not submit work or alter the pilot.
The default cap remains four token-epochs; C1/C2 have five-epoch extensions.
The 2026-09-09 bold C2 left search uses an explicitly recorded ten-epoch
ceiling to cover the 3.06M/2.42M points at unchanged compute.
Historical fixed-D observations remain separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_domainwise_bracketing_supplement_plan import _model_specs, _scheduler_steps
from build_domainwise_cluster_completion_plan import atomic_json, now
from collect_domainwise_cluster_results import read_plan_list_files
from domainwise_peptide_isoflop import BUDGETS, EFFECTIVE_BATCH, PROTOCOL, collect, curves_for, make_plan
from run_domainwise_cluster_completion import validate_manifest
from run_domainwise_supplement_queue import validate_supplement_plan


def make_supplement(manifest, manifest_path, root, completed, pending, budget, model_id, reason,
                    *, maximum_epoch_fraction=4):
    allowed_caps = {"C0": (4,), "C1": (4, 5), "C2": (4, 5, 10)}
    if maximum_epoch_fraction not in allowed_caps[budget]:
        raise ValueError("only C1/C2 have max5 extensions and C2 has a max10 left-search extension; default cap is four")
    spec = _model_specs()[model_id]
    n = int(spec["parameter_count"])
    for row in completed + pending:
        if row["budget"] == budget and (row["model_id"] == model_id or int(row["parameter_count"]) == n):
            raise ValueError(f"duplicate peptide IsoFLOP coordinate: {budget}/{model_id}")
    curve = next(c for c in curves_for(completed, pending) if c["budget"] == budget)
    if curve["pending"] or curve["analysis"] is None:
        raise ValueError(f"wait for the current curve's full validation: {budget}")
    if curve["visual_category"] == "accepted":
        raise ValueError(f"curve already has the requested bracketing support: {budget}")

    plan = make_plan(manifest, manifest_path, root)
    row = dict(next(r for r in plan["rows"] if r["budget"] == budget))
    tokens = round(BUDGETS[budget] / (6 * n))
    epoch_tokens = row["train_input_tokens_per_epoch"]
    if tokens > maximum_epoch_fraction * epoch_tokens:
        cap_name = {4: "four", 5: "five", 10: "ten"}[maximum_epoch_fraction]
        raise ValueError(f"{cap_name}-token-epoch limit exceeded: {budget}/{model_id}")
    steps = _scheduler_steps(tokens, row["train_records"], epoch_tokens, EFFECTIVE_BATCH)
    run_id = f"peptide_Iso{budget}_{model_id}_s42"
    run_dir = root / "runs" / run_id
    row.update({k: int(spec[k]) for k in (
        "parameter_count", "non_embedding_parameter_count", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")})
    row.update(model_id=model_id, run_id=run_id, run_dir=str(run_dir),
               attempt_dir=str(run_dir / "attempt_001"), priority_rank=1,
               target_input_tokens_with_eos=tokens, target_epoch_fraction=tokens / epoch_tokens,
               maximum_epoch_fraction=maximum_epoch_fraction,
               scheduler_total_steps=steps, warmup_steps=max(1, round(steps * row["warmup_ratio"])),
               approval_source="user_goal", supplement_reason=reason,
               source_curve_analysis=curve["analysis"],
               source_curve_run_ids=[r["run_id"] for r in curve["points"]])
    plan.update(rows=[row], row_count=1, approval_source="user_goal",
                analysis_note="Existing model registry and peptide IsoFLOP protocol; keep all completed observations.")
    if maximum_epoch_fraction == 5:
        approval_date = {"C1": "20260908", "C2": "20260909"}[budget]
        extension = f"peptide_{budget}_max5_token_epochs_{approval_date}"
        row.update(experiment_extension=extension, approval_source=f"user_{approval_date}")
        plan.update(experiment_extension=extension, approval_source=f"user_{approval_date}",
                    analysis_note=f"{budget}-only max5-token-epoch extension; unchanged 6ND, batch, split and codebook. Retain all prior observations.")
    elif maximum_epoch_fraction == 10:
        extension = "peptide_C2_bold_left_max10_token_epochs_20260909"
        row.update(experiment_extension=extension, approval_source="user_bold_left_20260909",
                   extension_note="Operational ceiling for the disclosed 7.67/9.71-epoch left probes; unchanged C2 FLOPs, not a global epoch change.")
        plan.update(experiment_extension=extension, approval_source="user_bold_left_20260909",
                    analysis_note="C2 bold left search at unchanged 6ND; named smaller models require more token epochs. Keep old observations and all other curve budgets unchanged.")
    elif int(spec["hidden_size"]) == 64:
        row.update(experiment_extension="peptide_C0_h64_head128_20260909", approval_source="user_20260909")
        plan.update(experiment_extension=row["experiment_extension"], approval_source="user_20260909")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-list", type=Path, required=True)
    parser.add_argument("--budget", choices=BUDGETS, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--maximum-epoch-fraction", type=int, choices=(4, 5, 10), default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    paths = read_plan_list_files([args.plan_list])
    if not paths:
        raise ValueError("peptide plan list is empty")
    plans = [json.loads(p.read_text()) for p in paths]
    manifest_path = Path(plans[0]["domain_split_manifest"])
    manifest = json.loads(manifest_path.read_text())
    completed, pending = [], []
    for plan in plans:
        if plan["scientific_protocol"] != PROTOCOL or Path(plan["domain_split_manifest"]) != manifest_path:
            raise ValueError("source plans must share the approved peptide protocol and manifest")
        done, missing = collect(plan, manifest)
        completed.extend(done)
        pending.extend(missing)
    plan = make_supplement(manifest, manifest_path, root, completed, pending,
                           args.budget, args.model_id, args.reason,
                           maximum_epoch_fraction=args.maximum_epoch_fraction)
    plan["source_plans"] = [str(p) for p in paths]
    validate_supplement_plan(plan, root / "plan.json")
    validate_manifest(manifest_path, plan["rows"])
    root.mkdir(parents=True)
    atomic_json(root / "plan.json", plan)
    print(json.dumps({"at": now(), "status": "prepared_not_launched", "plan": str(root / "plan.json"),
                      "row": plan["rows"][0]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
