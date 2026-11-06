#!/usr/bin/env python3
"""Add released-code and paper-convention inner-TM to native DPLM folds."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.dplm_tm import run_tmalign, score_inner_tm
from pipeline.runtime_host import require_compute_host

TASK = "dplm_length_conditioned_protein"
CONDITIONS = ("100", "200", "300", "400")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-results", type=Path, required=True)
    parser.add_argument("--tmalign", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute_host("DPLM native inner-TM evaluation")
    fold_summary = json.loads((args.fold_results / "summary.json").read_text())
    if fold_summary["task"] != TASK or not fold_summary["plddt_scoring_complete"]:
        raise ValueError("completed native DPLM fold results required")
    rows = [json.loads(line) for line in (args.fold_results / "per_slot.jsonl").read_text().splitlines()]
    if len(rows) != fold_summary["requested"] or sum(row["folded"] for row in rows) != fold_summary["folded"]:
        raise ValueError("raw folded-slot accounting differs from source summary")
    summary, per_slot, pairs = score_inner_tm(rows, partial(run_tmalign, args.tmalign),
                                             conditions=CONDITIONS, workers=args.workers)
    summary.update(task=TASK, fold_results=str(args.fold_results), fold_provenance=fold_summary,
                   evaluator={"tmalign": str(args.tmalign), "workers": args.workers,
                       "source": "bytedance/dplm analysis/TMalign.cpp (Version 20220412), cal_tmscore.py and uncond_analysis.ipynb"})
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in (("per_slot.jsonl", per_slot), ("unique_pair_scores.jsonl", pairs)):
        with (args.output / name).open("x") as handle:
            for row in records:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    with (args.output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"DPLM_INNER_TM_COMPLETE {args.output}; pdb-TM still required", flush=True)


if __name__ == "__main__":
    main()
