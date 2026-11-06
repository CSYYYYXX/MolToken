#!/usr/bin/env python3
"""Aggregate the frozen three PepBench seeds, retaining per-seed metrics.

This only summarizes existing evaluations. It does not change their protocol,
promote partial scores to formal, or pool sequence-level pPPL standard errors
as if they were between-seed variation.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host

METRICS = {
    "validity": "counts.validity",
    "uniqueness": "counts.uniqueness_within_valid",
    "exact_length_rate": "counts.exact_length_rate",
    "novelty_sft_train": "novelty.sft_train.exact_novelty_within_valid",
    "nearest_sft_train": "novelty.sft_train.nearest_normalized_edit_distance.mean",
    "novelty_pep_source_only": "novelty.pepbenchmark_pretraining_source_only.exact_novelty_within_valid",
    "nearest_pep_source_only": "novelty.pepbenchmark_pretraining_source_only.nearest_normalized_edit_distance.mean",
    "diversity": "diversity.mean_normalized_edit_distance",
    "esm_fbd": "fbd",
    "esm_ppl": "esm_ppl.mean",
    "precision": "embedding_manifold.precision",
    "recall": "embedding_manifold.recall",
    "aa_jsd_bits": "marginal_distribution.aa_composition_jsd_bits",
    "length_jsd_bits": "marginal_distribution.length_jsd_bits",
    "embedding_coverage": "embedding_coverage_over_requested",
    "ppl_coverage": "ppl_coverage_over_requested",
}


def summarize(rows):
    rows = sorted(rows, key=lambda r: r["seed"])
    if [r["seed"] for r in rows] != [42, 43, 44]:
        raise ValueError("exactly one completed evaluation for each seed 42/43/44 is required")
    if any(not r["complete"] for r in rows):
        raise ValueError("an evaluator did not complete")
    if any(r["counts"]["requested"] != 10000 for r in rows):
        raise ValueError("three-seed benchmark summary requires 10000 raw slots per seed")
    if len({r["protocol"] for r in rows}) != 1:
        raise ValueError("mixed evaluation protocols")
    # Compare scientific feature settings, not paths, device or incidental batch.
    esm_keys = ("model", "layer", "embedding_dimension", "pooling", "dtype", "attention", "ppl_masking")
    for row in rows[1:]:
        if any(row["esm"][key] != rows[0]["esm"][key] for key in esm_keys):
            raise ValueError("mixed ESM feature definitions")
        if row["embedding_manifold"]["k"] != rows[0]["embedding_manifold"]["k"]:
            raise ValueError("mixed PR neighborhood definitions")
    metrics = {}
    for name, path in METRICS.items():
        values = []
        for row in rows:
            value = row
            for key in path.split("."):
                value = value[key]
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name}: non-finite seed value")
            values.append(value)
        defined = sum(v is not None for v in values)
        metrics[name] = {"per_seed": dict(zip((42, 43, 44), values, strict=True)),
                         "defined_seeds": defined,
                         "mean": mean(values) if defined == 3 else None,
                         "std_between_seeds_ddof1": stdev(values) if defined == 3 else None}
    return {
        "complete": True, "formal": all(r["formal"] for r in rows),
        "protocol": rows[0]["protocol"], "seeds": [42, 43, 44], "raw_slots_per_seed": 10000,
        "official_generation_leaderboard": False, "pretraining_exposed": True,
        "full_pretraining_novelty_available": all(r["full_pretraining_novelty_available"] for r in rows),
        "partial_result_reasons": sorted(set(reason for r in rows for reason in r["partial_result_reasons"])),
        "reference_paths_per_seed": {r["seed"]: r["reference_paths"] for r in rows},
        "sampling_schedules": {r["seed"]: r["sampling_schedule"] for r in rows},
        "esm_definition": {key: rows[0]["esm"][key] for key in esm_keys},
        "aggregation": "arithmetic mean and sample standard deviation of the three seed-level metrics; undefined seed metrics are not dropped",
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("PepBench three-seed metric summary")
    result = summarize([json.loads(path.read_text()) for path in args.summaries])
    result["input_summaries"] = [str(path) for path in args.summaries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
