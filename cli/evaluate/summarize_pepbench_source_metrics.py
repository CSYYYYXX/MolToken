#!/usr/bin/env python3
"""Aggregate completed conservative-AA source metrics, not a formal leaderboard."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host


def summarize(exact_rows, distance_rows):
    exact_rows = sorted(exact_rows, key=lambda row: row["seed"])
    distance_rows = sorted(distance_rows, key=lambda row: row["seed"])
    for rows in (exact_rows, distance_rows):
        if [row["seed"] for row in rows] != [42, 43, 44]:
            raise ValueError("exactly seeds 42/43/44 are required for each component")
    for exact, distance in zip(exact_rows, distance_rows, strict=True):
        if not exact["complete"] or not distance["complete"] or not distance["computation"]["complete"]:
            raise ValueError("source component is incomplete")
        if exact["counts"] != distance["counts"] or exact["counts"]["requested"] != 10000:
            raise ValueError("source component raw-slot counts differ")
        if exact["exact_novel_valid_slots"] != distance["exact_novel_valid_slots"]:
            raise ValueError("exact novelty disagrees with zero nearest distances")
        if exact["reference_records"] != distance["computation"]["source_records_total"]:
            raise ValueError("source reference record counts differ")
        if exact["reference_scope"] != distance["computation"]["reference_scope"]:
            raise ValueError("source reference scopes differ")
        if not distance["computation"]["unindexed_lengths_proven_unable_to_improve"]:
            raise ValueError("nearest distances do not cover omitted reference lengths")
    if len({row["reference_records"] for row in exact_rows}) != 1:
        raise ValueError("seeds use different reference record counts")
    if any(row["reference_scope"] != exact_rows[0]["reference_scope"] for row in exact_rows):
        raise ValueError("seeds use different reference scopes")
    metrics = {}
    for name, values in {
        "exact_novelty_within_valid": [r["exact_novelty_within_valid"] for r in exact_rows],
        "exact_novel_valid_over_requested": [r["exact_novel_valid_over_requested"] for r in exact_rows],
        "nearest_normalized_edit_distance": [r["nearest_normalized_edit_distance"]["mean"] for r in distance_rows],
    }.items():
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError(f"non-finite source metric: {name}")
        defined = sum(value is not None for value in values)
        metrics[name] = {
            "per_seed": dict(zip((42, 43, 44), values, strict=True)),
            "defined_seeds": defined,
            "mean": mean(values) if defined == 3 else None,
            "std_between_seeds_ddof1": stdev(values) if defined == 3 else None,
        }
    return {
        "complete": True, "formal_benchmark_complete": False,
        "official_generation_leaderboard": False, "pretraining_exposed": True,
        "per_record_gradient_exposure_proven": False,
        "includes_small_molecule_graph_equivalents": False,
        "seeds": [42, 43, 44], "raw_slots_per_seed": 10000,
        "reference_records": exact_rows[0]["reference_records"],
        "reference_scope": exact_rows[0]["reference_scope"],
        "counts_per_seed": {row["seed"]: row["counts"] for row in exact_rows},
        "aggregation": "mean and sample SD of three seed-level metrics; valid slots retain multiplicity; undefined seeds are not dropped",
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, nargs=3, required=True)
    parser.add_argument("--distances", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("PepBench conservative source metric aggregation")
    result = summarize([json.loads(p.read_text()) for p in args.exact],
                       [json.loads(p.read_text()) for p in args.distances])
    result["input_exact"] = list(map(str, args.exact))
    result["input_distances"] = list(map(str, args.distances))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
