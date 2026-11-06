#!/usr/bin/env python3
"""Add exact conservative source novelty without rereading giant JSON into RAM.

Writes a separate evidence artifact; does not promote a partial ESM evaluator
result to formal, claim complete nearest-distance scoring, or overwrite it.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.aa_source_index import exact_source_matches
from pipeline.benchmarks.pepbench_metrics import slot_metrics
from cli.evaluate.pepbench_generation_protocol import TASK, load_schedule
from pipeline.runtime_host import require_compute_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-samples", type=Path, required=True)
    parser.add_argument("--condition-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("PepBench conservative-source exact novelty")
    schedule = load_schedule(args.condition_manifest, seed=args.seed)
    rows = [json.loads(line) for line in args.decoded_samples.read_text().splitlines()]
    if sorted(r["sample_index"] for r in rows) != list(range(schedule["samples"])):
        raise ValueError("sample slots differ from the frozen schedule")
    rows.sort(key=lambda r: r["sample_index"])
    for row, condition in zip(rows, schedule["conditions"], strict=True):
        if row["task"] != TASK or row["condition"] != condition:
            raise ValueError("sample task/condition mismatch")
    counts, valid = slot_metrics(rows)
    matches, index_summary = exact_source_matches(valid, args.index)
    matched_slots = sum(seq in matches for seq in valid)
    novel = len(valid) - matched_slots
    result = {
        "complete": True, "formal_benchmark_complete": False, "seed": args.seed,
        "decoded_samples": str(args.decoded_samples), "index": str(args.index),
        "counts": counts, "matched_valid_slots": matched_slots,
        "exact_novel_valid_slots": novel,
        "exact_novelty_within_valid": novel / len(valid) if valid else None,
        "exact_novel_valid_over_requested": novel / len(rows),
        "matched_unique_sequences": sorted(matches),
        "reference_scope": index_summary["reference_scope"],
        "reference_records": index_summary["source_records_total"],
        "per_record_gradient_exposure_proven": False,
        "includes_small_molecule_graph_equivalents": False,
        "nearest_normalized_edit_distance_computed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
