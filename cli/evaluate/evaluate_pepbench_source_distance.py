#!/usr/bin/env python3
"""Complete conservative AA source nearest distances using an exact index scan."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.aa_source_distance import nearest_source_distances
from pipeline.benchmarks.aa_source_index import source_rows
from pipeline.benchmarks.pepbench_metrics import distribution_summary, slot_metrics
from cli.evaluate.pepbench_generation_protocol import TASK, load_schedule
from pipeline.runtime_host import require_compute_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--condition-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--anchor-source", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("exact full-AA-source nearest edit distances")
    schedule = load_schedule(args.condition_manifest, seed=args.seed)
    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    if sorted(r["sample_index"] for r in rows) != list(range(schedule["samples"])):
        raise ValueError("sample slots differ from the frozen schedule")
    rows.sort(key=lambda r: r["sample_index"])
    for row, condition in zip(rows, schedule["conditions"], strict=True):
        if row["task"] != TASK or row["condition"] != condition:
            raise ValueError("task/condition mismatch")
    counts, valid = slot_metrics(rows)
    index_summary = json.loads((args.index / "summary.json").read_text())
    source = next(item for item in index_summary["sources"] if Path(item["source_path"]) == args.anchor_source)
    anchors = [row["normalized"] for row in source_rows(args.anchor_source)]
    if len(anchors) != source["records"]:
        raise ValueError("anchor source record count changed since index construction")
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"DISTANCE_START seed={args.seed} valid={len(valid)} anchor_records={len(anchors)} workers={args.workers}", flush=True)
    nearest, computation = nearest_source_distances(valid, args.index, anchors, workers=args.workers)
    with (args.output / "per_unique_sequence.json").open("x") as handle:
        json.dump(nearest, handle, allow_nan=False)
        handle.write("\n")
    values = [nearest[seq] for seq in valid]
    novel = sum(value > 0 for value in values)
    result = {"complete": True, "formal_benchmark_complete": False, "seed": args.seed,
              "counts": counts, "samples": str(args.samples), "index": str(args.index),
              "anchor_source": str(args.anchor_source), "computation": computation,
              "exact_novel_valid_slots": novel,
              "exact_novelty_within_valid": novel / len(valid) if valid else None,
              "exact_novel_valid_over_requested": novel / len(rows),
              "nearest_normalized_edit_distance": distribution_summary(values),
              "per_record_gradient_exposure_proven": False,
              "includes_small_molecule_graph_equivalents": False,
              "definition": "min over all listed conservative AA sources of Levenshtein(g,r)/max(len(g),len(r)); raw valid-slot weighting"}
    with (args.output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
