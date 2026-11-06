#!/usr/bin/env python3
"""Freeze a local, uniformly sampled UR50 natural reference before folding.

Uses the complete canonical exact-length candidate scan, BEFORE Axis codec or
context filtering. This is a local reference, not the original paper's sample
IDs and not generated protein output. No training data or split is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host

LENGTHS = (100, 200, 300, 400)
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def select_reference(records, *, per_length, seed):
    if per_length < 1:
        raise ValueError("per_length must be positive")
    pools = {length: [] for length in LENGTHS}
    rngs = {length: random.Random(f"natural-ur50:{seed}:{length}") for length in LENGTHS}
    counts, seen = Counter(), set()
    for row in records:
        length, accession, sequence = row["length"], row["accession"], row["sequence"]
        if length not in pools or len(sequence) != length or set(sequence) - AA:
            raise ValueError(f"noncanonical or wrong-length source record: {accession}")
        if accession in seen:
            raise ValueError(f"duplicate source accession: {accession}")
        seen.add(accession)
        counts[length] += 1
        if len(pools[length]) < per_length:
            pools[length].append(row)
        else:
            position = rngs[length].randrange(counts[length])
            if position < per_length:
                pools[length][position] = row
    result = []
    for length in LENGTHS:
        if counts[length] < per_length:
            raise ValueError(f"length {length}: only {counts[length]} candidates")
        for i, row in enumerate(pools[length]):
            result.append({"id": f"natural:{row['accession']}", "condition": str(length),
                           "sample_index": i, "sequence": row["sequence"],
                           "source_accession": row["accession"], "source_header": row["header"],
                           "source_record_number": row["record_number"],
                           "source_candidate_split": row["split"]})
    return result, {str(length): counts[length] for length in LENGTHS}


def candidate_records(scan_dir, manifest):
    files = sorted(manifest["files"], key=lambda item: item["relative_path"])
    for item in files:
        count = 0
        with (scan_dir / item["relative_path"]).open() as handle:
            for line in handle:
                count += 1
                yield json.loads(line)
        if count != item["records"]:
            raise ValueError(f"incomplete candidate shard {item['relative_path']}: {count} records")


def materialize(scan_dir, source_manifest, output, *, per_length=40, seed=42):
    source = json.loads(source_manifest.read_text())
    scan = json.loads((scan_dir / "manifest.json").read_text())
    if not source["complete"] or not scan["complete"]:
        raise ValueError("complete source scan required")
    rows, counts = select_reference(candidate_records(scan_dir, scan),
                                    per_length=per_length, seed=seed)
    if counts != scan["canonical_selected_records"] or counts != source["selection_audit"]["canonical_selected_records"]:
        raise ValueError("natural reference population differs from complete canonical source scan")
    summary = {
        "complete": True, "kind": "natural_reference", "is_axis_model_result": False,
        "source_release": source["source"]["release"], "source_path": source["source"]["path"],
        "scan_dir": str(scan_dir), "source_manifest": str(source_manifest),
        "candidate_counts": counts, "selected_per_length": per_length, "seed": seed,
        "selected_total": len(rows), "unique_sequences": len({row["sequence"] for row in rows}),
        "selected_source_splits": dict(Counter(row["source_candidate_split"] for row in rows)),
        "selection": "uniform reservoir without replacement by source record, independently per exact length; sorted candidate shard order; Python random.Random string seed natural-ur50:{seed}:{length}",
        "source_filter": "20 canonical amino acids, exact whole-protein lengths 100/200/300/400",
        "source_noncanonical_exclusions": source["selection_audit"]["rejected_noncanonical_records"],
        "codec_or_context_filter_applied": False, "sequence_deduplication_applied": False,
        "structures_used_for_selection": False, "training_split_modified": False,
        "original_paper_reference_ids": False,
        "scope": "local UniRef50 natural baseline, potentially pretraining/SFT-exposed; not a clean unseen set or original DPLM snapshot",
        "paper": "https://proceedings.mlr.press/v235/wang24ct.html",
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "natural_references.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (output / "manifest.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require_compute_host("natural UR50 reference materialization")
    summary = materialize(args.scan_dir, args.source_manifest, args.output,
                          per_length=args.per_length, seed=args.seed)
    print("NATURAL_UR50_REFERENCE_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
