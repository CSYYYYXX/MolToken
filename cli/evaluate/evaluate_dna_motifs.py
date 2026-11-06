#!/usr/bin/env python3
"""Score decoded DNA with the verified released-hit MOODS reconstruction.

This component retains raw slot coverage and multiplicities. Population Gini
is diagnostic: neither it nor this command is the complete published headline
evaluator. Invalid/wrong-length slots are never repaired or replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.dna_moods import ReleasedDNAMotifScanner
from pipeline.benchmarks.dna_motif_metrics import motif_diversity
from pipeline.runtime_host import require_compute_host

TASK = "dna_diffusion_celltype_generation"
CONDITIONS = ("GM12878", "K562", "HepG2")
STATUSES = {"valid", "wrong_length", "invalid_model_id", "missing_eos",
            "empty_generation", "decode_error"}


def score_rows(rows, scanner, *, samples_per_condition=1000):
    if samples_per_condition <= 0:
        raise ValueError("samples_per_condition must be positive")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate raw slot IDs")
    groups = {condition: [] for condition in CONDITIONS}
    for row in rows:
        if row["task"] != TASK or row["condition"] not in groups:
            raise ValueError("expected one of the three headline DNA conditions")
        if row["status"] not in STATUSES:
            raise ValueError(f"unknown decoder status: {row['status']}")
        if row["status"] == "valid":
            seq = row["decoded_sequence"]
            if (not isinstance(seq, str) or len(seq) != 200 or
                    set(seq) - set("ACGT") or row["ended_with_eos"] is not True):
                raise ValueError("declared-valid slot is not EOS-complete canonical 200-bp DNA")
        groups[row["condition"]].append(row)
    for condition, selected in groups.items():
        if len(selected) != samples_per_condition:
            raise ValueError(f"{condition}: expected {samples_per_condition} raw slots, got {len(selected)}")
        if sorted(row["sample_index"] for row in selected) != list(range(samples_per_condition)):
            raise ValueError(f"{condition}: missing or duplicate sample indices")

    cache, per_condition, per_slot = {}, {}, []
    matrix_ids = set(scanner.matrix_ids)
    for condition, selected in groups.items():
        counts, valid_sequences = Counter(), Counter()
        for row in selected:
            slot = {key: row[key] for key in ("id", "condition", "sample_index", "status")}
            slot.update(scanned=False, motif_hits=None)
            if row["status"] == "valid":
                sequence = row["decoded_sequence"]
                if sequence not in cache:
                    values = scanner.counts(sequence)
                    if set(values) - matrix_ids:
                        raise ValueError("scanner returned a matrix outside its declared collection")
                    if any(type(n) is not int or n < 0 for n in values.values()):
                        raise ValueError("scanner returned a non-integer or negative hit count")
                    cache[sequence] = values
                counts.update(cache[sequence])  # Once per raw slot, not per unique sequence.
                valid_sequences[sequence] += 1
                slot.update(scanned=True, motif_hits=sum(cache[sequence].values()))
            per_slot.append(slot)
        n_valid = sum(valid_sequences.values())
        per_condition[condition] = {
            "requested": len(selected), "scanned_valid_200bp": n_valid,
            "scan_coverage": n_valid / len(selected),
            "unique_sequences_scanned": len(valid_sequences),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "total_motif_hits": sum(counts.values()),
            "counts_by_matrix": {m: counts[m] for m in scanner.matrix_ids},
            "diagnostic_plain_gini_diversity": motif_diversity([counts[m] for m in scanner.matrix_ids]),
        }
    values = [g["diagnostic_plain_gini_diversity"] for g in per_condition.values()]
    scanned = sum(g["scanned_valid_200bp"] for g in per_condition.values())
    return {
        "task": TASK, "requested": len(rows), "scanned_valid_200bp": scanned,
        "scan_coverage": scanned / len(rows), "unique_scans_computed": len(cache),
        "per_condition": per_condition,
        "diagnostic_macro_plain_gini_diversity": sum(values) / len(values) if all(v is not None for v in values) else None,
        "motif_scan_complete": True, "official_headline_complete": False,
        "gini_status": "diagnostic population Gini over the declared matrix universe; Figure7 aggregation remains unresolved",
        "denominator_policy": "scan only canonical 200-bp raw slots; retain invalid slots in coverage; duplicates retain original multiplicity",
        "failed_slots_replaced": False, "sequences_repaired_or_truncated": False,
        "evaluator": scanner.provenance(),
    }, per_slot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-samples", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-condition", type=int, default=1000,
                        help="1000 for the planned model run; smaller counts are explicit controls")
    args = parser.parse_args(argv)
    require_compute_host("native DNA motif evaluation")
    rows = [json.loads(line) for line in args.decoded_samples.read_text().splitlines()]
    scanner = ReleasedDNAMotifScanner(args.profile_json)
    summary, per_slot = score_rows(rows, scanner, samples_per_condition=args.samples_per_condition)
    summary.update(decoded_samples=str(args.decoded_samples), samples_per_condition=args.samples_per_condition)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_slot.jsonl").open("x") as handle:
        for row in per_slot:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    with (args.output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"DNA_MOTIF_COMPONENT_COMPLETE {args.output}; official_headline_complete=false", flush=True)


if __name__ == "__main__":
    main()
