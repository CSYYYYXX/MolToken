#!/usr/bin/env python3
"""Empirical TRAIN unigram sanity baseline on the shared frozen length schedule."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cli.evaluate.evaluate_pepbench_gen import read_references
from cli.evaluate.pepbench_generation_protocol import TASK, load_schedule
from pipeline.benchmarks.pepbench_metrics import AA
from pipeline.runtime_host import require_compute_host


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--condition-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute_host("PepBench empirical baseline generation", socket.gethostname())
    _, sequences = read_references(args.train_reference, split="train", canonical=True)
    counts = Counter("".join(sequences))
    schedule = load_schedule(args.condition_manifest, seed=args.seed)
    # Separate stream from the Python seed used to construct target lengths.
    rng = random.Random(args.seed + 100_000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for i, length in enumerate(schedule["conditions"]):
            seq = "".join(rng.choices(AA, weights=[counts[aa] for aa in AA], k=int(length)))
            handle.write(json.dumps({"id": f"unigram:{args.seed}:{i}", "task": TASK,
                "target_type": "peptide", "sample_index": i, "condition": length,
                "decoded_sequence": seq, "status": "valid", "ended_with_eos": True,
                "baseline": "empirical train-AA unigram at prescribed length",
                "raw_output": seq, "invalid_outputs_replaced": False}) + "\n")
    summary = {"model": "empirical train-AA unigram", "requested": schedule["samples"],
               "seed": args.seed, "residue_rng_seed": args.seed + 100_000,
               "train_records": len(sequences), "train_aa_counts": dict(counts),
               "condition_manifest": str(args.condition_manifest), "official_test_records_read": 0,
               "prescribed_length_baseline": True,
               "axis_comparison_caveat": "Axis conditions on length but uses free EOS; unigram draws exactly the prescribed residue count"}
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
