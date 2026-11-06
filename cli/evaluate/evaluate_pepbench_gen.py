#!/usr/bin/env python3
"""Evaluate raw PepBench-Gen slots with a fixed sequence/ESM-2 protocol.

This is a pretraining-exposed reconstruction, not an official AMP leaderboard.
Missing full-pretraining references are permitted ONLY in explicitly partial
diagnostics. Invalid outputs remain in raw-slot denominators; no resampling.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import ks_2samp

from cli.evaluate.pepbench_generation_protocol import TASK, FORMAL_SAMPLES, load_schedule
from pipeline.benchmarks.pepbench_esm import Esm8MFeatures
from pipeline.benchmarks.pepbench_metrics import (
    AA, distribution_summary, diversity_metric, embedding_precision_recall,
    frechet_distance, marginal_jsd, novelty_metrics, sequence_is_valid, slot_metrics,
)
from pipeline.runtime_host import require_compute_host

PROTOCOL = "axis-pepbench-gen-esm8m-v1-20260915"
# Published modlAMP "eisenberg" scale (including its rounded F/I/L/R/V values),
# as used by OmegAMP. Alpha-helix moment: full sequence, 100 degrees/residue.
# https://modlamp.org/_modules/modlamp/core.html#load_scale
EISENBERG = dict(zip("ACDEFGHIKLMNPQRSTVWY", [0.62, 0.29, -0.90, -0.74, 1.2, 0.48, -0.40, 1.4, -1.50, 1.1, 0.64, -0.78, 0.12, -0.85, -2.5, -0.18, -0.05, 1.1, 0.81, 0.26], strict=True))


def read_references(path, *, split=None, canonical=False):
    rows, sequences = [], []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if split is not None and row["split"] != split:
                raise ValueError(f"{path}: reference is not exclusively {split}")
            sequence = row["sequence"] if "sequence" in row else row["original_sequence"]
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"{path}: empty/non-string reference sequence")
            if canonical and not sequence_is_valid(sequence):
                raise ValueError(f"{path}: noncanonical reference sequence")
            rows.append(row)
            sequences.append(sequence)
    if not rows:
        raise ValueError(f"empty reference: {path}")
    return rows, sequences


def read_slots(path, schedule):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indices = [r["sample_index"] for r in rows]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(schedule["samples"])):
        raise ValueError("raw sample indices have duplicates, missing or additional slots")
    rows.sort(key=lambda r: r["sample_index"])
    for row, condition in zip(rows, schedule["conditions"], strict=True):
        if row["task"] != TASK or row["condition"] != condition:
            raise ValueError("raw task/length conditions differ from the frozen prompt schedule")
    return rows


def physicochemical(sequences):
    result = {k: [] for k in ("charge_pH7", "gravy", "pI", "instability", "molecular_weight", "hydrophobic_moment_alpha", "aa_entropy_bits")}
    for seq in sequences:
        analysis = ProteinAnalysis(seq)
        result["charge_pH7"].append(analysis.charge_at_pH(7.0))
        result["gravy"].append(analysis.gravy())
        result["pI"].append(analysis.isoelectric_point())
        result["instability"].append(analysis.instability_index())
        result["molecular_weight"].append(analysis.molecular_weight())
        angles = np.arange(len(seq)) * np.deg2rad(100.0)
        h = np.array([EISENBERG[aa] for aa in seq])
        moment = np.hypot((h * np.cos(angles)).sum(), (h * np.sin(angles)).sum()) / len(seq)
        result["hydrophobic_moment_alpha"].append(float(moment))
        p = np.array(list(Counter(seq).values()), dtype=np.float64) / len(seq)
        result["aa_entropy_bits"].append(float(-(p * np.log2(p)).sum()))
    return result


def compare_physicochemical(sequences, reference):
    generated, real = physicochemical(sequences), physicochemical(reference)
    return {k: {"generated": distribution_summary(generated[k]), "reference": distribution_summary(real[k]),
                "ks_distance": float(ks_2samp(generated[k], real[k]).statistic) if generated[k] else None}
            for k in generated}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-samples", type=Path, required=True)
    parser.add_argument("--condition-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--test-reference", type=Path, required=True)
    parser.add_argument("--pepbenchmark-source-reference", type=Path, required=True)
    parser.add_argument("--full-pretraining-reference", type=Path)
    parser.add_argument("--esm-model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--diagnostic", action="store_true", help="explicitly partial: allow fewer slots and/or no full-pretraining reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute_host("PepBench generated-sequence evaluation", socket.gethostname())
    schedule = load_schedule(args.condition_manifest, seed=args.seed)
    if not args.diagnostic and (schedule["samples"] != FORMAL_SAMPLES or args.full_pretraining_reference is None):
        raise ValueError("formal evaluation needs 10000 slots and a full pretraining reference; use --diagnostic only for explicit partial results")
    if args.cpu_threads < 1:
        raise ValueError("positive CPU thread count required")
    torch.set_num_threads(args.cpu_threads)
    rows = read_slots(args.decoded_samples, schedule)
    counts, valid = slot_metrics(rows)
    train_rows, train = read_references(args.train_reference, split="train", canonical=True)
    test_rows, test = read_references(args.test_reference, split="test", canonical=True)
    if set(train) & set(test) or {r["id"] for r in train_rows} & {r["id"] for r in test_rows}:
        raise ValueError("SFT train and test reference overlap")
    _, pep_source = read_references(args.pepbenchmark_source_reference)
    refs = {"sft_train": train, "pepbenchmark_pretraining_source_only": pep_source}
    if args.full_pretraining_reference is not None:
        _, refs["full_pretraining"] = read_references(args.full_pretraining_reference)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    novelty = {}
    for name, reference in refs.items():
        result = novelty_metrics(valid, reference, workers=args.cpu_threads)
        per_sequence = result.pop("per_unique_sequence_nearest_distance", {})
        (args.output / f"nearest_{name}.json").write_text(json.dumps(per_sequence) + "\n")
        result["exact_novel_valid_over_requested"] = result.get("exact_novel_slots", 0) / len(rows)
        novelty[name] = result
        print(f"NOVELTY_COMPLETE reference={name} valid={len(valid)}", flush=True)
    diversity = diversity_metric(valid, seed=args.seed)
    properties = compare_physicochemical(valid, test)
    features = Esm8MFeatures(args.esm_model, device=args.device)
    unique = sorted(set(valid))
    unique_embeddings = features.embeddings(unique)
    embedding_lookup = dict(zip(unique, unique_embeddings, strict=True))
    generated_embeddings = (np.stack([embedding_lookup[s] for s in valid]) if valid else np.empty((0, 320), dtype=np.float32))
    reference_embeddings = features.embeddings(test)
    np.savez_compressed(args.output / "esm_features.npz", generated=generated_embeddings, reference=reference_embeddings,
                        generated_valid_slot_ids=np.array([r["id"] for r in rows if r["status"] == "valid"]),
                        reference_ids=np.array([r["id"] for r in test_rows]))
    print(f"EMBEDDINGS_COMPLETE generated={len(valid)} reference={len(test)}", flush=True)
    ppl = {}
    with (args.output / "ppl_per_unique_sequence.jsonl").open("x") as handle:
        for i, seq in enumerate(unique):
            value = features.pseudo_perplexity(seq)
            ppl[seq] = value
            handle.write(json.dumps({"sequence": seq, "ppl": value}) + "\n")
            if (i + 1) % 100 == 0 or i + 1 == len(unique):
                handle.flush()
                print(f"PPL_SCORED {i + 1}/{len(unique)}", flush=True)
    summary = {
        "protocol": PROTOCOL, "complete": True, "formal": not args.diagnostic,
        "official_generation_leaderboard": False, "pretraining_exposed": True,
        "full_pretraining_novelty_available": args.full_pretraining_reference is not None,
        "partial_result_reasons": (["diagnostic mode"] if args.diagnostic else []) +
            (["missing complete foundation-model pretraining reference"] if args.full_pretraining_reference is None else []),
        "seed": args.seed, "sampling_schedule": str(args.condition_manifest), "counts": counts,
        "novelty": novelty, "diversity": diversity,
        "marginal_distribution": marginal_jsd(valid, test), "physicochemical": properties,
        "esm": features.provenance(), "esm_ppl": distribution_summary([ppl[s] for s in valid]),
        "fbd": frechet_distance(reference_embeddings, generated_embeddings),
        "embedding_manifold": embedding_precision_recall(reference_embeddings, generated_embeddings),
        "embedding_coverage_over_requested": len(valid) / len(rows),
        "ppl_coverage_over_requested": len(valid) / len(rows),
        "reference_paths": {"sft_train": str(args.train_reference), "sft_test": str(args.test_reference),
                            "pepbenchmark_source_only": str(args.pepbenchmark_source_reference),
                            "full_pretraining": str(args.full_pretraining_reference) if args.full_pretraining_reference else None},
        "metric_selection": "frozen before model benchmark scoring; no test-selected checkpoint or sampling parameters",
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
