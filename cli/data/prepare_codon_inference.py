#!/usr/bin/env python3
"""Prepare Figure4 prompts using only proteins, hosts and TRAIN length statistics.

No official DNA or published score is read. The complete retained protein prefix
is represented with the production codec. The p99 TRAIN generation-token/codon
density reserves output space without claiming unseen targets are certain to fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import multiprocessing as mp
from pathlib import Path
import socket
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host
from pipeline.finetune.codon_inference import fit_codon_input
from pipeline.finetune import sequence_codec_runtime as runtime
from pipeline.benchmarks.common import REFERENCE_LABEL

TEXT_LENGTHS = {}
DENSITY = None


def describe(values):
    values = sorted(values)
    if not values:
        raise ValueError("empty length summary")
    return {"n": len(values), "min": values[0], "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1], "max": values[-1]}


def prepare_request(request):
    organism = request["organism"]
    text = REFERENCE_LABEL[organism]  # Same literal organism label as SFT.
    fit = fit_codon_input(request["protein"], text_tokens=TEXT_LENGTHS[organism],
                          generation_tokens_per_codon=DENSITY,
                          encode_protein=lambda p: runtime.encode_sequence("protein", p)[1])
    return {"id": request["id"], "task": "codontransformer_protein_to_dna", "target_type": "dna",
            "organism": organism, "protein_name": request["protein_name"],
            "condition": {"type": "both", "text": text, "molecule_bpe_ids": fit["protein_bpe_ids"]},
            "protein": fit["protein"], "source_protein": fit["source_protein"],
            "terminal_stop": fit["terminal_stop"], "scope": fit["scope"],
            "max_new_tokens": fit["max_new_tokens"],
            "meta": {key: value for key, value in fit.items()
                     if key not in ("protein_bpe_ids", "protein", "source_protein")}}


def main():
    global TEXT_LENGTHS, DENSITY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True,
                        help="protein/host requests.jsonl from the unchanged Figure4 release")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require_compute_host("CodonTransformer input-only benchmark preparation", socket.gethostname())
    if args.workers < 1:
        raise ValueError("workers must be positive")
    manifest = json.loads((args.data_root / "manifest.json").read_text())
    if not manifest["complete"] or not manifest["training_ready"]:
        raise ValueError("requires completed formal TRAIN materialization")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    density, target_tokens, condition_tokens, lengths = [], [], [], []
    counts = Counter()
    with (args.data_root / "train.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            meta = row["meta"]
            if meta["split"] != "train":
                raise ValueError("budget statistics must use TRAIN, not validation/test")
            codons = meta["dna_bases"] // 3
            if codons < 1 or meta["dna_bases"] % 3:
                raise ValueError("noncoding training target in budget calculation")
            density.append((len(row["target_bpe_ids"]) + 1) / codons)
            target_tokens.append(len(row["target_bpe_ids"]))
            condition_tokens.append(len(row["condition"]["molecule_bpe_ids"]))
            lengths.append(meta["protein_residues"])
            counts["train"] += 1
            counts["source_dna_truncated"] += int(meta["source_dna_truncated"])
            counts["condition_truncated"] += int(meta["condition_truncated"])
            counts["protein_condition_empty"] += int(meta["protein_condition_empty"])
    if counts["train"] != manifest["split_counts"]["train"]:
        raise ValueError("TRAIN record count differs from published materialization")
    DENSITY = describe(density)["p99"]
    policy = {"name": "train_p99_dna_density_input_only_prefix_v1", "data_root": str(args.data_root),
              "training_counts": dict(counts), "generation_tokens_per_codon": DENSITY,
              "density_definition": "(TRAIN target BPE tokens + model EOS) / retained coding DNA codons",
              "density_statistics": describe(density), "target_bpe_tokens": describe(target_tokens),
              "condition_bpe_tokens": describe(condition_tokens), "retained_protein_residues": describe(lengths),
              "selection_data": "TRAIN only; no validation/test targets or model scores",
              "generation": "deterministic greedy, one output per request; free EOS within remaining context; no repair/resampling",
              "decoding_reference": "https://github.com/Adibvafa/CodonTransformer/blob/v1.6.7/CodonTransformer/CodonPrediction.py",
              "estimated_capacity_is_not_a_guarantee": True}
    (args.output_root / "policy.json").write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("duplicate official request IDs")
    for request in requests:
        if request["scope"] != "full_CDS" or not request["terminal_stop"]:
            raise ValueError("input must be original full-protein requests, not an earlier adapted subset")
    from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec
    from cli.evaluate.eval_axis_b_samples import BPEDecoder, FDDFSVocab
    from transformers import AutoTokenizer
    shared = args.shared_root
    vocab = shared / "data/fddfs_guide_recount_v1/vocab.json"
    merges = shared / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1/codebook_fddfs_prefix70_mf80000_v1/merges.csv"
    runtime.CODEC = FrozenAxisCodec(shared / "data/fddfs_guide_recount_v1/guide_counts.json", vocab, merges)
    runtime.DECODER = BPEDecoder(merges, FDDFSVocab.load(vocab))
    tokenizer = AutoTokenizer.from_pretrained(str(shared / "data/pretrained_encoders/all-mpnet-base-v2"),
                                             local_files_only=True)
    TEXT_LENGTHS = {org: len(tokenizer(REFERENCE_LABEL[org])["input_ids"])
                    for org in {r["organism"] for r in requests}}
    sys.setrecursionlimit(100000)
    runtime.encode_sequence("protein", "MA")
    print(json.dumps({"policy": policy, "stage": "encoding_input_proteins", "requests": len(requests)}), flush=True)
    results = []
    with mp.get_context("fork").Pool(args.workers) as pool, (args.output_root / "requests.jsonl").open("x") as handle:
        for result in pool.imap(prepare_request, requests, chunksize=1):
            results.append(result)
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            if len(results) % 20 == 0:
                print(json.dumps({"encoded": len(results), "total": len(requests)}), flush=True)
    from pipeline.finetune.axis_b_data import AxisBSFTCollator
    collator = AxisBSFTCollator(tokenizer, qwen_max_context=512, condition_token_budget=512,
                               molecule_condition_mode="direct_qwen", strict_no_runtime_truncation=True,
                               return_tensors="list")
    for row in results:
        # The production sampling builder keeps only target type/BOS. A single
        # dummy token satisfies the training collator; it is never generated
        # from or replaced with a hidden DNA reference.
        prompt = {key: row[key] for key in ("id", "task", "target_type", "condition")}
        batch = collator([{**prompt, "target_bpe_ids": [0]}])
        observed_prompt = sum(batch["full_attention_mask"][0]) - 2  # dummy + EOS
        if observed_prompt != row["meta"]["prompt_tokens"] or batch["molecule_truncated"][0]:
            raise ValueError(f"production collator changed input-only prompt {row['id']}")
    summary = {"complete": True, "requests": len(results), "scope_counts": dict(Counter(r["scope"] for r in results)),
               "condition_bpe_tokens": describe(len(r["condition"]["molecule_bpe_ids"]) for r in results),
               "retained_residues": describe(len(r["protein"]) for r in results),
               "max_new_tokens": describe(r["max_new_tokens"] for r in results),
               "all_prompt_plus_generation_equal_512":
                   all(r["meta"]["prompt_tokens"] + r["max_new_tokens"] == 512 for r in results),
               "complete_protein_prefix_encoded": True, "hidden_reference_DNA_read": False,
               "strict_production_prompt_checks": len(results),
               "baseline_scoring_complete": False, "HoloMol_evaluation": False,
               "elapsed_seconds": time.monotonic() - started}
    with (args.output_root / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
