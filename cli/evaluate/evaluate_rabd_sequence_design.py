#!/usr/bin/env python3
"""RAbD: 10,000 unconditional draws per prompt, own-NLL top100, residue AAR.

Run with torchrun on the allocated worker; ranks own disjoint antibodies.
Uses the published 58-case test set, no reference-conditioned generation or
ranking. Variable-length/invalid outputs stay in the fixed-slot denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TASK = "rabd_cdrh3_sequence_design"
PROTOCOL = "axis-rabd-58-ancestral10k-own-nll-top100-v1-20260923"
DECODER = None


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def init_decoder(merges, vocab):
    global DECODER
    from cli.evaluate.sample_tcr_translate_released_top20_auxiliary import AxisPeptideDecoder
    DECODER = AxisPeptideDecoder(Path(merges), Path(vocab))


@lru_cache(maxsize=16384)
def decode_ids(ids):
    # Invalid generated graphs are an expected outcome, retained with errors.
    # Unexpected programming/runtime errors are not swallowed.
    try:
        return DECODER.decode(list(ids)), None
    except ValueError as exc:
        return None, str(exc)


def evaluate_case(path, reference, pool, draws, top):
    from pipeline.benchmarks.rabd_metrics import rank_candidates, score_selected
    rows = read_rows(path)
    eligible = [r for r in rows if r["ended_with_eos"] and not r["invalid_model_id"]
                and r["generated_bpe_ids"]]
    unique = list(dict.fromkeys(tuple(r["generated_bpe_ids"]) for r in eligible))
    decoded = dict(zip(unique, pool.map(decode_ids, unique, chunksize=32), strict=True))
    counts = Counter(requested=len(rows))
    for row in rows:
        row["decoded_sequence"] = None
        if row["invalid_model_id"]:
            status = "invalid_model_id"
        elif not row["ended_with_eos"]:
            status = "missing_eos"
        elif not row["generated_bpe_ids"]:
            status = "empty_generation"
        else:
            sequence, error = decoded[tuple(row["generated_bpe_ids"])]
            row["decoded_sequence"] = sequence
            row["decode_error"] = error
            status = "decoded" if sequence is not None else "decode_error"
            counts["exact_requested_length"] += int(sequence is not None
                                                       and len(sequence) == len(reference))
        row["status"] = status
        counts[status] += 1
    # No filtering of invalid or wrong-length candidates before ranking.
    selected = rank_candidates(rows, expected=draws, top=top)
    score = score_selected(reference, selected, top=top)
    with path.with_suffix(".decoded.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    dump(path.with_suffix(".top100.json"), selected)
    score["all_draw_counts"] = dict(counts)
    return score


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--text-encoder", required=True, type=Path)
    p.add_argument("--vocab", required=True, type=Path)
    p.add_argument("--merges", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--draws", type=int, default=10000)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1280)
    p.add_argument("--limit-cases", type=int, default=0, help="validation smoke only")
    p.add_argument("--seed", type=int, default=7, help="upstream evaluation seed; case streams seed+index")
    p.add_argument("--decode-workers", type=int, default=6)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--condition-token-budget", type=int, default=512)
    return p


def main():
    args = parser().parse_args()
    from pipeline.runtime_host import require_compute_host
    require_compute_host("RAbD checkpoint evaluation", socket.gethostname())
    import torch
    from cli.evaluate.sample_axis_b_sft import load_model, model_batch_from_records
    from cli.evaluate.sample_tcr_translate_released_top20_auxiliary import convert_model_sequence
    from pipeline.finetune.axis_b_data import AxisBSFTCollator
    from pipeline.benchmarks.rabd_sampling import prompt_only, scored_generate
    from pipeline.benchmarks.rabd_metrics import aggregate_cases

    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)  # torchrun's local rank, not CUDA_VISIBLE_DEVICES manipulation
    device = torch.device("cuda", rank)
    if args.draws < args.top or args.top < 1 or args.batch_size < 1:
        raise ValueError("invalid sampling budget")
    if args.split == "test" and (args.draws != 10000 or args.top != 100 or args.limit_cases):
        raise ValueError("test requires all58 cases with10000 draws/top100")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rank_dir = args.output_root / f"rank{rank}"
    rank_dir.mkdir(exist_ok=False)
    records = read_rows(args.data_root / f"{args.split}.jsonl")
    if args.split == "test" and len(records) != 58:
        raise ValueError("released test must have58 cases")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate input IDs")
    if args.limit_cases:
        # Only validation. Longest prompt first for a realistic capacity smoke.
        records = sorted(records, key=lambda r: len(r["condition"]["molecule_bpe_ids"]),
                         reverse=True)[:args.limit_cases]
    for record in records:
        if record["task"] != TASK or record["target_type"] != "peptide":
            raise ValueError("wrong task/target in sampling input")
        if int(record["condition"]["text"]) != record["meta"]["requested_length"]:
            raise ValueError("length condition disagrees with materialization")
    config = json.loads((args.checkpoint / "axis_b_config.json").read_text())
    args.molecule_condition_mode = "direct_qwen"
    model, tokenizer = load_model(args, device)
    collator = AxisBSFTCollator(tokenizer, qwen_max_context=512, return_tensors="pt",
                               condition_token_budget=512, molecule_condition_mode="direct_qwen",
                               strict_no_runtime_truncation=True)
    started = time.time()
    dump(rank_dir / "run_config.json", {
        "protocol": PROTOCOL, "checkpoint": str(args.checkpoint), "checkpoint_metadata": config,
        "split": args.split, "rank": rank, "world": world,
        "case_indices": list(range(rank, len(records), world)),
        "draws_per_case": args.draws, "top": args.top, "batch_size": args.batch_size,
        "dtype": args.dtype, "seed": args.seed, "case_seed_rule": "seed+original_case_index",
        "sampling": "ancestral temperature1/top_p1/top_k0; free EOS, remaining context512",
        "ranking": "sum negative log probabilities of emitted model tokens including EOS / requested AA count",
        "ppl_caveat": "generated graph-token path likelihood, NOT exact AA-sequence marginal PPL",
        "length_or_validity_filter_before_ranking": False, "resampling": False,
        "expected_cases": len(records), "score_assumption": "official AAR only if allselected arevalid/same-length",
    })
    print(json.dumps({"rank": rank, "sampling_started": True, "cases": len(records[rank::world]),
                      "batch_size": args.batch_size}), flush=True)
    for index in range(rank, len(records), world):
        record = records[index]
        prompt = prompt_only(record)
        requested = int(prompt["condition"]["text"])
        torch.manual_seed(args.seed + index)
        torch.cuda.manual_seed(args.seed + index)
        with torch.inference_mode():
            build, _ = model_batch_from_records([prompt], collator, device)
            inputs = model.build_qwen_inputs(**build)
            length = int(inputs["attention_mask"].sum().item())
            embeds = inputs["inputs_embeds"][:, :length]
        case_started = time.time()
        path = rank_dir / f"case{index:03d}.jsonl"
        with path.open("x") as handle:
            for start in range(0, args.draws, args.batch_size):
                size = min(args.batch_size, args.draws - start)
                model_rows, nlls, token_counts = scored_generate(model.qwen, embeds, size)
                for offset, (ids, nll, count) in enumerate(zip(model_rows, nlls, token_counts, strict=True)):
                    # Preserve complete model draws, including malformed tokens.
                    ids = ids[:count]
                    raw, eos, invalid = convert_model_sequence(ids)
                    row = {"case_id": record["id"], "case_index": index,
                           "sample_index": start + offset, "requested_length": requested,
                           "model_token_ids": ids, "generated_bpe_ids": raw,
                           "ended_with_eos": eos, "invalid_model_id": invalid,
                           "sequence_nll": nll, "scored_token_count": count,
                           "nll_per_requested_residue": nll / requested}
                    handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                handle.flush()
                print(json.dumps({"rank": rank, "case": index, "sampled": start + size,
                                  "case_sec": round(time.time() - case_started, 2),
                                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}), flush=True)
    dump(rank_dir / "sampling_complete.json", {"cases": len(records[rank::world]),
                                               "elapsed_sec": time.time() - started})
    # References are read ONLY after all sampling is finalized for this rank.
    references = {r["id"]: r["target"] for r in read_rows(args.data_root / f"{args.split}_references.jsonl")}
    cases = []
    with ProcessPoolExecutor(max_workers=args.decode_workers, mp_context=mp.get_context("spawn"),
                             initializer=init_decoder, initargs=(str(args.merges), str(args.vocab))) as pool:
        for index in range(rank, len(records), world):
            record = records[index]
            reference = references[record["id"]]
            if len(reference) != int(record["condition"]["text"]):
                raise ValueError("reference length mismatch")
            score = evaluate_case(rank_dir / f"case{index:03d}.jsonl", reference,
                                  pool, args.draws, args.top)
            score.update(case_index=index, case_id=record["id"], pdb=record["meta"]["pdb"])
            dump(rank_dir / f"case{index:03d}.metrics.json", score)
            cases.append(score)
            print(json.dumps({"rank": rank, "scored_case": index, **score}), flush=True)
    dump(rank_dir / "metrics.json", {"complete": True, "protocol": PROTOCOL,
                                     "case_count": len(cases), "cases": cases,
                                     "aggregate": aggregate_cases(cases),
                                     "elapsed_sec": time.time() - started})
    print(f"RABD_RANK_COMPLETE rank={rank}", flush=True)


if __name__ == "__main__":
    main()
