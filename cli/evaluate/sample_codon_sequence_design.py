#!/usr/bin/env python3
"""Sample all prepared Codon Figure4 requests from the matching Full/best.

Run on an allocated GPU worker, optionally via torchrun. Ranks own disjoint
requests. This entry never reads DNA references or benchmark scores.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host
from pipeline.benchmarks.codon_generation import TASK, PROTOCOL, generate_requests, validate_checkpoint
from pipeline.benchmarks.common import completed_lineage


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    require_compute_host("Codon checkpoint sampling", socket.gethostname())
    import torch
    from cli.evaluate.sample_axis_b_sft import load_model
    from pipeline.finetune.axis_b_data import AxisBSFTCollator

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    records = [json.loads(line) for line in args.requests.read_text().splitlines()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("empty or duplicate requests")
    for row in records:
        if row["task"] != TASK or row["target_type"] != "dna" or row["meta"]["reference_DNA_used_for_budget"]:
            raise ValueError(f"invalid Codon input-only request {row['id']}")
    config = json.loads((args.checkpoint / "axis_b_config.json").read_text())
    validate_checkpoint(config)
    lineage = completed_lineage(args.checkpoint.parent.parent, TASK)
    args.condition_token_budget, args.molecule_condition_mode = 512, "direct_qwen"
    model, tokenizer = load_model(args, device)
    collator = AxisBSFTCollator(tokenizer, qwen_max_context=512, return_tensors="pt",
                               condition_token_budget=512, molecule_condition_mode="direct_qwen",
                               strict_no_runtime_truncation=True)
    rank_root = args.output_root / f"rank{rank}"
    rank_root.mkdir(parents=True, exist_ok=False)
    assigned = records[rank::world]
    dump(rank_root / "run_config.json", {"protocol": PROTOCOL, "rank": rank, "world": world,
                                         "checkpoint": str(args.checkpoint), "checkpoint_metadata": config,
                                         "training": lineage,
                                         "requests": str(args.requests), "expected_ids": [r["id"] for r in assigned],
                                         "input_records": assigned,
                                         "batch_size": args.batch_size, "dtype": args.dtype,
                                         "decoding": "one deterministic greedy output per request; free EOS",
                                         "resampling": False, "hidden_DNA_read": False})
    # Equal-length batches reduce decoder padding without changing outputs or
    # request identity. There are no repeated draws for memory occupancy.
    assigned.sort(key=lambda r: (r["meta"]["prompt_tokens"], r["id"]))
    started = time.monotonic()
    with (rank_root / "samples.jsonl").open("x") as handle:
        for start in range(0, len(assigned), args.batch_size):
            batch = assigned[start:start+args.batch_size]
            for result in generate_requests(model, collator, batch, device):
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            handle.flush()
            print(json.dumps({"rank": rank, "sampled": min(start+args.batch_size, len(assigned)),
                              "total": len(assigned), "elapsed_seconds": time.monotonic()-started}), flush=True)
    dump(rank_root / "sampling_complete.json", {"complete": True, "protocol": PROTOCOL,
                                                "requests": len(assigned), "rank": rank, "world": world,
                                                "elapsed_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
