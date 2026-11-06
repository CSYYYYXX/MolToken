#!/usr/bin/env python3
"""Freeze 3 x 10,000 model-independent lengths from the actual SFT TRAIN view."""
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cli.evaluate.pepbench_generation_protocol import FORMAL_SAMPLES, FORMAL_SEEDS, TASK, make_schedule
from pipeline.finetune.axis_b_data import AxisBIndex, AxisBJsonlDataset
from pipeline.runtime_host import require_compute_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("PepBench train-length schedule materialization", socket.gethostname())
    index = AxisBIndex.load(args.data_root, args.data_root / "axis_b_index_v1")
    dataset = AxisBJsonlDataset(args.data_root, split="train", index=index, task=TASK)
    lengths = []
    try:
        for i in range(len(dataset)):
            record = dataset[i]
            condition = record["condition"]["text"]
            length = int(condition)
            if str(length) != condition:
                raise ValueError(f"noncanonical length condition in {record['id']}")
            lengths.append(length)
    finally:
        dataset.close()
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in FORMAL_SEEDS:
        schedule = make_schedule(lengths, seed=seed, samples=FORMAL_SAMPLES)
        schedule["train_data_root"] = str(args.data_root.resolve())
        path = args.output / f"conditions_seed{seed}.json"
        path.write_text(json.dumps(schedule, indent=2) + "\n")
        print(json.dumps({"output": str(path), "seed": seed,
                          "train_records": len(lengths), "slots": FORMAL_SAMPLES}), flush=True)


if __name__ == "__main__":
    main()
