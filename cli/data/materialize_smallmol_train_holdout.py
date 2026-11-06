#!/usr/bin/env python3
"""Reuse active frozen BPE TRAIN files with explicit train-only ES holdouts.

No re-encoding, test/official-validation reading, or source changes occur.
The new root contains both active datasets but every SFT lineage selects one
task. Benchmark evaluators keep their separate unmodified official references.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.finetune.active_scope import ACTIVE_DATASET_IDS
from pipeline.finetune.axis_b_data import AxisBJsonlDataset, AxisBSFTCollator, build_axis_b_index, load_formal_manifest
from pipeline.runtime_host import require_compute_host

TASKS = ("unconditional_moses", "unconditional_guacamol")


def prepare(source: Path, output: Path, *, fraction=.01, seed=20260827):
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must lie between zero and one")
    manifest, files, _ = load_formal_manifest(source)
    selected = [f for f in files if f.split == "train"]
    if {f.dataset for f in selected} != set(ACTIVE_DATASET_IDS) or len(selected) != 2:
        raise ValueError("expected exactly the two active frozen-BPE training files")
    output.mkdir(parents=True, exist_ok=False)
    derived = copy.deepcopy(manifest)
    summaries = {item["relative_source"]: item for item in derived["file_summaries"]}
    derived["file_summaries"] = []
    for source_file in selected:
        destination = output / source_file.relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_file.path, destination)
        item = summaries[source_file.relative_source]
        item["output"] = str(destination)
        item["reused_frozen_bpe_source"] = str(source_file.path)
        derived["file_summaries"].append(item)
    derived.update(input_root=str(source), output_root=str(output), files=2,
                   records=sum(int(f.records) for f in selected))
    derived["protocol"]["validation_holdout"] = {
        "method": "sha256_record_id_fraction_v1",
        "tasks": {task: {"fraction": fraction, "seed": seed} for task in TASKS}}
    derived["early_stopping_materialization"] = {
        "source": str(source), "train_files_only": True,
        "official_validation_and_test_records_read": 0,
        "bpe_reencoded": False, "source_files_modified": False,
        "split_note": "deterministic record-ID hashing chooses a split, not an acceptance gate; seed/fraction chosen before new model results"}
    (output / "manifest.json").write_text(json.dumps(derived, indent=2) + "\n")
    index = build_axis_b_index(output)
    counts = {split: index.task_counts(split) for split in ("train", "eval", "test")}
    if counts["test"]:
        raise ValueError("official test rows entered the train-only root")
    if sum(sum(v.values()) for v in counts.values()) != derived["records"]:
        raise ValueError("split accounting does not cover exactly the two training sources")

    collate = AxisBSFTCollator(None, max_context=512, return_tensors="np",
                              strict_no_runtime_truncation=True)
    controls = []
    for task in TASKS:
        for split in ("train", "eval"):
            data = AxisBJsonlDataset(index=index, split=split, task=task)
            if len(data) <= 0:
                raise ValueError(f"empty {task}/{split}")
            positions = sorted({int(i * (len(data) - 1) / 63) for i in range(64)})
            records = [data[i] for i in positions]
            batch = collate(records)
            controls.append({"task": task, "split": split, "records": len(data),
                "sampled_controls": len(records),
                "sampled_max_decoder_tokens": int(batch["decoder_attention_mask"].sum(axis=1).max())})
    summary = {"complete": True, "source": str(source), "output": str(output),
        "selected_source_records": derived["records"], "split_counts": counts,
        "holdout": derived["protocol"]["validation_holdout"], "collator_controls": controls,
        "official_validation_and_test_records_read": 0, "gpu_memory_probes_done": False,
        "training_started": False, "source_bpe_files_modified": False,
        "context_scope": "64 evenly spaced indexed controls per task/split; not a full length audit"}
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    print("ACTIVE_SMALLMOL_TRAIN_HOLDOUT_PREPARATION_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("active small-molecule train-only SFT holdout preparation")
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
