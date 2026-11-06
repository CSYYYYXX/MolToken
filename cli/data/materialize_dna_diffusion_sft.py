#!/usr/bin/env python3
"""Materialize DNA-Diffusion into the frozen Axis biological SFT schema.

The released chromosome split is preserved for benchmark evaluation, but
early stopping uses an exact, deterministic 5% holdout taken only from the
released training rows.  Released chr2 validation records are retained as a
separate tuning/reference artifact and are never indexed as the SFT eval view.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any, Iterable, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.active_scope import FORMAL_TASK_DATASET_SCOPES  # noqa: E402
from pipeline.finetune.axis_b_data import build_axis_b_index  # noqa: E402
from pipeline.runtime_host import require_compute_host  # noqa: E402


ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / '')
)
DEFAULT_SOURCE = (
    ROOT
    / "data/downloads_raw/AxisBiologicalSequenceBenchmarks-20260830/dna_diffusion"
    / "K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/processed/axis_b_biological_sft/dna_diffusion_official_fourlabel_v1"
)
DEFAULT_GUIDE = ROOT / "data/fddfs_guide_recount_v1/guide_counts.json"
DEFAULT_VOCAB = ROOT / "data/fddfs_guide_recount_v1/vocab.json"
DEFAULT_MERGES = (
    ROOT
    / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1"
    / "codebook_fddfs_prefix70_mf80000_v1/merges.csv"
)
DEFAULT_TOKENIZER = ROOT / "data/pretrained_encoders/all-mpnet-base-v2/tokenizer.json"

TASK = "dna_diffusion_celltype_generation"
DATASET_ID = "dna_diffusion_official_fourlabel_v1"
PROTOCOL = "axis-dna-diffusion-sft-v1-20260830"
LABELS = ("GM12878", "hESCT0", "K562", "HepG2")
ALPHABET = frozenset("ACGT")
EXPECTED_RELEASED_COUNTS = {
    "training": 39_254,
    "validation": 4_053,
    "test": 4_565,
}


def require_worker() -> None:
    require_compute_host("DNA-Diffusion materialization", socket.gethostname())


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def stable_score(seed: int, label: str, record_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{record_id}".encode("utf-8")).digest()


def exact_stratified_inner_validation(
    rows: Sequence[dict[str, str]], *, fraction: float, seed: int
) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("inner-validation fraction must be in (0, 1)")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["cell_type"]].append(row)
    if set(grouped) != set(LABELS):
        raise ValueError(f"released training labels differ from {LABELS}: {sorted(grouped)}")
    selected: set[str] = set()
    for label in LABELS:
        ordered = sorted(
            grouped[label],
            key=lambda row: (stable_score(seed, label, row["id"]), row["id"]),
        )
        count = max(1, int(round(len(ordered) * fraction)))
        selected.update(row["id"] for row in ordered[:count])
    return selected


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ids: set[str] = set()
    sequences: set[str] = set()
    released_counts: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"dhs_id", "chr", "sequence", "TAG", "data_label"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"DNA-Diffusion TSV missing columns: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            source_id = str(raw["dhs_id"]).strip()
            chromosome = str(raw["chr"]).strip()
            sequence = str(raw["sequence"]).strip().upper()
            cell_type = str(raw["TAG"]).split("_", 1)[0]
            released_split = str(raw["data_label"]).strip().lower()
            if not source_id or not chromosome:
                raise ValueError(f"{path}:{line_number}: empty id or chromosome")
            if cell_type not in LABELS:
                raise ValueError(f"{path}:{line_number}: unexpected cell type {cell_type!r}")
            if released_split not in EXPECTED_RELEASED_COUNTS:
                raise ValueError(
                    f"{path}:{line_number}: unexpected released split {released_split!r}"
                )
            if len(sequence) != 200 or set(sequence).difference(ALPHABET):
                raise ValueError(f"{path}:{line_number}: target is not strict 200-bp ACGT")
            record_id = f"dna-diffusion:{source_id}:{cell_type}"
            if record_id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id}")
            if sequence in sequences:
                raise ValueError(f"{path}:{line_number}: duplicate sequence")
            ids.add(record_id)
            sequences.add(sequence)
            released_counts[released_split] += 1
            rows.append(
                {
                    "id": record_id,
                    "source_id": source_id,
                    "chromosome": chromosome,
                    "sequence": sequence,
                    "cell_type": cell_type,
                    "released_split": released_split,
                }
            )
    if dict(released_counts) != EXPECTED_RELEASED_COUNTS:
        raise ValueError(
            f"released split counts changed: expected={EXPECTED_RELEASED_COUNTS} "
            f"actual={dict(released_counts)}"
        )
    return rows


def quantile(values: Sequence[int], q: float) -> float:
    ordered = sorted(int(value) for value in values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (rank - low) * (ordered[high] - ordered[low]))


def summarize(values: Iterable[int]) -> dict[str, Any]:
    items = list(int(value) for value in values)
    return {
        "count": len(items),
        "min": min(items),
        "mean": sum(items) / len(items),
        "p50": quantile(items, 0.50),
        "p95": quantile(items, 0.95),
        "p99": quantile(items, 0.99),
        "max": max(items),
    }


def json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    if FORMAL_TASK_DATASET_SCOPES.get(TASK) != (DATASET_ID,):
        raise RuntimeError("DNA task registry and materializer dataset id disagree")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    source = require_file(args.source, "DNA-Diffusion source TSV")
    guide = require_file(args.guide, "FDDFS guide")
    vocab = require_file(args.vocab, "FDDFS vocabulary")
    merges = require_file(args.merges, "frozen BPE merges")
    tokenizer_path = require_file(args.tokenizer, "MPNet tokenizer")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    staging.mkdir(parents=True)

    rows = load_rows(source)
    training_rows = [row for row in rows if row["released_split"] == "training"]
    inner_validation_ids = exact_stratified_inner_validation(
        training_rows,
        fraction=args.inner_validation_fraction,
        seed=args.split_seed,
    )

    from tokenizers import Tokenizer
    from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    condition_tokens = {
        label: int(sum(tokenizer.encode(label).attention_mask)) for label in LABELS
    }
    if any(value <= 0 for value in condition_tokens.values()):
        raise ValueError(f"cell label encoded to zero text tokens: {condition_tokens}")
    codec = FrozenAxisCodec(guide, vocab, merges)

    output_paths = {
        "train": staging / "train.jsonl",
        "validation": staging / "validation.jsonl",
        "test": staging / "test.jsonl",
        "released_validation": staging / "released_validation.jsonl",
    }
    reference_paths = {
        "test": staging / "test_references.jsonl",
        "released_validation": staging / "released_validation_references.jsonl",
    }
    handles = {name: path.open("x", encoding="utf-8") for name, path in output_paths.items()}
    reference_handles = {
        name: path.open("x", encoding="utf-8") for name, path in reference_paths.items()
    }
    counts: dict[str, int] = defaultdict(int)
    target_lengths: list[int] = []
    context_lengths: list[int] = []
    try:
        for index, row in enumerate(rows, start=1):
            released_split = row["released_split"]
            if released_split == "training":
                split = "validation" if row["id"] in inner_validation_ids else "train"
            elif released_split == "validation":
                split = "released_validation"
            else:
                split = "test"
            metrics, _fddfs, target_bpe = codec.encode_sequence(
                "dna",
                row["sequence"],
                row["id"],
                text_condition_tokens=condition_tokens[row["cell_type"]],
            )
            if int(metrics["full_context_tokens"]) > 512:
                raise ValueError(
                    f"{row['id']}: full context {metrics['full_context_tokens']} exceeds 512"
                )
            record = {
                "id": row["id"],
                "task": TASK,
                "target_type": "dna",
                "condition": {
                    "type": "text",
                    "text": row["cell_type"],
                    "molecule_bpe_ids": [],
                },
                "target_bpe_ids": target_bpe,
                "meta": {
                    "benchmark": "DNA-Diffusion released chromosome split",
                    "split": split,
                    "released_split": released_split,
                    "source_id": row["source_id"],
                    "chromosome": row["chromosome"],
                    "cell_type": row["cell_type"],
                    "sequence_length": 200,
                    "text_prefix_tokens": condition_tokens[row["cell_type"]],
                    "target_bpe_tokens": len(target_bpe),
                    "full_context_tokens": int(metrics["full_context_tokens"]),
                    "target_truncated": False,
                },
            }
            handles[split].write(json_line(record))
            counts[split] += 1
            target_lengths.append(len(target_bpe))
            context_lengths.append(int(metrics["full_context_tokens"]))
            if split in reference_handles:
                reference_handles[split].write(
                    json_line(
                        {
                            "id": row["id"],
                            "cell_type": row["cell_type"],
                            "chromosome": row["chromosome"],
                            "sequence": row["sequence"],
                        }
                    )
                )
            if index % args.progress_every == 0 or index == len(rows):
                print(json.dumps({"encoded": index, "total": len(rows)}), flush=True)
    finally:
        for handle in (*handles.values(), *reference_handles.values()):
            handle.close()

    if counts["train"] + counts["validation"] != EXPECTED_RELEASED_COUNTS["training"]:
        raise RuntimeError("inner train/validation no longer partitions released training")
    if counts["test"] != EXPECTED_RELEASED_COUNTS["test"]:
        raise RuntimeError("formal test count differs from released chr1 test")
    if counts["released_validation"] != EXPECTED_RELEASED_COUNTS["validation"]:
        raise RuntimeError("released chr2 validation count changed")

    file_summaries = []
    for split in ("train", "validation", "test"):
        path = output_paths[split]
        file_summaries.append(
            {
                "dataset": DATASET_ID,
                "split": split,
                "relative_source": path.name,
                "records": counts[split],
                "output_size": path.stat().st_size,
            }
        )
    manifest = {
        "complete": True,
        "task": TASK,
        "dataset_id": DATASET_ID,
        "records": sum(int(item["records"]) for item in file_summaries),
        "files": len(file_summaries),
        "file_summaries": file_summaries,
        "protocol": {
            "name": PROTOCOL,
            "serializer": "fddfs",
            "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
            "model_token_offset": 3,
            "condition": "bare cell-type label",
            "target": "strict 200-bp ACGT sequence",
            "inner_validation": {
                "source": "released training rows only",
                "method": "exact per-cell-type rank by sha256(seed, label, stable id)",
                "fraction": args.inner_validation_fraction,
                "seed": args.split_seed,
                "official_test_used_for_checkpoint_selection": False,
                "released_chr2_validation_used_for_checkpoint_selection": False,
            },
            "headline_cell_types": ["GM12878", "K562", "HepG2"],
            "supplemental_cell_type": "hESCT0",
        },
        "source": str(source),
        "released_counts": EXPECTED_RELEASED_COUNTS,
        "sft_split_counts": {
            "train": counts["train"],
            "validation": counts["validation"],
            "test": counts["test"],
        },
        "released_validation_records": counts["released_validation"],
        "released_validation_output": output_paths["released_validation"].name,
        "reference_outputs": {
            key: path.name for key, path in reference_paths.items()
        },
        "condition_tokens": condition_tokens,
        "target_bpe_tokens": summarize(target_lengths),
        "full_context_tokens": summarize(context_lengths),
        "all_records_fit_512_without_truncation": max(context_lengths) <= 512,
        "codec": codec.codebook,
    }
    atomic_json(staging / "manifest.json", manifest)
    index = build_axis_b_index(staging)
    expected_index_counts = {
        "train": counts["train"],
        "eval": counts["validation"],
        "test": counts["test"],
    }
    for split, expected in expected_index_counts.items():
        actual = int(index.task_counts(split).get(TASK, 0))
        if actual != expected:
            raise RuntimeError(
                f"indexed {split} count mismatch: expected={expected} actual={actual}"
            )
    # The staging directory is atomically renamed below.  Keep the audit-only
    # data_root field pointed at the durable published location instead of the
    # now-defunct ``.building.<pid>`` path.
    index_manifest_path = staging / "axis_b_index_v1" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["data_root"] = str(output_root)
    atomic_json(index_manifest_path, index_manifest)
    os.replace(staging, output_root)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(output_root),
                "counts": dict(counts),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
