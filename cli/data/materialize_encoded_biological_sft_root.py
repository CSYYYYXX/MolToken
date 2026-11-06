#!/usr/bin/env python3
"""Publish already encoded biological split JSONL as one formal Axis SFT root.

The upstream materializer owns sequence cleaning, split construction, heavy
graph conversion, FDDFS serialization, and frozen-BPE compression.  This
worker performs the final transport step only: it validates the encoded record
schema, hard-links train/validation/test JSONL into an isolated task root,
writes the standard Axis manifest, and builds the random-access index.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Iterator, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.active_scope import (  # noqa: E402
    FORMAL_TASK_DATASET_SCOPES,
)
from pipeline.finetune.axis_b_data import (  # noqa: E402
    RAW_BPE_MAX_ID,
    TARGET_TYPE_TOKEN_IDS,
    AxisBIndex,
    build_axis_b_index,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402


PROTOCOL = "axis-biological-encoded-sft-root-v1-20260830"
SPLITS = ("train", "validation", "test")


def require_worker() -> None:
    require_compute_host("biological SFT publication", socket.gethostname())


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield line_number, record


def validate_record(
    record: Mapping[str, Any],
    *,
    task: str,
    split: str,
    source: Path,
    line_number: int,
) -> str:
    label = f"{source}:{line_number}"
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{label}: missing non-empty id")
    if record.get("task") != task:
        raise ValueError(
            f"{label}: task={record.get('task')!r}, expected {task!r}"
        )
    target_type = record.get("target_type")
    if target_type not in TARGET_TYPE_TOKEN_IDS:
        raise ValueError(f"{label}: unsupported target_type={target_type!r}")
    target = record.get("target_bpe_ids")
    if not isinstance(target, list) or not target:
        raise ValueError(f"{label}: target_bpe_ids must be a non-empty list")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        or value < 0 or value > RAW_BPE_MAX_ID
        for value in target
    ):
        raise ValueError(f"{label}: target_bpe_ids contains an invalid raw BPE id")
    condition = record.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError(f"{label}: condition must be an object")
    condition_type = condition.get("type")
    if condition_type not in {"none", "text", "molecule", "both"}:
        raise ValueError(f"{label}: invalid condition.type={condition_type!r}")
    molecule_ids = condition.get("molecule_bpe_ids")
    if molecule_ids is not None:
        if not isinstance(molecule_ids, list):
            raise ValueError(f"{label}: condition.molecule_bpe_ids must be a list")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            or value < 0 or value > RAW_BPE_MAX_ID
            for value in molecule_ids
        ):
            raise ValueError(f"{label}: condition contains an invalid raw BPE id")
    metadata = record.get("meta")
    if isinstance(metadata, Mapping):
        declared_split = metadata.get("split")
        if declared_split is not None and str(declared_split) != split:
            raise ValueError(
                f"{label}: meta.split={declared_split!r}, expected {split!r}"
            )
        full_context = metadata.get("full_context_tokens")
        if full_context is not None and int(full_context) > 512:
            raise ValueError(
                f"{label}: materialized full context {full_context} exceeds 512"
            )
    return record_id


def scan_split(path: Path, *, task: str, split: str, seen_ids: set[str]) -> int:
    count = 0
    for line_number, record in iter_records(path):
        record_id = validate_record(
            record,
            task=task,
            split=split,
            source=path,
            line_number=line_number,
        )
        if record_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id across splits: {record_id}")
        seen_ids.add(record_id)
        count += 1
    if count == 0:
        raise ValueError(f"{path}: split is empty")
    return count


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_index(
    index: AxisBIndex,
    *,
    task: str,
    dataset_id: str,
    counts: Mapping[str, int],
) -> None:
    indexed_datasets = {str(item.get("dataset")) for item in index.files}
    if indexed_datasets != {dataset_id}:
        raise RuntimeError(
            f"indexed dataset mismatch: expected={dataset_id!r} actual={indexed_datasets}"
        )
    expected = {
        "train": int(counts["train"]),
        "eval": int(counts["validation"]),
        "test": int(counts["test"]),
    }
    for split, expected_count in expected.items():
        observed = int(index.task_counts(split).get(task, 0))
        if observed != expected_count:
            raise RuntimeError(
                f"indexed {split} count mismatch for {task}: "
                f"expected={expected_count} actual={observed}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(FORMAL_TASK_DATASET_SCOPES))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    expected_scope = FORMAL_TASK_DATASET_SCOPES[args.task]
    if tuple(expected_scope) != (args.dataset_id,):
        raise ValueError(
            f"dataset id {args.dataset_id!r} does not match registered scope "
            f"for {args.task!r}: {expected_scope}"
        )

    sources = {
        "train": require_file(args.train, "train JSONL"),
        "validation": require_file(args.validation, "validation JSONL"),
        "test": require_file(args.test, "test JSONL"),
    }
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    split_dir = staging / "splits"
    split_dir.mkdir()

    seen_ids: set[str] = set()
    counts: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for split in SPLITS:
        source = sources[split]
        count = scan_split(source, task=args.task, split=split, seen_ids=seen_ids)
        destination = split_dir / f"{split}.jsonl"
        os.link(source, destination)
        counts[split] = count
        summaries.append(
            {
                "dataset": args.dataset_id,
                "split": split,
                "relative_source": str(destination.relative_to(staging)),
                "records": count,
                "output_size": destination.stat().st_size,
            }
        )

    manifest = {
        "complete": True,
        "protocol": {
            "name": PROTOCOL,
            "serializer": "fddfs",
            "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
            "model_token_offset": 3,
            "validation_source": "benchmark-provided or train-derived validation; never official test",
        },
        "task": args.task,
        "dataset_id": args.dataset_id,
        "files": len(summaries),
        "records": sum(counts.values()),
        "split_counts": counts,
        "file_summaries": summaries,
        "source_files": {split: str(path) for split, path in sources.items()},
        "transport": "same-filesystem hard links; encoded payload bytes unchanged",
    }
    atomic_json(staging / "manifest.json", manifest)
    index = build_axis_b_index(staging)
    validate_index(
        index,
        task=args.task,
        dataset_id=args.dataset_id,
        counts=counts,
    )
    index_manifest_path = staging / "axis_b_index_v1" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["data_root"] = str(output_root)
    atomic_json(index_manifest_path, index_manifest)
    os.replace(staging, output_root)
    print(
        json.dumps(
            {
                "status": "complete",
                "task": args.task,
                "dataset_id": args.dataset_id,
                "output_root": str(output_root),
                "split_counts": counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
