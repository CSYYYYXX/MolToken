#!/usr/bin/env python3
"""Materialize released TCR-TRANSLATE pairs for formal Axis two-stage SFT.

The released training pair is filtered to the strict canonical amino-acid
domain and split by complete pMHC source.  Early stopping therefore sees only
a deterministic, train-derived validation set.  The released validation pair
is retained in the generic ``test`` slot solely for the released top-20
auxiliary evaluator; the separate external-13 sparse benchmark is the formal
headline test.

Each pMHC condition is encoded without a natural-language prompt.  The
epitope and 34-residue HLA pseudosequence are independently converted to
heavy graphs and FDDFS token streams, a raw ``<SEP>`` token is inserted
between those streams, and frozen BPE is applied to the joined stream.  The
stored condition contains only the resulting raw BPE ids.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import chain, zip_longest
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.active_scope import FORMAL_TASK_DATASET_SCOPES  # noqa: E402
from pipeline.finetune.axis_b_data import (  # noqa: E402
    RAW_BPE_MAX_ID,
    AxisBIndex,
    build_axis_b_index,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402


ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / '')
)
DEFAULT_INPUT_ROOT = (
    ROOT
    / "data/downloads_raw/AxisBiologicalSequenceBenchmarks-20260830"
    / "tcr_translate/data"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data/processed/axis_b_biological_sft/tcr_translate_released_v1"
)
DEFAULT_GUIDE = ROOT / "data/fddfs_guide_recount_v1/guide_counts.json"
DEFAULT_VOCAB = ROOT / "data/fddfs_guide_recount_v1/vocab.json"
DEFAULT_MERGES = (
    ROOT
    / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1"
    / "codebook_fddfs_prefix70_mf80000_v1/merges.csv"
)

TRAIN_SOURCE_NAME = "pmhc2tcr_topk_train_source_dedup_uf.txt"
TRAIN_TARGET_NAME = "pmhc2tcr_topk_train_target_dedup_uf.txt"
RELEASED_VALIDATION_SOURCE_NAME = "pmhc2tcr_topk_val_source_dedup.txt"
RELEASED_VALIDATION_TARGET_NAME = "pmhc2tcr_topk_val_target_dedup.txt"

TASK = "tcr_translate_pmhc_to_cdr3b"
DATASET_ID = "tcr_translate_released_v1"
PROTOCOL = "axis-tcr-translate-released-sft-v1-20260830"
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_TRAIN_RAW = 328_541
EXPECTED_TRAIN_CANONICAL = 324_435
EXPECTED_RELEASED_VALIDATION_RAW = 92_602
EXPECTED_RELEASED_VALIDATION_CANONICAL = 92_602
INNER_VALIDATION_FRACTION = 0.05
SPLIT_SEED = 42
CONTEXT_BUDGET = 512
TARGET_FRAME_TOKENS = 3


class AxisCodec(Protocol):
    codebook: Mapping[str, Any]

    def encode_sequence(
        self,
        target_type: str,
        sequence: str,
        label: str,
        *,
        text_condition_tokens: int = 0,
        molecule_condition_bpe_tokens: int = 0,
    ) -> tuple[dict[str, int], list[str], list[int]]: ...

    def bpe_ids(self, tokens: Sequence[str], label: str) -> list[int]: ...


@dataclass(frozen=True)
class CanonicalPair:
    release_split: str
    line_number: int
    epitope: str
    hla_pseudosequence: str
    cdr3b: str

    @property
    def source_key(self) -> str:
        return f"{self.epitope} {self.hla_pseudosequence}"

    @property
    def record_id(self) -> str:
        return (
            f"tcr-translate:{self.release_split}:"
            f"{self.line_number:09d}"
        )


@dataclass(frozen=True)
class ReleaseAudit:
    raw_rows: int
    canonical_rows: int
    rejected_rows: int
    rejection_reasons: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_rows": self.raw_rows,
            "canonical_rows": self.canonical_rows,
            "rejected_rows": self.rejected_rows,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


@dataclass(frozen=True)
class GroupSplit:
    validation_groups: frozenset[str]
    total_groups: int
    train_groups: int
    validation_group_count: int
    total_records: int
    target_validation_records: int
    validation_records: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": (
                "group by complete canonical pMHC source; order groups by "
                "sha256(seed\\0source); select the non-empty prefix whose "
                "cumulative record count is closest to round(fraction*N), "
                "with ties preferring fewer validation records"
            ),
            "group_key": "epitope + one space + complete 34-aa HLA pseudosequence",
            "total_groups": self.total_groups,
            "train_groups": self.train_groups,
            "validation_groups": self.validation_group_count,
            "total_records": self.total_records,
            "target_validation_records": self.target_validation_records,
            "validation_records": self.validation_records,
            "train_records": self.total_records - self.validation_records,
        }


def require_worker() -> None:
    require_compute_host("TCR-TRANSLATE materialization", socket.gethostname())


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _canonical_pair(
    source: str,
    target: str,
    *,
    release_split: str,
    line_number: int,
) -> tuple[CanonicalPair | None, tuple[str, ...]]:
    reasons: list[str] = []
    parts = source.split()
    if len(parts) != 2:
        reasons.append("source_component_count_not_two")
    else:
        epitope, hla = parts
        if not epitope or set(epitope).difference(AA_ALPHABET):
            reasons.append("epitope_not_strict_20aa")
        if not hla or set(hla).difference(AA_ALPHABET):
            reasons.append("hla_not_strict_20aa")
        if len(hla) != 34:
            reasons.append("hla_length_not_34")
    if not target or set(target).difference(AA_ALPHABET):
        reasons.append("target_not_strict_20aa")
    if reasons:
        return None, tuple(reasons)
    return (
        CanonicalPair(
            release_split=release_split,
            line_number=line_number,
            epitope=parts[0],
            hla_pseudosequence=parts[1],
            cdr3b=target,
        ),
        (),
    )


def load_parallel_release(
    source_path: Path,
    target_path: Path,
    *,
    release_split: str,
    expected_raw: int,
    expected_canonical: int,
) -> tuple[list[CanonicalPair], ReleaseAudit]:
    source_path = require_file(source_path, f"{release_split} source")
    target_path = require_file(target_path, f"{release_split} target")
    records: list[CanonicalPair] = []
    reasons: Counter[str] = Counter()
    raw_rows = 0
    with source_path.open("r", encoding="utf-8") as source_handle, target_path.open(
        "r", encoding="utf-8"
    ) as target_handle:
        for line_number, pair in enumerate(
            zip_longest(source_handle, target_handle), start=1
        ):
            source_line, target_line = pair
            if source_line is None or target_line is None:
                raise ValueError(
                    f"unpaired {release_split} source/target files at line {line_number}"
                )
            raw_rows += 1
            record, rejected_for = _canonical_pair(
                source_line.strip(),
                target_line.strip(),
                release_split=release_split,
                line_number=line_number,
            )
            if record is None:
                reasons.update(rejected_for)
            else:
                records.append(record)
    if raw_rows != expected_raw:
        raise ValueError(
            f"{release_split} raw row count changed: expected={expected_raw} "
            f"actual={raw_rows}"
        )
    if len(records) != expected_canonical:
        raise ValueError(
            f"{release_split} canonical row count changed: "
            f"expected={expected_canonical} actual={len(records)} "
            f"reasons={dict(reasons)}"
        )
    return records, ReleaseAudit(
        raw_rows=raw_rows,
        canonical_rows=len(records),
        rejected_rows=raw_rows - len(records),
        rejection_reasons=dict(reasons),
    )


def stable_group_score(source_key: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{source_key}".encode("utf-8")).digest()


def closest_group_disjoint_validation(
    records: Sequence[CanonicalPair],
    *,
    fraction: float = INNER_VALIDATION_FRACTION,
    seed: int = SPLIT_SEED,
) -> GroupSplit:
    if not 0.0 < fraction < 1.0:
        raise ValueError("inner-validation fraction must be in (0, 1)")
    sizes = Counter(record.source_key for record in records)
    if len(sizes) < 2:
        raise ValueError("group-disjoint split requires at least two complete pMHC groups")
    ordered = sorted(
        sizes,
        key=lambda source_key: (
            stable_group_score(source_key, seed=seed),
            source_key,
        ),
    )
    target_records = max(1, min(len(records) - 1, int(round(len(records) * fraction))))
    cumulative = 0
    best: tuple[int, int, int] | None = None
    for prefix_length, source_key in enumerate(ordered[:-1], start=1):
        cumulative += sizes[source_key]
        candidate = (abs(cumulative - target_records), cumulative, prefix_length)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("no non-empty group-disjoint validation prefix exists")
    _distance, validation_records, prefix_length = best
    selected = frozenset(ordered[:prefix_length])
    if validation_records != sum(sizes[key] for key in selected):
        raise RuntimeError("internal pMHC group split accounting mismatch")
    return GroupSplit(
        validation_groups=selected,
        total_groups=len(sizes),
        train_groups=len(sizes) - len(selected),
        validation_group_count=len(selected),
        total_records=len(records),
        target_validation_records=target_records,
        validation_records=validation_records,
    )


def _validate_raw_bpe_ids(values: Sequence[int], *, label: str) -> list[int]:
    result = [int(value) for value in values]
    if not result:
        raise ValueError(f"{label}: frozen BPE produced an empty sequence")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        or value < 0 or value > RAW_BPE_MAX_ID
        for value in values
    ):
        raise ValueError(f"{label}: invalid raw frozen-BPE id")
    return result


class TcrPairEncoder:
    """Cache deterministic sequence encodings without changing codec semantics."""

    def __init__(self, codec: AxisCodec, *, context_budget: int = CONTEXT_BUDGET) -> None:
        self.codec = codec
        self.context_budget = int(context_budget)
        self._component_cache: dict[
            tuple[str, str], tuple[tuple[str, ...], tuple[int, ...]]
        ] = {}
        self._condition_cache: dict[str, tuple[int, ...]] = {}
        self._target_encoding_calls = 0

    def _component_encoding(
        self, target_type: str, sequence: str, label: str
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        key = (target_type, sequence)
        cached = self._component_cache.get(key)
        if cached is not None:
            return cached
        _metrics, tokens, bpe = self.codec.encode_sequence(
            target_type,
            sequence,
            label,
        )
        encoded = (
            tuple(str(token) for token in tokens),
            tuple(_validate_raw_bpe_ids(bpe, label=label)),
        )
        self._component_cache[key] = encoded
        return encoded

    def encode_condition(
        self,
        epitope: str,
        hla_pseudosequence: str,
        *,
        label: str,
    ) -> list[int]:
        source_key = f"{epitope} {hla_pseudosequence}"
        condition = self._condition_cache.get(source_key)
        if condition is None:
            epitope_tokens, _epitope_bpe = self._component_encoding(
                "peptide", epitope, f"{label}:epitope"
            )
            hla_tokens, _hla_bpe = self._component_encoding(
                "protein",
                hla_pseudosequence,
                f"{label}:hla-pseudosequence",
            )
            joined_tokens = [*epitope_tokens, "<SEP>", *hla_tokens]
            condition = tuple(
                _validate_raw_bpe_ids(
                    self.codec.bpe_ids(
                        joined_tokens,
                        f"{label}:joined-pmhc-condition",
                    ),
                    label=f"{label}:joined-pmhc-condition",
                )
            )
            self._condition_cache[source_key] = condition
        return list(condition)

    def encode(self, pair: CanonicalPair) -> tuple[list[int], list[int], int]:
        condition = self.encode_condition(
            pair.epitope,
            pair.hla_pseudosequence,
            label=pair.record_id,
        )
        _target_metrics, _target_tokens, raw_target = self.codec.encode_sequence(
            "peptide",
            pair.cdr3b,
            f"{pair.record_id}:cdr3b-target",
        )
        self._target_encoding_calls += 1
        target = _validate_raw_bpe_ids(
            raw_target, label=f"{pair.record_id}:cdr3b-target"
        )
        full_context = len(condition) + len(target) + TARGET_FRAME_TOKENS
        if full_context > self.context_budget:
            raise ValueError(
                f"{pair.record_id}: full context {full_context} exceeds "
                f"{self.context_budget}; formal TCR data must not be truncated"
            )
        return condition, list(target), full_context

    @property
    def unique_component_encodings(self) -> int:
        return len(self._component_cache)

    @property
    def unique_condition_encodings(self) -> int:
        return len(self._condition_cache)

    @property
    def target_encoding_calls(self) -> int:
        return self._target_encoding_calls


def quantile(values: Sequence[int], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(int(value) for value in values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (rank - low) * (ordered[high] - ordered[low]))


def summarize(values: Iterable[int]) -> dict[str, Any]:
    items = [int(value) for value in values]
    if not items:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(items),
        "min": min(items),
        "mean": sum(items) / len(items),
        "p50": quantile(items, 0.50),
        "p95": quantile(items, 0.95),
        "p99": quantile(items, 0.99),
        "max": max(items),
    }


def axis_record(
    pair: CanonicalPair,
    *,
    split: str,
    condition_bpe_ids: Sequence[int],
    target_bpe_ids: Sequence[int],
    full_context_tokens: int,
) -> dict[str, Any]:
    return {
        "id": pair.record_id,
        "task": TASK,
        "target_type": "peptide",
        "condition": {
            "type": "molecule",
            "molecule_bpe_ids": list(condition_bpe_ids),
        },
        "target_bpe_ids": list(target_bpe_ids),
        "meta": {
            "benchmark": "TCR-TRANSLATE released pMHC-to-CDR3beta",
            "split": split,
            "released_split": pair.release_split,
            "source_line_number": pair.line_number,
            "epitope_residues": len(pair.epitope),
            "hla_pseudosequence_residues": len(pair.hla_pseudosequence),
            "target_residues": len(pair.cdr3b),
            "molecule_prefix_tokens": len(condition_bpe_ids),
            "target_bpe_tokens": len(target_bpe_ids),
            "full_context_tokens": int(full_context_tokens),
            "target_truncated": False,
        },
    }


def reference_record(pair: CanonicalPair, *, split: str) -> dict[str, Any]:
    return {
        "id": pair.record_id,
        "split": split,
        "released_split": pair.release_split,
        "source_line_number": pair.line_number,
        "epitope": pair.epitope,
        "hla_pseudosequence": pair.hla_pseudosequence,
        "cdr3b": pair.cdr3b,
    }


def validate_index(
    index: AxisBIndex,
    *,
    split_counts: Mapping[str, int],
) -> None:
    indexed_datasets = {str(item.get("dataset")) for item in index.files}
    if indexed_datasets != {DATASET_ID}:
        raise RuntimeError(
            f"indexed dataset mismatch: expected={DATASET_ID!r} "
            f"actual={indexed_datasets}"
        )
    expected = {
        "train": int(split_counts["train"]),
        "eval": int(split_counts["validation"]),
        "test": int(split_counts["test"]),
    }
    for split, expected_count in expected.items():
        actual = int(index.task_counts(split).get(TASK, 0))
        if actual != expected_count:
            raise RuntimeError(
                f"indexed {split} count mismatch: expected={expected_count} "
                f"actual={actual}"
            )


def materialize_release(
    *,
    input_root: Path,
    output_root: Path,
    codec: AxisCodec,
    validation_fraction: float = INNER_VALIDATION_FRACTION,
    split_seed: int = SPLIT_SEED,
    expected_train_raw: int = EXPECTED_TRAIN_RAW,
    expected_train_canonical: int = EXPECTED_TRAIN_CANONICAL,
    expected_released_validation_raw: int = EXPECTED_RELEASED_VALIDATION_RAW,
    expected_released_validation_canonical: int = EXPECTED_RELEASED_VALIDATION_CANONICAL,
    context_budget: int = CONTEXT_BUDGET,
    progress_every: int = 5000,
    allow_non_worker_index: bool = False,
    index_builder: Callable[..., AxisBIndex] = build_axis_b_index,
) -> dict[str, Any]:
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    if int(context_budget) != CONTEXT_BUDGET:
        raise ValueError(f"formal TCR context budget must be {CONTEXT_BUDGET}")
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")

    train_records, train_audit = load_parallel_release(
        input_root / TRAIN_SOURCE_NAME,
        input_root / TRAIN_TARGET_NAME,
        release_split="released-train",
        expected_raw=expected_train_raw,
        expected_canonical=expected_train_canonical,
    )
    released_validation_records, released_validation_audit = load_parallel_release(
        input_root / RELEASED_VALIDATION_SOURCE_NAME,
        input_root / RELEASED_VALIDATION_TARGET_NAME,
        release_split="released-validation",
        expected_raw=expected_released_validation_raw,
        expected_canonical=expected_released_validation_canonical,
    )
    group_split = closest_group_disjoint_validation(
        train_records,
        fraction=validation_fraction,
        seed=split_seed,
    )

    staging.mkdir(parents=True)
    output_paths = {
        "train": staging / "train.jsonl",
        "validation": staging / "validation.jsonl",
        "test": staging / "test.jsonl",
    }
    reference_paths = {
        split: staging / f"{split}_references.jsonl" for split in output_paths
    }
    output_handles = {
        split: path.open("x", encoding="utf-8")
        for split, path in output_paths.items()
    }
    reference_handles = {
        split: path.open("x", encoding="utf-8")
        for split, path in reference_paths.items()
    }
    encoder = TcrPairEncoder(codec, context_budget=context_budget)
    split_counts: Counter[str] = Counter()
    condition_lengths: list[int] = []
    target_lengths: list[int] = []
    context_lengths: list[int] = []
    total_records = len(train_records) + len(released_validation_records)
    try:
        for index, pair in enumerate(
            chain(train_records, released_validation_records), start=1
        ):
            if pair.release_split == "released-train":
                split = (
                    "validation"
                    if pair.source_key in group_split.validation_groups
                    else "train"
                )
            else:
                split = "test"
            condition_bpe, target_bpe, full_context = encoder.encode(pair)
            output_handles[split].write(
                json_line(
                    axis_record(
                        pair,
                        split=split,
                        condition_bpe_ids=condition_bpe,
                        target_bpe_ids=target_bpe,
                        full_context_tokens=full_context,
                    )
                )
            )
            reference_handles[split].write(
                json_line(reference_record(pair, split=split))
            )
            split_counts[split] += 1
            condition_lengths.append(len(condition_bpe))
            target_lengths.append(len(target_bpe))
            context_lengths.append(full_context)
            if index % progress_every == 0 or index == total_records:
                print(
                    json.dumps({"encoded": index, "total": total_records}),
                    flush=True,
                )
    finally:
        for handle in (*output_handles.values(), *reference_handles.values()):
            handle.close()

    expected_split_counts = {
        "train": len(train_records) - group_split.validation_records,
        "validation": group_split.validation_records,
        "test": len(released_validation_records),
    }
    if dict(split_counts) != expected_split_counts:
        raise RuntimeError(
            f"materialized split counts differ: expected={expected_split_counts} "
            f"actual={dict(split_counts)}"
        )
    train_groups = {
        record.source_key
        for record in train_records
        if record.source_key not in group_split.validation_groups
    }
    if train_groups.intersection(group_split.validation_groups):
        raise RuntimeError("complete pMHC source leaked across inner train/validation")
    if max(context_lengths) > context_budget:
        raise RuntimeError("materialized TCR record exceeds formal context budget")

    file_summaries: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = output_paths[split]
        file_summaries.append(
            {
                "dataset": DATASET_ID,
                "split": split,
                "relative_source": path.name,
                "records": int(split_counts[split]),
                "output_size": path.stat().st_size,
            }
        )
    split_description = group_split.as_dict()
    split_description.update(
        {
            "fraction": validation_fraction,
            "seed": split_seed,
            "group_disjoint": True,
            "source": "strict canonical rows from released training pair only",
            "released_validation_used_for_checkpoint_selection": False,
        }
    )
    manifest = {
        "complete": True,
        "task": TASK,
        "dataset_id": DATASET_ID,
        "files": len(file_summaries),
        "records": sum(int(item["records"]) for item in file_summaries),
        "file_summaries": file_summaries,
        "protocol": {
            "name": PROTOCOL,
            "serializer": "fddfs",
            "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
            "model_token_offset": 3,
            "target_type": "peptide",
            "context_budget": CONTEXT_BUDGET,
            "target_frame_tokens": TARGET_FRAME_TOKENS,
            "condition": {
                "schema": "molecule_bpe_ids only; no text or metadata prompt",
                "construction": (
                    "FDDFS(peptide epitope) + raw <SEP> + "
                    "FDDFS(protein HLA-34), then frozen BPE over the whole joined stream"
                ),
                "vocabulary_extension": False,
                "stackk": False,
            },
            "inner_validation": split_description,
            "released_validation_auxiliary": {
                "source": "released validation source/target pair",
                "index_split_name": "test",
                "purpose": "released top-20 auxiliary reference/evaluation only",
                "headline": False,
                "used_for_checkpoint_selection": False,
                "preserve_duplicate_source_target_rows": True,
            },
            "headline_test": {
                "track": "external13_sparse_headline",
                "source": "benchmark_data_w_preds.csv derived from the released mixed-source external benchmark",
                "groups": 13,
                "excluded_deep_simulation_pmhc": "RVRAYTYSK_A*03:01",
                "materialized_by": "materialize_tcr_translate_external13_headline_worker.py",
                "used_for_checkpoint_selection": False,
            },
        },
        "source_files": {
            "released_train_source": str((input_root / TRAIN_SOURCE_NAME).resolve()),
            "released_train_target": str((input_root / TRAIN_TARGET_NAME).resolve()),
            "released_validation_source": str(
                (input_root / RELEASED_VALIDATION_SOURCE_NAME).resolve()
            ),
            "released_validation_target": str(
                (input_root / RELEASED_VALIDATION_TARGET_NAME).resolve()
            ),
            "released_grouped_validation_table": str(
                (input_root / "topk_val_df.csv").resolve()
            ),
            "released_external_benchmark_table": str(
                (input_root / "benchmark_data_w_preds.csv").resolve()
            ),
        },
        "canonical_filter": {
            "source_components": "exactly epitope and HLA pseudosequence",
            "alphabet": "20 canonical amino acids",
            "hla_pseudosequence_length": 34,
            "train": train_audit.as_dict(),
            "released_validation": released_validation_audit.as_dict(),
        },
        "split_counts": dict(expected_split_counts),
        "reference_outputs": {
            split: path.name for split, path in reference_paths.items()
        },
        "condition_bpe_tokens": summarize(condition_lengths),
        "target_bpe_tokens": summarize(target_lengths),
        "full_context_tokens": summarize(context_lengths),
        "all_records_fit_512_without_truncation": True,
        "encoding_cache": {
            "unique_component_encodings": encoder.unique_component_encodings,
            "unique_complete_pmhc_conditions": encoder.unique_condition_encodings,
            "target_encoding_calls": encoder.target_encoding_calls,
        },
        "codec": dict(codec.codebook),
    }
    atomic_json(staging / "manifest.json", manifest)
    index = index_builder(
        staging,
        allow_non_worker=allow_non_worker_index,
    )
    validate_index(index, split_counts=expected_split_counts)
    # The staging directory is atomically renamed below.  Keep the audit-only
    # data_root field pointed at the durable published location instead of the
    # now-defunct ``.building.<pid>`` path.
    index_manifest_path = staging / "axis_b_index_v1" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["data_root"] = str(output_root)
    atomic_json(index_manifest_path, index_manifest)
    os.replace(staging, output_root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    if FORMAL_TASK_DATASET_SCOPES.get(TASK) != (DATASET_ID,):
        raise RuntimeError("TCR task registry and materializer dataset id disagree")
    guide = require_file(args.guide, "FDDFS guide")
    vocab = require_file(args.vocab, "FDDFS vocabulary")
    merges = require_file(args.merges, "frozen BPE merges")
    from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec

    manifest = materialize_release(
        input_root=args.input_root,
        output_root=args.output_root,
        codec=FrozenAxisCodec(guide, vocab, merges),
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "task": TASK,
                "dataset_id": DATASET_ID,
                "output_root": str(args.output_root.expanduser().resolve()),
                "split_counts": manifest["split_counts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
