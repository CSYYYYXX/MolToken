#!/usr/bin/env python3
"""Materialize the Axis DPLM-aligned length-conditioned protein task.

The local source is the project's UniRef50 2026_01 FASTA, not the older
``airkingbd/uniref50`` snapshot used by DPLM.  This materializer therefore
builds a disclosed, pretraining-exposed partial reproduction rather than
claiming exact DPLM training-data identity.

Only strict canonical 20-amino-acid records whose natural lengths are exactly
100, 200, 300, or 400 are retained.  The condition is the bare decimal length
and the target is encoded by the frozen heavy-graph -> FDDFS -> BPE codec.
Every complete SFT record must fit the fixed 512-token Axis context; overflow
is a hard error and no sequence or token truncation is permitted.

The worker runs in two bounded, resumable stages.  Stage 1 scans the gzip once
(``pigz`` when available) and writes only exact canonical candidates into
deterministic static shards.  Stage 2 lets each process initialize one frozen
codec and encode complete shards.  A deterministic held-out test is assigned
first, then a separate hash namespace selects inner validation only from the
test-excluded source pool.  The test split is never a checkpoint-selection
split.  Completed scan and encoded-shard directories are reusable with
``--resume``; interrupted gzip scans restart from the beginning.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.active_scope import FORMAL_TASK_DATASET_SCOPES  # noqa: E402
from pipeline.finetune.axis_b_data import (  # noqa: E402
    RAW_BPE_MAX_ID,
    AxisBIndex,
    build_axis_b_index,
)
from pipeline.finetune.frozen_axis_errors import (  # noqa: E402
    RingIdOverflow,
    UnknownFeatureError,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402


ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / '')
)
DEFAULT_SOURCE = ROOT / "data/raw/protein/uniref50/uniref50.fasta.gz"
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data/processed/axis_b_biological_sft"
    / "dplm_uniref50_length100_400_partial_v1"
)
DEFAULT_GUIDE = ROOT / "data/fddfs_guide_recount_v1/guide_counts.json"
DEFAULT_VOCAB = ROOT / "data/fddfs_guide_recount_v1/vocab.json"
DEFAULT_MERGES = (
    ROOT
    / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1"
    / "codebook_fddfs_prefix70_mf80000_v1/merges.csv"
)
DEFAULT_TOKENIZER = ROOT / "data/pretrained_encoders/all-mpnet-base-v2/tokenizer.json"

TASK = "dplm_length_conditioned_protein"
DATASET_ID = "dplm_uniref50_length100_400_partial_v1"
PROTOCOL = "axis-dplm-uniref50-length100-400-partial-sft-v1-20260830"
SOURCE_RELEASE = "UniRef50 2026_01"
TARGET_LENGTHS = (100, 200, 300, 400)
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
CONTEXT_BUDGET = 512
TARGET_FRAME_TOKENS = 3
SPLIT_SEED = 42
INNER_VALIDATION_FRACTION = 0.05
HELDOUT_TEST_FRACTION = 0.05
RELEASED_CODE_SAMPLES_PER_LENGTH = 40
DEFAULT_CANDIDATE_SHARDS = 128
DEFAULT_READ_THREADS = 8
DEFAULT_ENCODE_WORKERS = max(1, min(32, (os.cpu_count() or 4) - 4))
STAGING_SCHEMA = "axis-dplm-uniref50-candidate-staging-v1"
ENCODED_SHARD_SCHEMA = "axis-dplm-uniref50-encoded-shard-v1"
CONTEXT_OVERFLOW_POLICIES = ("fail", "record-exclude")
CONTEXT_OVERFLOW_REASON = "full_context_tokens_exceeds_context_budget"
REPRESENTATION_LIMIT_POLICIES = ("fail", "record-exclude")
RING_ID_OVERFLOW_REASON = "ring_id_exceeds_frozen_vocab_capacity"
UNKNOWN_FEATURE_REASON = "feature_token_absent_from_frozen_vocab"
REPRESENTATION_LIMIT_REASONS = (
    RING_ID_OVERFLOW_REASON,
    UNKNOWN_FEATURE_REASON,
)
EXCLUSION_REASONS = (CONTEXT_OVERFLOW_REASON, *REPRESENTATION_LIMIT_REASONS)
EXCLUSION_RELATIVE_PATH = "excluded_records.jsonl"


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


class AxisCodecFactory(Protocol):
    def __call__(self) -> AxisCodec: ...


@dataclass(frozen=True)
class FrozenAxisCodecFactory:
    """Pickle-safe factory; every encoder process initializes its own codec."""

    guide_path: Path
    vocab_path: Path
    merges_path: Path

    def __call__(self) -> AxisCodec:
        from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec

        return FrozenAxisCodec(
            self.guide_path,
            self.vocab_path,
            self.merges_path,
        )


@dataclass(frozen=True)
class FastaRecord:
    record_number: int
    accession: str
    header: str
    sequence: str

    @property
    def record_id(self) -> str:
        return f"uniref50:{self.accession}"


@dataclass
class StreamingIntegerStats:
    """Exact small-domain integer summary without retaining every record."""

    count: int = 0
    total: int = 0
    minimum: int | None = None
    maximum: int | None = None
    histogram: Counter[int] = field(default_factory=Counter)

    def add(self, value: int) -> None:
        value = int(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.histogram[value] += 1

    def _value_at(self, index: int) -> int:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        cumulative = 0
        for value, frequency in sorted(self.histogram.items()):
            cumulative += frequency
            if index < cumulative:
                return value
        raise RuntimeError("integer histogram accounting mismatch")

    def quantile(self, q: float) -> float:
        if self.count == 0:
            raise ValueError("cannot summarize empty integer statistics")
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        rank = (self.count - 1) * q
        low = math.floor(rank)
        high = math.ceil(rank)
        low_value = self._value_at(low)
        if low == high:
            return float(low_value)
        high_value = self._value_at(high)
        return float(low_value + (rank - low) * (high_value - low_value))

    def as_dict(self) -> dict[str, int | float]:
        if self.count == 0 or self.minimum is None or self.maximum is None:
            raise ValueError("cannot summarize empty integer statistics")
        return {
            "count": self.count,
            "min": self.minimum,
            "mean": self.total / self.count,
            "p50": self.quantile(0.50),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "max": self.maximum,
        }


def require_worker() -> None:
    require_compute_host("DPLM UniRef50 materialization", socket.gethostname())


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


class _ProcessTextStream:
    """Context manager for a fail-loud external gzip decompressor."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.stream = process.stdout

    def __enter__(self):
        if self.stream is None:
            raise RuntimeError("decompressor stdout pipe is unavailable")
        return self.stream

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stream is not None:
            self.stream.close()
        return_code = self.process.wait()
        if exc_type is None and return_code not in (0, -13):
            raise RuntimeError(
                f"input decompressor failed with return code {return_code}"
            )


def resolve_decompressor(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "pigz", "python-gzip"}:
        raise ValueError(
            "decompressor must be one of: auto, pigz, python-gzip"
        )
    if normalized == "auto":
        return "pigz" if shutil.which("pigz") else "python-gzip"
    if normalized == "pigz" and not shutil.which("pigz"):
        raise FileNotFoundError("pigz was requested but is not installed")
    return normalized


def open_gzip_text(
    path: Path,
    *,
    decompressor: str,
    read_threads: int,
):
    backend = resolve_decompressor(decompressor)
    if backend == "python-gzip":
        return gzip.open(path, "rt", encoding="ascii", errors="strict", newline="")
    process = subprocess.Popen(
        ["pigz", "-dc", "-p", str(read_threads), str(path)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="ascii",
        errors="strict",
        bufsize=1024 * 1024,
    )
    return _ProcessTextStream(process)


def iter_gzip_fasta(
    path: Path,
    *,
    decompressor: str = "python-gzip",
    read_threads: int = 1,
) -> Iterator[FastaRecord]:
    """Yield strict UniRef50 FASTA records without loading the corpus."""

    if read_threads < 1:
        raise ValueError("read_threads must be positive")

    record_number = 0
    header: str | None = None
    sequence_parts: list[str] = []

    def finish_record() -> FastaRecord:
        nonlocal record_number
        if header is None:
            raise RuntimeError("internal FASTA parser state has no header")
        sequence = "".join(sequence_parts)
        if not sequence:
            raise ValueError(f"{path}: FASTA record {record_number + 1} is empty")
        accession = header.split(None, 1)[0]
        if not accession.startswith("UniRef50_"):
            raise ValueError(
                f"{path}: FASTA record {record_number + 1} has non-UniRef50 "
                f"accession {accession!r}"
            )
        record_number += 1
        return FastaRecord(record_number, accession, header, sequence)

    with open_gzip_text(
        path,
        decompressor=decompressor,
        read_threads=read_threads,
    ) as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                raise ValueError(f"{path}:{line_number}: blank FASTA line")
            if line.startswith(">"):
                if len(line) == 1:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
                if header is not None:
                    yield finish_record()
                header = line[1:]
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(
                        f"{path}:{line_number}: sequence data precedes first FASTA header"
                    )
                sequence_parts.append(line)
    if header is None:
        raise ValueError(f"{path}: FASTA contains no records")
    yield finish_record()


def _hash_threshold(fraction: float) -> int:
    if not 0.0 < fraction < 1.0:
        raise ValueError("split fractions must be in (0, 1)")
    return int(fraction * (1 << 64))


def _split_score(record_id: str, *, seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\0{TASK}\0{namespace}\0{record_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def assign_split(
    record_id: str,
    *,
    seed: int = SPLIT_SEED,
    inner_validation_fraction: float = INNER_VALIDATION_FRACTION,
    heldout_test_fraction: float = HELDOUT_TEST_FRACTION,
) -> str:
    """Assign test first, then validation from the independent remaining pool."""

    test_threshold = _hash_threshold(heldout_test_fraction)
    validation_threshold = _hash_threshold(inner_validation_fraction)
    if _split_score(record_id, seed=seed, namespace="heldout-test") < test_threshold:
        return "test"
    if (
        _split_score(record_id, seed=seed, namespace="inner-validation")
        < validation_threshold
    ):
        return "validation"
    return "train"


def condition_token_counts(tokenizer: Any) -> dict[int, int]:
    counts: dict[int, int] = {}
    for length in TARGET_LENGTHS:
        encoded = tokenizer.encode(str(length))
        count = int(sum(int(value) for value in encoded.attention_mask))
        if count <= 0:
            raise ValueError(f"length condition {length} encoded to zero text tokens")
        counts[length] = count
    return counts


def _validate_condition_counts(values: Mapping[int, int]) -> dict[int, int]:
    if set(values) != set(TARGET_LENGTHS):
        raise ValueError(
            f"condition token lengths must be exactly {TARGET_LENGTHS}: "
            f"got {sorted(values)}"
        )
    parsed = {int(length): int(count) for length, count in values.items()}
    if any(count <= 0 for count in parsed.values()):
        raise ValueError("every bare numeric condition must have at least one text token")
    return parsed


def _validate_target_bpe(values: Sequence[int], *, label: str) -> list[int]:
    result = [int(value) for value in values]
    if not result:
        raise ValueError(f"{label}: frozen BPE produced an empty protein target")
    if any(value < 0 or value > RAW_BPE_MAX_ID for value in result):
        raise ValueError(f"{label}: frozen BPE produced an out-of-range raw id")
    return result


def axis_record(
    source: FastaRecord,
    *,
    split: str,
    target_bpe_ids: Sequence[int],
    text_tokens: int,
    full_context_tokens: int,
) -> dict[str, Any]:
    length = len(source.sequence)
    return {
        "id": source.record_id,
        "task": TASK,
        "target_type": "protein",
        "condition": {
            "type": "text",
            "text": str(length),
            "molecule_bpe_ids": [],
        },
        "target_bpe_ids": list(target_bpe_ids),
        "meta": {
            "benchmark": "DPLM-aligned length-100-400 partial protocol",
            "split": split,
            "source_accession": source.accession,
            "source_record_number": source.record_number,
            "source_release": SOURCE_RELEASE,
            "sequence_length": length,
            "requested_length": length,
            "text_tokens": int(text_tokens),
            "text_prefix_tokens": int(text_tokens),
            "molecule_prefix_tokens": 0,
            "target_bpe_tokens": len(target_bpe_ids),
            "full_context_tokens": int(full_context_tokens),
            "context_budget": CONTEXT_BUDGET,
            "target_truncated": False,
            "pretraining_exposed": True,
        },
    }


def _reference_fasta(record: FastaRecord) -> str:
    return f">{record.header}\n{record.sequence}\n"


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
    for index_split, materialized_split in (
        ("train", "train"),
        ("eval", "validation"),
        ("test", "test"),
    ):
        expected = int(split_counts[materialized_split])
        actual = int(index.task_counts(index_split).get(TASK, 0))
        if actual != expected:
            raise RuntimeError(
                f"indexed {index_split} count mismatch: expected={expected} actual={actual}"
            )


def _nested_counts(
    counts: Mapping[str, Counter[int]],
) -> dict[str, dict[str, int]]:
    return {
        split: {str(length): int(by_length[length]) for length in TARGET_LENGTHS}
        for split, by_length in counts.items()
    }


def log_event(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, **fields},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _candidate_shard(record_id: str, shards: int) -> int:
    digest = hashlib.sha256(f"candidate-shard\0{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(shards)


def _load_complete_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("complete"):
        raise ValueError(f"{label} is not marked complete: {path}")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


def scan_uniref50_candidates(
    *,
    source_path: Path,
    staging_root: Path,
    candidate_shards: int,
    inner_validation_fraction: float,
    heldout_test_fraction: float,
    split_seed: int,
    decompressor: str,
    read_threads: int,
    progress_every: int,
    resume: bool,
) -> dict[str, Any]:
    """Stage 1: one raw pass, retaining only exact canonical candidates."""

    if candidate_shards < 1:
        raise ValueError("candidate_shards must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    _hash_threshold(inner_validation_fraction)
    _hash_threshold(heldout_test_fraction)
    source_path = require_file(source_path, "UniRef50 gzip FASTA")
    staging_root.mkdir(parents=True, exist_ok=True)
    scan_dir = staging_root / "scan"
    expected_resume = {
        "schema": STAGING_SCHEMA,
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "target_lengths": list(TARGET_LENGTHS),
        "alphabet": "".join(sorted(AA_ALPHABET)),
        "candidate_shards": int(candidate_shards),
        "split_seed": int(split_seed),
        "inner_validation_fraction": float(inner_validation_fraction),
        "heldout_test_fraction": float(heldout_test_fraction),
    }
    if scan_dir.exists():
        if not resume:
            raise FileExistsError(
                f"candidate scan exists; pass --resume to reuse it: {scan_dir}"
            )
        manifest = _load_complete_json(scan_dir / "manifest.json", "candidate scan manifest")
        for key, expected in expected_resume.items():
            _require_equal(manifest.get(key), expected, f"candidate scan {key}")
        for item in manifest.get("files") or []:
            path = scan_dir / str(item["relative_path"])
            if not path.is_file():
                raise FileNotFoundError(f"candidate shard is missing: {path}")
            _require_equal(
                path.stat().st_size,
                int(item["bytes"]),
                f"candidate shard size {path.name}",
            )
        log_event(
            "dplm_scan_resume",
            scan_dir=str(scan_dir),
            source_records=int(manifest["source_records_scanned"]),
            candidates=int(manifest["canonical_selected_total"]),
        )
        return manifest

    backend = resolve_decompressor(decompressor)
    building = staging_root / f"scan.building.{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"scan build directory already exists: {building}")
    building.mkdir()
    candidate_paths = [
        building / f"candidate-{index:05d}.jsonl"
        for index in range(candidate_shards)
    ]
    shard_records = [0 for _ in candidate_paths]
    shard_bytes = [0 for _ in candidate_paths]
    source_records = 0
    ignored_other_length = 0
    requested_length_records: Counter[int] = Counter()
    canonical_records: Counter[int] = Counter()
    rejected_noncanonical_records: Counter[int] = Counter()
    rejected_characters: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_counts_by_length = {
        split: Counter() for split in ("train", "validation", "test")
    }
    started = time.time()
    log_event(
        "dplm_scan_start",
        source=str(source_path),
        decompressor=backend,
        read_threads=read_threads,
        candidate_shards=candidate_shards,
    )
    with ExitStack() as stack:
        handles = [
            stack.enter_context(path.open("x", encoding="utf-8"))
            for path in candidate_paths
        ]
        for record in iter_gzip_fasta(
            source_path,
            decompressor=backend,
            read_threads=read_threads,
        ):
            source_records += 1
            length = len(record.sequence)
            if length not in TARGET_LENGTHS:
                ignored_other_length += 1
            else:
                requested_length_records[length] += 1
                invalid = set(record.sequence).difference(AA_ALPHABET)
                if invalid:
                    rejected_noncanonical_records[length] += 1
                    rejected_characters.update(invalid)
                else:
                    split = assign_split(
                        record.record_id,
                        seed=split_seed,
                        inner_validation_fraction=inner_validation_fraction,
                        heldout_test_fraction=heldout_test_fraction,
                    )
                    shard = _candidate_shard(record.record_id, candidate_shards)
                    line = json_line(
                        {
                            "record_number": record.record_number,
                            "accession": record.accession,
                            "header": record.header,
                            "sequence": record.sequence,
                            "length": length,
                            "split": split,
                        }
                    )
                    handles[shard].write(line)
                    shard_records[shard] += 1
                    shard_bytes[shard] += len(line.encode("utf-8"))
                    canonical_records[length] += 1
                    split_counts[split] += 1
                    split_counts_by_length[split][length] += 1
            if source_records % progress_every == 0:
                log_event(
                    "dplm_scan_progress",
                    source_records=source_records,
                    candidates=sum(canonical_records.values()),
                    elapsed_sec=round(time.time() - started, 3),
                )

    for length in TARGET_LENGTHS:
        if canonical_records[length] == 0:
            raise ValueError(f"no strict canonical UniRef50 records at length {length}")
        for split in ("train", "validation", "test"):
            if split_counts_by_length[split][length] == 0:
                raise ValueError(
                    f"deterministic split {split!r} is empty at length {length}"
                )
    if sum(split_counts.values()) != sum(canonical_records.values()):
        raise RuntimeError("candidate selection and split accounting differ")

    manifest = {
        "complete": True,
        **expected_resume,
        "source_release": SOURCE_RELEASE,
        "decompressor": backend,
        "read_threads": int(read_threads),
        "source_records_scanned": int(source_records),
        "ignored_nonrequested_length_records": int(ignored_other_length),
        "requested_length_records": {
            str(length): int(requested_length_records[length])
            for length in TARGET_LENGTHS
        },
        "rejected_noncanonical_records": {
            str(length): int(rejected_noncanonical_records[length])
            for length in TARGET_LENGTHS
        },
        "rejected_characters": dict(sorted(rejected_characters.items())),
        "canonical_selected_records": {
            str(length): int(canonical_records[length])
            for length in TARGET_LENGTHS
        },
        "canonical_selected_total": int(sum(canonical_records.values())),
        "split_counts": {
            split: int(split_counts[split])
            for split in ("train", "validation", "test")
        },
        "split_counts_by_length": _nested_counts(split_counts_by_length),
        "files": [
            {
                "shard": index,
                "relative_path": path.name,
                "records": int(shard_records[index]),
                "bytes": int(shard_bytes[index]),
            }
            for index, path in enumerate(candidate_paths)
        ],
        "elapsed_sec": time.time() - started,
        "resume_semantics": (
            "A complete scan directory is reused. An interrupted gzip scan is "
            "rescanned from the beginning because gzip has no reliable record seek."
        ),
    }
    atomic_json(building / "manifest.json", manifest)
    os.replace(building, scan_dir)
    log_event(
        "dplm_scan_complete",
        source_records=source_records,
        candidates=sum(canonical_records.values()),
        elapsed_sec=round(time.time() - started, 3),
        scan_dir=str(scan_dir),
    )
    return manifest


_ENCODER_CODEC: AxisCodec | None = None


def _initialize_encoder(codec_factory: AxisCodecFactory) -> None:
    global _ENCODER_CODEC
    _ENCODER_CODEC = codec_factory()


@dataclass(frozen=True)
class EncodeShardTask:
    shard: int
    candidate_path: Path
    expected_records: int
    expected_bytes: int
    encoded_root: Path
    condition_tokens_by_length: dict[int, int]
    context_budget: int
    split_seed: int
    inner_validation_fraction: float
    heldout_test_fraction: float
    progress_every: int
    context_overflow_policy: str
    representation_limit_policy: str


def representation_limit_exclusion(
    *,
    exc: RingIdOverflow | UnknownFeatureError,
    record: FastaRecord,
    split: str,
    text_condition_tokens: int,
    context_budget: int,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "id": record.record_id,
        "accession": record.accession,
        "source_record_number": record.record_number,
        "length": len(record.sequence),
        "split": split,
        "target_bpe_tokens": None,
        "text_condition_tokens": int(text_condition_tokens),
        "full_context_tokens": None,
        "context_budget": int(context_budget),
        "exception_type": type(exc).__name__,
        "target_truncated": False,
        "target_repaired": False,
    }
    if isinstance(exc, RingIdOverflow):
        common.update(
            {
                "reason": RING_ID_OVERFLOW_REASON,
                "max_ring_id": int(exc.max_ring_id),
                "ring_capacity": int(exc.ring_capacity),
                "first_colliding_token_id": int(exc.first_colliding_token_id),
            }
        )
        return common
    if isinstance(exc, UnknownFeatureError):
        unknown = sorted(str(token) for token in exc.tokens)
        common.update(
            {
                "reason": UNKNOWN_FEATURE_REASON,
                "unknown_feature_count": len(unknown),
                "unknown_node_feature_count": sum(
                    token.startswith("N:") for token in unknown
                ),
                "unknown_edge_feature_count": sum(
                    token.startswith("E:") for token in unknown
                ),
                "unknown_features": unknown,
            }
        )
        return common
    raise TypeError(f"unsupported representation-limit exception: {type(exc).__name__}")


def validate_exclusion_record(
    exclusion: Mapping[str, Any],
    *,
    context_budget: int,
) -> str:
    record_id = str(exclusion["id"])
    reason = str(exclusion["reason"])
    if reason not in EXCLUSION_REASONS:
        raise ValueError(f"{record_id}: unknown exclusion reason {reason!r}")
    _require_equal(
        exclusion.get("target_truncated"),
        False,
        f"{record_id} target_truncated",
    )
    _require_equal(
        exclusion.get("target_repaired"),
        False,
        f"{record_id} target_repaired",
    )
    if reason == CONTEXT_OVERFLOW_REASON:
        full_context = int(exclusion["full_context_tokens"])
        if full_context <= context_budget:
            raise ValueError(
                f"{record_id}: excluded context {full_context} does not exceed "
                f"{context_budget}"
            )
        _require_equal(exclusion.get("exception_type"), None, f"{record_id} exception")
        return reason
    _require_equal(
        exclusion.get("full_context_tokens"),
        None,
        f"{record_id} unavailable context",
    )
    _require_equal(
        exclusion.get("target_bpe_tokens"),
        None,
        f"{record_id} unavailable BPE length",
    )
    if reason == RING_ID_OVERFLOW_REASON:
        _require_equal(
            exclusion.get("exception_type"),
            "RingIdOverflow",
            f"{record_id} exception",
        )
        max_ring_id = int(exclusion["max_ring_id"])
        ring_capacity = int(exclusion["ring_capacity"])
        if max_ring_id <= ring_capacity:
            raise ValueError(
                f"{record_id}: max_ring_id={max_ring_id} does not exceed "
                f"ring_capacity={ring_capacity}"
            )
        return reason
    _require_equal(
        exclusion.get("exception_type"),
        "UnknownFeatureError",
        f"{record_id} exception",
    )
    unknown = [str(token) for token in exclusion["unknown_features"]]
    _require_equal(unknown, sorted(set(unknown)), f"{record_id} unknown features")
    _require_equal(
        int(exclusion["unknown_feature_count"]),
        len(unknown),
        f"{record_id} unknown feature count",
    )
    if not unknown:
        raise ValueError(f"{record_id}: empty unknown feature exclusion")
    return reason


def _encode_candidate_shard(task: EncodeShardTask) -> dict[str, Any]:
    if _ENCODER_CODEC is None:
        raise RuntimeError("encoder process was not initialized")
    if task.context_overflow_policy not in CONTEXT_OVERFLOW_POLICIES:
        raise ValueError(
            f"unknown context overflow policy: {task.context_overflow_policy!r}"
        )
    if task.representation_limit_policy not in REPRESENTATION_LIMIT_POLICIES:
        raise ValueError(
            "unknown representation limit policy: "
            f"{task.representation_limit_policy!r}"
        )
    if not task.candidate_path.is_file():
        raise FileNotFoundError(f"candidate shard is missing: {task.candidate_path}")
    _require_equal(
        task.candidate_path.stat().st_size,
        task.expected_bytes,
        f"candidate shard size {task.candidate_path.name}",
    )
    final_dir = task.encoded_root / f"shard-{task.shard:05d}"
    if final_dir.exists():
        raise FileExistsError(f"encoded shard already exists: {final_dir}")
    building = task.encoded_root / f".shard-{task.shard:05d}.building.{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"encoded shard build directory exists: {building}")
    building.mkdir(parents=True)
    output_path = building / "encoded.jsonl"
    exclusion_path = building / EXCLUSION_RELATIVE_PATH
    split_counts: Counter[str] = Counter()
    split_counts_by_length = {
        split: Counter() for split in ("train", "validation", "test")
    }
    exclusion_counts: Counter[str] = Counter()
    exclusion_counts_by_length: Counter[int] = Counter()
    exclusion_counts_by_split_length = {
        split: Counter() for split in ("train", "validation", "test")
    }
    exclusion_counts_by_reason: Counter[str] = Counter()
    exclusion_counts_by_length_reason = {
        length: Counter() for length in TARGET_LENGTHS
    }
    encoded_records = 0
    excluded_records = 0
    processed_records = 0
    started = time.time()
    with (
        task.candidate_path.open("r", encoding="utf-8") as source,
        output_path.open("x", encoding="utf-8") as output,
        exclusion_path.open("x", encoding="utf-8") as exclusions,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(
                    f"blank candidate line {task.candidate_path}:{line_number}"
                )
            candidate = json.loads(line)
            record = FastaRecord(
                record_number=int(candidate["record_number"]),
                accession=str(candidate["accession"]),
                header=str(candidate["header"]),
                sequence=str(candidate["sequence"]),
            )
            length = len(record.sequence)
            _require_equal(int(candidate["length"]), length, f"{record.record_id} length")
            if length not in TARGET_LENGTHS:
                raise ValueError(
                    f"{record.record_id}: staged non-target natural length {length}"
                )
            invalid = set(record.sequence).difference(AA_ALPHABET)
            if invalid:
                raise ValueError(
                    f"{record.record_id}: staged noncanonical residues {sorted(invalid)}"
                )
            split = str(candidate["split"])
            expected_split = assign_split(
                record.record_id,
                seed=task.split_seed,
                inner_validation_fraction=task.inner_validation_fraction,
                heldout_test_fraction=task.heldout_test_fraction,
            )
            _require_equal(split, expected_split, f"{record.record_id} split")
            text_tokens = int(task.condition_tokens_by_length[length])
            try:
                metrics, _fddfs, raw_target = _ENCODER_CODEC.encode_sequence(
                    "protein",
                    record.sequence,
                    record.record_id,
                    text_condition_tokens=text_tokens,
                )
            except RingIdOverflow as exc:
                if task.representation_limit_policy == "fail":
                    raise
                exclusion = representation_limit_exclusion(
                    exc=exc,
                    record=record,
                    split=split,
                    text_condition_tokens=text_tokens,
                    context_budget=task.context_budget,
                )
                exclusions.write(json_line(exclusion))
                reason = str(exclusion["reason"])
                excluded_records += 1
                exclusion_counts[split] += 1
                exclusion_counts_by_length[length] += 1
                exclusion_counts_by_split_length[split][length] += 1
                exclusion_counts_by_reason[reason] += 1
                exclusion_counts_by_length_reason[length][reason] += 1
            except UnknownFeatureError as exc:
                if task.representation_limit_policy == "fail":
                    raise
                exclusion = representation_limit_exclusion(
                    exc=exc,
                    record=record,
                    split=split,
                    text_condition_tokens=text_tokens,
                    context_budget=task.context_budget,
                )
                exclusions.write(json_line(exclusion))
                reason = str(exclusion["reason"])
                excluded_records += 1
                exclusion_counts[split] += 1
                exclusion_counts_by_length[length] += 1
                exclusion_counts_by_split_length[split][length] += 1
                exclusion_counts_by_reason[reason] += 1
                exclusion_counts_by_length_reason[length][reason] += 1
            else:
                target_bpe = _validate_target_bpe(raw_target, label=record.record_id)
                expected_context = text_tokens + len(target_bpe) + TARGET_FRAME_TOKENS
                reported_context = int(metrics["full_context_tokens"])
                _require_equal(
                    reported_context,
                    expected_context,
                    f"{record.record_id} full context tokens",
                )
                if reported_context > task.context_budget:
                    if task.context_overflow_policy == "fail":
                        raise ValueError(
                            f"{record.record_id}: complete {length}-aa target needs "
                            f"{reported_context} tokens, exceeding {task.context_budget}; "
                            "formal DPLM data must not be truncated"
                        )
                    exclusions.write(
                        json_line(
                            {
                                "id": record.record_id,
                                "accession": record.accession,
                                "source_record_number": record.record_number,
                                "length": length,
                                "split": split,
                                "target_bpe_tokens": len(target_bpe),
                                "text_condition_tokens": text_tokens,
                                "full_context_tokens": reported_context,
                                "context_budget": int(task.context_budget),
                                "reason": CONTEXT_OVERFLOW_REASON,
                                "exception_type": None,
                                "target_truncated": False,
                                "target_repaired": False,
                            }
                        )
                    )
                    excluded_records += 1
                    exclusion_counts[split] += 1
                    exclusion_counts_by_length[length] += 1
                    exclusion_counts_by_split_length[split][length] += 1
                    exclusion_counts_by_reason[CONTEXT_OVERFLOW_REASON] += 1
                    exclusion_counts_by_length_reason[length][
                        CONTEXT_OVERFLOW_REASON
                    ] += 1
                else:
                    output.write(
                        json_line(
                            {
                                "split": split,
                                "record": axis_record(
                                    record,
                                    split=split,
                                    target_bpe_ids=target_bpe,
                                    text_tokens=text_tokens,
                                    full_context_tokens=reported_context,
                                ),
                                "reference_header": record.header,
                                "reference_sequence": record.sequence,
                            }
                        )
                    )
                    encoded_records += 1
                    split_counts[split] += 1
                    split_counts_by_length[split][length] += 1
            processed_records += 1
            if task.progress_every > 0 and processed_records % task.progress_every == 0:
                log_event(
                    "dplm_encode_shard_progress",
                    shard=task.shard,
                    processed_records=processed_records,
                    fit_records=encoded_records,
                    excluded_records=excluded_records,
                    expected_records=task.expected_records,
                    elapsed_sec=round(time.time() - started, 3),
                )
    _require_equal(
        processed_records,
        task.expected_records,
        f"processed shard {task.shard} record count",
    )
    summary = {
        "complete": True,
        "schema": ENCODED_SHARD_SCHEMA,
        "shard": int(task.shard),
        "candidate_path": str(task.candidate_path),
        "candidate_records": int(task.expected_records),
        "candidate_bytes": int(task.expected_bytes),
        "processed_records": int(processed_records),
        "encoded_records": int(encoded_records),
        "excluded_records": int(excluded_records),
        "encoded_relative_path": "encoded.jsonl",
        "encoded_bytes": output_path.stat().st_size,
        "exclusion_relative_path": EXCLUSION_RELATIVE_PATH,
        "exclusion_bytes": exclusion_path.stat().st_size,
        "context_overflow_policy": task.context_overflow_policy,
        "context_overflow_reason": CONTEXT_OVERFLOW_REASON,
        "representation_limit_policy": task.representation_limit_policy,
        "representation_limit_reasons": list(REPRESENTATION_LIMIT_REASONS),
        "condition_tokens_by_length": {
            str(length): int(task.condition_tokens_by_length[length])
            for length in TARGET_LENGTHS
        },
        "context_budget": int(task.context_budget),
        "split_seed": int(task.split_seed),
        "inner_validation_fraction": float(task.inner_validation_fraction),
        "heldout_test_fraction": float(task.heldout_test_fraction),
        "split_counts": {
            split: int(split_counts[split])
            for split in ("train", "validation", "test")
        },
        "split_counts_by_length": _nested_counts(split_counts_by_length),
        "exclusion_counts": {
            split: int(exclusion_counts[split])
            for split in ("train", "validation", "test")
        },
        "exclusion_counts_by_length": {
            str(length): int(exclusion_counts_by_length[length])
            for length in TARGET_LENGTHS
        },
        "exclusion_counts_by_split_length": _nested_counts(
            exclusion_counts_by_split_length
        ),
        "exclusion_counts_by_reason": {
            reason: int(exclusion_counts_by_reason[reason])
            for reason in EXCLUSION_REASONS
        },
        "exclusion_counts_by_length_reason": {
            str(length): {
                reason: int(exclusion_counts_by_length_reason[length][reason])
                for reason in EXCLUSION_REASONS
            }
            for length in TARGET_LENGTHS
        },
        "codec": dict(_ENCODER_CODEC.codebook),
        "elapsed_sec": time.time() - started,
    }
    atomic_json(building / "summary.json", summary)
    os.replace(building, final_dir)
    return summary


def _validate_encoded_resume(
    *,
    summary: Mapping[str, Any],
    directory: Path,
    candidate: Mapping[str, Any],
    token_counts: Mapping[int, int],
    context_budget: int,
    split_seed: int,
    inner_validation_fraction: float,
    heldout_test_fraction: float,
    context_overflow_policy: str,
    representation_limit_policy: str,
) -> dict[str, Any]:
    _require_equal(summary.get("schema"), ENCODED_SHARD_SCHEMA, "encoded schema")
    _require_equal(summary.get("shard"), int(candidate["shard"]), "encoded shard id")
    _require_equal(
        summary.get("candidate_records"),
        int(candidate["records"]),
        "encoded candidate records",
    )
    _require_equal(
        summary.get("candidate_bytes"),
        int(candidate["bytes"]),
        "encoded candidate bytes",
    )
    _require_equal(
        summary.get("condition_tokens_by_length"),
        {str(length): int(token_counts[length]) for length in TARGET_LENGTHS},
        "encoded condition token counts",
    )
    _require_equal(summary.get("context_budget"), int(context_budget), "encoded context budget")
    _require_equal(summary.get("split_seed"), int(split_seed), "encoded split seed")
    _require_equal(
        summary.get("inner_validation_fraction"),
        float(inner_validation_fraction),
        "encoded inner validation fraction",
    )
    _require_equal(
        summary.get("heldout_test_fraction"),
        float(heldout_test_fraction),
        "encoded heldout test fraction",
    )
    encoded_path = directory / str(summary["encoded_relative_path"])
    if not encoded_path.is_file():
        raise FileNotFoundError(f"encoded shard payload is missing: {encoded_path}")
    _require_equal(
        encoded_path.stat().st_size,
        int(summary["encoded_bytes"]),
        f"encoded shard size {encoded_path.name}",
    )
    normalized = dict(summary)
    excluded_records = int(summary.get("excluded_records", 0))
    processed_records = int(
        summary.get(
            "processed_records",
            int(summary["encoded_records"]) + excluded_records,
        )
    )
    _require_equal(
        processed_records,
        int(candidate["records"]),
        "resumed processed candidate records",
    )
    existing_policy = summary.get("context_overflow_policy")
    if existing_policy is None:
        _require_equal(excluded_records, 0, "legacy resumed exclusions")
        _require_equal(
            int(summary["encoded_records"]),
            int(candidate["records"]),
            "legacy resumed fit records",
        )
        log_event(
            "dplm_encode_shard_resume_legacy_no_overflow",
            shard=int(candidate["shard"]),
            requested_policy=context_overflow_policy,
        )
    elif existing_policy != context_overflow_policy and excluded_records > 0:
        raise ValueError(
            f"encoded shard {candidate['shard']} used context policy "
            f"{existing_policy!r} with {excluded_records} exclusions and cannot be "
            f"reused under {context_overflow_policy!r}"
        )
    elif existing_policy != context_overflow_policy:
        log_event(
            "dplm_encode_shard_resume_no_overflow_policy_compatible",
            shard=int(candidate["shard"]),
            existing_policy=existing_policy,
            requested_policy=context_overflow_policy,
        )
    existing_representation_policy = summary.get("representation_limit_policy")
    if existing_representation_policy is None and excluded_records > 0:
        raise ValueError(
            f"encoded shard {candidate['shard']} predates representation-limit "
            f"audit and has {excluded_records} exclusions; it cannot be reused"
        )
    if (
        existing_representation_policy is not None
        and existing_representation_policy != representation_limit_policy
        and excluded_records > 0
    ):
        raise ValueError(
            f"encoded shard {candidate['shard']} used representation policy "
            f"{existing_representation_policy!r} with {excluded_records} exclusions "
            f"and cannot be reused under {representation_limit_policy!r}"
        )
    if existing_representation_policy != representation_limit_policy:
        log_event(
            "dplm_encode_shard_resume_no_exclusion_representation_policy_compatible",
            shard=int(candidate["shard"]),
            existing_policy=existing_representation_policy,
            requested_policy=representation_limit_policy,
        )
    exclusion_relative = summary.get("exclusion_relative_path")
    if exclusion_relative is not None:
        exclusion_path = directory / str(exclusion_relative)
        if not exclusion_path.is_file():
            raise FileNotFoundError(
                f"encoded shard exclusion payload is missing: {exclusion_path}"
            )
        _require_equal(
            exclusion_path.stat().st_size,
            int(summary.get("exclusion_bytes", -1)),
            f"encoded exclusion size {exclusion_path.name}",
        )
    elif excluded_records > 0:
        raise ValueError(
            f"encoded shard {candidate['shard']} reports exclusions without an exclusion payload"
        )
    normalized.update(
        {
            "processed_records": processed_records,
            "excluded_records": excluded_records,
            "context_overflow_policy": context_overflow_policy,
            "context_overflow_reason": CONTEXT_OVERFLOW_REASON,
            "representation_limit_policy": representation_limit_policy,
            "representation_limit_reasons": list(REPRESENTATION_LIMIT_REASONS),
            "exclusion_relative_path": exclusion_relative,
            "exclusion_bytes": int(summary.get("exclusion_bytes", 0)),
            "exclusion_counts": dict(
                summary.get("exclusion_counts")
                or {split: 0 for split in ("train", "validation", "test")}
            ),
            "exclusion_counts_by_length": dict(
                summary.get("exclusion_counts_by_length")
                or {str(length): 0 for length in TARGET_LENGTHS}
            ),
            "exclusion_counts_by_split_length": dict(
                summary.get("exclusion_counts_by_split_length")
                or {
                    split: {str(length): 0 for length in TARGET_LENGTHS}
                    for split in ("train", "validation", "test")
                }
            ),
            "exclusion_counts_by_reason": dict(
                summary.get("exclusion_counts_by_reason")
                or {reason: 0 for reason in EXCLUSION_REASONS}
            ),
            "exclusion_counts_by_length_reason": dict(
                summary.get("exclusion_counts_by_length_reason")
                or {
                    str(length): {reason: 0 for reason in EXCLUSION_REASONS}
                    for length in TARGET_LENGTHS
                }
            ),
        }
    )
    _require_equal(
        sum(int(value) for value in normalized["exclusion_counts_by_reason"].values()),
        excluded_records,
        "resumed exclusion reason count",
    )
    return normalized


def encode_candidate_shards(
    *,
    scan_manifest: Mapping[str, Any],
    staging_root: Path,
    codec_factory: AxisCodecFactory,
    condition_tokens_by_length: Mapping[int, int],
    context_budget: int,
    encode_workers: int,
    encode_progress_every: int,
    mp_start_method: str,
    context_overflow_policy: str,
    representation_limit_policy: str = "fail",
    resume: bool = False,
) -> dict[str, Any]:
    """Stage 2: static candidate shards encoded by process-local codecs."""

    if encode_workers < 1:
        raise ValueError("encode_workers must be positive")
    if encode_progress_every < 0:
        raise ValueError("encode_progress_every must be non-negative")
    if context_overflow_policy not in CONTEXT_OVERFLOW_POLICIES:
        raise ValueError(
            f"context_overflow_policy must be one of {CONTEXT_OVERFLOW_POLICIES}"
        )
    if representation_limit_policy not in REPRESENTATION_LIMIT_POLICIES:
        raise ValueError(
            "representation_limit_policy must be one of "
            f"{REPRESENTATION_LIMIT_POLICIES}"
        )
    token_counts = _validate_condition_counts(condition_tokens_by_length)
    scan_dir = staging_root / "scan"
    encoded_root = staging_root / "encoded"
    encoded_root.mkdir(parents=True, exist_ok=True)
    incomplete_dirs = sorted(encoded_root.glob(".shard-*.building.*"))
    if incomplete_dirs and not resume:
        raise FileExistsError(
            f"incomplete encoded shard directories exist; pass --resume: {incomplete_dirs[0]}"
        )
    for incomplete in incomplete_dirs:
        if not incomplete.is_dir():
            raise ValueError(f"incomplete encoded shard path is not a directory: {incomplete}")
        log_event(
            "dplm_encode_recovery_remove_incomplete",
            path=str(incomplete),
        )
        shutil.rmtree(incomplete)
    candidates = sorted(scan_manifest.get("files") or [], key=lambda item: int(item["shard"]))
    summaries: dict[int, dict[str, Any]] = {}
    tasks: list[EncodeShardTask] = []
    split_seed = int(scan_manifest["split_seed"])
    inner_validation_fraction = float(scan_manifest["inner_validation_fraction"])
    heldout_test_fraction = float(scan_manifest["heldout_test_fraction"])
    for candidate in candidates:
        shard = int(candidate["shard"])
        candidate_path = scan_dir / str(candidate["relative_path"])
        final_dir = encoded_root / f"shard-{shard:05d}"
        if final_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"encoded shard exists; pass --resume to reuse it: {final_dir}"
                )
            summary = _load_complete_json(final_dir / "summary.json", "encoded shard summary")
            excluded_records = int(summary.get("excluded_records", 0))
            incompatible_policy: str | None = None
            if excluded_records > 0:
                existing_context_policy = summary.get("context_overflow_policy")
                existing_representation_policy = summary.get(
                    "representation_limit_policy"
                )
                if existing_context_policy != context_overflow_policy:
                    incompatible_policy = "context-overflow-policy"
                elif existing_representation_policy != representation_limit_policy:
                    incompatible_policy = "representation-limit-policy"
                elif summary.get("exclusion_counts_by_reason") is None:
                    incompatible_policy = "missing-reason-audit"
            if incompatible_policy is not None:
                log_event(
                    "dplm_encode_recovery_rebuild_incompatible_shard",
                    shard=shard,
                    excluded_records=excluded_records,
                    reason=incompatible_policy,
                    path=str(final_dir),
                )
                shutil.rmtree(final_dir)
            else:
                summary = _validate_encoded_resume(
                    summary=summary,
                    directory=final_dir,
                    candidate=candidate,
                    token_counts=token_counts,
                    context_budget=context_budget,
                    split_seed=split_seed,
                    inner_validation_fraction=inner_validation_fraction,
                    heldout_test_fraction=heldout_test_fraction,
                    context_overflow_policy=context_overflow_policy,
                    representation_limit_policy=representation_limit_policy,
                )
                summaries[shard] = summary
                log_event(
                    "dplm_encode_shard_resume",
                    shard=shard,
                    records=int(summary["encoded_records"]),
                )
                continue
        tasks.append(
            EncodeShardTask(
                shard=shard,
                candidate_path=candidate_path,
                expected_records=int(candidate["records"]),
                expected_bytes=int(candidate["bytes"]),
                encoded_root=encoded_root,
                condition_tokens_by_length=token_counts,
                context_budget=int(context_budget),
                split_seed=split_seed,
                inner_validation_fraction=inner_validation_fraction,
                heldout_test_fraction=heldout_test_fraction,
                progress_every=int(encode_progress_every),
                context_overflow_policy=context_overflow_policy,
                representation_limit_policy=representation_limit_policy,
            )
        )

    started = time.time()
    if tasks:
        workers = min(int(encode_workers), len(tasks))
        log_event(
            "dplm_encode_start",
            pending_shards=len(tasks),
            reused_shards=len(summaries),
            workers=workers,
            mp_start_method=mp_start_method,
        )
        context = mp.get_context(mp_start_method)
        with context.Pool(
            processes=workers,
            initializer=_initialize_encoder,
            initargs=(codec_factory,),
        ) as pool:
            for result in pool.imap_unordered(_encode_candidate_shard, tasks, chunksize=1):
                shard = int(result["shard"])
                summaries[shard] = result
                log_event(
                    "dplm_encode_shard_complete",
                    shard=shard,
                    fit_records=int(result["encoded_records"]),
                    excluded_records=int(result["excluded_records"]),
                    completed_shards=len(summaries),
                    total_shards=len(candidates),
                    elapsed_sec=round(time.time() - started, 3),
                )
    _require_equal(len(summaries), len(candidates), "encoded shard count")
    ordered = [summaries[int(candidate["shard"])] for candidate in candidates]
    encoded_records = sum(int(item["encoded_records"]) for item in ordered)
    excluded_records = sum(int(item["excluded_records"]) for item in ordered)
    processed_records = sum(int(item["processed_records"]) for item in ordered)
    _require_equal(
        processed_records,
        int(scan_manifest["canonical_selected_total"]),
        "processed total records",
    )
    _require_equal(
        encoded_records + excluded_records,
        processed_records,
        "fit plus excluded records",
    )
    codec_variants = {
        json.dumps(item["codec"], ensure_ascii=False, sort_keys=True)
        for item in ordered
    }
    if len(codec_variants) != 1:
        raise ValueError("encoded shards were produced with inconsistent codec metadata")
    codec = json.loads(next(iter(codec_variants)))
    exclusion_counts_by_reason = {
        reason: sum(
            int(item["exclusion_counts_by_reason"][reason]) for item in ordered
        )
        for reason in EXCLUSION_REASONS
    }
    exclusion_counts_by_length_reason = {
        str(length): {
            reason: sum(
                int(item["exclusion_counts_by_length_reason"][str(length)][reason])
                for item in ordered
            )
            for reason in EXCLUSION_REASONS
        }
        for length in TARGET_LENGTHS
    }
    _require_equal(
        sum(exclusion_counts_by_reason.values()),
        excluded_records,
        "encoded exclusion reason count",
    )
    manifest = {
        "complete": True,
        "schema": ENCODED_SHARD_SCHEMA,
        "context_overflow_policy": context_overflow_policy,
        "context_overflow_reason": CONTEXT_OVERFLOW_REASON,
        "representation_limit_policy": representation_limit_policy,
        "representation_limit_reasons": list(REPRESENTATION_LIMIT_REASONS),
        "shards": len(ordered),
        "records": int(encoded_records),
        "candidate_records": int(processed_records),
        "excluded_records": int(excluded_records),
        "exclusion_counts_by_reason": exclusion_counts_by_reason,
        "exclusion_counts_by_length_reason": exclusion_counts_by_length_reason,
        "codec": codec,
        "summaries": ordered,
        "elapsed_sec_this_invocation": time.time() - started,
        "resume": bool(resume),
    }
    atomic_json(encoded_root / "manifest.json", manifest)
    log_event(
        "dplm_encode_complete",
        shards=len(ordered),
        fit_records=encoded_records,
        excluded_records=excluded_records,
        candidate_records=processed_records,
        elapsed_sec=round(time.time() - started, 3),
    )
    return manifest


def publish_materialization(
    *,
    source_path: Path,
    output_root: Path,
    staging_root: Path,
    scan_manifest: Mapping[str, Any],
    encoded_manifest: Mapping[str, Any],
    condition_tokens_by_length: Mapping[int, int],
    context_budget: int,
    allow_non_worker_index: bool,
    index_builder: Callable[..., AxisBIndex],
) -> dict[str, Any]:
    """Deterministically merge encoded shards, index, then atomically publish."""

    token_counts = _validate_condition_counts(condition_tokens_by_length)
    building = output_root.with_name(f".{output_root.name}.publishing.{os.getpid()}")
    if building.exists():
        raise FileExistsError(f"publish build directory already exists: {building}")
    building.mkdir(parents=True)
    output_paths = {
        split: building / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    reference_paths = {
        "validation": building / "validation_references.fasta",
        "test": building / "test_references.fasta",
    }
    exclusion_output_path = building / EXCLUSION_RELATIVE_PATH
    split_counts: Counter[str] = Counter()
    split_counts_by_length = {
        split: Counter() for split in ("train", "validation", "test")
    }
    target_stats = StreamingIntegerStats()
    context_stats = StreamingIntegerStats()
    target_stats_by_length = {
        length: StreamingIntegerStats() for length in TARGET_LENGTHS
    }
    context_stats_by_length = {
        length: StreamingIntegerStats() for length in TARGET_LENGTHS
    }
    exclusion_counts: Counter[str] = Counter()
    exclusion_counts_by_length: Counter[int] = Counter()
    exclusion_counts_by_split_length = {
        split: Counter() for split in ("train", "validation", "test")
    }
    exclusion_counts_by_reason: Counter[str] = Counter()
    exclusion_counts_by_length_reason = {
        length: Counter() for length in TARGET_LENGTHS
    }
    exclusion_counts_by_split_reason = {
        split: Counter() for split in ("train", "validation", "test")
    }
    exclusion_counts_by_split_length_reason = {
        split: {length: Counter() for length in TARGET_LENGTHS}
        for split in ("train", "validation", "test")
    }
    started = time.time()
    with ExitStack() as stack:
        outputs = {
            split: stack.enter_context(path.open("x", encoding="utf-8"))
            for split, path in output_paths.items()
        }
        references = {
            split: stack.enter_context(path.open("x", encoding="utf-8"))
            for split, path in reference_paths.items()
        }
        exclusion_output = stack.enter_context(
            exclusion_output_path.open("x", encoding="utf-8")
        )
        for summary in sorted(
            encoded_manifest["summaries"], key=lambda item: int(item["shard"])
        ):
            shard = int(summary["shard"])
            encoded_path = (
                staging_root
                / "encoded"
                / f"shard-{shard:05d}"
                / str(summary["encoded_relative_path"])
            )
            observed = 0
            with encoded_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ValueError(f"blank encoded line {encoded_path}:{line_number}")
                    payload = json.loads(line)
                    split = str(payload["split"])
                    if split not in outputs:
                        raise ValueError(f"unknown split {split!r} in {encoded_path}:{line_number}")
                    record = payload["record"]
                    _require_equal(
                        record["meta"]["split"],
                        split,
                        f"encoded split {encoded_path}:{line_number}",
                    )
                    outputs[split].write(json_line(record))
                    if split in references:
                        references[split].write(
                            f">{payload['reference_header']}\n"
                            f"{payload['reference_sequence']}\n"
                        )
                    length = int(record["meta"]["sequence_length"])
                    target_length = len(record["target_bpe_ids"])
                    full_context = int(record["meta"]["full_context_tokens"])
                    if full_context > context_budget:
                        raise ValueError(
                            f"{record['id']}: merged context {full_context} exceeds {context_budget}"
                        )
                    split_counts[split] += 1
                    split_counts_by_length[split][length] += 1
                    target_stats.add(target_length)
                    context_stats.add(full_context)
                    target_stats_by_length[length].add(target_length)
                    context_stats_by_length[length].add(full_context)
                    observed += 1
            _require_equal(
                observed,
                int(summary["encoded_records"]),
                f"merged encoded shard {shard} records",
            )
            exclusion_relative = summary.get("exclusion_relative_path")
            observed_exclusions = 0
            shard_exclusion_counts: Counter[str] = Counter()
            shard_exclusion_counts_by_length: Counter[int] = Counter()
            shard_exclusion_counts_by_reason: Counter[str] = Counter()
            shard_exclusion_counts_by_length_reason = {
                length: Counter() for length in TARGET_LENGTHS
            }
            if exclusion_relative is not None:
                exclusion_path = (
                    staging_root
                    / "encoded"
                    / f"shard-{shard:05d}"
                    / str(exclusion_relative)
                )
                with exclusion_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            raise ValueError(
                                f"blank exclusion line {exclusion_path}:{line_number}"
                            )
                        exclusion = json.loads(line)
                        split = str(exclusion["split"])
                        length = int(exclusion["length"])
                        if split not in split_counts_by_length:
                            raise ValueError(
                                f"unknown exclusion split {split!r} in "
                                f"{exclusion_path}:{line_number}"
                            )
                        if length not in TARGET_LENGTHS:
                            raise ValueError(
                                f"unknown exclusion length {length} in "
                                f"{exclusion_path}:{line_number}"
                            )
                        reason = validate_exclusion_record(
                            exclusion,
                            context_budget=context_budget,
                        )
                        exclusion_output.write(json_line(exclusion))
                        observed_exclusions += 1
                        shard_exclusion_counts[split] += 1
                        shard_exclusion_counts_by_length[length] += 1
                        shard_exclusion_counts_by_reason[reason] += 1
                        shard_exclusion_counts_by_length_reason[length][reason] += 1
                        exclusion_counts[split] += 1
                        exclusion_counts_by_length[length] += 1
                        exclusion_counts_by_split_length[split][length] += 1
                        exclusion_counts_by_reason[reason] += 1
                        exclusion_counts_by_length_reason[length][reason] += 1
                        exclusion_counts_by_split_reason[split][reason] += 1
                        exclusion_counts_by_split_length_reason[split][length][reason] += 1
            _require_equal(
                observed_exclusions,
                int(summary["excluded_records"]),
                f"merged encoded shard {shard} exclusions",
            )
            _require_equal(
                {
                    split: int(shard_exclusion_counts[split])
                    for split in ("train", "validation", "test")
                },
                {
                    split: int(summary["exclusion_counts"][split])
                    for split in ("train", "validation", "test")
                },
                f"merged encoded shard {shard} exclusion split counts",
            )
            _require_equal(
                {
                    str(length): int(shard_exclusion_counts_by_length[length])
                    for length in TARGET_LENGTHS
                },
                {
                    str(length): int(summary["exclusion_counts_by_length"][str(length)])
                    for length in TARGET_LENGTHS
                },
                f"merged encoded shard {shard} exclusion length counts",
            )
            _require_equal(
                {
                    reason: int(shard_exclusion_counts_by_reason[reason])
                    for reason in EXCLUSION_REASONS
                },
                {
                    reason: int(summary["exclusion_counts_by_reason"][reason])
                    for reason in EXCLUSION_REASONS
                },
                f"merged encoded shard {shard} exclusion reason counts",
            )
            _require_equal(
                {
                    str(length): {
                        reason: int(
                            shard_exclusion_counts_by_length_reason[length][reason]
                        )
                        for reason in EXCLUSION_REASONS
                    }
                    for length in TARGET_LENGTHS
                },
                {
                    str(length): {
                        reason: int(
                            summary["exclusion_counts_by_length_reason"][str(length)][
                                reason
                            ]
                        )
                        for reason in EXCLUSION_REASONS
                    }
                    for length in TARGET_LENGTHS
                },
                f"merged encoded shard {shard} exclusion length/reason counts",
            )

    expected_split_counts = {
        split: int(scan_manifest["split_counts"][split])
        - int(exclusion_counts[split])
        for split in ("train", "validation", "test")
    }
    _require_equal(
        {split: int(split_counts[split]) for split in expected_split_counts},
        expected_split_counts,
        "published split counts",
    )
    _require_equal(
        sum(exclusion_counts.values()),
        int(encoded_manifest["excluded_records"]),
        "published exclusion count",
    )
    _require_equal(
        {
            reason: int(exclusion_counts_by_reason[reason])
            for reason in EXCLUSION_REASONS
        },
        {
            reason: int(encoded_manifest["exclusion_counts_by_reason"][reason])
            for reason in EXCLUSION_REASONS
        },
        "published exclusion reason counts",
    )
    _require_equal(
        {
            str(length): {
                reason: int(exclusion_counts_by_length_reason[length][reason])
                for reason in EXCLUSION_REASONS
            }
            for length in TARGET_LENGTHS
        },
        {
            str(length): {
                reason: int(
                    encoded_manifest["exclusion_counts_by_length_reason"][str(length)][
                        reason
                    ]
                )
                for reason in EXCLUSION_REASONS
            }
            for length in TARGET_LENGTHS
        },
        "published exclusion length/reason counts",
    )
    _require_equal(
        sum(split_counts.values()),
        int(encoded_manifest["records"]),
        "published fit record count",
    )
    for length in TARGET_LENGTHS:
        for split in ("train", "validation", "test"):
            if split_counts_by_length[split][length] == 0:
                raise ValueError(
                    f"published split {split!r} is empty at length {length}"
                )

    file_summaries = [
        {
            "dataset": DATASET_ID,
            "split": split,
            "relative_source": path.name,
            "records": int(split_counts[split]),
            "output_size": path.stat().st_size,
        }
        for split, path in output_paths.items()
    ]
    canonical_records = {
        length: int(scan_manifest["canonical_selected_records"][str(length)])
        for length in TARGET_LENGTHS
    }
    exclusion_reason_counts = {
        reason: int(exclusion_counts_by_reason[reason])
        for reason in EXCLUSION_REASONS
    }
    exclusion_length_reason_counts = {
        str(length): {
            reason: int(exclusion_counts_by_length_reason[length][reason])
            for reason in EXCLUSION_REASONS
        }
        for length in TARGET_LENGTHS
    }
    by_length_audit = {
        str(length): {
            "condition": str(length),
            "condition_text_tokens": token_counts[length],
            "canonical_source_records": canonical_records[length],
            "excluded_records": int(exclusion_counts_by_length[length]),
            "excluded_records_by_reason": exclusion_length_reason_counts[str(length)],
            "context_overflow_excluded_records": int(
                exclusion_counts_by_length_reason[length][CONTEXT_OVERFLOW_REASON]
            ),
            "representation_limit_excluded_records": int(
                sum(
                    exclusion_counts_by_length_reason[length][reason]
                    for reason in REPRESENTATION_LIMIT_REASONS
                )
            ),
            "fit_records": int(
                canonical_records[length] - exclusion_counts_by_length[length]
            ),
            "split_counts": {
                split: int(split_counts_by_length[split][length])
                for split in ("train", "validation", "test")
            },
            "target_bpe_tokens": target_stats_by_length[length].as_dict(),
            "full_context_tokens": context_stats_by_length[length].as_dict(),
            "all_sft_records_fit_512_without_truncation": True,
        }
        for length in TARGET_LENGTHS
    }
    generation_audit = {
        "complete": True,
        "task": TASK,
        "dataset_id": DATASET_ID,
        "scope": "DPLM-aligned length-100-400 partial protocol",
        "supported_lengths": list(TARGET_LENGTHS),
        "excluded_lengths": {
            "500": (
                "representation-limited diagnostic only: the frozen Axis encoding "
                "does not provide zero-truncation coverage for all audited 500-aa targets"
            )
        },
        "conditions": by_length_audit,
        "exclusion_audit": {
            "candidate_records": int(encoded_manifest["candidate_records"]),
            "materialized_records": int(encoded_manifest["records"]),
            "excluded_records": int(encoded_manifest["excluded_records"]),
            "excluded_records_by_reason": exclusion_reason_counts,
            "excluded_records_by_length_reason": exclusion_length_reason_counts,
            "target_truncation": False,
            "target_repair": False,
            "exclusion_output": exclusion_output_path.name,
        },
        "context_overflow_audit": {
            "policy": encoded_manifest["context_overflow_policy"],
            "reason": CONTEXT_OVERFLOW_REASON,
            "context_budget": int(context_budget),
            "candidate_records": int(encoded_manifest["candidate_records"]),
            "fit_records": int(encoded_manifest["records"]),
            "excluded_records": exclusion_reason_counts[CONTEXT_OVERFLOW_REASON],
            "excluded_records_by_length": {
                str(length): int(
                    exclusion_counts_by_length_reason[length][CONTEXT_OVERFLOW_REASON]
                )
                for length in TARGET_LENGTHS
            },
            "target_truncation": False,
            "target_repair": False,
            "exclusion_output": exclusion_output_path.name,
        },
        "representation_limit_audit": {
            "policy": encoded_manifest["representation_limit_policy"],
            "reasons": list(REPRESENTATION_LIMIT_REASONS),
            "excluded_records": sum(
                exclusion_reason_counts[reason]
                for reason in REPRESENTATION_LIMIT_REASONS
            ),
            "excluded_records_by_reason": {
                reason: exclusion_reason_counts[reason]
                for reason in REPRESENTATION_LIMIT_REASONS
            },
            "excluded_records_by_length_reason": {
                str(length): {
                    reason: exclusion_length_reason_counts[str(length)][reason]
                    for reason in REPRESENTATION_LIMIT_REASONS
                }
                for length in TARGET_LENGTHS
            },
            "target_truncation": False,
            "target_repair": False,
            "exclusion_output": exclusion_output_path.name,
        },
        "released_code_sampling_reference": {
            "samples_per_length": RELEASED_CODE_SAMPLES_PER_LENGTH,
            "seed": 42,
            "temperature": 1.0,
            "note": (
                "These are DPLM released-code comparison settings. Axis decoding "
                "must be reported as an autoregressive adaptation, not as diffusion "
                "gumbel-argmax sampling."
            ),
        },
        "generation_acceptance": {
            "condition": "bare decimal requested length",
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "exact_requested_residue_length_required": True,
            "score_every_raw_draw": True,
            "post_generation_repair": False,
            "validity_filter_before_metric_aggregation": False,
            "resample_to_replace_invalid_draws": False,
            "test_references_used_for_checkpoint_selection": False,
        },
    }
    atomic_json(building / "generation_audit.json", generation_audit)

    split_seed = int(scan_manifest["split_seed"])
    inner_validation_fraction = float(scan_manifest["inner_validation_fraction"])
    heldout_test_fraction = float(scan_manifest["heldout_test_fraction"])
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
            "target_type": "protein",
            "context_budget": CONTEXT_BUDGET,
            "target_frame_tokens": TARGET_FRAME_TOKENS,
            "selection": {
                "natural_residue_lengths": list(TARGET_LENGTHS),
                "length_padding": False,
                "length_truncation": False,
                "alphabet": "20 canonical amino acids",
                "character_repair_or_replacement": False,
            },
            "condition": {
                "schema": "bare decimal length as text; no instruction or modality prose",
                "values": [str(length) for length in TARGET_LENGTHS],
                "vocabulary_extension": False,
                "molecule_condition": False,
            },
            "split": {
                "seed": split_seed,
                "heldout_test": {
                    "source": f"{SOURCE_RELEASE} canonical selected records",
                    "method": "sha256 heldout-test namespace threshold",
                    "fraction": heldout_test_fraction,
                    "assigned_before_inner_validation": True,
                    "used_for_checkpoint_selection": False,
                },
                "inner_validation": {
                    "source": "test-excluded local UniRef50 source pool",
                    "method": "independent sha256 inner-validation namespace threshold",
                    "fraction_of_test_excluded_pool": inner_validation_fraction,
                    "used_for_early_stopping": True,
                    "official_or_heldout_test_used_for_selection": False,
                },
            },
        },
        "scope_disclosure": {
            "protocol_scope": "DPLM-aligned length-100-400 partial reproduction",
            "pretraining_exposed": True,
            "clean_unseen_downstream_generalization": False,
            "local_source_snapshot": SOURCE_RELEASE,
            "dplm_source_snapshot": "airkingbd/uniref50 Hugging Face snapshot",
            "source_snapshots_identical": False,
            "interpretation": (
                "Measures in-domain length-conditioned adaptation of the Axis "
                "foundation model; it is not exact DPLM data reproduction."
            ),
        },
        "source": {
            "path": str(source_path),
            "format": "single-pass streamed gzip FASTA",
            "release": SOURCE_RELEASE,
            "source_size_bytes": int(scan_manifest["source_size_bytes"]),
            "source_records_scanned": int(scan_manifest["source_records_scanned"]),
            "decompressor": scan_manifest["decompressor"],
            "read_threads": int(scan_manifest["read_threads"]),
        },
        "materialization": {
            "mode": "two-stage-static-candidate-shards",
            "staging_root": str(staging_root),
            "candidate_schema": STAGING_SCHEMA,
            "encoded_shard_schema": ENCODED_SHARD_SCHEMA,
            "candidate_shards": int(scan_manifest["candidate_shards"]),
            "resume_granularity": "complete scan plus each complete encoded shard",
            "context_overflow_policy": encoded_manifest["context_overflow_policy"],
            "representation_limit_policy": encoded_manifest[
                "representation_limit_policy"
            ],
            "deterministic_merge_order": "ascending candidate shard, then source order within shard",
            "publish_elapsed_sec": time.time() - started,
        },
        "selection_audit": {
            "ignored_nonrequested_length_records": int(
                scan_manifest["ignored_nonrequested_length_records"]
            ),
            "requested_length_records": dict(scan_manifest["requested_length_records"]),
            "rejected_noncanonical_records": dict(
                scan_manifest["rejected_noncanonical_records"]
            ),
            "rejected_characters": dict(scan_manifest["rejected_characters"]),
            "canonical_selected_records": dict(
                scan_manifest["canonical_selected_records"]
            ),
        },
        "split_counts": {
            split: int(split_counts[split])
            for split in ("train", "validation", "test")
        },
        "split_counts_by_length": _nested_counts(split_counts_by_length),
        "exclusion_audit": {
            "candidate_records": int(encoded_manifest["candidate_records"]),
            "materialized_records": int(encoded_manifest["records"]),
            "excluded_records": int(encoded_manifest["excluded_records"]),
            "excluded_records_by_split": {
                split: int(exclusion_counts[split])
                for split in ("train", "validation", "test")
            },
            "excluded_records_by_length": {
                str(length): int(exclusion_counts_by_length[length])
                for length in TARGET_LENGTHS
            },
            "excluded_records_by_reason": exclusion_reason_counts,
            "excluded_records_by_length_reason": exclusion_length_reason_counts,
            "target_truncation": False,
            "target_repair": False,
            "exclusion_output": exclusion_output_path.name,
            "exclusion_output_size": exclusion_output_path.stat().st_size,
        },
        "context_overflow_audit": {
            "policy": encoded_manifest["context_overflow_policy"],
            "reason": CONTEXT_OVERFLOW_REASON,
            "context_budget": int(context_budget),
            "candidate_records": int(encoded_manifest["candidate_records"]),
            "fit_records": int(encoded_manifest["records"]),
            "excluded_records": exclusion_reason_counts[CONTEXT_OVERFLOW_REASON],
            "excluded_records_by_split": {
                split: int(
                    exclusion_counts_by_split_reason[split][CONTEXT_OVERFLOW_REASON]
                )
                for split in ("train", "validation", "test")
            },
            "excluded_records_by_length": {
                str(length): int(
                    exclusion_counts_by_length_reason[length][CONTEXT_OVERFLOW_REASON]
                )
                for length in TARGET_LENGTHS
            },
            "excluded_records_by_split_length": {
                split: {
                    str(length): int(
                        exclusion_counts_by_split_length_reason[split][length][
                            CONTEXT_OVERFLOW_REASON
                        ]
                    )
                    for length in TARGET_LENGTHS
                }
                for split in ("train", "validation", "test")
            },
            "target_truncation": False,
            "target_repair": False,
            "exclusion_output": exclusion_output_path.name,
            "exclusion_output_size": exclusion_output_path.stat().st_size,
        },
        "representation_limit_audit": {
            "policy": encoded_manifest["representation_limit_policy"],
            "reasons": list(REPRESENTATION_LIMIT_REASONS),
            "excluded_records": sum(
                exclusion_reason_counts[reason]
                for reason in REPRESENTATION_LIMIT_REASONS
            ),
            "excluded_records_by_reason": {
                reason: exclusion_reason_counts[reason]
                for reason in REPRESENTATION_LIMIT_REASONS
            },
            "excluded_records_by_length_reason": {
                str(length): {
                    reason: exclusion_length_reason_counts[str(length)][reason]
                    for reason in REPRESENTATION_LIMIT_REASONS
                }
                for length in TARGET_LENGTHS
            },
            "target_truncation": False,
            "target_repair": False,
            "exclusion_output": exclusion_output_path.name,
            "exclusion_output_size": exclusion_output_path.stat().st_size,
        },
        "condition_text_tokens": {
            str(length): token_counts[length] for length in TARGET_LENGTHS
        },
        "target_bpe_tokens": target_stats.as_dict(),
        "full_context_tokens": context_stats.as_dict(),
        "target_bpe_tokens_by_length": {
            str(length): target_stats_by_length[length].as_dict()
            for length in TARGET_LENGTHS
        },
        "full_context_tokens_by_length": {
            str(length): context_stats_by_length[length].as_dict()
            for length in TARGET_LENGTHS
        },
        "all_materialized_records_fit_512_without_truncation": True,
        "all_canonical_candidates_fit_512": int(encoded_manifest["excluded_records"]) == 0,
        "reference_outputs": {
            split: path.name for split, path in reference_paths.items()
        },
        "generation_audit_output": "generation_audit.json",
        "exclusion_output": exclusion_output_path.name,
        "codec": dict(encoded_manifest["codec"]),
    }
    atomic_json(building / "manifest.json", manifest)
    index = index_builder(building, allow_non_worker=allow_non_worker_index)
    validate_index(index, split_counts=manifest["split_counts"])
    index_manifest_path = building / "axis_b_index_v1" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["data_root"] = str(output_root)
    atomic_json(index_manifest_path, index_manifest)
    os.replace(building, output_root)
    log_event(
        "dplm_publish_complete",
        output_root=str(output_root),
        records=int(manifest["records"]),
        elapsed_sec=round(time.time() - started, 3),
    )
    return manifest


def materialize_uniref50(
    *,
    source_path: Path,
    output_root: Path,
    codec_factory: AxisCodecFactory,
    condition_tokens_by_length: Mapping[int, int],
    staging_root: Path | None = None,
    candidate_shards: int = DEFAULT_CANDIDATE_SHARDS,
    encode_workers: int = DEFAULT_ENCODE_WORKERS,
    decompressor: str = "auto",
    read_threads: int = DEFAULT_READ_THREADS,
    mp_start_method: str = "spawn",
    inner_validation_fraction: float = INNER_VALIDATION_FRACTION,
    heldout_test_fraction: float = HELDOUT_TEST_FRACTION,
    split_seed: int = SPLIT_SEED,
    context_budget: int = CONTEXT_BUDGET,
    progress_every: int = 1_000_000,
    encode_progress_every: int = 5_000,
    context_overflow_policy: str = "fail",
    representation_limit_policy: str = "fail",
    resume: bool = False,
    allow_non_worker_index: bool = False,
    index_builder: Callable[..., AxisBIndex] = build_axis_b_index,
) -> dict[str, Any]:
    if int(context_budget) != CONTEXT_BUDGET:
        raise ValueError(f"formal DPLM partial protocol context must be {CONTEXT_BUDGET}")
    if context_overflow_policy not in CONTEXT_OVERFLOW_POLICIES:
        raise ValueError(
            f"context_overflow_policy must be one of {CONTEXT_OVERFLOW_POLICIES}"
        )
    if representation_limit_policy not in REPRESENTATION_LIMIT_POLICIES:
        raise ValueError(
            "representation_limit_policy must be one of "
            f"{REPRESENTATION_LIMIT_POLICIES}"
        )
    token_counts = _validate_condition_counts(condition_tokens_by_length)
    source_path = require_file(source_path, "UniRef50 gzip FASTA")
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite output root: {output_root}")
        manifest = _load_complete_json(output_root / "manifest.json", "published manifest")
        _require_equal(manifest.get("task"), TASK, "published task")
        _require_equal(manifest.get("dataset_id"), DATASET_ID, "published dataset id")
        _require_equal(
            (manifest.get("context_overflow_audit") or {}).get("policy"),
            context_overflow_policy,
            "published context overflow policy",
        )
        _require_equal(
            (manifest.get("representation_limit_audit") or {}).get("policy"),
            representation_limit_policy,
            "published representation limit policy",
        )
        log_event(
            "dplm_materialization_resume_complete",
            output_root=str(output_root),
            records=int(manifest["records"]),
        )
        return manifest
    staging = (
        staging_root.expanduser().resolve()
        if staging_root is not None
        else output_root.with_name(f".{output_root.name}.staging")
    )
    if staging.exists() and not resume:
        raise FileExistsError(
            f"staging root exists; pass --resume to reuse it: {staging}"
        )
    staging.mkdir(parents=True, exist_ok=True)
    scan_manifest = scan_uniref50_candidates(
        source_path=source_path,
        staging_root=staging,
        candidate_shards=int(candidate_shards),
        inner_validation_fraction=float(inner_validation_fraction),
        heldout_test_fraction=float(heldout_test_fraction),
        split_seed=int(split_seed),
        decompressor=decompressor,
        read_threads=int(read_threads),
        progress_every=int(progress_every),
        resume=bool(resume),
    )
    encoded_manifest = encode_candidate_shards(
        scan_manifest=scan_manifest,
        staging_root=staging,
        codec_factory=codec_factory,
        condition_tokens_by_length=token_counts,
        context_budget=int(context_budget),
        encode_workers=int(encode_workers),
        encode_progress_every=int(encode_progress_every),
        mp_start_method=mp_start_method,
        context_overflow_policy=context_overflow_policy,
        representation_limit_policy=representation_limit_policy,
        resume=bool(resume),
    )
    return publish_materialization(
        source_path=source_path,
        output_root=output_root,
        staging_root=staging,
        scan_manifest=scan_manifest,
        encoded_manifest=encoded_manifest,
        condition_tokens_by_length=token_counts,
        context_budget=int(context_budget),
        allow_non_worker_index=allow_non_worker_index,
        index_builder=index_builder,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="Persistent scan/encoded staging root; defaults beside output-root.",
    )
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--candidate-shards", type=int, default=DEFAULT_CANDIDATE_SHARDS)
    parser.add_argument("--encode-workers", type=int, default=DEFAULT_ENCODE_WORKERS)
    parser.add_argument("--read-threads", type=int, default=DEFAULT_READ_THREADS)
    parser.add_argument(
        "--decompressor",
        choices=("auto", "pigz", "python-gzip"),
        default="auto",
    )
    parser.add_argument(
        "--mp-start-method",
        choices=tuple(mp.get_all_start_methods()),
        default="spawn",
    )
    parser.add_argument(
        "--inner-validation-fraction",
        type=float,
        default=INNER_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--heldout-test-fraction",
        type=float,
        default=HELDOUT_TEST_FRACTION,
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--encode-progress-every", type=int, default=5_000)
    parser.add_argument(
        "--context-overflow-policy",
        choices=CONTEXT_OVERFLOW_POLICIES,
        default="fail",
        help=(
            "fail immediately (default), or record every overflow in an audit "
            "JSONL and exclude it without truncation or repair"
        ),
    )
    parser.add_argument(
        "--representation-limit-policy",
        choices=REPRESENTATION_LIMIT_POLICIES,
        default="fail",
        help=(
            "fail immediately (default), or audit and exclude only typed frozen-"
            "codec representation limits (RingIdOverflow and UnknownFeatureError)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the complete scan and each complete encoded shard.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    if FORMAL_TASK_DATASET_SCOPES.get(TASK) != (DATASET_ID,):
        raise RuntimeError("DPLM task registry and materializer dataset id disagree")
    source = require_file(args.source, "UniRef50 gzip FASTA")
    guide = require_file(args.guide, "FDDFS guide")
    vocab = require_file(args.vocab, "FDDFS vocabulary")
    merges = require_file(args.merges, "frozen BPE merges")
    tokenizer_path = require_file(args.tokenizer, "MPNet tokenizer")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    manifest = materialize_uniref50(
        source_path=source,
        output_root=args.output_root,
        staging_root=args.staging_root,
        codec_factory=FrozenAxisCodecFactory(guide, vocab, merges),
        condition_tokens_by_length=condition_token_counts(tokenizer),
        candidate_shards=args.candidate_shards,
        encode_workers=args.encode_workers,
        decompressor=args.decompressor,
        read_threads=args.read_threads,
        mp_start_method=args.mp_start_method,
        inner_validation_fraction=args.inner_validation_fraction,
        heldout_test_fraction=args.heldout_test_fraction,
        split_seed=args.split_seed,
        progress_every=args.progress_every,
        encode_progress_every=args.encode_progress_every,
        context_overflow_policy=args.context_overflow_policy,
        representation_limit_policy=args.representation_limit_policy,
        resume=args.resume,
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
