#!/usr/bin/env python3
"""Materialize current Axis-B FDDFS JSONL records with the fixed pretraining BPE.

This script is deliberately FDDFS-only. It consumes the current MOSES/GuacaMol
string-token records written by ``preprocess_finetune_dataset.py`` and applies
the complete ``mf80000`` merge table used by pretraining. Any legacy ``K<num>``
token is a hard error.

Output JSONL keeps record metadata and text conditions, replaces
``condition.molecule_fddfs`` with ``condition.molecule_bpe_ids``, and replaces
``target_fddfs`` with ``target_bpe_ids``.  IDs are raw BPE IDs (0..17203): the
model-side +3 offset and BOS/EOS are intentionally not materialized here.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any, Iterator, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.active_scope import (  # noqa: E402
    ACTIVE_DATASET_IDS,
    ACTIVE_VALIDATION_HOLDOUT,
)


DEFAULT_DATA_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / '')
)
DEFAULT_INPUT_ROOT = DEFAULT_DATA_ROOT / "finetune_datasets"
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT / "finetune_datasets_bpe_fddfs_mf80000_moses_guacamol_v1"
)
DEFAULT_VOCAB = _moltoken_paths.TOKENIZER_ASSET_ROOT / "vocab.json"
DEFAULT_MERGES = (
    _moltoken_paths.TOKENIZER_ASSET_ROOT / "merges.csv"
)


# Current formal downstream products only. Historical and retired benchmarks
# live under the shared archive and must be re-enabled explicitly before they
# can be materialized again.
DATASET_FILES: dict[str, tuple[str, ...]] = {
    "unconditional_guacamol_v1": ("train.jsonl", "valid.jsonl", "test.jsonl"),
    "unconditional_moses_v1": ("train.jsonl", "test.jsonl", "test_scaffold.jsonl"),
}
if tuple(DATASET_FILES) != ACTIVE_DATASET_IDS:
    raise RuntimeError("active dataset file map and formal downstream scope disagree")


RING_RE = re.compile(r"^R([1-9][0-9]*)$")
FORBIDDEN_STACK_TOKEN_RE = re.compile(r"^K[0-9]+$")


class ForbiddenSerializerError(ValueError):
    """Raised when a legacy stack-token sequence reaches this FDDFS pipeline."""


class LengthStats:
    """Exact streaming length statistics represented as integer histograms."""

    def __init__(self) -> None:
        self.sequences = 0
        self.empty_or_missing = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_hist: Counter[int] = Counter()
        self.output_hist: Counter[int] = Counter()

    def add(self, input_len: int, output_len: int) -> None:
        if input_len <= 0:
            self.empty_or_missing += 1
            return
        if output_len <= 0:
            raise ValueError("non-empty FDDFS sequence encoded to empty BPE sequence")
        if output_len > input_len:
            raise ValueError(
                f"BPE increased sequence length: FDDFS={input_len}, BPE={output_len}"
            )
        self.sequences += 1
        self.input_tokens += int(input_len)
        self.output_tokens += int(output_len)
        self.input_hist[int(input_len)] += 1
        self.output_hist[int(output_len)] += 1

    def merge(self, other: "LengthStats") -> None:
        self.sequences += other.sequences
        self.empty_or_missing += other.empty_or_missing
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.input_hist.update(other.input_hist)
        self.output_hist.update(other.output_hist)

    def to_payload(self, *, include_histograms: bool = True) -> dict[str, Any]:
        payload = {
            "sequences": int(self.sequences),
            "empty_or_missing": int(self.empty_or_missing),
            "fddfs_tokens": int(self.input_tokens),
            "bpe_tokens": int(self.output_tokens),
            "compression_ratio_fddfs_over_bpe": (
                self.input_tokens / self.output_tokens if self.output_tokens else None
            ),
            "fddfs_length": summarize_histogram(self.input_hist),
            "bpe_length": summarize_histogram(self.output_hist),
        }
        if include_histograms:
            payload["fddfs_histogram"] = {
                str(k): int(v) for k, v in sorted(self.input_hist.items())
            }
            payload["bpe_histogram"] = {
                str(k): int(v) for k, v in sorted(self.output_hist.items())
            }
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LengthStats":
        out = cls()
        out.sequences = int(payload.get("sequences", 0))
        out.empty_or_missing = int(payload.get("empty_or_missing", 0))
        out.input_tokens = int(payload.get("fddfs_tokens", 0))
        out.output_tokens = int(payload.get("bpe_tokens", 0))
        out.input_hist = Counter(
            {int(k): int(v) for k, v in payload.get("fddfs_histogram", {}).items()}
        )
        out.output_hist = Counter(
            {int(k): int(v) for k, v in payload.get("bpe_histogram", {}).items()}
        )
        return out


def histogram_value_at(hist: Counter[int], index: int) -> int:
    if index < 0:
        raise IndexError(index)
    seen = 0
    for value, count in sorted(hist.items()):
        next_seen = seen + int(count)
        if index < next_seen:
            return int(value)
        seen = next_seen
    raise IndexError(index)


def histogram_quantile(hist: Counter[int], q: float) -> float | None:
    """Return NumPy-style linear quantile without materializing all lengths."""

    n = sum(hist.values())
    if n == 0:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError(q)
    rank = (n - 1) * q
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    lo_value = histogram_value_at(hist, lo)
    if hi == lo:
        return float(lo_value)
    hi_value = histogram_value_at(hist, hi)
    return float(lo_value + (rank - lo) * (hi_value - lo_value))


def summarize_histogram(hist: Counter[int]) -> dict[str, Any]:
    n = int(sum(hist.values()))
    total = int(sum(int(length) * int(count) for length, count in hist.items()))
    return {
        "count": n,
        "mean": (total / n if n else None),
        "median": histogram_quantile(hist, 0.5),
        "p99": histogram_quantile(hist, 0.99),
        "min": (int(min(hist)) if hist else None),
        "max": (int(max(hist)) if hist else None),
        "quantile_method": "linear_rank_(n-1)q",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_vocab(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    token_to_id = payload.get("token_to_id")
    if not isinstance(token_to_id, dict) or not token_to_id:
        raise ValueError(f"missing token_to_id in {path}")
    return {
        "token_to_id": {str(k): int(v) for k, v in token_to_id.items()},
        "ring_base": int(payload["ring_base"]),
        "ring_min": int(payload.get("ring_min", 1)),
    }


def load_merge_rules(path: Path) -> tuple[list[tuple[int, int, int]], int, int]:
    rules: list[tuple[int, int, int]] = []
    max_id = -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "left", "right", "new_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        expected_step = 0
        for row in reader:
            step = int(row["step"])
            left = int(row["left"])
            right = int(row["right"])
            new_id = int(row["new_id"])
            if step != expected_step:
                raise ValueError(
                    f"{path}: non-contiguous merge step {step}, expected {expected_step}"
                )
            rules.append((left, right, new_id))
            max_id = max(max_id, left, right, new_id)
            expected_step += 1
    if not rules:
        raise ValueError(f"empty merge table: {path}")
    return rules, max_id + 1, min(new_id for _left, _right, new_id in rules)


def validate_merge_dependencies(
    rules: Sequence[tuple[int, int, int]], *, initial_max_token: int
) -> None:
    available = set(range(int(initial_max_token) + 1))
    for index, (left, right, new_id) in enumerate(rules):
        if left not in available or right not in available:
            raise ValueError(
                f"merge {index} depends on unavailable token: ({left}, {right})->{new_id}"
            )
        available.add(int(new_id))


def fddfs_tokens_to_base_ids(
    tokens: Sequence[Any],
    *,
    token_to_id: dict[str, int],
    ring_base: int,
    ring_min: int,
    first_merge_id: int,
    sample_label: str,
) -> list[int]:
    if not isinstance(tokens, (list, tuple)):
        raise TypeError(f"{sample_label}: FDDFS sequence must be a list, got {type(tokens).__name__}")
    out: list[int] = []
    for position, raw_token in enumerate(tokens):
        token = str(raw_token)
        if FORBIDDEN_STACK_TOKEN_RE.fullmatch(token):
            raise ForbiddenSerializerError(
                f"{sample_label}: forbidden legacy token {token!r} at position {position}"
            )
        token_id = token_to_id.get(token)
        if token_id is None:
            match = RING_RE.fullmatch(token)
            if match is None:
                raise KeyError(
                    f"{sample_label}: unknown/non-FDDFS token {token!r} at position {position}"
                )
            ring_number = int(match.group(1))
            token_id = int(ring_base) + ring_number - int(ring_min)
        if token_id < 0 or token_id >= first_merge_id:
            raise ValueError(
                f"{sample_label}: base token {token!r} mapped to {token_id}, "
                f"outside pre-BPE alphabet [0,{first_merge_id - 1}]"
            )
        out.append(int(token_id))
    return out


def unfold_bpe_ids(
    encoded: Sequence[int], merge_expansion: dict[int, tuple[int, int]]
) -> list[int]:
    out: list[int] = []
    stack = [int(x) for x in reversed(encoded)]
    while stack:
        token = stack.pop()
        pair = merge_expansion.get(token)
        if pair is None:
            out.append(token)
        else:
            left, right = pair
            stack.append(int(right))
            stack.append(int(left))
    return out


def discover_inputs(
    input_root: Path, selected_datasets: set[str] | None
) -> list[tuple[str, Path]]:
    unknown = (selected_datasets or set()).difference(DATASET_FILES)
    if unknown:
        raise ValueError(f"unknown dataset selection: {sorted(unknown)}")
    discovered: list[tuple[str, Path]] = []
    for dataset, patterns in DATASET_FILES.items():
        if selected_datasets and dataset not in selected_datasets:
            continue
        dataset_dir = input_root / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(dataset_dir)
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(sorted(dataset_dir.glob(pattern)))
        paths = sorted(dict.fromkeys(paths))
        if not paths:
            raise FileNotFoundError(f"no input JSONL for {dataset}: patterns={patterns}")
        for path in paths:
            if not path.is_file() or path.suffix != ".jsonl":
                raise ValueError(f"invalid input path: {path}")
            discovered.append((dataset, path))
    return discovered


def infer_split(path: Path, dataset: str) -> str:
    name = path.stem.lower()
    for split in ("train", "validation", "valid", "eval", "test_scaffold", "test"):
        if name == split or name.startswith(f"{split}_"):
            return "valid" if split == "validation" else split
    return "unknown"


_ENGINE: Any = None
_VOCAB: dict[str, Any] | None = None
_FIRST_MERGE_ID = -1
_MAX_RAW_BPE_ID = -1
_MERGE_EXPANSION: dict[int, tuple[int, int]] | None = None
_CODEBOOK_INFO: dict[str, Any] | None = None
_VERIFY_SAMPLES_PER_FILE = 0
_BATCH_RECORDS = 0
_BATCH_TOKENS = 0
_LOG_EVERY = 0
_MAX_RECORDS = 0


def init_worker(
    vocab_path: str,
    merges_path: str,
    verify_samples_per_file: int,
    batch_records: int,
    batch_tokens: int,
    log_every: int,
    max_records: int,
) -> None:
    global _ENGINE, _VOCAB, _FIRST_MERGE_ID, _MAX_RAW_BPE_ID, _MERGE_EXPANSION
    global _CODEBOOK_INFO, _VERIFY_SAMPLES_PER_FILE, _BATCH_RECORDS
    global _BATCH_TOKENS, _LOG_EVERY, _MAX_RECORDS

    from src.algorithms.compression.bpe_engine import BPEEngine

    vocab_file = Path(vocab_path)
    merges_file = Path(merges_path)
    vocab = load_vocab(vocab_file)
    rules, bpe_vocab_size, first_merge_id = load_merge_rules(merges_file)
    if int(vocab["ring_base"]) != len(vocab["token_to_id"]):
        raise ValueError(
            "FDDFS vocab is not contiguous before ring IDs: "
            f"ring_base={vocab['ring_base']} token_to_id={len(vocab['token_to_id'])}"
        )
    validate_merge_dependencies(rules, initial_max_token=first_merge_id - 1)
    if bpe_vocab_size > 65536:
        raise ValueError(f"BPE IDs do not fit uint16: vocab_size={bpe_vocab_size}")
    _ENGINE = BPEEngine.from_codebook_dict(
        {"merge_rules": rules, "vocab_size": bpe_vocab_size}, encode_backend="cpp"
    )
    _VOCAB = vocab
    _FIRST_MERGE_ID = int(first_merge_id)
    _MAX_RAW_BPE_ID = int(bpe_vocab_size - 1)
    _MERGE_EXPANSION = {int(new): (int(left), int(right)) for left, right, new in rules}
    _CODEBOOK_INFO = {
        "serializer": "fddfs",
        "merge_rules": len(rules),
        "first_merge_id": first_merge_id,
        "max_raw_bpe_id": bpe_vocab_size - 1,
        "raw_bpe_vocab_size": bpe_vocab_size,
        "model_token_offset": 3,
        "model_pad_token_id": 0,
        "model_bos_token_id": 1,
        "model_eos_token_id": 2,
        "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
    }
    _VERIFY_SAMPLES_PER_FILE = max(0, int(verify_samples_per_file))
    _BATCH_RECORDS = max(1, int(batch_records))
    _BATCH_TOKENS = max(1, int(batch_tokens))
    _LOG_EVERY = max(0, int(log_every))
    _MAX_RECORDS = max(0, int(max_records))


def json_loads(raw: bytes) -> dict[str, Any]:
    try:
        import orjson

        value = orjson.loads(raw)
    except ImportError:
        value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"JSON record must be an object, got {type(value).__name__}")
    return value


def json_dumps(value: dict[str, Any]) -> bytes:
    try:
        import orjson

        return orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE)
    except ImportError:
        return (
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")


def iter_record_batches(handle: Any) -> Iterator[list[tuple[int, bytes, dict[str, Any]]]]:
    batch: list[tuple[int, bytes, dict[str, Any]]] = []
    batch_tokens = 0
    for line_number, raw_line in enumerate(handle, start=1):
        if _MAX_RECORDS and line_number > _MAX_RECORDS:
            break
        if not raw_line.strip():
            raise ValueError(f"blank line at input line {line_number}")
        record = json_loads(raw_line)
        condition = record.get("condition") or {}
        condition_tokens = condition.get("molecule_fddfs") or []
        target_tokens = record.get("target_fddfs") or []
        estimate = len(condition_tokens) + len(target_tokens)
        if batch and (
            len(batch) >= _BATCH_RECORDS or batch_tokens + estimate > _BATCH_TOKENS
        ):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append((line_number, raw_line, record))
        batch_tokens += estimate
    if batch:
        yield batch


def transform_batch(
    batch: list[tuple[int, bytes, dict[str, Any]]],
    *,
    source_label: str,
    verified_so_far: int,
) -> tuple[list[bytes], list[tuple[str, int, int]], int]:
    assert _ENGINE is not None and _VOCAB is not None and _MERGE_EXPANSION is not None
    to_encode: list[list[int]] = []
    locations: list[tuple[int, str, int]] = []

    for record_index, (line_number, _raw_line, record) in enumerate(batch):
        condition = record.get("condition")
        if not isinstance(condition, dict):
            raise TypeError(f"{source_label}:{line_number}: condition must be an object")
        condition_tokens = condition.get("molecule_fddfs") or []
        target_tokens = record.get("target_fddfs") or []
        for field, tokens in (("condition", condition_tokens), ("target", target_tokens)):
            if not tokens:
                continue
            base_ids = fddfs_tokens_to_base_ids(
                tokens,
                token_to_id=_VOCAB["token_to_id"],
                ring_base=int(_VOCAB["ring_base"]),
                ring_min=int(_VOCAB["ring_min"]),
                first_merge_id=_FIRST_MERGE_ID,
                sample_label=f"{source_label}:{line_number}:{field}",
            )
            locations.append((record_index, field, len(base_ids)))
            to_encode.append(base_ids)

    encoded_sequences = _ENGINE.batch_encode(to_encode) if to_encode else []
    if len(encoded_sequences) != len(locations):
        raise RuntimeError("BPE batch result count mismatch")

    encoded_by_record: dict[int, dict[str, tuple[list[int], int]]] = defaultdict(dict)
    for base_ids, encoded, (record_index, field, input_len) in zip(
        to_encode, encoded_sequences, locations
    ):
        encoded_ids = [int(x) for x in encoded]
        if encoded_ids and (
            min(encoded_ids) < 0 or max(encoded_ids) > _MAX_RAW_BPE_ID
        ):
            raise ValueError(
                f"encoded IDs outside fixed codebook [0,{_MAX_RAW_BPE_ID}] "
                f"for {source_label}"
            )
        if len(encoded_ids) > input_len:
            raise ValueError(
                f"BPE length increased for {source_label}: {input_len}->{len(encoded_ids)}"
            )
        if verified_so_far < _VERIFY_SAMPLES_PER_FILE:
            unfolded = unfold_bpe_ids(encoded_ids, _MERGE_EXPANSION)
            if unfolded != base_ids:
                raise AssertionError(
                    f"BPE roundtrip mismatch in {source_label}: "
                    f"base_len={len(base_ids)} bpe_len={len(encoded_ids)}"
                )
            verified_so_far += 1
        encoded_by_record[record_index][field] = (encoded_ids, input_len)

    output_lines: list[bytes] = []
    lengths: list[tuple[str, int, int]] = []
    for record_index, (_line_number, _raw_line, record) in enumerate(batch):
        output = dict(record)
        condition = dict(output.get("condition") or {})
        condition.pop("molecule_fddfs", None)
        condition_encoded = encoded_by_record.get(record_index, {}).get("condition")
        if condition_encoded is None:
            condition["molecule_bpe_ids"] = []
            lengths.append(("condition", 0, 0))
        else:
            ids, input_len = condition_encoded
            condition["molecule_bpe_ids"] = ids
            lengths.append(("condition", input_len, len(ids)))
        output["condition"] = condition
        output.pop("target_fddfs", None)
        target_encoded = encoded_by_record.get(record_index, {}).get("target")
        if target_encoded is None:
            output["target_bpe_ids"] = []
            lengths.append(("target", 0, 0))
        else:
            ids, input_len = target_encoded
            output["target_bpe_ids"] = ids
            lengths.append(("target", input_len, len(ids)))
        output_lines.append(json_dumps(output))
    return output_lines, lengths, verified_so_far


def summary_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".bpe_summary.json")


def resume_summary_matches(
    existing: dict[str, Any],
    *,
    source_stat: os.stat_result,
    output: Path,
    codebook_sha256: str,
    vocab_sha256: str,
    max_records: int,
) -> bool:
    if not (
        output.is_file()
        and int(existing.get("source_size", -1)) == source_stat.st_size
        and int(existing.get("source_mtime_ns", -1)) == source_stat.st_mtime_ns
        and existing.get("codebook_sha256") == codebook_sha256
        and existing.get("vocab_sha256") == vocab_sha256
        and int(existing.get("partial_max_records", -1)) == int(max_records)
        and bool(existing.get("complete"))
    ):
        return False
    if int(existing.get("output_size", -1)) != output.stat().st_size:
        return False
    expected_digest = existing.get("output_sha256")
    return isinstance(expected_digest, str) and sha256_file(output) == expected_digest


def process_one_file(task: dict[str, Any]) -> dict[str, Any]:
    assert _CODEBOOK_INFO is not None
    dataset = str(task["dataset"])
    source = Path(task["source"])
    output = Path(task["output"])
    relative = str(task["relative"])
    split = infer_split(source, dataset)
    summary_path = summary_path_for(output)
    source_stat = source.stat()

    if bool(task.get("resume")) and output.is_file() and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if resume_summary_matches(
            existing,
            source_stat=source_stat,
            output=output,
            codebook_sha256=str(task["codebook_sha256"]),
            vocab_sha256=str(task["vocab_sha256"]),
            max_records=_MAX_RECORDS,
        ):
            return {"status": "resumed", **existing}

    if output.exists() or summary_path.exists():
        raise FileExistsError(f"output exists without a valid --resume manifest: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    started = time.time()
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    stats = {"condition": LengthStats(), "target": LengthStats()}
    task_stats: dict[str, dict[str, LengthStats]] = defaultdict(
        lambda: {"condition": LengthStats(), "target": LengthStats()}
    )
    records = 0
    verified = 0
    input_bytes = 0

    try:
        with source.open("rb") as input_handle, tmp_output.open("wb") as output_handle:
            for batch in iter_record_batches(input_handle):
                output_lines, lengths, verified = transform_batch(
                    batch, source_label=relative, verified_so_far=verified
                )
                for (_line_number, raw_line, record), output_line in zip(batch, output_lines):
                    source_digest.update(raw_line)
                    input_bytes += len(raw_line)
                    output_digest.update(output_line)
                    output_handle.write(output_line)
                    records += 1
                for batch_index, (_line_number, _raw_line, record) in enumerate(batch):
                    task_name = str(record.get("task") or "unknown")
                    condition_row = lengths[batch_index * 2]
                    target_row = lengths[batch_index * 2 + 1]
                    for field, input_len, output_len in (condition_row, target_row):
                        stats[field].add(input_len, output_len)
                        task_stats[task_name][field].add(input_len, output_len)
                if _LOG_EVERY and records % _LOG_EVERY < len(batch):
                    elapsed = max(1e-9, time.time() - started)
                    print(
                        f"progress file={relative} records={records} "
                        f"input_gib={input_bytes / (1024**3):.3f} "
                        f"records_per_sec={records / elapsed:.1f} elapsed_sec={elapsed:.1f}",
                        flush=True,
                    )
        os.replace(tmp_output, output)
    except BaseException:
        tmp_output.unlink(missing_ok=True)
        raise

    payload = {
        "complete": True,
        "status": "written",
        "host": socket.gethostname(),
        "dataset": dataset,
        "split": split,
        "relative_source": relative,
        "source": str(source),
        "output": str(output),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "source_sha256": source_digest.hexdigest(),
        "output_size": int(output.stat().st_size),
        "output_sha256": output_digest.hexdigest(),
        "codebook_sha256": str(task["codebook_sha256"]),
        "vocab_sha256": str(task["vocab_sha256"]),
        "records": int(records),
        "verified_bpe_roundtrips": int(verified),
        "partial_max_records": int(_MAX_RECORDS),
        "encoding": dict(_CODEBOOK_INFO),
        "lengths": {
            field: item.to_payload(include_histograms=True) for field, item in stats.items()
        },
        "by_task": {
            name: {
                field: item.to_payload(include_histograms=True)
                for field, item in fields.items()
            }
            for name, fields in sorted(task_stats.items())
        },
        "elapsed_sec": time.time() - started,
    }
    atomic_write_json(summary_path, payload)
    return payload


def empty_field_pair() -> dict[str, LengthStats]:
    return {"condition": LengthStats(), "target": LengthStats()}


def merge_summary_into(
    destination: dict[str, LengthStats], payload: dict[str, Any]
) -> None:
    for field in ("condition", "target"):
        destination[field].merge(LengthStats.from_payload(payload["lengths"][field]))


def compact_stats(fields: dict[str, LengthStats]) -> dict[str, Any]:
    return {
        name: value.to_payload(include_histograms=False) for name, value in fields.items()
    }


def histogram_stats(fields: dict[str, LengthStats]) -> dict[str, Any]:
    return {
        name: value.to_payload(include_histograms=True) for name, value in fields.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges-csv", type=Path, default=DEFAULT_MERGES)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Restrict to a current dataset directory; repeat as needed.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-records", type=int, default=2048)
    parser.add_argument("--batch-tokens", type=int, default=2_000_000)
    parser.add_argument("--verify-samples-per-file", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-jump-host",
        action="store_true",
        help="Only for tiny synthetic tests; full materialization must run on a worker.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hostname = socket.gethostname()
    if not __import__("pipeline.runtime_host", fromlist=["is_compute_host"]).is_compute_host(hostname.lower()) and not args.allow_jump_host:
        raise RuntimeError(
            f"refusing to materialize on non-worker host {hostname!r}; run on a cluster worker"
        )
    if args.workers < 1:
        raise ValueError("--workers must be >=1")
    if not args.vocab.is_file() or not args.merges_csv.is_file():
        raise FileNotFoundError(f"missing vocab/codebook: {args.vocab}, {args.merges_csv}")

    selected = set(args.dataset) or None
    inputs = discover_inputs(args.input_root, selected)
    codebook_sha = sha256_file(args.merges_csv)
    vocab_sha = sha256_file(args.vocab)

    if args.output_root.exists() and not args.resume:
        if any(args.output_root.iterdir()):
            raise FileExistsError(
                f"output root is non-empty; pass --resume: {args.output_root}"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dataset, source in inputs:
        relative = source.relative_to(args.input_root)
        tasks.append(
            {
                "dataset": dataset,
                "source": str(source),
                "relative": str(relative),
                "output": str(args.output_root / relative),
                "resume": bool(args.resume),
                "codebook_sha256": codebook_sha,
                "vocab_sha256": vocab_sha,
            }
        )

    print(
        f"fddfs_bpe_materialize_start host={hostname} files={len(tasks)} "
        f"workers={args.workers} output={args.output_root}",
        flush=True,
    )
    print(f"codebook_sha256={codebook_sha} vocab_sha256={vocab_sha}", flush=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    init_args = (
        str(args.vocab),
        str(args.merges_csv),
        int(args.verify_samples_per_file),
        int(args.batch_records),
        int(args.batch_tokens),
        int(args.log_every),
        int(args.max_records),
    )
    if args.workers == 1:
        init_worker(*init_args)
        for task in tasks:
            result = process_one_file(task)
            results.append(result)
            print(
                f"file_done source={result['relative_source']} records={result['records']} "
                f"elapsed_sec={result['elapsed_sec']:.1f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker, initargs=init_args
        ) as pool:
            futures = {pool.submit(process_one_file, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"file_done source={result['relative_source']} records={result['records']} "
                    f"elapsed_sec={result['elapsed_sec']:.1f}",
                    flush=True,
                )

    results.sort(key=lambda item: str(item["relative_source"]))
    global_stats = empty_field_pair()
    by_dataset: dict[str, dict[str, LengthStats]] = defaultdict(empty_field_pair)
    by_task: dict[str, dict[str, LengthStats]] = defaultdict(empty_field_pair)
    by_file: dict[str, dict[str, Any]] = {}
    total_records = 0
    for result in results:
        merge_summary_into(global_stats, result)
        merge_summary_into(by_dataset[str(result["dataset"])], result)
        total_records += int(result["records"])
        for task_name, task_payload in result.get("by_task", {}).items():
            for field in ("condition", "target"):
                by_task[task_name][field].merge(
                    LengthStats.from_payload(task_payload[field])
                )
        by_file[str(result["relative_source"])] = compact_stats(
            {
                field: LengthStats.from_payload(result["lengths"][field])
                for field in ("condition", "target")
            }
        )

    protocol = {
        "serializer": "fddfs",
        "legacy_stack_tokens_allowed": False,
        "codebook": str(args.merges_csv),
        "codebook_sha256": codebook_sha,
        "vocab": str(args.vocab),
        "vocab_sha256": vocab_sha,
        "merge_scope": "all_rules_from_step_0; input is uncompressed FDDFS",
        "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
        "model_token_offset": 3,
        "lengths_exclude_bos_eos": True,
        "source_policy": "current Axis-B products only; historical/smoke outputs excluded",
        "validation_holdout": ACTIVE_VALIDATION_HOLDOUT,
    }
    stats_payload = {
        "protocol": protocol,
        "records": int(total_records),
        "files": len(results),
        "global": compact_stats(global_stats),
        "by_dataset": {
            name: compact_stats(fields) for name, fields in sorted(by_dataset.items())
        },
        "by_task": {name: compact_stats(fields) for name, fields in sorted(by_task.items())},
        "by_file": by_file,
        "elapsed_sec": time.time() - started,
    }
    histogram_payload = {
        "protocol": protocol,
        "global": histogram_stats(global_stats),
        "by_dataset": {
            name: histogram_stats(fields) for name, fields in sorted(by_dataset.items())
        },
        "by_task": {
            name: histogram_stats(fields) for name, fields in sorted(by_task.items())
        },
    }
    manifest = {
        "complete": True,
        "created_at_unix": time.time(),
        "host": hostname,
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "protocol": protocol,
        "records": int(total_records),
        "files": len(results),
        "partial_max_records_per_file": int(args.max_records),
        "file_summaries": [
            {
                key: result[key]
                for key in (
                    "dataset",
                    "split",
                    "relative_source",
                    "source",
                    "output",
                    "source_size",
                    "source_sha256",
                    "output_size",
                    "output_sha256",
                    "records",
                    "verified_bpe_roundtrips",
                    "elapsed_sec",
                )
            }
            for result in results
        ],
    }
    atomic_write_json(args.output_root / "length_stats.json", stats_payload)
    atomic_write_json(args.output_root / "length_histograms.json", histogram_payload)
    atomic_write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(stats_payload["global"], ensure_ascii=False, indent=2), flush=True)
    print(
        f"FDDFS_BPE_MATERIALIZE_OK files={len(results)} records={total_records} "
        f"elapsed_sec={time.time() - started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
