"""Axis-B SFT data, indexing, collation, and task-balanced sampling.

The materialized Axis-B files are newline-delimited JSON produced by
``compress_finetune_fddfs_bpe.py``.  They are intentionally kept as JSONL:
this module builds a small random-access sidecar on a worker and never loads a
whole corpus into memory.  The sidecar stores ``(file_id, byte_offset,
task_id)`` rows as atomic ``numpy.uint64`` arrays.

There are two token-id spaces in this module. BPE ids on disk are raw ids
``0..17203``. Qwen target ids are obtained by adding ``MODEL_TOKEN_OFFSET`` (3)
at collation time, after which target type/BOS/EOS are added. The molBERT
condition protocol is body+EOS without BOS. Keeping conversion here and
nowhere in the index prevents accidentally double-offsetting a dataset.

The code deliberately has no dependency on a model implementation.  PyTorch
and NumPy are imported lazily, so schema/collator tests can run in a small
Python environment and data indexing can be performed explicitly on a worker.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import random
import socket
import struct
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from pipeline.runtime_host import is_compute_host


# Raw FDDFS-BPE ids are fixed by the materialization protocol.
RAW_BPE_MIN_ID = 0
RAW_BPE_MAX_ID = 17_203
MODEL_TOKEN_OFFSET = 3
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
# FDDFS reserves raw ids 0/1/2 for ``(``, ``)`` and ``<SEP>``.  After the
# model-side offset, the component separator is Qwen id 5; it must not be
# confused with Qwen's EOS id 2.
FDDFS_COMPONENT_SEP_RAW_ID = 2
FDDFS_COMPONENT_SEP_MODEL_ID = FDDFS_COMPONENT_SEP_RAW_ID + MODEL_TOKEN_OFFSET
# Keep the historical 384-token default for existing tasks, but allow a caller
# to use MPNet's full 512-token encoder window.  Long text tasks can then pool
# the complete encoder sequence into a smaller Qwen prefix via
# ``condition_token_budget`` instead of truncating the source text merely to
# reserve decoder space.
TEXT_MAX_LENGTH = 384
TEXT_ENCODER_MAX_LENGTH = 512
MOLECULE_MAX_LENGTH = 512
# The flagship was pretrained/evaluated with fixed 512-token sequences.  A
# checkpoint may expose a larger rotary-position limit, but Axis-B's data
# protocol must still reserve room for the target inside this window.
DEFAULT_QWEN_CONTEXT = 512

TARGET_TYPE_TOKEN_IDS: dict[str, int] = {
    "small_molecule": 17_208,
    "peptide": 17_209,
    "protein": 17_210,
    "dna": 17_211,
    "rna": 17_212,
}

CONDITION_TYPES = frozenset(("none", "text", "molecule", "both"))
# These are source modalities, not task ids.  The set is intentionally small
# and extensible; task descriptions remain ordinary text rather than Qwen
# vocabulary entries.
CONDITION_MODALITIES = frozenset(("none", "text", "small_molecule", "peptide", "protein", "dna", "rna"))
DEFAULT_CONDITION_TOKEN_BUDGET = MOLECULE_MAX_LENGTH
MOLECULE_CONDITION_MODES = frozenset(("direct_qwen", "molbert"))
INDEX_VERSION = 1
_ROW_STRUCT = struct.Struct("<QQQ")


class AxisBDataError(ValueError):
    """Base class for malformed Axis-B data and sidecars."""


class ManifestError(AxisBDataError):
    """The formal BPE manifest is missing or inconsistent."""


class SchemaError(AxisBDataError):
    """A JSONL record does not satisfy the SFT schema."""


def _numpy() -> Any:
    """Import NumPy only when an index operation actually needs it."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Axis-B indexing/sampling requires NumPy; run it in the allocated worker environment"
        ) from exc
    return np


def _canonical_split(value: str) -> str:
    value = str(value).strip().lower()
    if value == "train":
        return "train"
    if value in {"eval", "valid", "validation", "dev"}:
        return "eval"
    if value == "test" or value.startswith("test_"):
        return "test"
    raise ManifestError(f"unsupported manifest split {value!r}; expected train/eval/test")


def _is_tomg(dataset: str) -> bool:
    return "tomg" in str(dataset).lower()


def _validation_holdout_specs(manifest: Mapping[str, Any]) -> dict[str, tuple[float, int]]:
    """Return deterministic train-derived validation rules from the manifest."""

    protocol = manifest.get("protocol") or {}
    raw = protocol.get("validation_holdout")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ManifestError("protocol.validation_holdout must be an object")
    if raw.get("method") != "sha256_record_id_fraction_v1":
        raise ManifestError(
            "protocol.validation_holdout.method must be "
            "'sha256_record_id_fraction_v1'"
        )
    tasks = raw.get("tasks")
    if not isinstance(tasks, Mapping) or not tasks:
        raise ManifestError("protocol.validation_holdout.tasks must be a non-empty object")
    parsed: dict[str, tuple[float, int]] = {}
    for raw_task, raw_spec in tasks.items():
        task = str(raw_task).strip()
        if not task or not isinstance(raw_spec, Mapping):
            raise ManifestError("every validation holdout task needs an object spec")
        fraction = float(raw_spec.get("fraction", 0.0))
        seed = int(raw_spec.get("seed"))
        if not 0.0 < fraction < 1.0:
            raise ManifestError(
                f"validation holdout fraction for {task!r} must be in (0, 1)"
            )
        parsed[task] = (fraction, seed)
    return parsed


def _record_split(
    *,
    source_split: str,
    task: str,
    record: Mapping[str, Any],
    holdouts: Mapping[str, tuple[float, int]],
    source_label: str,
) -> str:
    """Assign a train record to train/eval without consulting benchmark test data."""

    spec = holdouts.get(task)
    if source_split != "train" or spec is None:
        return source_split
    sample_id = record.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise SchemaError(
            f"validation holdout requires a non-empty record id: {source_label}"
        )
    fraction, seed = spec
    digest = hashlib.sha256(
        f"{seed}\0{task}\0{sample_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    threshold = int(fraction * (1 << 64))
    return "eval" if value < threshold else "train"


def _index_semantics(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Select execution-relevant protocol fields for index reuse."""

    return {
        key: protocol.get(key)
        for key in (
            "serializer",
            "ids_on_disk",
            "model_token_offset",
            "validation_holdout",
        )
    }


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _default_index_dir(root: Path) -> Path:
    """Choose a schema-specific sidecar name without breaking v1 roots."""

    if (root / "condition_schema.json").is_file():
        return root / "axis_b_index_v2"
    return root / "axis_b_index_v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write metadata through a same-directory temporary file and rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class ManifestFile:
    """One source JSONL named by the formal materialization manifest."""

    file_id: int
    dataset: str
    manifest_split: str
    split: str
    relative_source: str
    path: Path
    records: int | None
    output_size: int | None
    output_sha256: str | None


def load_formal_manifest(data_root: Path | str) -> tuple[dict[str, Any], list[ManifestFile], str]:
    """Load and validate the formal BPE JSONL manifest.

    Paths are resolved from ``relative_source`` rather than trusting absolute
    paths embedded in the manifest.  This allows a worker mount to differ from
    the host that materialized the files while retaining the same manifest.
    """

    root = Path(data_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"formal BPE manifest not found: {manifest_path}")
    raw_manifest = manifest_path.read_bytes()
    try:
        payload = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("complete"):
        raise ManifestError(f"{manifest_path} is not a complete materialization manifest")

    protocol = payload.get("protocol") or {}
    if protocol:
        ids_on_disk = protocol.get("ids_on_disk")
        if ids_on_disk and ids_on_disk != "raw_bpe_before_model_offset_without_bos_eos":
            raise ManifestError(f"unexpected on-disk id protocol: {ids_on_disk!r}")
        offset = protocol.get("model_token_offset")
        if offset is not None and int(offset) != MODEL_TOKEN_OFFSET:
            raise ManifestError(f"expected model token offset 3, got {offset!r}")
        if protocol.get("serializer") not in (None, "fddfs"):
            raise ManifestError(f"Axis-B requires FDDFS data, got {protocol.get('serializer')!r}")

    summaries = payload.get("file_summaries")
    if not isinstance(summaries, list) or not summaries:
        raise ManifestError("manifest.file_summaries is empty or not a list")

    files: list[ManifestFile] = []
    seen: set[str] = set()
    # Sorting makes file ids stable even if a producer changes JSON object order.
    ordered = sorted(summaries, key=lambda item: str(item.get("relative_source", "")))
    for file_id, item in enumerate(ordered):
        if not isinstance(item, dict):
            raise ManifestError(f"file_summaries[{file_id}] is not an object")
        relative = str(item.get("relative_source") or "")
        if not relative or relative in seen:
            raise ManifestError(f"duplicate/missing relative_source: {relative!r}")
        seen.add(relative)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"source escapes data root: {relative!r}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"manifest source missing: {candidate}")
        manifest_split = str(item.get("split") or "")
        split = _canonical_split(manifest_split)
        dataset = str(item.get("dataset") or "")
        if not dataset:
            raise ManifestError(f"missing dataset for {relative}")
        # TOMG is a prompt-only test benchmark.  Refuse a tampered manifest
        # that labels it train rather than silently leaking prompts into SFT.
        if _is_tomg(dataset) and split != "test":
            raise ManifestError(f"TOMG source must be test-only: {relative}")
        stat = candidate.stat()
        expected_size = item.get("output_size")
        expected_size_i = int(expected_size) if expected_size is not None else None
        if expected_size_i is not None and expected_size_i >= 0 and stat.st_size != expected_size_i:
            raise ManifestError(
                f"size mismatch for {relative}: manifest={expected_size_i}, actual={stat.st_size}"
            )
        expected_records = item.get("records")
        records_i = int(expected_records) if expected_records is not None else None
        files.append(
            ManifestFile(
                file_id=file_id,
                dataset=dataset,
                manifest_split=manifest_split,
                split=split,
                relative_source=relative,
                path=candidate,
                records=records_i,
                output_size=expected_size_i,
                output_sha256=(str(item["output_sha256"]) if item.get("output_sha256") else None),
            )
        )
    return payload, files, _sha256_bytes(raw_manifest)


def _worker_guard(*, allow_non_worker: bool) -> None:
    if allow_non_worker:
        return
    hostname = socket.gethostname().lower()
    if not is_compute_host(hostname):
        raise RuntimeError(
            f"refusing Axis-B index construction on non-worker host {hostname!r}; "
            "use an allocated worker, or explicitly enable the approved physical "
            "host (tests may pass allow_non_worker=True)"
        )


@dataclass
class AxisBIndex:
    """Loaded index metadata and random-access row arrays."""

    data_root: Path
    index_dir: Path
    metadata: dict[str, Any]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.metadata.get("splits", {}))

    @property
    def task_names(self) -> dict[int, str]:
        tasks = self.metadata.get("tasks") or {}
        if not isinstance(tasks, dict):
            raise AxisBDataError("invalid task metadata in Axis-B index")
        return {int(k): str(v) for k, v in tasks.items()}

    @property
    def files(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.metadata.get("files") or ())

    def rows(self, split: str = "train") -> Any:
        canonical = _canonical_split(split)
        item = (self.metadata.get("splits") or {}).get(canonical)
        if not isinstance(item, dict):
            raise KeyError(f"index has no split {canonical!r}")
        path = self.index_dir / str(item["rows_file"])
        np = _numpy()
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        if rows.dtype != np.dtype(np.uint64) or rows.ndim != 2 or rows.shape[1] != 3:
            raise AxisBDataError(f"invalid uint64 row index: {path}")
        if int(rows.shape[0]) != int(item.get("records", -1)):
            raise AxisBDataError(f"row count mismatch in {path}")
        return rows

    def task_counts(self, split: str = "train") -> dict[str, int]:
        canonical = _canonical_split(split)
        item = (self.metadata.get("splits") or {}).get(canonical) or {}
        return {str(k): int(v) for k, v in (item.get("task_counts") or {}).items()}

    @classmethod
    def load(cls, data_root: Path | str, index_dir: Path | str | None = None) -> "AxisBIndex":
        root = Path(data_root).expanduser().resolve()
        directory = Path(index_dir) if index_dir is not None else _default_index_dir(root)
        metadata_path = directory / "index_manifest.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Axis-B index missing: {metadata_path}; build it on a worker with build_axis_b_index()"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AxisBDataError(f"invalid index metadata: {metadata_path}") from exc
        if not isinstance(metadata, dict) or not metadata.get("complete"):
            raise AxisBDataError(f"incomplete Axis-B index: {metadata_path}")
        if int(metadata.get("index_version", -1)) != INDEX_VERSION:
            raise AxisBDataError(f"unsupported Axis-B index version: {metadata.get('index_version')}")
        split_metadata = metadata.get("splits")
        if not isinstance(split_metadata, dict):
            raise AxisBDataError(f"invalid split metadata: {metadata_path}")
        for split, item in split_metadata.items():
            if not isinstance(item, dict):
                raise AxisBDataError(f"invalid split entry {split!r}: {metadata_path}")
            path = directory / str(item.get("rows_file", ""))
            if not path.is_file():
                raise AxisBDataError(f"index rows missing for {split}: {path}")
        return cls(root, directory, metadata)


def _index_is_reusable(
    index: AxisBIndex,
    *,
    files: Sequence[ManifestFile],
    protocol: Mapping[str, Any],
) -> bool:
    metadata = index.metadata
    if _index_semantics(metadata.get("protocol") or {}) != _index_semantics(protocol):
        return False
    indexed_files = metadata.get("files") or []
    if len(indexed_files) != len(files):
        return False
    for current, saved in zip(files, indexed_files):
        if (
            str(saved.get("relative_source")) != current.relative_source
            or int(saved.get("size", -1)) != current.path.stat().st_size
            or str(saved.get("split")) != current.split
        ):
            return False
    return True


def _write_raw_rows_as_npy(raw_path: Path, output_path: Path, count: int) -> None:
    """Convert a compact raw uint64 spool to an atomic ``.npy`` array."""

    np = _numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.tmp.", suffix=".npy", dir=str(output_path.parent)
    )
    os.close(fd)
    try:
        mapped = np.lib.format.open_memmap(
            temp_name, mode="w+", dtype=np.uint64, shape=(int(count), 3), fortran_order=False
        )
        with raw_path.open("rb") as source:
            position = 0
            while position < count:
                want = min(1_048_576, count - position)
                raw = source.read(want * _ROW_STRUCT.size)
                if len(raw) != want * _ROW_STRUCT.size:
                    raise AxisBDataError(f"truncated row spool {raw_path}")
                chunk = np.frombuffer(raw, dtype="<u8").reshape((-1, 3))
                mapped[position : position + want] = chunk
                position += want
        mapped.flush()
        del mapped
        with open(temp_name, "rb") as check:
            os.fsync(check.fileno())
        os.replace(temp_name, output_path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def build_axis_b_index(
    data_root: Path | str,
    *,
    index_dir: Path | str | None = None,
    allow_non_worker: bool = False,
    allow_jump_host: bool = False,
    force: bool = False,
) -> AxisBIndex:
    """Build or reuse the worker-side uint64 random-access index.

    This is the only operation that scans JSONL.  It streams one line at a
    time, writes compact temporary row spools, and atomically publishes each
    NumPy array followed by ``index_manifest.json``.  Existing compatible
    indexes are reused without touching source data.
    """

    _worker_guard(allow_non_worker=bool(allow_non_worker or allow_jump_host))
    root = Path(data_root).expanduser().resolve()
    directory = Path(index_dir) if index_dir is not None else _default_index_dir(root)
    directory = directory.expanduser().resolve()
    manifest, manifest_files, manifest_digest = load_formal_manifest(root)
    holdouts = _validation_holdout_specs(manifest)
    if directory.exists() and (directory / "index_manifest.json").is_file() and not force:
        existing = AxisBIndex.load(root, directory)
        if _index_is_reusable(
            existing,
            files=manifest_files,
            protocol=manifest.get("protocol") or {},
        ):
            return existing
        raise AxisBDataError(
            f"existing Axis-B index is stale: {directory}; use a new index_dir or force=True"
        )
    directory.mkdir(parents=True, exist_ok=True)

    _numpy()
    spool_paths: dict[str, Path] = {}
    spool_handles: dict[str, Any] = {}
    counts: dict[str, int] = defaultdict(int)
    task_names: dict[str, int] = {}
    file_rows: dict[int, int] = defaultdict(int)
    file_split_rows: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    split_task_counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "eval": defaultdict(int),
        "test": defaultdict(int),
    }
    try:
        for split in ("train", "eval", "test"):
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{split}.rows.raw.", dir=str(directory)
            )
            path = Path(temp_name)
            spool_paths[split] = path
            spool_handles[split] = os.fdopen(fd, "wb")

        observed_by_file: dict[int, int] = {}
        for source in manifest_files:
            line_count = 0
            handle = source.path.open("rb")
            try:
                while True:
                    offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    line_count += 1
                    if not raw_line.strip():
                        raise SchemaError(f"blank line in {source.relative_source}:{line_count}")
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise SchemaError(
                            f"invalid JSON in {source.relative_source}:{line_count}: {exc}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise SchemaError(f"record is not an object: {source.relative_source}:{line_count}")
                    target = record.get("target_bpe_ids")
                    if target is None:
                        # A missing target is equivalent to an empty target for
                        # prompt-only records, but a non-list value is malformed.
                        target = []
                    if not isinstance(target, list):
                        raise SchemaError(
                            f"target_bpe_ids must be a list: {source.relative_source}:{line_count}"
                        )
                    if not target:
                        continue
                    task = record.get("task")
                    if not isinstance(task, str) or not task:
                        raise SchemaError(f"missing task: {source.relative_source}:{line_count}")
                    if task not in task_names:
                        task_names[task] = len(task_names)
                    task_id = task_names[task]
                    row_split = _record_split(
                        source_split=source.split,
                        task=task,
                        record=record,
                        holdouts=holdouts,
                        source_label=f"{source.relative_source}:{line_count}",
                    )
                    split_handle = spool_handles[row_split]
                    split_handle.write(_ROW_STRUCT.pack(source.file_id, offset, task_id))
                    counts[row_split] += 1
                    file_rows[source.file_id] += 1
                    file_split_rows[source.file_id][row_split] += 1
                    split_task_counts[row_split][task] += 1
            finally:
                handle.close()
            observed_by_file[source.file_id] = line_count
            if source.records is not None and line_count != source.records:
                raise ManifestError(
                    f"record count mismatch for {source.relative_source}: "
                    f"manifest={source.records}, actual={line_count}"
                )
        manifest_total = manifest.get("records")
        if manifest_total is not None and int(manifest_total) != sum(observed_by_file.values()):
            raise ManifestError(
                f"manifest total record mismatch: manifest={manifest_total}, "
                f"files={sum(observed_by_file.values())}"
            )
        for task in holdouts:
            train_records = int(split_task_counts["train"].get(task, 0))
            eval_records = int(split_task_counts["eval"].get(task, 0))
            if train_records <= 0 or eval_records <= 0:
                raise ManifestError(
                    f"validation holdout for {task!r} produced "
                    f"train={train_records} eval={eval_records}"
                )
        for handle in spool_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

        split_meta: dict[str, Any] = {}
        for split in ("train", "eval", "test"):
            rows_name = f"{split}.rows.npy"
            _write_raw_rows_as_npy(spool_paths[split], directory / rows_name, counts[split])
            task_counts = {
                str(task): int(n)
                for task, n in split_task_counts[split].items()
                if int(n) > 0
            }
            split_meta[split] = {
                "rows_file": rows_name,
                "records": int(counts[split]),
                "task_counts": task_counts,
            }

        metadata = {
            "complete": True,
            "index_version": INDEX_VERSION,
            "data_root": str(root),
            "manifest_sha256": manifest_digest,
            "protocol": manifest.get("protocol") or {},
            "tasks": {str(task_id): task for task, task_id in sorted(task_names.items(), key=lambda item: item[1])},
            "splits": split_meta,
            "files": [
                {
                    "file_id": source.file_id,
                    "dataset": source.dataset,
                    "manifest_split": source.manifest_split,
                    "split": source.split,
                    "relative_source": source.relative_source,
                    "size": source.path.stat().st_size,
                    "mtime_ns": source.path.stat().st_mtime_ns,
                    "records": observed_by_file[source.file_id],
                    "indexed_records": file_rows.get(source.file_id, 0),
                    "indexed_split_records": dict(
                        sorted(file_split_rows.get(source.file_id, {}).items())
                    ),
                }
                for source in manifest_files
            ],
        }
        _atomic_json(directory / "index_manifest.json", metadata)
        return AxisBIndex.load(root, directory)
    finally:
        for handle in spool_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        for path in spool_paths.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass


# A plural alias reads naturally in launch scripts and preserves compatibility
# with early callers that named the operation after its three split arrays.
build_axis_b_indexes = build_axis_b_index


class AxisBJsonlDataset:
    """Random-access view over one canonical manifest split.

    ``task`` creates a backwards-compatible one-task view.  ``tasks`` creates
    an explicit multi-task view without rewriting source JSONL or rebuilding
    the shared index.  The row sidecar remains memory-mapped; a filtered view
    stores only matching compact-index row positions.
    """

    def __init__(
        self,
        data_root: Path | str | AxisBIndex | None = None,
        *,
        split: str = "train",
        index_dir: Path | str | None = None,
        index: AxisBIndex | None = None,
        task: str | None = None,
        tasks: Sequence[str] | None = None,
    ) -> None:
        if isinstance(data_root, AxisBIndex):
            if index is not None and index is not data_root:
                raise ValueError("conflicting data_root/index values")
            index = data_root
            data_root = index.data_root
        if data_root is None:
            if index is None:
                raise ValueError("data_root or index must be provided")
            data_root = index.data_root
        self.data_root = Path(data_root).expanduser().resolve()
        self.index = index or AxisBIndex.load(self.data_root, index_dir)
        self.split = _canonical_split(split)
        self._rows = self.index.rows(self.split)
        self._files_by_id = {
            int(item["file_id"]): self.data_root / str(item["relative_source"])
            for item in self.index.files
        }
        self._file_meta_by_id = {int(item["file_id"]): item for item in self.index.files}
        self._handles: dict[int, Any] = {}
        self._task_names = self.index.task_names
        self._task_indices_cache: dict[str, list[int]] | None = None
        if task is not None and tasks is not None:
            raise ValueError("pass either task or tasks, not both")
        if tasks is None:
            selected_tasks = (str(task).strip(),) if task is not None else ()
        else:
            selected_tasks = tuple(dict.fromkeys(str(value).strip() for value in tasks))
        if any(not value for value in selected_tasks):
            raise ValueError("task filters must be non-empty strings")
        self.tasks = selected_tasks
        self.task = selected_tasks[0] if len(selected_tasks) == 1 else None
        self._row_positions: Any | None = None
        if self.tasks:
            unknown = set(self.tasks).difference(self._task_names.values())
            if unknown:
                raise AxisBDataError(f"unknown Axis-B task(s): {sorted(unknown)}")
            np = _numpy()
            selected = set(self.tasks)
            task_ids = [
                task_id for task_id, name in self._task_names.items() if name in selected
            ]
            if len(task_ids) != len(self.tasks):
                raise AxisBDataError(f"task metadata is ambiguous for {self.tasks!r}")
            self._row_positions = np.flatnonzero(
                np.isin(self._rows[:, 2], np.asarray(task_ids, dtype=np.uint64))
            )
        if self.split == "train":
            for item in self.index.files:
                if item.get("split") == "train" and _is_tomg(str(item.get("dataset", ""))):
                    raise AxisBDataError("TOMG records are forbidden in the training split")

    def __len__(self) -> int:
        if self._row_positions is not None:
            return int(self._row_positions.shape[0])
        return int(self._rows.shape[0])

    def task_counts(self, split: str | None = None) -> dict[str, int]:
        """Return task counts for this view, excluding other tasks."""

        requested = self.split if split is None else _canonical_split(split)
        counts = self.index.task_counts(requested)
        if self.tasks:
            return {
                task: int(counts[task])
                for task in self.tasks
                if int(counts.get(task, 0)) > 0
            }
        return counts

    def _handle(self, file_id: int) -> Any:
        handle = self._handles.get(file_id)
        if handle is None:
            path = self._files_by_id.get(file_id)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"indexed source file missing for file_id={file_id}: {path}")
            handle = path.open("rb")
            self._handles[file_id] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, numbers.Integral):
            index = int(index)
        else:
            raise TypeError(f"dataset index must be an integer, got {type(index).__name__}")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        base_index = int(self._row_positions[index]) if self._row_positions is not None else index
        row = self._rows[base_index]
        file_id, offset, task_id = (int(value) for value in row.tolist())
        handle = self._handle(file_id)
        handle.seek(offset)
        raw_line = handle.readline()
        if not raw_line:
            raise AxisBDataError(f"indexed offset points past EOF: file_id={file_id}, offset={offset}")
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid JSON at indexed offset {offset}") from exc
        if not isinstance(record, dict):
            raise SchemaError("indexed JSON record is not an object")
        target = record.get("target_bpe_ids")
        if not isinstance(target, list) or not target:
            raise SchemaError("index points to an empty target record")
        task = record.get("task")
        expected_task = self._task_names.get(task_id)
        if task != expected_task:
            raise SchemaError(
                f"task mismatch at index {index}: row={expected_task!r}, record={task!r}"
            )
        output = dict(record)
        output["_axis_b"] = {
            "dataset_index": index,
            "source_dataset_index": base_index,
            "file_id": file_id,
            "byte_offset": offset,
            "task_id": task_id,
            "task": expected_task,
            "split": self.split,
            "source": str(self._file_meta_by_id[file_id]["relative_source"]),
        }
        return output

    def task_names_for_split(self) -> tuple[str, ...]:
        if self.tasks:
            counts = self.task_counts(self.split)
            return tuple(task for task in self.tasks if counts.get(task, 0) > 0)
        names = (self.index.metadata.get("splits") or {}).get(self.split, {}).get("task_counts", {})
        return tuple(str(name) for name in names)

    def task_indices(self) -> dict[str, list[int]]:
        """Return exact dataset positions grouped by task.

        This scans only the compact uint64 sidecar, never the JSONL payload.
        """

        if self._task_indices_cache is not None:
            return self._task_indices_cache
        if self.task:
            self._task_indices_cache = {self.task: list(range(len(self)))} if len(self) else {}
            return self._task_indices_cache
        result: dict[str, list[int]] = defaultdict(list)
        for position in range(len(self)):
            base_index = int(self._row_positions[position]) if self._row_positions is not None else position
            task_id = int(self._rows[base_index, 2])
            result[self._task_names[task_id]].append(position)
        self._task_indices_cache = dict(result)
        return self._task_indices_cache

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown details
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        # Worker dataset copies random-access records but never need the
        # sampler's large task-position cache.
        state["_task_indices_cache"] = None
        return state


def _resolve_task_key(
    key: str | int,
    task_indices: Mapping[str, Sequence[int]],
    task_ids: Mapping[int, str] | None = None,
) -> str:
    if isinstance(key, numbers.Integral):
        if task_ids is not None and int(key) in task_ids and task_ids[int(key)] in task_indices:
            return str(task_ids[int(key)])
        names = tuple(task_indices)
        value = int(key)
        if value < 0 or value >= len(names):
            raise KeyError(key)
        return names[value]
    key = str(key)
    if key not in task_indices:
        raise KeyError(key)
    return key


class _TaskCursor:
    def __init__(self, indices: Sequence[int], rng: random.Random) -> None:
        self.indices = [int(value) for value in indices]
        if not self.indices:
            raise ValueError("cannot sample an empty task")
        self.rng = rng
        self.permutation = list(self.indices)
        self.rng.shuffle(self.permutation)
        self.position = 0

    def draw(self, count: int) -> list[int]:
        count = int(count)
        if count <= 0:
            return []
        n = len(self.permutation)
        if n < count:
            # Small tasks necessarily use replacement.  For normal tasks,
            # restarting a permutation at a boundary keeps each global batch
            # duplicate-free and makes DDP rank slices disjoint.
            return [self.rng.choice(self.permutation) for _ in range(count)]
        if self.position + count > n:
            self.rng.shuffle(self.permutation)
            self.position = 0
        out = self.permutation[self.position : self.position + count]
        self.position += count
        return out


class HomogeneousTaskBatchSampler:
    """Deterministic task-homogeneous, DDP-aware batch sampler.

    By default the number of global steps assigned to task ``i`` is allocated
    proportional to ``sqrt(n_i)`` using an exact largest-remainder allocation,
    where ``n_i`` is read from the sidecar task index.  Thus task weighting is
    based on eligible records (empty targets already removed), not on file
    sizes or a stale manifest estimate.  ``task_group_size`` keeps consecutive
    micro-batches on the same task.  A one-rank sampler with group size four
    then draws and slices exactly the same virtual global batch as four DDP
    ranks with group size one, which preserves task-homogeneous optimizer
    updates under gradient accumulation.
    """

    def __init__(
        self,
        dataset: AxisBJsonlDataset | Any,
        *,
        batch_sizes: int | Mapping[str | int, int] = 1,
        batch_size_by_task: int | Mapping[str | int, int] | None = None,
        per_task_batch_sizes: int | Mapping[str | int, int] | None = None,
        steps_per_epoch: int | None = None,
        steps_by_task: Mapping[str | int, int] | None = None,
        steps: Mapping[str | int, int] | None = None,
        per_task_steps: Mapping[str | int, int] | None = None,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        task_group_size: int = 1,
    ) -> None:
        self.dataset = dataset
        task_indices_raw = dataset.task_indices()
        self.task_indices: dict[str, list[int]] = {
            str(name): [int(x) for x in values] for name, values in task_indices_raw.items() if values
        }
        self._task_ids = getattr(getattr(dataset, "index", None), "task_names", None)
        if not self.task_indices:
            raise ValueError("cannot sample a dataset with no non-empty-target tasks")
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.rank, self.world_size = self._resolve_dist(rank, world_size)
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank/world_size out of range: {self.rank}/{self.world_size}")
        self.task_group_size = int(task_group_size)
        if self.task_group_size <= 0:
            raise ValueError("task_group_size must be positive")
        batch_aliases = [value for value in (batch_size_by_task, per_task_batch_sizes) if value is not None]
        if batch_aliases:
            if len(batch_aliases) > 1 or batch_sizes != 1:
                raise ValueError("pass only one of batch_sizes/batch_size_by_task/per_task_batch_sizes")
            batch_sizes = batch_aliases[0]
        self.batch_sizes = self._resolve_batch_sizes(batch_sizes)
        step_aliases = [value for value in (steps, per_task_steps) if value is not None]
        if step_aliases:
            if len(step_aliases) > 1 or steps_by_task is not None:
                raise ValueError("pass only one of steps_by_task/steps/per_task_steps")
            steps_by_task = step_aliases[0]
        if steps_by_task is not None and steps_per_epoch is not None:
            raise ValueError("pass either steps_by_task or steps_per_epoch, not both")
        if steps_by_task is not None:
            self.steps_by_task = self._resolve_steps(steps_by_task)
            self.steps_per_epoch = sum(self.steps_by_task.values())
            non_divisible = {
                task: count
                for task, count in self.steps_by_task.items()
                if count % self.task_group_size != 0
            }
            if non_divisible:
                raise ValueError(
                    "steps_by_task must be divisible by task_group_size: "
                    f"{non_divisible}"
                )
            self.group_steps_by_task = {
                task: count // self.task_group_size
                for task, count in self.steps_by_task.items()
            }
        else:
            if steps_per_epoch is None or int(steps_per_epoch) <= 0:
                raise ValueError("steps_per_epoch must be positive when steps_by_task is omitted")
            self.steps_per_epoch = int(steps_per_epoch)
            if self.steps_per_epoch % self.task_group_size != 0:
                raise ValueError("steps_per_epoch must be divisible by task_group_size")
            group_steps = self.steps_per_epoch // self.task_group_size
            self.group_steps_by_task = self._sqrt_step_allocation(group_steps)
            self.steps_by_task = {
                task: count * self.task_group_size
                for task, count in self.group_steps_by_task.items()
            }

    @staticmethod
    def _resolve_dist(rank: int | None, world_size: int | None) -> tuple[int, int]:
        if rank is not None or world_size is not None:
            return int(0 if rank is None else rank), int(1 if world_size is None else world_size)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank()), int(dist.get_world_size())
        except Exception:
            pass
        return 0, 1

    def _resolve_batch_sizes(self, value: int | Mapping[str | int, int]) -> dict[str, int]:
        if isinstance(value, numbers.Integral):
            size = int(value)
            if size <= 0:
                raise ValueError("batch size must be positive")
            return {task: size for task in self.task_indices}
        result: dict[str, int] = {}
        for key, raw_size in value.items():
            task = _resolve_task_key(key, self.task_indices, self._task_ids)
            size = int(raw_size)
            if size <= 0:
                raise ValueError(f"batch size for {task} must be positive")
            result[task] = size
        missing = set(self.task_indices).difference(result)
        if missing:
            raise ValueError(f"missing batch sizes for tasks: {sorted(missing)}")
        return result

    def _resolve_steps(self, value: Mapping[str | int, int]) -> dict[str, int]:
        result = {task: 0 for task in self.task_indices}
        for key, raw_steps in value.items():
            task = _resolve_task_key(key, self.task_indices, self._task_ids)
            steps = int(raw_steps)
            if steps < 0:
                raise ValueError(f"steps for {task} cannot be negative")
            result[task] = steps
        if sum(result.values()) <= 0:
            raise ValueError("steps_by_task allocates zero total steps")
        return result

    def _sqrt_step_allocation(self, total_steps: int) -> dict[str, int]:
        tasks = tuple(self.task_indices)
        weights = [math.sqrt(len(self.task_indices[task])) for task in tasks]
        denominator = sum(weights)
        exact = [total_steps * weight / denominator for weight in weights]
        floors = [int(math.floor(value)) for value in exact]
        remainder = int(total_steps - sum(floors))
        # Stable task order breaks equal fractional ties, making the schedule
        # reproducible across Python processes and DDP ranks.
        order = sorted(range(len(tasks)), key=lambda i: (-(exact[i] - floors[i]), i))
        for index in order[:remainder]:
            floors[index] += 1
        return {task: floors[i] for i, task in enumerate(tasks)}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _group_schedule(self) -> list[str]:
        schedule: list[str] = []
        for task in self.task_indices:
            schedule.extend([task] * int(self.group_steps_by_task.get(task, 0)))
        rng = random.Random(self.seed + 1_000_003 * self.epoch + 17)
        rng.shuffle(schedule)
        return schedule

    def _schedule(self) -> list[str]:
        return [
            task
            for task in self._group_schedule()
            for _ in range(self.task_group_size)
        ]

    def __len__(self) -> int:
        return int(self.steps_per_epoch)

    def __iter__(self) -> Iterator[list[int]]:
        cursors = {
            task: _TaskCursor(
                values,
                random.Random(self.seed + 7_919 * (position + 1) + 104_729 * self.epoch),
            )
            for position, (task, values) in enumerate(self.task_indices.items())
        }
        for task in self._group_schedule():
            local_batch_size = self.batch_sizes[task]
            virtual_world_size = self.world_size * self.task_group_size
            global_indices = cursors[task].draw(local_batch_size * virtual_world_size)
            for group_rank in range(self.task_group_size):
                virtual_rank = group_rank * self.world_size + self.rank
                start = virtual_rank * local_batch_size
                yield global_indices[start : start + local_batch_size]

    @property
    def task_step_counts(self) -> dict[str, int]:
        return dict(self.steps_by_task)

    @property
    def task_group_counts(self) -> dict[str, int]:
        return dict(self.group_steps_by_task)

    @property
    def task_record_counts(self) -> dict[str, int]:
        return {task: len(values) for task, values in self.task_indices.items()}

    @property
    def task_weights(self) -> dict[str, float]:
        raw = {task: math.sqrt(float(len(values))) for task, values in self.task_indices.items()}
        total = sum(raw.values())
        return {task: value / total for task, value in raw.items()}


def _to_plain(value: Any) -> Any:
    """Turn tokenizer tensors/arrays into nested Python lists."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _validate_raw_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise SchemaError(f"{label} must be a JSON list")
    result: list[int] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, numbers.Integral) or isinstance(raw, bool):
            raise SchemaError(f"{label}[{position}] is not an integer")
        token = int(raw)
        if token < RAW_BPE_MIN_ID or token > RAW_BPE_MAX_ID:
            raise SchemaError(
                f"{label}[{position}]={token} outside raw BPE range "
                f"[{RAW_BPE_MIN_ID},{RAW_BPE_MAX_ID}]"
            )
        result.append(token)
    return result


def _pad_sequences(sequences: Sequence[Sequence[int]], pad: int) -> list[list[int]]:
    width = max((len(sequence) for sequence in sequences), default=0)
    return [list(sequence) + [pad] * (width - len(sequence)) for sequence in sequences]


class AxisBSFTCollator:
    """Validate and collate formal BPE records for Axis-B SFT.

    ``return_tensors='auto'`` uses PyTorch when available, then NumPy, and
    finally plain lists.  Set it explicitly to ``'pt'`` or ``'np'`` in a
    training job.  The returned ``decoder_*`` keys match the model-independent
    contract used by ``axis_b_model.py``; aliases and logical variable-prefix
    masks are included for wrappers that compact absent conditions.
    """

    def __init__(
        self,
        tokenizer: Any | None,
        *,
        qwen_max_context: int | None = None,
        max_context: int | None = None,
        return_tensors: str = "auto",
        text_max_length: int = TEXT_MAX_LENGTH,
        molecule_max_length: int = MOLECULE_MAX_LENGTH,
        condition_token_budget: int = DEFAULT_CONDITION_TOKEN_BUDGET,
        molecule_prepend_bos: bool = False,
        molecule_condition_mode: str = "direct_qwen",
        strict_no_runtime_truncation: bool = False,
    ) -> None:
        if qwen_max_context is None:
            qwen_max_context = max_context
        elif max_context is not None and int(max_context) != int(qwen_max_context):
            raise ValueError("qwen_max_context and max_context disagree")
        if qwen_max_context is None or int(qwen_max_context) <= 0:
            raise ValueError("qwen_max_context must be positive")
        if int(qwen_max_context) < 4:
            raise ValueError("qwen_max_context must leave room for type/BOS/raw/EOS")
        if int(text_max_length) <= 0 or int(text_max_length) > TEXT_ENCODER_MAX_LENGTH:
            raise ValueError(
                f"text_max_length must be in [1,{TEXT_ENCODER_MAX_LENGTH}]"
            )
        if molecule_condition_mode not in MOLECULE_CONDITION_MODES:
            raise ValueError(
                f"molecule_condition_mode must be one of {sorted(MOLECULE_CONDITION_MODES)}"
            )
        minimum_molecule_length = 2 if molecule_condition_mode == "molbert" else 1
        if int(molecule_max_length) < minimum_molecule_length or int(molecule_max_length) > MOLECULE_MAX_LENGTH:
            raise ValueError(
                f"molecule_max_length must be in [{minimum_molecule_length},{MOLECULE_MAX_LENGTH}]"
            )
        if int(condition_token_budget) <= 0 or int(condition_token_budget) > max(TEXT_ENCODER_MAX_LENGTH, MOLECULE_MAX_LENGTH):
            raise ValueError("condition_token_budget must be positive and within encoder limits")
        if return_tensors not in {"auto", "pt", "np", "list"}:
            raise ValueError("return_tensors must be auto, pt, np, or list")
        self.tokenizer = tokenizer
        self.qwen_max_context = int(qwen_max_context)
        self.return_tensors = return_tensors
        self.text_max_length = int(text_max_length)
        self.molecule_max_length = int(molecule_max_length)
        self.condition_token_budget = int(condition_token_budget)
        self.molecule_condition_mode = str(molecule_condition_mode)
        self.strict_no_runtime_truncation = bool(strict_no_runtime_truncation)
        # MolBERT's pretraining stream is body+EOS (the online-pad dataset
        # does not add BOS).  Keep this explicit so a caller using a different
        # encoder protocol can opt into the documented BOS-framed variant.
        self.molecule_prepend_bos = bool(molecule_prepend_bos)

    def _convert(self, value: Any, *, dtype: str) -> Any:
        mode = self.return_tensors
        if mode == "list":
            return value
        if mode in {"auto", "pt"}:
            try:
                import torch

                torch_dtype = {
                    "long": torch.long,
                    "bool": torch.bool,
                    "float": torch.float32,
                }[dtype]
                return torch.tensor(value, dtype=torch_dtype)
            except ImportError:
                if mode == "pt":
                    raise ImportError("return_tensors='pt' requires PyTorch")
        if mode in {"auto", "np"}:
            try:
                np = _numpy()
                np_dtype = {"long": np.int64, "bool": np.bool_, "float": np.float32}[dtype]
                return np.asarray(value, dtype=np_dtype)
            except ImportError:
                if mode == "np":
                    raise
        return value

    def _tokenize_text(self, texts: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        if self.tokenizer is None:
            if any(texts):
                raise SchemaError("a tokenizer is required for text conditions")
            return [[PAD_TOKEN_ID] for _ in texts], [[0] for _ in texts]
        tokenizer_kwargs = {
            "padding": True,
            "truncation": not self.strict_no_runtime_truncation,
            "return_attention_mask": True,
        }
        if not self.strict_no_runtime_truncation:
            tokenizer_kwargs["max_length"] = self.text_max_length
        try:
            encoded = self.tokenizer(texts, **tokenizer_kwargs)
        except TypeError as exc:
            # Tiny test/dry-run tokenizers sometimes expose only the three
            # standard length arguments; HF tokenizers accept the richer form.
            if "return_attention_mask" not in str(exc):
                raise
            tokenizer_kwargs.pop("return_attention_mask")
            encoded = self.tokenizer(texts, **tokenizer_kwargs)
        if not isinstance(encoded, Mapping):
            raise SchemaError("injected tokenizer must return a mapping")
        if "input_ids" not in encoded:
            raise SchemaError("tokenizer output lacks input_ids")
        ids_raw = _to_plain(encoded["input_ids"])
        masks_raw = _to_plain(encoded.get("attention_mask"))
        if not isinstance(ids_raw, list):
            raise SchemaError("tokenizer input_ids must be a list or tensor")
        if isinstance(ids_raw, list) and (not ids_raw or isinstance(ids_raw[0], numbers.Integral)):
            ids_raw = [ids_raw]
        if masks_raw is not None and isinstance(masks_raw, list) and (
            not masks_raw or isinstance(masks_raw[0], numbers.Integral)
        ):
            masks_raw = [masks_raw]
        if masks_raw is None:
            masks_raw = [[1] * len(row) for row in ids_raw]
        if not isinstance(ids_raw, list) or not isinstance(masks_raw, list):
            raise SchemaError("tokenizer output must be batched lists")
        if len(ids_raw) != len(texts) or len(masks_raw) != len(texts):
            raise SchemaError("tokenizer batch size mismatch")
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for position, (row, mask) in enumerate(zip(ids_raw, masks_raw)):
            row = [int(value) for value in row]
            mask = [int(value) for value in mask]
            if len(row) != len(mask):
                raise SchemaError("tokenizer returned inconsistent ids/mask lengths")
            active_tokens = int(sum(mask))
            if (
                self.strict_no_runtime_truncation
                and bool(texts[position])
                and active_tokens > self.text_max_length
            ):
                raise SchemaError(
                    "strict_no_runtime_truncation: text condition would be truncated "
                    f"from {active_tokens} to text_max_length={self.text_max_length}"
                )
            if len(row) > self.text_max_length:
                raise SchemaError("tokenizer violated text max_length or mask shape")
            ids.append(row)
            masks.append(mask)
        return ids, masks

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("cannot collate an empty batch")

        raw_targets: list[list[int]] = []
        target_types: list[str] = []
        molecule_sequences: list[list[int]] = []
        molecule_present: list[bool] = []
        text_present: list[bool] = []
        texts: list[str] = []
        condition_modalities: list[str] = []
        condition_roles: list[str | None] = []
        condition_instructions: list[str | None] = []
        molecule_truncated: list[bool] = []
        task_names: list[str] = []
        record_ids: list[str] = []

        for batch_index, record in enumerate(batch):
            if not isinstance(record, Mapping):
                raise SchemaError(f"batch[{batch_index}] is not a mapping")
            target_type = record.get("target_type")
            if target_type not in TARGET_TYPE_TOKEN_IDS:
                raise SchemaError(f"unsupported target_type: {target_type!r}")
            raw_target = _validate_raw_ids(record.get("target_bpe_ids"), "target_bpe_ids")
            if not raw_target:
                raise SchemaError("SFT collator cannot train on an empty target")

            condition = record.get("condition")
            if not isinstance(condition, Mapping):
                raise SchemaError("condition must be an object")
            condition_type = condition.get("type")
            if condition_type not in CONDITION_TYPES:
                raise SchemaError(f"unsupported condition.type: {condition_type!r}")
            text_value = condition.get("text")
            if text_value is None:
                text_value = ""
            if not isinstance(text_value, str):
                raise SchemaError("condition.text must be a string or null")
            instruction_value = condition.get("instruction")
            if instruction_value is None:
                instruction_value = ""
            if not isinstance(instruction_value, str):
                raise SchemaError("condition.instruction must be a string or null")
            has_schema_metadata = any(
                key in condition for key in ("modality", "role", "instruction")
            )
            modality = condition.get("modality")
            if modality is None:
                modality = "text" if not condition.get("molecule_bpe_ids") and (text_value.strip() or instruction_value.strip()) else (
                    "small_molecule" if condition.get("molecule_bpe_ids") else "none"
                )
            if not isinstance(modality, str) or not modality.strip():
                raise SchemaError("condition.modality must be a non-empty string when present")
            modality = modality.strip()
            if modality not in CONDITION_MODALITIES:
                raise SchemaError(f"unsupported condition.modality: {modality!r}")
            role = condition.get("role")
            if role is not None and not isinstance(role, str):
                raise SchemaError("condition.role must be a string or null")
            role = role.strip() if isinstance(role, str) else None
            prompt_parts: list[str] = []
            if has_schema_metadata and modality != "none":
                prompt_parts.append(f"Input modality: {modality}.")
            if has_schema_metadata and role:
                prompt_parts.append(f"Input role: {role}.")
            if has_schema_metadata and instruction_value.strip():
                prompt_parts.append(instruction_value.strip())
            if text_value.strip():
                prompt_parts.append(text_value.strip())
            prompt = " ".join(prompt_parts)
            text_is_present = bool(prompt)
            raw_molecule = _validate_raw_ids(
                condition.get("molecule_bpe_ids") if condition.get("molecule_bpe_ids") is not None else [],
                "condition.molecule_bpe_ids",
            )
            molecule_is_present = bool(raw_molecule)
            expected = {
                "none": (False, False),
                "text": (True, False),
                "molecule": (False, True),
                "both": (True, True),
            }[str(condition_type)]
            if (text_is_present, molecule_is_present) != expected:
                raise SchemaError(
                    f"condition.type={condition_type!r} disagrees with present fields "
                    f"text={text_is_present}, molecule={molecule_is_present}"
                )

            # Direct mode reuses the exact Qwen FDDFS-BPE token stream.  The
            # flat stream already contains model id 5 for FDDFS ``<SEP>``
            # between disconnected components.  Qwen EOS=2 is a sample
            # terminator, not a component separator, so do not insert it here.
            # Legacy molBERT mode retains its historical body+EOS protocol.
            molecule_model = [x + MODEL_TOKEN_OFFSET for x in raw_molecule]
            if self.molecule_condition_mode == "molbert":
                molecule_model.append(EOS_TOKEN_ID)
                if self.molecule_prepend_bos:
                    molecule_model = [BOS_TOKEN_ID] + molecule_model
            molecule_was_truncated = False
            if molecule_is_present and len(molecule_model) > self.molecule_max_length:
                if self.strict_no_runtime_truncation:
                    raise SchemaError(
                        "strict_no_runtime_truncation: molecule condition for "
                        f"{record.get('id', batch_index)!r} would be truncated from "
                        f"{len(molecule_model)} to molecule_max_length="
                        f"{self.molecule_max_length}"
                    )
                if self.molecule_condition_mode == "molbert":
                    molecule_model = molecule_model[: self.molecule_max_length - 1] + [EOS_TOKEN_ID]
                else:
                    molecule_model = molecule_model[: self.molecule_max_length]
                molecule_was_truncated = True
            if not molecule_is_present:
                molecule_model = []

            raw_targets.append(raw_target)
            target_types.append(str(target_type))
            molecule_sequences.append(molecule_model)
            molecule_present.append(molecule_is_present)
            text_present.append(text_is_present)
            texts.append(prompt if text_is_present else "")
            condition_modalities.append(modality)
            condition_roles.append(role)
            condition_instructions.append(instruction_value.strip() or None)
            molecule_truncated.append(molecule_was_truncated)
            task_names.append(str(record.get("task", "")))
            record_ids.append(str(record.get("id", batch_index)))

        text_ids, text_masks = self._tokenize_text(texts)
        prefix_lengths: list[int] = []
        text_prefix_lengths: list[int] = []
        molecule_prefix_lengths: list[int] = []
        target_sequences: list[list[int]] = []
        target_labels: list[list[int]] = []
        target_type_ids: list[int] = []
        target_truncated: list[bool] = []
        raw_target_lengths: list[int] = []
        for index, raw_target in enumerate(raw_targets):
            unbounded_text_length = (
                int(sum(text_masks[index])) if text_present[index] else 0
            )
            unbounded_molecule_length = (
                len(molecule_sequences[index]) if molecule_present[index] else 0
            )
            if (
                self.strict_no_runtime_truncation
                and unbounded_text_length > self.condition_token_budget
            ):
                raise SchemaError(
                    "strict_no_runtime_truncation: text condition for "
                    f"{record_ids[index]!r} exceeds condition_token_budget: "
                    f"{unbounded_text_length}>{self.condition_token_budget}"
                )
            if (
                self.strict_no_runtime_truncation
                and unbounded_molecule_length > self.condition_token_budget
            ):
                raise SchemaError(
                    "strict_no_runtime_truncation: molecule condition for "
                    f"{record_ids[index]!r} exceeds condition_token_budget: "
                    f"{unbounded_molecule_length}>{self.condition_token_budget}"
                )
            text_length = min(
                unbounded_text_length, self.condition_token_budget
            )
            molecule_length = min(
                unbounded_molecule_length, self.condition_token_budget
            )
            # The operational Qwen window is a total budget, not a per-stream
            # budget.  Preserve the text stream first (it carries the task
            # instruction), then use the remaining slots for the structural
            # stream.  Normal records are far below this cap; this branch only
            # protects long/combined conditions from making target training
            # impossible.
            available_prefix = self.qwen_max_context - 4
            if text_length + molecule_length > available_prefix:
                if self.strict_no_runtime_truncation:
                    raise SchemaError(
                        "strict_no_runtime_truncation: combined condition prefix for "
                        f"{record_ids[index]!r} exceeds the context prefix budget: "
                        f"text={text_length} molecule={molecule_length} "
                        f"available={available_prefix} qwen_max_context="
                        f"{self.qwen_max_context}"
                    )
                text_length = min(text_length, available_prefix)
                molecule_length = min(
                    molecule_length, max(0, available_prefix - text_length)
                )
            prefix_length = text_length + molecule_length
            target_budget = self.qwen_max_context - prefix_length
            if target_budget < 4:
                raise SchemaError(
                    f"qwen_max_context={self.qwen_max_context} leaves no room for type/BOS/raw/EOS"
                )
            max_raw = target_budget - 3  # target type + BOS + raw + EOS
            target_raw_model = [x + MODEL_TOKEN_OFFSET for x in raw_target]
            was_truncated = len(target_raw_model) > max_raw
            if was_truncated:
                if self.strict_no_runtime_truncation:
                    raise SchemaError(
                        "strict_no_runtime_truncation: target for "
                        f"{record_ids[index]!r} would be truncated from "
                        f"{len(target_raw_model)} to {max_raw} raw tokens after a "
                        f"prefix of {prefix_length} within qwen_max_context="
                        f"{self.qwen_max_context}"
                    )
                target_raw_model = target_raw_model[:max_raw]
            target_model = [TARGET_TYPE_TOKEN_IDS[target_types[index]], BOS_TOKEN_ID]
            target_model.extend(target_raw_model)
            target_model.append(EOS_TOKEN_ID)
            labels = [-100, -100] + list(target_raw_model) + [EOS_TOKEN_ID]
            if target_model[-1] != EOS_TOKEN_ID:
                raise AssertionError("target EOS was not preserved")
            prefix_lengths.append(prefix_length)
            text_prefix_lengths.append(text_length)
            molecule_prefix_lengths.append(molecule_length)
            target_sequences.append(target_model)
            target_labels.append(labels)
            target_type_ids.append(TARGET_TYPE_TOKEN_IDS[target_types[index]])
            target_truncated.append(was_truncated)
            raw_target_lengths.append(len(raw_target))

        target_ids = _pad_sequences(target_sequences, PAD_TOKEN_ID)
        target_mask = _pad_sequences([[1] * len(row) for row in target_sequences], 0)
        labels = _pad_sequences(target_labels, -100)
        # A dummy two-position tensor keeps frozen encoders shape-stable when a
        # whole batch has no molecule condition; its attention is all zero.
        molecule_width = max(2, max((len(row) for row in molecule_sequences), default=0))
        molecule_ids = [row + [PAD_TOKEN_ID] * (molecule_width - len(row)) for row in molecule_sequences]
        molecule_masks = [
            ([1] * len(row) + [0] * (molecule_width - len(row))) if row else [0] * molecule_width
            for row in molecule_sequences
        ]

        full_lengths = [prefix + len(target) for prefix, target in zip(prefix_lengths, target_sequences)]
        full_width = max(full_lengths)
        full_attention = [
            [1] * length + [0] * (full_width - length) for length in full_lengths
        ]
        full_labels = []
        for prefix, row_labels, target_sequence in zip(prefix_lengths, labels, target_sequences):
            values = [-100] * prefix + row_labels[: len(target_sequence)]
            full_labels.append(values + [-100] * (full_width - len(values)))
        # ``labels`` is target-only; wrappers that prepend compact variable
        # prefixes can use ``full_labels`` directly.
        condition_prefix_mask = [[int(text), int(molecule)] for text, molecule in zip(text_present, molecule_present)]
        batch_out: dict[str, Any] = {
            "decoder_input_ids": self._convert(target_ids, dtype="long"),
            "decoder_attention_mask": self._convert(target_mask, dtype="long"),
            "decoder_labels": self._convert(labels, dtype="long"),
            "target_input_ids": self._convert(target_ids, dtype="long"),
            "target_attention_mask": self._convert(target_mask, dtype="long"),
            "target_labels": self._convert(labels, dtype="long"),
            "input_ids": self._convert(target_ids, dtype="long"),
            "attention_mask": self._convert(target_mask, dtype="long"),
            "labels": self._convert(labels, dtype="long"),
            "molecule_input_ids": self._convert(molecule_ids, dtype="long"),
            # Explicit alias: in direct mode these are Qwen model-space token
            # ids, while the historical name is retained for old callers.
            "molecule_token_ids": self._convert(molecule_ids, dtype="long"),
            "molecule_encoder_ids": self._convert(molecule_ids, dtype="long"),
            "molecule_attention_mask": self._convert(molecule_masks, dtype="long"),
            "text_input_ids": self._convert(text_ids, dtype="long"),
            "text_attention_mask": self._convert(text_masks, dtype="long"),
            "text_present": self._convert(text_present, dtype="bool"),
            "molecule_present": self._convert(molecule_present, dtype="bool"),
            "condition_prefix_mask": self._convert(condition_prefix_mask, dtype="long"),
            "target_type_ids": self._convert(target_type_ids, dtype="long"),
            "condition_prefix_lengths": self._convert(prefix_lengths, dtype="long"),
            "prefix_lengths": self._convert(prefix_lengths, dtype="long"),
            "text_prefix_lengths": self._convert(text_prefix_lengths, dtype="long"),
            "molecule_prefix_lengths": self._convert(molecule_prefix_lengths, dtype="long"),
            "target_lengths": self._convert([len(row) for row in target_sequences], dtype="long"),
            "full_attention_mask": self._convert(full_attention, dtype="long"),
            "full_labels": self._convert(full_labels, dtype="long"),
            "loss_start_positions": self._convert([prefix + 2 for prefix in prefix_lengths], dtype="long"),
            "loss_token_counts": self._convert([len(row) - 2 for row in target_sequences], dtype="long"),
            "raw_target_lengths": self._convert(raw_target_lengths, dtype="long"),
            "target_truncated": self._convert(target_truncated, dtype="bool"),
            "molecule_truncated": self._convert(molecule_truncated, dtype="bool"),
            "condition_token_budget": int(self.condition_token_budget),
            "condition_modalities": condition_modalities,
            "condition_roles": condition_roles,
            "condition_instructions": condition_instructions,
            "task_names": task_names,
            "molecule_sequences": molecule_sequences,
            "molecule_condition_mode": self.molecule_condition_mode,
            "strict_no_runtime_truncation": self.strict_no_runtime_truncation,
            "target_sequences": target_sequences,
        }
        return batch_out


# Short aliases used by launch scripts and notebooks.
AxisBSFTDataset = AxisBJsonlDataset
AxisBCollator = AxisBSFTCollator
AxisBTaskBatchSampler = HomogeneousTaskBatchSampler
TaskHomogeneousBatchSampler = HomogeneousTaskBatchSampler


__all__ = [
    "AxisBDataError",
    "AxisBIndex",
    "AxisBJsonlDataset",
    "AxisBSFTDataset",
    "AxisBSFTCollator",
    "AxisBCollator",
    "BOS_TOKEN_ID",
    "CONDITION_TYPES",
    "EOS_TOKEN_ID",
    "FDDFS_COMPONENT_SEP_MODEL_ID",
    "FDDFS_COMPONENT_SEP_RAW_ID",
    "HomogeneousTaskBatchSampler",
    "AxisBTaskBatchSampler",
    "TaskHomogeneousBatchSampler",
    "INDEX_VERSION",
    "MODEL_TOKEN_OFFSET",
    "ManifestError",
    "MOLECULE_MAX_LENGTH",
    "MOLECULE_CONDITION_MODES",
    "PAD_TOKEN_ID",
    "RAW_BPE_MAX_ID",
    "RAW_BPE_MIN_ID",
    "SchemaError",
    "TARGET_TYPE_TOKEN_IDS",
    "TEXT_ENCODER_MAX_LENGTH",
    "TEXT_MAX_LENGTH",
    "build_axis_b_index",
    "build_axis_b_indexes",
    "load_formal_manifest",
]
