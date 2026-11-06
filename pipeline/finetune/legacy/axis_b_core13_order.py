"""Core-13 natural-one-pass order sidecars and sampler.

The official protocol is intentionally independent of Python RNG.  A
worker-only builder scans record IDs once, partitions every task by its
record-ID SHA-256 order, applies the mixed-lineage stage interleave where
required, and publishes two reusable ``uint64`` NumPy arrays.  Training only
validates and mmap-loads those arrays; it never rescans the JSONL payloads.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import struct
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence


PROTOCOL_ID = "axis-core13-official-v1-20260821"
PROTOCOL = "core13-natural-one-pass-v1"
SEED = 42
MOSES_LINEAGE_ID = "core13_moses"
GUACAMOL_LINEAGE_ID = "core13_guacamol"
MIXED_LINEAGE_ID = "core13_mol1d_mix"
STAGES = ("alignment", "full")
SIDECAR_FORMAT = "axis_core13_order_sidecar_v2"

CORE13_MIXED_TASKS = (
    "smolinstruct_molecule_generation",
    "smolinstruct_forward_synthesis",
    "smolinstruct_retrosynthesis",
    "s2_moledit_addcomponent",
    "s2_moledit_delcomponent",
    "s2_moledit_subcomponent",
    "s2_molcustom_bondnum",
    "s2_molcustom_functionalgroup",
    "s2_molopt_logp",
    "s2_molopt_mr",
    "s2_molopt_qed",
)

CORE13_LINEAGE_TASKS: dict[str, tuple[str, ...]] = {
    MOSES_LINEAGE_ID: ("moses",),
    GUACAMOL_LINEAGE_ID: ("guacamol_distribution",),
    MIXED_LINEAGE_ID: CORE13_MIXED_TASKS,
}
CORE13_LINEAGES = tuple(CORE13_LINEAGE_TASKS)


class Core13OrderError(ValueError):
    """The Core-13 order contract or a sidecar is inconsistent."""


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - worker dependency
        raise ImportError("Core-13 order sidecars require NumPy on a worker") from exc
    return np


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
                    "utf-8"
                )
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def is_atomnum_task(task: str) -> bool:
    normalized = str(task).strip().lower().replace("_", "").replace("-", "")
    return normalized.endswith("atomnum")


def canonical_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    values = tuple(sorted(dict.fromkeys(str(task).strip() for task in tasks)))
    if not values or any(not task for task in values):
        raise Core13OrderError("Core-13 requires explicit non-empty task filters")
    atomnum = [task for task in values if is_atomnum_task(task)]
    if atomnum:
        raise Core13OrderError(
            f"AtomNum is materialize-only and forbidden in Core-13 training: {atomnum}"
        )
    return values


def canonical_lineage_tasks(
    lineage_id: str, tasks: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    """Validate one of the three frozen Core-13 lineage/task pairings."""

    lineage = str(lineage_id).strip()
    if lineage not in CORE13_LINEAGE_TASKS:
        raise Core13OrderError(
            f"unsupported Core-13 lineage {lineage!r}; expected one of "
            f"{list(CORE13_LINEAGES)}"
        )
    selected = canonical_tasks(tasks)
    expected = canonical_tasks(CORE13_LINEAGE_TASKS[lineage])
    if selected != expected:
        raise Core13OrderError(
            f"Core-13 lineage/task mismatch for {lineage}: "
            f"expected={list(expected)} actual={list(selected)}"
        )
    return lineage, selected


def stage_order_mode(lineage_id: str) -> str:
    """Mixed stages interleave by stage hash; single-task stages keep base order."""

    if lineage_id not in CORE13_LINEAGE_TASKS:
        raise Core13OrderError(f"unsupported Core-13 lineage: {lineage_id!r}")
    return "lineage_stage_hash_interleave" if lineage_id == MIXED_LINEAGE_ID else "task_record_hash"


def record_partition_hash(task: str, record_id: str, *, seed: int = SEED) -> bytes:
    payload = f"{PROTOCOL_ID}|{int(seed)}|{task}|{record_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def stage_interleave_hash(
    lineage_id: str,
    stage: str,
    task: str,
    record_id: str,
    *,
    seed: int = SEED,
) -> bytes:
    if stage not in STAGES:
        raise Core13OrderError(f"unsupported Core-13 stage: {stage!r}")
    payload = (
        f"{PROTOCOL_ID}|{int(seed)}|{lineage_id}|{stage}|{task}|{record_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _worker_guard(*, allow_non_worker: bool) -> None:
    if allow_non_worker:
        return
    hostname = socket.gethostname().lower()
    if not __import__("pipeline.runtime_host", fromlist=["is_compute_host"]).is_compute_host(hostname):
        raise RuntimeError(
            f"refusing Core-13 order construction on non-worker host {hostname!r}; "
            "build the sidecar inside an allocated CPU worker"
        )


def dataset_view_row_sha256(dataset: Any, *, chunk_rows: int = 1_000_000) -> str:
    """Fingerprint a filtered view using only the compact Axis-B row index."""

    np = _numpy()
    rows = getattr(dataset, "_rows", None)
    positions = getattr(dataset, "_row_positions", None)
    if rows is None:
        raise Core13OrderError("dataset has no compact Axis-B row index")
    digest = hashlib.sha256()
    digest.update(str(getattr(dataset, "split", "")).encode("utf-8"))
    digest.update(b"\0")
    tasks = canonical_tasks(tuple(getattr(dataset, "tasks", ()) or ()))
    digest.update(canonical_json_bytes(list(tasks)))
    digest.update(b"\0")
    length = int(len(dataset))
    digest.update(struct.pack("<Q", length))
    for start in range(0, length, int(chunk_rows)):
        end = min(length, start + int(chunk_rows))
        if positions is None:
            chunk = rows[start:end]
        else:
            chunk = rows[positions[start:end]]
        digest.update(np.asarray(chunk, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


def _index_manifest_sha256(dataset: Any) -> str:
    path = Path(dataset.index.index_dir) / "index_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def _descriptor_bytes(task: str, record_id: str, occurrence_index: int) -> bytes:
    return canonical_json_bytes([task, record_id, int(occurrence_index)]) + b"\n"


def _position_order_sha256(values: Any, *, chunk_rows: int = 1_000_000) -> str:
    np = _numpy()
    digest = hashlib.sha256()
    for start in range(0, int(len(values)), int(chunk_rows)):
        chunk = np.asarray(values[start : start + int(chunk_rows)], dtype="<u8")
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _safe_sidecar_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise Core13OrderError(f"sidecar path escapes root: {relative!r}") from exc
    return path


def _write_stage_array(
    connection: sqlite3.Connection,
    *,
    stage: str,
    count: int,
    output: Path,
) -> dict[str, Any]:
    np = _numpy()
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.", suffix=".npy", dir=str(output.parent)
    )
    os.close(fd)
    try:
        mapped = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.uint64, shape=(int(count),)
        )
        cursor = connection.execute(
            "SELECT task, record_id, occurrence_index, position "
            "FROM selected WHERE stage=? "
            "ORDER BY stage_hash, task, record_id, occurrence_index",
            (stage,),
        )
        position_digest = hashlib.sha256()
        record_digest = hashlib.sha256()
        written = 0
        while True:
            rows = cursor.fetchmany(65_536)
            if not rows:
                break
            positions = np.asarray([int(row[3]) for row in rows], dtype="<u8")
            mapped[written : written + len(rows)] = positions
            position_digest.update(positions.tobytes(order="C"))
            for task, record_id, occurrence, _position in rows:
                record_digest.update(
                    _descriptor_bytes(str(task), str(record_id), int(occurrence))
                )
            written += len(rows)
        if written != int(count):
            raise Core13OrderError(
                f"{stage} SQL/order count mismatch: expected={count} actual={written}"
            )
        mapped.flush()
        del mapped
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "file": output.name,
        "dtype": "uint64",
        "shape": [int(count)],
        "records": int(count),
        "bytes": int(output.stat().st_size),
        "sha256": sha256_file(output),
        "position_order_sha256": position_digest.hexdigest(),
        "record_order_sha256": record_digest.hexdigest(),
    }


def audit_partition_arrays(
    alignment: Any,
    full: Any,
    *,
    dataset_records: int,
    chunk_rows: int = 1_000_000,
) -> dict[str, Any]:
    """Audit duplicate, overlap, omission, and union without Python sets."""

    np = _numpy()
    total = int(dataset_records)
    seen = np.zeros(total, dtype=np.uint8)
    duplicates = {"alignment": 0, "full": 0}
    overlap = 0
    out_of_range = 0
    for stage_code, (stage, values) in enumerate(
        (("alignment", alignment), ("full", full)), start=1
    ):
        for start in range(0, int(len(values)), int(chunk_rows)):
            raw = np.asarray(values[start : start + int(chunk_rows)], dtype=np.uint64)
            valid = raw[raw < total]
            out_of_range += int(raw.shape[0] - valid.shape[0])
            unique = np.unique(valid)
            duplicates[stage] += int(valid.shape[0] - unique.shape[0])
            previous = seen[unique]
            duplicates[stage] += int(np.count_nonzero(previous == stage_code))
            overlap += int(np.count_nonzero((previous != 0) & (previous != stage_code)))
            seen[unique] = stage_code
    union = int(np.count_nonzero(seen))
    omitted = total - union
    exact = bool(
        out_of_range == 0
        and duplicates["alignment"] == 0
        and duplicates["full"] == 0
        and overlap == 0
        and omitted == 0
        and int(len(alignment)) + int(len(full)) == total
    )
    return {
        "dataset_records": total,
        "alignment_records": int(len(alignment)),
        "full_records": int(len(full)),
        "alignment_duplicate_positions": int(duplicates["alignment"]),
        "full_duplicate_positions": int(duplicates["full"]),
        "alignment_full_overlap_positions": int(overlap),
        "out_of_range_positions": int(out_of_range),
        "union_positions": union,
        "omitted_positions": int(omitted),
        "disjoint": overlap == 0,
        "union_is_dataset": omitted == 0 and union == total,
        "exact_once": exact,
    }


def _stage_task_counts(dataset: Any, values: Any) -> dict[str, int]:
    """Count stage tasks through compact row arrays, never JSONL payloads."""

    np = _numpy()
    rows = dataset._rows
    view_positions = dataset._row_positions
    task_names = dataset.index.task_names
    counts: Counter[str] = Counter()
    for start in range(0, int(len(values)), 1_000_000):
        selected = np.asarray(values[start : start + 1_000_000], dtype=np.int64)
        base = selected if view_positions is None else view_positions[selected]
        task_ids, task_counts = np.unique(rows[base, 2], return_counts=True)
        for task_id, count in zip(task_ids.tolist(), task_counts.tolist()):
            counts[str(task_names[int(task_id)])] += int(count)
    return dict(counts)


@dataclass(frozen=True)
class Core13OrderSidecar:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    def order(self, stage: str) -> Any:
        if stage not in STAGES:
            raise Core13OrderError(stage)
        item = self.manifest["orders"][stage]
        path = _safe_sidecar_path(self.root, str(item["file"]))
        np = _numpy()
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def natural_epoch_audit(self, stage: str) -> dict[str, Any]:
        if stage not in STAGES:
            raise Core13OrderError(stage)
        orders = self.manifest["orders"]
        selected = orders[stage]
        return {
            "protocol": PROTOCOL,
            "protocol_id": self.manifest["protocol_id"],
            "lineage_id": self.manifest["lineage_id"],
            "stage_order_mode": self.manifest["stage_order_mode"],
            "tasks": list(self.manifest["tasks"]),
            "stage": stage,
            "seed": int(self.manifest["seed"]),
            "dataset_records": int(self.manifest["dataset_records"]),
            "task_counts": dict(self.manifest["task_counts"]),
            "selected_records": int(selected["records"]),
            "selected_position_order_sha256": selected["position_order_sha256"],
            "selected_record_order_sha256": selected["record_order_sha256"],
            "alignment_position_order_sha256": orders["alignment"][
                "position_order_sha256"
            ],
            "full_position_order_sha256": orders["full"]["position_order_sha256"],
            "alignment_record_order_sha256": orders["alignment"][
                "record_order_sha256"
            ],
            "full_record_order_sha256": orders["full"]["record_order_sha256"],
            "data_manifest_sha256": self.manifest["data_manifest_sha256"],
            "index_manifest_sha256": self.manifest["index_manifest_sha256"],
            "dataset_view_row_sha256": self.manifest["dataset_view_row_sha256"],
            "order_sidecar_manifest_sha256": self.manifest_sha256,
            "partition_audit_sha256": self.manifest["partition_audit_sha256"],
            "partition_audit": dict(self.manifest["partition_audit"]),
        }


def build_core13_order_sidecar(
    dataset: Any,
    output_dir: Path | str,
    *,
    lineage_id: str = MIXED_LINEAGE_ID,
    tasks: Sequence[str] = CORE13_MIXED_TASKS,
    seed: int = SEED,
    temp_dir: Path | str | None = None,
    allow_non_worker: bool = False,
    log_every: int = 100_000,
) -> Core13OrderSidecar:
    """Scan record IDs once and atomically publish reusable stage orders."""

    _worker_guard(allow_non_worker=allow_non_worker)
    if int(seed) != SEED:
        raise Core13OrderError(f"Core-13 seed is frozen at {SEED}, got {seed}")
    if getattr(dataset, "split", None) != "train":
        raise Core13OrderError("Core-13 order builder requires the train split")
    lineage_id, selected_tasks = canonical_lineage_tasks(lineage_id, tasks)
    dataset_tasks = canonical_tasks(tuple(getattr(dataset, "tasks", ()) or ()))
    if dataset_tasks != selected_tasks:
        raise Core13OrderError(
            f"dataset task view mismatch: dataset={dataset_tasks} requested={selected_tasks}"
        )
    task_counts = {str(k): int(v) for k, v in dataset.task_counts("train").items()}
    if set(task_counts) != set(selected_tasks) or any(value <= 0 for value in task_counts.values()):
        raise Core13OrderError(
            f"every selected task must have train records: counts={task_counts}"
        )
    if int(len(dataset)) != sum(task_counts.values()):
        raise Core13OrderError("filtered dataset length does not equal selected task counts")
    order_mode = stage_order_mode(lineage_id)

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        return load_core13_order_sidecar(
            dataset,
            root,
            lineage_id=lineage_id,
            tasks=selected_tasks,
            seed=seed,
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"incomplete/non-empty Core-13 sidecar directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    temp_root = Path(
        temp_dir or os.environ.get("TMPDIR") or "./outputs/tmp"
    ).expanduser().resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    fd, database_name = tempfile.mkstemp(
        prefix="core13_order_", suffix=".sqlite3", dir=str(temp_root)
    )
    os.close(fd)
    database_path = Path(database_name)
    started = time.time()
    try:
        connection = sqlite3.connect(str(database_path))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE records ("
                "task TEXT NOT NULL, record_id TEXT NOT NULL, "
                "position INTEGER PRIMARY KEY, base_hash BLOB NOT NULL)"
            )
            pending: list[tuple[str, str, int, bytes]] = []
            observed: Counter[str] = Counter()
            for position in range(len(dataset)):
                record = dataset[position]
                task = record.get("task")
                record_id = record.get("id")
                if task not in selected_tasks:
                    raise Core13OrderError(
                        f"record {position} has unselected task {task!r}"
                    )
                if not isinstance(record_id, str) or not record_id:
                    raise Core13OrderError(
                        f"record {position} in task {task!r} has no stable string id"
                    )
                pending.append(
                    (
                        str(task),
                        record_id,
                        int(position),
                        record_partition_hash(str(task), record_id, seed=seed),
                    )
                )
                observed[str(task)] += 1
                if len(pending) >= 10_000:
                    connection.executemany(
                        "INSERT INTO records(task,record_id,position,base_hash) VALUES(?,?,?,?)",
                        pending,
                    )
                    connection.commit()
                    pending.clear()
                if log_every and (position + 1) % int(log_every) == 0:
                    print(
                        f"core13_order_scan records={position + 1:,}/{len(dataset):,}",
                        flush=True,
                    )
            if pending:
                connection.executemany(
                    "INSERT INTO records(task,record_id,position,base_hash) VALUES(?,?,?,?)",
                    pending,
                )
                connection.commit()
            if dict(observed) != task_counts:
                raise Core13OrderError(
                    f"JSONL/index task counts disagree: observed={dict(observed)} "
                    f"index={task_counts}"
                )
            connection.execute(
                "CREATE INDEX records_task_order "
                "ON records(task,base_hash,record_id,position)"
            )
            connection.execute(
                "CREATE TABLE selected ("
                "stage TEXT NOT NULL, task TEXT NOT NULL, record_id TEXT NOT NULL, "
                "occurrence_index INTEGER NOT NULL, position INTEGER PRIMARY KEY, "
                "stage_hash BLOB NOT NULL)"
            )

            duplicate_row = connection.execute(
                "SELECT COALESCE(SUM(n-1),0), COUNT(*) FROM ("
                "SELECT COUNT(*) AS n FROM records GROUP BY task,record_id HAVING n>1)"
            ).fetchone()
            duplicate_occurrences = int(duplicate_row[0])
            duplicate_ids = int(duplicate_row[1])
            task_audits: dict[str, Any] = {}
            selected_pending: list[tuple[str, str, str, int, int, bytes]] = []
            for task in selected_tasks:
                count = int(task_counts[task])
                alignment_count = count // 5
                full_count = count - alignment_count
                task_digest = hashlib.sha256()
                cursor = connection.execute(
                    "SELECT record_id,position,base_hash FROM records WHERE task=? "
                    "ORDER BY base_hash,task,record_id,position",
                    (task,),
                )
                previous_id: str | None = None
                occurrence = -1
                seen = 0
                for record_id, position, base_hash in cursor:
                    record_id = str(record_id)
                    if record_id == previous_id:
                        occurrence += 1
                    else:
                        previous_id = record_id
                        occurrence = 0
                    stage = "alignment" if seen < alignment_count else "full"
                    task_digest.update(_descriptor_bytes(task, record_id, occurrence))
                    selected_pending.append(
                        (
                            stage,
                            task,
                            record_id,
                            occurrence,
                            int(position),
                            (
                                stage_interleave_hash(
                                    lineage_id,
                                    stage,
                                    task,
                                    record_id,
                                    seed=seed,
                                )
                                if order_mode == "lineage_stage_hash_interleave"
                                else bytes(base_hash)
                            ),
                        )
                    )
                    if len(selected_pending) >= 10_000:
                        connection.executemany(
                            "INSERT INTO selected(stage,task,record_id,occurrence_index,position,stage_hash) "
                            "VALUES(?,?,?,?,?,?)",
                            selected_pending,
                        )
                        connection.commit()
                        selected_pending.clear()
                    seen += 1
                if seen != count:
                    raise Core13OrderError(
                        f"ordered task count mismatch for {task}: {seen} != {count}"
                    )
                task_audits[task] = {
                    "records": count,
                    "alignment_records": alignment_count,
                    "full_records": full_count,
                    "alignment_expected_floor_n_over_5": alignment_count,
                    "ordered_record_ids_sha256": task_digest.hexdigest(),
                    "disjoint": True,
                    "union_records": count,
                }
            if selected_pending:
                connection.executemany(
                    "INSERT INTO selected(stage,task,record_id,occurrence_index,position,stage_hash) "
                    "VALUES(?,?,?,?,?,?)",
                    selected_pending,
                )
                connection.commit()
            connection.execute(
                "CREATE INDEX selected_stage_order "
                "ON selected(stage,stage_hash,task,record_id,occurrence_index)"
            )
            connection.commit()

            orders: dict[str, Any] = {}
            for stage in STAGES:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM selected WHERE stage=?", (stage,)
                    ).fetchone()[0]
                )
                if count <= 0:
                    raise Core13OrderError(f"Core-13 {stage} partition is empty")
                orders[stage] = _write_stage_array(
                    connection,
                    stage=stage,
                    count=count,
                    output=root / f"{stage}.order.npy",
                )
        finally:
            connection.close()

        np = _numpy()
        alignment = np.load(root / orders["alignment"]["file"], mmap_mode="r")
        full = np.load(root / orders["full"]["file"], mmap_mode="r")
        global_audit = audit_partition_arrays(
            alignment, full, dataset_records=int(len(dataset))
        )
        if not global_audit["exact_once"]:
            raise Core13OrderError(f"Core-13 partition audit failed: {global_audit}")
        partition_audit = {
            "global": global_audit,
            "per_task": task_audits,
        }
        payload = {
            "complete": True,
            "format": SIDECAR_FORMAT,
            "protocol": PROTOCOL,
            "protocol_id": PROTOCOL_ID,
            "seed": int(seed),
            "lineage_id": lineage_id,
            "stage_order_mode": order_mode,
            "tasks": list(selected_tasks),
            "dataset_records": int(len(dataset)),
            "task_counts": {task: int(task_counts[task]) for task in selected_tasks},
            "data_root": str(dataset.data_root),
            "data_manifest_sha256": str(dataset.index.metadata.get("manifest_sha256") or ""),
            "index_dir": str(dataset.index.index_dir),
            "index_manifest_sha256": _index_manifest_sha256(dataset),
            "dataset_view_row_sha256": dataset_view_row_sha256(dataset),
            "orders": orders,
            "partition_audit": partition_audit,
            "partition_audit_sha256": hashlib.sha256(
                canonical_json_bytes(partition_audit)
            ).hexdigest(),
            "duplicate_record_ids": duplicate_ids,
            "duplicate_record_occurrences": duplicate_occurrences,
            "tie_break": "task_id,record_id,occurrence_index; occurrence follows dataset position",
            "hash_contract": {
                "task_partition": (
                    'SHA256("axis-core13-official-v1-20260821|42|<task_id>|<record_id>")'
                ),
                "stage_interleave": (
                    f'SHA256("{PROTOCOL_ID}|{int(seed)}|{lineage_id}|'
                    '<stage>|<task_id>|<record_id>")'
                    if order_mode == "lineage_stage_hash_interleave"
                    else None
                ),
                "stage_order": (
                    "stage_interleave_hash,task_id,record_id,occurrence_index"
                    if order_mode == "lineage_stage_hash_interleave"
                    else "task_partition_hash,task_id,record_id,occurrence_index"
                ),
            },
            "created_at_unix": time.time(),
            "host": socket.gethostname(),
            "elapsed_sec": time.time() - started,
        }
        _atomic_json(manifest_path, payload)
        return load_core13_order_sidecar(
            dataset,
            root,
            lineage_id=lineage_id,
            tasks=selected_tasks,
            seed=seed,
        )
    finally:
        try:
            database_path.unlink()
        except FileNotFoundError:
            pass


def load_core13_order_sidecar(
    dataset: Any,
    root: Path | str,
    *,
    lineage_id: str = MIXED_LINEAGE_ID,
    tasks: Sequence[str] = CORE13_MIXED_TASKS,
    seed: int = SEED,
) -> Core13OrderSidecar:
    """Strictly validate and mmap-ready a prebuilt order sidecar."""

    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Core13OrderError(f"invalid Core-13 order manifest: {manifest_path}") from exc
    lineage_id, selected_tasks = canonical_lineage_tasks(lineage_id, tasks)
    if getattr(dataset, "split", None) != "train":
        raise Core13OrderError("Core-13 order sidecars can only load against train")
    dataset_tasks = canonical_tasks(tuple(getattr(dataset, "tasks", ()) or ()))
    if dataset_tasks != selected_tasks:
        raise Core13OrderError(
            f"dataset task view mismatch: dataset={dataset_tasks} requested={selected_tasks}"
        )
    current_task_counts = {
        str(task): int(count)
        for task, count in dataset.task_counts("train").items()
    }
    if (
        set(current_task_counts) != set(selected_tasks)
        or any(count <= 0 for count in current_task_counts.values())
        or sum(current_task_counts.values()) != int(len(dataset))
    ):
        raise Core13OrderError(
            f"invalid filtered Core-13 train view: counts={current_task_counts} "
            f"records={len(dataset)}"
        )
    expected = {
        "complete": True,
        "format": SIDECAR_FORMAT,
        "protocol": PROTOCOL,
        "protocol_id": PROTOCOL_ID,
        "seed": int(seed),
        "lineage_id": str(lineage_id),
        "stage_order_mode": stage_order_mode(lineage_id),
        "tasks": list(selected_tasks),
        "dataset_records": int(len(dataset)),
        "task_counts": {
            task: int(current_task_counts[task])
            for task in selected_tasks
        },
        "data_root": str(dataset.data_root),
        "data_manifest_sha256": str(dataset.index.metadata.get("manifest_sha256") or ""),
        "index_dir": str(dataset.index.index_dir),
        "index_manifest_sha256": _index_manifest_sha256(dataset),
        "dataset_view_row_sha256": dataset_view_row_sha256(dataset),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise Core13OrderError(
            f"Core-13 order sidecar identity mismatch: {json.dumps(mismatches, sort_keys=True)}"
        )
    orders = payload.get("orders")
    if not isinstance(orders, dict) or set(orders) != set(STAGES):
        raise Core13OrderError("Core-13 order manifest must contain alignment/full")
    arrays: dict[str, Any] = {}
    np = _numpy()
    for stage in STAGES:
        item = orders[stage]
        if not isinstance(item, dict):
            raise Core13OrderError(f"invalid {stage} order metadata")
        path = _safe_sidecar_path(root, str(item.get("file") or ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(item.get("bytes", -1)) != int(path.stat().st_size):
            raise Core13OrderError(f"{stage} order size mismatch")
        if sha256_file(path) != item.get("sha256"):
            raise Core13OrderError(f"{stage} order SHA-256 mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype(np.uint64) or array.ndim != 1:
            raise Core13OrderError(f"{stage} order must be one-dimensional uint64")
        if item.get("dtype") != "uint64" or item.get("shape") != [int(array.shape[0])]:
            raise Core13OrderError(f"{stage} order dtype/shape metadata mismatch")
        if int(array.shape[0]) != int(item.get("records", -1)):
            raise Core13OrderError(f"{stage} order record count mismatch")
        if _position_order_sha256(array) != item.get("position_order_sha256"):
            raise Core13OrderError(f"{stage} position-order digest mismatch")
        arrays[stage] = array

    audit = audit_partition_arrays(
        arrays["alignment"], arrays["full"], dataset_records=len(dataset)
    )
    saved_audit = (payload.get("partition_audit") or {}).get("global")
    if audit != saved_audit or not audit["exact_once"]:
        raise Core13OrderError(
            f"Core-13 alignment/full disjoint-union audit mismatch: {audit}"
        )
    saved_per_task = (payload.get("partition_audit") or {}).get("per_task")
    if not isinstance(saved_per_task, dict) or set(saved_per_task) != set(selected_tasks):
        raise Core13OrderError("Core-13 sidecar has no per-task partition audit")
    alignment_counts = _stage_task_counts(dataset, arrays["alignment"])
    full_counts = _stage_task_counts(dataset, arrays["full"])
    for task in selected_tasks:
        total = int(expected["task_counts"][task])
        expected_alignment = total // 5
        actual_alignment = int(alignment_counts.get(task, 0))
        actual_full = int(full_counts.get(task, 0))
        saved = saved_per_task.get(task) or {}
        if (
            actual_alignment != expected_alignment
            or actual_full != total - expected_alignment
            or int(saved.get("records", -1)) != total
            or int(saved.get("alignment_records", -1)) != actual_alignment
            or int(saved.get("full_records", -1)) != actual_full
            or int(saved.get("union_records", -1)) != total
            or not bool(saved.get("disjoint"))
        ):
            raise Core13OrderError(
                f"Core-13 per-task partition mismatch for {task}: "
                f"alignment={actual_alignment} full={actual_full} saved={saved}"
            )
    partition_digest = hashlib.sha256(
        canonical_json_bytes(payload["partition_audit"])
    ).hexdigest()
    if partition_digest != payload.get("partition_audit_sha256"):
        raise Core13OrderError("Core-13 partition audit SHA-256 mismatch")
    return Core13OrderSidecar(
        root=root,
        manifest=payload,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def validate_core13_resume_audits(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Require full to resume the exact alignment lineage/task/data/order."""

    if current.get("stage") != "full":
        raise Core13OrderError("current Core-13 resume audit must be the full stage")
    required = {
        "protocol": PROTOCOL,
        "protocol_id": PROTOCOL_ID,
        "lineage_id": current.get("lineage_id"),
        "stage_order_mode": current.get("stage_order_mode"),
        "tasks": current.get("tasks"),
        "seed": current.get("seed"),
        "dataset_records": current.get("dataset_records"),
        "task_counts": current.get("task_counts"),
        "data_manifest_sha256": current.get("data_manifest_sha256"),
        "index_manifest_sha256": current.get("index_manifest_sha256"),
        "dataset_view_row_sha256": current.get("dataset_view_row_sha256"),
        "order_sidecar_manifest_sha256": current.get(
            "order_sidecar_manifest_sha256"
        ),
        "partition_audit_sha256": current.get("partition_audit_sha256"),
        "partition_audit": current.get("partition_audit"),
        "alignment_position_order_sha256": current.get(
            "alignment_position_order_sha256"
        ),
        "full_position_order_sha256": current.get("full_position_order_sha256"),
        "alignment_record_order_sha256": current.get(
            "alignment_record_order_sha256"
        ),
        "full_record_order_sha256": current.get("full_record_order_sha256"),
        "selected_position_order_sha256": current.get(
            "alignment_position_order_sha256"
        ),
        "selected_record_order_sha256": current.get(
            "alignment_record_order_sha256"
        ),
        "selected_records": int(
            ((current.get("partition_audit") or {}).get("global") or {}).get(
                "alignment_records", -1
            )
        ),
        "stage": "alignment",
        "complete": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": previous.get(key)}
        for key, expected in required.items()
        if previous.get(key) != expected
    }
    if mismatches:
        raise Core13OrderError(
            "Core-13 full resume does not match alignment lineage/task/data/order: "
            + json.dumps(mismatches, sort_keys=True)
        )


class NaturalOnePassOrderBatchSampler:
    """Yield a frozen mmap order exactly once, including a real short tail."""

    def __init__(self, order: Sequence[int], batch_size: int) -> None:
        self.order = order
        self.batch_size = int(batch_size)
        if len(order) <= 0:
            raise Core13OrderError("natural one-pass stage order is empty")
        if self.batch_size <= 0:
            raise Core13OrderError("batch_size must be positive")
        self._yielded_records = 0
        self._yielded_digest = hashlib.sha256()

    def __len__(self) -> int:
        return int(math.ceil(len(self.order) / self.batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        self._yielded_records = 0
        self._yielded_digest = hashlib.sha256()
        for start in range(0, len(self.order), self.batch_size):
            values = [int(value) for value in self.order[start : start + self.batch_size]]
            self._yielded_digest.update(
                struct.pack("<" + "Q" * len(values), *values)
            )
            self._yielded_records += len(values)
            yield values

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != 0:
            raise Core13OrderError("Core-13 natural one-pass supports exactly epoch 0")

    @property
    def selected_records(self) -> int:
        return int(len(self.order))

    @property
    def final_batch_size(self) -> int:
        remainder = len(self.order) % self.batch_size
        return int(remainder if remainder else self.batch_size)

    @property
    def yielded_records(self) -> int:
        return int(self._yielded_records)

    @property
    def yielded_order_sha256(self) -> str:
        return self._yielded_digest.hexdigest()
