"""Bounded, per-device NVIDIA observations for fail-closed slot admission.

A successful empty compute query means idle; a failed query never does.  This
module does not launch workloads or signal existing processes. On timeout it
may kill only the query child it created; it never waits indefinitely to reap it.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable


INVENTORY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "temperature.gpu",
    "utilization.gpu",
)
COMPUTE_FIELDS = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
QUERY_TIMEOUT_SECONDS = 2.0
QUERY_CLEANUP_TIMEOUT_SECONDS = 0.2
STUCK_QUERY_MIN_AGE_SECONDS = 5.0
# Strong references prevent Popen finalization from losing track of a query
# that SIGKILL cannot yet reap (for example, an uninterruptible driver call).
_PENDING_CHILDREN: dict[int, subprocess.Popen] = {}
_PENDING_LOCK = threading.Lock()
_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_.]*\Z")
_INTEGER = re.compile(r"[0-9]+\Z")


def _validate_gpu_id(gpu_id: int) -> None:
    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise RuntimeError(f"invalid physical GPU index: {gpu_id!r}")


def stuck_nvidia_smi_pids(proc_root: Path = Path("/proc")) -> list[int]:
    """Read-only detection of this UID's aged, uninterruptible NVIDIA queries."""
    try:
        uptime = float((proc_root / "uptime").read_text(encoding="utf-8").split()[0])
        ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
        if not math.isfinite(uptime) or uptime < 0 or ticks_per_second <= 0:
            raise ValueError("invalid process age clock")
        entries = list(proc_root.iterdir())
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError(f"cannot inspect existing NVIDIA query processes: {exc}") from exc
    uid = os.getuid()
    blocked: list[int] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            raw = (entry / "stat").read_text(encoding="utf-8")
            opening, closing = raw.find("("), raw.rfind(")")
            if opening < 0 or closing <= opening:
                continue
            tail = raw[closing + 2 :].split()
            if len(tail) < 20 or tail[0] != "D":
                continue
            age = uptime - int(tail[19]) / ticks_per_second
            if age < STUCK_QUERY_MIN_AGE_SECONDS:
                continue
            comm = raw[opening + 1 : closing]
            if comm != "nvidia-smi":
                first_arg = (entry / "cmdline").read_bytes().split(b"\0", 1)[0]
                if Path(os.fsdecode(first_arg)).name != "nvidia-smi":
                    continue
            blocked.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            # /proc entries can disappear while being inspected. Nothing in
            # this scan ever signals a PID, regardless of its apparent identity.
            continue
    return sorted(blocked)


def _reap_pending_locked() -> list[int]:
    """Caller holds _PENDING_LOCK; poll is waitpid(WNOHANG), never wait()."""
    for pid, child in list(_PENDING_CHILDREN.items()):
        try:
            exited = child.poll() is not None
        except OSError:
            exited = False
        if exited:
            del _PENDING_CHILDREN[pid]
    return sorted(_PENDING_CHILDREN)


def _close_query_pipes(child: subprocess.Popen) -> None:
    for pipe in (child.stdout, child.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


def _retire_failed_query(child: subprocess.Popen) -> None:
    # Register before attempting cleanup so another thread cannot start more
    # queries while this timed-out child is being killed/reaped.
    with _PENDING_LOCK:
        _PENDING_CHILDREN[child.pid] = child
    try:
        try:
            child.kill()
        except OSError:
            pass
        try:
            child.communicate(timeout=QUERY_CLEANUP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            pass
    finally:
        _close_query_pipes(child)
        with _PENDING_LOCK:
            _reap_pending_locked()


def query_gpu_csv(
    gpu_id: int,
    selector: str,
    fields: tuple[str, ...],
    timeout_seconds: float = QUERY_TIMEOUT_SECONDS,
) -> list[list[str]]:
    """Query exactly one physical GPU, rejecting partial or malformed output."""
    _validate_gpu_id(gpu_id)
    if selector not in {"gpu", "compute-apps"}:
        raise RuntimeError(f"invalid NVIDIA query selector: {selector!r}")
    if (
        not isinstance(fields, tuple)
        or not fields
        or any(not isinstance(field, str) or not _FIELD.fullmatch(field) for field in fields)
        or len(set(fields)) != len(fields)
    ):
        raise RuntimeError(f"invalid NVIDIA query fields: {fields!r}")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise RuntimeError(f"invalid NVIDIA query timeout: {timeout_seconds!r}")
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        f"--query-{selector}={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    existing_stuck = stuck_nvidia_smi_pids()
    if existing_stuck:
        raise RuntimeError(f"existing aged D-state NVIDIA queries block new probes: {existing_stuck}")
    child: subprocess.Popen | None = None
    try:
        with _PENDING_LOCK:
            pending = _reap_pending_locked()
            if pending:
                raise RuntimeError(f"unreaped owned NVIDIA query children block new probes: {pending}")
            child = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        stdout, stderr = child.communicate(timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        if child is not None:
            _retire_failed_query(child)
        raise RuntimeError(
            f"GPU {gpu_id} NVIDIA {selector} query failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if child is not None:
            _close_query_pipes(child)
    if child.returncode != 0:
        # In particular, exit 255 may include plausible partial stdout. Never
        # parse it or use it as evidence that any process/GPU is absent.
        raise RuntimeError(
            f"GPU {gpu_id} NVIDIA {selector} query exited {child.returncode}: "
            f"{stderr.strip()}"
        )
    rows: list[list[str]] = []
    try:
        for raw in csv.reader(io.StringIO(stdout), skipinitialspace=True, strict=True):
            if not raw or (len(raw) == 1 and not raw[0].strip()):
                continue
            row = [cell.strip() for cell in raw]
            if len(row) != len(fields) or any(not cell for cell in row):
                raise RuntimeError(
                    f"GPU {gpu_id} invalid NVIDIA {selector} row: {raw!r}"
                )
            rows.append(row)
    except csv.Error as exc:
        raise RuntimeError(f"GPU {gpu_id} malformed NVIDIA {selector} CSV: {exc}") from exc
    return rows


def _integer(value: str, name: str, *, positive: bool = False) -> int:
    if not _INTEGER.fullmatch(value):
        raise RuntimeError(f"invalid {name}: {value!r}")
    number = int(value)
    if positive and number == 0:
        raise RuntimeError(f"invalid non-positive {name}: {value!r}")
    return number


def _nonnegative_float(value: str, name: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid {name}: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"invalid non-finite/negative {name}: {value!r}")
    return number


def _unavailable(error: str) -> dict[str, Any]:
    return {"status": "unavailable", "inventory": None, "compute": None, "error": error}


def _probe_one(gpu_id: int) -> dict[str, Any]:
    try:
        rows = query_gpu_csv(gpu_id, "gpu", INVENTORY_FIELDS, QUERY_TIMEOUT_SECONDS)
        if len(rows) != 1:
            raise RuntimeError(f"expected exactly one GPU inventory row, got {len(rows)}")
        values = rows[0]
        index = _integer(values[0], "GPU index")
        if index != gpu_id:
            raise RuntimeError(f"GPU index mismatch: requested {gpu_id}, observed {index}")
        uuid = values[1]
        if not _GPU_UUID.fullmatch(uuid):
            raise RuntimeError(f"invalid physical GPU UUID: {uuid!r}")
        if values[2].lower() in {"n/a", "[n/a]", "unknown", "[not supported]"}:
            raise RuntimeError(f"invalid GPU name: {values[2]!r}")
        total = _integer(values[3], "memory.total", positive=True)
        used = _integer(values[4], "memory.used")
        temperature = _nonnegative_float(values[5], "temperature.gpu")
        utilization = _nonnegative_float(values[6], "utilization.gpu")
        if used > total or utilization > 100:
            raise RuntimeError("GPU memory/utilization exceeds its physical range")
        inventory = {
            "index": index,
            "uuid": uuid,
            "name": values[2],
            "memory_total_mib": total,
            "memory_used_mib": used,
            "temperature_c": temperature,
            "utilization_gpu": utilization,
        }
        compute: list[dict[str, Any]] = []
        pids: set[int] = set()
        for row in query_gpu_csv(gpu_id, "compute-apps", COMPUTE_FIELDS, QUERY_TIMEOUT_SECONDS):
            if row[0] != uuid:
                raise RuntimeError(f"GPU compute UUID mismatch: expected {uuid}, got {row[0]}")
            pid = _integer(row[1], "compute PID", positive=True)
            if pid in pids:
                raise RuntimeError(f"duplicate compute PID on GPU {gpu_id}: {pid}")
            pids.add(pid)
            memory = _integer(row[3], "compute used_gpu_memory")
            if memory > total:
                raise RuntimeError("compute memory exceeds physical GPU memory")
            compute.append(
                {
                    "gpu_uuid": row[0],
                    "pid": pid,
                    "process_name": row[2],
                    "used_gpu_memory_mib": memory,
                }
            )
        return {"status": "ok", "inventory": inventory, "compute": compute, "error": None}
    except RuntimeError as exc:
        return _unavailable(f"GPU {gpu_id}: {exc}")


def probe_gpus(gpu_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Probe cards independently, with at most four concurrent 2-second queries.

    Each card needs at most two sequential queries (inventory, then compute).
    Four cards therefore have a query-time budget of four seconds, not sixteen.
    Callers must not treat ``unavailable`` or ``compute=None`` as an idle card.
    """
    ids: list[int] = []
    for gpu_id in gpu_ids:
        _validate_gpu_id(gpu_id)
        if gpu_id not in ids:
            ids.append(gpu_id)
    if not ids:
        return {}
    with ThreadPoolExecutor(max_workers=min(4, len(ids))) as executor:
        results = dict(zip(ids, executor.map(_probe_one, ids)))
    by_uuid: dict[str, list[int]] = {}
    for gpu_id, result in results.items():
        if result["status"] == "ok":
            by_uuid.setdefault(result["inventory"]["uuid"], []).append(gpu_id)
    for uuid, duplicate_ids in by_uuid.items():
        if len(duplicate_ids) > 1:
            for gpu_id in duplicate_ids:
                results[gpu_id] = _unavailable(
                    f"physical GPU UUID {uuid} is duplicated across indexes {duplicate_ids}"
                )
    return results
