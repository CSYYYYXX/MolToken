#!/usr/bin/env python3
"""Run the domain-wise supplement queue with persistent atomic reservations.

The original queue uses ``flock`` for cross-worker ownership.  That lock is
not globally exclusive on the cluster's shared mount.  This entry point keeps
the original queue, plan validation, profiling, training, and admission logic,
but replaces both run claims and allocation ownership with atomic directory
creation.  Reservation directories are deliberately never removed by this
process: a completed run is released semantically by its validated completion
marker, while a failed or interrupted owner remains visible for explicit
operator recovery.

Do not run this entry point concurrently with the legacy queue on the same
allocation or with any overlapping candidate.  The two ownership mechanisms
cannot coordinate.  Operator-approved queues may coexist on different
allocations only when their candidate sets are completely disjoint.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from . import run_domainwise_supplement_queue as legacy
except ImportError:  # direct script execution
    import run_domainwise_supplement_queue as legacy


RESERVATION_SCHEMA = "domainwise_supplement_atomic_reservation_v1"
RUN_RESERVATION_SUFFIX = ".atomic_reservation"
ALLOCATION_RESERVATION_SUFFIX = ".atomic_reservation"


def _write_owner(path: Path, payload: dict[str, Any]) -> None:
    """Create owner metadata once; never replace another owner's record."""

    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reserve_directory(path: Path, owner: dict[str, Any]) -> bool:
    """Atomically reserve ``path``; return False when another owner won."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"atomic reservation path is not a directory: {path}")
        return False

    payload = {
        "schema_version": RESERVATION_SCHEMA,
        "status": "reserved",
        "reservation_path": str(path),
        **owner,
    }
    try:
        _write_owner(path / "owner.json", payload)
    except Exception as exc:
        # The directory is the ownership primitive.  Keep it after a metadata
        # failure so another worker cannot take over an ambiguous reservation.
        raise RuntimeError(
            f"atomic reservation was created but owner metadata failed; "
            f"manual inspection is required: {path}"
        ) from exc
    return True


def run_reservation_path(claims_dir: Path, candidate: legacy.Candidate) -> Path:
    row = candidate.row
    return claims_dir / (
        f"{row['run_id']}.{row['attempt_name']}{RUN_RESERVATION_SUFFIX}"
    )


@dataclass
class AtomicRunReservation:
    """Duck-typed replacement for the legacy descriptor-backed Claim."""

    candidate: legacy.Candidate
    path: Path
    formal_thread: Any = None
    formal_outcome: legacy.FormalOutcome | None = None

    def close(self) -> None:
        # Persistence is intentional.  Success is recognized from complete.json;
        # failure/crash requires explicit recovery and must not become stealable.
        return None


def try_reserve_run(
    candidate: legacy.Candidate,
    claims_dir: Path,
    metadata: dict[str, Any],
) -> AtomicRunReservation | None:
    path = run_reservation_path(claims_dir, candidate)
    row = candidate.row
    acquired = _reserve_directory(
        path,
        {
            "reservation_kind": "supplement_run",
            "reserved_at": legacy.now(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "run_id": row["run_id"],
            "attempt_number": row["attempt_number"],
            "attempt_name": row["attempt_name"],
            "attempt_dir": row["attempt_dir"],
            "plan_path": str(candidate.plan_path),
            **metadata,
        },
    )
    if not acquired:
        return None
    return AtomicRunReservation(candidate=candidate, path=path)


@dataclass
class AtomicAllocationReservation:
    path: Path

    def close(self) -> None:
        # A queue has no natural successful terminal state: it watches for new
        # plans indefinitely.  Preserve ownership on every exit so restart is
        # always an explicit, audited operator action.
        return None


def acquire_atomic_allocation_reservation(
    status_dir: Path, global_gpu_offset: int
) -> AtomicAllocationReservation:
    path = status_dir / (
        f"supplement_queue_g{global_gpu_offset}{ALLOCATION_RESERVATION_SUFFIX}"
    )
    acquired = _reserve_directory(
        path,
        {
            "reservation_kind": "supplement_allocation",
            "reserved_at": legacy.now(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "global_gpu_offset": global_gpu_offset,
        },
    )
    if not acquired:
        raise RuntimeError(
            f"atomic allocation reservation already exists for global GPU "
            f"offset {global_gpu_offset}: {path}"
        )
    return AtomicAllocationReservation(path=path)


class AtomicSupplementQueue(legacy.SupplementQueue):
    """Legacy scheduler with only its cross-worker claim primitive replaced."""

    def _claim_next(self, lane: legacy.BaselineLane) -> AtomicRunReservation | None:
        busy: list[str] = []
        busy_paths: list[str] = []
        for candidate in self.refresh_candidates():
            if self._candidate_complete(candidate):
                continue
            reservation = try_reserve_run(
                candidate,
                self.claims_dir,
                {
                    "device_index": lane.device_index,
                    "global_gpu_index": lane.global_gpu_index,
                    "baseline_lane": lane.lane_within_gpu,
                },
            )
            if reservation is None:
                busy.append(str(candidate.row["run_id"]))
                busy_paths.append(str(run_reservation_path(self.claims_dir, candidate)))
                continue
            if self._candidate_complete(candidate):
                # Keep the reservation as durable provenance.  The validated
                # completion marker is the semantic release signal.
                reservation.close()
                continue
            attempt_dir = Path(str(candidate.row["attempt_dir"])).expanduser().resolve()
            if attempt_dir.exists():
                reservation.close()
                raise FileExistsError(
                    f"incomplete supplement attempt already exists after atomic "
                    f"reservation: {attempt_dir}; reservation retained at "
                    f"{reservation.path}"
                )
            if busy:
                self.event(
                    "claim_contention_continued",
                    claim_scheme="atomic_directory_v1",
                    lane=lane.label,
                    busy_run_ids=busy,
                    busy_reservation_paths=busy_paths,
                    claimed_run_id=candidate.row["run_id"],
                    reservation_path=str(reservation.path),
                )
            return reservation
        if busy:
            self.event(
                "claim_contention_no_candidate",
                claim_scheme="atomic_directory_v1",
                lane=lane.label,
                busy_run_ids=busy,
                busy_reservation_paths=busy_paths,
            )
        return None


def main(argv: list[str] | None = None) -> int:
    """Reuse the legacy CLI/bootstrap while injecting atomic ownership."""

    original_queue = legacy.SupplementQueue
    original_allocation_lock = legacy.acquire_allocation_lock
    legacy.SupplementQueue = AtomicSupplementQueue
    legacy.acquire_allocation_lock = acquire_atomic_allocation_reservation
    try:
        return legacy.main(argv)
    finally:
        legacy.SupplementQueue = original_queue
        legacy.acquire_allocation_lock = original_allocation_lock


if __name__ == "__main__":
    raise SystemExit(main())
