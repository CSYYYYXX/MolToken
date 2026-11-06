"""Exact, deterministic single-pass sampling for Axis-B ablations.

This module is intentionally separate from ``HomogeneousTaskBatchSampler``.
The latter implements a fixed-step, task-balanced training contract and may
cycle a task many times.  Here the contract is the opposite: a single-task,
single-rank run must consume every selected dataset position exactly once,
including a short final batch.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Iterator, Sequence


PROTOCOL = "single-task-natural-one-pass-v1"


def deterministic_partition(
    length: int,
    *,
    seed: int,
    start_part: int = 0,
    end_part: int = 1,
    parts: int = 1,
) -> tuple[list[int], dict[str, int | str]]:
    """Return one contiguous slice of a deterministic permutation.

    Integer partition boundaries make adjacent stage slices exhaustive and
    disjoint even when ``length`` is not divisible by ``parts``.
    """

    length = int(length)
    seed = int(seed)
    start_part = int(start_part)
    end_part = int(end_part)
    parts = int(parts)
    if length <= 0:
        raise ValueError("natural one-pass sampling requires a non-empty dataset")
    if parts <= 0:
        raise ValueError("parts must be positive")
    if not 0 <= start_part < end_part <= parts:
        raise ValueError(
            "partition must satisfy 0 <= start_part < end_part <= parts; "
            f"got {start_part}:{end_part}/{parts}"
        )

    order = list(range(length))
    random.Random(seed).shuffle(order)
    start = length * start_part // parts
    end = length * end_part // parts
    selected = order[start:end]

    def digest(values: Sequence[int]) -> str:
        payload = json.dumps(list(values), separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    metadata: dict[str, int | str] = {
        "protocol": PROTOCOL,
        "dataset_records": length,
        "seed": seed,
        "partition_start_part": start_part,
        "partition_end_part": end_part,
        "partition_parts": parts,
        "partition_start_index": start,
        "partition_end_index": end,
        "selected_records": len(selected),
        "full_order_sha256": digest(order),
        "selected_order_sha256": digest(selected),
    }
    return selected, metadata


class SingleTaskNaturalEpochBatchSampler:
    """Yield a fixed list of positions once, with an optional short tail."""

    def __init__(self, indices: Sequence[int], batch_size: int) -> None:
        self.indices = [int(value) for value in indices]
        self.batch_size = int(batch_size)
        if not self.indices:
            raise ValueError("natural epoch partition cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("natural epoch indices contain duplicates")
        self._yielded_indices: list[int] = []

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        self._yielded_indices = []
        for start in range(0, len(self.indices), self.batch_size):
            batch = self.indices[start : start + self.batch_size]
            self._yielded_indices.extend(batch)
            yield batch

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != 0:
            raise ValueError("natural one-pass sampler supports exactly epoch 0")

    @property
    def selected_records(self) -> int:
        return len(self.indices)

    @property
    def final_batch_size(self) -> int:
        remainder = len(self.indices) % self.batch_size
        return remainder if remainder else self.batch_size

    def flattened(self) -> Iterable[int]:
        for batch in self:
            yield from batch

    @property
    def yielded_indices(self) -> tuple[int, ...]:
        return tuple(self._yielded_indices)

    @property
    def yielded_order_sha256(self) -> str:
        payload = json.dumps(self._yielded_indices, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def _rank_slice(global_size: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return one non-overlapping, near-even rank slice of a global batch."""

    quotient, remainder = divmod(int(global_size), int(world_size))
    length = quotient + (1 if int(rank) < remainder else 0)
    start = int(rank) * quotient + min(int(rank), remainder)
    if length <= 0:
        raise ValueError(
            f"global batch {global_size} leaves rank {rank}/{world_size} empty"
        )
    return start, length


def distributed_global_batch_sizes(
    selected_records: int,
    per_rank_batch_size: int,
    world_size: int,
) -> tuple[int, ...]:
    """Partition an exact pass into synchronized, non-empty DDP microbatches."""

    records = int(selected_records)
    local_capacity = int(per_rank_batch_size)
    ranks = int(world_size)
    if records <= 0 or local_capacity <= 0 or ranks <= 0:
        raise ValueError("records, per-rank batch size, and world size must be positive")
    if records < ranks:
        raise ValueError(
            f"{records} selected records cannot keep {ranks} DDP ranks non-empty"
        )

    global_capacity = local_capacity * ranks
    steps = math.ceil(records / global_capacity)
    sizes = [global_capacity] * max(0, steps - 1)
    sizes.append(records - global_capacity * (steps - 1))
    if len(sizes) > 1 and sizes[-1] < ranks:
        transfer = ranks - sizes[-1]
        sizes[-2] -= transfer
        sizes[-1] += transfer
    if sum(sizes) != records or any(
        size < ranks or size > global_capacity for size in sizes
    ):
        raise RuntimeError(
            "cannot form exact synchronized DDP batches: "
            f"records={records} local_capacity={local_capacity} ranks={ranks} "
            f"tail={sizes[-2:]}"
        )
    return tuple(int(size) for size in sizes)


class DistributedSingleTaskNaturalEpochBatchSampler:
    """Rank-local slices of one global deterministic single-task pass.

    Every rank holds the same global order but yields only its disjoint slice.
    No padding, duplication, cycling, or record omission is used, including in
    the final short global batch.
    """

    def __init__(
        self,
        indices: Sequence[int],
        per_rank_batch_size: int,
        *,
        rank: int,
        world_size: int,
    ) -> None:
        self.indices = [int(value) for value in indices]
        self.batch_size = int(per_rank_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not self.indices:
            raise ValueError("natural epoch partition cannot be empty")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("natural epoch indices contain duplicates")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0,{self.world_size}), got {self.rank}"
            )
        self.global_batch_sizes = distributed_global_batch_sizes(
            len(self.indices), self.batch_size, self.world_size
        )
        self.final_local_batch_sizes = tuple(
            _rank_slice(self.global_batch_sizes[-1], candidate, self.world_size)[1]
            for candidate in range(self.world_size)
        )
        self.expected_local_records_by_rank = tuple(
            sum(
                _rank_slice(global_size, candidate, self.world_size)[1]
                for global_size in self.global_batch_sizes
            )
            for candidate in range(self.world_size)
        )
        self._yielded_local_indices: list[int] = []
        self._yielded_global_indices: list[int] = []

    def __len__(self) -> int:
        return len(self.global_batch_sizes)

    def __iter__(self) -> Iterator[list[int]]:
        self._yielded_local_indices = []
        self._yielded_global_indices = []
        cursor = 0
        for global_size in self.global_batch_sizes:
            global_values = self.indices[cursor : cursor + global_size]
            if len(global_values) != global_size:
                raise RuntimeError("distributed natural sampler read a short global batch")
            local_start, local_size = _rank_slice(
                global_size, self.rank, self.world_size
            )
            local_values = global_values[local_start : local_start + local_size]
            self._yielded_global_indices.extend(global_values)
            self._yielded_local_indices.extend(local_values)
            cursor += global_size
            yield local_values
        if cursor != len(self.indices):
            raise RuntimeError(
                f"distributed natural sampler omitted records: {cursor}/{len(self.indices)}"
            )

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != 0:
            raise ValueError("natural one-pass sampler supports exactly epoch 0")

    @property
    def selected_records(self) -> int:
        return len(self.indices)

    @property
    def final_batch_size(self) -> int:
        return int(self.final_local_batch_sizes[self.rank])

    @property
    def final_global_batch_size(self) -> int:
        return int(self.global_batch_sizes[-1])

    def final_optimizer_batch_records(self, accumulation_steps: int) -> int:
        accumulation = int(accumulation_steps)
        if accumulation <= 0:
            raise ValueError("accumulation_steps must be positive")
        tail_microsteps = len(self) % accumulation or accumulation
        return int(sum(self.global_batch_sizes[-tail_microsteps:]))

    @property
    def yielded_indices(self) -> tuple[int, ...]:
        """Global yielded order, used for an exact-once identity audit."""

        return tuple(self._yielded_global_indices)

    @property
    def yielded_local_indices(self) -> tuple[int, ...]:
        return tuple(self._yielded_local_indices)

    @property
    def yielded_order_sha256(self) -> str:
        payload = json.dumps(
            self._yielded_global_indices, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()
