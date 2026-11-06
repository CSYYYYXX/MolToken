"""Transparent exact-one-epoch sampling for each Axis-B SFT stage.

Unlike the historical ``natural-one-pass`` protocol, this protocol does not
partition one permutation between alignment and full.  A caller constructs a
fresh sampler for each stage and each sampler consumes the complete selected
training view exactly once.

The sampler works for both a single task and a mixed task view.  Every
microbatch is task homogeneous.  When gradient accumulation is used,
microbatches are arranged in same-task groups so one optimizer update never
mixes tasks.  Distributed ranks receive disjoint slices of the same global
microbatch; records are never padded, repeated, or dropped.

Auditing is deliberately semantic.  Counts, duplicates, omissions, task
membership, and real tail sizes are recorded directly; file or order hashes
are not acceptance gates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
import math
import random
from typing import Any


PROTOCOL = "stagewise-full-epoch-v1"


def _rank_slice(global_size: int, rank: int, world_size: int) -> tuple[int, int]:
    quotient, remainder = divmod(int(global_size), int(world_size))
    length = quotient + (1 if int(rank) < remainder else 0)
    start = int(rank) * quotient + min(int(rank), remainder)
    if length <= 0:
        raise ValueError(
            f"global microbatch {global_size} leaves rank {rank}/{world_size} empty"
        )
    return start, length


def distributed_global_batch_sizes(
    records: int, per_rank_batch_size: int, world_size: int
) -> tuple[int, ...]:
    """Return an exact, non-empty DDP partition with a real short tail."""

    records = int(records)
    per_rank_batch_size = int(per_rank_batch_size)
    world_size = int(world_size)
    if records <= 0 or per_rank_batch_size <= 0 or world_size <= 0:
        raise ValueError("records, per-rank batch size, and world size must be positive")
    if records < world_size:
        raise ValueError(
            f"task has {records} records and cannot keep {world_size} DDP ranks non-empty"
        )

    capacity = per_rank_batch_size * world_size
    steps = math.ceil(records / capacity)
    sizes = [capacity] * max(0, steps - 1)
    sizes.append(records - capacity * (steps - 1))
    if len(sizes) > 1 and sizes[-1] < world_size:
        transfer = world_size - sizes[-1]
        sizes[-2] -= transfer
        sizes[-1] += transfer
    if sum(sizes) != records or any(
        size < world_size or size > capacity for size in sizes
    ):
        raise RuntimeError(
            "cannot build exact distributed batches: "
            f"records={records} per_rank_batch={per_rank_batch_size} "
            f"world_size={world_size} sizes={sizes}"
        )
    return tuple(sizes)


def _split_for_accumulation(
    batches: list[list[int]], *, group_size: int, world_size: int
) -> list[list[int]]:
    """Split existing batches until their count is a multiple of group size."""

    target = math.ceil(len(batches) / group_size) * group_size
    while len(batches) < target:
        candidates = [
            (len(batch), index)
            for index, batch in enumerate(batches)
            if len(batch) >= 2 * world_size
        ]
        if not candidates:
            raise ValueError(
                "not enough records to form task-homogeneous gradient-accumulation "
                f"groups without duplication: batches={len(batches)} target={target} "
                f"world_size={world_size}"
            )
        _size, index = max(candidates)
        batch = batches[index]
        midpoint = len(batch) // 2
        midpoint = max(world_size, min(midpoint, len(batch) - world_size))
        batches[index : index + 1] = [batch[:midpoint], batch[midpoint:]]
    return batches


def stage_microsteps(
    dataset: Any,
    batch_sizes: Mapping[str, int],
    world_size: int,
    accumulation_steps: int,
) -> int:
    """Compute the exact number of microsteps without reading JSONL payloads."""

    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    task_indices = dataset.task_indices()
    total = 0
    for task in sorted(task_indices):
        count = len(task_indices[task])
        if count <= 0:
            continue
        base = len(
            distributed_global_batch_sizes(
                count, int(batch_sizes[task]), int(world_size)
            )
        )
        total += math.ceil(base / accumulation_steps) * accumulation_steps
    if total <= 0:
        raise ValueError("selected training view is empty")
    return total


class StagewiseFullEpochBatchSampler:
    """One complete, deterministic stage pass over a single or mixed view."""

    def __init__(
        self,
        dataset: Any,
        *,
        batch_sizes: Mapping[str, int],
        steps_per_epoch: int,
        seed: int,
        epoch: int,
        rank: int,
        world_size: int,
        task_group_size: int,
    ) -> None:
        self.dataset = dataset
        self.batch_sizes = {str(key): int(value) for key, value in batch_sizes.items()}
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.task_group_size = int(task_group_size)
        if self.epoch != 0:
            raise ValueError("stagewise full-epoch sampling supports exactly epoch 0")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank/world_size out of range: {self.rank}/{self.world_size}")
        if self.task_group_size <= 0:
            raise ValueError("task_group_size must be positive")

        raw_task_indices = dataset.task_indices()
        self.task_indices = {
            str(task): [int(value) for value in values]
            for task, values in raw_task_indices.items()
            if values
        }
        if set(self.task_indices) != set(self.batch_sizes):
            raise ValueError(
                "task/batch-size mismatch: "
                f"tasks={sorted(self.task_indices)} batch_sizes={sorted(self.batch_sizes)}"
            )

        groups: list[tuple[str, list[list[int]]]] = []
        self.expected_task_records: dict[str, int] = {}
        self.microsteps_by_task: dict[str, int] = {}
        selected: list[int] = []
        for task in sorted(self.task_indices):
            order = list(self.task_indices[task])
            random.Random(f"{self.seed}:{task}").shuffle(order)
            sizes = distributed_global_batch_sizes(
                len(order), self.batch_sizes[task], self.world_size
            )
            cursor = 0
            task_batches: list[list[int]] = []
            for size in sizes:
                task_batches.append(order[cursor : cursor + size])
                cursor += size
            task_batches = _split_for_accumulation(
                task_batches,
                group_size=self.task_group_size,
                world_size=self.world_size,
            )
            for start in range(0, len(task_batches), self.task_group_size):
                groups.append(
                    (task, task_batches[start : start + self.task_group_size])
                )
            self.expected_task_records[task] = len(order)
            self.microsteps_by_task[task] = len(task_batches)
            selected.extend(order)

        random.Random(f"{self.seed}:task-groups").shuffle(groups)
        self.global_batches: list[tuple[str, list[int]]] = [
            (task, batch)
            for task, task_batches in groups
            for batch in task_batches
        ]
        if len(self.global_batches) != int(steps_per_epoch):
            raise ValueError(
                "trainer/sampler microstep mismatch: "
                f"trainer={steps_per_epoch} sampler={len(self.global_batches)}"
            )
        selected_unique = len(set(selected))
        if len(selected) != len(dataset) or selected_unique != len(selected):
            raise ValueError(
                "selected task views are not one disjoint exhaustive dataset pass: "
                f"selected={len(selected)} unique={selected_unique} dataset={len(dataset)}"
            )
        self.selected_indices = tuple(selected)
        self.expected_local_records_by_rank = tuple(
            sum(
                _rank_slice(len(batch), candidate, self.world_size)[1]
                for _task, batch in self.global_batches
            )
            for candidate in range(self.world_size)
        )
        self.final_local_batch_sizes = tuple(
            _rank_slice(len(self.global_batches[-1][1]), candidate, self.world_size)[1]
            for candidate in range(self.world_size)
        )
        expected_local_indices: list[int] = []
        expected_local_tasks: list[str] = []
        for task, global_batch in self.global_batches:
            local_start, local_size = _rank_slice(
                len(global_batch), self.rank, self.world_size
            )
            expected_local_indices.extend(
                global_batch[local_start : local_start + local_size]
            )
            expected_local_tasks.extend([task] * local_size)
        self.expected_local_indices = tuple(expected_local_indices)
        self.expected_local_tasks = tuple(expected_local_tasks)

    def __len__(self) -> int:
        return len(self.global_batches)

    def __iter__(self) -> Iterator[list[int]]:
        for _task, global_batch in self.global_batches:
            local_start, local_size = _rank_slice(
                len(global_batch), self.rank, self.world_size
            )
            local_batch = global_batch[local_start : local_start + local_size]
            yield local_batch

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != 0:
            raise ValueError("stagewise full-epoch sampling supports exactly epoch 0")

    def plan_audit(self, *, stage: str, accumulation_steps: int) -> dict[str, Any]:
        """Describe the expected stage pass without claiming runtime consumption."""

        return {
            "protocol": PROTOCOL,
            "semantics": "this stage consumes the complete selected train view once",
            "stage": str(stage),
            "dataset_records": len(self.selected_indices),
            "selected_records": len(self.selected_indices),
            "expected_task_records": dict(self.expected_task_records),
            "batch_sizes_per_rank": dict(self.batch_sizes),
            "world_size": self.world_size,
            "gradient_accumulation_steps": int(accumulation_steps),
            "task_homogeneous_accumulation": True,
            "expected_microsteps": len(self.global_batches),
            "expected_optimizer_steps": math.ceil(
                len(self.global_batches) / int(accumulation_steps)
            ),
            "microsteps_by_task": dict(self.microsteps_by_task),
            "expected_local_records_by_rank": list(
                self.expected_local_records_by_rank
            ),
            "expected_final_global_batch_size": len(self.global_batches[-1][1]),
            "expected_final_local_batch_sizes": list(self.final_local_batch_sizes),
            "no_padding_duplication_cycling_or_oversampling": True,
        }

    def local_forward_audit(
        self,
        *,
        actual_indices: Sequence[int],
        actual_tasks: Sequence[str],
        actual_microsteps: int,
        actual_final_batch_size: int,
    ) -> dict[str, Any]:
        """Compare records that reached forward with this rank's exact plan."""

        actual = tuple(int(value) for value in actual_indices)
        tasks = tuple(str(value) for value in actual_tasks)
        actual_set = set(actual)
        expected_set = set(self.expected_local_indices)
        duplicates = len(actual) - len(actual_set)
        omitted = len(expected_set.difference(actual_set))
        unexpected = len(actual_set.difference(expected_set))
        expected_task_records = dict(Counter(self.expected_local_tasks))
        actual_task_records = dict(Counter(tasks))
        complete = (
            actual == self.expected_local_indices
            and tasks == self.expected_local_tasks
            and int(actual_microsteps) == len(self.global_batches)
            and int(actual_final_batch_size) == self.final_local_batch_sizes[self.rank]
            and duplicates == 0
            and omitted == 0
            and unexpected == 0
            and actual_task_records == expected_task_records
        )
        return {
            "local_complete": complete,
            "expected_local_records": len(self.expected_local_indices),
            "actual_local_records": len(actual),
            "expected_local_task_records": expected_task_records,
            "actual_local_task_records": actual_task_records,
            "actual_local_microsteps": int(actual_microsteps),
            "actual_local_final_batch_size": int(actual_final_batch_size),
            "actual_local_duplicate_records": duplicates,
            "actual_local_omitted_records": omitted,
            "actual_local_unexpected_records": unexpected,
            "actual_local_order_matches_plan": actual == self.expected_local_indices,
        }


__all__ = [
    "PROTOCOL",
    "StagewiseFullEpochBatchSampler",
    "distributed_global_batch_sizes",
    "stage_microsteps",
]
