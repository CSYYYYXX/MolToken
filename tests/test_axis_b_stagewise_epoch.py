#!/usr/bin/env python3
"""Pure synthetic tests for stagewise full-epoch sampling."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "pipeline/finetune/axis_b_stagewise_epoch.py"
)
SPEC = importlib.util.spec_from_file_location("axis_b_stagewise_epoch_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load sampler module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PROTOCOL = MODULE.PROTOCOL
StagewiseFullEpochBatchSampler = MODULE.StagewiseFullEpochBatchSampler
distributed_global_batch_sizes = MODULE.distributed_global_batch_sizes
stage_microsteps = MODULE.stage_microsteps


class FakeDataset:
    def __init__(self, task_counts: dict[str, int]) -> None:
        cursor = 0
        self._tasks: dict[str, list[int]] = {}
        for task, count in task_counts.items():
            self._tasks[task] = list(range(cursor, cursor + count))
            cursor += count
        self.length = cursor

    def __len__(self) -> int:
        return self.length

    def task_indices(self) -> dict[str, list[int]]:
        return {task: list(values) for task, values in self._tasks.items()}


class StagewiseEpochTests(unittest.TestCase):
    def test_real_ddp_tail_has_no_padding(self) -> None:
        self.assertEqual(distributed_global_batch_sizes(33, 8, 2), (16, 15, 2))
        self.assertEqual(sum(distributed_global_batch_sizes(33, 8, 2)), 33)

    def test_single_task_each_stage_is_one_complete_epoch(self) -> None:
        dataset = FakeDataset({"protein_generation": 33})
        steps = stage_microsteps(dataset, {"protein_generation": 8}, 2, 4)
        self.assertEqual(steps, 4)
        for stage in ("alignment", "full"):
            samplers = [
                StagewiseFullEpochBatchSampler(
                    dataset,
                    batch_sizes={"protein_generation": 8},
                    steps_per_epoch=steps,
                    seed=42,
                    epoch=0,
                    rank=rank,
                    world_size=2,
                    task_group_size=4,
                )
                for rank in range(2)
            ]
            rank_batches = [list(sampler) for sampler in samplers]
            rank_records = [
                [value for batch in batches for value in batch]
                for batches in rank_batches
            ]
            self.assertFalse(set(rank_records[0]) & set(rank_records[1]))
            self.assertEqual(
                set(rank_records[0]) | set(rank_records[1]), set(range(33))
            )
            for rank, sampler in enumerate(samplers):
                local = sampler.local_forward_audit(
                    actual_indices=rank_records[rank],
                    actual_tasks=sampler.expected_local_tasks,
                    actual_microsteps=len(rank_batches[rank]),
                    actual_final_batch_size=len(rank_batches[rank][-1]),
                )
                self.assertTrue(local["local_complete"])
                self.assertEqual(
                    local["actual_local_records"],
                    sampler.expected_local_records_by_rank[rank],
                )

    def test_mixed_batches_and_optimizer_groups_are_task_homogeneous(self) -> None:
        dataset = FakeDataset({"task_a": 37, "task_b": 29, "task_c": 21})
        batch_sizes = {"task_a": 5, "task_b": 4, "task_c": 3}
        accumulation = 3
        world_size = 2
        steps = stage_microsteps(
            dataset, batch_sizes, world_size, accumulation
        )
        sampler = StagewiseFullEpochBatchSampler(
            dataset,
            batch_sizes=batch_sizes,
            steps_per_epoch=steps,
            seed=17,
            epoch=0,
            rank=0,
            world_size=world_size,
            task_group_size=accumulation,
        )
        labels = [task for task, _batch in sampler.global_batches]
        for start in range(0, len(labels), accumulation):
            self.assertEqual(len(set(labels[start : start + accumulation])), 1)
        batches = list(sampler)
        actual_indices = [value for batch in batches for value in batch]
        local = sampler.local_forward_audit(
            actual_indices=actual_indices,
            actual_tasks=sampler.expected_local_tasks,
            actual_microsteps=len(batches),
            actual_final_batch_size=len(batches[-1]),
        )
        plan = sampler.plan_audit(
            stage="alignment", accumulation_steps=accumulation
        )
        self.assertTrue(local["local_complete"])
        self.assertEqual(plan["protocol"], PROTOCOL)
        self.assertEqual(
            plan["expected_task_records"],
            {task: len(values) for task, values in dataset.task_indices().items()},
        )

    def test_one_gpu_acc4_matches_four_gpu_acc1_without_repeat_or_omit(self) -> None:
        dataset = FakeDataset({"task_a": 101, "task_b": 67})
        batch_sizes = {"task_a": 8, "task_b": 5}
        geometries = ((4, 1), (1, 4))
        optimizer_steps = []
        for world_size, accumulation in geometries:
            steps = stage_microsteps(
                dataset, batch_sizes, world_size, accumulation
            )
            sampler = StagewiseFullEpochBatchSampler(
                dataset,
                batch_sizes=batch_sizes,
                steps_per_epoch=steps,
                seed=19,
                epoch=0,
                rank=0,
                world_size=world_size,
                task_group_size=accumulation,
            )
            plan = sampler.plan_audit(
                stage="alignment", accumulation_steps=accumulation
            )
            optimizer_steps.append(plan["expected_optimizer_steps"])
            labels = [task for task, _batch in sampler.global_batches]
            for start in range(0, len(labels), accumulation):
                self.assertEqual(
                    len(set(labels[start : start + accumulation])), 1
                )
            flattened = [
                value
                for _task, batch in sampler.global_batches
                for value in batch
            ]
            self.assertEqual(len(flattened), len(dataset))
            self.assertEqual(len(set(flattened)), len(dataset))
        self.assertEqual(optimizer_steps, [8, 8])

        one_gpu_steps = stage_microsteps(dataset, batch_sizes, 1, 4)
        one_gpu = StagewiseFullEpochBatchSampler(
            dataset,
            batch_sizes=batch_sizes,
            steps_per_epoch=one_gpu_steps,
            seed=19,
            epoch=0,
            rank=0,
            world_size=1,
            task_group_size=4,
        )
        self.assertTrue(
            any(
                len(batch) < batch_sizes[task]
                for task, batch in one_gpu.global_batches
            )
        )

    def test_audit_contains_no_hash_acceptance_fields(self) -> None:
        dataset = FakeDataset({"task": 20})
        steps = stage_microsteps(dataset, {"task": 4}, 1, 1)
        sampler = StagewiseFullEpochBatchSampler(
            dataset,
            batch_sizes={"task": 4},
            steps_per_epoch=steps,
            seed=42,
            epoch=0,
            rank=0,
            world_size=1,
            task_group_size=1,
        )
        batches = list(sampler)
        local = sampler.local_forward_audit(
            actual_indices=[value for batch in batches for value in batch],
            actual_tasks=sampler.expected_local_tasks,
            actual_microsteps=len(batches),
            actual_final_batch_size=len(batches[-1]),
        )
        audit = sampler.plan_audit(stage="full", accumulation_steps=1) | local
        self.assertNotIn("sha", json.dumps(audit).lower())

    def test_prefetched_sampler_yields_do_not_count_as_forward_consumption(self) -> None:
        dataset = FakeDataset({"task": 20})
        steps = stage_microsteps(dataset, {"task": 4}, 1, 1)
        sampler = StagewiseFullEpochBatchSampler(
            dataset,
            batch_sizes={"task": 4},
            steps_per_epoch=steps,
            seed=42,
            epoch=0,
            rank=0,
            world_size=1,
            task_group_size=1,
        )
        prefetched_batches = list(sampler)
        forwarded = sampler.expected_local_indices[:-1]
        forwarded_tasks = sampler.expected_local_tasks[:-1]
        local = sampler.local_forward_audit(
            actual_indices=forwarded,
            actual_tasks=forwarded_tasks,
            actual_microsteps=len(prefetched_batches),
            actual_final_batch_size=len(prefetched_batches[-1]) - 1,
        )
        self.assertFalse(local["local_complete"])
        self.assertEqual(local["actual_local_omitted_records"], 1)

if __name__ == "__main__":
    unittest.main()
