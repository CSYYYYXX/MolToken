#!/usr/bin/env python3
"""Focused synthetic tests for the Axis-B SFT data layer.

These tests never touch the formal multi-gigabyte corpus.  The index tests use
small JSONL files with a formal-looking manifest; the collator tests use a
minimal tokenizer stub.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PIPELINE_ROOT / "pipeline" / "finetune" / "axis_b_data.py"
SPEC = importlib.util.spec_from_file_location("axis_b_data_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {MODULE_PATH}")
AXIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AXIS
SPEC.loader.exec_module(AXIS)

try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except ImportError:  # pragma: no cover - jump-host fallback
    HAS_NUMPY = False


class TinyTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, texts, **kwargs):
        self.calls.append(dict(kwargs))
        assert kwargs["truncation"] is True
        assert kwargs["padding"] is True
        assert kwargs["max_length"] == 384
        rows = []
        masks = []
        for text in texts:
            # A deterministic, deliberately tiny stand-in for a HF tokenizer.
            words = str(text).split()
            row = [101] + [200 + i for i in range(len(words))]
            rows.append(row)
            masks.append([1] * len(row))
        width = max(len(row) for row in rows)
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in rows],
            "attention_mask": [mask + [0] * (width - len(mask)) for mask in masks],
        }


class PrefixTruncatingTokenizer:
    """Tiny HF-like tokenizer that makes prefix truncation observable."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(dict(kwargs))
        max_length = int(kwargs["max_length"])
        rows = []
        for text in texts:
            words = str(text).split()
            full = [101] + [200 + index for index in range(len(words))] + [102]
            rows.append(full[:max_length])
        width = max(len(row) for row in rows)
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in rows],
            "attention_mask": [
                [1] * len(row) + [0] * (width - len(row)) for row in rows
            ],
        }


class StrictLengthTokenizer:
    """Tokenizer stub that exposes the untruncated active token length."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(dict(kwargs))
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is True
        assert "max_length" not in kwargs
        rows = []
        for text in texts:
            words = str(text).split()
            rows.append([101] + [200 + index for index in range(len(words))] + [102])
        width = max(len(row) for row in rows)
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in rows],
            "attention_mask": [
                [1] * len(row) + [0] * (width - len(row)) for row in rows
            ],
        }


def _record(
    sample_id: str,
    task: str,
    *,
    target: list[int] | None = None,
    condition_type: str = "none",
    text: str | None = None,
    molecule: list[int] | None = None,
    target_type: str = "small_molecule",
) -> dict[str, object]:
    return {
        "id": sample_id,
        "task": task,
        "target_type": target_type,
        "condition": {
            "type": condition_type,
            "text": text,
            "molecule_bpe_ids": list(molecule or []),
        },
        "target_bpe_ids": list([4, 5] if target is None else target),
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_formal_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="axis-b-data-test-"))
    files = [
        (
            "toy_a",
            "train",
            "toy_a/train.jsonl",
            [
                _record("a0", "task_a"),
                _record("a1", "task_a", target=[6]),
                _record("a_empty", "task_a", target=[]),
                _record("a2", "task_a", target=[7, 8]),
            ],
        ),
        (
            "toy_b",
            "train",
            "toy_b/train.jsonl",
            [
                _record("b0", "task_b"),
                _record("b1", "task_b", target=[9]),
            ],
        ),
        (
            "toy_a",
            "validation",
            "toy_a/validation.jsonl",
            [_record("e0", "task_a", target=[10])],
        ),
        (
            "toy_test",
            "test",
            "toy_test/test.jsonl",
            [_record("t0", "task_a", target=[11])],
        ),
        (
            "tomgbench_v1",
            "test",
            "tomgbench_v1/MolCustom_AtomNum.jsonl",
            [_record("prompt", "molcustom_atomnum", target=[])],
        ),
    ]
    summaries = []
    for dataset, split, relative, records in files:
        path = root / relative
        _write_jsonl(path, records)
        summaries.append(
            {
                "dataset": dataset,
                "split": split,
                "relative_source": relative,
                "output_size": path.stat().st_size,
                "output_sha256": _sha256(path),
                "records": len(records),
            }
        )
    manifest = {
        "complete": True,
        "protocol": {
            "serializer": "fddfs",
            "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
            "model_token_offset": 3,
        },
        "file_summaries": summaries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@unittest.skipUnless(HAS_NUMPY, "NumPy is installed in the worker test environment")
class IndexAndDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _make_formal_root()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_atomic_uint64_rows_and_empty_target_filter(self) -> None:
        index = AXIS.build_axis_b_index(self.root, allow_non_worker=True)
        rows = index.rows("train")
        self.assertEqual(rows.dtype, np.dtype(np.uint64))
        # Five non-empty train records; the empty row is absent.
        self.assertEqual(rows.shape, (5, 3))
        self.assertEqual(index.task_counts("train"), {"task_a": 3, "task_b": 2})
        self.assertEqual(index.rows("eval").shape[0], 1)
        self.assertEqual(index.rows("test").shape[0], 1)
        self.assertTrue((index.index_dir / "train.rows.npy").is_file())
        self.assertFalse(any(index.index_dir.glob("*.raw.*")))

        dataset = AXIS.AxisBJsonlDataset(self.root, split="train", index=index)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset[0]["target_bpe_ids"], [4, 5])
        self.assertEqual(dataset[-1]["_axis_b"]["task"], "task_b")
        self.assertEqual({k: len(v) for k, v in dataset.task_indices().items()}, {"task_a": 3, "task_b": 2})
        dataset.close()

        task_a = AXIS.AxisBJsonlDataset(self.root, split="train", index=index, task="task_a")
        self.assertEqual(len(task_a), 3)
        self.assertEqual(task_a.task_counts(), {"task_a": 3})
        self.assertEqual(task_a.task_indices(), {"task_a": [0, 1, 2]})
        self.assertTrue(all(task_a[i]["task"] == "task_a" for i in range(len(task_a))))
        task_a.close()

        explicit_multi = AXIS.AxisBJsonlDataset(
            self.root,
            split="train",
            index=index,
            tasks=("task_b", "task_a"),
        )
        self.assertEqual(len(explicit_multi), 5)
        self.assertEqual(explicit_multi.tasks, ("task_b", "task_a"))
        self.assertIsNone(explicit_multi.task)
        self.assertEqual(explicit_multi.task_counts(), {"task_b": 2, "task_a": 3})
        self.assertEqual(
            {key: len(value) for key, value in explicit_multi.task_indices().items()},
            {"task_a": 3, "task_b": 2},
        )
        explicit_multi.close()
        with self.assertRaisesRegex(ValueError, "either task or tasks"):
            AXIS.AxisBJsonlDataset(
                self.root,
                split="train",
                index=index,
                task="task_a",
                tasks=("task_b",),
            )

        # A task with no validation file is a valid empty eval view; it must
        # not silently fall back to another task's records.
        task_b_eval = AXIS.AxisBJsonlDataset(self.root, split="eval", index=index, task="task_b")
        self.assertEqual(len(task_b_eval), 0)
        self.assertEqual(task_b_eval.task_counts(), {})
        task_b_eval.close()

        # A second call must reuse the compatible sidecar without changing it.
        reused = AXIS.build_axis_b_index(self.root, allow_non_worker=True)
        self.assertEqual(reused.metadata["manifest_sha256"], index.metadata["manifest_sha256"])

    def test_tomg_train_leak_is_rejected(self) -> None:
        # The formal loader refuses a TOMG file mislabeled as train, even if it
        # contains a non-empty target due to an accidental preprocessing change.
        manifest_path = self.root / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        for item in payload["file_summaries"]:
            if item["dataset"] == "tomgbench_v1":
                item["split"] = "train"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(AXIS.ManifestError):
            AXIS.build_axis_b_index(self.root, allow_non_worker=True)

    def test_index_reuse_has_no_manifest_sha_or_mtime_gate(self) -> None:
        index = AXIS.build_axis_b_index(self.root, allow_non_worker=True)
        metadata_path = index.index_dir / "index_manifest.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["manifest_sha256"] = "provenance-only-not-an-acceptance-gate"
        metadata["sentinel"] = "reused"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["non_semantic_note"] = "changing this must not reject the index"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (self.root / "toy_a" / "train.jsonl").touch()

        loaded = AXIS.AxisBIndex.load(self.root, index.index_dir)
        self.assertEqual(loaded.metadata["sentinel"], "reused")
        reused = AXIS.build_axis_b_index(self.root, allow_non_worker=True)
        self.assertEqual(reused.metadata["sentinel"], "reused")

    def test_train_derived_validation_holdout_is_deterministic_and_disjoint(self) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["protocol"]["validation_holdout"] = {
            "method": "sha256_record_id_fraction_v1",
            "tasks": {"task_b": {"fraction": 0.5, "seed": 1}},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        index = AXIS.build_axis_b_index(self.root, allow_non_worker=True)
        train = AXIS.AxisBJsonlDataset(
            self.root, split="train", index=index, task="task_b"
        )
        evaluation = AXIS.AxisBJsonlDataset(
            self.root, split="eval", index=index, task="task_b"
        )
        train_ids = {train[position]["id"] for position in range(len(train))}
        eval_ids = {
            evaluation[position]["id"] for position in range(len(evaluation))
        }
        self.assertEqual(train_ids, {"b1"})
        self.assertEqual(eval_ids, {"b0"})
        self.assertFalse(train_ids.intersection(eval_ids))
        self.assertEqual(train_ids.union(eval_ids), {"b0", "b1"})
        train.close()
        evaluation.close()

        reused = AXIS.build_axis_b_index(self.root, allow_non_worker=True)
        self.assertEqual(reused.task_counts("train")["task_b"], 1)
        self.assertEqual(reused.task_counts("eval")["task_b"], 1)

        manifest["protocol"]["validation_holdout"]["tasks"]["task_b"]["seed"] = 2
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(AXIS.AxisBDataError, "index is stale"):
            AXIS.build_axis_b_index(self.root, allow_non_worker=True)


@unittest.skipUnless(HAS_NUMPY, "NumPy is installed in the worker test environment")
class SamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _make_formal_root()
        self.index = AXIS.build_axis_b_index(self.root, allow_non_worker=True)
        self.dataset = AXIS.AxisBJsonlDataset(self.root, split="train", index=self.index)

    def tearDown(self) -> None:
        import shutil

        self.dataset.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_sqrt_allocation_and_ddp_determinism(self) -> None:
        kwargs = {
            "batch_sizes": {"task_a": 1, "task_b": 1},
            "steps_per_epoch": 12,
            "seed": 91,
        }
        rank0 = AXIS.HomogeneousTaskBatchSampler(self.dataset, rank=0, world_size=2, **kwargs)
        rank1 = AXIS.HomogeneousTaskBatchSampler(self.dataset, rank=1, world_size=2, **kwargs)
        # sqrt(3):sqrt(2), largest remainder for 12 steps is 7:5.
        self.assertEqual(rank0.task_step_counts, {"task_a": 7, "task_b": 5})
        batches0 = list(rank0)
        batches1 = list(rank1)
        self.assertEqual(len(batches0), 12)
        self.assertEqual(batches0, list(AXIS.HomogeneousTaskBatchSampler(self.dataset, rank=0, world_size=2, **kwargs)))
        for left, right in zip(batches0, batches1):
            self.assertEqual(self.dataset[left[0]]["task"], self.dataset[right[0]]["task"])
        rank0.set_epoch(1)
        self.assertNotEqual(batches0, list(rank0))

    def test_grouped_accumulation_matches_four_rank_virtual_batch(self) -> None:
        base_kwargs = {
            "batch_sizes": {"task_a": 1, "task_b": 1},
            "steps_per_epoch": 12,
            "seed": 137,
        }
        ddp = [
            list(
                AXIS.HomogeneousTaskBatchSampler(
                    self.dataset, rank=rank, world_size=4, **base_kwargs
                )
            )
            for rank in range(4)
        ]
        single = AXIS.HomogeneousTaskBatchSampler(
            self.dataset,
            rank=0,
            world_size=1,
            task_group_size=4,
            batch_sizes=base_kwargs["batch_sizes"],
            steps_per_epoch=48,
            seed=base_kwargs["seed"],
        )
        single_batches = list(single)
        self.assertEqual(len(single_batches), 48)
        self.assertEqual(single.task_group_counts, {"task_a": 7, "task_b": 5})
        self.assertEqual(single.task_step_counts, {"task_a": 28, "task_b": 20})
        for step in range(12):
            group = single_batches[step * 4 : (step + 1) * 4]
            task = self.dataset[group[0][0]]["task"]
            self.assertTrue(all(self.dataset[row[0]]["task"] == task for row in group))
            self.assertEqual(group, [ddp[rank][step] for rank in range(4)])

    def test_group_size_requires_divisible_microsteps(self) -> None:
        with self.assertRaises(ValueError):
            AXIS.HomogeneousTaskBatchSampler(
                self.dataset,
                batch_sizes=1,
                steps_per_epoch=10,
                task_group_size=4,
            )


class GroupedSamplerContractTest(unittest.TestCase):
    class FakeDataset:
        def __init__(self) -> None:
            self.rows = {
                "task_a": list(range(3)),
                "task_b": list(range(3, 5)),
            }
            self.lookup = {
                index: task for task, indices in self.rows.items() for index in indices
            }

        def task_indices(self):
            return self.rows

        def __getitem__(self, index):
            return {"task": self.lookup[int(index)]}

    def test_single_rank_a4_exactly_matches_four_rank_draws(self) -> None:
        dataset = self.FakeDataset()
        base = {
            "batch_sizes": {"task_a": 1, "task_b": 1},
            "steps_per_epoch": 12,
            "seed": 137,
        }
        ddp = [
            list(
                AXIS.HomogeneousTaskBatchSampler(
                    dataset, rank=rank, world_size=4, **base
                )
            )
            for rank in range(4)
        ]
        single_sampler = AXIS.HomogeneousTaskBatchSampler(
            dataset,
            rank=0,
            world_size=1,
            task_group_size=4,
            batch_sizes=base["batch_sizes"],
            steps_per_epoch=48,
            seed=base["seed"],
        )
        single = list(single_sampler)
        self.assertEqual(single_sampler.task_group_counts, {"task_a": 7, "task_b": 5})
        self.assertEqual(single_sampler.task_step_counts, {"task_a": 28, "task_b": 20})
        for update in range(12):
            group = single[update * 4 : (update + 1) * 4]
            self.assertEqual(group, [ddp[rank][update] for rank in range(4)])
            tasks = {dataset[row[0]]["task"] for row in group}
            self.assertEqual(len(tasks), 1)

    def test_group_size_rejects_incomplete_optimizer_group(self) -> None:
        with self.assertRaises(ValueError):
            AXIS.HomogeneousTaskBatchSampler(
                self.FakeDataset(),
                batch_sizes=1,
                steps_per_epoch=10,
                task_group_size=4,
            )


class CollatorTest(unittest.TestCase):
    def test_strict_mode_rejects_every_runtime_truncation_site(self) -> None:
        with self.subTest("text tokenizer cap"):
            with self.assertRaisesRegex(
                AXIS.SchemaError, "text condition would be truncated"
            ):
                AXIS.AxisBSFTCollator(
                    StrictLengthTokenizer(),
                    qwen_max_context=32,
                    text_max_length=4,
                    condition_token_budget=8,
                    return_tensors="list",
                    strict_no_runtime_truncation=True,
                )([
                    _record(
                        "long-text",
                        "task",
                        target=[1],
                        condition_type="text",
                        text="zero one two",
                    )
                ])

        with self.subTest("molecule stream cap"):
            with self.assertRaisesRegex(
                AXIS.SchemaError, "molecule condition.*would be truncated"
            ):
                AXIS.AxisBSFTCollator(
                    None,
                    qwen_max_context=32,
                    molecule_max_length=4,
                    condition_token_budget=8,
                    return_tensors="list",
                    molecule_condition_mode="direct_qwen",
                    strict_no_runtime_truncation=True,
                )([
                    _record(
                        "long-molecule",
                        "task",
                        target=[1],
                        condition_type="molecule",
                        molecule=[0, 1, 2, 3, 4],
                    )
                ])

        with self.subTest("text condition-token budget"):
            with self.assertRaisesRegex(
                AXIS.SchemaError, "text condition.*condition_token_budget"
            ):
                AXIS.AxisBSFTCollator(
                    StrictLengthTokenizer(),
                    qwen_max_context=32,
                    text_max_length=8,
                    condition_token_budget=4,
                    return_tensors="list",
                    strict_no_runtime_truncation=True,
                )([
                    _record(
                        "text-budget",
                        "task",
                        target=[1],
                        condition_type="text",
                        text="zero one two",
                    )
                ])

        with self.subTest("molecule condition-token budget"):
            with self.assertRaisesRegex(
                AXIS.SchemaError, "molecule condition.*condition_token_budget"
            ):
                AXIS.AxisBSFTCollator(
                    None,
                    qwen_max_context=32,
                    molecule_max_length=8,
                    condition_token_budget=4,
                    return_tensors="list",
                    strict_no_runtime_truncation=True,
                )([
                    _record(
                        "molecule-budget",
                        "task",
                        target=[1],
                        condition_type="molecule",
                        molecule=[0, 1, 2, 3, 4],
                    )
                ])

        with self.subTest("combined prefix budget"):
            with self.assertRaisesRegex(
                AXIS.SchemaError, "combined condition prefix"
            ):
                AXIS.AxisBSFTCollator(
                    StrictLengthTokenizer(),
                    qwen_max_context=10,
                    text_max_length=8,
                    molecule_max_length=8,
                    condition_token_budget=8,
                    return_tensors="list",
                    strict_no_runtime_truncation=True,
                )([
                    _record(
                        "long-prefix",
                        "task",
                        target=[1],
                        condition_type="both",
                        text="one",
                        molecule=[0, 1, 2, 3],
                    )
                ])

        with self.subTest("target budget"):
            with self.assertRaisesRegex(
                AXIS.SchemaError, "target.*would be truncated"
            ):
                AXIS.AxisBSFTCollator(
                    None,
                    qwen_max_context=8,
                    return_tensors="list",
                    strict_no_runtime_truncation=True,
                )([
                    _record(
                        "long-target",
                        "task",
                        target=[0, 1, 2, 3, 4, 5],
                    )
                ])

    def test_equal_text_max_and_condition_budget_directly_truncate_tokens(self) -> None:
        tokenizer = PrefixTruncatingTokenizer()
        output = AXIS.AxisBSFTCollator(
            tokenizer,
            qwen_max_context=16,
            return_tensors="list",
            text_max_length=4,
            condition_token_budget=4,
        )(
            [
                _record(
                    "direct-prefix",
                    "mollang_generation",
                    target=[7],
                    condition_type="text",
                    text="zero one two three four five",
                )
            ]
        )
        self.assertEqual(tokenizer.calls[0]["max_length"], 4)
        self.assertEqual(output["text_input_ids"], [[101, 200, 201, 202]])
        self.assertEqual(output["text_attention_mask"], [[1, 1, 1, 1]])
        self.assertEqual(output["text_prefix_lengths"], [4])

    def test_offset_truncation_eos_and_prefix_lengths(self) -> None:
        tokenizer = TinyTokenizer()
        collator = AXIS.AxisBSFTCollator(
            tokenizer,
            qwen_max_context=16,
            return_tensors="list",
            molecule_prepend_bos=False,
        )
        batch = [
            _record(
                "x",
                "task",
                target=list(range(10)),
                condition_type="both",
                text="a small description",
                molecule=[0, 1, 2, 3],
            )
        ]
        output = collator(batch)
        # The text encoder emits four positions ([CLS] + three words) and the
        # direct Qwen molecule stream emits four model-space tokens.  The raw
        # FDDFS <SEP> id 2 maps to Qwen id 5; Qwen EOS=2 is not inserted here.
        # Context 16 leaves room for five target BPE ids plus type/BOS/EOS.
        self.assertEqual(output["decoder_input_ids"], [[17208, 1, 3, 4, 5, 6, 7, 2]])
        self.assertEqual(output["decoder_labels"], [[-100, -100, 3, 4, 5, 6, 7, 2]])
        self.assertEqual(output["molecule_input_ids"], [[3, 4, 5, 6]])
        self.assertEqual(output["prefix_lengths"], [8])
        self.assertEqual(output["text_prefix_lengths"], [4])
        self.assertEqual(output["molecule_prefix_lengths"], [4])
        self.assertEqual(output["loss_start_positions"], [10])
        self.assertEqual(output["loss_token_counts"], [6])
        self.assertEqual(output["target_truncated"], [True])
        self.assertEqual(tokenizer.calls[0]["max_length"], 384)

    def test_direct_mode_preserves_component_separator_without_eos(self) -> None:
        record = _record(
            "sep",
            "task",
            target=[1],
            condition_type="molecule",
            molecule=[0, 2, 3],
        )
        output = AXIS.AxisBSFTCollator(
            None, qwen_max_context=32, return_tensors="list", molecule_condition_mode="direct_qwen"
        )([record])
        self.assertEqual(output["molecule_input_ids"], [[3, 5, 6]])
        self.assertNotIn(AXIS.EOS_TOKEN_ID, output["molecule_input_ids"][0])

    def test_optional_bos_for_molecule_encoder(self) -> None:
        record = _record(
            "m",
            "task",
            target=[1],
            condition_type="molecule",
            molecule=[7],
        )
        output = AXIS.AxisBSFTCollator(
            None,
            qwen_max_context=32,
            return_tensors="list",
            molecule_prepend_bos=True,
            molecule_condition_mode="molbert",
        )([record])
        self.assertEqual(output["molecule_input_ids"], [[1, 10, 2]])

    def test_raw_id_validation(self) -> None:
        record = _record("bad", "task", target=[17_204])
        with self.assertRaises(AXIS.SchemaError):
            AXIS.AxisBSFTCollator(None, qwen_max_context=32, return_tensors="list")([record])

    def test_molecule_cap_preserves_eos(self) -> None:
        record = _record(
            "long-mol",
            "task",
            target=[1],
            condition_type="molecule",
            molecule=list(range(AXIS.RAW_BPE_MAX_ID + 1))[:600],
        )
        output = AXIS.AxisBSFTCollator(
            None,
            qwen_max_context=1024,
            return_tensors="list",
            molecule_max_length=512,
            molecule_condition_mode="molbert",
        )([record])
        self.assertEqual(len(output["molecule_sequences"][0]), 512)
        self.assertEqual(output["molecule_sequences"][0][-1], AXIS.EOS_TOKEN_ID)
        self.assertEqual(output["molecule_truncated"], [True])

    def test_total_context_clips_condition_and_keeps_target_room(self) -> None:
        record = _record(
            "long-condition",
            "task",
            target=list(range(20)),
            condition_type="molecule",
            molecule=list(range(20)),
        )
        output = AXIS.AxisBSFTCollator(
            None,
            qwen_max_context=12,
            return_tensors="list",
            condition_token_budget=512,
        )([record])
        # 12 total positions reserve four for target type/BOS/raw/EOS.
        self.assertEqual(output["molecule_prefix_lengths"], [8])
        self.assertEqual(output["prefix_lengths"], [8])
        self.assertEqual(output["target_truncated"], [True])
        self.assertEqual(output["decoder_input_ids"][0][-1], AXIS.EOS_TOKEN_ID)


if __name__ == "__main__":
    unittest.main()
