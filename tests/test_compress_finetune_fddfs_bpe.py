#!/usr/bin/env python3
"""Unit tests for FDDFS-only Axis-B BPE materialization helpers."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from compress_finetune_fddfs_bpe import (
    ForbiddenSerializerError,
    LengthStats,
    fddfs_tokens_to_base_ids,
    histogram_quantile,
    infer_split,
    resume_summary_matches,
    summarize_histogram,
    unfold_bpe_ids,
)


class FddfsTokenMappingTest(unittest.TestCase):
    def test_base_and_ring_tokens(self) -> None:
        ids = fddfs_tokens_to_base_ids(
            ["(", "N:test", "R1", "R2", ")"],
            token_to_id={"(": 0, ")": 1, "N:test": 2},
            ring_base=3,
            ring_min=1,
            first_merge_id=5,
            sample_label="sample",
        )
        self.assertEqual(ids, [0, 2, 3, 4, 1])

    def test_forbidden_legacy_token(self) -> None:
        with self.assertRaises(ForbiddenSerializerError):
            fddfs_tokens_to_base_ids(
                ["K1"],
                token_to_id={"(": 0},
                ring_base=1,
                ring_min=1,
                first_merge_id=3,
                sample_label="sample",
            )

    def test_ring_overflow_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fddfs_tokens_to_base_ids(
                ["R3"],
                token_to_id={"(": 0},
                ring_base=1,
                ring_min=1,
                first_merge_id=3,
                sample_label="sample",
            )


class BpeRoundtripTest(unittest.TestCase):
    def test_recursive_unfold(self) -> None:
        expansion = {5: (0, 1), 6: (5, 2), 7: (6, 6)}
        self.assertEqual(unfold_bpe_ids([7, 3], expansion), [0, 1, 2, 0, 1, 2, 3])


class LengthStatsTest(unittest.TestCase):
    def test_exact_statistics(self) -> None:
        hist = Counter({1: 1, 3: 1, 5: 1, 7: 1})
        self.assertEqual(histogram_quantile(hist, 0.5), 4.0)
        self.assertAlmostEqual(histogram_quantile(hist, 0.99), 6.94)
        summary = summarize_histogram(hist)
        self.assertEqual(summary["mean"], 4.0)
        self.assertEqual(summary["median"], 4.0)

    def test_merge_and_empty_count(self) -> None:
        left = LengthStats()
        left.add(10, 4)
        left.add(0, 0)
        right = LengthStats()
        right.add(8, 3)
        left.merge(right)
        payload = left.to_payload()
        self.assertEqual(payload["sequences"], 2)
        self.assertEqual(payload["empty_or_missing"], 1)
        self.assertEqual(payload["bpe_tokens"], 7)


class ResumeValidationTest(unittest.TestCase):
    def test_rejects_partial_or_tampered_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "train.jsonl"
            output.write_bytes(b"record\n")
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            source_stat = SimpleNamespace(st_size=123, st_mtime_ns=456)
            summary = {
                "complete": True,
                "source_size": 123,
                "source_mtime_ns": 456,
                "codebook_sha256": "codebook",
                "vocab_sha256": "vocab",
                "partial_max_records": 20,
                "output_size": output.stat().st_size,
                "output_sha256": digest,
            }
            self.assertTrue(
                resume_summary_matches(
                    summary,
                    source_stat=source_stat,
                    output=output,
                    codebook_sha256="codebook",
                    vocab_sha256="vocab",
                    max_records=20,
                )
            )
            self.assertFalse(
                resume_summary_matches(
                    summary,
                    source_stat=source_stat,
                    output=output,
                    codebook_sha256="codebook",
                    vocab_sha256="vocab",
                    max_records=0,
                )
            )
            output.write_bytes(b"tamper\n")
            self.assertFalse(
                resume_summary_matches(
                    summary,
                    source_stat=source_stat,
                    output=output,
                    codebook_sha256="codebook",
                    vocab_sha256="vocab",
                    max_records=20,
                )
            )


class SplitInferenceTest(unittest.TestCase):
    def test_eval_and_validation_labels_remain_distinct(self) -> None:
        self.assertEqual(
            infer_split(Path("eval_holdout.jsonl"), "unconditional_moses_v1"),
            "eval",
        )
        self.assertEqual(
            infer_split(Path("validation.jsonl"), "unconditional_guacamol_v1"),
            "valid",
        )


if __name__ == "__main__":
    unittest.main()
