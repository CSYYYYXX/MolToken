from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cli.data.materialize_tcr_translate_sft import (
    DATASET_ID,
    TASK,
    CanonicalPair,
    TcrPairEncoder,
    _canonical_pair,
    closest_group_disjoint_validation,
    materialize_release,
    stable_group_score,
)


HLA_A = "A" * 34
HLA_C = "C" * 34
HLA_D = "D" * 34


class FakeCodec:
    def __init__(self, *, joined_length: int | None = None) -> None:
        self.codebook = {"name": "fake-frozen-codebook"}
        self.sequence_calls: list[tuple[str, str, str]] = []
        self.joined_calls: list[tuple[list[str], str]] = []
        self.joined_length = joined_length

    def encode_sequence(
        self,
        target_type: str,
        sequence: str,
        label: str,
        *,
        text_condition_tokens: int = 0,
        molecule_condition_bpe_tokens: int = 0,
    ):
        self.sequence_calls.append((target_type, sequence, label))
        tokens = [f"{target_type}:{residue}" for residue in sequence]
        bpe = [100 + index for index in range(len(sequence))]
        metrics = {
            "full_context_tokens": (
                text_condition_tokens
                + molecule_condition_bpe_tokens
                + len(bpe)
                + 3
            )
        }
        return metrics, tokens, bpe

    def bpe_ids(self, tokens, label):
        values = list(tokens)
        self.joined_calls.append((values, label))
        length = self.joined_length if self.joined_length is not None else len(values)
        return [500 + index for index in range(length)]


class FakeIndex:
    def __init__(self, counts: dict[str, int]) -> None:
        self.files = ({"dataset": DATASET_ID},)
        self._counts = counts

    def task_counts(self, split: str):
        return {TASK: self._counts[split]}


def fake_index_builder(root: Path, *, allow_non_worker: bool):
    assert allow_non_worker
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["split_counts"]
    canonical_counts = {
        "train": counts["train"],
        "eval": counts["validation"],
        "test": counts["test"],
    }
    index_dir = root / "axis_b_index_v1"
    index_dir.mkdir()
    (index_dir / "index_manifest.json").write_text(
        json.dumps({"complete": True}) + "\n", encoding="utf-8"
    )
    return FakeIndex(canonical_counts)


def pair(
    line_number: int,
    epitope: str,
    hla: str,
    target: str,
    *,
    release_split: str = "released-train",
) -> CanonicalPair:
    return CanonicalPair(release_split, line_number, epitope, hla, target)


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(value + "\n" for value in values), encoding="utf-8")


class TcrTranslateMaterializerTests(unittest.TestCase):
    def test_strict_canonical_filter_requires_two_components_hla34_and_20aa(self) -> None:
        accepted, reasons = _canonical_pair(
            f"ACD {HLA_A}",
            "CASSL",
            release_split="released-train",
            line_number=7,
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(reasons, ())
        self.assertEqual(accepted.record_id, "tcr-translate:released-train:000000007")

        missing_hla, reasons = _canonical_pair(
            "ACD", "CASSL", release_split="released-train", line_number=8
        )
        self.assertIsNone(missing_hla)
        self.assertIn("source_component_count_not_two", reasons)

        short_hla, reasons = _canonical_pair(
            "ACD AAAAA", "CASSL", release_split="released-train", line_number=9
        )
        self.assertIsNone(short_hla)
        self.assertIn("hla_length_not_34", reasons)

        modified_target, reasons = _canonical_pair(
            f"ACD {HLA_A}",
            "CASS*",
            release_split="released-train",
            line_number=10,
        )
        self.assertIsNone(modified_target)
        self.assertIn("target_not_strict_20aa", reasons)

    def test_inner_validation_is_deterministic_and_complete_source_disjoint(self) -> None:
        records = [
            *[pair(index + 1, "AC", HLA_A, "CASS") for index in range(7)],
            *[pair(index + 20, "CD", HLA_C, "CASR") for index in range(5)],
            *[pair(index + 40, "DE", HLA_D, "CAST") for index in range(8)],
        ]
        first = closest_group_disjoint_validation(records, fraction=0.25, seed=42)
        second = closest_group_disjoint_validation(records, fraction=0.25, seed=42)
        self.assertEqual(first, second)
        self.assertTrue(first.validation_groups)
        self.assertLess(first.validation_group_count, first.total_groups)
        validation_sources = {
            record.source_key
            for record in records
            if record.source_key in first.validation_groups
        }
        train_sources = {
            record.source_key
            for record in records
            if record.source_key not in first.validation_groups
        }
        self.assertTrue(validation_sources.isdisjoint(train_sources))
        sizes = {
            source: sum(record.source_key == source for record in records)
            for source in {record.source_key for record in records}
        }
        ordered = sorted(
            sizes,
            key=lambda source: (stable_group_score(source, seed=42), source),
        )
        cumulative = 0
        prefix_counts = []
        for source in ordered[:-1]:
            cumulative += sizes[source]
            prefix_counts.append(cumulative)
        closest_distance = min(abs(value - 5) for value in prefix_counts)
        self.assertEqual(abs(first.validation_records - 5), closest_distance)

    def test_condition_joins_raw_fddfs_with_separator_before_whole_stream_bpe(self) -> None:
        codec = FakeCodec()
        encoder = TcrPairEncoder(codec)
        condition, target, full_context = encoder.encode(
            pair(1, "AC", HLA_D, "CASS")
        )
        self.assertEqual(
            codec.joined_calls[0][0],
            ["peptide:A", "peptide:C", "<SEP>"]
            + ["protein:D"] * 34,
        )
        self.assertEqual(
            [(target_type, sequence) for target_type, sequence, _ in codec.sequence_calls],
            [("peptide", "AC"), ("protein", HLA_D), ("peptide", "CASS")],
        )
        self.assertEqual(full_context, len(condition) + len(target) + 3)

    def test_context_overflow_fails_instead_of_truncating(self) -> None:
        codec = FakeCodec(joined_length=506)
        with self.assertRaisesRegex(ValueError, "exceeds 512"):
            TcrPairEncoder(codec).encode(pair(1, "AC", HLA_A, "CASS"))

    def test_end_to_end_writes_formal_splits_manifest_references_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            write_lines(
                input_root / "pmhc2tcr_topk_train_source_dedup_uf.txt",
                [
                    f"AC {HLA_A}",
                    f"AC {HLA_A}",
                    f"CD {HLA_C}",
                    f"DE {HLA_D}",
                    "INVALID_SOURCE",
                ],
            )
            write_lines(
                input_root / "pmhc2tcr_topk_train_target_dedup_uf.txt",
                ["CASS", "CASR", "CAST", "CASY", "CASS"],
            )
            write_lines(
                input_root / "pmhc2tcr_topk_val_source_dedup.txt",
                [f"AC {HLA_A}", f"CD {HLA_C}"],
            )
            write_lines(
                input_root / "pmhc2tcr_topk_val_target_dedup.txt",
                ["CASS", "CASR"],
            )
            output_root = root / "output"
            manifest = materialize_release(
                input_root=input_root,
                output_root=output_root,
                codec=FakeCodec(),
                validation_fraction=0.25,
                expected_train_raw=5,
                expected_train_canonical=4,
                expected_released_validation_raw=2,
                expected_released_validation_canonical=2,
                progress_every=100,
                allow_non_worker_index=True,
                index_builder=fake_index_builder,
            )

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["task"], "tcr_translate_pmhc_to_cdr3b")
            self.assertEqual(manifest["dataset_id"], "tcr_translate_released_v1")
            self.assertEqual(manifest["split_counts"]["test"], 2)
            self.assertFalse(
                manifest["protocol"]["released_validation_auxiliary"]["headline"]
            )
            self.assertEqual(
                manifest["protocol"]["headline_test"]["track"],
                "external13_sparse_headline",
            )
            self.assertEqual(
                manifest["canonical_filter"]["train"]["canonical_rows"], 4
            )
            self.assertTrue((output_root / "axis_b_index_v1/index_manifest.json").is_file())
            for split in ("train", "validation", "test"):
                self.assertTrue((output_root / f"{split}.jsonl").is_file())
                self.assertTrue((output_root / f"{split}_references.jsonl").is_file())

            records = []
            for split in ("train", "validation", "test"):
                records.extend(
                    json.loads(raw)
                    for raw in (output_root / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            self.assertTrue(records)
            for record in records:
                self.assertEqual(record["target_type"], "peptide")
                self.assertEqual(
                    set(record["condition"]), {"type", "molecule_bpe_ids"}
                )
                self.assertEqual(record["condition"]["type"], "molecule")
                self.assertNotIn("text", record["condition"])
                self.assertLessEqual(record["meta"]["full_context_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
