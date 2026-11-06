from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from cli.data.materialize_rfamgen_figure2_axis import (
    EncodedRecord,
    FastaRecord,
    axis_sft_record,
    encode_with_context_cap,
    iter_fasta,
    resolve_post_cap_collisions,
    sklearn_70_15_15,
)


class LengthCodec:
    """Tiny codec whose target BPE length equals RNA residue length."""

    def __init__(self) -> None:
        self.seen_lengths: list[int] = []

    def encode_sequence(
        self,
        target_type: str,
        sequence: str,
        label: str,
        *,
        text_condition_tokens: int = 0,
        molecule_condition_bpe_tokens: int = 0,
    ):
        del label
        self.seen_lengths.append(len(sequence))
        assert target_type == "rna"
        bpe = list(range(len(sequence)))
        metrics = {
            "residue_length": len(sequence),
            "heavy_nodes": len(sequence),
            "fddfs_tokens": len(sequence),
            "bpe_tokens": len(bpe),
            "text_condition_tokens": text_condition_tokens,
            "molecule_condition_bpe_tokens": molecule_condition_bpe_tokens,
            "full_context_tokens": text_condition_tokens + len(bpe) + 3,
        }
        return metrics, list(sequence), bpe


def source(identifier: str, sequence: str, index: int = 0) -> FastaRecord:
    return FastaRecord(identifier, f"header {identifier}", sequence, index)


class RfamAxisMaterializerTests(unittest.TestCase):
    def test_rf01960_cap_removes_minimal_three_prime_suffix(self) -> None:
        codec = LengthCodec()
        encoded = encode_with_context_cap(
            codec,
            source("RF01960_000000000", "ACGUAC"),
            "RF01960",
            condition_tokens=1,
            context_budget=8,
        )
        self.assertEqual(encoded.training_sequence, "ACGU")
        self.assertEqual(encoded.deleted_3prime_suffix, "AC")
        self.assertEqual(codec.seen_lengths, [6, 5, 4])
        self.assertEqual(encoded.final_metrics["full_context_tokens"], 8)

    def test_other_family_over_context_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowed only for RF01960"):
            encode_with_context_cap(
                LengthCodec(),
                source("RF00177_000000000", "ACGUAC"),
                "RF00177",
                condition_tokens=1,
                context_budget=8,
            )

    def test_post_cap_collision_prefers_uncapped_reference(self) -> None:
        codec = LengthCodec()
        uncapped = encode_with_context_cap(
            codec,
            source("uncapped", "ACGU", 1),
            "RF01960",
            condition_tokens=1,
            context_budget=8,
        )
        capped = encode_with_context_cap(
            codec,
            source("capped", "ACGUAC", 0),
            "RF01960",
            condition_tokens=1,
            context_budget=8,
        )
        retained, collisions = resolve_post_cap_collisions([capped, uncapped])
        self.assertEqual([record.source.identifier for record in retained], ["uncapped"])
        self.assertEqual(collisions[0]["excluded_id"], "capped")

    def test_split_uses_two_step_family_partition(self) -> None:
        calls: list[tuple[list[str], float, int]] = []

        def splitter(values, *, test_size, random_state):
            calls.append((list(values), test_size, random_state))
            cut = 7 if len(values) == 10 else 2
            return list(values[:cut]), list(values[cut:])

        result = sklearn_70_15_15(
            [str(index) for index in range(10)], seed=42, splitter=splitter
        )
        self.assertEqual(result["train"], [str(index) for index in range(7)])
        self.assertEqual(result["validation"], ["7", "8"])
        self.assertEqual(result["test"], ["9"])
        self.assertEqual([(call[1], call[2]) for call in calls], [(0.30, 42), (0.50, 42)])

    def test_axis_record_keeps_original_reference_provenance(self) -> None:
        encoded = encode_with_context_cap(
            LengthCodec(),
            source("sample", "ACGUAC"),
            "RF01960",
            condition_tokens=1,
            context_budget=8,
        )
        payload = axis_sft_record(
            encoded,
            family="RF01960",
            split="train",
            condition_tokens=1,
            context_budget=8,
        )
        self.assertEqual(payload["condition"]["text"], "RF01960")
        self.assertEqual(payload["target_bpe_ids"], [0, 1, 2, 3])
        self.assertTrue(payload["meta"]["target_truncated"])
        self.assertTrue(payload["meta"]["evaluation_reference_uses_original_sequence"])

    def test_fasta_reader_preserves_original_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.fasta"
            path.write_text(">id source description\nAC\nGU\n", encoding="utf-8")
            records = list(iter_fasta(path))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].identifier, "id")
        self.assertEqual(records[0].sequence, "ACGU")
        self.assertEqual(records[0].input_index, 0)
        self.assertEqual(records[0].original_header, "source description")


if __name__ == "__main__":
    unittest.main()
