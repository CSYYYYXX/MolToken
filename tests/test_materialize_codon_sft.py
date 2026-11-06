import unittest
from unittest.mock import patch

from cli.data import materialize_codon_sft as m


class CodonMaterializationTests(unittest.TestCase):
    def test_source_flag_selects_training_not_a_test_split(self):
        self.assertTrue(m.select_finetuning_row({"fine_tuned": "True"}))
        self.assertFalse(m.select_finetuning_row({"fine_tuned": "False"}))
        with self.assertRaises(ValueError):
            m.select_finetuning_row({"fine_tuned": "maybe"})

    def test_identical_retained_protein_never_crosses_holdout(self):
        refs = [{"id": str(i), "retained_protein": str(i // 2)} for i in range(40)]
        held = m.split_groups(refs)
        self.assertEqual(held, m.split_groups(list(reversed(refs))))
        for i in range(0, 40, 2):
            self.assertEqual(str(i) in held, str(i+1) in held)
        self.assertGreater(len(held), 0)

    def test_short_pair_complete_and_translation_checked(self):
        ref = {"id": "x", "source_line": 1, "organism": "test", "protein": "MA", "dna": "ATGGCTTAA"}
        with patch.dict(m.TEXT_LENGTHS, {"test": 3}), patch.object(m, "encode", side_effect=lambda k, s: [4] * len(s)):
            record, provenance = m.materialize_pair(ref)
            self.assertFalse(record["meta"]["source_dna_truncated"])
            self.assertEqual(len(record["target_bpe_ids"]), 9)
            self.assertEqual(provenance["retained_dna"], ref["dna"])
            with self.assertRaisesRegex(ValueError, "translate"):
                m.materialize_pair({**ref, "protein": "MM"})

    def test_long_pair_is_aligned_fragment_not_bpe_slice(self):
        ref = {"id": "x", "source_line": 1, "organism": "test", "protein": "M" * 300,
               "dna": "ATG" * 300 + "TAA"}
        with patch.dict(m.TEXT_LENGTHS, {"test": 3}), patch.object(m, "encode", side_effect=lambda k, s: [4] * len(s)):
            record, provenance = m.materialize_pair(ref)
        self.assertTrue(record["meta"]["source_dna_truncated"])
        self.assertEqual(len(provenance["retained_dna"]), 3 * len(provenance["retained_protein"]))
        self.assertLessEqual(record["meta"]["full_context_tokens"], 512)
        self.assertEqual(len(record["target_bpe_ids"]), len(provenance["retained_dna"]))

    def test_condition_ring_overflow_preserves_complete_dna(self):
        def encoder(kind, seq):
            if kind == "protein" and len(seq) > 10:
                raise m.RingIdOverflow("test", 300, 249, 583)
            return [4] * len(seq)
        ref = {"id": "x", "source_line": 1, "organism": "test", "protein": "M" * 20,
               "dna": "ATG" * 20 + "TAA"}
        with patch.dict(m.TEXT_LENGTHS, {"test": 3}), patch.object(m, "encode", side_effect=encoder):
            record, provenance = m.materialize_pair(ref)
        self.assertFalse(record["meta"]["source_dna_truncated"])
        self.assertTrue(record["meta"]["condition_ring_limit_errors"])
        self.assertEqual(provenance["retained_dna"], ref["dna"])

    def test_dna_ring_overflow_is_explicit_aligned_prefix(self):
        def encoder(kind, seq):
            if kind == "dna" and len(seq) > 900:
                raise m.RingIdOverflow("test", 300, 249, 583)
            return [4] * len(seq)
        ref = {"id": "x", "source_line": 1, "organism": "test", "protein": "M" * 400,
               "dna": "ATG" * 400 + "TAA"}
        with patch.dict(m.TEXT_LENGTHS, {"test": 3}), patch.object(m, "encode", side_effect=encoder):
            record, provenance = m.materialize_pair(ref)
        self.assertTrue(record["meta"]["source_dna_truncated"])
        self.assertIsNotNone(record["meta"]["target_ring_limit_error"])
        self.assertLessEqual(record["meta"]["full_context_tokens"], 512)
        self.assertEqual(len(provenance["retained_dna"]), 3*len(provenance["retained_protein"]))

    def test_unexpected_encoding_error_propagates(self):
        with patch.object(m, "encode", side_effect=RuntimeError("unrelated failure")):
            with self.assertRaisesRegex(RuntimeError, "unrelated failure"):
                m.encode_condition_prefix("M" * 20)


if __name__ == "__main__":
    unittest.main()
