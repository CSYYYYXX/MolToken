import unittest
from unittest.mock import Mock

from pipeline.benchmarks.codon_metrics import decode_coding_slot, score_coding_output, summarize_coding_slots


class CodonMetricTests(unittest.TestCase):
    def setUp(self):
        self.official = Mock()
        self.official.get_CSI_value.return_value = 0.25
        self.official.get_GC_content.return_value = 40.0
        self.official.get_cfd.return_value = 40.0
        self.weights = {"GCC": 0.25, "TTT": 0.25}
        self.frequencies = {"A": (["GCC"], [1.0])}

    def score(self, dna, protein="MAFW", terminal_stop=True):
        return score_coding_output(dna, protein=protein, terminal_stop=terminal_stop,
                                   official=self.official, csi_weights=self.weights,
                                   codon_frequencies=self.frequencies)

    def test_author_functions_receive_unchanged_output(self):
        dna = "ATGGCCTTTTGGTAA"
        row = self.score(dna)
        self.assertTrue(row["translation_matches"])
        self.assertTrue(row["exact_length"])
        self.official.get_CSI_value.assert_called_once_with(dna, self.weights)
        self.official.get_GC_content.assert_called_once_with(dna, lower=False)
        self.official.get_cfd.assert_called_once_with(dna, self.frequencies, threshold=0.3)

    def test_fragments_do_not_get_fabricated_stop_codons(self):
        row = self.score("ATGGCCTTTTGG", terminal_stop=False)
        self.assertTrue(row["translation_matches"])
        self.assertFalse(self.score("ATGGCCTTTTGG")["translation_matches"])

    def test_wrong_translation_is_retained_and_marked(self):
        row = self.score("ATGGCCTTTTGGTAA", protein="MAYW")
        self.assertTrue(row["frame_valid"])
        self.assertTrue(row["exact_length"])
        self.assertFalse(row["translation_matches"])
        self.assertEqual(row["CSI"], 0.25)

    def test_ineligible_outputs_keep_slots_without_fake_property_scores(self):
        for dna, status in [(None, "decode_failure"), ("", "empty_dna"),
                            ("ATN", "noncanonical_dna"), ("AT", "incomplete_codon")]:
            row = self.score(dna)
            self.assertEqual(row["status"], status)
            self.assertFalse(row["frame_valid"])
            self.assertIsNone(row["CSI"])
            self.assertIsNone(row["GC_percent"])
        self.official.get_CSI_value.assert_not_called()

    def test_no_zip_truncation_or_length_repair(self):
        row = self.score("ATGGCCTAA", protein="MAFW")
        self.assertEqual(row["dna"], "ATGGCCTAA")
        self.assertFalse(row["exact_length"])
        self.assertFalse(row["translation_matches"])

    def test_expected_undefined_csi_is_not_zero(self):
        row = self.score("ATGTGGTAA", protein="MW")
        self.assertTrue(row["translation_matches"])
        self.assertIsNone(row["CSI"])
        self.assertEqual(row["csi_undefined_reason"], "no_synonymous_sense_codons")
        self.official.get_CSI_value.assert_not_called()

    def test_unexpected_evaluator_error_propagates(self):
        self.official.get_cfd.side_effect = KeyError("missing frequency")
        with self.assertRaisesRegex(KeyError, "missing frequency"):
            self.score("ATGGCCTTTTGGTAA")

    def test_bad_requests_or_nonfinite_scores_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.score("ATG", protein="X")
        with self.assertRaises(TypeError):
            self.score(["A", "T", "G"])
        self.official.get_CSI_value.return_value = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.score("ATGGCCTTTTGGTAA")

    def test_decode_failures_keep_slots_and_unexpected_errors_raise(self):
        sample = {"invalid_model_id": False, "ended_with_eos": True, "generated_bpe_ids": [9]}
        decoder = Mock(return_value="ATG")
        self.assertEqual(decode_coding_slot(sample, decoder), ("ATG", "decoded", None))
        for change, status in [({"invalid_model_id": True}, "invalid_model_id"),
                               ({"ended_with_eos": False}, "missing_eos"),
                               ({"generated_bpe_ids": []}, "empty_generation")]:
            decoder.reset_mock()
            self.assertEqual(decode_coding_slot({**sample, **change}, decoder), (None, status, None))
            decoder.assert_not_called()
        decoder.side_effect = ValueError("malformed FDDFS")
        self.assertEqual(decode_coding_slot(sample, decoder), (None, "graph_decode_failure", "malformed FDDFS"))
        decoder.side_effect = RuntimeError("unexpected decoder failure")
        with self.assertRaisesRegex(RuntimeError, "unexpected decoder"):
            decode_coding_slot(sample, decoder)

    def test_summary_carries_all_requests_and_translation_coverage(self):
        rows = [{"generation_status": "decoded", "metrics": self.score("ATGGCCTTTTGGTAA")},
                {"generation_status": "decoded", "metrics": self.score("ATGGCCTTTTGGTAA", protein="MAYW")},
                {"generation_status": "missing_eos", "metrics": self.score(None)}]
        result = summarize_coding_slots(rows)
        self.assertEqual(result["requested"], 3)
        self.assertEqual(result["frame_valid"], {"count": 2, "rate_all_requests": 2/3})
        self.assertEqual(result["translation_matches"], {"count": 1, "rate_all_requests": 1/3})
        self.assertEqual(result["all_frame_valid"]["CSI"]["scored"], 2)
        self.assertEqual(result["translation_matched"]["CSI"]["scored"], 1)
        self.assertEqual(result["translation_matched"]["CSI"]["coverage_all_requests"], 1/3)

    def test_all_failed_summary_has_null_not_zero_property_means(self):
        result = summarize_coding_slots([{"generation_status": "missing_eos", "metrics": self.score(None)}])
        self.assertEqual(result["translation_matches"]["rate_all_requests"], 0)
        self.assertIsNone(result["all_frame_valid"]["CSI"]["mean"])
        with self.assertRaises(ValueError):
            summarize_coding_slots([])


if __name__ == "__main__":
    unittest.main()
