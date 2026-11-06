import unittest

from pipeline.finetune.codon_context import coding_prefix, fit_coding_prefix


class CodonContextTests(unittest.TestCase):
    def test_codon_prefix_and_no_invented_stop(self):
        aa, dna = coding_prefix("MAA", "ATGGCTGCTTAA", 2)
        self.assertEqual((aa, dna), ("MA", "ATGGCT"))
        self.assertEqual(len(dna), 3 * len(aa))

    def test_bad_source_lengths_fail(self):
        with self.assertRaisesRegex(ValueError, "terminal stop"):
            coding_prefix("MAA", "ATGGCT", 1)

    def test_both_parts_are_reencoded_and_fit_verified(self):
        calls = []
        def encoder(kind, seq):
            calls.append((kind, seq))
            return [4] * len(seq)
        r = fit_coding_prefix("M" * 100, "ATG" * 100 + "TAA", text_tokens=3,
                              original_protein_bpe=100, original_dna_bpe=303,
                              encode_sequence=encoder, max_context=100)
        self.assertLessEqual(r["full_context_tokens"], 100)
        self.assertEqual(len(r["dna"]), len(r["protein"]) * 3)
        self.assertEqual(calls[0][0], "protein")
        self.assertEqual(calls[1][0], "dna")

    def test_underestimated_bpe_requires_actual_smaller_prefix(self):
        r = fit_coding_prefix("M" * 100, "ATG" * 100 + "TAA", text_tokens=3,
                              original_protein_bpe=50, original_dna_bpe=50,
                              encode_sequence=lambda kind, seq: [4] * (2 * len(seq)), max_context=100)
        self.assertGreater(r["encoding_attempts"], 1)
        self.assertLessEqual(r["full_context_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
