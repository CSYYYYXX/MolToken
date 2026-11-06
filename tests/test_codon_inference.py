import unittest

from pipeline.finetune.codon_inference import fit_codon_input
from pipeline.finetune.frozen_axis_errors import RingIdOverflow


class CodonInferenceTests(unittest.TestCase):
    def fit(self, protein, **kwargs):
        args = dict(text_tokens=5, generation_tokens_per_codon=2.0,
                    encode_protein=lambda s: [7] * len(s))
        args.update(kwargs)
        return fit_codon_input(protein, **args)

    def test_short_full_gene_and_generation_includes_eos_budget(self):
        result = self.fit("MA")
        self.assertTrue(result["terminal_stop"])
        self.assertEqual(result["estimated_generation_tokens"], 6)
        self.assertEqual(result["prompt_tokens"] + result["max_new_tokens"], 512)
        self.assertFalse(result["reference_DNA_used_for_budget"])

    def test_long_input_is_aligned_source_prefix_not_bpe_slice(self):
        encoded = []
        def encode(s):
            encoded.append(s)
            return [7] * (len(s) * 2)
        protein = "MA" * 300
        result = self.fit(protein, encode_protein=encode)
        self.assertEqual(result["protein"], protein[:result["retained_residues"]])
        self.assertEqual(result["protein"], encoded[-1])
        self.assertEqual(len(result["protein_bpe_ids"]), 2 * len(result["protein"]))
        self.assertFalse(result["terminal_stop"])
        self.assertEqual(result["scope"], "coding_prefix")
        self.assertLessEqual(result["estimated_generation_tokens"], result["max_new_tokens"])

    def test_input_only_and_repeatable(self):
        self.assertEqual(self.fit("MA" * 300), self.fit("MA" * 300))
        with self.assertRaises(TypeError):
            self.fit("MA", hidden_dna="ATGGCTTAA")

    def test_known_ring_bound_reduces_source(self):
        def encode(s):
            if len(s) > 20:
                raise RingIdOverflow("test", 300, 249, 583)
            return [7] * len(s)
        result = self.fit("M" * 40, encode_protein=encode)
        self.assertLessEqual(result["retained_residues"], 20)
        self.assertEqual(result["attempts"][0]["reason"], "ring_capacity")

    def test_unexpected_encoder_failure_is_not_suppressed(self):
        def fail(s):
            raise RuntimeError("encoder broken")
        with self.assertRaisesRegex(RuntimeError, "encoder broken"):
            self.fit("MA", encode_protein=fail)

    def test_impossible_budget_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "do not fit"):
            self.fit("MA", generation_tokens_per_codon=1000)
        with self.assertRaises(ValueError):
            self.fit("MA", generation_tokens_per_codon=float("nan"))
        with self.assertRaises(ValueError):
            self.fit("MX")


if __name__ == "__main__":
    unittest.main()
