import unittest

from pipeline.finetune.target_priority import pack_target_first, target_priority_budget
from pipeline.finetune.axis_b_data import AxisBSFTCollator, MODEL_TOKEN_OFFSET, EOS_TOKEN_ID


class SpeciesTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[11, 12, 13] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts]}


def record(target=300, protein=300):
    return {"id": "toy", "task": "codontransformer_protein_to_dna", "target_type": "dna",
            "condition": {"type": "both", "text": "species", "molecule_bpe_ids": [7] * protein},
            "target_bpe_ids": [8] * target}


class TargetPriorityTests(unittest.TestCase):
    def test_cap_depends_on_complete_target_and_framing(self):
        plan = target_priority_budget(target_tokens=300, text_tokens=3, molecule_tokens=300)
        self.assertEqual(plan["molecule_token_cap"], 206)
        self.assertEqual(plan["packed_context_tokens"], 512)
        self.assertFalse(plan["target_truncated"])

    def test_short_condition_kept_without_padding(self):
        result, plan = pack_target_first(record(target=20, protein=10), text_tokens=3)
        self.assertEqual(len(result["condition"]["molecule_bpe_ids"]), 10)
        self.assertEqual(plan["packed_context_tokens"], 36)
        self.assertFalse(plan["condition_truncated"])

    def test_original_and_complete_target_unchanged(self):
        original = record()
        result, plan = pack_target_first(original, text_tokens=3)
        self.assertEqual(len(original["condition"]["molecule_bpe_ids"]), 300)
        self.assertEqual(result["target_bpe_ids"], original["target_bpe_ids"])
        self.assertEqual(len(result["condition"]["molecule_bpe_ids"]), 206)

    def test_target_alone_cannot_fit_is_explicit(self):
        original = record(target=510)
        result, plan = pack_target_first(original, text_tokens=3)
        self.assertIsNone(result)
        self.assertEqual(plan["status"], "complete_target_exceeds_context")
        self.assertEqual(len(original["target_bpe_ids"]), 510)

    def test_species_text_must_also_fit(self):
        result, plan = pack_target_first(record(target=507), text_tokens=3)
        self.assertIsNone(result)
        self.assertEqual(plan["status"], "complete_target_and_text_exceed_context")

    def test_zero_remaining_condition_is_explicit_not_faked(self):
        result, plan = pack_target_first(record(target=506), text_tokens=3)
        self.assertEqual(result["condition"]["type"], "text")
        self.assertEqual(result["condition"]["molecule_bpe_ids"], [])
        self.assertTrue(plan["protein_condition_empty"])

    def test_actual_strict_collator_receives_full_target_and_loss_labels(self):
        rows = [pack_target_first(record(target=n), text_tokens=3)[0] for n in (20, 300, 506)]
        collator = AxisBSFTCollator(SpeciesTokenizer(), qwen_max_context=512,
                                   condition_token_budget=512, molecule_condition_mode="direct_qwen",
                                   strict_no_runtime_truncation=True, return_tensors="list")
        batch = collator(rows)
        for i, row in enumerate(rows):
            n = len(row["target_bpe_ids"])
            expected = [8 + MODEL_TOKEN_OFFSET] * n + [EOS_TOKEN_ID]
            self.assertEqual(batch["decoder_input_ids"][i][2:2+n+1], expected)
            self.assertEqual(batch["decoder_labels"][i][2:2+n+1], expected)
            self.assertFalse(batch["target_truncated"][i])
            self.assertFalse(batch["molecule_truncated"][i])
            self.assertLessEqual(sum(batch["full_attention_mask"][i]), 512)


if __name__ == "__main__":
    unittest.main()
