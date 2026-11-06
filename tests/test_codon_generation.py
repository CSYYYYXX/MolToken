import unittest
from unittest.mock import patch

import torch

from pipeline.benchmarks.codon_generation import TASK, generate_requests, prompt_only, validate_checkpoint
from pipeline.finetune.axis_b_data import EOS_TOKEN_ID


class CodonGenerationTests(unittest.TestCase):
    def request(self, rid, length):
        return {"id": rid, "condition": {"type": "both", "text": "Homo sapiens", "molecule_bpe_ids": [7]},
                "meta": {"prompt_tokens": length}, "max_new_tokens": 512-length,
                "protein": "MA", "target_bpe_ids": [999], "hidden_dna": "SECRET"}

    def test_prompt_excludes_references_and_real_target(self):
        request = self.request("x", 5)
        prompt = prompt_only(request)
        self.assertEqual(set(prompt), {"id", "task", "target_type", "condition", "target_bpe_ids"})
        self.assertEqual(prompt["target_bpe_ids"], [0])
        prompt["condition"]["molecule_bpe_ids"].append(8)
        self.assertEqual(request["condition"]["molecule_bpe_ids"], [7])

    def test_semantic_checkpoint_match_not_batch_or_hash(self):
        meta = dict(stage="full", checkpoint_role="best", lineage_id=TASK,
                    max_context=512, molecule_condition_mode="direct_qwen", batch_size=1234)
        validate_checkpoint({"metadata": meta})
        for field, value in [("stage", "alignment"), ("lineage_id", "other_task"), ("checkpoint_role", "final")]:
            with self.assertRaises(ValueError):
                validate_checkpoint({"metadata": {**meta, field: value}})

    def test_variable_prompt_groups_and_raw_failure_slots(self):
        calls = []
        class Qwen:
            def generate(self, **kwargs):
                calls.append(kwargs)
                size, length = kwargs["inputs_embeds"].shape[:2]
                self_test.assertEqual(kwargs["attention_mask"].sum().item(), size * length)
                config = kwargs["generation_config"]
                self_test.assertFalse(config.do_sample)
                self_test.assertEqual(config.num_beams, 1)
                self_test.assertEqual(config.max_new_tokens, 512-length)
                self_test.assertFalse(kwargs["use_model_defaults"])
                ids = [3, EOS_TOKEN_ID] if length == 5 else [0, EOS_TOKEN_ID]
                return torch.tensor([ids for _ in range(size)])
        class Model:
            qwen = Qwen()
            def build_qwen_inputs(self, **kwargs):
                return {"inputs_embeds": torch.zeros(3, 8, 4),
                        "attention_mask": torch.tensor([[1]*5+[0]*3, [1]*8, [1]*5+[0]*3])}
        self_test = self
        records = [self.request("a", 5), self.request("b", 8), self.request("c", 5)]
        def build(prompts, collator, device):
            self.assertTrue(all(p["target_bpe_ids"] == [0] and "hidden_dna" not in p for p in prompts))
            return {}, {"molecule_truncated": torch.tensor([False]*3)}
        with patch("cli.evaluate.sample_axis_b_sft.model_batch_from_records", side_effect=build):
            outputs = generate_requests(Model(), None, records, torch.device("cpu"))
        self.assertEqual(len(calls), 2)
        self.assertEqual([r["id"] for r in outputs], ["a", "b", "c"])
        self.assertEqual(outputs[0]["model_token_ids"], [3, EOS_TOKEN_ID])
        self.assertTrue(outputs[1]["invalid_model_id"])
        self.assertEqual(len(outputs), 3)  # no failure filtering/resampling

    def test_unexpected_generation_error_propagates(self):
        class Model:
            def build_qwen_inputs(self, **kwargs):
                raise RuntimeError("native model failure")
        with patch("cli.evaluate.sample_axis_b_sft.model_batch_from_records",
                   return_value=({}, {"molecule_truncated": torch.tensor([False])})):
            with self.assertRaisesRegex(RuntimeError, "native model failure"):
                generate_requests(Model(), None, [self.request("a", 5)], torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
