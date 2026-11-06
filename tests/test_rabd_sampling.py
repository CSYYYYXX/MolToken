import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, GenerationConfig, LogitsProcessorList
from pipeline.benchmarks.rabd_sampling import SequenceNLLRecorder, prompt_only, scored_generate


class RAbDSamplingTests(unittest.TestCase):
    def test_prompt_drops_answer(self):
        a = {"id": "a", "task": "rabd", "target_type": "peptide",
             "condition": {"type": "both", "text": "3", "molecule_bpe_ids": [9]},
             "target_bpe_ids": [999], "meta": {"answer": "ABC"}}
        b = {**a, "target_bpe_ids": [111], "meta": {"answer": "XYZ"}}
        self.assertEqual(prompt_only(a), prompt_only(b))
        self.assertNotIn("meta", prompt_only(a))
        self.assertEqual(a["target_bpe_ids"], [999])

    def test_eos_counted_padding_not_counted(self):
        recorder = SequenceNLLRecorder(2)
        scores = torch.tensor([[0., 1., 2.], [0., 1., 2.]])
        recorder(torch.empty((2, 0), dtype=torch.long), scores)
        recorder(torch.tensor([[2], [1]]), scores)
        nll, count = recorder.finish(torch.tensor([[2, 0], [1, 2]]))
        p = scores.log_softmax(-1)
        self.assertEqual(count, [1, 2])
        self.assertAlmostEqual(nll[0], -p[0, 2].item(), places=6)
        self.assertAlmostEqual(nll[1], -(p[1, 1] + p[1, 2]).item(), places=6)

    def test_recording_matches_hf_transition_scores_and_does_not_change_draws(self):
        torch.manual_seed(123)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16,
                            max_position_embeddings=32)
        model = Qwen3ForCausalLM(config).eval()
        embeds = model.get_input_embeddings()(torch.tensor([[3, 4, 1]]))
        mask = torch.ones((4, 3), dtype=torch.long)
        generation = GenerationConfig(do_sample=True, temperature=1., top_k=0, top_p=1.,
                                      max_new_tokens=9, eos_token_id=2, pad_token_id=0,
                                      use_cache=True)
        recorder = SequenceNLLRecorder(2)
        with torch.inference_mode():
            torch.manual_seed(7)
            out = model.generate(inputs_embeds=embeds.expand(4, -1, -1), attention_mask=mask,
                                 generation_config=generation, return_dict_in_generate=True,
                                 output_scores=True, logits_processor=LogitsProcessorList([recorder]))
            nll, count = recorder.finish(out.sequences)
            scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            for row, n in enumerate(count):
                self.assertAlmostEqual(nll[row], -scores[row, :n].sum().item(), places=5)
            torch.manual_seed(7)
            plain = model.generate(inputs_embeds=embeds.expand(4, -1, -1), attention_mask=mask,
                                   generation_config=generation)
            self.assertTrue(torch.equal(out.sequences, plain))
            # Training checkpoints can save use_cache=False and other defaults.
            # The sampler must not inherit them into the fixed evaluation rule.
            model.generation_config.use_cache = False
            model.generation_config.repetition_penalty = 1.5
            torch.manual_seed(7)
            ids, measured, counts = scored_generate(model, embeds, 4, max_context=12)
            self.assertEqual(ids, plain.tolist())
            self.assertEqual(counts, count)
            for a, b in zip(measured, nll):
                self.assertAlmostEqual(a, b, places=5)


if __name__ == "__main__":
    unittest.main()
