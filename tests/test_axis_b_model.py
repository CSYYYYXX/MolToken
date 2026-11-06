#!/usr/bin/env python3
"""Synthetic model-contract tests; no project checkpoints or large data."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.axis_b_model import (  # noqa: E402
    AxisBConditionalModel,
    TARGET_TYPE_TOKEN_IDS,
    resize_qwen_vocab_for_axis_b,
)
from sample_axis_b_sft import _generation_groups  # noqa: E402
from pipeline.finetune.axis_b_constants import (  # noqa: E402
    QWEN_AXIS_B_MARKER,
)


class _FakeEmbedding:
    def __init__(self, rows: int) -> None:
        self.weight = SimpleNamespace(shape=(rows, 4))


class _FakeQwen:
    def __init__(
        self,
        rows: int,
        *,
        config_rows: int | None = None,
        marker: object | None = None,
        reject_mean: bool = False,
    ) -> None:
        self.input = _FakeEmbedding(rows)
        self.output = _FakeEmbedding(rows)
        self.config = SimpleNamespace(
            vocab_size=rows if config_rows is None else config_rows,
            tie_word_embeddings=True,
            axis_b_checkpoint_interface=marker,
        )
        self.reject_mean = reject_mean
        self.resize_calls: list[tuple[int, dict[str, object]]] = []
        self.tie_calls = 0

    def get_input_embeddings(self) -> _FakeEmbedding:
        return self.input

    def get_output_embeddings(self) -> _FakeEmbedding:
        return self.output

    def resize_token_embeddings(self, size: int, **kwargs: object) -> None:
        self.resize_calls.append((size, dict(kwargs)))
        if self.reject_mean:
            raise TypeError("fixture lacks mean_resizing")
        self.input.weight.shape = (size, 4)
        self.output.weight.shape = (size, 4)

    def tie_weights(self) -> None:
        self.tie_calls += 1


class AxisBModelContractTest(unittest.TestCase):
    def test_resize_requires_exact_mean_path_and_valid_resume_marker(self) -> None:
        base = _FakeQwen(17_207)
        resize_qwen_vocab_for_axis_b(base)
        self.assertEqual(base.resize_calls, [(17_213, {"mean_resizing": True})])
        self.assertEqual(base.config.axis_b_checkpoint_interface, QWEN_AXIS_B_MARKER)
        self.assertEqual(base.tie_calls, 1)

        unsupported = _FakeQwen(17_207, reject_mean=True)
        with self.assertRaisesRegex(RuntimeError, "mean_resizing=True"):
            resize_qwen_vocab_for_axis_b(unsupported)
        self.assertEqual(
            unsupported.resize_calls, [(17_213, {"mean_resizing": True})]
        )

        resume = _FakeQwen(17_213, marker=QWEN_AXIS_B_MARKER)
        resize_qwen_vocab_for_axis_b(resume)
        self.assertEqual(resume.resize_calls, [])
        self.assertEqual(resume.tie_calls, 1)

        forged = _FakeQwen(17_213)
        with self.assertRaisesRegex(ValueError, "not a complete legal"):
            resize_qwen_vocab_for_axis_b(forged)

        for invalid_size in (17_208, 17_212, 17_214):
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaisesRegex(ValueError, "must be exactly 17,207"):
                    resize_qwen_vocab_for_axis_b(_FakeQwen(invalid_size))

        mismatch = _FakeQwen(17_207, config_rows=17_208)
        with self.assertRaisesRegex(ValueError, "config/embedding vocabulary mismatch"):
            resize_qwen_vocab_for_axis_b(mismatch)

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from transformers import BertConfig, BertModel, Qwen3Config, Qwen3ForCausalLM
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"transformers Qwen3 unavailable: {exc}")
        qwen_config = Qwen3Config(
            vocab_size=17_207,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            tie_word_embeddings=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )
        qwen = Qwen3ForCausalLM(qwen_config)
        resize_qwen_vocab_for_axis_b(qwen)
        bert_config = BertConfig(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        cls.model = AxisBConditionalModel(
            qwen=qwen,
            text_encoder=BertModel(bert_config),
            encoder_dim=8,
            molecule_condition_mode="direct_qwen",
        )

    def _batch(self) -> dict[str, torch.Tensor]:
        # Row 0 has both prefixes, row 1 has none.  Decoder lengths differ and
        # therefore exercise variable-prefix right-padding.
        return {
            "decoder_input_ids": torch.tensor(
                [
                    [TARGET_TYPE_TOKEN_IDS["small_molecule"], 1, 3, 4, 2, 0],
                    [TARGET_TYPE_TOKEN_IDS["protein"], 1, 7, 2, 0, 0],
                ],
                dtype=torch.long,
            ),
            "decoder_attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.long
            ),
            "decoder_labels": torch.tensor(
                [[-100, -100, 3, 4, 2, -100], [-100, -100, 7, 2, -100, -100]],
                dtype=torch.long,
            ),
            "text_input_ids": torch.tensor([[5, 6], [0, 0]], dtype=torch.long),
            "text_attention_mask": torch.tensor([[1, 1], [0, 0]], dtype=torch.long),
            "text_present": torch.tensor([True, False]),
            # Direct Qwen model-space ids; 5 is raw FDDFS <SEP> id 2 + offset 3.
            "molecule_input_ids": torch.tensor([[8, 9, 5], [0, 0, 0]], dtype=torch.long),
            "molecule_attention_mask": torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.long),
            "molecule_present": torch.tensor([True, False]),
            # Row 0 retains two text positions and three molecule positions.
            "prefix_lengths": torch.tensor([5, 0], dtype=torch.long),
            "text_prefix_lengths": torch.tensor([2, 0], dtype=torch.long),
            "molecule_prefix_lengths": torch.tensor([3, 0], dtype=torch.long),
        }

    def test_variable_prefix_and_loss(self) -> None:
        model = self.model
        model.configure_training_stage("alignment")
        output = model(**self._batch())
        self.assertEqual(tuple(output.logits.shape[:2]), (2, 10))
        self.assertEqual(output.loss_tokens, 5)
        self.assertTrue(torch.isfinite(output.loss))

    def test_encoder_length_equal_to_budget_never_adaptive_pools(self) -> None:
        model = self.model
        previous_budget = model.condition_token_budget
        model.condition_token_budget = 2
        try:
            with patch(
                "pipeline.finetune.axis_b_model.F.adaptive_avg_pool1d",
                side_effect=AssertionError("adaptive pooling must not run"),
            ) as adaptive_pool:
                built = model.build_qwen_inputs(**self._batch())
            adaptive_pool.assert_not_called()
            self.assertEqual(built["text_prefix_lengths"].tolist(), [2, 0])
        finally:
            model.condition_token_budget = previous_budget

    def test_generation_groups_remove_right_padding_query(self) -> None:
        mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
            dtype=torch.long,
        )
        self.assertEqual(_generation_groups(mask), [(4, [0, 2]), (6, [1])])

    def test_alignment_gradient_mask_and_full_stage(self) -> None:
        model = self.model
        batch = self._batch()
        model.configure_training_stage("alignment")
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        embedding_grad = model.qwen.get_input_embeddings().weight.grad
        self.assertIsNotNone(embedding_grad)
        self.assertEqual(float(embedding_grad[:17_207].abs().max()), 0.0)
        self.assertGreater(float(embedding_grad[17_208:17_213].abs().sum()), 0.0)
        model.configure_training_stage("full")
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        self.assertGreater(float(model.qwen.model.layers[0].self_attn.q_proj.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
