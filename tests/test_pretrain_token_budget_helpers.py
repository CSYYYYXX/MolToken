from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from cli.pretrain import train_qwen3_pretrain as pretrain
from cli.pretrain.train_qwen3_pretrain import (
    FinalEvalHeartbeat,
    count_input_tokens,
    count_loss_tokens,
    evaluate,
    optimizer_step_due,
    resolve_attention_backend,
    scale_gradients_for_token_mean,
    trim_sample_aligned_batch_to_token_budget,
)


def test_attention_backend_reports_actual_packed_xformers_path() -> None:
    config = SimpleNamespace(_attn_implementation="sdpa")
    assert resolve_attention_backend("packed-xformers", config, "sdpa") == (
        "xformers_memory_efficient_block_diagonal_causal"
    )
    assert resolve_attention_backend("hf", config, "sdpa") == (
        "torch_scaled_dot_product_attention"
    )


def _online_batch(lengths: list[int], width: int = 8) -> dict[str, torch.Tensor]:
    batch_size = len(lengths)
    input_ids = torch.zeros((batch_size, width), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = torch.arange(1, length + 1)
        attention_mask[row, :length] = 1
        labels[row, :length] = input_ids[row, :length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_lengths": torch.tensor(lengths, dtype=torch.long),
        "truncated": torch.zeros(batch_size, dtype=torch.long),
        "sample_indices": torch.arange(batch_size, dtype=torch.long),
        "domain_ids": torch.full((batch_size,), 3, dtype=torch.long),
    }


def test_input_and_loss_token_counts_keep_eos_but_exclude_padding() -> None:
    batch = _online_batch([3, 5, 2])
    assert count_input_tokens(batch) == 10
    assert count_loss_tokens(batch) == 7


def test_token_budget_trim_keeps_first_whole_record_that_crosses_target() -> None:
    batch = _online_batch([3, 5, 2, 7])
    trimmed, kept = trim_sample_aligned_batch_to_token_budget(batch, 7)

    assert kept == 2
    assert trimmed["input_ids"].shape == (2, 8)
    assert trimmed["segment_lengths"].tolist() == [3, 5]
    assert trimmed["sample_indices"].tolist() == [0, 1]
    assert trimmed["domain_ids"].tolist() == [3, 3]
    assert count_input_tokens(trimmed) == 8
    # Whole-record stopping overshoots by less than the crossing record length.
    assert count_input_tokens(trimmed) - 7 < 5


def test_token_budget_trim_returns_full_batch_when_it_is_below_remainder() -> None:
    batch = _online_batch([3, 5, 2])
    trimmed, kept = trim_sample_aligned_batch_to_token_budget(batch, 100)
    assert trimmed is batch
    assert kept == 3


def test_token_sum_gradient_normalization_matches_global_token_mean() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    # Two variable-size microbatches have mean losses (w - 0)^2 and
    # (w - 2)^2 over 1 and 3 tokens.  Accumulate loss sums exactly as the
    # training loop does, then normalize once at the optimizer boundary.
    weight = model.weight.reshape(())
    ((weight - 0.0).square() * 1).backward(retain_graph=True)
    ((weight - 2.0).square() * 3).backward()
    scale_gradients_for_token_mean(model, 4)

    expected = (2.0 * (1.0 - 0.0) * 1 + 2.0 * (1.0 - 2.0) * 3) / 4
    assert torch.allclose(model.weight.grad, torch.tensor([[expected]]))


def test_optimizer_step_due_flushes_short_final_accumulation_window() -> None:
    assert not optimizer_step_due(2, 4, False)
    assert optimizer_step_due(4, 4, False)
    assert optimizer_step_due(2, 4, True)


def test_final_eval_heartbeat_is_durable_and_throttled_to_under_60_seconds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monotonic = iter((100.0, 129.0, 130.0, 160.0))
    fsync_calls: list[int] = []
    monkeypatch.setattr(pretrain.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(pretrain.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(pretrain.os, "fsync", lambda fd: fsync_calls.append(fd))
    path = tmp_path / "train_log.jsonl"

    with FinalEvalHeartbeat(
        path,
        step=12,
        training_started_at=900.0,
        interval_seconds=30.0,
    ) as heartbeat:
        heartbeat({"batches": 1, "tokens": 10, "input_tokens": 11, "samples": 2})
        heartbeat({"batches": 2, "tokens": 20, "input_tokens": 22, "samples": 4})
        heartbeat({"batches": 3, "tokens": 30, "input_tokens": 33, "samples": 6})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert pretrain.FINAL_EVAL_HEARTBEAT_INTERVAL_SECONDS <= 60.0
    assert [record["event"] for record in records] == [
        "final_eval_start",
        "final_eval_heartbeat",
        "final_eval_heartbeat",
    ]
    assert [record["sequence"] for record in records] == [0, 1, 2]
    assert records[-1]["batches"] == 3
    assert len(fsync_calls) == len(records)


def test_nonfinite_validation_batch_never_emits_progress() -> None:
    class NonfiniteModel:
        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, **_kwargs):
            return SimpleNamespace(loss=torch.tensor(float("nan")))

    progress: list[dict[str, int]] = []
    batch = _online_batch([3])
    with pytest.raises(FloatingPointError, match="non-finite validation loss"):
        evaluate(
            NonfiniteModel(),
            [batch],
            torch.device("cpu"),
            0,
            False,
            progress.append,
        )
    assert progress == []


def test_finite_validation_batch_emits_progress_after_completion() -> None:
    class FiniteModel:
        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, **_kwargs):
            return SimpleNamespace(loss=torch.tensor(2.0))

    progress: list[dict[str, int]] = []
    result = evaluate(
        FiniteModel(),
        [_online_batch([3])],
        torch.device("cpu"),
        0,
        False,
        progress.append,
    )
    assert result["batches"] == 1
    assert progress == [
        {"batches": 1, "tokens": 2, "input_tokens": 3, "samples": 1}
    ]
