#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import os
import random
import struct
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half

try:
    import xformers.ops as xops
    _XFORMERS_IMPORT_ERROR = None
except Exception as exc:
    xops = None
    _XFORMERS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


COMPACT_MAGIC = b"YTTMIBPE"
COMPACT_VERSION = 1
COMPACT_HEADER_STRUCT = struct.Struct("<8sIQQi")
COMPACT_LEN_STRUCT = struct.Struct("<I")
MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "labels", "segment_lengths"}
FINAL_EVAL_HEARTBEAT_INTERVAL_SECONDS = 30.0


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON record and make it durable before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


class FinalEvalHeartbeat:
    """Durable progress records from completed full-validation batches."""

    def __init__(
        self,
        path: Path,
        *,
        step: int,
        training_started_at: float,
        interval_seconds: float = FINAL_EVAL_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds <= 0 or interval_seconds > 60:
            raise ValueError("final-eval heartbeat interval must be in (0, 60] seconds")
        self.path = path
        self.step = int(step)
        self.training_started_at = float(training_started_at)
        self.interval_seconds = float(interval_seconds)
        self._sequence = 0
        self._last_heartbeat_at: float | None = None

    def _event(self, name: str, progress: dict[str, int] | None = None) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event": name,
            "step": self.step,
            "sequence": self._sequence,
            "elapsed_sec": time.time() - self.training_started_at,
        }
        if progress is not None:
            event.update(progress)
        return event

    def __enter__(self) -> FinalEvalHeartbeat:
        append_jsonl_durable(self.path, self._event("final_eval_start"))
        self._last_heartbeat_at = time.monotonic()
        return self

    def __call__(self, progress: dict[str, int]) -> None:
        current = time.monotonic()
        if (
            self._last_heartbeat_at is not None
            and current - self._last_heartbeat_at < self.interval_seconds
        ):
            return
        self._sequence += 1
        append_jsonl_durable(
            self.path, self._event("final_eval_heartbeat", progress)
        )
        self._last_heartbeat_at = current

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def expand_paths(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if matches:
            out.extend(matches)
        else:
            p = Path(pattern)
            if p.exists():
                out.append(p)
    dedup = sorted(dict.fromkeys(out))
    if not dedup:
        raise FileNotFoundError(f"no token files matched: {patterns}")
    return dedup


def sample_index_paths(index_dir: Path, token_path: Path) -> tuple[Path, Path, Path]:
    stem = token_path.name
    return (
        index_dir / f"{stem}.starts.npy",
        index_dir / f"{stem}.lengths.npy",
        index_dir / f"{stem}.meta.json",
    )


def compact_index_paths(index_dir: Path, compact_path: Path) -> tuple[Path, Path, Path]:
    stem = compact_path.name
    return (
        index_dir / f"{stem}.payload_offsets.u64",
        index_dir / f"{stem}.lengths.u32",
        index_dir / f"{stem}.meta.json",
    )


def compact_domain_ids_path(domain_id_dir: Path, compact_path: Path) -> Path:
    return domain_id_dir / f"{compact_path.name}.domain_ids.u16"


def parse_domain_names(raw: str) -> list[str]:
    names = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(set(names)) != len(names):
        raise ValueError(f"--domain-names contains duplicates: {raw}")
    return names


def model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}


def splitmix64(x: int) -> int:
    x = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def count_loss_tokens(batch_or_labels: dict[str, torch.Tensor] | torch.Tensor) -> int:
    if isinstance(batch_or_labels, dict):
        if "segment_lengths" in batch_or_labels:
            lengths = batch_or_labels["segment_lengths"].long()
            return int(torch.clamp(lengths - 1, min=0).sum().item())
        labels = batch_or_labels["labels"]
    else:
        labels = batch_or_labels
    # HF causal LM loss 会把 labels 向左错一位：位置 i 预测位置 i+1。
    # 因此最后一列永远没有监督，PAD 对应的 -100 不参与 loss。
    if labels.ndim != 2 or labels.shape[1] <= 1:
        return 0
    return int(labels[:, 1:].ne(-100).sum().item())


def count_input_tokens(batch: dict[str, torch.Tensor]) -> int:
    """Count tokens that actually enter the Transformer (EOS included, PAD excluded)."""
    if "segment_lengths" in batch:
        return int(batch["segment_lengths"].long().sum().item())
    if "attention_mask" in batch:
        return int(batch["attention_mask"].long().sum().item())
    if "input_ids" not in batch:
        raise KeyError("batch has no input_ids")
    return int(batch["input_ids"].numel())


def trim_sample_aligned_batch_to_token_budget(
    batch: dict[str, torch.Tensor],
    remaining_input_tokens: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """Keep the shortest record prefix that reaches a positive token remainder.

    Records are never split.  Consequently the final actual token count can
    exceed the target by at most one record.  This helper intentionally accepts
    only sample-aligned batches such as ``online-pad`` where one
    ``segment_lengths`` entry corresponds to one row of ``input_ids``.
    """
    remaining = int(remaining_input_tokens)
    if remaining <= 0:
        raise ValueError("remaining_input_tokens must be positive")
    if "input_ids" not in batch or "segment_lengths" not in batch:
        raise ValueError("token-budget trimming requires input_ids and segment_lengths")
    input_ids = batch["input_ids"]
    lengths = batch["segment_lengths"].long().flatten()
    if input_ids.ndim < 1:
        raise ValueError("input_ids must have a batch dimension")
    batch_size = int(input_ids.shape[0])
    if batch_size <= 0 or int(lengths.numel()) != batch_size:
        raise ValueError(
            "token-budget trimming requires exactly one segment length per batch row: "
            f"rows={batch_size} segment_lengths={int(lengths.numel())}"
        )
    if bool((lengths <= 0).any()):
        raise ValueError("segment_lengths contains a non-positive record length")

    cumulative = torch.cumsum(lengths, dim=0)
    if int(cumulative[-1].item()) < remaining:
        return batch, batch_size
    # searchsorted(..., right=False) finds the first prefix whose token count is
    # >= remaining.  Include that entire record rather than truncating a sample.
    keep = int(torch.searchsorted(cumulative, torch.tensor(remaining, device=cumulative.device)).item()) + 1
    if keep >= batch_size:
        return batch, batch_size

    trimmed: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            trimmed[key] = value[:keep]
        else:
            trimmed[key] = value
    return trimmed, keep


def scale_gradients_for_token_mean(model: torch.nn.Module, loss_tokens: float) -> None:
    """Turn accumulated token-sum gradients into an exact token-mean gradient."""
    denominator = float(loss_tokens)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"loss_tokens must be positive and finite, got {loss_tokens}")
    scale = 1.0 / denominator
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def optimizer_step_due(
    micro_steps_in_update: int,
    gradient_accumulation_steps: int,
    budget_reached: bool,
) -> bool:
    """Return whether to step, including a short final accumulation window."""
    micro_steps = int(micro_steps_in_update)
    accumulation = int(gradient_accumulation_steps)
    if micro_steps <= 0 or accumulation <= 0:
        raise ValueError("micro-step and gradient-accumulation counts must be positive")
    return micro_steps >= accumulation or bool(budget_reached)


def build_next_token_labels(labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """把右 padding 的二维 labels 转成 packed causal LM target。"""
    lengths = attention_mask.long().sum(dim=1).tolist()
    flat_labels = labels[attention_mask.bool()]
    targets = torch.full_like(flat_labels, -100)
    offset = 0
    for length in lengths:
        length = int(length)
        if length > 1:
            targets[offset: offset + length - 1] = flat_labels[offset + 1: offset + length]
        offset += length
    return targets


def build_packed_position_ids(segment_lengths: torch.Tensor) -> torch.Tensor:
    """根据每个样本段的长度生成 packed RoPE position ids，每个样本段都从 0 开始。"""
    lengths = segment_lengths.long()
    if lengths.ndim != 1:
        raise RuntimeError("segment_lengths must be a 1D tensor")
    if bool((lengths <= 0).any()):
        raise RuntimeError("segment_lengths contains non-positive length")
    total_tokens = int(lengths.sum().item())
    segment_starts = torch.cumsum(lengths, dim=0) - lengths
    token_starts = torch.repeat_interleave(segment_starts, lengths)
    return (torch.arange(total_tokens, device=lengths.device, dtype=torch.long) - token_starts).unsqueeze(0)


def build_next_token_labels_from_segments(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_lengths: torch.Tensor,
) -> torch.Tensor:
    """按 packed 样本段生成 next-token target，样本段最后一个 token 不参与 loss。"""
    flat_labels = labels[attention_mask.bool()]
    lengths = segment_lengths.to(device=flat_labels.device, dtype=torch.long)
    if int(lengths.sum().item()) != int(flat_labels.numel()):
        raise RuntimeError(
            "segment_lengths sum does not match valid label count: "
            f"{int(lengths.sum().item())} vs {int(flat_labels.numel())}"
        )
    targets = torch.full_like(flat_labels, -100)
    if flat_labels.numel() <= 1:
        return targets
    targets[:-1] = flat_labels[1:]
    segment_ends = torch.cumsum(lengths, dim=0) - 1
    targets[segment_ends] = -100
    return targets


def new_domain_stats(num_domains: int) -> dict[str, torch.Tensor]:
    return {
        "sample_counts": torch.zeros(num_domains, dtype=torch.float64),
        "input_tokens": torch.zeros(num_domains, dtype=torch.float64),
        "loss_tokens": torch.zeros(num_domains, dtype=torch.float64),
        "loss_sums": torch.zeros(num_domains, dtype=torch.float64),
    }


def add_domain_stats(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor]) -> None:
    for key in dst:
        dst[key].add_(src[key].to(dtype=torch.float64, device=dst[key].device))


def reduce_domain_stats(
    stats: dict[str, torch.Tensor],
    device: torch.device,
    ddp: bool,
) -> dict[str, torch.Tensor]:
    reduced: dict[str, torch.Tensor] = {}
    for key, value in stats.items():
        tensor = value.to(device=device, dtype=torch.float64)
        if ddp:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor.detach().cpu()
    return reduced


def domain_stats_to_log(
    stats: dict[str, torch.Tensor],
    domain_names: Sequence[str],
) -> dict[str, Any]:
    names = list(domain_names)
    sample_counts = stats["sample_counts"]
    input_tokens = stats["input_tokens"]
    loss_tokens = stats["loss_tokens"]
    loss_sums = stats["loss_sums"]
    total_samples = float(sample_counts.sum().item())
    total_input_tokens = float(input_tokens.sum().item())
    total_loss_tokens = float(loss_tokens.sum().item())
    total_loss_sum = float(loss_sums.sum().item())

    def keyed_values(values: torch.Tensor, cast: str) -> dict[str, int | float]:
        out: dict[str, int | float] = {}
        for name, value in zip(names, values.tolist()):
            out[name] = int(round(float(value))) if cast == "int" else float(value)
        return out

    def keyed_fraction(values: torch.Tensor, total: float) -> dict[str, float]:
        return {name: (float(value) / total if total > 0.0 else 0.0) for name, value in zip(names, values.tolist())}

    domain_loss: dict[str, float | None] = {}
    for name, loss_sum, token_count in zip(names, loss_sums.tolist(), loss_tokens.tolist()):
        token_count = float(token_count)
        domain_loss[name] = (float(loss_sum) / token_count) if token_count > 0.0 else None

    return {
        "domain_samples": keyed_values(sample_counts, "int"),
        "domain_sample_fraction": keyed_fraction(sample_counts, total_samples),
        "domain_input_tokens": keyed_values(input_tokens, "int"),
        "domain_input_token_fraction": keyed_fraction(input_tokens, total_input_tokens),
        "domain_loss_tokens": keyed_values(loss_tokens, "int"),
        "domain_loss_token_fraction": keyed_fraction(loss_tokens, total_loss_tokens),
        "domain_loss_sum": keyed_values(loss_sums, "float"),
        "domain_loss_contribution": keyed_fraction(loss_sums, total_loss_sum),
        "domain_loss": domain_loss,
    }


def compute_domain_loss_stats(
    batch: dict[str, torch.Tensor],
    logits: torch.Tensor,
    num_domains: int,
) -> dict[str, torch.Tensor]:
    if "domain_ids" not in batch:
        raise RuntimeError("--log-domain-loss requires domain_ids in the training batch")
    if "segment_lengths" not in batch:
        raise RuntimeError("--log-domain-loss currently requires segment_lengths")
    if logits.ndim != 2:
        raise RuntimeError("--log-domain-loss currently expects packed logits with shape [loss_tokens, vocab]")

    domain_ids = batch["domain_ids"].to(device=logits.device, dtype=torch.long).flatten()
    segment_lengths = batch["segment_lengths"].to(device=logits.device, dtype=torch.long).flatten()
    if int(domain_ids.numel()) != int(segment_lengths.numel()):
        raise RuntimeError(
            "domain_ids count must match segment count: "
            f"{int(domain_ids.numel())} vs {int(segment_lengths.numel())}"
        )
    if bool((domain_ids < 0).any()) or bool((domain_ids >= int(num_domains)).any()):
        raise RuntimeError(f"domain_ids must be in [0, {int(num_domains)})")

    labels = batch["labels"].to(device=logits.device)
    attention_mask = batch["attention_mask"].to(device=logits.device)
    targets = build_next_token_labels_from_segments(labels, attention_mask, segment_lengths)
    loss_positions = targets.ne(-100)
    selected_targets = targets[loss_positions].to(device=logits.device, dtype=torch.long)
    if int(selected_targets.numel()) != int(logits.shape[0]):
        raise RuntimeError(
            "packed logits count does not match selected target count: "
            f"{int(logits.shape[0])} vs {int(selected_targets.numel())}"
        )

    token_domain_ids = torch.repeat_interleave(domain_ids, segment_lengths)
    selected_domain_ids = token_domain_ids[loss_positions]
    per_token_loss = torch.nn.functional.cross_entropy(logits.detach().float(), selected_targets, reduction="none")

    sample_counts = torch.bincount(domain_ids, minlength=num_domains).to(torch.float64)
    input_tokens = torch.bincount(
        domain_ids,
        weights=segment_lengths.to(torch.float64),
        minlength=num_domains,
    )
    loss_tokens = torch.bincount(selected_domain_ids, minlength=num_domains).to(torch.float64)
    loss_sums = torch.zeros(num_domains, device=logits.device, dtype=torch.float64)
    loss_sums.scatter_add_(0, selected_domain_ids, per_token_loss.detach().to(torch.float64))

    return {
        "sample_counts": sample_counts.detach().cpu(),
        "input_tokens": input_tokens.detach().cpu(),
        "loss_tokens": loss_tokens.detach().cpu(),
        "loss_sums": loss_sums.detach().cpu(),
    }


def apply_rotary_pos_emb_packed(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for packed [1, total_tokens, heads, head_dim] tensors."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class U16TokenBlockDataset(Dataset):
    """Memory-map uint16 token streams and expose fixed causal-LM blocks."""

    def __init__(
        self,
        paths: Sequence[Path],
        block_size: int,
        split: str,
        val_fraction: float,
        split_strategy: str,
    ):
        if block_size < 2:
            raise ValueError("--block-size must be >= 2")
        if not (0.0 <= val_fraction < 0.5):
            raise ValueError("--val-fraction must be in [0, 0.5)")
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if split_strategy not in {"stride", "tail"}:
            raise ValueError("--split-strategy must be stride or tail")
        self.paths = list(paths)
        self.block_size = int(block_size)
        self.split = split
        self.split_strategy = split_strategy
        self.val_stride = max(2, int(round(1.0 / val_fraction))) if val_fraction > 0.0 else 0
        self.arrays: List[np.memmap] = []
        self.starts: List[int] = []
        self.lengths: List[int] = []
        self.block_counts: List[int] = []
        self.raw_block_counts: List[int] = []
        self.cum_blocks: List[int] = [0]
        for path in self.paths:
            n_tokens = path.stat().st_size // np.dtype(np.uint16).itemsize
            if n_tokens < block_size:
                continue

            if split_strategy == "tail":
                split_at = int(n_tokens * (1.0 - val_fraction))
                if split == "train":
                    start, end = 0, max(0, split_at)
                else:
                    start, end = max(0, split_at), n_tokens
                length = max(0, end - start)
                raw_blocks = length // block_size
                blocks = raw_blocks
            else:
                start, length = 0, n_tokens
                raw_blocks = length // block_size
                val_blocks = (raw_blocks + self.val_stride - 1) // self.val_stride if self.val_stride else 0
                blocks = val_blocks if split == "val" else raw_blocks - val_blocks

            if blocks <= 0:
                continue
            arr = np.memmap(path, mode="r", dtype=np.uint16)
            self.arrays.append(arr)
            self.starts.append(start)
            self.lengths.append(length)
            self.block_counts.append(blocks)
            self.raw_block_counts.append(raw_blocks)
            self.cum_blocks.append(self.cum_blocks[-1] + blocks)
        if self.cum_blocks[-1] <= 0:
            raise ValueError(f"empty {split} dataset after block split")

    def __len__(self) -> int:
        return self.cum_blocks[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = bisect.bisect_right(self.cum_blocks, idx) - 1
        local_idx = idx - self.cum_blocks[file_idx]
        if self.split_strategy == "stride":
            if self.split == "val":
                raw_block_idx = local_idx * self.val_stride
            else:
                raw_block_idx = local_idx + 1 + local_idx // (self.val_stride - 1) if self.val_stride else local_idx
        else:
            raw_block_idx = local_idx
        offset = self.starts[file_idx] + raw_block_idx * self.block_size
        chunk = np.asarray(self.arrays[file_idx][offset: offset + self.block_size], dtype=np.int64)
        if chunk.shape[0] != self.block_size:
            raise RuntimeError("short token block; dataset block accounting is inconsistent")
        input_ids = torch.from_numpy(chunk.copy()).long()
        # Qwen3ForCausalLM/HF causal LM loss 会在模型内部右移 labels；
        # 这里 labels 必须与 input_ids 相同，不能预先手动 shift。
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    def token_count(self) -> int:
        return int(sum(blocks * self.block_size for blocks in self.block_counts))


class DistributedStrideSampler(Sampler[int]):
    """不物化全量索引的 DDP 顺序分片 sampler，用于超大样本数数据集。"""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def __iter__(self):
        n = len(self.dataset)
        return iter(range(self.rank, n, self.num_replicas))

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.num_replicas + 1


class DistributedAffineShuffleSampler(Sampler[int]):
    """不物化全量索引的 DDP 随机置换 sampler，用于超大 online-pad 数据集。

    这个 sampler 用每个 epoch 的仿射置换 `(a * i + b) % n` 打乱全局样本位置。
    只要 `gcd(a, n) == 1`，它就是 `[0, n)` 上的双射，不会重复或漏样本；
    每个 rank 再取自己的全局位置 `rank, rank + world_size, ...`。
    """

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int, seed: int) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _permutation_params(self, n: int) -> tuple[int, int]:
        if n <= 1:
            return 1, 0
        key = self.seed ^ ((self.epoch & 0xFFFFFFFF) << 32)
        a = int(splitmix64(key) % n)
        if a <= 0:
            a = 1
        # a 必须和 n 互质，仿射映射才是全排列。通常只需跳几次。
        while math.gcd(a, n) != 1:
            a += 1
            if a >= n:
                a = 1
        b = int(splitmix64(key ^ 0xD1B54A32D192ED03) % n)
        return a, b

    def __iter__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return iter(())
        a, b = self._permutation_params(n)

        def iterator():
            for pos in range(self.rank, n, self.num_replicas):
                yield (a * pos + b) % n

        return iterator()

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.num_replicas + 1


class U16EosSampleDataset(Dataset):
    """按 EOS 索引读取 uint16 token 流，每次返回一条完整样本。"""

    def __init__(
        self,
        paths: Sequence[Path],
        index_dir: Path,
        split: str,
        val_fraction: float,
        split_strategy: str,
        eos_token_id: int,
        bos_token_id: int,
        add_bos: bool,
        sample_max_len: int,
    ):
        if index_dir is None:
            raise ValueError("--sample-index-dir is required when --dataset-mode=sample")
        if not index_dir.exists():
            raise FileNotFoundError(f"sample index dir does not exist: {index_dir}")
        if not (0.0 <= val_fraction < 0.5):
            raise ValueError("--val-fraction must be in [0, 0.5)")
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if split_strategy not in {"stride", "tail"}:
            raise ValueError("--split-strategy must be stride or tail")
        if sample_max_len < 0:
            raise ValueError("--sample-max-len must be >= 0")
        self.paths = list(paths)
        self.index_dir = index_dir
        self.split = split
        self.split_strategy = split_strategy
        self.val_stride = max(2, int(round(1.0 / val_fraction))) if val_fraction > 0.0 else 0
        self.eos_token_id = int(eos_token_id)
        self.bos_token_id = int(bos_token_id)
        self.add_bos = bool(add_bos)
        self.sample_max_len = int(sample_max_len)
        self.arrays: List[np.memmap] = []
        self.start_arrays: List[np.memmap] = []
        self.length_arrays: List[np.memmap] = []
        self.split_starts: List[int] = []
        self.sample_counts: List[int] = []
        self.raw_sample_counts: List[int] = []
        self.raw_token_counts: List[int] = []
        self.cum_samples: List[int] = [0]
        self._token_count = 0

        for path in self.paths:
            starts_path, lengths_path, meta_path = sample_index_paths(index_dir, path)
            missing = [str(p) for p in (starts_path, lengths_path, meta_path) if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "missing sample index files for "
                    f"{path}: {missing}. Build them on a worker with "
                    "cli/data/build_u16_eos_sample_index.py."
                )

            starts = np.load(starts_path, mmap_mode="r")
            lengths = np.load(lengths_path, mmap_mode="r")
            if starts.shape != lengths.shape:
                raise ValueError(f"index shape mismatch for {path}: starts={starts.shape}, lengths={lengths.shape}")
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            n_samples = int(starts.shape[0])
            if n_samples <= 0:
                continue

            if split_strategy == "tail":
                split_at = int(n_samples * (1.0 - val_fraction))
                if split == "train":
                    split_start, count = 0, max(0, split_at)
                else:
                    split_start, count = max(0, split_at), n_samples - max(0, split_at)
            else:
                split_start = 0
                if self.val_stride:
                    val_count = (n_samples + self.val_stride - 1) // self.val_stride
                    count = val_count if split == "val" else n_samples - val_count
                else:
                    count = 0 if split == "val" else n_samples

            if count <= 0:
                continue

            arr = np.memmap(path, mode="r", dtype=np.uint16)
            self.arrays.append(arr)
            self.start_arrays.append(starts)
            self.length_arrays.append(lengths)
            self.split_starts.append(split_start)
            self.sample_counts.append(count)
            self.raw_sample_counts.append(n_samples)
            self.raw_token_counts.append(int(meta.get("token_count", 0)))
            self.cum_samples.append(self.cum_samples[-1] + count)
            self._token_count += self._selected_token_count(lengths, split_start, count, n_samples)

        if self.cum_samples[-1] <= 0:
            raise ValueError(f"empty {split} sample dataset after split")

    def _selected_token_count(
        self,
        lengths: np.ndarray,
        split_start: int,
        count: int,
        n_samples: int,
    ) -> int:
        if self.split_strategy == "tail":
            total = int(np.asarray(lengths[split_start: split_start + count], dtype=np.uint64).sum())
        elif self.val_stride and self.split == "val":
            total = int(np.asarray(lengths[0:n_samples:self.val_stride], dtype=np.uint64).sum())
        elif self.val_stride and self.split == "train":
            all_tokens = int(np.asarray(lengths[:], dtype=np.uint64).sum())
            val_tokens = int(np.asarray(lengths[0:n_samples:self.val_stride], dtype=np.uint64).sum())
            total = all_tokens - val_tokens
        else:
            total = int(np.asarray(lengths[:], dtype=np.uint64).sum())
        if self.add_bos:
            total += int(count)
        return total

    def __len__(self) -> int:
        return self.cum_samples[-1]

    def _raw_sample_index(self, file_idx: int, local_idx: int) -> int:
        if self.split_strategy == "tail":
            return self.split_starts[file_idx] + local_idx
        if self.val_stride and self.split == "val":
            return local_idx * self.val_stride
        if self.val_stride and self.split == "train":
            return local_idx + 1 + local_idx // (self.val_stride - 1)
        return local_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = bisect.bisect_right(self.cum_samples, idx) - 1
        local_idx = idx - self.cum_samples[file_idx]
        raw_idx = self._raw_sample_index(file_idx, local_idx)
        start = int(self.start_arrays[file_idx][raw_idx])
        length = int(self.length_arrays[file_idx][raw_idx])
        if length <= 0:
            raise RuntimeError(f"bad empty sample in {self.paths[file_idx]} at raw sample {raw_idx}")
        chunk = np.asarray(self.arrays[file_idx][start: start + length], dtype=np.int64)
        if chunk.shape[0] != length:
            raise RuntimeError(
                f"short sample read from {self.paths[file_idx]} at raw sample {raw_idx}: "
                f"wanted {length}, got {chunk.shape[0]}"
            )
        if int(chunk[-1]) != self.eos_token_id:
            raise RuntimeError(
                f"sample in {self.paths[file_idx]} at raw sample {raw_idx} does not end with EOS={self.eos_token_id}"
            )
        if self.add_bos:
            ids = np.empty(length + 1, dtype=np.int64)
            ids[0] = self.bos_token_id
            ids[1:] = chunk
        else:
            ids = chunk
        if self.sample_max_len > 0 and ids.shape[0] > self.sample_max_len:
            raise RuntimeError(
                f"sample length {ids.shape[0]} exceeds --sample-max-len={self.sample_max_len}; "
                "increase --sample-max-len instead of silently truncating data"
            )
        return {"input_ids": torch.from_numpy(ids.copy()).long()}

    def token_count(self) -> int:
        return int(self._token_count)


class BpeUnfoldPolicy:
    """BPE 逆向展开采样策略接口。"""

    def should_unfold(self, token_id: int, merge_step: int, depth: int, rng: np.random.Generator) -> bool:
        raise NotImplementedError


class ConstantBpeUnfoldPolicy(BpeUnfoldPolicy):
    """每个可展开 token 用同一个概率展开；后续可替换成分布采样。"""

    def __init__(self, prob: float) -> None:
        if not (0.0 <= prob <= 1.0):
            raise ValueError("--bpe-unfold-prob must be in [0, 1]")
        self.prob = float(prob)

    def should_unfold(self, token_id: int, merge_step: int, depth: int, rng: np.random.Generator) -> bool:
        if self.prob <= 0.0:
            return False
        if self.prob >= 1.0:
            return True
        return bool(rng.random() < self.prob)


class BpeUnfolder:
    """按 merges.csv 逆向展开 BPE token。"""

    def __init__(
        self,
        merge_pairs: dict[int, tuple[int, int]],
        merge_steps: dict[int, int],
        policy: BpeUnfoldPolicy,
        max_depth: int,
        max_output_len: int,
    ) -> None:
        if max_depth < 0:
            raise ValueError("--bpe-unfold-max-depth must be >= 0")
        if max_output_len <= 0:
            raise ValueError("max_output_len must be > 0")
        self.merge_pairs = merge_pairs
        self.merge_steps = merge_steps
        self.policy = policy
        self.max_depth = int(max_depth)
        self.max_output_len = int(max_output_len)

    @classmethod
    def from_csv(
        cls,
        path: Path,
        policy_name: str,
        prob: float,
        max_depth: int,
        max_output_len: int,
    ) -> "BpeUnfolder":
        if policy_name == "none" or prob <= 0.0:
            policy: BpeUnfoldPolicy = ConstantBpeUnfoldPolicy(0.0)
        elif policy_name == "constant":
            policy = ConstantBpeUnfoldPolicy(prob)
        else:
            raise ValueError(f"unsupported --bpe-unfold-policy={policy_name!r}")
        if not path.exists():
            raise FileNotFoundError(f"BPE merge CSV does not exist: {path}")
        merge_pairs: dict[int, tuple[int, int]] = {}
        merge_steps: dict[int, int] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"left", "right", "new_id"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"merge CSV missing columns {sorted(missing)}: {path}")
            for fallback_step, row in enumerate(reader):
                new_id = int(row["new_id"])
                merge_pairs[new_id] = (int(row["left"]), int(row["right"]))
                merge_steps[new_id] = int(row.get("step") or fallback_step)
        return cls(merge_pairs, merge_steps, policy, max_depth, max_output_len)

    def expand(self, tokens: np.ndarray, rng: np.random.Generator, max_output_len: int | None = None) -> list[int]:
        limit = self.max_output_len if max_output_len is None else min(self.max_output_len, int(max_output_len))
        out: list[int] = []
        for token in tokens.tolist():
            self._append(int(token), 0, rng, out, limit)
            if len(out) >= limit:
                break
        return out[:limit]

    def _append(self, token_id: int, depth: int, rng: np.random.Generator, out: list[int], max_output_len: int) -> None:
        if len(out) >= max_output_len:
            return
        pair = self.merge_pairs.get(token_id)
        if pair is None or depth >= self.max_depth:
            out.append(token_id)
            return
        merge_step = self.merge_steps.get(token_id, -1)
        if not self.policy.should_unfold(token_id, merge_step, depth, rng):
            out.append(token_id)
            return
        left, right = pair
        self._append(left, depth + 1, rng, out, max_output_len)
        self._append(right, depth + 1, rng, out, max_output_len)


class CompactIntOnlinePadDataset(Dataset):
    """从 compact-int shard 在线读取样本，随机展开 BPE 后截断并 pad 到固定长度。"""

    def __init__(
        self,
        paths: Sequence[Path],
        index_dir: Path,
        split: str,
        val_fraction: float,
        split_strategy: str,
        token_offset: int,
        eos_token_id: int,
        pad_token_id: int,
        pad_length: int,
        truncate: bool,
        seed: int,
        unfold_max_ratio: float,
        unfolder: BpeUnfolder | None,
        domain_id_dir: Path | None = None,
        domain_split_manifest: Path | None = None,
        domain_name: str | None = None,
    ) -> None:
        if index_dir is None:
            raise ValueError("--compact-index-dir is required when --dataset-mode=online-pad")
        if not index_dir.exists():
            raise FileNotFoundError(f"compact index dir does not exist: {index_dir}")
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if split_strategy not in {"stride", "tail"}:
            raise ValueError("--split-strategy must be stride or tail")
        if not (0.0 <= val_fraction < 0.5):
            raise ValueError("--val-fraction must be in [0, 0.5)")
        if pad_length < 2:
            raise ValueError("--online-pad-length must be >= 2")
        self.paths = list(paths)
        self.index_dir = index_dir
        self.split = split
        self.split_strategy = split_strategy
        self.val_stride = max(2, int(round(1.0 / val_fraction))) if val_fraction > 0.0 else 0
        self.token_offset = int(token_offset)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.pad_length = int(pad_length)
        self.truncate = bool(truncate)
        self.seed = int(seed)
        self.unfold_max_ratio = float(unfold_max_ratio)
        if self.unfold_max_ratio <= 0.0:
            raise ValueError("--bpe-unfold-max-ratio must be > 0")
        self.epoch = 0
        self.unfolder = unfolder if split == "train" else None
        self.payload_offsets: List[np.memmap] = []
        self.length_arrays: List[np.memmap] = []
        self.byte_arrays: List[np.memmap] = []
        self.domain_id_arrays: List[np.memmap | None] = []
        self.raw_index_arrays: List[np.memmap | None] = []
        self.split_starts: List[int] = []
        self.sample_counts: List[int] = []
        self.raw_sample_counts: List[int] = []
        self.cum_samples: List[int] = [0]
        self._token_count = 0

        manifest_entries: dict[Path, dict[str, Any]] | None = None
        manifest_view: dict[str, Any] | None = None
        if domain_split_manifest is not None:
            if not domain_name:
                raise ValueError("--domain-name is required with --domain-split-manifest")
            manifest_path = Path(domain_split_manifest).expanduser().resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "domain_split_raw_indices_v1":
                raise ValueError(f"unsupported domain split manifest schema: {manifest_path}")
            if manifest.get("status") != "complete":
                raise ValueError(f"domain split manifest is not complete: {manifest_path}")
            if int(manifest["dataset"]["context_length"]) != self.pad_length:
                raise ValueError(
                    "domain split manifest context length differs from --online-pad-length: "
                    f"{manifest['dataset']['context_length']} vs {self.pad_length}"
                )
            manifest_split = "validation" if split == "val" else split
            try:
                manifest_view = manifest["views"][domain_name][manifest_split]
            except KeyError as exc:
                raise ValueError(
                    f"domain/split missing from manifest: {domain_name}/{manifest_split}"
                ) from exc
            manifest_entries = {
                Path(row["token_file"]).expanduser().resolve(): row
                for row in manifest_view["shards"]
            }
            requested_paths = {Path(path).expanduser().resolve() for path in self.paths}
            if set(manifest_entries) != requested_paths:
                missing = sorted(str(path) for path in requested_paths - set(manifest_entries))
                extra = sorted(str(path) for path in set(manifest_entries) - requested_paths)
                raise ValueError(
                    f"domain split manifest/token-file mismatch; missing={missing} extra={extra}"
                )

        for path in self.paths:
            offsets_path, lengths_path, meta_path = compact_index_paths(index_dir, path)
            missing = [str(p) for p in (offsets_path, lengths_path, meta_path) if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"missing compact index files for {path}: {missing}. "
                    "Build them on a worker with cli/data/build_compact_int_index.py."
                )
            offsets = np.memmap(offsets_path, mode="r", dtype=np.uint64)
            lengths = np.memmap(lengths_path, mode="r", dtype=np.uint32)
            if offsets.shape != lengths.shape:
                raise ValueError(f"compact index shape mismatch for {path}: offsets={offsets.shape}, lengths={lengths.shape}")
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            n_samples = int(lengths.shape[0])
            if n_samples <= 0:
                continue
            if int(meta.get("records", n_samples)) != n_samples:
                raise ValueError(f"compact index meta record mismatch for {path}")
            domain_ids = None
            if domain_id_dir is not None:
                domain_ids_path = compact_domain_ids_path(domain_id_dir, path)
                if not domain_ids_path.exists():
                    raise FileNotFoundError(f"missing compact domain id sidecar for {path}: {domain_ids_path}")
                domain_ids = np.memmap(domain_ids_path, mode="r", dtype=np.uint16)
                if int(domain_ids.shape[0]) != n_samples:
                    raise ValueError(
                        f"domain id sidecar length mismatch for {path}: "
                        f"domain_ids={int(domain_ids.shape[0])} samples={n_samples}"
                    )

            raw_indices = None
            manifest_tokens = None
            if manifest_entries is not None:
                entry = manifest_entries[path.expanduser().resolve()]
                raw_indices_path = Path(entry["raw_indices_file"]).expanduser().resolve()
                if not raw_indices_path.is_file():
                    raise FileNotFoundError(
                        f"missing frozen raw-index view for {path}: {raw_indices_path}"
                    )
                raw_indices = np.memmap(raw_indices_path, mode="r", dtype="<u4")
                if int(raw_indices.shape[0]) != int(entry["records"]):
                    raise ValueError(
                        f"raw-index view count mismatch for {path}: "
                        f"indices={int(raw_indices.shape[0])} manifest={entry['records']}"
                    )
                # The materializer emits indices in strictly increasing raw
                # order, so the last element is the maximum without a full
                # O(records) scan at every worker startup.
                if raw_indices.size and int(raw_indices[-1]) >= n_samples:
                    raise ValueError(f"raw-index view exceeds source shard for {path}")
                manifest_tokens = int(entry["effective_input_tokens_with_eos"])

            if raw_indices is not None:
                split_start, count = 0, int(raw_indices.shape[0])
            elif split_strategy == "tail":
                split_at = int(n_samples * (1.0 - val_fraction))
                if split == "train":
                    split_start, count = 0, max(0, split_at)
                else:
                    split_start, count = max(0, split_at), n_samples - max(0, split_at)
            else:
                split_start = 0
                if self.val_stride:
                    val_count = (n_samples + self.val_stride - 1) // self.val_stride
                    count = val_count if split == "val" else n_samples - val_count
                else:
                    count = 0 if split == "val" else n_samples
            if count <= 0:
                continue

            self.payload_offsets.append(offsets)
            self.length_arrays.append(lengths)
            self.byte_arrays.append(np.memmap(path, mode="r", dtype=np.uint8))
            self.domain_id_arrays.append(domain_ids)
            self.raw_index_arrays.append(raw_indices)
            self.split_starts.append(split_start)
            self.sample_counts.append(count)
            self.raw_sample_counts.append(n_samples)
            self.cum_samples.append(self.cum_samples[-1] + count)
            self._token_count += (
                manifest_tokens
                if manifest_tokens is not None
                else self._selected_token_count(lengths, split_start, count, n_samples)
            )

        if self.cum_samples[-1] <= 0:
            raise ValueError(f"empty {split} online-pad dataset after split")
        if manifest_view is not None:
            if len(self) != int(manifest_view["records"]):
                raise ValueError("domain split manifest record total is inconsistent")
            if self._token_count != int(manifest_view["effective_input_tokens_with_eos"]):
                raise ValueError("domain split manifest token total is inconsistent")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.cum_samples[-1]

    def _selected_token_count(
        self,
        lengths: np.ndarray,
        split_start: int,
        count: int,
        n_samples: int,
    ) -> int:
        cap = self.pad_length - 1
        if self.split_strategy == "tail":
            selected = np.asarray(lengths[split_start: split_start + count], dtype=np.uint64)
        elif self.val_stride and self.split == "val":
            selected = np.asarray(lengths[0:n_samples:self.val_stride], dtype=np.uint64)
        elif self.val_stride and self.split == "train":
            all_tokens = int(np.minimum(np.asarray(lengths[:], dtype=np.uint64), cap).sum())
            val_tokens = int(np.minimum(np.asarray(lengths[0:n_samples:self.val_stride], dtype=np.uint64), cap).sum())
            return all_tokens - val_tokens + int(count)
        else:
            selected = np.asarray(lengths[:], dtype=np.uint64)
        return int(np.minimum(selected, cap).sum()) + int(count)

    def _raw_sample_index(self, file_idx: int, local_idx: int) -> int:
        raw_indices = self.raw_index_arrays[file_idx]
        if raw_indices is not None:
            return int(raw_indices[local_idx])
        if self.split_strategy == "tail":
            return self.split_starts[file_idx] + local_idx
        if self.val_stride and self.split == "val":
            return local_idx * self.val_stride
        if self.val_stride and self.split == "train":
            return local_idx + 1 + local_idx // (self.val_stride - 1)
        return local_idx

    def _rng_for(self, file_idx: int, raw_idx: int) -> np.random.Generator:
        key = self.seed
        key ^= (int(self.epoch) & 0xFFFFFFFF) << 32
        key ^= (int(file_idx) & 0xFFFF) << 48
        key ^= int(raw_idx) & 0xFFFFFFFF
        return np.random.default_rng(splitmix64(key))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = bisect.bisect_right(self.cum_samples, idx) - 1
        local_idx = idx - self.cum_samples[file_idx]
        raw_idx = self._raw_sample_index(file_idx, local_idx)
        length = int(self.length_arrays[file_idx][raw_idx])
        if length <= 0:
            raise RuntimeError(f"bad empty compact-int sample in {self.paths[file_idx]} at raw sample {raw_idx}")
        payload_offset = int(self.payload_offsets[file_idx][raw_idx])
        raw = self.byte_arrays[file_idx][payload_offset: payload_offset + length * 4]
        seq = np.frombuffer(raw, dtype="<i4", count=length).astype(np.int64, copy=False)
        max_body_len = self.pad_length - 1
        if self.unfolder is not None:
            unfold_cap = min(max_body_len, max(1, int(math.ceil(length * self.unfold_max_ratio))))
            body = self.unfolder.expand(seq, self._rng_for(file_idx, raw_idx), unfold_cap)
        else:
            body = seq.tolist()
        was_truncated = len(body) > max_body_len
        if was_truncated:
            if not self.truncate:
                raise RuntimeError(
                    f"online-pad sample length {len(body)} exceeds max body length {max_body_len}; "
                    "pass --online-truncate-long-samples to cut the tail"
                )
            body = body[:max_body_len]
        shifted = [int(x) + self.token_offset for x in body]
        shifted.append(self.eos_token_id)
        input_ids = torch.full((self.pad_length,), self.pad_token_id, dtype=torch.long)
        real_len = len(shifted)
        input_ids[:real_len] = torch.tensor(shifted, dtype=torch.long)
        labels = input_ids.clone()
        if real_len < self.pad_length:
            labels[real_len:] = -100
        attention_mask = torch.zeros((self.pad_length,), dtype=torch.long)
        attention_mask[:real_len] = 1
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "segment_lengths": torch.tensor([real_len], dtype=torch.long),
            "truncated": torch.tensor([1 if was_truncated else 0], dtype=torch.long),
            "sample_indices": torch.tensor([idx], dtype=torch.long),
        }
        domain_ids = self.domain_id_arrays[file_idx]
        if domain_ids is not None:
            item["domain_ids"] = torch.tensor([int(domain_ids[raw_idx])], dtype=torch.long)
        return item

    def token_count(self) -> int:
        return int(self._token_count)


class PackedU16SequenceDataset(Dataset):
    """读取固定长度 packed sequence，并用 sidecar 恢复每条样本段长度。"""

    def __init__(
        self,
        paths: Sequence[Path],
        segment_dir: Path,
        sequence_length: int,
        pad_token_id: int,
        split_name: str,
    ):
        if segment_dir is None:
            raise ValueError("--packed-segment-dir is required when --dataset-mode=packed-sequence")
        if not segment_dir.exists():
            raise FileNotFoundError(f"packed segment dir does not exist: {segment_dir}")
        if sequence_length < 2:
            raise ValueError("--packed-sequence-length must be >= 2")
        self.paths = list(paths)
        self.segment_dir = segment_dir
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)
        self.split_name = split_name
        self.arrays: List[np.memmap] = []
        self.segment_offsets: List[np.memmap] = []
        self.segment_lengths: List[np.memmap] = []
        self.sequence_counts: List[int] = []
        self.cum_sequences: List[int] = [0]
        self._token_count = 0

        for path in self.paths:
            n_tokens = path.stat().st_size // np.dtype(np.uint16).itemsize
            if n_tokens % self.sequence_length != 0:
                raise ValueError(
                    f"packed token file size is not divisible by sequence length: "
                    f"{path} tokens={n_tokens} sequence_length={self.sequence_length}"
                )
            n_sequences = n_tokens // self.sequence_length
            if n_sequences <= 0:
                continue

            stem = path.name
            if stem.endswith(".u16"):
                stem = stem[: -len(".u16")]
            offsets_path = segment_dir / split_name / f"{stem}.segment_offsets.u64"
            lengths_path = segment_dir / split_name / f"{stem}.segment_lengths.u16"
            missing = [str(p) for p in (offsets_path, lengths_path) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"missing packed segment sidecar for {path}: {missing}")

            arr = np.memmap(path, mode="r", dtype=np.uint16)
            offsets = np.memmap(offsets_path, mode="r", dtype=np.uint64)
            lengths = np.memmap(lengths_path, mode="r", dtype=np.uint16)
            if int(offsets.shape[0]) != int(n_sequences + 1):
                raise ValueError(
                    f"segment offset count mismatch for {path}: "
                    f"offsets={offsets.shape[0]} expected={n_sequences + 1}"
                )
            if int(offsets[-1]) > int(lengths.shape[0]):
                raise ValueError(
                    f"segment offsets exceed length sidecar for {path}: "
                    f"offset_last={int(offsets[-1])} lengths={int(lengths.shape[0])}"
                )

            self.arrays.append(arr)
            self.segment_offsets.append(offsets)
            self.segment_lengths.append(lengths)
            self.sequence_counts.append(int(n_sequences))
            self.cum_sequences.append(self.cum_sequences[-1] + int(n_sequences))
            self._token_count += int(np.asarray(lengths[:], dtype=np.uint64).sum())

        if self.cum_sequences[-1] <= 0:
            raise ValueError(f"empty packed sequence dataset for split={split_name}")

    def __len__(self) -> int:
        return self.cum_sequences[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = bisect.bisect_right(self.cum_sequences, idx) - 1
        local_idx = idx - self.cum_sequences[file_idx]
        start = local_idx * self.sequence_length
        end = start + self.sequence_length
        ids_np = np.asarray(self.arrays[file_idx][start:end], dtype=np.int64)
        if ids_np.shape[0] != self.sequence_length:
            raise RuntimeError("short packed sequence read; dataset accounting is inconsistent")
        off0 = int(self.segment_offsets[file_idx][local_idx])
        off1 = int(self.segment_offsets[file_idx][local_idx + 1])
        if off1 <= off0:
            raise RuntimeError(
                f"packed sequence has no segments: split={self.split_name} file={self.paths[file_idx]} idx={local_idx}"
            )
        seg_np = np.asarray(self.segment_lengths[file_idx][off0:off1], dtype=np.int64)
        valid_count = int(np.count_nonzero(ids_np != self.pad_token_id))
        if int(seg_np.sum()) != valid_count:
            raise RuntimeError(
                "packed segment length sum does not match non-PAD token count: "
                f"split={self.split_name} file={self.paths[file_idx]} idx={local_idx} "
                f"segments={int(seg_np.sum())} valid={valid_count}"
            )
        input_ids = torch.from_numpy(ids_np.copy()).long()
        attention_mask = input_ids.ne(self.pad_token_id).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "segment_lengths": torch.from_numpy(seg_np.copy()).long(),
        }

    def token_count(self) -> int:
        return int(self._token_count)


def collate_blocks(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "labels": torch.stack([x["labels"] for x in batch], dim=0),
    }


def eos_segment_lengths(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """用 EOS 把一个固定长度 token block 切成多个样本段，返回每段长度。"""
    if input_ids.ndim != 1:
        raise RuntimeError("eos_segment_lengths expects a 1D token block")
    eos_positions = torch.nonzero(input_ids.eq(int(eos_token_id)), as_tuple=False).flatten()
    lengths: list[int] = []
    start = 0
    for pos_tensor in eos_positions:
        end = int(pos_tensor.item()) + 1
        if end > start:
            lengths.append(end - start)
        start = end
    if start < int(input_ids.numel()):
        lengths.append(int(input_ids.numel()) - start)
    if not lengths:
        lengths.append(int(input_ids.numel()))
    return torch.tensor(lengths, dtype=torch.long)


def collate_packed_blocks(
    batch: Sequence[dict[str, torch.Tensor]],
    eos_token_id: int,
) -> dict[str, torch.Tensor]:
    """固定长度 block 训练用 collate：保留 block 形状，同时提供 block 内样本段长度。"""
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    segment_lengths = torch.cat([eos_segment_lengths(row, eos_token_id) for row in input_ids], dim=0)
    if int(segment_lengths.sum().item()) != int(input_ids.numel()):
        raise RuntimeError(
            "packed block segment accounting is inconsistent: "
            f"{int(segment_lengths.sum().item())} vs {int(input_ids.numel())}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_lengths": segment_lengths,
    }


def collate_samples(
    batch: Sequence[dict[str, torch.Tensor]],
    pad_token_id: int,
    sample_pad_length: int,
    pad_to_multiple_of: int,
) -> dict[str, torch.Tensor]:
    """普通 HF sample 训练用 collate：右侧补 PAD，并用 attention_mask/labels 屏蔽 PAD。"""
    if not batch:
        raise ValueError("empty batch")
    lengths = [int(x["input_ids"].numel()) for x in batch]
    max_len = max(lengths)
    if sample_pad_length > 0:
        if max_len > sample_pad_length:
            raise RuntimeError(
                f"batch has sample length {max_len} > --sample-pad-length={sample_pad_length}; "
                "increase the fixed pad length"
            )
        out_len = sample_pad_length
    else:
        out_len = max_len
        if pad_to_multiple_of > 1:
            out_len = ((out_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    input_ids = torch.full((len(batch), out_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), out_len), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        n = int(ids.numel())
        input_ids[i, :n] = ids
        attention_mask[i, :n] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_packed_samples(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """packed sample 训练用 collate：不补 PAD，直接把一批完整样本拼成一条有效 token 串。"""
    if not batch:
        raise ValueError("empty batch")
    lengths = torch.tensor([int(x["input_ids"].numel()) for x in batch], dtype=torch.long)
    if bool((lengths <= 0).any()):
        raise RuntimeError("packed sample batch contains an empty sample")
    input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).unsqueeze(0).long()
    labels = input_ids.clone()
    if int(lengths.sum().item()) != int(input_ids.numel()):
        raise RuntimeError(
            "packed sample segment accounting is inconsistent: "
            f"{int(lengths.sum().item())} vs {int(input_ids.numel())}"
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "segment_lengths": lengths,
    }


def collate_packed_sequences(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """固定长度 packed sequence 训练用 collate：batch size 是 packed sequence 数量。"""
    if not batch:
        raise ValueError("empty batch")
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    segment_lengths = torch.cat([x["segment_lengths"] for x in batch], dim=0)
    if int(segment_lengths.sum().item()) != int(attention_mask.sum().item()):
        raise RuntimeError(
            "packed sequence segment accounting is inconsistent: "
            f"{int(segment_lengths.sum().item())} vs {int(attention_mask.sum().item())}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_lengths": segment_lengths,
    }


def collate_online_padded_samples(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """在线增强样本的 collate：样本已 pad 到固定长度，segment_lengths 保留真实边界。"""
    if not batch:
        raise ValueError("empty batch")
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    segment_lengths = torch.cat([x["segment_lengths"] for x in batch], dim=0)
    truncated = torch.cat([x["truncated"] for x in batch], dim=0)
    if int(segment_lengths.sum().item()) != int(attention_mask.sum().item()):
        raise RuntimeError(
            "online padded sample segment accounting is inconsistent: "
            f"{int(segment_lengths.sum().item())} vs {int(attention_mask.sum().item())}"
        )
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_lengths": segment_lengths,
        "truncated": truncated,
        "sample_indices": torch.cat([x["sample_indices"] for x in batch], dim=0),
    }
    if "domain_ids" in batch[0]:
        out["domain_ids"] = torch.cat([x["domain_ids"] for x in batch], dim=0)
    return out


class PackedXformersQwen3Attention(torch.nn.Module):
    """Qwen3 attention on packed valid tokens, using xFormers block-diagonal causal attention."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.hidden_size = int(config.hidden_size)
        self.attention_dropout = float(config.attention_dropout)
        self.q_proj = torch.nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = torch.nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_bias: object,
    ) -> torch.Tensor:
        if xops is None:
            raise RuntimeError(
                "packed-xformers model requires a working xformers import; "
                f"import_error={_XFORMERS_IMPORT_ERROR}"
            )
        total_tokens = hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(1, total_tokens, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(1, total_tokens, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(1, total_tokens, self.num_key_value_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_packed(q, k, cos, sin)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if self.num_key_value_groups != 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)

        context = xops.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attention_bias,
            p=self.attention_dropout if self.training else 0.0,
        )
        context = context.reshape(1, total_tokens, self.num_attention_heads * self.head_dim).contiguous()
        return self.o_proj(context)


class PackedXformersQwen3DecoderLayer(torch.nn.Module):
    """Qwen3 decoder layer that keeps only valid tokens in the MLP/attention stream."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = PackedXformersQwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_bias: object,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_bias)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class PackedXformersQwen3ForCausalLM(torch.nn.Module):
    """Qwen3-style causal LM with GTE-style unpadding and block-diagonal causal attention."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        if xops is None:
            raise RuntimeError(
                "packed-xformers model requires a working xformers import; "
                f"import_error={_XFORMERS_IMPORT_ERROR}"
            )
        fmha = getattr(xops, "fmha", None)
        if fmha is None:
            raise RuntimeError("xformers is installed but xops.fmha is unavailable")
        attn_bias_mod = fmha.attn_bias
        if not hasattr(attn_bias_mod, "BlockDiagonalCausalMask"):
            raise RuntimeError("xformers is installed but BlockDiagonalCausalMask is unavailable")
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [PackedXformersQwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if bool(config.tie_word_embeddings):
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def save_pretrained(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(out_dir)
        torch.save(self.state_dict(), out_dir / "pytorch_model.bin")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        segment_lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        if attention_mask is None:
            if segment_lengths is None:
                attention_mask = input_ids.ne(int(self.config.pad_token_id))
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        valid_input_ids = input_ids[attention_mask].unsqueeze(0)
        total_tokens = int(valid_input_ids.shape[1])
        if total_tokens <= 0:
            raise RuntimeError("packed-xformers received an empty batch")

        if segment_lengths is None:
            lengths_tensor = attention_mask.long().sum(dim=1)
            if int(lengths_tensor.min().item()) <= 0:
                raise RuntimeError("packed-xformers does not support empty samples in a batch")
            segment_lengths_for_device = lengths_tensor.to(device=input_ids.device, dtype=torch.long)
        else:
            segment_lengths_for_device = segment_lengths.to(device=input_ids.device, dtype=torch.long).flatten()
            if segment_lengths_for_device.ndim != 1:
                raise RuntimeError("segment_lengths must be a 1D tensor")
            if int(segment_lengths_for_device.numel()) <= 0:
                raise RuntimeError("segment_lengths is empty")
            if bool((segment_lengths_for_device <= 0).any()):
                raise RuntimeError("segment_lengths contains non-positive length")
            if int(segment_lengths_for_device.sum().item()) != total_tokens:
                raise RuntimeError(
                    "segment_lengths sum does not match valid token count: "
                    f"{int(segment_lengths_for_device.sum().item())} vs {total_tokens}"
                )
        lengths = [int(x) for x in segment_lengths_for_device.detach().cpu().tolist()]

        position_ids = build_packed_position_ids(segment_lengths_for_device)
        hidden_states = self.embed_tokens(valid_input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(lengths)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_bias)
        hidden_states = self.norm(hidden_states)

        loss = None
        logits = None
        if labels is not None:
            if segment_lengths is None:
                targets = build_next_token_labels(labels, attention_mask)
            else:
                targets = build_next_token_labels_from_segments(labels, attention_mask, segment_lengths_for_device)
            loss_positions = targets.ne(-100)
            if not bool(loss_positions.any()):
                raise RuntimeError("batch contains no causal LM targets")
            selected_hidden = hidden_states.squeeze(0)[loss_positions]
            selected_targets = targets[loss_positions].to(selected_hidden.device)
            logits = self.lm_head(selected_hidden)
            loss = torch.nn.functional.cross_entropy(logits.float(), selected_targets)
        else:
            logits = self.lm_head(hidden_states[:, :total_tokens, :])

        return CausalLMOutputWithPast(loss=loss, logits=logits)


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return world_size > 1, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        hidden_act="silu",
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=args.num_hidden_layers,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
    )
    set_attention_implementation(config, args.attn_implementation)
    return config


def set_attention_implementation(config: Qwen3Config, attn_implementation: str) -> None:
    if attn_implementation == "auto":
        return
    try:
        config._attn_implementation = attn_implementation
    except Exception:
        setattr(config, "_attn_implementation_internal", attn_implementation)


def resolve_attention_backend(
    model_implementation: str,
    config: Qwen3Config,
    requested_implementation: str,
) -> str:
    """Report the implementation that actually executes attention.

    ``_attn_implementation`` remains a useful Transformers config tag, but the
    custom packed model does not dispatch through it: it directly calls
    xFormers with a block-diagonal causal bias.
    """

    if model_implementation == "packed-xformers":
        return "xformers_memory_efficient_block_diagonal_causal"
    tag = str(
        getattr(
            config,
            "_attn_implementation",
            getattr(config, "_attn_implementation_internal", requested_implementation),
        )
    )
    return {
        "eager": "transformers_eager_attention",
        "sdpa": "torch_scaled_dot_product_attention",
        "flash_attention_2": "flash_attention_2",
        "flex_attention": "torch_flex_attention",
    }.get(tag, f"transformers_attention_{tag}")


def configure_sdpa_kernel(mode: str) -> dict[str, Any]:
    """选择 PyTorch 原生 scaled_dot_product_attention 可使用的 CUDA kernel。"""
    cuda_backend = getattr(torch.backends, "cuda", None)
    status: dict[str, Any] = {"requested": mode, "configured": False}
    if cuda_backend is None or not torch.cuda.is_available():
        status["reason"] = "cuda backend unavailable"
        return status

    setters = {
        "flash": getattr(cuda_backend, "enable_flash_sdp", None),
        "mem_efficient": getattr(cuda_backend, "enable_mem_efficient_sdp", None),
        "math": getattr(cuda_backend, "enable_math_sdp", None),
    }
    getters = {
        "flash": getattr(cuda_backend, "flash_sdp_enabled", None),
        "mem_efficient": getattr(cuda_backend, "mem_efficient_sdp_enabled", None),
        "math": getattr(cuda_backend, "math_sdp_enabled", None),
    }
    if any(func is None for func in setters.values()):
        status["reason"] = "sdpa kernel setters unavailable"
        return status

    if mode == "auto":
        pass
    elif mode == "flash":
        setters["flash"](True)
        setters["mem_efficient"](False)
        setters["math"](False)
    elif mode == "mem_efficient":
        setters["flash"](False)
        setters["mem_efficient"](True)
        setters["math"](False)
    elif mode == "math":
        setters["flash"](False)
        setters["mem_efficient"](False)
        setters["math"](True)
    else:
        raise ValueError(f"unknown --sdpa-kernel: {mode}")

    status["configured"] = True
    status["enabled"] = {name: bool(func()) for name, func in getters.items() if func is not None}
    return status


def build_model(args: argparse.Namespace) -> tuple[Qwen3ForCausalLM, Qwen3Config, str]:
    if args.model_implementation == "packed-xformers":
        if args.init_checkpoint is not None:
            raise RuntimeError("--model-implementation=packed-xformers does not support --init-checkpoint yet")
        config = build_config(args)
        return PackedXformersQwen3ForCausalLM(config), config, "random initialization; packed-xformers implementation"
    if args.init_checkpoint is not None:
        config = Qwen3Config.from_pretrained(args.init_checkpoint)
        config.use_cache = False
        set_attention_implementation(config, args.attn_implementation)
        if int(config.vocab_size) != int(args.vocab_size):
            raise ValueError(f"--vocab-size={args.vocab_size} does not match checkpoint vocab_size={config.vocab_size}")
        from moltoken.checkpoints import load_pretrained
        model = load_pretrained(args.init_checkpoint, attention=config._attn_implementation)
        config = model.config
        config.use_cache = False
        return model, config, f"loaded from {args.init_checkpoint}"
    config = build_config(args)
    return Qwen3ForCausalLM(config), config, "random initialization"


def describe_architecture(config: Qwen3Config, init_source: str, param_count: int) -> str:
    """把当前实际训练的 Qwen3-style 架构写成人能读懂的 run_config 说明。"""

    return (
        "Qwen3-style causal LM: "
        f"vocab_size={config.vocab_size}, "
        f"hidden_size={config.hidden_size}, "
        f"intermediate_size={config.intermediate_size}, "
        f"num_hidden_layers={config.num_hidden_layers}, "
        f"num_attention_heads={config.num_attention_heads}, "
        f"num_key_value_heads={config.num_key_value_heads}, "
        f"head_dim={getattr(config, 'head_dim', None)}, "
        f"rope_theta={config.rope_theta}, "
        f"attention_dropout={config.attention_dropout}, "
        f"rms_norm_eps={config.rms_norm_eps}, "
        f"tie_word_embeddings={config.tie_word_embeddings}; "
        f"{init_source}; "
        f"parameter_count={param_count}; "
        "no Qwen checkpoint is loaded unless --init-checkpoint is explicitly set."
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    raw = model.module if hasattr(model, "module") else model
    return raw._orig_mod if hasattr(raw, "_orig_mod") else raw


def save_model(model: torch.nn.Module, out_dir: Path, rank: int) -> None:
    if rank != 0:
        return
    raw = unwrap_model(model)
    raw.save_pretrained(out_dir)


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = model.parameters()
    kwargs = {
        "lr": args.lr,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_eps,
        "weight_decay": args.weight_decay,
    }
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if args.optimizer == "adamw-fused":
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except TypeError as exc:
            raise RuntimeError("torch.optim.AdamW(fused=True) is not available in this PyTorch build") from exc
    if args.optimizer == "deepspeed-fused-adam":
        try:
            from deepspeed.ops.adam import FusedAdam
        except Exception as exc:
            raise RuntimeError("optimizer=deepspeed-fused-adam requires deepspeed to be installed") from exc
        return FusedAdam(params, **kwargs)
    raise ValueError(f"unknown optimizer: {args.optimizer}")


def build_deepspeed_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.deepspeed_config is not None:
        with args.deepspeed_config.open("r", encoding="utf-8") as f:
            return json.load(f)
    zero_stage = int(args.deepspeed_zero_stage)
    if zero_stage not in {0, 1, 2}:
        raise ValueError("--deepspeed-zero-stage must be 0, 1, or 2 for this save path")
    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": bool(args.bf16)},
        "zero_optimization": {"stage": zero_stage},
        "gradient_clipping": float(args.max_grad_norm),
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
    }
    return config


def set_scheduler_step(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    base_lr: float,
    step: int,
    lr_lambda: Any,
) -> None:
    if step <= 0:
        return
    scale = float(lr_lambda(step))
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", base_lr)
        group["lr"] = float(group["initial_lr"]) * scale
    scheduler.last_epoch = int(step)
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    use_bf16: bool,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    token_sum = 0
    input_token_sum = 0
    sample_sum = 0
    batch_count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            input_token_sum += count_input_tokens(batch)
            sample_sum += (
                int(batch["segment_lengths"].numel())
                if "segment_lengths" in batch
                else int(batch["input_ids"].shape[0])
            )
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                out = model(**model_inputs(batch))
            tokens = count_loss_tokens(batch)
            batch_loss = float(out.loss.detach().cpu())
            if not math.isfinite(batch_loss):
                raise FloatingPointError(f"non-finite validation loss at batch={i}: {batch_loss}")
            loss_sum += batch_loss * tokens
            token_sum += tokens
            batch_count += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "batches": batch_count,
                        "tokens": token_sum,
                        "input_tokens": input_token_sum,
                        "samples": sample_sum,
                    }
                )
    model.train()
    loss = loss_sum / max(token_sum, 1)
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "tokens": token_sum,
        "input_tokens": input_token_sum,
        "samples": sample_sum,
        "batches": batch_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a Qwen3-style causal LM on FDDFS integer-BPE streams."
    )
    ap.add_argument("--token-files", nargs="+", required=True, help="Glob(s) or paths for train uint16 token streams or compact-int shards.")
    ap.add_argument("--val-token-files", nargs="+", default=None, help="Glob(s) or paths for validation uint16 token streams / compact-int shards; defaults to --token-files.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--dataset-mode", choices=["block", "packed-block", "sample", "packed-sequence", "online-pad"], default="block")
    ap.add_argument("--model-implementation", choices=["hf", "packed-xformers"], default="hf")
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--packed-segment-dir", type=Path, default=None, help="Root directory containing packed segment sidecars for train/val/test splits.")
    ap.add_argument("--packed-sequence-length", type=int, default=512, help="Fixed packed sequence length for --dataset-mode=packed-sequence.")
    ap.add_argument("--sample-index-dir", type=Path, default=None)
    ap.add_argument("--sample-max-len", type=int, default=0, help="Maximum unpadded sample length after optional BOS; 0 disables this check.")
    ap.add_argument("--sample-pad-length", type=int, default=0, help="HF sample mode only: pad every sample batch to this fixed length; 0 pads to batch max.")
    ap.add_argument("--sample-pad-to-multiple-of", type=int, default=8, help="HF sample mode only: when --sample-pad-length=0, round batch length up to this multiple.")
    ap.add_argument("--sample-add-bos", action="store_true", default=False, help="Prepend BOS to every EOS-delimited sample.")
    ap.add_argument("--compact-index-dir", type=Path, default=None, help="Index dir for compact-int shards used by --dataset-mode=online-pad.")
    ap.add_argument("--compact-domain-id-dir", type=Path, default=None, help="Optional sidecar dir containing *.domain_ids.u16 files for compact-int shards.")
    ap.add_argument("--domain-split-manifest", type=Path, default=None, help="Frozen domain_split_raw_indices_v1 manifest for domain-wise online-pad training.")
    ap.add_argument("--domain-name", default=None, help="Domain view to select from --domain-split-manifest.")
    ap.add_argument("--domain-names", default="small_molecule,peptide,protein,dna,rna", help="Comma-separated domain names, ordered by domain id.")
    ap.add_argument("--log-domain-loss", action="store_true", help="Log per-domain sample/token counts and training loss without changing the optimization objective.")
    ap.add_argument("--token-offset", type=int, default=3, help="Offset added to raw BPE token ids before feeding the model.")
    ap.add_argument("--online-pad-length", type=int, default=512, help="Fixed pad length for --dataset-mode=online-pad.")
    ap.add_argument("--online-truncate-long-samples", action="store_true", help="Cut online samples to online-pad-length-1 before appending EOS.")
    ap.add_argument("--bpe-merges-csv", type=Path, default=None, help="BPE merges.csv for online random reverse-merge augmentation.")
    ap.add_argument("--bpe-unfold-policy", choices=["none", "constant"], default="none", help="Sampling policy for online BPE reverse expansion.")
    ap.add_argument("--bpe-unfold-prob", type=float, default=0.0, help="Constant per-token unfold probability for --bpe-unfold-policy=constant.")
    ap.add_argument("--bpe-unfold-max-depth", type=int, default=8, help="Maximum recursive BPE unfold depth.")
    ap.add_argument("--bpe-unfold-max-ratio", type=float, default=4.0, help="Cap unfolded body length at min(pad_length-1, ceil(original_len * ratio)).")
    ap.set_defaults(online_shuffle=True)
    ap.add_argument("--online-shuffle", dest="online_shuffle", action="store_true", help="Shuffle online-pad samples with an O(1)-memory deterministic permutation sampler.")
    ap.add_argument("--no-online-shuffle", dest="online_shuffle", action="store_false", help="Disable online-pad shuffle and use sequential DDP striding.")
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--split-strategy", choices=["stride", "tail"], default="stride")
    ap.add_argument("--max-position-embeddings", type=int, default=40960)
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--intermediate-size", type=int, default=3072)
    ap.add_argument("--num-hidden-layers", type=int, default=18)
    ap.add_argument("--num-attention-heads", type=int, default=16)
    ap.add_argument("--num-key-value-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--pad-token-id", type=int, default=0)
    ap.add_argument("--bos-token-id", type=int, default=1)
    ap.add_argument("--eos-token-id", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument(
        "--target-input-tokens",
        type=int,
        default=0,
        help=(
            "Authoritative non-PAD input-token budget, including EOS.  When positive, "
            "training stops after the first whole record that reaches this budget; "
            "--max-steps remains the LR-scheduler horizon."
        ),
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.95)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--final-eval", action="store_true", help="Run validation once after the token/step budget is reached.")
    ap.add_argument(
        "--final-eval-batches",
        type=int,
        default=0,
        help="Maximum batches for --final-eval; 0 evaluates the complete frozen validation view.",
    )
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", action="store_false", dest="bf16")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    ap.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    ap.add_argument("--sdpa-kernel", choices=["auto", "flash", "mem_efficient", "math"], default="auto")
    ap.add_argument("--optimizer", choices=["adamw", "adamw-fused", "deepspeed-fused-adam"], default="adamw")
    ap.add_argument("--torch-compile", action="store_true", help="Use torch.compile on the model before DDP wrapping.")
    ap.add_argument("--torch-compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed engine instead of plain DDP.")
    ap.add_argument("--deepspeed-zero-stage", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--deepspeed-config", type=Path, default=None)
    ap.add_argument("--init-checkpoint", type=Path, default=None, help="Load model weights from a HF checkpoint before training.")
    ap.add_argument(
        "--resume-global-step",
        type=int,
        default=0,
        help="Deprecated safety guard; nonzero is rejected because full-state resume is not implemented.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Build datasets/model, write config, then exit before training.")
    args = ap.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.target_input_tokens < 0:
        raise ValueError("--target-input-tokens must be non-negative")
    if args.final_eval_batches < 0:
        raise ValueError("--final-eval-batches must be non-negative")
    if args.resume_global_step != 0:
        raise ValueError(
            "--resume-global-step is disabled: it only changed counters/LR and never restored model, "
            "optimizer, RNG, or data position. Restart this run from step 0 until a full-state resume "
            "checkpoint is implemented."
        )
    if args.target_input_tokens > 0 and args.dataset_mode != "online-pad":
        raise ValueError("--target-input-tokens currently requires --dataset-mode=online-pad")
    if args.target_input_tokens > 0 and args.deepspeed:
        raise ValueError(
            "--target-input-tokens currently requires the native optimizer path so the final short "
            "gradient window can be normalized exactly by its loss-token count"
        )
    domain_names = parse_domain_names(args.domain_names)
    num_domains = len(domain_names)
    if args.log_domain_loss:
        if args.dataset_mode != "online-pad":
            raise ValueError("--log-domain-loss currently supports --dataset-mode=online-pad")
        if args.model_implementation != "packed-xformers":
            raise ValueError("--log-domain-loss currently expects --model-implementation=packed-xformers")
        if args.compact_domain_id_dir is None:
            raise ValueError("--compact-domain-id-dir is required with --log-domain-loss")
    if (args.domain_split_manifest is None) != (args.domain_name is None):
        raise ValueError("--domain-split-manifest and --domain-name must be supplied together")
    if args.domain_split_manifest is not None and args.dataset_mode != "online-pad":
        raise ValueError("--domain-split-manifest currently requires --dataset-mode=online-pad")

    ddp, rank, local_rank, world_size = setup_distributed()
    try:
        if args.target_input_tokens > 0 and world_size != 1:
            raise RuntimeError(
                "exact --target-input-tokens stopping currently supports one process per run; "
                "launch independent single-GPU scaling runs instead of DDP"
            )
        set_seed(args.seed, rank)
        is_main = rank == 0
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        sdpa_status = configure_sdpa_kernel(args.sdpa_kernel)
        token_paths = expand_paths(args.token_files)
        val_token_paths = expand_paths(args.val_token_files) if args.val_token_files is not None else token_paths
        # online-pad stores exactly one molecule per batch row, so ordinary HF
        # causal attention also isolates samples. Concatenated packed modes do
        # require the block-diagonal xFormers backend.
        if args.dataset_mode in {"packed-block", "packed-sequence"} and args.model_implementation != "packed-xformers":
            raise RuntimeError(f"--dataset-mode={args.dataset_mode} requires --model-implementation=packed-xformers")
        if args.dataset_mode == "block":
            train_ds = U16TokenBlockDataset(token_paths, args.block_size, "train", args.val_fraction, args.split_strategy)
            val_ds = U16TokenBlockDataset(val_token_paths, args.block_size, "val", args.val_fraction, args.split_strategy)
            data_collate = collate_blocks
        elif args.dataset_mode == "packed-block":
            train_ds = U16TokenBlockDataset(token_paths, args.block_size, "train", args.val_fraction, args.split_strategy)
            val_ds = U16TokenBlockDataset(val_token_paths, args.block_size, "val", args.val_fraction, args.split_strategy)
            data_collate = partial(collate_packed_blocks, eos_token_id=args.eos_token_id)
        elif args.dataset_mode == "packed-sequence":
            train_ds = PackedU16SequenceDataset(
                token_paths,
                args.packed_segment_dir,
                args.packed_sequence_length,
                args.pad_token_id,
                "train",
            )
            val_ds = PackedU16SequenceDataset(
                val_token_paths,
                args.packed_segment_dir,
                args.packed_sequence_length,
                args.pad_token_id,
                "val",
            )
            data_collate = collate_packed_sequences
        elif args.dataset_mode == "online-pad":
            if args.bpe_unfold_policy != "none" or args.bpe_unfold_prob > 0.0:
                if args.bpe_merges_csv is None:
                    raise ValueError("--bpe-merges-csv is required when online BPE unfolding is enabled")
                train_unfold_max_len = max(1, args.online_pad_length - 1)
                train_unfolder = BpeUnfolder.from_csv(
                    args.bpe_merges_csv,
                    args.bpe_unfold_policy,
                    args.bpe_unfold_prob,
                    args.bpe_unfold_max_depth,
                    train_unfold_max_len,
                )
            else:
                train_unfolder = None
            train_ds = CompactIntOnlinePadDataset(
                token_paths,
                args.compact_index_dir,
                "train",
                args.val_fraction,
                args.split_strategy,
                args.token_offset,
                args.eos_token_id,
                args.pad_token_id,
                args.online_pad_length,
                args.online_truncate_long_samples,
                args.seed,
                args.bpe_unfold_max_ratio,
                train_unfolder,
                args.compact_domain_id_dir,
                args.domain_split_manifest,
                args.domain_name,
            )
            val_ds = CompactIntOnlinePadDataset(
                val_token_paths,
                args.compact_index_dir,
                "val",
                args.val_fraction,
                args.split_strategy,
                args.token_offset,
                args.eos_token_id,
                args.pad_token_id,
                args.online_pad_length,
                True,
                args.seed,
                args.bpe_unfold_max_ratio,
                None,
                args.compact_domain_id_dir,
                args.domain_split_manifest,
                args.domain_name,
            )
            data_collate = collate_online_padded_samples
        else:
            train_ds = U16EosSampleDataset(
                token_paths,
                args.sample_index_dir,
                "train",
                args.val_fraction,
                args.split_strategy,
                args.eos_token_id,
                args.bos_token_id,
                args.sample_add_bos,
                args.sample_max_len,
            )
            val_ds = U16EosSampleDataset(
                token_paths,
                args.sample_index_dir,
                "val",
                args.val_fraction,
                args.split_strategy,
                args.eos_token_id,
                args.bos_token_id,
                args.sample_add_bos,
                args.sample_max_len,
            )
            if args.model_implementation == "packed-xformers":
                data_collate = collate_packed_samples
            else:
                data_collate = partial(
                    collate_samples,
                    pad_token_id=args.pad_token_id,
                    sample_pad_length=args.sample_pad_length,
                    pad_to_multiple_of=args.sample_pad_to_multiple_of,
                )
        if args.dataset_mode == "online-pad":
            if args.online_shuffle:
                # online-pad 数据按 shard 顺序存放；单进程训练也必须显式打乱，
                # 否则 DataLoader(shuffle=False) 会顺序读取，导致前后训练分布偏移。
                train_sampler = DistributedAffineShuffleSampler(train_ds, num_replicas=world_size, rank=rank, seed=args.seed)
            else:
                train_sampler = DistributedStrideSampler(train_ds, num_replicas=world_size, rank=rank) if ddp else None
        elif ddp:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        else:
            train_sampler = None
        train_sampler_name = type(train_sampler).__name__ if train_sampler is not None else ("DataLoaderShuffle" if args.dataset_mode != "online-pad" else "None")
        collate_mode = (
            "packed_blocks_by_eos"
            if args.dataset_mode == "packed-block"
            else (
                "fixed_packed_sequences_no_pad"
                if args.dataset_mode == "packed-sequence"
                else (
                    "online_padded_samples_with_optional_bpe_unfold"
                    if args.dataset_mode == "online-pad"
                    else ("packed_samples_no_pad" if args.dataset_mode == "sample" and args.model_implementation == "packed-xformers" else "hf_padded")
                )
            )
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None and args.dataset_mode != "online-pad"),
            sampler=train_sampler,
            collate_fn=data_collate,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            # Domain-wise token budgets may cross an epoch boundary.  Dropping
            # this batch would silently omit the same tail records every epoch
            # and make the audited per-epoch token count false.
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        config = build_config(args) if args.init_checkpoint is None else Qwen3Config.from_pretrained(args.init_checkpoint)
        set_attention_implementation(config, args.attn_implementation)
        model, config, init_source = build_model(args)
        if args.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            else:
                raise RuntimeError("--gradient-checkpointing is not supported by this model implementation")
        model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        attention_backend = resolve_attention_backend(
            args.model_implementation,
            config,
            args.attn_implementation,
        )
        compile_status = {"enabled": False}
        if args.torch_compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("--torch-compile requires torch.compile, but this PyTorch build does not expose it")
            model = torch.compile(model, mode=args.torch_compile_mode)
            compile_status = {"enabled": True, "mode": args.torch_compile_mode}
        if ddp and not args.deepspeed:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        if is_main:
            write_json(args.out_dir / "run_config.json", {
                "args": jsonable_args(args),
                "token_files": [str(p) for p in token_paths],
                "val_token_files": [str(p) for p in val_token_paths],
                "num_token_files": len(token_paths),
                "num_val_token_files": len(val_token_paths),
                "dataset_mode": args.dataset_mode,
                "model_implementation": args.model_implementation,
                "collate_mode": collate_mode,
                "domain_loss_logging": {
                    "enabled": bool(args.log_domain_loss),
                    "domain_names": domain_names,
                    "compact_domain_id_dir": str(args.compact_domain_id_dir) if args.compact_domain_id_dir else None,
                },
                "domain_split_view": {
                    "manifest": str(args.domain_split_manifest) if args.domain_split_manifest else None,
                    "domain": args.domain_name,
                    "train_split": "train" if args.domain_split_manifest else None,
                    "validation_split": "validation" if args.domain_split_manifest else None,
                },
                "train_items": len(train_ds),
                "val_items": len(val_ds),
                "train_tokens_visible": train_ds.token_count(),
                "val_tokens_visible": val_ds.token_count(),
                "world_size": world_size,
                "training_engine": "deepspeed" if args.deepspeed else ("ddp" if ddp else "single_process"),
                "train_sampler": train_sampler_name,
                "attention_implementation": getattr(config, "_attn_implementation", args.attn_implementation),
                "attention_backend": attention_backend,
                "sdpa_kernel": sdpa_status,
                "optimizer": args.optimizer,
                "torch_compile": compile_status,
                "init_source": init_source,
                "parameter_count": param_count,
                "trainable_parameter_count": trainable_count,
                "architecture_source": describe_architecture(config, init_source, param_count),
                "effective_config": config.to_dict(),
            })
            print(json.dumps({
                "event": "model_built",
                "params": param_count,
                "dataset_mode": args.dataset_mode,
                "model_implementation": args.model_implementation,
                "collate_mode": collate_mode,
                "domain_loss_logging": bool(args.log_domain_loss),
                "train_items": len(train_ds),
                "val_items": len(val_ds),
                "engine": "deepspeed" if args.deepspeed else ("ddp" if ddp else "single_process"),
                "train_sampler": train_sampler_name,
                "attn": getattr(config, "_attn_implementation", args.attn_implementation),
                "attention_backend": attention_backend,
                "sdpa_kernel": sdpa_status,
                "optimizer": args.optimizer,
                "torch_compile": compile_status,
                "init_source": init_source,
            }, ensure_ascii=False), flush=True)
        if args.dry_run:
            return

        optimizer = build_optimizer(model, args)
        warmup_steps = max(1, int(round(args.max_steps * args.warmup_ratio)))
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        set_scheduler_step(optimizer, scheduler, args.lr, args.resume_global_step, lr_lambda)

        if args.deepspeed:
            try:
                import deepspeed
            except Exception as exc:
                raise RuntimeError("--deepspeed requires deepspeed to be installed in the worker environment") from exc
            ds_config = build_deepspeed_config(args)
            ds_kwargs = {
                "model": model,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "config": ds_config,
            }
            if dist.is_initialized():
                ds_kwargs["dist_init_required"] = False
            model, optimizer, _, scheduler = deepspeed.initialize(**ds_kwargs)

        log_path = args.out_dir / "train_log.jsonl"
        global_step = 0
        micro_step = 0
        micro_steps_in_update = 0
        accumulated_loss_tokens = 0
        running_loss_sum = 0.0
        running_count = 0
        running_loss_tokens = 0
        running_input_tokens = 0
        running_truncated_samples = 0
        running_domain_stats = new_domain_stats(num_domains) if args.log_domain_loss else None
        last_train_loss = None
        last_train_loss_tokens = 0
        total_train_loss_weighted = 0.0
        total_train_loss_tokens = 0
        total_train_input_tokens = 0
        total_train_samples = 0
        final_update_micro_steps = 0
        final_update_loss_tokens = 0
        start = time.time()
        last_log_time = start
        if args.deepspeed:
            model.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        epoch = 0

        def training_budget_reached() -> bool:
            if args.target_input_tokens > 0:
                return total_train_input_tokens >= args.target_input_tokens
            return global_step >= args.max_steps

        while not training_budget_reached():
            epoch += 1
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            for batch in train_loader:
                if args.target_input_tokens > 0:
                    remaining_input_tokens = args.target_input_tokens - total_train_input_tokens
                    if remaining_input_tokens <= 0:
                        break
                    batch, _ = trim_sample_aligned_batch_to_token_budget(batch, remaining_input_tokens)
                batch_input_tokens = count_input_tokens(batch)
                batch_loss_tokens = count_loss_tokens(batch)
                if batch_input_tokens <= 0:
                    raise RuntimeError("training batch contains no non-PAD input tokens")
                if batch_loss_tokens <= 0:
                    raise RuntimeError("training batch contains no causal LM loss tokens")
                batch_samples = (
                    int(batch["segment_lengths"].numel())
                    if "segment_lengths" in batch
                    else int(batch["input_ids"].shape[0])
                )
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                    out = model(**model_inputs(batch))
                    if out.loss is None:
                        raise RuntimeError("model returned no training loss")
                    # The model returns a mean over this microbatch's loss
                    # tokens.  Backpropagate its token sum, then divide the
                    # accumulated gradients by the complete update window's
                    # token count immediately before optimizer.step().  This
                    # makes variable-length and final-short microbatches exact
                    # token-micro NLL rather than equally weighted batch means.
                    backward_loss = (
                        out.loss / args.grad_accum
                        if args.deepspeed
                        else out.loss * float(batch_loss_tokens)
                    )
                batch_truncated = int(batch["truncated"].sum().item()) if "truncated" in batch else 0
                if args.log_domain_loss:
                    if out.logits is None:
                        raise RuntimeError("--log-domain-loss requires logits in model output")
                    add_domain_stats(running_domain_stats, compute_domain_loss_stats(batch, out.logits, num_domains))
                if args.deepspeed:
                    model.backward(backward_loss)
                else:
                    backward_loss.backward()
                detached_loss = float(out.loss.detach().cpu())
                if not math.isfinite(detached_loss):
                    raise FloatingPointError(
                        f"non-finite training loss at micro_step={micro_step + 1}: {detached_loss}"
                    )
                running_loss_sum += detached_loss * batch_loss_tokens
                running_count += 1
                running_loss_tokens += batch_loss_tokens
                running_input_tokens += batch_input_tokens
                running_truncated_samples += batch_truncated
                total_train_loss_weighted += detached_loss * batch_loss_tokens
                total_train_loss_tokens += batch_loss_tokens
                total_train_input_tokens += batch_input_tokens
                total_train_samples += batch_samples
                micro_step += 1
                micro_steps_in_update += 1
                accumulated_loss_tokens += batch_loss_tokens
                reached_target = (
                    args.target_input_tokens > 0
                    and total_train_input_tokens >= args.target_input_tokens
                )
                if not optimizer_step_due(micro_steps_in_update, args.grad_accum, reached_target):
                    continue
                if args.deepspeed:
                    model.step()
                else:
                    gradient_denominator = float(accumulated_loss_tokens)
                    if ddp:
                        denominator_tensor = torch.tensor(
                            gradient_denominator,
                            dtype=torch.float64,
                            device=device,
                        )
                        dist.all_reduce(denominator_tensor, op=dist.ReduceOp.SUM)
                        # DDP gradients are averaged across ranks.  Dividing by
                        # global_tokens/world_size therefore yields the global
                        # per-token mean gradient.
                        gradient_denominator = float(denominator_tensor.item()) / world_size
                    scale_gradients_for_token_mean(model, gradient_denominator)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    grad_norm_value = float(grad_norm.detach().cpu())
                    if not math.isfinite(grad_norm_value):
                        raise FloatingPointError(
                            f"non-finite gradient norm before optimizer step={global_step + 1}: "
                            f"{grad_norm_value}"
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1
                final_update_micro_steps = micro_steps_in_update
                final_update_loss_tokens = accumulated_loss_tokens
                micro_steps_in_update = 0
                accumulated_loss_tokens = 0

                should_log = (
                    global_step == 1
                    or global_step % args.log_every == 0
                    or training_budget_reached()
                )
                domain_log = None
                if args.log_domain_loss and should_log:
                    domain_log = domain_stats_to_log(
                        reduce_domain_stats(running_domain_stats, device, ddp),
                        domain_names,
                    )

                if is_main and should_log:
                    now = time.time()
                    dt = max(now - last_log_time, 1e-9)
                    last_train_loss = running_loss_sum / max(running_loss_tokens, 1)
                    last_train_loss_tokens = running_loss_tokens
                    row = {
                        "event": "train",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": last_train_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "loss_tokens": running_loss_tokens,
                        "input_tokens": running_input_tokens,
                        "cumulative_input_tokens": total_train_input_tokens,
                        "target_input_tokens": args.target_input_tokens or None,
                        "truncated_samples": running_truncated_samples,
                        "loss_tokens_per_sec_global_approx": running_loss_tokens * world_size / dt,
                        "input_tokens_per_sec_global_approx": running_input_tokens * world_size / dt,
                        "optimizer_micro_steps": final_update_micro_steps,
                        "optimizer_loss_tokens": final_update_loss_tokens,
                        "elapsed_sec": now - start,
                    }
                    if domain_log is not None:
                        row.update(domain_log)
                    if device.type == "cuda":
                        row["cuda_memory_gb_rank0"] = {
                            "allocated": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
                            "reserved": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
                            "max_allocated": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
                            "max_reserved": round(torch.cuda.max_memory_reserved(device) / (1024**3), 3),
                        }
                        torch.cuda.reset_peak_memory_stats(device)
                if should_log and args.log_domain_loss:
                    running_domain_stats = new_domain_stats(num_domains)
                if is_main and should_log:
                    running_loss_sum = 0.0
                    running_count = 0
                    running_loss_tokens = 0
                    running_input_tokens = 0
                    running_truncated_samples = 0
                    last_log_time = now
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps(row, ensure_ascii=False), flush=True)

                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    val = evaluate(model, val_loader, device, args.eval_batches, args.bf16)
                    if is_main:
                        row = {"event": "eval", "step": global_step, **val, "elapsed_sec": time.time() - start}
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(json.dumps(row, ensure_ascii=False), flush=True)

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_model(model, args.out_dir / f"checkpoint-{global_step}", rank)

                if training_budget_reached():
                    break
        save_model(model, args.out_dir / "final", rank)
        final_validation = None
        if args.final_eval:
            heartbeat: FinalEvalHeartbeat | None = (
                FinalEvalHeartbeat(
                    log_path,
                    step=global_step,
                    training_started_at=start,
                )
                if is_main
                else None
            )
            with heartbeat if heartbeat is not None else nullcontext():
                final_validation = evaluate(
                    model,
                    val_loader,
                    device,
                    args.final_eval_batches,
                    args.bf16,
                    heartbeat,
                )
            final_validation["complete_view"] = args.final_eval_batches == 0
            final_validation["expected_input_tokens"] = val_ds.token_count()
            final_validation["input_token_coverage"] = (
                float(final_validation["input_tokens"]) / val_ds.token_count()
            )
            if is_main:
                row = {
                    "event": "final_eval",
                    "step": global_step,
                    **final_validation,
                    "elapsed_sec": time.time() - start,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)
        if is_main:
            trailing_train_loss = None
            trailing_train_loss_tokens = 0
            if running_count > 0:
                trailing_train_loss = running_loss_sum / max(running_loss_tokens, 1)
                trailing_train_loss_tokens = running_loss_tokens
            mean_train_loss = None
            if total_train_loss_tokens > 0:
                mean_train_loss = total_train_loss_weighted / total_train_loss_tokens
            actual_flops_6nd = 6 * param_count * total_train_input_tokens
            write_json(args.out_dir / "summary.json", {
                "global_step": global_step,
                "elapsed_sec": time.time() - start,
                "parameter_count": param_count,
                "model_implementation": args.model_implementation,
                "attention_backend": attention_backend,
                "final_checkpoint": str(args.out_dir / "final"),
                "final_logged_train_loss": last_train_loss,
                "final_logged_train_loss_tokens": last_train_loss_tokens,
                "trailing_train_loss": trailing_train_loss,
                "trailing_train_loss_tokens": trailing_train_loss_tokens,
                "mean_train_loss": mean_train_loss,
                "total_train_loss_tokens": total_train_loss_tokens,
                "target_input_tokens": args.target_input_tokens or None,
                "actual_input_tokens": total_train_input_tokens,
                "input_token_overshoot": (
                    total_train_input_tokens - args.target_input_tokens
                    if args.target_input_tokens > 0
                    else None
                ),
                "input_token_relative_error": (
                    (total_train_input_tokens - args.target_input_tokens)
                    / args.target_input_tokens
                    if args.target_input_tokens > 0
                    else None
                ),
                "target_flops_6nd": (
                    6 * param_count * args.target_input_tokens
                    if args.target_input_tokens > 0
                    else None
                ),
                "actual_flops_6nd": actual_flops_6nd,
                "train_input_tokens_per_epoch": train_ds.token_count(),
                "actual_epoch_fraction": total_train_input_tokens / train_ds.token_count(),
                "total_train_samples": total_train_samples,
                "scheduler_total_steps": args.max_steps,
                "warmup_steps": warmup_steps,
                "final_update_micro_steps": final_update_micro_steps,
                "final_update_loss_tokens": final_update_loss_tokens,
                "termination_reason": "target_input_tokens" if args.target_input_tokens > 0 else "max_steps",
                "final_validation": final_validation,
            })
    finally:
        cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
