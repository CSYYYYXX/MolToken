#!/usr/bin/env python3
"""Private Axis-B training engine used by the formal repeated-stage wrapper.

The script intentionally uses the standard Hugging Face Qwen3 implementation:
the conditional prefix is assembled as ``inputs_embeds`` outside the model.
It supports a one-GPU smoke run and single-node torchrun/DDP production runs.
All heavy data/index/model work is expected to run on an allocated worker.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

PIPELINE_ROOT = Path(__file__).resolve().parents[3]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.axis_b_data import (
    AxisBIndex,
    AxisBJsonlDataset,
    AxisBSFTCollator,
    DEFAULT_QWEN_CONTEXT,
    HomogeneousTaskBatchSampler,
    build_axis_b_index,
)
from pipeline.finetune.axis_b_model import (
    AxisBConditionalModel,
    MODEL_VOCAB_SIZE,
    load_projection_state,
    resize_qwen_vocab_for_axis_b,
)
from pipeline.finetune.active_scope import (
    ACTIVE_TRAIN_TASK_IDS,
    FORMAL_TASK_DATASET_SCOPES,
)
from pipeline.finetune.legacy.axis_b_core13_order import (
    CORE13_LINEAGE_TASKS,
    CORE13_LINEAGES,
    PROTOCOL as CORE13_NATURAL_ONE_PASS_PROTOCOL,
    SEED as CORE13_SEED,
    NaturalOnePassOrderBatchSampler,
    canonical_lineage_tasks as canonical_core13_lineage_tasks,
    is_atomnum_task,
    load_core13_order_sidecar,
    validate_core13_resume_audits,
)
from pipeline.finetune.legacy.axis_b_natural_epoch import (
    PROTOCOL as NATURAL_ONE_PASS_PROTOCOL,
    DistributedSingleTaskNaturalEpochBatchSampler,
    SingleTaskNaturalEpochBatchSampler,
    deterministic_partition,
)


DATA_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / '')
)
RUN_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / 'runs')
)
DEFAULT_DATA = DATA_ROOT / "finetune_datasets_bpe_fddfs_mf80000_moses_guacamol_v1"
DEFAULT_PRETRAINED = RUN_ROOT / "flagship_safetensors" / "small_0p55b"
DEFAULT_TEXT_ENCODER = DATA_ROOT / "pretrained_encoders" / "all-mpnet-base-v2"
DEFAULT_MOLECULE_ENCODER = RUN_ROOT / "molbert_base_v2" / "final"

MODEL_INPUT_KEYS = frozenset(
    {
        "decoder_input_ids",
        "decoder_attention_mask",
        "decoder_labels",
        "text_input_ids",
        "text_attention_mask",
        "text_present",
        "molecule_input_ids",
        "molecule_attention_mask",
        "molecule_present",
        "prefix_lengths",
        "text_prefix_lengths",
        "molecule_prefix_lengths",
    }
)


def enforce_offline_runtime() -> None:
    """Make the worker-side training process incapable of HF network fallback."""

    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--pretrained-ckpt", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--molecule-encoder", type=Path, default=DEFAULT_MOLECULE_ENCODER)
    parser.add_argument(
        "--molecule-condition-mode",
        choices=("direct_qwen", "molbert"),
        default="direct_qwen",
        help="Use pretrained Qwen FDDFS-BPE embeddings or the legacy MolBERT projector.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("alignment", "full"), required=True)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help=(
            "Explicit Axis-B task filter; repeat for a multi-task view. "
            "AtomNum is rejected from all training views."
        ),
    )
    parser.add_argument(
        "--lineage-id",
        default=None,
        help=(
            "Frozen Core-13 lineage: core13_moses, core13_guacamol, or "
            "core13_mol1d_mix. It is inferred only from an exact frozen task set."
        ),
    )
    parser.add_argument(
        "--natural-order-dir",
        type=Path,
        default=None,
        help="Prebuilt Core-13 alignment/full .npy order sidecar directory.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--sampling-protocol",
        choices=(
            "fixed_steps",
            NATURAL_ONE_PASS_PROTOCOL,
            CORE13_NATURAL_ONE_PASS_PROTOCOL,
        ),
        default="fixed_steps",
        help=(
            "Use historical fixed steps, the legacy single-task natural pass, "
            "or a frozen Core-13 lineage order sidecar."
        ),
    )
    parser.add_argument("--natural-partition-start", type=int, default=0)
    parser.add_argument("--natural-partition-end", type=int, default=1)
    parser.add_argument("--natural-partition-parts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--protein-batch-size", type=int, default=8)
    parser.add_argument("--pocket-batch-size", type=int, default=32)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-context", type=int, default=DEFAULT_QWEN_CONTEXT)
    parser.add_argument("--text-max-length", type=int, default=384)
    parser.add_argument("--molecule-max-length", type=int, default=512)
    parser.add_argument(
        "--condition-token-budget",
        type=int,
        default=512,
        help="Maximum condition tokens per text/molecule stream.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many consecutive epochs without validation-loss "
            "improvement; zero disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss decrease counted as an improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=0,
        help="Do not stop before this many complete training epochs.",
    )
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--strict-no-runtime-truncation",
        action="store_true",
        help=(
            "Fail if text, molecule, combined-prefix, or target tokens would be "
            "truncated by the runtime collator. Historical runs keep the legacy "
            "truncating behavior unless this flag is set."
        ),
    )
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP SFT requires CUDA on every rank")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + 1009 * int(rank)
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def log(message: str, *, rank: int, handle: Any | None = None) -> None:
    if rank != 0:
        return
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


def load_text_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def load_axis_b_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AxisBConditionalModel:
    from transformers import AutoModel, Qwen3ForCausalLM

    qwen_path = args.resume_dir / "qwen3" if args.resume_dir else args.pretrained_ckpt
    from moltoken.checkpoints import load_pretrained
    qwen = load_pretrained(qwen_path, dtype=dtype if device.type == "cuda" else torch.float32)
    resize_qwen_vocab_for_axis_b(qwen)
    qwen.config.use_cache = False

    text_encoder = AutoModel.from_pretrained(
        str(args.text_encoder),
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    molecule_encoder = None
    if args.molecule_condition_mode == "molbert":
        from transformers import BertForMaskedLM

        # Legacy compatibility only.  New runs should use direct_qwen.
        molbert_mlm = BertForMaskedLM.from_pretrained(
            str(args.molecule_encoder),
            torch_dtype=dtype if device.type == "cuda" else torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        molecule_encoder = molbert_mlm.bert
        del molbert_mlm

    model = AxisBConditionalModel(
        qwen=qwen,
        text_encoder=text_encoder,
        molecule_encoder=molecule_encoder,
        condition_token_budget=int(args.condition_token_budget),
        molecule_condition_mode=args.molecule_condition_mode,
    )
    if args.resume_dir:
        load_projection_state(model, args.resume_dir)
    model.to(device)
    model.configure_training_stage(args.stage)
    return model


def task_batch_sizes(dataset: AxisBJsonlDataset, args: argparse.Namespace) -> dict[str, int]:
    result: dict[str, int] = {}
    for task in dataset.task_counts(dataset.split):
        if task == "text2protein":
            result[task] = int(args.protein_batch_size)
        elif task == "pocket2ligand":
            result[task] = int(args.pocket_batch_size)
        else:
            result[task] = int(args.batch_size)
    return result


def choose_steps_per_epoch(
    dataset: AxisBJsonlDataset,
    batch_sizes: Mapping[str, int],
    world_size: int,
    requested: int,
) -> int:
    if requested > 0:
        return int(requested)
    counts = dataset.task_counts(dataset.split)
    weights = {task: math.sqrt(float(count)) for task, count in counts.items() if count > 0}
    denominator = sum(weights.values())
    expected_global_batch = sum(
        (weight / denominator) * int(batch_sizes[task]) * int(world_size)
        for task, weight in weights.items()
    )
    return max(len(weights) * 2, int(math.ceil(len(dataset) / max(expected_global_batch, 1.0))))


class SequentialBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        length: int,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_length = max(
            0, (self.length - self.rank + self.world_size - 1) // self.world_size
        )

    def __iter__(self) -> Iterable[list[int]]:
        indices = range(self.rank, self.length, self.world_size)
        batch: list[int] = []
        for index in indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.local_length / self.batch_size)


def move_model_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in MODEL_INPUT_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value):
            result[key] = value.to(device=device, non_blocking=True)
    required = MODEL_INPUT_KEYS.difference(result)
    if required:
        raise KeyError(f"collator batch missing model keys: {sorted(required)}")
    return result


def all_reduce_sum(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_max(value: float, device: torch.device) -> float:
    """Return the worst-rank scalar for memory and other diagnostics."""

    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def all_gather_int(value: int, device: torch.device) -> list[int]:
    """Gather an exact integer counter from every DDP rank."""

    if not dist.is_available() or not dist.is_initialized():
        return [int(value)]
    tensor = torch.tensor([int(value)], dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [int(item.item()) for item in gathered]


def build_optimizer(model: AxisBConditionalModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    lr = args.learning_rate
    if lr is None:
        lr = 1e-4 if args.stage == "alignment" else 5e-5
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        # Alignment's tied embedding has a gradient mask; decoupled weight
        # decay would still mutate frozen rows, so keep it in no_decay.
        if args.stage == "alignment" and name.endswith("embed_tokens.weight"):
            no_decay.append(parameter)
        elif name.endswith("bias") or parameter.ndim < 2:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(args.weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise RuntimeError("no trainable parameters after stage configuration")
    return torch.optim.AdamW(groups, lr=float(lr), betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(max(0, warmup_steps))
    total_steps = int(max(1, total_steps))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(
    model: AxisBConditionalModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    output_dir: Path,
    *,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    stats: Mapping[str, Any],
    rank: int,
) -> None:
    if rank != 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_axis_b_pretrained(
        output_dir,
        metadata={
            "checkpoint_role": (
                "best"
                if output_dir.name == "best"
                else "final"
                if output_dir.name == "final"
                else "periodic"
            ),
            "stage": args.stage,
            "epoch": int(epoch),
            "step": int(step),
            "data_root": str(args.data_root),
            "pretrained_ckpt": str(args.pretrained_ckpt),
            "resume_dir": str(args.resume_dir) if args.resume_dir else None,
            "text_encoder": str(args.text_encoder),
            "molecule_condition_mode": args.molecule_condition_mode,
            "molecule_encoder": str(args.molecule_encoder) if args.molecule_condition_mode == "molbert" else None,
            "max_context": int(args.max_context),
            "text_max_length": int(args.text_max_length),
            "strict_no_runtime_truncation": bool(
                args.strict_no_runtime_truncation
            ),
            "world_size": int(getattr(args, "world_size", 1)),
            "grad_accumulation_steps": int(args.grad_accumulation_steps),
            "effective_batch_multiplier": int(getattr(args, "world_size", 1))
            * int(args.grad_accumulation_steps),
            "effective_global_batch_records": int(getattr(args, "effective_global_batch_records", 0)),
            "task_homogeneous_accumulation": (
                str(args.sampling_protocol) != CORE13_NATURAL_ONE_PASS_PROTOCOL
                or len(getattr(args, "tasks", ())) == 1
            ),
            "gradient_normalization": "target_token_weighted_across_accumulation_group",
            "sampling_protocol": str(args.sampling_protocol),
            "natural_epoch_audit": dict(getattr(args, "natural_epoch_audit", {})),
            "stats": dict(stats),
            "task": args.task,
            "tasks": list(getattr(args, "tasks", ())),
            "lineage_id": getattr(args, "lineage_id", None),
        },
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "step": int(step),
            "args": vars(args),
        },
        output_dir / "trainer_state.pt",
    )
    (output_dir / "trainer_state.json").write_text(
        json.dumps(
            {"epoch": int(epoch), "step": int(step), "stage": args.stage, "stats": dict(stats)},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_sum = 0.0
    truncation_sum = 0.0
    batches = 0
    for batch in loader:
        if max_batches > 0 and batches >= max_batches:
            break
        model_batch = move_model_batch(batch, device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and use_bf16,
        ):
            output = model(**model_batch)
        tokens = float(output.loss_tokens)
        if output.loss is not None and tokens > 0:
            loss_sum += float(output.loss.detach().float().item()) * tokens
            token_sum += tokens
        if torch.is_tensor(batch.get("target_truncated")):
            truncation_sum += float(batch["target_truncated"].sum().item())
        batches += 1
    loss_sum = all_reduce_sum(loss_sum, device)
    token_sum = all_reduce_sum(token_sum, device)
    truncation_sum = all_reduce_sum(truncation_sum, device)
    model.train()
    return {
        "loss": loss_sum / token_sum if token_sum else float("nan"),
        "loss_tokens": token_sum,
        "batches": float(batches),
        "truncated_samples": truncation_sum,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_view_row_sha256(dataset: AxisBJsonlDataset) -> str:
    """Fingerprint the exact ordered task view without reading JSONL payloads."""

    positions = getattr(dataset, "_row_positions", None)
    rows = getattr(dataset, "_rows", None)
    if positions is None or rows is None:
        raise RuntimeError("natural one-pass requires an indexed single-task dataset view")
    selected_rows = rows[positions]
    digest = hashlib.sha256()
    digest.update(str(dataset.task).encode("utf-8"))
    digest.update(b"\0")
    digest.update(selected_rows.tobytes(order="C"))
    return digest.hexdigest()


def validate_natural_one_pass_resume(
    args: argparse.Namespace, current_audit: Mapping[str, Any]
) -> None:
    """Require full stage to continue an adjacent alignment partition."""

    if args.stage != "full":
        return
    if args.resume_dir is None:
        raise ValueError("natural one-pass full stage requires an alignment checkpoint")
    config_path = args.resume_dir / "axis_b_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("alignment checkpoint is missing Axis-B metadata")
    previous = metadata.get("natural_epoch_audit")
    if not isinstance(previous, dict) or not previous.get("complete"):
        raise ValueError("alignment checkpoint has no completed natural one-pass audit")
    checks = {
        "protocol": NATURAL_ONE_PASS_PROTOCOL,
        "task": args.task,
        "stage": "alignment",
        "dataset_records": current_audit.get("dataset_records"),
        "seed": current_audit.get("seed"),
        "partition_parts": current_audit.get("partition_parts"),
        "partition_start_part": 0,
        "partition_end_part": current_audit.get("partition_start_part"),
    }
    mismatches = {
        key: {"expected": expected, "actual": previous.get(key)}
        for key, expected in checks.items()
        if previous.get(key) != expected
    }
    if current_audit.get("partition_end_part") != current_audit.get("partition_parts"):
        mismatches["full_partition_end"] = {
            "expected": current_audit.get("partition_parts"),
            "actual": current_audit.get("partition_end_part"),
        }
    parent_interface_checks = {
        "data_root": str(args.data_root),
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "text_encoder": str(args.text_encoder),
        "molecule_condition_mode": str(args.molecule_condition_mode),
        "max_context": int(args.max_context),
    }
    for key, expected in parent_interface_checks.items():
        if metadata.get(key) != expected:
            mismatches[f"parent_{key}"] = {
                "expected": expected,
                "actual": metadata.get(key),
            }
    if int(payload.get("condition_token_budget", -1)) != int(args.condition_token_budget):
        mismatches["parent_condition_token_budget"] = {
            "expected": int(args.condition_token_budget),
            "actual": payload.get("condition_token_budget"),
        }
    if mismatches:
        raise ValueError(
            "natural one-pass alignment/full partitions are not one exhaustive "
            f"lineage: {json.dumps(mismatches, sort_keys=True)}"
        )
    current_audit["parent_checkpoint"] = str(args.resume_dir)


def validate_core13_natural_one_pass_resume(
    args: argparse.Namespace, current_audit: dict[str, Any]
) -> None:
    """Validate that full resumes the exact frozen Core-13 alignment lineage."""

    if args.stage != "full":
        return
    if args.resume_dir is None:
        raise ValueError("Core-13 full stage requires its alignment final checkpoint")
    config_path = args.resume_dir / "axis_b_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Core-13 alignment checkpoint is missing Axis-B metadata")
    previous = metadata.get("natural_epoch_audit")
    if not isinstance(previous, dict):
        raise ValueError("Core-13 alignment checkpoint has no natural-epoch audit")
    validate_core13_resume_audits(previous, current_audit)

    parent_interface_checks = {
        "stage": "alignment",
        "data_root": str(args.data_root),
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "text_encoder": str(args.text_encoder),
        "molecule_condition_mode": str(args.molecule_condition_mode),
        "max_context": int(args.max_context),
        "sampling_protocol": CORE13_NATURAL_ONE_PASS_PROTOCOL,
        "task": args.task,
        "tasks": list(args.tasks),
        "lineage_id": str(args.lineage_id),
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in parent_interface_checks.items()
        if metadata.get(key) != expected
    }
    if int(payload.get("condition_token_budget", -1)) != int(args.condition_token_budget):
        mismatches["condition_token_budget"] = {
            "expected": int(args.condition_token_budget),
            "actual": payload.get("condition_token_budget"),
        }
    if mismatches:
        raise ValueError(
            "Core-13 full resume parent interface mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    current_audit["parent_axis_b_config_sha256"] = sha256_file(config_path)
    current_audit["parent_checkpoint"] = str(args.resume_dir)


def main() -> int:
    enforce_offline_runtime()
    args = parse_args()
    raw_tasks = tuple(args.task or ())
    tasks = tuple(sorted(dict.fromkeys(str(task).strip() for task in raw_tasks)))
    if any(not task for task in tasks):
        raise ValueError("--task values must be non-empty")
    forbidden_atomnum = [task for task in tasks if is_atomnum_task(task)]
    if forbidden_atomnum:
        raise ValueError(
            f"AtomNum is materialize-only and cannot enter training: {forbidden_atomnum}"
        )
    args.tasks = tasks
    # Preserve the historical singular metadata field without conflating a
    # filtered mixed view with a single task.
    args.task = tasks[0] if len(tasks) == 1 else None
    rank, world_size, _local_rank, device = setup_distributed()
    set_seed(args.seed, rank)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.index_dir = args.index_dir.expanduser().resolve() if args.index_dir else None
    args.pretrained_ckpt = args.pretrained_ckpt.expanduser().resolve()
    args.resume_dir = args.resume_dir.expanduser().resolve() if args.resume_dir else None
    args.natural_order_dir = (
        args.natural_order_dir.expanduser().resolve() if args.natural_order_dir else None
    )
    args.text_encoder = args.text_encoder.expanduser().resolve()
    args.molecule_encoder = args.molecule_encoder.expanduser().resolve()
    args.world_size = int(world_size)
    if int(args.grad_accumulation_steps) <= 0:
        raise ValueError("--grad-accumulation-steps must be positive")
    if int(args.early_stopping_patience) < 0:
        raise ValueError("--early-stopping-patience cannot be negative")
    if float(args.early_stopping_min_delta) < 0:
        raise ValueError("--early-stopping-min-delta cannot be negative")
    if int(args.early_stopping_min_epochs) < 0:
        raise ValueError("--early-stopping-min-epochs cannot be negative")
    args.effective_global_batch_records = (
        int(args.batch_size) * int(world_size) * int(args.grad_accumulation_steps)
    )
    single_task_natural = args.sampling_protocol == NATURAL_ONE_PASS_PROTOCOL
    core13_natural = args.sampling_protocol == CORE13_NATURAL_ONE_PASS_PROTOCOL
    natural_one_pass = single_task_natural or core13_natural
    if not core13_natural and args.lineage_id in CORE13_LINEAGES:
        raise ValueError(
            "Frozen Core-13 lineage IDs require the record-ID-hash "
            f"{CORE13_NATURAL_ONE_PASS_PROTOCOL} protocol; legacy position-based "
            "sampling remains available only for non-Core-13 compatibility runs"
        )
    if natural_one_pass:
        if args.epochs != 1 or args.steps_per_epoch != 0 or args.max_steps != 0:
            raise ValueError(
                "natural one-pass sampling requires --epochs 1, "
                "--steps-per-epoch 0, and --max-steps 0"
            )
    if core13_natural and world_size != 1:
        raise ValueError(
            "the legacy Core-13 natural-order adapter remains single-rank; "
            "use its dedicated distributed lineage adapter for DDP"
        )
    if single_task_natural and len(args.tasks) != 1:
        raise ValueError("legacy natural one-pass sampling requires one explicit --task")
    if core13_natural:
        if args.lineage_id is None:
            matching_lineages = [
                lineage
                for lineage, lineage_tasks in CORE13_LINEAGE_TASKS.items()
                if tuple(sorted(lineage_tasks)) == args.tasks
            ]
            if len(matching_lineages) != 1:
                raise ValueError(
                    "Core-13 natural one-pass requires one frozen --lineage-id/task set: "
                    f"lineages={list(CORE13_LINEAGES)} tasks={list(args.tasks)}"
                )
            args.lineage_id = matching_lineages[0]
        args.lineage_id, selected_core13_tasks = canonical_core13_lineage_tasks(
            args.lineage_id, args.tasks
        )
        args.tasks = selected_core13_tasks
        args.task = args.tasks[0] if len(args.tasks) == 1 else None
        if args.seed != CORE13_SEED:
            raise ValueError(f"Core-13 seed is frozen at {CORE13_SEED}")
        if args.natural_order_dir is None:
            raise ValueError("Core-13 natural one-pass training requires --natural-order-dir")
        if args.force_index:
            raise ValueError(
                "Core-13 training forbids --force-index; build/reuse the compact index "
                "and order sidecar on a data worker before training"
            )
        if args.index_dir is None:
            schema_version = "v2" if (args.data_root / "condition_schema.json").is_file() else "v1"
            args.index_dir = args.data_root / f"axis_b_index_{schema_version}"
        if not (args.index_dir / "index_manifest.json").is_file():
            raise FileNotFoundError(
                "Core-13 prebuilt Axis-B index is missing; run "
                "build_axis_b_core13_order.py on a CPU/data worker first: "
                f"{args.index_dir / 'index_manifest.json'}"
            )
        if args.molecule_condition_mode != "direct_qwen":
            raise ValueError("Core-13 training requires --molecule-condition-mode direct_qwen")
        expected_learning_rate = 1e-4 if args.stage == "alignment" else 5e-5
        if args.learning_rate is not None and float(args.learning_rate) != expected_learning_rate:
            raise ValueError(
                f"Core-13 {args.stage} peak learning rate is frozen at "
                f"{expected_learning_rate:g}"
            )
        args.learning_rate = expected_learning_rate
        frozen_scalars = {
            "weight_decay": (float(args.weight_decay), 0.1),
            "warmup_ratio": (float(args.warmup_ratio), 0.03),
            "min_lr_ratio": (float(args.min_lr_ratio), 0.1),
            "max_grad_norm": (float(args.max_grad_norm), 1.0),
        }
        scalar_mismatches = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in frozen_scalars.items()
            if actual != expected
        }
        if scalar_mismatches:
            raise ValueError(
                "Core-13 frozen optimizer/schedule mismatch: "
                + json.dumps(scalar_mismatches, sort_keys=True)
            )
        if args.no_bf16:
            raise ValueError("Core-13 precision is frozen at bf16")
        if args.no_gradient_checkpointing:
            raise ValueError("Core-13 requires gradient checkpointing")
        if device.type != "cuda" and not args.dry_run:
            raise ValueError("Core-13 formal training requires a CUDA worker for bf16")
        if (
            args.natural_partition_start,
            args.natural_partition_end,
            args.natural_partition_parts,
        ) != (0, 1, 1):
            raise ValueError(
                "Core-13 alignment/full membership comes only from the frozen order sidecar; "
                "do not pass legacy --natural-partition-* overrides"
            )
        if args.stage == "alignment" and args.resume_dir is not None:
            raise ValueError("Core-13 alignment must initialize independently from the base")
    elif args.lineage_id is None:
        args.lineage_id = args.task
    if args.stage == "full" and args.resume_dir is None:
        raise ValueError(
            "full SFT requires --resume-dir pointing to the completed alignment checkpoint"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8") if rank == 0 else None
    try:
        if rank == 0:
            index = build_axis_b_index(
                args.data_root,
                index_dir=args.index_dir,
                allow_non_worker=False,
                force=args.force_index,
            )
        barrier()
        index = AxisBIndex.load(args.data_root, args.index_dir)
        dataset_filter = {"tasks": args.tasks} if args.tasks else {}
        train_dataset = AxisBJsonlDataset(
            args.data_root,
            split="train",
            index_dir=args.index_dir,
            index=index,
            **dataset_filter,
        )
        eval_dataset = AxisBJsonlDataset(
            args.data_root,
            split="eval",
            index_dir=args.index_dir,
            index=index,
            **dataset_filter,
        )
        if getattr(args, "formal_active_scope", False):
            indexed_datasets = {
                str(item.get("dataset")) for item in index.files
            }
            if len(args.tasks) != 1:
                raise RuntimeError(
                    "formal repeated-stage SFT requires exactly one task scope"
                )
            formal_task = str(args.tasks[0])
            expected_datasets = FORMAL_TASK_DATASET_SCOPES.get(formal_task)
            if expected_datasets is None:
                raise RuntimeError(
                    f"formal SFT task has no registered data scope: {formal_task!r}"
                )
            if indexed_datasets != set(expected_datasets):
                raise RuntimeError(
                    "formal SFT data root does not match the registered task scope: "
                    f"task={formal_task!r} expected={list(expected_datasets)} "
                    f"actual={sorted(indexed_datasets)}"
                )
            if set(args.tasks) - set(ACTIVE_TRAIN_TASK_IDS):
                raise RuntimeError(
                    f"formal SFT task scope escaped active allowlist: {list(args.tasks)}"
                )
            if len(eval_dataset) <= 0:
                raise RuntimeError(
                    f"formal task {args.tasks[0]!r} has no train-derived validation "
                    "view; official benchmark test data cannot be used for early stopping"
                )
        batch_sizes = task_batch_sizes(train_dataset, args)
        observed_atomnum = [task for task in batch_sizes if is_atomnum_task(task)]
        if observed_atomnum:
            raise RuntimeError(
                f"AtomNum is materialize-only and leaked into the training view: {observed_atomnum}"
            )
        accumulation = max(1, int(args.grad_accumulation_steps))
        if core13_natural:
            if set(batch_sizes) != set(args.tasks):
                raise RuntimeError(
                    f"Core-13 task filter/index mismatch: expected={args.tasks} "
                    f"actual={sorted(batch_sizes)}"
                )
            if set(batch_sizes.values()) != {int(args.batch_size)}:
                raise RuntimeError(
                    f"Core-13 lineage tasks require one uniform microbatch: {batch_sizes}"
                )
            sidecar = load_core13_order_sidecar(
                train_dataset,
                args.natural_order_dir,
                lineage_id=args.lineage_id,
                tasks=args.tasks,
                seed=args.seed,
            )
            stage_order = sidecar.order(args.stage)
            sampler = NaturalOnePassOrderBatchSampler(stage_order, args.batch_size)
            steps_per_epoch = len(sampler)
            optimizer_steps_per_epoch = int(math.ceil(steps_per_epoch / accumulation))
            natural_audit = sidecar.natural_epoch_audit(args.stage)
            per_task_partition = natural_audit["partition_audit"]["per_task"]
            selected_task_counts = {
                task: int(per_task_partition[task][f"{args.stage}_records"])
                for task in args.tasks
            }
            expected_optimizer_tail = (
                sampler.selected_records % args.effective_global_batch_records
                or args.effective_global_batch_records
            )
            natural_audit.update(
                {
                    "data_root": str(args.data_root),
                    "index_dir": str(index.index_dir),
                    "order_sidecar_dir": str(sidecar.root),
                    "selected_task_counts": selected_task_counts,
                    "batch_size": int(args.batch_size),
                    "world_size": int(world_size),
                    "gradient_accumulation_steps": accumulation,
                    "effective_global_batch_records": int(
                        args.effective_global_batch_records
                    ),
                    "expected_microsteps": steps_per_epoch,
                    "expected_optimizer_steps": optimizer_steps_per_epoch,
                    "expected_final_batch_size": sampler.final_batch_size,
                    "expected_final_optimizer_batch_records": int(
                        expected_optimizer_tail
                    ),
                    "expected_final_accumulation_microsteps": (
                        steps_per_epoch % accumulation or accumulation
                    ),
                    "no_balancing_cycling_replication_or_oversampling": True,
                }
            )
            args.natural_epoch_audit = natural_audit
            validate_core13_natural_one_pass_resume(
                args, args.natural_epoch_audit
            )
        elif single_task_natural:
            if len(batch_sizes) != 1 or args.task not in batch_sizes:
                raise RuntimeError(
                    "natural one-pass task view did not resolve to exactly one task"
                )
            selected_indices, natural_audit = deterministic_partition(
                len(train_dataset),
                seed=args.seed,
                start_part=args.natural_partition_start,
                end_part=args.natural_partition_end,
                parts=args.natural_partition_parts,
            )
            natural_audit.update(
                {
                    "data_root": str(args.data_root),
                    "index_dir": str(index.index_dir),
                }
            )
            if world_size == 1:
                sampler = SingleTaskNaturalEpochBatchSampler(
                    selected_indices, batch_sizes[args.task]
                )
                final_local_batch_sizes = [int(sampler.final_batch_size)]
                expected_local_records_by_rank = [int(sampler.selected_records)]
                final_global_batch_size = int(sampler.final_batch_size)
                expected_optimizer_tail = (
                    int(sampler.selected_records) % args.effective_global_batch_records
                    or args.effective_global_batch_records
                )
            else:
                sampler = DistributedSingleTaskNaturalEpochBatchSampler(
                    selected_indices,
                    batch_sizes[args.task],
                    rank=rank,
                    world_size=world_size,
                )
                final_local_batch_sizes = [
                    int(value) for value in sampler.final_local_batch_sizes
                ]
                expected_local_records_by_rank = [
                    int(value) for value in sampler.expected_local_records_by_rank
                ]
                final_global_batch_size = int(sampler.final_global_batch_size)
                expected_optimizer_tail = sampler.final_optimizer_batch_records(
                    accumulation
                )
            steps_per_epoch = len(sampler)
            optimizer_steps_per_epoch = int(math.ceil(steps_per_epoch / accumulation))
            natural_audit.update(
                {
                    "task": str(args.task),
                    "stage": str(args.stage),
                    "batch_size": int(batch_sizes[args.task]),
                    "world_size": int(world_size),
                    "gradient_accumulation_steps": accumulation,
                    "effective_global_batch_records": int(
                        args.effective_global_batch_records
                    ),
                    "expected_microsteps": steps_per_epoch,
                    "expected_optimizer_steps": optimizer_steps_per_epoch,
                    "expected_final_batch_size": final_local_batch_sizes[0],
                    "expected_final_global_batch_size": final_global_batch_size,
                    "expected_final_local_batch_sizes": final_local_batch_sizes,
                    "expected_local_records_by_rank": expected_local_records_by_rank,
                    "expected_final_optimizer_batch_records": int(
                        expected_optimizer_tail
                    ),
                    "expected_final_accumulation_microsteps": (
                        steps_per_epoch % accumulation or accumulation
                    ),
                    "no_padding_duplication_cycling_or_oversampling": True,
                }
            )
            args.natural_epoch_audit = natural_audit
            validate_natural_one_pass_resume(args, args.natural_epoch_audit)
        else:
            steps_per_epoch = choose_steps_per_epoch(
                train_dataset, batch_sizes, world_size, args.steps_per_epoch
            )
            steps_per_epoch = int(math.ceil(steps_per_epoch / accumulation) * accumulation)
            optimizer_steps_per_epoch = steps_per_epoch // accumulation
            sampler = HomogeneousTaskBatchSampler(
                train_dataset,
                batch_sizes=batch_sizes,
                steps_per_epoch=steps_per_epoch,
                seed=args.seed,
                epoch=0,
                rank=rank,
                world_size=world_size,
                task_group_size=accumulation,
            )
        log(
            f"Axis-B data: train={len(train_dataset):,} eval={len(eval_dataset):,} "
            f"task_filter={list(args.tasks) if args.tasks else 'all'} "
            f"tasks={len(batch_sizes)} microsteps/epoch={steps_per_epoch} "
            f"optimizer_steps/epoch={optimizer_steps_per_epoch} world={world_size} "
            f"accumulation={accumulation} "
            f"effective_global_batch_records={args.effective_global_batch_records}",
            rank=rank,
            handle=log_handle,
        )
        if natural_one_pass:
            log(
                "natural_one_pass="
                + json.dumps(args.natural_epoch_audit, sort_keys=True),
                rank=rank,
                handle=log_handle,
            )
        if args.dry_run:
            log(f"batch_sizes={json.dumps(batch_sizes, sort_keys=True)}", rank=rank, handle=log_handle)
            return 0

        tokenizer = load_text_tokenizer(args.text_encoder)
        collator = AxisBSFTCollator(
            tokenizer,
            qwen_max_context=args.max_context,
            return_tensors="pt",
            text_max_length=args.text_max_length,
            molecule_max_length=args.molecule_max_length,
            condition_token_budget=args.condition_token_budget,
            molecule_condition_mode=args.molecule_condition_mode,
            strict_no_runtime_truncation=args.strict_no_runtime_truncation,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        eval_sampler = SequentialBatchSampler(
            len(eval_dataset),
            args.eval_batch_size,
            rank=rank,
            world_size=world_size,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        dtype = torch.float32 if args.no_bf16 or device.type != "cuda" else torch.bfloat16
        model = load_axis_b_model(args, device=device, dtype=dtype)
        if not args.no_gradient_checkpointing and hasattr(model.qwen, "gradient_checkpointing_enable"):
            # Reentrant checkpointing fires DDP parameter hooks more than once
            # when find_unused_parameters=True. Non-reentrant checkpointing is
            # required for the full-stage conditional graph.
            model.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.qwen.config.use_cache = False
        report = model.trainable_parameter_report()
        log(f"parameter_report={json.dumps(report, sort_keys=True)}", rank=rank, handle=log_handle)
        if world_size > 1:
            model_for_train: torch.nn.Module = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        else:
            model_for_train = model
        optimizer = build_optimizer(model, args)
        total_steps = int(args.epochs * optimizer_steps_per_epoch)
        if args.max_steps > 0:
            total_steps = min(total_steps, int(args.max_steps))
        warmup_steps = int(round(total_steps * float(args.warmup_ratio)))
        scheduler = build_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

        # Exclude model/optimizer loading from the training-memory diagnostic.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        global_step = 0
        optimizer_step = 0
        running_loss = 0.0
        running_tokens = 0.0
        running_truncated = 0.0
        accumulated_global_tokens = 0.0
        accumulated_global_records = 0
        processed_train_samples = 0
        last_train_batch_size = 0
        last_optimizer_batch_records = 0
        completed_epochs = 0
        best_eval_loss = float("inf")
        best_eval_epoch: int | None = None
        epochs_without_improvement = 0
        stopped_early = False
        optimizer.zero_grad(set_to_none=True)
        model_for_train.train()
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(train_loader):
                if args.max_steps > 0 and optimizer_step >= args.max_steps:
                    break
                model_batch = move_model_batch(batch, device)
                last_train_batch_size = int(model_batch["decoder_input_ids"].shape[0])
                processed_train_samples += last_train_batch_size
                accumulated_global_records += int(
                    all_reduce_sum(last_train_batch_size, device)
                )
                accumulation_index = batch_index % max(1, args.grad_accumulation_steps)
                sync = (
                    accumulation_index == max(1, args.grad_accumulation_steps) - 1
                    or batch_index == len(train_loader) - 1
                )
                context = model_for_train.no_sync() if world_size > 1 and not sync else None
                if context is None:
                    context_manager = torch.enable_grad()
                else:
                    context_manager = context
                with context_manager:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and not args.no_bf16
                    ):
                        output = model_for_train(**model_batch)
                        if output.loss is None:
                            raise RuntimeError("model returned no loss")
                        local_tokens = float(output.loss_tokens)
                        global_tokens = all_reduce_sum(local_tokens, device)
                        if global_tokens <= 0:
                            raise RuntimeError("batch has no target loss tokens")
                        # Accumulate token-summed gradients.  DDP averages the
                        # synchronized gradient across ranks; at the optimizer
                        # boundary we normalize once by all target tokens from
                        # every rank and micro-batch in this task group.
                        loss = output.loss * local_tokens
                    loss.backward()
                accumulated_global_tokens += global_tokens
                running_loss += float(output.loss.detach().float().item()) * local_tokens
                running_tokens += local_tokens
                if torch.is_tensor(batch.get("target_truncated")):
                    running_truncated += float(batch["target_truncated"].sum().item())
                global_step += 1
                if sync:
                    if accumulated_global_tokens <= 0:
                        raise RuntimeError("accumulation group has no target loss tokens")
                    gradient_scale = float(world_size) / accumulated_global_tokens
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(gradient_scale)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(args.max_grad_norm)
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_global_tokens = 0.0
                    last_optimizer_batch_records = int(accumulated_global_records)
                    accumulated_global_records = 0
                    optimizer_step += 1
                    if optimizer_step % max(1, args.logging_steps) == 0:
                        loss_total = all_reduce_sum(running_loss, device)
                        token_total = all_reduce_sum(running_tokens, device)
                        trunc_total = all_reduce_sum(running_truncated, device)
                        mem = 0.0
                        mem_reserved = 0.0
                        mem_allocated_percent = 0.0
                        mem_percent = 0.0
                        if device.type == "cuda":
                            mem = torch.cuda.max_memory_allocated(device) / (1024**3)
                            mem_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
                            mem = all_reduce_max(mem, device)
                            mem_reserved = all_reduce_max(mem_reserved, device)
                            total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                            mem_allocated_percent = 100.0 * mem / max(total_mem, 1e-9)
                            mem_percent = 100.0 * mem_reserved / max(total_mem, 1e-9)
                        log(
                            f"step={optimizer_step} epoch={epoch} loss={loss_total / max(token_total,1):.6f} "
                            f"tokens={token_total:.0f} trunc={trunc_total:.0f} "
                            f"lr={scheduler.get_last_lr()[0]:.3e} grad_norm={float(grad_norm):.3f} "
                            f"max_mem_gib={mem:.2f} reserved_gib={mem_reserved:.2f} "
                            f"allocated_pct={mem_allocated_percent:.1f} reserved_pct={mem_percent:.1f}",
                            rank=rank,
                            handle=log_handle,
                        )
                        running_loss = running_tokens = running_truncated = 0.0
                    if args.save_steps > 0 and optimizer_step % args.save_steps == 0:
                        barrier()
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            args.output_dir / f"checkpoint-{optimizer_step}",
                            args=args,
                            epoch=epoch,
                            step=optimizer_step,
                            stats={"last_train_loss": float(output.loss.detach().float().item())},
                            rank=rank,
                        )
                        barrier()
            if args.max_steps > 0 and optimizer_step >= args.max_steps:
                break
            barrier()
            eval_stats = evaluate(
                # Evaluation is read-only and rank-sharded.  Rank shards can
                # contain different batch counts, so bypass DDP forward
                # bookkeeping here; collectives are performed explicitly by
                # ``evaluate`` after every rank finishes its local shard.
                model,
                eval_loader,
                device=device,
                max_batches=args.eval_batches,
                use_bf16=not args.no_bf16,
            )
            log(f"eval epoch={epoch} {json.dumps(eval_stats, sort_keys=True)}", rank=rank, handle=log_handle)
            completed_epochs = epoch + 1
            if args.early_stopping_patience > 0:
                eval_loss = float(eval_stats["loss"])
                if not math.isfinite(eval_loss):
                    raise RuntimeError(
                        f"validation loss is not finite at epoch {completed_epochs}: {eval_loss}"
                    )
                improved = eval_loss < (
                    best_eval_loss - float(args.early_stopping_min_delta)
                )
                if improved:
                    best_eval_loss = eval_loss
                    best_eval_epoch = completed_epochs
                    epochs_without_improvement = 0
                    barrier()
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        args.output_dir / "best",
                        args=args,
                        epoch=completed_epochs,
                        step=optimizer_step,
                        stats={
                            "eval": eval_stats,
                            "early_stopping": {
                                "best_eval_loss": best_eval_loss,
                                "best_epoch": best_eval_epoch,
                                "patience": int(args.early_stopping_patience),
                                "min_delta": float(args.early_stopping_min_delta),
                                "min_epochs": int(args.early_stopping_min_epochs),
                            },
                        },
                        rank=rank,
                    )
                    barrier()
                    log(
                        f"early_stopping best epoch={best_eval_epoch} "
                        f"loss={best_eval_loss:.6f}",
                        rank=rank,
                        handle=log_handle,
                    )
                else:
                    epochs_without_improvement += 1
                    log(
                        "early_stopping no_improvement "
                        f"epoch={completed_epochs} count={epochs_without_improvement}/"
                        f"{args.early_stopping_patience} best_epoch={best_eval_epoch} "
                        f"best_loss={best_eval_loss:.6f}",
                        rank=rank,
                        handle=log_handle,
                    )
                    if (
                        completed_epochs >= args.early_stopping_min_epochs
                        and epochs_without_improvement
                        >= args.early_stopping_patience
                    ):
                        stopped_early = True
            barrier()
            if stopped_early:
                log(
                    f"EARLY_STOP epoch={completed_epochs} best_epoch={best_eval_epoch} "
                    f"best_eval_loss={best_eval_loss:.6f}",
                    rank=rank,
                    handle=log_handle,
                )
                break

        if natural_one_pass:
            expected_records = int(args.natural_epoch_audit["selected_records"])
            expected_microsteps = int(args.natural_epoch_audit["expected_microsteps"])
            expected_optimizer_steps = int(
                args.natural_epoch_audit["expected_optimizer_steps"]
            )
            processed_local_by_rank = all_gather_int(processed_train_samples, device)
            processed_global = int(sum(processed_local_by_rank))
            final_local_batch_sizes = all_gather_int(last_train_batch_size, device)
            final_global_batch_size = int(sum(final_local_batch_sizes))
            if processed_global != expected_records:
                raise RuntimeError(
                    "natural one-pass sample count mismatch: "
                    f"expected={expected_records} actual={processed_global} "
                    f"per_rank={processed_local_by_rank}"
                )
            expected_local_records = [
                int(value)
                for value in args.natural_epoch_audit.get(
                    "expected_local_records_by_rank", [expected_records]
                )
            ]
            if processed_local_by_rank != expected_local_records:
                raise RuntimeError(
                    "natural one-pass per-rank sample count mismatch: "
                    f"expected={expected_local_records} "
                    f"actual={processed_local_by_rank}"
                )
            if global_step != expected_microsteps:
                raise RuntimeError(
                    "natural one-pass microstep mismatch: "
                    f"expected={expected_microsteps} actual={global_step}"
                )
            if optimizer_step != expected_optimizer_steps:
                raise RuntimeError(
                    "natural one-pass optimizer-step mismatch: "
                    f"expected={expected_optimizer_steps} actual={optimizer_step}"
                )
            if accumulated_global_tokens != 0.0:
                raise RuntimeError("natural one-pass ended with unsynchronized gradients")
            if accumulated_global_records != 0:
                raise RuntimeError("natural one-pass ended with an unstepped record tail")
            expected_final_local = list(
                args.natural_epoch_audit.get(
                    "expected_final_local_batch_sizes",
                    [args.natural_epoch_audit["expected_final_batch_size"]],
                )
            )
            if final_local_batch_sizes != [int(value) for value in expected_final_local]:
                raise RuntimeError(
                    "natural one-pass final microbatch mismatch: "
                    f"expected={expected_final_local} "
                    f"actual={final_local_batch_sizes}"
                )
            expected_final_global = int(
                args.natural_epoch_audit.get(
                    "expected_final_global_batch_size", sum(expected_final_local)
                )
            )
            if final_global_batch_size != expected_final_global:
                raise RuntimeError(
                    "natural one-pass final global microbatch mismatch: "
                    f"expected={expected_final_global} actual={final_global_batch_size}"
                )
            yielded_digest = sampler.yielded_order_sha256
            if core13_natural:
                yielded_records = int(sampler.yielded_records)
                yielded_unique = yielded_records
                yielded_duplicates = 0
                yielded_omitted = 0
                expected_digest = args.natural_epoch_audit[
                    "selected_position_order_sha256"
                ]
                expected_tail = int(
                    args.natural_epoch_audit[
                        "expected_final_optimizer_batch_records"
                    ]
                )
                if last_optimizer_batch_records != expected_tail:
                    raise RuntimeError(
                        "Core-13 real optimizer tail mismatch: "
                        f"expected={expected_tail} actual={last_optimizer_batch_records}"
                    )
                identity_method = (
                    "validated_npy_disjoint_union_exact_once_plus_runtime_"
                    "position_order_digest_and_forward_record_count"
                )
            else:
                yielded_indices = sampler.yielded_indices
                yielded_records = len(yielded_indices)
                yielded_unique = len(set(yielded_indices))
                yielded_duplicates = yielded_records - yielded_unique
                yielded_omitted = len(set(selected_indices).difference(yielded_indices))
                expected_tail = int(
                    args.natural_epoch_audit[
                        "expected_final_optimizer_batch_records"
                    ]
                )
                if last_optimizer_batch_records != expected_tail:
                    raise RuntimeError(
                        "natural one-pass real optimizer tail mismatch: "
                        f"expected={expected_tail} actual={last_optimizer_batch_records}"
                    )
                identity_method = (
                    "runtime_exact_index_order_plus_forward_batch_record_count"
                )
            if yielded_records != expected_records:
                raise RuntimeError(
                    "natural one-pass yielded-index count mismatch: "
                    f"expected={expected_records} actual={yielded_records}"
                )
            if yielded_duplicates or yielded_omitted:
                raise RuntimeError(
                    "natural one-pass yielded-index identity mismatch: "
                    f"duplicates={yielded_duplicates} omitted={yielded_omitted}"
                )
            if core13_natural:
                if yielded_digest != expected_digest:
                    raise RuntimeError("Core-13 natural one-pass yielded order digest mismatch")
            elif tuple(yielded_indices) != tuple(selected_indices):
                raise RuntimeError("natural one-pass yielded index order mismatch")
            args.natural_epoch_audit.update(
                {
                    "actual_microsteps": global_step,
                    "actual_optimizer_steps": optimizer_step,
                    "actual_processed_records": processed_global,
                    "actual_local_records_by_rank": processed_local_by_rank,
                    "actual_unique_records": yielded_unique,
                    "actual_duplicate_records": yielded_duplicates,
                    "actual_omitted_selected_records": yielded_omitted,
                    "actual_final_batch_size": final_local_batch_sizes[0],
                    "actual_final_global_batch_size": final_global_batch_size,
                    "actual_final_local_batch_sizes": final_local_batch_sizes,
                    "actual_final_optimizer_batch_records": last_optimizer_batch_records,
                    "actual_yielded_order_sha256": yielded_digest,
                    "identity_audit_method": identity_method,
                    "complete": True,
                }
            )
            log(
                "natural_one_pass_complete="
                + json.dumps(args.natural_epoch_audit, sort_keys=True),
                rank=rank,
                handle=log_handle,
            )

        barrier()
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            args.output_dir / "final",
            args=args,
            epoch=completed_epochs,
            step=optimizer_step,
            stats={
                "eval": eval_stats if "eval_stats" in locals() else {},
                "early_stopping": {
                    "enabled": bool(args.early_stopping_patience > 0),
                    "stopped_early": stopped_early,
                    "completed_epochs": completed_epochs,
                    "best_eval_loss": (
                        best_eval_loss if best_eval_epoch is not None else None
                    ),
                    "best_epoch": best_eval_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "patience": int(args.early_stopping_patience),
                    "min_delta": float(args.early_stopping_min_delta),
                    "min_epochs": int(args.early_stopping_min_epochs),
                    "best_checkpoint": (
                        str(args.output_dir / "best")
                        if best_eval_epoch is not None
                        else None
                    ),
                },
            },
            rank=rank,
        )
        barrier()
        if natural_one_pass and rank == 0:
            audit_text = json.dumps(args.natural_epoch_audit, indent=2, sort_keys=True) + "\n"
            (args.output_dir / "natural_epoch_audit.json").write_text(
                audit_text, encoding="utf-8"
            )
            (args.output_dir / "final" / "natural_epoch_audit.json").write_text(
                audit_text, encoding="utf-8"
            )
        log("AXIS_B_SFT_COMPLETE", rank=rank, handle=log_handle)
        return 0
    finally:
        if log_handle is not None:
            log_handle.close()
        cleanup_distributed()
