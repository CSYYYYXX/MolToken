#!/usr/bin/env python3
"""Sample one Axis-B task from a completed conditional SFT checkpoint.

This GPU-side script writes raw generated BPE ids plus ground truth.  Strict
FDDFS graph decoding and chemistry/protein task metrics remain separate CPU
worker steps, matching the existing unconditional-generation workflow.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.axis_b_data import (  # noqa: E402
    AxisBIndex,
    AxisBJsonlDataset,
    AxisBSFTCollator,
    DEFAULT_QWEN_CONTEXT,
    EOS_TOKEN_ID,
    MODEL_TOKEN_OFFSET,
    PAD_TOKEN_ID,
    RAW_BPE_MAX_ID,
)
from pipeline.finetune.axis_b_model import (  # noqa: E402
    AxisBConditionalModel,
    load_projection_state,
)
from pipeline.finetune.active_scope import ACTIVE_TRAIN_TASK_IDS  # noqa: E402
from pipeline.paths import PipelinePaths  # noqa: E402


DATA_ROOT = PipelinePaths().data_root
DEFAULT_DATA = DATA_ROOT / "finetune_datasets_bpe_fddfs_mf80000_moses_guacamol_v1"
DEFAULT_TEXT_ENCODER = Path(
    os.environ.get(
        "HOLOMOL_TEXT_ENCODER",
        str(_moltoken_paths.DEFAULT_TEXT_ENCODER),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument(
        "--molecule-condition-mode",
        choices=("direct_qwen",),
        default="direct_qwen",
        help="Current formal checkpoints use frozen FDDFS/BPE ids directly in Qwen.",
    )
    parser.add_argument("--split", choices=("eval", "test"), default="test")
    parser.add_argument("--task", required=True, choices=ACTIVE_TRAIN_TASK_IDS)
    parser.add_argument(
        "--source-contains",
        default=None,
        help="Only use records whose indexed source path contains this string.",
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=DEFAULT_QWEN_CONTEXT)
    parser.add_argument("--condition-token-budget", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_molecule_condition_mode(checkpoint: Path, requested: str) -> str:
    if requested != "direct_qwen":
        raise ValueError("current formal sampling supports direct_qwen only")
    config_path = Path(checkpoint) / "axis_b_config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        mode = payload.get("molecule_condition_mode")
        metadata = payload.get("metadata") or {}
        mode = mode or metadata.get("molecule_condition_mode")
        if mode is not None and mode != "direct_qwen":
            raise ValueError(
                f"checkpoint records retired molecule condition mode {mode!r}; "
                "current formal sampling requires direct_qwen"
            )
    return "direct_qwen"


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[AxisBConditionalModel, Any]:
    """Load offline weights in the requested dtype on either device, without fallback."""
    from moltoken import load_sft

    args.molecule_condition_mode = resolve_molecule_condition_mode(
        args.checkpoint, getattr(args, "molecule_condition_mode", "auto")
    )

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    model, tokenizer = load_sft(args.checkpoint, text_encoder=args.text_encoder,
                                device=device, dtype=dtype)
    if int(getattr(args, "condition_token_budget", 512)) != model.condition_token_budget:
        raise ValueError("sampling condition token budget differs from checkpoint metadata")
    model.qwen.config.use_cache = True
    return model, tokenizer


def model_batch_from_records(
    records: list[dict[str, Any]],
    collator: AxisBSFTCollator,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
    collated = collator(records)
    batch_size = len(records)
    # Generation prompt stops after target type and BOS.  Ground-truth target
    # ids are retained only in the JSONL output for downstream metrics.
    prompt_ids = collated["decoder_input_ids"][:, :2].to(device)
    prompt_mask = torch.ones((batch_size, 2), dtype=torch.long, device=device)
    build = {
        "decoder_input_ids": prompt_ids,
        "decoder_attention_mask": prompt_mask,
        "decoder_labels": None,
        "text_input_ids": collated["text_input_ids"].to(device),
        "text_attention_mask": collated["text_attention_mask"].to(device),
        "text_present": collated["text_present"].to(device),
        "molecule_input_ids": collated["molecule_input_ids"].to(device),
        "molecule_attention_mask": collated["molecule_attention_mask"].to(device),
        "molecule_present": collated["molecule_present"].to(device),
        "prefix_lengths": collated["prefix_lengths"].to(device),
        "text_prefix_lengths": collated["text_prefix_lengths"].to(device),
        "molecule_prefix_lengths": collated["molecule_prefix_lengths"].to(device),
    }
    return build, collated


def condition_type(record: dict[str, Any]) -> str:
    condition = record.get("condition")
    if not isinstance(condition, dict):
        raise ValueError("record condition must be an object")
    value = condition.get("type")
    if value not in {"none", "text", "molecule", "both"}:
        raise ValueError(f"unsupported condition.type={value!r}")
    return str(value)


def collated_target_raw(collated: dict[str, Any], row: int) -> list[int]:
    """Return the target actually presented to the model after context truncation."""

    ids = collated["decoder_input_ids"][row].tolist()
    mask = collated["decoder_attention_mask"][row].tolist()
    raw: list[int] = []
    for token_id, active in zip(ids[2:], mask[2:]):
        if not int(active) or int(token_id) == PAD_TOKEN_ID:
            break
        token_id = int(token_id)
        if token_id == EOS_TOKEN_ID:
            break
        raw.append(token_id - MODEL_TOKEN_OFFSET)
    return raw


def _generation_groups(attention_mask: torch.Tensor) -> list[tuple[int, list[int]]]:
    """Group rows by active prompt length for safe ``inputs_embeds`` generation.

    ``transformers.generate`` consumes the final logit column for every row.
    Right-padding variable-length soft prefixes would therefore query a zero
    pad position for shorter rows.  Equal-length groups keep that final
    column at each row's actual BOS position.
    """

    lengths = attention_mask.to(dtype=torch.long).sum(dim=1).tolist()
    grouped: dict[int, list[int]] = {}
    for row, length in enumerate(lengths):
        grouped.setdefault(int(length), []).append(row)
    return list(grouped.items())


@torch.no_grad()
def generate_batch(
    model: AxisBConditionalModel,
    build_batch: dict[str, torch.Tensor | None],
    args: argparse.Namespace,
) -> tuple[list[list[int]], list[bool], list[bool]]:
    # Prefix spans may differ within one condition type.  Build once, then
    # generate equal-length prompt groups so HF never reads a right-pad query.
    qwen_inputs = model.build_qwen_inputs(**build_batch)
    context_limit = int(getattr(args, "max_context", model.qwen.config.max_position_embeddings))
    batch_size = int(qwen_inputs["attention_mask"].shape[0])
    sequences: list[list[int] | None] = [None] * batch_size
    ended_with_eos: list[bool | None] = [None] * batch_size
    invalid_model_id: list[bool | None] = [None] * batch_size
    for active_length, rows in _generation_groups(qwen_inputs["attention_mask"]):
        row_ids = torch.tensor(rows, dtype=torch.long, device=qwen_inputs["inputs_embeds"].device)
        embeds = qwen_inputs["inputs_embeds"].index_select(0, row_ids)[:, :active_length]
        mask = qwen_inputs["attention_mask"].index_select(0, row_ids)[:, :active_length]
        max_new_tokens = min(int(args.max_new_tokens), max(1, context_limit - active_length))
        generation_args: dict[str, Any] = {
            "inputs_embeds": embeds,
            "attention_mask": mask,
            "do_sample": not args.greedy,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
            "use_cache": True,
        }
        if not args.greedy:
            generation_args.update(
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
            )
        generated = model.qwen.generate(**generation_args)
        for local, row in enumerate(generated.tolist()):
            raw: list[int] = []
            eos = False
            invalid = False
            for token_id in row:
                token_id = int(token_id)
                if token_id == EOS_TOKEN_ID:
                    eos = True
                    break
                if token_id == PAD_TOKEN_ID:
                    # PAD is only valid after an EOS terminator.  A PAD emitted
                    # before EOS means generation stopped without a valid target.
                    invalid = True
                    break
                candidate = token_id - MODEL_TOKEN_OFFSET
                if candidate < 0 or candidate > RAW_BPE_MAX_ID:
                    invalid = True
                    break
                raw.append(candidate)
            original_row = rows[local]
            sequences[original_row] = raw
            ended_with_eos[original_row] = eos
            invalid_model_id[original_row] = invalid
    return (
        [row if row is not None else [] for row in sequences],
        [bool(value) for value in ended_with_eos],
        [bool(value) for value in invalid_model_id],
    )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = AxisBIndex.load(args.data_root, args.index_dir)
    dataset = AxisBJsonlDataset(
        args.data_root,
        split=args.split,
        index_dir=args.index_dir,
        index=index,
        task=args.task,
    )
    positions = list(range(len(dataset)))
    if args.source_contains:
        needle = str(args.source_contains)
        positions = [
            position
            for position in positions
            if needle in str(dataset[position].get("_axis_b", {}).get("source", ""))
        ]
        if not positions:
            raise KeyError(
                f"no records for task={args.task!r} with source filter={needle!r}"
            )
    rng = random.Random(args.seed)
    rng.shuffle(positions)
    positions = positions[: min(args.n_samples, len(positions))]
    records = [dataset[position] for position in positions]

    args.molecule_condition_mode = resolve_molecule_condition_mode(
        args.checkpoint, args.molecule_condition_mode
    )
    model, tokenizer = load_model(args, device)
    collator = AxisBSFTCollator(
        tokenizer,
        qwen_max_context=int(args.max_context),
        return_tensors="pt",
        condition_token_budget=int(args.condition_token_budget),
        molecule_condition_mode=args.molecule_condition_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    eos_count = 0
    invalid_count = 0
    exact_count = 0
    exact_comparable_count = 0
    truncated_target_count = 0
    empty_generation_count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            groups: dict[str, list[dict[str, Any]]] = {}
            for record in batch_records:
                groups.setdefault(condition_type(record), []).append(record)
            for grouped_records in groups.values():
                build_batch, collated = model_batch_from_records(
                    grouped_records, collator, device
                )
                generated, ended, invalid = generate_batch(model, build_batch, args)
                for row, (record, raw_ids, has_eos, has_invalid) in enumerate(
                    zip(grouped_records, generated, ended, invalid)
                ):
                    target = collated_target_raw(collated, row)
                    target_was_truncated = bool(
                        collated["target_truncated"][row].item()
                    )
                    empty_generation = not raw_ids
                    target_exceeds_max_new_tokens = len(target) + 1 > int(
                        args.max_new_tokens
                    )
                    token_exact_comparable = (
                        not target_exceeds_max_new_tokens and not target_was_truncated
                    )
                    # A token match only counts when generation terminated
                    # normally, every emitted id was valid, and the comparison
                    # target is exactly the context-limited target seen by the
                    # collator.
                    exact = bool(
                        token_exact_comparable
                        and has_eos
                        and not has_invalid
                        and raw_ids == target
                    )
                    handle.write(
                        json.dumps(
                            {
                                "id": record.get("id"),
                                "task": record.get("task"),
                                "meta": record.get("meta"),
                                "prompt_only": bool(record.get("_prompt_only", False)),
                                "target_type": record.get("target_type"),
                                "generated_bpe_ids": raw_ids,
                                "bpe_ids": raw_ids,
                                "target_bpe_ids": target,
                                "target_truncated": target_was_truncated,
                                "target_exceeds_max_new_tokens": target_exceeds_max_new_tokens,
                                "token_exact_comparable": token_exact_comparable,
                                "empty_generation": empty_generation,
                                "ended_with_eos": has_eos,
                                "invalid_model_id": has_invalid,
                                "token_exact_match": exact,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    written += 1
                    eos_count += int(has_eos)
                    invalid_count += int(has_invalid)
                    exact_count += int(exact)
                    exact_comparable_count += int(token_exact_comparable)
                    truncated_target_count += int(target_was_truncated)
                    empty_generation_count += int(empty_generation)
            print(f"sampled {written}/{len(records)}", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "task": args.task,
        "samples": written,
        "ended_with_eos": eos_count,
        "invalid_model_id": invalid_count,
        "target_truncated": truncated_target_count,
        "empty_generation": empty_generation_count,
        "token_exact_match": exact_count,
        "token_exact_comparable": exact_comparable_count,
        "token_exact_match_rate": (
            exact_count / exact_comparable_count if exact_comparable_count else None
        ),
        "elapsed_sec": time.time() - t0,
        "generation": {
            "greedy": args.greedy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context,
            "seed": args.seed,
        },
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
