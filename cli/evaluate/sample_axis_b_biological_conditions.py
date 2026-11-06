#!/usr/bin/env python3
"""Sample a frozen biological condition grid from Full/best.

This entry point never opens a benchmark train/validation/test file.  It builds
prompt-only records from the frozen benchmark condition grid, so official test
targets cannot leak into checkpoint or sampling-parameter selection.  Every
requested slot is written exactly once; invalid or incomplete generations are
not filtered or replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.axis_b_data import (  # noqa: E402
    AxisBSFTCollator,
    DEFAULT_QWEN_CONTEXT,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402
from cli.evaluate.pepbench_generation_protocol import (  # noqa: E402
    TASK as PEP_TASK, FORMAL_SAMPLES as PEP_SAMPLES, load_schedule,
)


PROTOCOL = "axis-biological-frozen-condition-sampling-v1-20260902"
DNA_TASK = "dna_diffusion_celltype_generation"
RFAM_TASK = "rfamgen_figure2_reconstruction17"
DPLM_TASK = "dplm_length_conditioned_protein"
DPLM_CONDITIONS = ("100", "200", "300", "400")
DPLM_SAMPLES_PER_CONDITION = 40
DNA_HEADLINE_CONDITIONS = ("GM12878", "K562", "HepG2")
DNA_SUPPLEMENTAL_CONDITION = "hESCT0"
RFAM_CONDITIONS = (
    "RF00001",
    "RF00003",
    "RF00004",
    "RF00017",
    "RF00029",
    "RF00059",
    "RF00100",
    "RF00163",
    "RF00174",
    "RF00177",
    "RF00230",
    "RF00906",
    "RF00994",
    "RF01960",
    "RF02271",
    "RF04021",
    "RF04088",
)
FORMAL_SAMPLES_PER_CONDITION = 1000


def task_spec(task: str, include_dna_supplemental: bool = False) -> dict[str, Any]:
    if task == PEP_TASK:
        if include_dna_supplemental:
            raise ValueError("--include-dna-supplemental is only valid for DNA")
        return {
            "target_type": "peptide", "conditions": tuple(str(n) for n in range(5, 81)),
            "default_batch_size": 32, "default_max_new_tokens": DEFAULT_QWEN_CONTEXT,
            "samples_per_condition": 0, "default_seed": 42,
            "condition_scope": (
                "PepBench-Gen pretraining-exposed reconstruction; fixed TRAIN-derived "
                "length schedule, bare numeric prompts, free EOS; no repair or resampling"
            ),
        }
    if task == DNA_TASK:
        conditions = DNA_HEADLINE_CONDITIONS
        if include_dna_supplemental:
            conditions += (DNA_SUPPLEMENTAL_CONDITION,)
        return {
            "target_type": "dna",
            "conditions": conditions,
            "default_batch_size": 16,
            "default_max_new_tokens": 128,
            "samples_per_condition": FORMAL_SAMPLES_PER_CONDITION,
            "default_seed": 20260902,
            "condition_scope": (
                "official three-cell headline"
                if not include_dna_supplemental
                else "official three-cell headline plus hESCT0 supplemental"
            ),
        }
    if task == RFAM_TASK:
        if include_dna_supplemental:
            raise ValueError("--include-dna-supplemental is only valid for DNA")
        return {
            "target_type": "rna",
            "conditions": RFAM_CONDITIONS,
            "default_batch_size": 1,
            "default_max_new_tokens": DEFAULT_QWEN_CONTEXT,
            "samples_per_condition": FORMAL_SAMPLES_PER_CONDITION,
            "default_seed": 20260902,
            "condition_scope": (
                "Axis RfamGen Figure 2 paper-protocol reconstruction-17; "
                "RF00005_CCU excluded because exact public membership is unavailable"
            ),
        }
    if task == DPLM_TASK:
        if include_dna_supplemental:
            raise ValueError("--include-dna-supplemental is only valid for DNA")
        return {
            "target_type": "protein",
            "conditions": DPLM_CONDITIONS,
            "default_batch_size": 8,
            "default_max_new_tokens": DEFAULT_QWEN_CONTEXT,
            "samples_per_condition": DPLM_SAMPLES_PER_CONDITION,
            "default_seed": 42,
            "condition_scope": (
                "DPLM-aligned context/representation-filtered length-100–400 "
                "partial protocol; released-code 40 raw slots per length; "
                "Axis free EOS, no forced residue length or resampling"
            ),
        }
    raise ValueError(f"unsupported task: {task!r}")


def prompt_record(task: str, target_type: str, condition: str, slot: int) -> dict[str, Any]:
    return {
        "id": f"{task}:{condition}:sample-{slot:06d}",
        "task": task,
        "target_type": target_type,
        "condition": {
            "type": "text",
            "text": condition,
            "molecule_bpe_ids": [],
        },
        # The collator requires a body.  The shared generation helper slices
        # the decoder prompt immediately after target-type + BOS, so this raw
        # token is never embedded or exposed to generation.
        "target_bpe_ids": [0],
        "_prompt_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--task", choices=(DNA_TASK, RFAM_TASK, DPLM_TASK, PEP_TASK), required=True)
    parser.add_argument(
        "--condition-manifest", type=Path,
        help="PepBench only: frozen TRAIN-derived length schedule shared by both models",
    )
    parser.add_argument("--include-dna-supplemental", action="store_true")
    parser.add_argument(
        "--samples-per-condition", type=int, default=0,
        help="0 selects DNA/Rfam 1000 or DPLM 40; PepBench uses its condition manifest",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 selects the task default (DNA 16, Rfam 1, DPLM 8, PepBench 32)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 selects 128 for DNA and the remaining context for other tasks",
    )
    parser.add_argument("--max-context", type=int, default=DEFAULT_QWEN_CONTEXT)
    parser.add_argument("--condition-token-budget", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Explicit sampling device; no automatic CPU or dtype fallback")
    parser.add_argument("--cuda-device-index", type=int, default=0,
                        help="Local index among scheduler-assigned GPUs")
    parser.add_argument("--condition-shards", type=int, default=1)
    parser.add_argument("--condition-shard-index", type=int, default=0)
    parser.add_argument("--gpu-memory-fraction", type=float,
                        help="Optional CUDA allocator fraction when sharing a worker; OOM is an error, not a resampling trigger")
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="Torch intra-op threads when --device cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace, spec: Mapping[str, Any]) -> None:
    if args.task == PEP_TASK:
        if args.condition_manifest is None or args.samples_per_condition != 0:
            raise ValueError("PepBench requires --condition-manifest, not --samples-per-condition")
    elif args.condition_manifest is not None:
        raise ValueError("--condition-manifest is only used for PepBench")
    elif args.samples_per_condition <= 0:
        raise ValueError("--samples-per-condition must be positive")
    if args.batch_size < 0 or args.max_new_tokens < 0:
        raise ValueError("batch size and max-new-tokens must be non-negative")
    if args.max_context != DEFAULT_QWEN_CONTEXT:
        raise ValueError("formal biological sampling requires context 512")
    if not 0.0 < args.temperature:
        raise ValueError("temperature must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must lie in (0, 1]")
    if args.top_k < 0:
        raise ValueError("top-k must be non-negative")
    if args.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    if args.gpu_memory_fraction is not None:
        if args.device != "cuda" or not 0 < args.gpu_memory_fraction <= 1:
            raise ValueError("gpu-memory-fraction requires CUDA and must lie in (0, 1]")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".summary.json").exists():
        raise FileExistsError(f"refusing to overwrite sampling output: {args.output}")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.text_encoder.is_dir():
        raise FileNotFoundError(args.text_encoder)
    if not tuple(spec["conditions"]):
        raise ValueError("condition grid is empty")
    if not 0 <= args.condition_shard_index < args.condition_shards <= len(spec["conditions"]):
        raise ValueError("condition shard index/count are outside the condition grid")
    if args.task == PEP_TASK and args.condition_shards != 1:
        raise ValueError("PepBench's slot schedule does not support condition sharding")


def sampling_batches(args, spec, schedule):
    """Keep the fixed prompt order; never inspect or substitute output targets."""
    if schedule is not None:
        conditions = schedule["conditions"]
        for start in range(0, len(conditions), args.batch_size):
            yield [prompt_record(args.task, spec["target_type"], conditions[i], i)
                   for i in range(start, min(start + args.batch_size, len(conditions)))]
    else:
        for condition in spec["conditions"]:
            for start in range(0, args.samples_per_condition, args.batch_size):
                yield [prompt_record(args.task, spec["target_type"], condition, i)
                       for i in range(start, min(start + args.batch_size, args.samples_per_condition))]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = task_spec(args.task, args.include_dna_supplemental)
    if args.samples_per_condition == 0:
        args.samples_per_condition = spec["samples_per_condition"]
    if args.seed is None:
        args.seed = spec["default_seed"]
    _validate_args(args, spec)
    all_conditions = tuple(spec["conditions"])
    spec = {**spec, "conditions": all_conditions[args.condition_shard_index::args.condition_shards]}
    schedule = (load_schedule(args.condition_manifest, seed=args.seed)
                if args.task == PEP_TASK else None)
    require_compute_host("Axis biological checkpoint sampling", socket.gethostname())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("checkpoint sampling requires one visible CUDA GPU")
    if device.type == "cuda":
        torch.cuda.set_device(args.cuda_device_index)
        device = torch.device("cuda", args.cuda_device_index)
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    from transformers import AutoTokenizer
    from cli.evaluate.sample_axis_b_sft import (
        generate_batch,
        load_model,
        model_batch_from_records,
    )

    args.batch_size = int(args.batch_size or spec["default_batch_size"])
    args.max_new_tokens = int(args.max_new_tokens or spec["default_max_new_tokens"])
    args.molecule_condition_mode = "direct_qwen"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, _tokenizer_from_model = load_model(args, device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.text_encoder), local_files_only=True
    )
    collator = AxisBSFTCollator(
        tokenizer,
        qwen_max_context=args.max_context,
        return_tensors="pt",
        condition_token_budget=args.condition_token_budget,
        molecule_condition_mode="direct_qwen",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    per_condition: dict[str, Counter[str]] = {
        condition: Counter() for condition in spec["conditions"]
    }
    started = time.time()
    with args.output.open("x", encoding="utf-8") as handle:
        for records in sampling_batches(args, spec, schedule):
            build, _collated = model_batch_from_records(records, collator, device)
            generated, ended, invalid = generate_batch(model, build, args)
            for record, raw_ids, has_eos, has_invalid in zip(
                records, generated, ended, invalid, strict=True
            ):
                condition = record["condition"]["text"]
                empty = not raw_ids
                payload = {
                    "protocol": PROTOCOL,
                    "id": record["id"],
                    "task": args.task,
                    "target_type": spec["target_type"],
                    "condition": condition,
                    "prompt_only": True,
                    "sample_index": int(record["id"].rsplit("-", 1)[-1]),
                    "generated_bpe_ids": raw_ids,
                    "ended_with_eos": bool(has_eos),
                    "invalid_model_id": bool(has_invalid),
                    "empty_generation": bool(empty),
                }
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                for counter in (counts, per_condition[condition]):
                    counter["requested"] += 1
                    counter["ended_with_eos"] += int(has_eos)
                    counter["invalid_model_id"] += int(has_invalid)
                    counter["empty_generation"] += int(empty)
            handle.flush()
            print(json.dumps({"sampled_total": counts["requested"],
                              "last_condition": condition}, separators=(",", ":")), flush=True)

    expected = (len(schedule["conditions"]) if schedule is not None
                else len(spec["conditions"]) * args.samples_per_condition)
    formal_slots = (PEP_SAMPLES if schedule is not None
                    else len(all_conditions) * spec["samples_per_condition"])
    if counts["requested"] != expected:
        raise RuntimeError(f"wrote {counts['requested']} rows, expected {expected}")
    summary = {
        "protocol": PROTOCOL,
        "complete": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "task": args.task,
        "target_type": spec["target_type"],
        "condition_scope": spec["condition_scope"],
        "conditions": list(spec["conditions"]),
        "condition_shards": args.condition_shards,
        "condition_shard_index": args.condition_shard_index,
        "samples_per_condition": args.samples_per_condition,
        "formal_sample_count": spec["samples_per_condition"] if schedule is None else None,
        "formal_slot_count": formal_slots,
        "formal_count_complete": expected == formal_slots,
        "condition_manifest": str(args.condition_manifest) if schedule is not None else None,
        "length_schedule": schedule,
        "official_test_records_read": 0,
        "sampling_parameter_selection_rule": (
            "freeze on train-derived validation only; official benchmark test is final-use only"
        ),
        "failed_slots_replaced": False,
        "counts": dict(counts),
        "per_condition": {key: dict(value) for key, value in per_condition.items()},
        "generation": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context,
            "greedy": args.greedy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": str(device),
            "qwen_dtype": str(model.qwen.dtype),
            "text_encoder_dtype": str(model.text_encoder.dtype),
            "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        },
        "elapsed_sec": time.time() - started,
        "output": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
