#!/usr/bin/env python3
"""Generate the TCR-TRANSLATE external-13 headline predictions.

The paper's sparse test uses one 1,000-way deterministic beam for each of 13
pMHC conditions.  All 1,000 ranked slots are retained.  Invalid, incomplete,
empty, undecodable, or duplicate Axis outputs are recorded in place and are
never filtered, repaired, or replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from cli.evaluate.sample_tcr_translate_released_top20_auxiliary import (  # noqa: E402
    AxisPeptideDecoder,
    DEFAULT_MERGES,
    DEFAULT_TEXT_ENCODER,
    DEFAULT_VOCAB,
    apply_gpu_memory_fraction,
    candidate_payload,
    convert_model_sequence,
)
from pipeline.finetune.axis_b_data import (  # noqa: E402
    DEFAULT_QWEN_CONTEXT,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402


TASK = "tcr_translate_pmhc_to_cdr3b"
TRACK = "external13_sparse_headline"
PROMPT_PROTOCOL = "axis-tcr-translate-external13-headline-groups-v1-20260830"
PREDICTION_PROTOCOL = (
    "axis-tcr-translate-external13-headline-predictions-v1-20260830"
)
EXPECTED_GROUPS = 13
HEADLINE_K = 1_000
HEADLINE_NUM_BEAMS = 1_000
HEADLINE_MAX_NEW_TOKENS = 25
HEADLINE_NO_REPEAT_NGRAM_SIZE = 5
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def require_worker() -> None:
    require_compute_host(
        "TCR-TRANSLATE external-13 sampling", socket.gethostname()
    )


def headline_beam_generation_kwargs() -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": HEADLINE_NUM_BEAMS,
        "num_return_sequences": HEADLINE_K,
        "no_repeat_ngram_size": HEADLINE_NO_REPEAT_NGRAM_SIZE,
        "early_stopping": False,
        "length_penalty": 1.0,
        "max_new_tokens": HEADLINE_MAX_NEW_TOKENS,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "use_cache": True,
    }


def _integer_ids(value: Any, *, label: str) -> list[int]:
    from pipeline.finetune.axis_b_data import RAW_BPE_MAX_ID

    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if any(
        type(item) is not int or item < 0 or item > RAW_BPE_MAX_ID
        for item in value
    ):
        raise ValueError(f"{label} contains an invalid raw BPE id")
    return [int(item) for item in value]


def validate_prompt(record: Any, *, line_number: int) -> dict[str, Any]:
    label = f"external-13 prompt line {line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{label}: record must be an object")
    if record.get("task") != TASK or record.get("target_type") != "peptide":
        raise ValueError(f"{label}: task/target_type does not match TCR protocol")
    if record.get("track") != TRACK or record.get("headline") is not True:
        raise ValueError(f"{label}: prompt is not the external-13 headline track")
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{label}: id must be non-empty")
    condition = record.get("condition")
    if not isinstance(condition, dict) or set(condition) != {
        "type",
        "molecule_bpe_ids",
    }:
        raise ValueError(
            f"{label}: condition must contain only type and molecule_bpe_ids"
        )
    if condition.get("type") != "molecule":
        raise ValueError(f"{label}: condition.type must be molecule")
    if not _integer_ids(
        condition.get("molecule_bpe_ids"), label=f"{label}.molecule_bpe_ids"
    ):
        raise ValueError(f"{label}: molecule condition is empty")
    references = record.get("reference_translations")
    if not isinstance(references, list) or not references:
        raise ValueError(f"{label}: reference_translations must be non-empty")
    for index, reference in enumerate(references):
        if (
            not isinstance(reference, str)
            or not reference
            or set(reference).difference(AA_ALPHABET)
        ):
            raise ValueError(f"{label}: invalid reference at index {index}")
    pmhc = record.get("pmhc")
    if not isinstance(pmhc, dict) or not isinstance(pmhc.get("official_key"), str):
        raise ValueError(f"{label}: pmhc.official_key is required")
    return record


def load_prompt_groups(
    path: Path, *, expected_groups: int = EXPECTED_GROUPS
) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"external-13 prompts or manifest missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest.get("complete"):
        raise ValueError(f"incomplete external-13 prompt manifest: {manifest_path}")
    if manifest.get("protocol") != PROMPT_PROTOCOL:
        raise ValueError(f"unexpected prompt protocol: {manifest.get('protocol')!r}")
    if manifest.get("track") != TRACK or manifest.get("headline") is not True:
        raise ValueError("prompt manifest is not the external-13 headline track")
    contract = manifest.get("generation_contract") or {}
    expected_contract = {
        "k": HEADLINE_K,
        "num_beams": HEADLINE_NUM_BEAMS,
        "max_new_tokens": HEADLINE_MAX_NEW_TOKENS,
        "no_repeat_ngram_size": HEADLINE_NO_REPEAT_NGRAM_SIZE,
        "filter_invalid_or_duplicate_outputs": False,
        "replacement_sampling": False,
        "post_hoc_sequence_repair": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"external-13 generation contract {key}={contract.get(key)!r}, "
                f"expected {expected!r}"
            )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pmhc: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            record = validate_prompt(json.loads(raw), line_number=line_number)
            pmhc_key = record["pmhc"]["official_key"]
            if record["id"] in seen_ids or pmhc_key in seen_pmhc:
                raise ValueError(f"{path}:{line_number}: duplicate id or pMHC key")
            seen_ids.add(record["id"])
            seen_pmhc.add(pmhc_key)
            records.append(record)
    if len(records) != expected_groups:
        raise ValueError(
            f"external-13 group count changed: expected={expected_groups} "
            f"actual={len(records)}"
        )
    return records


def _prompt_model_record(prompt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": prompt["id"],
        "task": TASK,
        "target_type": "peptide",
        "condition": prompt["condition"],
        "target_bpe_ids": [0],
    }


def generate_model_rows(
    model: Any,
    collator: Any,
    prompt: Mapping[str, Any],
    device: Any,
) -> list[list[int]]:
    import torch
    from cli.evaluate.sample_axis_b_sft import model_batch_from_records

    build_batch, _collated = model_batch_from_records(
        [_prompt_model_record(prompt)], collator, device
    )
    qwen_inputs = model.build_qwen_inputs(**build_batch)
    active_length = int(qwen_inputs["attention_mask"].sum().item())
    if active_length + HEADLINE_MAX_NEW_TOKENS > DEFAULT_QWEN_CONTEXT:
        raise ValueError(
            f"{prompt['id']}: prompt length {active_length} cannot retain "
            f"max_new_tokens={HEADLINE_MAX_NEW_TOKENS} in context 512"
        )
    embeds = qwen_inputs["inputs_embeds"][:, :active_length]
    mask = qwen_inputs["attention_mask"][:, :active_length]
    with torch.no_grad():
        generated = model.qwen.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            **headline_beam_generation_kwargs(),
        )
    rows = generated.tolist()
    if len(rows) != HEADLINE_K:
        raise RuntimeError(
            f"{prompt['id']}: generate returned {len(rows)} rows, "
            f"expected {HEADLINE_K}"
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gpu-memory-fraction", type=float,
                        help="Optional CUDA allocator ceiling; fixed beams and output slots stay unchanged")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    import torch
    from pipeline.finetune.axis_b_data import AxisBSFTCollator
    from cli.evaluate.sample_axis_b_sft import load_model

    prompts = load_prompt_groups(args.prompts)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("TCR external-13 formal beam sampling requires CUDA")
    apply_gpu_memory_fraction(torch, args.gpu_memory_fraction)
    device = torch.device("cuda")
    args.molecule_condition_mode = "direct_qwen"
    args.condition_token_budget = 512
    model, tokenizer = load_model(args, device)
    collator = AxisBSFTCollator(
        tokenizer,
        qwen_max_context=DEFAULT_QWEN_CONTEXT,
        return_tensors="pt",
        condition_token_budget=512,
        molecule_condition_mode="direct_qwen",
    )
    decoder = AxisPeptideDecoder(args.merges, args.vocab)
    counters: Counter[str] = Counter()
    started = time.time()
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts, start=1):
            generated_rows = generate_model_rows(model, collator, prompt, device)
            candidates = []
            for rank, model_ids in enumerate(generated_rows, start=1):
                raw, ended, invalid = convert_model_sequence(model_ids)
                candidate = candidate_payload(
                    raw,
                    ended_with_eos=ended,
                    invalid_model_id=invalid,
                    decoder=decoder,
                    rank=rank,
                )
                candidates.append(candidate)
                counters[candidate["status"]] += 1
            payload = {
                "protocol": PREDICTION_PROTOCOL,
                "track": TRACK,
                "headline": True,
                "id": prompt["id"],
                "task": TASK,
                "pmhc": prompt["pmhc"],
                "generation": {
                    "mode": "beam",
                    "k": HEADLINE_K,
                    "num_beams": HEADLINE_NUM_BEAMS,
                    "max_new_tokens": HEADLINE_MAX_NEW_TOKENS,
                    "no_repeat_ngram_size": HEADLINE_NO_REPEAT_NGRAM_SIZE,
                    "do_sample": False,
                    "invalid_or_duplicate_filtering": False,
                    "replacement_sampling": False,
                    "post_hoc_sequence_repair": False,
                },
                "beam_candidates": candidates,
                "translations": [
                    candidate["official_translation"] for candidate in candidates
                ],
                "reference_translations": prompt["reference_translations"],
            }
            handle.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            print(f"TCR external-13 beam sampled {index}/{len(prompts)}", flush=True)
    os.replace(temporary, args.output)
    summary = {
        "complete": True,
        "protocol": PREDICTION_PROTOCOL,
        "track": TRACK,
        "headline": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "prompts": str(args.prompts.resolve()),
        "output": str(args.output.resolve()),
        "groups": len(prompts),
        "beam_slots": len(prompts) * HEADLINE_K,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "generation": {
            "k": HEADLINE_K,
            "num_beams": HEADLINE_NUM_BEAMS,
            "max_new_tokens": HEADLINE_MAX_NEW_TOKENS,
            "no_repeat_ngram_size": HEADLINE_NO_REPEAT_NGRAM_SIZE,
        },
        "counts": dict(sorted(counters.items())),
        "elapsed_seconds": time.time() - started,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
