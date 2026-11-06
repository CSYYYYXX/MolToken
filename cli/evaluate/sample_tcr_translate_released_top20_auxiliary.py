#!/usr/bin/env python3
"""Generate the released top-20 auxiliary view from an Axis checkpoint.

One deterministic 300-beam search returns 100 ranked candidates for each of
the 20 released pMHC prompts.  A separate greedy decode supplies the official
Char-BLEU hypothesis.  Every requested rank is retained: incomplete, invalid,
undecodable, empty, and duplicate generations are never removed or replaced.
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

from pipeline.finetune.axis_b_data import (  # noqa: E402
    DEFAULT_QWEN_CONTEXT,
    EOS_TOKEN_ID,
    MODEL_TOKEN_OFFSET,
    PAD_TOKEN_ID,
    RAW_BPE_MAX_ID,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402


DATA_ROOT = Path(
    str(Path(__file__).resolve().parents[2])
)
DEFAULT_TEXT_ENCODER = DATA_ROOT / "all-mpnet-base-v2"
DEFAULT_VOCAB = DATA_ROOT / "fddfs_guide_recount_v1/vocab.json"
DEFAULT_MERGES = (
    DATA_ROOT
    / "codebook_fddfs_prefix70_mf80000_v1/merges.csv"
)

TASK = "tcr_translate_pmhc_to_cdr3b"
TRACK = "released_top20_validation_auxiliary"
PROMPT_PROTOCOL = "axis-tcr-translate-released-top20-auxiliary-groups-v1-20260830"
PREDICTION_PROTOCOL = "axis-tcr-translate-released-top20-auxiliary-predictions-v1-20260830"
EXPECTED_GROUPS = 20
OFFICIAL_K = 100
OFFICIAL_NUM_BEAMS = 300
OFFICIAL_MAX_NEW_TOKENS = 25
OFFICIAL_NO_REPEAT_NGRAM_SIZE = 5
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def require_worker() -> None:
    require_compute_host("TCR-TRANSLATE checkpoint sampling", socket.gethostname())


def official_beam_generation_kwargs() -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": OFFICIAL_NUM_BEAMS,
        "num_return_sequences": OFFICIAL_K,
        "no_repeat_ngram_size": OFFICIAL_NO_REPEAT_NGRAM_SIZE,
        "early_stopping": False,
        "length_penalty": 1.0,
        "max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "use_cache": True,
    }


def official_greedy_generation_kwargs() -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 0,
        "max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "use_cache": True,
    }


def _integer_ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if any(
        type(item) is not int or item < 0 or item > RAW_BPE_MAX_ID
        for item in value
    ):
        raise ValueError(f"{label} contains an invalid raw BPE id")
    return [int(item) for item in value]


def validate_prompt(record: Any, *, line_number: int) -> dict[str, Any]:
    label = f"prompt line {line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{label}: record must be an object")
    if record.get("task") != TASK or record.get("target_type") != "peptide":
        raise ValueError(f"{label}: task/target_type does not match TCR protocol")
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
    molecule_ids = _integer_ids(
        condition.get("molecule_bpe_ids"), label=f"{label} molecule_bpe_ids"
    )
    if not molecule_ids:
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


def load_prompt_groups(path: Path, *, expected_groups: int = EXPECTED_GROUPS) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"grouped prompts or manifest missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest.get("complete"):
        raise ValueError(f"incomplete grouped prompt manifest: {manifest_path}")
    if manifest.get("protocol") != PROMPT_PROTOCOL:
        raise ValueError(
            f"unexpected grouped prompt protocol: {manifest.get('protocol')!r}"
        )
    if manifest.get("track") != TRACK or manifest.get("headline") is not False:
        raise ValueError("grouped prompts are not the released top-20 auxiliary track")
    contract = manifest.get("generation_contract") or {}
    expected_contract = {
        "k": OFFICIAL_K,
        "num_beams": OFFICIAL_NUM_BEAMS,
        "max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
        "no_repeat_ngram_size": OFFICIAL_NO_REPEAT_NGRAM_SIZE,
        "filter_invalid_or_duplicate_outputs": False,
        "replacement_sampling": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"grouped prompt generation contract {key}={contract.get(key)!r}, "
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
            if record["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record['id']}")
            official_key = record["pmhc"]["official_key"]
            if official_key in seen_pmhc:
                raise ValueError(
                    f"{path}:{line_number}: duplicate pMHC key {official_key}"
                )
            seen_ids.add(record["id"])
            seen_pmhc.add(official_key)
            records.append(record)
    if len(records) != expected_groups:
        raise ValueError(
            f"grouped prompt count changed: expected={expected_groups} actual={len(records)}"
        )
    return records


def convert_model_sequence(model_ids: Sequence[int]) -> tuple[list[int], bool, bool]:
    raw: list[int] = []
    ended_with_eos = False
    invalid_model_id = False
    for value in model_ids:
        token_id = int(value)
        if token_id == EOS_TOKEN_ID:
            ended_with_eos = True
            break
        if token_id == PAD_TOKEN_ID:
            invalid_model_id = True
            break
        candidate = token_id - MODEL_TOKEN_OFFSET
        if candidate < 0 or candidate > RAW_BPE_MAX_ID:
            invalid_model_id = True
            break
        raw.append(candidate)
    return raw, ended_with_eos, invalid_model_id


class AxisPeptideDecoder:
    def __init__(self, merges_path: Path, vocab_path: Path) -> None:
        from cli.evaluate.eval_axis_b_samples import BPEDecoder, FDDFSVocab

        self.vocab = FDDFSVocab.load(vocab_path)
        self.decoder = BPEDecoder(merges_path, self.vocab)

    def decode(self, raw_ids: list[int]) -> str:
        from pipeline.graph.biopolymer_sequence import graph_to_biopolymer_sequence
        from pipeline.serialization.fddfs import deserialize_fddfs

        _base, tokens = self.decoder.decode_raw_ids(raw_ids)
        graph = deserialize_fddfs(tokens, self.vocab.default_edge)
        sequence = graph_to_biopolymer_sequence(graph, target_type="peptide")
        if not sequence or set(sequence).difference(AA_ALPHABET):
            raise ValueError("decoded graph is not a non-empty canonical peptide")
        return sequence


def candidate_payload(
    raw_ids: list[int],
    *,
    ended_with_eos: bool,
    invalid_model_id: bool,
    decoder: Any,
    rank: int,
) -> dict[str, Any]:
    decoded: str | None = None
    status: str
    error: str | None = None
    if invalid_model_id:
        status = "invalid_model_id"
    elif not ended_with_eos:
        status = "missing_eos"
    elif not raw_ids:
        status = "empty_generation"
    else:
        try:
            decoded = decoder.decode(raw_ids)
            status = "valid"
        except ValueError as exc:
            status = "decode_error"
            error = str(exc)
    return {
        "rank": int(rank),
        "bpe_ids": list(raw_ids),
        "ended_with_eos": bool(ended_with_eos),
        "invalid_model_id": bool(invalid_model_id),
        "status": status,
        "decoded_cdr3b": decoded,
        # The released evaluator consumes strings.  An invalid Axis slot is
        # represented by an empty hypothesis so it remains a miss at its
        # original rank; the detailed status above remains authoritative.
        "official_translation": decoded if decoded is not None else "",
        "decode_error": error,
    }


def _prompt_model_record(prompt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": prompt["id"],
        "task": TASK,
        "target_type": "peptide",
        "condition": prompt["condition"],
        # AxisBSFTCollator requires a target body, but model_batch_from_records
        # slices the generation prompt after target-type+BOS.  This raw token
        # is never embedded or exposed to generation.
        "target_bpe_ids": [0],
    }


def generate_model_rows(
    model: Any,
    collator: Any,
    prompt: Mapping[str, Any],
    device: Any,
) -> tuple[list[list[int]], list[int]]:
    import torch
    from cli.evaluate.sample_axis_b_sft import model_batch_from_records

    build_batch, _collated = model_batch_from_records(
        [_prompt_model_record(prompt)], collator, device
    )
    qwen_inputs = model.build_qwen_inputs(**build_batch)
    active_length = int(qwen_inputs["attention_mask"].sum().item())
    if active_length + OFFICIAL_MAX_NEW_TOKENS > DEFAULT_QWEN_CONTEXT:
        raise ValueError(
            f"{prompt['id']}: prompt length {active_length} cannot retain official "
            f"max_new_tokens={OFFICIAL_MAX_NEW_TOKENS} in context 512"
        )
    embeds = qwen_inputs["inputs_embeds"][:, :active_length]
    mask = qwen_inputs["attention_mask"][:, :active_length]
    with torch.no_grad():
        beam = model.qwen.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            **official_beam_generation_kwargs(),
        )
        greedy = model.qwen.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            **official_greedy_generation_kwargs(),
        )
    beam_rows = beam.tolist()
    greedy_rows = greedy.tolist()
    if len(beam_rows) != OFFICIAL_K or len(greedy_rows) != 1:
        raise RuntimeError(
            f"{prompt['id']}: generate returned beam={len(beam_rows)} "
            f"greedy={len(greedy_rows)}"
        )
    return beam_rows, greedy_rows[0]


def apply_gpu_memory_fraction(torch, fraction: float | None) -> None:
    """Optional allocator ceiling; never shrink beams or retry failed sampling."""
    if fraction is None:
        return
    if not 0 < fraction <= 1:
        raise ValueError("gpu-memory-fraction must lie in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(fraction)


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
        raise RuntimeError("TCR released top-20 formal beam sampling requires CUDA")
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
            beam_rows, greedy_row = generate_model_rows(
                model, collator, prompt, device
            )
            candidates = []
            for rank, model_ids in enumerate(beam_rows, start=1):
                raw, ended, invalid = convert_model_sequence(model_ids)
                candidate = candidate_payload(
                    raw,
                    ended_with_eos=ended,
                    invalid_model_id=invalid,
                    decoder=decoder,
                    rank=rank,
                )
                candidates.append(candidate)
                counters[f"beam_{candidate['status']}"] += 1
            greedy_raw, greedy_ended, greedy_invalid = convert_model_sequence(
                greedy_row
            )
            greedy = candidate_payload(
                greedy_raw,
                ended_with_eos=greedy_ended,
                invalid_model_id=greedy_invalid,
                decoder=decoder,
                rank=1,
            )
            counters[f"greedy_{greedy['status']}"] += 1
            payload = {
                "protocol": PREDICTION_PROTOCOL,
                "track": TRACK,
                "headline": False,
                "id": prompt["id"],
                "task": TASK,
                "pmhc": prompt["pmhc"],
                "generation": {
                    "mode": "beam",
                    "k": OFFICIAL_K,
                    "num_beams": OFFICIAL_NUM_BEAMS,
                    "max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
                    "no_repeat_ngram_size": OFFICIAL_NO_REPEAT_NGRAM_SIZE,
                    "do_sample": False,
                    "invalid_or_duplicate_filtering": False,
                    "replacement_sampling": False,
                },
                "beam_candidates": candidates,
                "translations": [
                    candidate["official_translation"] for candidate in candidates
                ],
                "greedy_candidate": greedy,
                "greedy_translation": greedy["official_translation"],
                "reference_translations": prompt["reference_translations"],
            }
            handle.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            print(f"TCR released top-20 auxiliary sampled {index}/{len(prompts)}", flush=True)
    os.replace(temporary, args.output)
    summary = {
        "complete": True,
        "protocol": PREDICTION_PROTOCOL,
        "track": TRACK,
        "headline": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "prompts": str(args.prompts.resolve()),
        "output": str(args.output.resolve()),
        "groups": len(prompts),
        "beam_slots": len(prompts) * OFFICIAL_K,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "generation": {
            "k": OFFICIAL_K,
            "num_beams": OFFICIAL_NUM_BEAMS,
            "max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
            "no_repeat_ngram_size": OFFICIAL_NO_REPEAT_NGRAM_SIZE,
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
