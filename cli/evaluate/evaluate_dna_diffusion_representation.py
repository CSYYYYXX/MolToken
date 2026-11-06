#!/usr/bin/env python3
"""Report strict DNA-Diffusion representation-level generation metrics.

This adapter intentionally does not compute or claim the paper's Enformer,
motif, locus, or regulatory headline metrics.  It validates the frozen
three-condition, 1,000-slot protocol and reports only model/FDDFS decoding
status, strict 200-bp validity, recovered length, and uniqueness.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


from cli.evaluate.decode_axis_b_biological_conditions import (
    PROTOCOL as DECODE_PROTOCOL,
)
from cli.evaluate.sample_axis_b_biological_conditions import (
    DNA_HEADLINE_CONDITIONS,
    DNA_TASK,
    FORMAL_SAMPLES_PER_CONDITION,
    PROTOCOL as SAMPLE_PROTOCOL,
)


PROTOCOL = "axis-dna-diffusion-representation-metrics-v1-20260902"
ALLOWED_STATUSES = frozenset(
    {
        "invalid_model_id",
        "missing_eos",
        "empty_generation",
        "decode_error",
        "wrong_length",
        "valid",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--decoded-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_jsonl(path: Path, *, protocol: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            row_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(row, dict) or row.get("protocol") != protocol:
                raise ValueError(f"{path}:{line_number}: protocol mismatch")
            if not isinstance(row_id, str) or not row_id or row_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate id")
            if row.get("task") != DNA_TASK or row.get("target_type") != "dna":
                raise ValueError(f"{path}:{line_number}: not a DNA-Diffusion row")
            seen.add(row_id)
            rows.append(row)
    return rows


def _length_summary(lengths: Sequence[int]) -> dict[str, float | int | None]:
    return {
        "count": len(lengths),
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "mean": mean(lengths) if lengths else None,
        "median": median(lengths) if lengths else None,
    }


def summarize(
    samples: Sequence[Mapping[str, Any]],
    decoded: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(samples) != len(decoded):
        raise ValueError("sample/decode row counts differ")
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {
        condition: [] for condition in DNA_HEADLINE_CONDITIONS
    }
    indices: dict[str, set[int]] = {condition: set() for condition in DNA_HEADLINE_CONDITIONS}
    for index, (sample, row) in enumerate(zip(samples, decoded, strict=True)):
        if sample.get("id") != row.get("id"):
            raise ValueError(f"sample/decode ID mismatch at row {index}")
        condition = sample.get("condition")
        if condition not in grouped or row.get("condition") != condition:
            raise ValueError(f"unknown or mismatched DNA condition at row {index}")
        if row.get("source_protocol") != SAMPLE_PROTOCOL:
            raise ValueError(f"decoded source protocol mismatch at row {index}")
        sample_index = sample.get("sample_index")
        if type(sample_index) is not int or not 0 <= sample_index < FORMAL_SAMPLES_PER_CONDITION:
            raise ValueError(f"invalid DNA sample index at row {index}")
        expected_id = f"{DNA_TASK}:{condition}:sample-{sample_index:06d}"
        if sample.get("id") != expected_id or sample_index in indices[str(condition)]:
            raise ValueError(f"DNA slot ID/index contract mismatch at row {index}")
        indices[str(condition)].add(sample_index)
        raw_ids = sample.get("generated_bpe_ids")
        if sample.get("prompt_only") is not True or not isinstance(raw_ids, list):
            raise ValueError(f"DNA raw slot representation mismatch at row {index}")
        if bool(sample.get("empty_generation")) != (len(raw_ids) == 0):
            raise ValueError(f"DNA empty-generation flag mismatch at row {index}")
        for key in ("sample_index", "generated_bpe_ids", "ended_with_eos", "invalid_model_id"):
            if row.get(key) != sample.get(key):
                raise ValueError(f"sample/decode {key} mismatch at row {index}")
        status = row.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unknown DNA decode status at row {index}: {status!r}")
        if bool(sample.get("invalid_model_id")):
            expected_statuses = {"invalid_model_id"}
        elif not bool(sample.get("ended_with_eos")):
            expected_statuses = {"missing_eos"}
        elif not raw_ids:
            expected_statuses = {"empty_generation"}
        else:
            expected_statuses = {"decode_error", "wrong_length", "valid"}
        if status not in expected_statuses:
            raise ValueError(f"DNA decode status precedence mismatch at row {index}")
        sequence = row.get("decoded_sequence")
        if row.get("sequence_length") != (len(sequence) if isinstance(sequence, str) else None):
            raise ValueError(f"DNA decoded sequence_length mismatch at row {index}")
        if status == "valid" and (
            not isinstance(sequence, str)
            or len(sequence) != 200
            or set(sequence).difference("ACGT")
        ):
            raise ValueError(f"invalid status=valid DNA at row {index}")
        if status == "wrong_length" and (
            not isinstance(sequence, str)
            or not sequence
            or len(sequence) == 200
            or set(sequence).difference("ACGT")
        ):
            raise ValueError(f"invalid status=wrong_length DNA at row {index}")
        grouped[str(condition)].append((sample, row))

    changed = {
        condition: len(rows)
        for condition, rows in grouped.items()
        if len(rows) != FORMAL_SAMPLES_PER_CONDITION
    }
    if changed:
        raise ValueError(f"formal DNA representation metrics require 1000 slots/condition: {changed}")
    for condition, observed in indices.items():
        if observed != set(range(FORMAL_SAMPLES_PER_CONDITION)):
            raise ValueError(f"formal DNA slot indices are incomplete for {condition}")

    def one_group(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
        requested = len(pairs)
        sample_rows = [sample for sample, _row in pairs]
        decoded_rows = [row for _sample, row in pairs]
        statuses = Counter(str(row.get("status")) for row in decoded_rows)
        recovered = [
            str(row["decoded_sequence"])
            for row in decoded_rows
            if isinstance(row.get("decoded_sequence"), str)
        ]
        valid = [
            str(row["decoded_sequence"])
            for row in decoded_rows
            if row.get("status") == "valid"
        ]
        if any(len(sequence) != 200 or set(sequence).difference("ACGT") for sequence in valid):
            raise ValueError("status=valid row violates strict 200-bp ACGT contract")
        unique = len(set(valid))
        exact_200 = sum(len(sequence) == 200 for sequence in recovered)
        ended = sum(bool(row.get("ended_with_eos")) for row in sample_rows)
        invalid_ids = sum(bool(row.get("invalid_model_id")) for row in sample_rows)
        empty = sum(bool(row.get("empty_generation")) for row in sample_rows)
        return {
            "requested": requested,
            "generation": {
                "ended_with_eos": ended,
                "ended_with_eos_rate": ended / requested,
                "invalid_model_id": invalid_ids,
                "invalid_model_id_rate": invalid_ids / requested,
                "empty_generation": empty,
                "empty_generation_rate": empty / requested,
            },
            "decode_status_counts": dict(statuses),
            "recovered_sequence": len(recovered),
            "recovered_sequence_over_requested": len(recovered) / requested,
            "strict_valid_200bp_acgt": len(valid),
            "strict_validity": len(valid) / requested,
            "exact_200bp_recovered": exact_200,
            "exact_200bp_over_requested": exact_200 / requested,
            "recovered_length": _length_summary([len(sequence) for sequence in recovered]),
            "unique_strict_valid": unique,
            "unique_strict_valid_over_requested": unique / requested,
            "unique_within_strict_valid": unique / len(valid) if valid else None,
        }

    per_condition = {condition: one_group(rows) for condition, rows in grouped.items()}
    overall_pairs = [pair for condition in DNA_HEADLINE_CONDITIONS for pair in grouped[condition]]
    return {
        "protocol": PROTOCOL,
        "complete": True,
        "formal": True,
        "task": DNA_TASK,
        "condition_scope": list(DNA_HEADLINE_CONDITIONS),
        "samples_per_condition": FORMAL_SAMPLES_PER_CONDITION,
        "requested": len(overall_pairs),
        "metric_scope": "representation_level_only",
        "strict_valid_definition": "decoded sequence has exactly 200 bases, alphabet A/C/G/T",
        "invalid_slots_filtered_or_replaced": False,
        "official_test_records_read_for_this_report": 0,
        "novelty": "not computed; no train/test reference set is read by this report",
        "headline_claims": {
            "enformer": "not computed",
            "motif": "not computed",
            "locus_or_regulatory_activity": "not computed",
            "official_dna_diffusion_headline": False,
            "reason": (
                "track/bin, locus coordinates, normalization, motif database/scanner, "
                "and aggregation are not sufficiently frozen for a comparable claim"
            ),
        },
        "per_condition": per_condition,
        "overall": one_group(overall_pairs),
    }


def validate_generation(generation: Mapping[str, Any]) -> None:
    """Check scientific sampling settings, not a legacy hardware batch size."""
    expected = {
        "max_new_tokens": 128, "max_context": 512, "greedy": False,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "seed": 20260902, "dtype": "bf16",
    }
    if any(generation.get(key) != value for key, value in expected.items()):
        raise ValueError("DNA scientific sampling settings differ from the frozen protocol")
    batch = generation.get("batch_size")
    if type(batch) is not int or batch <= 0:
        raise ValueError("DNA sampling batch_size must be a positive integer")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DNA representation metrics: {args.output}")
    samples = _read_jsonl(args.samples, protocol=SAMPLE_PROTOCOL)
    decoded = _read_jsonl(args.decoded_samples, protocol=DECODE_PROTOCOL)
    result = summarize(samples, decoded)
    sample_summary_path = args.samples.with_suffix(args.samples.suffix + ".summary.json")
    decoded_summary_path = args.decoded_samples.with_suffix(
        args.decoded_samples.suffix + ".summary.json"
    )
    for path in (sample_summary_path, decoded_summary_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    sample_summary = json.loads(sample_summary_path.read_text(encoding="utf-8"))
    decoded_summary = json.loads(decoded_summary_path.read_text(encoding="utf-8"))
    validate_generation(sample_summary["generation"])
    if (
        sample_summary.get("complete") is not True
        or sample_summary.get("protocol") != SAMPLE_PROTOCOL
        or sample_summary.get("task") != DNA_TASK
        or sample_summary.get("target_type") != "dna"
        or sample_summary.get("formal_count_complete") is not True
        or sample_summary.get("samples_per_condition") != FORMAL_SAMPLES_PER_CONDITION
        or sample_summary.get("formal_sample_count") != FORMAL_SAMPLES_PER_CONDITION
        or tuple(sample_summary.get("conditions") or ()) != DNA_HEADLINE_CONDITIONS
        or sample_summary.get("official_test_records_read") != 0
        or sample_summary.get("failed_slots_replaced") is not False
        or sample_summary.get("output") != str(args.samples.resolve())
    ):
        raise ValueError("DNA sampling summary is not the frozen formal representation view")
    if (
        decoded_summary.get("complete") is not True
        or decoded_summary.get("protocol") != DECODE_PROTOCOL
        or decoded_summary.get("task") != DNA_TASK
        or decoded_summary.get("target_type") != "dna"
        or decoded_summary.get("source") != str(args.samples.resolve())
        or decoded_summary.get("output") != str(args.decoded_samples.resolve())
        or decoded_summary.get("invalid_slots_filtered_or_replaced") is not False
    ):
        raise ValueError("DNA decode summary is incomplete")
    for condition in DNA_HEADLINE_CONDITIONS:
        expected = result["per_condition"][condition]
        observed = (decoded_summary.get("per_condition") or {}).get(condition) or {}
        if (
            observed.get("requested") != FORMAL_SAMPLES_PER_CONDITION
            or sum((observed.get("status_counts") or {}).values())
            != FORMAL_SAMPLES_PER_CONDITION
            or observed.get("valid") != expected["strict_valid_200bp_acgt"]
            or observed.get("unique_valid") != expected["unique_strict_valid"]
        ):
            raise ValueError(f"DNA decode summary mismatch for {condition}")
    overall = decoded_summary.get("overall") or {}
    if overall.get("requested") != len(samples) or sum((overall.get("status_counts") or {}).values()) != len(samples):
        raise ValueError("DNA decode summary overall denominator mismatch")
    result.update(
        {
            "checkpoint": sample_summary.get("checkpoint"),
            "samples": str(args.samples.resolve()),
            "decoded_samples": str(args.decoded_samples.resolve()),
            "sampling_summary": str(sample_summary_path.resolve()),
            "decode_summary": str(decoded_summary_path.resolve()),
            "generation": sample_summary["generation"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
