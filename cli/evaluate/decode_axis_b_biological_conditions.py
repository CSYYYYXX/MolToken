#!/usr/bin/env python3
"""Strictly decode biological samples from the frozen FDDFS/BPE representation.

The input JSONL is never rewritten.  One output row is emitted for every input
slot, including missing-EOS, invalid-token, empty, graph-decode, and
biopolymer-recovery failures.  This command reports only representation-level
metrics; it does not claim DNA-Diffusion Enformer/motif headline scores or
RfamGen Infernal scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from cli.evaluate.eval_axis_b_samples import BPEDecoder, FDDFSVocab  # noqa: E402
from cli.evaluate.sample_axis_b_biological_conditions import (  # noqa: E402
    DNA_TASK,
    DPLM_CONDITIONS,
    DPLM_TASK,
    PEP_TASK,
    PROTOCOL as SAMPLING_PROTOCOL,
    RFAM_TASK,
)
from pipeline.graph.biopolymer_sequence import (  # noqa: E402
    graph_to_biopolymer_sequence,
)
from pipeline.serialization.fddfs import deserialize_fddfs  # noqa: E402


PROTOCOL = "axis-biological-strict-sequence-decode-v1-20260902"
ALPHABETS = {
    DNA_TASK: frozenset("ACGT"),
    RFAM_TASK: frozenset("ACGU"),
    DPLM_TASK: frozenset("ACDEFGHIKLMNPQRSTVWY"),
    PEP_TASK: frozenset("ACDEFGHIKLMNPQRSTVWY"),
}
TARGET_TYPES = {DNA_TASK: "dna", RFAM_TASK: "rna", DPLM_TASK: "protein", PEP_TASK: "peptide"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--merges-csv", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _status_before_decode(record: Mapping[str, Any]) -> str | None:
    if bool(record.get("invalid_model_id")):
        return "invalid_model_id"
    if not bool(record.get("ended_with_eos")):
        return "missing_eos"
    raw = record.get("generated_bpe_ids")
    if not isinstance(raw, list) or not raw:
        return "empty_generation"
    return None


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested = len(rows)
    statuses = Counter(str(row["status"]) for row in rows)
    valid_sequences = [
        str(row["decoded_sequence"])
        for row in rows
        if row.get("status") == "valid"
    ]
    unique = len(set(valid_sequences))
    result = {
        "requested": requested,
        "status_counts": dict(statuses),
        "valid": len(valid_sequences),
        "validity": len(valid_sequences) / requested if requested else None,
        "unique_valid": unique,
        "unique_valid_over_requested": unique / requested if requested else None,
        "unique_within_valid": unique / len(valid_sequences) if valid_sequences else None,
        "length": {
            "min": min(map(len, valid_sequences)) if valid_sequences else None,
            "max": max(map(len, valid_sequences)) if valid_sequences else None,
            "mean": (
                sum(map(len, valid_sequences)) / len(valid_sequences)
                if valid_sequences
                else None
            ),
        },
    }
    if any(row.get("requested_length") is not None for row in rows):
        exact = sum(row.get("exact_length_match") is True for row in rows)
        result["exact_length_matches"] = exact
        result["exact_length_rate_requested_denominator"] = exact / requested
    return result


def decode_sequence(raw_ids: list[int], decoder: BPEDecoder, target_type: str) -> str:
    """Only representation/sequence ValueErrors are expected invalid outputs."""
    _base, tokens = decoder.decode_raw_ids(raw_ids)
    graph = deserialize_fddfs(tokens, decoder.vocab.default_edge)
    return graph_to_biopolymer_sequence(graph, target_type=target_type)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.samples, args.merges_csv, args.vocab_json):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite decode output: {args.output}")

    # Long linear nucleotide graphs exceed Python's default recursive parser
    # depth even for a normal 200-bp DNA target.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))
    vocab = FDDFSVocab.load(args.vocab_json)
    decoder = BPEDecoder(args.merges_csv, vocab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    task: str | None = None
    started = time.time()
    with args.samples.open("r", encoding="utf-8") as source, args.output.open(
        "x", encoding="utf-8"
    ) as output:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                raise ValueError(f"{args.samples}:{line_number}: blank line")
            record = json.loads(raw_line)
            if not isinstance(record, dict) or record.get("protocol") != SAMPLING_PROTOCOL:
                raise ValueError(f"{args.samples}:{line_number}: unexpected sampling protocol")
            row_task = str(record.get("task"))
            if row_task not in TARGET_TYPES:
                raise ValueError(f"{args.samples}:{line_number}: unsupported task {row_task!r}")
            if task is None:
                task = row_task
            elif task != row_task:
                raise ValueError("one decode input cannot mix tasks")
            target_type = TARGET_TYPES[row_task]
            if record.get("target_type") != target_type:
                raise ValueError(f"{args.samples}:{line_number}: target type mismatch")
            condition = record.get("condition")
            if not isinstance(condition, str) or not condition:
                raise ValueError(f"{args.samples}:{line_number}: condition must be non-empty")
            row_id = record.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise ValueError(f"{args.samples}:{line_number}: invalid or duplicate id")
            seen_ids.add(row_id)
            raw_ids = record.get("generated_bpe_ids")
            if not isinstance(raw_ids, list) or any(type(value) is not int for value in raw_ids):
                raise ValueError(f"{args.samples}:{line_number}: generated_bpe_ids must be integer IDs")
            requested_length = 200 if row_task == DNA_TASK else None
            if row_task == DPLM_TASK:
                if condition not in DPLM_CONDITIONS:
                    raise ValueError(f"unknown DPLM length condition: {condition!r}")
                requested_length = int(condition)
            elif row_task == PEP_TASK:
                if condition not in {str(n) for n in range(5, 81)}:
                    raise ValueError(f"unknown PepBench length condition: {condition!r}")
                requested_length = int(condition)
            status = _status_before_decode(record)
            sequence: str | None = None
            error: str | None = None
            if status is None:
                try:
                    sequence = decode_sequence(raw_ids, decoder, target_type)
                    invalid = set(sequence).difference(ALPHABETS[row_task])
                    if not sequence or invalid:
                        raise ValueError(
                            f"decoded {target_type} has invalid alphabet: {sorted(invalid)}"
                        )
                    if row_task == DNA_TASK and len(sequence) != 200:
                        status = "wrong_length"
                    elif row_task == PEP_TASK and not 5 <= len(sequence) <= 80:
                        status = "wrong_length"
                    else:
                        status = "valid"
                except ValueError as exc:
                    # Malformed generated graphs remain scored misses. Missing
                    # dependencies, programming errors and resource failures
                    # must abort visibly rather than masquerading as invalidity.
                    status = "decode_error"
                    error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    sequence = None
            result = {
                "protocol": PROTOCOL,
                "source_protocol": SAMPLING_PROTOCOL,
                "id": record.get("id"),
                "task": row_task,
                "target_type": target_type,
                "condition": condition,
                "sample_index": record.get("sample_index"),
                "generated_bpe_ids": record.get("generated_bpe_ids"),
                "ended_with_eos": bool(record.get("ended_with_eos")),
                "invalid_model_id": bool(record.get("invalid_model_id")),
                "status": status,
                "decoded_sequence": sequence,
                "sequence_length": len(sequence) if sequence is not None else None,
                "requested_length": requested_length,
                "exact_length_match": (
                    len(sequence) == requested_length if sequence is not None
                    and requested_length is not None else False
                ),
                "decode_error": error,
            }
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
            grouped.setdefault(condition, []).append(result)

    if task is None:
        raise ValueError("sampling file is empty")
    group_summaries = {
        condition: summarize_group(rows) for condition, rows in grouped.items()
    }
    total_rows = [row for rows in grouped.values() for row in rows]
    summary = {
        "protocol": PROTOCOL,
        "complete": True,
        "source": str(args.samples.resolve()),
        "task": task,
        "target_type": TARGET_TYPES[task],
        "invalid_slots_filtered_or_replaced": False,
        "novelty": (
            "not computed: supply a separately frozen train-reference view in the "
            "task evaluator; official test references must never select sampling parameters"
        ),
        "per_condition": group_summaries,
        "overall": summarize_group(total_rows),
        "elapsed_sec": time.time() - started,
        "output": str(args.output.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
