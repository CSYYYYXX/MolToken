#!/usr/bin/env python3
"""Evaluate the TCR-TRANSLATE external-13 headline prediction track.

For every pMHC this adapter reports the paper's counts among all 1,000 ranked
generation slots: exact reference matches, >=90% native sequence recovery,
and generations that GIANA 4.1 clusters with at least one released reference.
GIANA is run with CDR3beta only and its default settings (``-v``).  Invalid or
out-of-range Axis outputs remain misses in the 1,000-slot denominator; they
are never filtered from the prediction artifact or replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.runtime_host import require_compute_host  # noqa: E402


PREDICTION_PROTOCOL = (
    "axis-tcr-translate-external13-headline-predictions-v1-20260830"
)
TRACK = "external13_sparse_headline"
EXPECTED_GROUPS = 13
HEADLINE_K = 1_000
HEADLINE_NUM_BEAMS = 1_000
HEADLINE_MAX_NEW_TOKENS = 25
HEADLINE_NO_REPEAT_NGRAM_SIZE = 5
SEQUENCE_RECOVERY_THRESHOLD = 0.90
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
GIANA_MIN_LENGTH = 10
GIANA_MAX_LENGTH = 24


def require_worker() -> None:
    require_compute_host(
        "TCR-TRANSLATE external-13 evaluation", socket.gethostname()
    )


def _string_list(value: Any, *, label: str, allow_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or (not allow_empty and not item):
            raise ValueError(f"{label}[{index}] is not a valid string")
        result.append(item)
    return result


def validate_prediction(record: Any, *, line_number: int) -> dict[str, Any]:
    label = f"external-13 prediction line {line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{label}: record must be an object")
    if record.get("protocol") != PREDICTION_PROTOCOL:
        raise ValueError(f"{label}: unexpected protocol")
    if record.get("track") != TRACK or record.get("headline") is not True:
        raise ValueError(f"{label}: record is not the external-13 headline track")
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label}: generation must be an object")
    expected = {
        "mode": "beam",
        "k": HEADLINE_K,
        "num_beams": HEADLINE_NUM_BEAMS,
        "max_new_tokens": HEADLINE_MAX_NEW_TOKENS,
        "no_repeat_ngram_size": HEADLINE_NO_REPEAT_NGRAM_SIZE,
        "do_sample": False,
        "invalid_or_duplicate_filtering": False,
        "replacement_sampling": False,
        "post_hoc_sequence_repair": False,
    }
    for key, expected_value in expected.items():
        if generation.get(key) != expected_value:
            raise ValueError(
                f"{label}: generation.{key}={generation.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    translations = _string_list(
        record.get("translations"), label=f"{label}.translations", allow_empty=True
    )
    if len(translations) != HEADLINE_K:
        raise ValueError(
            f"{label}: expected {HEADLINE_K} translations, got {len(translations)}"
        )
    references = _string_list(
        record.get("reference_translations"),
        label=f"{label}.reference_translations",
        allow_empty=False,
    )
    if not references:
        raise ValueError(f"{label}: reference list is empty")
    for index, reference in enumerate(references):
        if set(reference).difference(AA_ALPHABET):
            raise ValueError(f"{label}: non-canonical reference at index {index}")
    candidates = record.get("beam_candidates")
    if not isinstance(candidates, list) or len(candidates) != HEADLINE_K:
        raise ValueError(f"{label}: beam_candidates must have {HEADLINE_K} slots")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or candidate.get("rank") != index + 1:
            raise ValueError(f"{label}: candidate rank order changed at {index}")
        if candidate.get("official_translation") != translations[index]:
            raise ValueError(
                f"{label}: candidate and translation disagree at rank {index + 1}"
            )
    pmhc = record.get("pmhc")
    if not isinstance(pmhc, dict) or not isinstance(pmhc.get("official_key"), str):
        raise ValueError(f"{label}: pmhc.official_key is required")
    return record


def load_predictions(
    path: Path, *, expected_groups: int = EXPECTED_GROUPS
) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"external-13 prediction JSONL missing: {path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pmhc: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            record = validate_prediction(json.loads(raw), line_number=line_number)
            record_id = record.get("id")
            pmhc_key = record["pmhc"]["official_key"]
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: missing id")
            if record_id in seen_ids or pmhc_key in seen_pmhc:
                raise ValueError(f"{path}:{line_number}: duplicate id or pMHC key")
            seen_ids.add(record_id)
            seen_pmhc.add(pmhc_key)
            records.append(record)
    if len(records) != expected_groups:
        raise ValueError(
            f"external-13 group count changed: expected={expected_groups} "
            f"actual={len(records)}"
        )
    return records


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def native_sequence_recovery(translation: str, references: Sequence[str]) -> float:
    """Match released ModelEvaluator._mean_sequence_recovery for one slot.

    Closest means minimum Levenshtein distance, retaining source order on
    ties; it is not the maximum positional identity over equal-length refs.
    Both released 6853e5c and data-source 97aeab1 use this rule.
    """

    if not translation:
        return 0.0
    same_length = [reference for reference in references if len(reference) == len(translation)]
    if same_length:
        closest = min(same_length, key=lambda value: levenshtein_distance(translation, value))
        return sum(left == right for left, right in zip(translation, closest)) / len(translation)
    closest = min(references, key=lambda value: levenshtein_distance(translation, value))
    return 1.0 - levenshtein_distance(translation, closest) / len(closest)


def compute_reference_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    translations = list(record["translations"])
    references = list(record["reference_translations"])
    reference_set = set(references)
    recoveries = [
        native_sequence_recovery(translation, references)
        for translation in translations
    ]
    canonical = [
        bool(value) and not set(value).difference(AA_ALPHABET)
        for value in translations
    ]
    return {
        "pMHC": record["pmhc"]["official_key"],
        "generation_slots": len(translations),
        "reference_count": len(references),
        "exact_reference_match_count": sum(
            translation in reference_set for translation in translations
        ),
        "sequence_recovery_ge_0_90_count": sum(
            value >= SEQUENCE_RECOVERY_THRESHOLD for value in recoveries
        ),
        "mean_native_sequence_recovery": sum(recoveries) / len(recoveries),
        "canonical_nonempty_count": sum(canonical),
        "unique_nonempty_count": len({value for value in translations if value}),
        "invalid_or_empty_count": sum(not value for value in canonical),
    }


def giana_eligible(sequence: str) -> bool:
    return (
        bool(sequence)
        and not set(sequence).difference(AA_ALPHABET)
        and GIANA_MIN_LENGTH <= len(sequence) <= GIANA_MAX_LENGTH
    )


def write_giana_input(record: Mapping[str, Any], path: Path) -> dict[str, int]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite GIANA input: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    references = list(record["reference_translations"])
    translations = list(record["translations"])
    ineligible_references = [value for value in references if not giana_eligible(value)]
    if ineligible_references:
        raise ValueError(
            f"{record['pmhc']['official_key']}: released reference is outside GIANA "
            f"CDR3-only domain: {ineligible_references[:3]}"
        )
    eligible_generations = 0
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write("CDR3\torigin\tslot\n")
        for index, sequence in enumerate(references):
            handle.write(f"{sequence}\treference\tref:{index:04d}\n")
        for index, sequence in enumerate(translations):
            if not giana_eligible(sequence):
                continue
            handle.write(f"{sequence}\tgeneration\tgen:{index:04d}\n")
            eligible_generations += 1
    return {
        "reference_rows": len(references),
        "generation_slots": len(translations),
        "giana_eligible_generation_slots": eligible_generations,
        "giana_ineligible_generation_slots": len(translations) - eligible_generations,
    }


def run_giana(
    *,
    giana_script: Path,
    giana_vgene_fasta: Path,
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    python_executable: str,
) -> None:
    giana_script = giana_script.expanduser().resolve()
    giana_vgene_fasta = giana_vgene_fasta.expanduser().resolve()
    if not giana_script.is_file() or not giana_vgene_fasta.is_file():
        raise FileNotFoundError(
            f"GIANA 4.1 script/assets missing: {giana_script}, {giana_vgene_fasta}"
        )
    query_module = giana_script.parent / "query.py"
    if not query_module.is_file():
        raise FileNotFoundError(f"GIANA query.py missing beside script: {query_module}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite GIANA output: {output_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    # GIANA 4.1 always prepends the directory containing GIANA4.1.py to the
    # value supplied through ``-V``.  Supplying the resolved absolute path
    # would therefore produce ``<script-dir>/<absolute-path>`` and fail even
    # when the official FASTA exists.  Preserve the caller-facing absolute
    # path contract, but translate it back to GIANA's script-relative form.
    giana_vgene_arg = os.path.relpath(giana_vgene_fasta, giana_script.parent)
    command = [
        python_executable,
        str(giana_script),
        "-f",
        str(input_path.resolve()),
        "-v",
        "-V",
        giana_vgene_arg,
        "-o",
        str(work_dir.resolve()),
        "-O",
        str(output_path.resolve()),
    ]
    subprocess.run(command, cwd=work_dir, check=True)
    if not output_path.is_file():
        raise RuntimeError(f"GIANA completed without output: {output_path}")


def parse_giana_reference_clustered(
    path: Path, *, generation_slots: int = HEADLINE_K
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"GIANA output missing: {path}")
    clusters: dict[str, list[tuple[str, str]]] = {}
    seen_slots: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if raw.startswith("##") or not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_number}: malformed GIANA output row")
            _sequence, cluster_id, origin, slot = parts[:4]
            if origin not in {"reference", "generation"}:
                raise ValueError(f"{path}:{line_number}: unknown origin {origin!r}")
            if slot in seen_slots:
                raise ValueError(f"{path}:{line_number}: duplicate GIANA slot {slot}")
            seen_slots.add(slot)
            clusters.setdefault(cluster_id, []).append((origin, slot))
    reference_clusters = {
        cluster_id
        for cluster_id, members in clusters.items()
        if any(origin == "reference" for origin, _slot in members)
    }
    generated_slots = {
        slot
        for cluster_id in reference_clusters
        for origin, slot in clusters[cluster_id]
        if origin == "generation"
    }
    for slot in generated_slots:
        if not slot.startswith("gen:"):
            raise ValueError(f"{path}: malformed generated slot {slot!r}")
        rank = int(slot.removeprefix("gen:"))
        if rank < 0 or rank >= generation_slots:
            raise ValueError(f"{path}: generation slot out of range: {slot}")
    return {
        "giana_4_1_reference_clustered_count": len(generated_slots),
        "giana_reference_clusters": len(reference_clusters),
        "giana_rows_emitted": len(seen_slots),
    }


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty pMHC values")
    return {
        "total": int(sum(values)),
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    giana_script: Path,
    giana_vgene_fasta: Path,
    work_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    work_root = work_root.expanduser().resolve()
    if work_root.exists():
        raise FileExistsError(f"refusing to overwrite GIANA work root: {work_root}")
    work_root.mkdir(parents=True)
    atomic: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        group_dir = work_root / f"{index:02d}_{record['pmhc']['official_key']}"
        group_dir.mkdir()
        input_path = group_dir / "combined_cdr3b.tsv"
        output_path = group_dir / "giana_clusters.tsv"
        giana_audit = write_giana_input(record, input_path)
        run_giana(
            giana_script=giana_script,
            giana_vgene_fasta=giana_vgene_fasta,
            input_path=input_path,
            output_path=output_path,
            work_dir=group_dir,
            python_executable=python_executable,
        )
        metrics = compute_reference_metrics(record)
        metrics.update(giana_audit)
        metrics.update(parse_giana_reference_clustered(output_path))
        atomic.append(metrics)
    metric_names = (
        "exact_reference_match_count",
        "sequence_recovery_ge_0_90_count",
        "giana_4_1_reference_clustered_count",
    )
    return {
        "complete": True,
        "protocol": PREDICTION_PROTOCOL,
        "track": TRACK,
        "headline": True,
        "groups": len(records),
        "generation_slots_per_pmhc": HEADLINE_K,
        "atomic_metrics": atomic,
        "aggregate_metrics": {
            name: _summary([int(row[name]) for row in atomic])
            for name in metric_names
        },
        "metric_contract": {
            "exact_reference_match_count": (
                "candidate occurrences exactly equal to any released cognate CDR3beta"
            ),
            "sequence_recovery_ge_0_90_count": (
                "candidate occurrences with native closest-reference recovery >= 0.90"
            ),
            "giana_4_1_reference_clustered_count": (
                "candidate occurrences sharing a GIANA 4.1 default CDR3-only cluster "
                "with at least one released reference"
            ),
            "invalid_or_duplicate_outputs": (
                "all requested ranks remain in the 1000-slot denominator; no repair, "
                "filtering, deduplication, or replacement"
            ),
        },
    }


def write_outputs(
    result: Mapping[str, Any], *, output_prefix: Path, model_name: str
) -> dict[str, str]:
    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(output_prefix) + ".json")
    atomic_path = Path(str(output_prefix) + ".atomic.csv")
    if json_path.exists() or atomic_path.exists():
        raise FileExistsError(
            f"refusing to overwrite external-13 metric outputs: {output_prefix}"
        )
    json_payload = {**result, "model": model_name}
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "model",
        "pMHC",
        "generation_slots",
        "reference_count",
        "exact_reference_match_count",
        "sequence_recovery_ge_0_90_count",
        "giana_4_1_reference_clustered_count",
        "mean_native_sequence_recovery",
        "canonical_nonempty_count",
        "unique_nonempty_count",
        "invalid_or_empty_count",
        "giana_eligible_generation_slots",
        "giana_ineligible_generation_slots",
        "giana_reference_clusters",
        "giana_rows_emitted",
    ]
    with atomic_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["atomic_metrics"]:
            writer.writerow({"model": model_name, **{key: row[key] for key in fields[1:]}})
    return {"json": str(json_path), "atomic_csv": str(atomic_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--giana-script", type=Path, required=True)
    parser.add_argument("--giana-vgene-fasta", type=Path, required=True)
    parser.add_argument("--giana-work-root", type=Path, required=True)
    parser.add_argument("--giana-python", default=sys.executable)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    records = load_predictions(args.predictions)
    result = evaluate_records(
        records,
        giana_script=args.giana_script,
        giana_vgene_fasta=args.giana_vgene_fasta,
        work_root=args.giana_work_root,
        python_executable=args.giana_python,
    )
    outputs = write_outputs(
        result, output_prefix=args.output_prefix, model_name=args.model_name
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "track": TRACK,
                "outputs": outputs,
                "aggregate_metrics": result["aggregate_metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
