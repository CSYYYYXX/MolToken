#!/usr/bin/env python3
"""Score decoded Axis RfamGen reconstruction-17 samples with Infernal 1.1.3.

For each frozen family, valid decoded RNA is aligned with that family's Rfam
14.7 CM using ``cmalign --notrunc --sfile``.  The per-sample paper quantity is
``bit_score / family_GA`` and GA pass is the strict paper/repository comparison
``bit_score > GA``.  Invalid model/decode slots remain in the requested
denominator and are never replaced.  Aggregates are computed within family
first and then macro-averaged across the 17 reconstruction families.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


from cli.evaluate.decode_axis_b_biological_conditions import (
    PROTOCOL as DECODE_PROTOCOL,
)
from cli.evaluate.sample_axis_b_biological_conditions import (
    FORMAL_SAMPLES_PER_CONDITION,
    RFAM_CONDITIONS,
    RFAM_TASK,
)
RFAM_DECODE_PROTOCOL = "biomatrix-public-sft-native-rna-zeroshot-v1"


PROTOCOL = "axis-rfamgen-figure2-reconstruction17-infernal-v1-20260902"
REQUIRED_INFERNAL_VERSION = "1.1.3"
GA_RE = re.compile(r"^GA\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))\b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-samples", type=Path, required=True)
    parser.add_argument("--cm-root", type=Path, required=True)
    parser.add_argument("--cmalign", type=Path, required=True)
    parser.add_argument("--cpu-per-family", type=int, default=4)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-representation", choices=("axis-fddfs", "biomatrix-native"),
                        default="axis-fddfs")
    parser.add_argument(
        "--allow-partial-smoke",
        action="store_true",
        help="Permit fewer than 17x1000 rows; output is explicitly non-formal",
    )
    return parser


def read_ga_threshold(path: Path) -> float:
    matches: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = GA_RE.match(line)
            if match:
                matches.append(float(match.group(1)))
    if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0:
        raise ValueError(f"{path}: expected one positive finite GA threshold, got {matches}")
    return matches[0]


def parse_cmalign_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: malformed cmalign score row")
            sequence_id = fields[1]
            if sequence_id in scores:
                raise ValueError(f"{path}:{line_number}: duplicate score id {sequence_id}")
            if fields[5] != "no":
                raise ValueError(
                    f"{path}:{line_number}: --notrunc score unexpectedly says {fields[5]!r}"
                )
            score = float(fields[6])
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: non-finite bit score")
            scores[sequence_id] = score
    return scores


def read_cmalign_version(path: Path) -> str:
    completed = subprocess.run(
        [str(path), "-h"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    match = re.search(r"^# INFERNAL\s+(\S+)", completed.stdout, re.MULTILINE)
    if match is None:
        raise ValueError(f"could not identify Infernal version from {path} -h")
    return match.group(1)


def _load_rows(path: Path, decode_protocol: str = DECODE_PROTOCOL) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(raw)
            if (
                not isinstance(row, dict)
                or row.get("protocol") != decode_protocol
                or row.get("task") != RFAM_TASK
                or row.get("target_type") != "rna"
            ):
                raise ValueError(f"{path}:{line_number}: not an Rfam decoded sample")
            row_id = row.get("id")
            condition = row.get("condition")
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate id")
            if condition not in RFAM_CONDITIONS:
                raise ValueError(f"{path}:{line_number}: unknown family {condition!r}")
            seen_ids.add(row_id)
            grouped.setdefault(str(condition), []).append(row)
    if not grouped:
        raise ValueError(f"empty decoded sample file: {path}")
    return grouped


def _family_metrics(rows: Sequence[Mapping[str, Any]], ga: float) -> dict[str, Any]:
    requested = len(rows)
    status_counts = Counter(str(row["status"]) for row in rows)
    scored = [row for row in rows if row.get("infernal_bit_score") is not None]
    scores = [float(row["infernal_bit_score"]) for row in scored]
    relatives = [float(row["relative_bit_score"]) for row in scored]
    ga_passes = sum(bool(row["ga_pass"]) for row in rows)
    valid_sequences = [
        str(row["decoded_sequence"])
        for row in rows
        if row.get("status") == "valid"
    ]
    unique = len(set(valid_sequences))
    return {
        "requested": requested,
        "decode_status_counts": dict(status_counts),
        "decoded_valid": len(valid_sequences),
        "validity": len(valid_sequences) / requested,
        "unique_valid": unique,
        "unique_valid_over_requested": unique / requested,
        "unique_within_valid": unique / len(valid_sequences) if valid_sequences else None,
        "ga_threshold": ga,
        "infernal_scored": len(scored),
        "infernal_score_coverage": len(scored) / requested,
        "bit_score_mean_scored": mean(scores) if scores else None,
        "bit_score_median_scored": median(scores) if scores else None,
        "relative_bit_score_mean_scored": mean(relatives) if relatives else None,
        "relative_bit_score_median_scored": median(relatives) if relatives else None,
        "ga_pass_requested": ga_passes,
        "ga_pass_rate_requested_denominator": ga_passes / requested,
        "ga_pass_rate_scored_denominator": ga_passes / len(scored) if scored else None,
    }


def _macro(families: Mapping[str, Mapping[str, Any]], key: str) -> float | None:
    values = [payload[key] for payload in families.values()]
    # A completely unscored requested family must not disappear from the
    # comparison. Undefined score means remain undefined; coverage/pass rates
    # still have their actual requested-slot denominators and remain usable.
    if not values or any(value is None for value in values):
        return None
    return mean(float(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cpu_per_family <= 0:
        raise ValueError("--cpu-per-family must be positive")
    if not args.decoded_samples.is_file() or args.decoded_samples.stat().st_size == 0:
        raise FileNotFoundError(args.decoded_samples)
    if not args.cmalign.is_file() or (args.cmalign.stat().st_mode & 0o111) == 0:
        raise FileNotFoundError(f"missing executable cmalign: {args.cmalign}")
    infernal_version = read_cmalign_version(args.cmalign)
    if infernal_version != REQUIRED_INFERNAL_VERSION:
        raise ValueError(
            "formal RfamGen reconstruction evaluation requires "
            f"Infernal {REQUIRED_INFERNAL_VERSION}, got {infernal_version}"
        )
    records_output = args.output.with_suffix(args.output.suffix + ".records.jsonl")
    if args.output.exists() or records_output.exists():
        raise FileExistsError(f"refusing to overwrite Rfam evaluation output: {args.output}")
    native = args.sample_representation == "biomatrix-native"
    decode_protocol = RFAM_DECODE_PROTOCOL if native else DECODE_PROTOCOL
    evaluation_protocol = "biomatrix-rfamgen-reconstruction17-infernal-v1" if native else PROTOCOL
    grouped = _load_rows(args.decoded_samples, decode_protocol)
    observed = tuple(family for family in RFAM_CONDITIONS if family in grouped)
    if not args.allow_partial_smoke:
        if observed != RFAM_CONDITIONS:
            raise ValueError(
                f"formal Rfam evaluation requires all 17 families; observed={observed}"
            )
        changed = {
            family: len(grouped[family])
            for family in RFAM_CONDITIONS
            if len(grouped[family]) != FORMAL_SAMPLES_PER_CONDITION
        }
        if changed:
            raise ValueError(
                f"formal Rfam evaluation requires 1000 requested slots per family: {changed}"
            )

    cm_paths: dict[str, Path] = {}
    thresholds: dict[str, float] = {}
    for family in observed:
        cm_path = args.cm_root / f"{family}.cm"
        if not cm_path.is_file() or cm_path.stat().st_size == 0:
            raise FileNotFoundError(cm_path)
        cm_paths[family] = cm_path
        thresholds[family] = read_ga_threshold(cm_path)

    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    scored_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="rfamgen_eval_", dir=str(args.tmp_dir) if args.tmp_dir else None
    ) as temporary:
        temp_root = Path(temporary)
        for family in observed:
            rows = grouped[family]
            valid = [row for row in rows if row.get("status") == "valid"]
            score_by_id: dict[str, float] = {}
            if valid:
                fasta = temp_root / f"{family}.fasta"
                score_file = temp_root / f"{family}.scores.txt"
                alignment = temp_root / f"{family}.sto"
                with fasta.open("x", encoding="utf-8") as handle:
                    for index, row in enumerate(valid):
                        handle.write(f">slot-{index:06d}\n{row['decoded_sequence']}\n")
                command = [
                    str(args.cmalign),
                    "--cpu",
                    str(args.cpu_per_family),
                    "--notrunc",
                    "-o",
                    str(alignment),
                    "--sfile",
                    str(score_file),
                    str(cm_paths[family]),
                    str(fasta),
                ]
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                parsed = parse_cmalign_scores(score_file)
                expected_ids = {f"slot-{index:06d}" for index in range(len(valid))}
                if set(parsed) != expected_ids:
                    raise RuntimeError(
                        f"{family}: cmalign scored IDs differ from valid decoded slots"
                    )
                score_by_id = parsed

            valid_index = 0
            ga = thresholds[family]
            for row in rows:
                bit_score: float | None = None
                if row.get("status") == "valid":
                    bit_score = score_by_id[f"slot-{valid_index:06d}"]
                    valid_index += 1
                result = dict(row)
                result["evaluation_protocol"] = evaluation_protocol
                result["family_ga_threshold"] = ga
                result["infernal_bit_score"] = bit_score
                result["relative_bit_score"] = bit_score / ga if bit_score is not None else None
                result["ga_pass"] = bool(bit_score is not None and bit_score > ga)
                scored_rows.append(result)

    per_family: dict[str, dict[str, Any]] = {}
    for family in observed:
        family_rows = [row for row in scored_rows if row["condition"] == family]
        per_family[family] = _family_metrics(family_rows, thresholds[family])
    macro_keys = (
        "validity",
        "unique_valid_over_requested",
        "infernal_score_coverage",
        "relative_bit_score_mean_scored",
        "relative_bit_score_median_scored",
        "ga_pass_rate_requested_denominator",
        "ga_pass_rate_scored_denominator",
    )
    summary = {
        "protocol": evaluation_protocol,
        "sample_representation": args.sample_representation,
        "complete": True,
        "formal": not args.allow_partial_smoke,
        "benchmark_label": ("BioMatrix public union-SFT zero-shot; same 17-family Infernal evaluation"
                            if native else (
            "Axis RfamGen Figure 2 paper-protocol reconstruction-17; "
            "G6-filtered, post-G6 Axis context-capped"
        )),
        "rf00005_ccu_included": False,
        "infernal_command_mode": ["cmalign", "--notrunc", "--sfile"],
        "ga_comparison": "bit_score > GA (strict)",
        "relative_bit_score": "bit_score / family GA",
        "invalid_slots_filtered_or_replaced": False,
        "family_aggregation": "within-family metrics followed by equal-family macro mean; undefined if any requested family is undefined, never drop that family",
        "macro_family_count": len(per_family),
        "families": per_family,
        "macro": {key: _macro(per_family, key) for key in macro_keys},
        "requested": len(scored_rows),
        "elapsed_sec": time.time() - started,
        "decoded_samples": str(args.decoded_samples.resolve()),
        "cm_root": str(args.cm_root.resolve()),
        "cmalign": str(args.cmalign.resolve()),
        "infernal_version": infernal_version,
        "records_output": str(records_output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with records_output.open("x", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
