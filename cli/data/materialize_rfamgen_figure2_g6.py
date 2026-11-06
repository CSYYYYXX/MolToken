#!/usr/bin/env python3
"""Materialize one RfamGen Figure 2 family's paper-protocol G6 pool.

This is a worker-side command.  Its input must already have passed the shared
Figure 2 cleaning, seed-removal, deduplication, and family length-band stages.
It runs ``cmalign --notrunc`` against the family CM, applies the public
RfamGen ``fold_by_SS_cons`` semantics, and retains sequences parsable by G6.

The command never downloads dependencies.  A fresh run requires a new output
directory.  ``--resume`` is explicit and only reuses batches whose semantic
batch marker and record manifest are complete; an incomplete batch is rerun.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterator, Sequence

from pipeline.benchmarks.rfamgen_figure2 import (
    build_nltk_g6_parser,
    count_fasta_records,
    fold_by_ss_cons,
    normalize_ss_cons,
    read_stockholm_alignment,
    recognize_g6,
    verify_direct_g6_against_nltk,
    verify_real_g6_case_against_nltk,
)


PROTOCOL = "axis-rfamgen-figure2-paper-g6-reconstruction-v1-20260830"
FAMILY_PATTERN = re.compile(r"RF\d{5}\Z")
RNA_ALPHABET = frozenset("ACGU")


@dataclass(frozen=True)
class SourceRecord:
    input_index: int
    original_header: str
    sequence: str
    alignment_id: str


@dataclass(frozen=True)
class BatchResult:
    batch_index: int
    input_records: int
    accepted_records: int
    rejection_reasons: dict[str, int]
    nltk_real_parity_eligible: int
    nltk_real_parity_checked: int


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    require_file(path, "JSON file")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_fasta(path: Path, family: str) -> Iterator[SourceRecord]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    seen_sequences: set[str] = set()
    header: str | None = None
    pieces: list[str] = []
    input_index = 0

    def finish_record() -> SourceRecord:
        nonlocal input_index
        if header is None:
            raise AssertionError("attempted to finish FASTA without a header")
        sequence = "".join(pieces).upper()
        if not sequence:
            raise ValueError(f"{path}: empty sequence for header {header!r}")
        invalid = set(sequence).difference(RNA_ALPHABET)
        if invalid:
            raise ValueError(
                f"{path}: input record {input_index} contains symbols outside A/C/G/U: "
                f"{sorted(invalid)}"
            )
        if sequence in seen_sequences:
            raise ValueError(
                f"{path}: duplicate canonical sequence at input record {input_index}"
            )
        seen_sequences.add(sequence)
        record = SourceRecord(
            input_index=input_index,
            original_header=header,
            sequence=sequence,
            alignment_id=f"{family}_{input_index:09d}",
        )
        input_index += 1
        return record

    with opener(path, "rt", encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield finish_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
                pieces = []
            else:
                if header is None:
                    raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
                pieces.append(line)
    if header is not None:
        yield finish_record()


def iter_batches(
    records: Iterator[SourceRecord], batch_size: int
) -> Iterator[list[SourceRecord]]:
    batch: list[SourceRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def inspect_family_cm(path: Path, family: str) -> dict[str, Any]:
    accessions: list[str] = []
    names: list[str] = []
    infernal_blocks = 0
    terminators = 0
    in_infernal_block = False
    with path.open("r", encoding="latin-1") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("INFERNAL1/"):
                infernal_blocks += 1
                in_infernal_block = True
            elif line.startswith("HMMER"):
                # A calibrated Infernal CM embeds its HMM filter after the CM
                # block.  That filter repeats NAME/ACC and has its own //; it
                # is not a second covariance model.
                in_infernal_block = False
            elif in_infernal_block and line.startswith("ACC"):
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"{path}: malformed ACC line {line!r}")
                accessions.append(fields[1])
            elif in_infernal_block and line.startswith("NAME"):
                fields = line.split(maxsplit=1)
                if len(fields) == 2:
                    names.append(fields[1])
            elif in_infernal_block and line == "//":
                terminators += 1
                in_infernal_block = False
    if infernal_blocks != 1 or accessions != [family] or terminators != 1:
        raise ValueError(
            f"{path}: expected exactly one CM with ACC {family}; "
            f"infernal_blocks={infernal_blocks}, accessions={accessions}, "
            f"terminators={terminators}"
        )
    return {
        "accession": accessions[0],
        "name": names[0] if names else None,
        "infernal_blocks": infernal_blocks,
    }


def cmalign_version(cmalign: Path) -> str:
    completed = subprocess.run(
        [str(cmalign), "-h"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{cmalign}: -h returned no version/banner text")
    return " | ".join(lines[:3])


def source_record_payload(record: SourceRecord) -> dict[str, Any]:
    return {
        "alignment_id": record.alignment_id,
        "input_index": record.input_index,
        "original_header": record.original_header,
        "sequence": record.sequence,
    }


def write_batch_inputs(batch_dir: Path, records: Sequence[SourceRecord]) -> None:
    source_tmp = batch_dir / "source_records.jsonl.tmp"
    fasta_tmp = batch_dir / "cmalign_input.fasta.tmp"
    with source_tmp.open("w", encoding="utf-8") as source_handle, fasta_tmp.open(
        "w", encoding="utf-8"
    ) as fasta_handle:
        for record in records:
            source_handle.write(
                json.dumps(source_record_payload(record), ensure_ascii=False) + "\n"
            )
            fasta_handle.write(f">{record.alignment_id}\n{record.sequence}\n")
    os.replace(source_tmp, batch_dir / "source_records.jsonl")
    os.replace(fasta_tmp, batch_dir / "cmalign_input.fasta")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require_file(path, "JSONL file")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
            records.append(payload)
    return records


def validate_completed_batch(
    batch_dir: Path,
    batch_index: int,
    records: Sequence[SourceRecord],
    nltk_real_parity_max_length: int,
) -> BatchResult:
    marker_path = batch_dir / "_SUCCESS.json"
    marker = load_json(marker_path)
    expected_source = [source_record_payload(record) for record in records]
    actual_source = read_jsonl(batch_dir / "source_records.jsonl")
    if actual_source != expected_source:
        raise ValueError(
            f"{batch_dir}: completed batch source records differ from current input"
        )
    record_manifest = read_jsonl(batch_dir / "records.jsonl")
    expected_ids = [record.alignment_id for record in records]
    actual_ids = [payload.get("alignment_id") for payload in record_manifest]
    if actual_ids != expected_ids:
        raise ValueError(f"{batch_dir}: record manifest IDs/order are inconsistent")
    accepted = sum(bool(payload.get("g6_parsable")) for payload in record_manifest)
    with (batch_dir / "accepted.fasta").open("r", encoding="utf-8") as handle:
        fasta_accepted = count_fasta_records(handle)
    if fasta_accepted != accepted:
        raise ValueError(
            f"{batch_dir}: accepted FASTA has {fasta_accepted}, manifest has {accepted}"
        )
    expected_marker = {
        "protocol": PROTOCOL,
        "batch_index": batch_index,
        "first_input_index": records[0].input_index,
        "last_input_index": records[-1].input_index,
        "input_records": len(records),
        "accepted_records": accepted,
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"{marker_path}: {key}={marker.get(key)!r}, expected {expected!r}"
            )
    reasons = Counter(
        str(payload["g6_reason"])
        for payload in record_manifest
        if not payload["g6_parsable"]
    )
    parity_eligible = sum(
        bool(payload.get("nltk_g6_parity_eligible")) for payload in record_manifest
    )
    expected_eligible = sum(
        len(record.sequence) <= nltk_real_parity_max_length for record in records
    )
    if parity_eligible != expected_eligible:
        raise ValueError(
            f"{batch_dir}: NLTK real-parity eligibility is inconsistent with "
            f"max length {nltk_real_parity_max_length}"
        )
    parity_checked = sum(
        bool(payload.get("nltk_g6_parity_checked")) for payload in record_manifest
    )
    if marker.get("nltk_real_parity_eligible") != parity_eligible:
        raise ValueError(f"{marker_path}: incorrect real-parity eligible count")
    if marker.get("nltk_real_parity_checked") != parity_checked:
        raise ValueError(f"{marker_path}: incorrect real-parity checked count")
    return BatchResult(
        batch_index,
        len(records),
        accepted,
        dict(reasons),
        parity_eligible,
        parity_checked,
    )


def run_cmalign(
    cmalign: Path,
    cm: Path,
    batch_dir: Path,
    cpu: int,
) -> list[str]:
    output_tmp = batch_dir / "alignment.sto.tmp"
    command = [
        str(cmalign),
        "--cpu",
        str(cpu),
        "--notrunc",
        "--outformat",
        "Pfam",
        "-o",
        str(output_tmp),
        str(cm),
        str(batch_dir / "cmalign_input.fasta"),
    ]
    with (batch_dir / "cmalign.log").open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            command,
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    require_file(output_tmp, "cmalign temporary output")
    os.replace(output_tmp, batch_dir / "alignment.sto")
    return command


def materialize_batch(
    batch_dir: Path,
    batch_index: int,
    records: Sequence[SourceRecord],
    cmalign: Path,
    cm: Path,
    cpu: int,
    nltk_parser: Any | None,
    nltk_real_parity_remaining: int,
    nltk_real_parity_max_length: int,
) -> BatchResult:
    write_batch_inputs(batch_dir, records)
    command = run_cmalign(cmalign, cm, batch_dir, cpu)
    alignment = read_stockholm_alignment(batch_dir / "alignment.sto")
    expected_ids = [record.alignment_id for record in records]
    if set(alignment.sequences) != set(expected_ids):
        missing = sorted(set(expected_ids).difference(alignment.sequences))
        extra = sorted(set(alignment.sequences).difference(expected_ids))
        raise ValueError(
            f"{batch_dir}: cmalign output IDs disagree; missing={missing}, extra={extra}"
        )

    normalized_consensus = normalize_ss_cons(alignment.ss_cons)
    manifest_tmp = batch_dir / "records.jsonl.tmp"
    accepted_tmp = batch_dir / "accepted.fasta.tmp"
    accepted_count = 0
    rejection_reasons: Counter[str] = Counter()
    parity_eligible = 0
    parity_checked = 0
    with manifest_tmp.open("w", encoding="utf-8") as manifest_handle, accepted_tmp.open(
        "w", encoding="utf-8"
    ) as accepted_handle:
        for record in records:
            folded_sequence, folded_structure = fold_by_ss_cons(
                alignment.sequences[record.alignment_id], normalized_consensus
            )
            if folded_sequence != record.sequence:
                raise ValueError(
                    f"{batch_dir}: cmalign/fold changed {record.alignment_id}: "
                    f"input length={len(record.sequence)}, folded length={len(folded_sequence)}"
                )
            g6 = recognize_g6(folded_sequence, folded_structure)
            eligible_for_parity = (
                len(folded_sequence) <= nltk_real_parity_max_length
            )
            if eligible_for_parity:
                parity_eligible += 1
            check_real_parity = (
                nltk_parser is not None
                and eligible_for_parity
                and parity_checked < nltk_real_parity_remaining
            )
            if check_real_parity:
                verify_real_g6_case_against_nltk(
                    folded_sequence,
                    folded_structure,
                    parser=nltk_parser,
                )
                parity_checked += 1
            payload = {
                "alignment_id": record.alignment_id,
                "input_index": record.input_index,
                "original_header": record.original_header,
                "residue_length": len(record.sequence),
                "folded_structure": folded_structure,
                "g6_parsable": g6.parsable,
                "g6_reason": g6.reason,
                "g6_derivation_rule_count": g6.derivation_rule_count,
                "nltk_g6_parity_eligible": eligible_for_parity,
                "nltk_g6_parity_checked": check_real_parity,
            }
            manifest_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if g6.parsable:
                accepted_handle.write(
                    f">{record.alignment_id} {record.original_header}\n{record.sequence}\n"
                )
                accepted_count += 1
            else:
                rejection_reasons[g6.reason] += 1
    os.replace(manifest_tmp, batch_dir / "records.jsonl")
    os.replace(accepted_tmp, batch_dir / "accepted.fasta")

    marker = {
        "protocol": PROTOCOL,
        "batch_index": batch_index,
        "first_input_index": records[0].input_index,
        "last_input_index": records[-1].input_index,
        "input_records": len(records),
        "accepted_records": accepted_count,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "nltk_real_parity_eligible": parity_eligible,
        "nltk_real_parity_checked": parity_checked,
        "cmalign_command": command,
    }
    atomic_write_json(batch_dir / "_SUCCESS.json", marker)
    return BatchResult(
        batch_index=batch_index,
        input_records=len(records),
        accepted_records=accepted_count,
        rejection_reasons=dict(rejection_reasons),
        nltk_real_parity_eligible=parity_eligible,
        nltk_real_parity_checked=parity_checked,
    )


def prepare_output_dir(output_dir: Path, resume: bool) -> None:
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"--resume output directory does not exist: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "batches").mkdir(exist_ok=True)


def validate_or_write_run_spec(output_dir: Path, spec: dict[str, Any], resume: bool) -> None:
    path = output_dir / "run_spec.json"
    if not resume:
        atomic_write_json(path, spec)
        return
    existing = load_json(path)
    semantic_keys = (
        "protocol",
        "family",
        "input_fasta",
        "family_cm",
        "cmalign",
        "cmalign_version",
        "batch_size",
        "cmalign_mode",
        "input_pool_mode",
        "nltk_parity",
    )
    for key in semantic_keys:
        if existing.get(key) != spec.get(key):
            raise ValueError(
                f"{path}: resume setting {key}={spec.get(key)!r} differs from "
                f"existing {existing.get(key)!r}"
            )


def consolidate_batches(
    output_dir: Path,
    batch_results: Sequence[BatchResult],
) -> tuple[Path, Path]:
    records_tmp = output_dir / "records.jsonl.tmp"
    accepted_tmp = output_dir / "accepted.fasta.tmp"
    with records_tmp.open("wb") as records_out, accepted_tmp.open("wb") as accepted_out:
        for result in batch_results:
            batch_dir = output_dir / "batches" / f"batch_{result.batch_index:06d}"
            with (batch_dir / "records.jsonl").open("rb") as source:
                shutil.copyfileobj(source, records_out)
            with (batch_dir / "accepted.fasta").open("rb") as source:
                shutil.copyfileobj(source, accepted_out)
    records_path = output_dir / "records.jsonl"
    accepted_path = output_dir / "accepted.fasta"
    os.replace(records_tmp, records_path)
    os.replace(accepted_tmp, accepted_path)
    return records_path, accepted_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--input-fasta", required=True, type=Path)
    parser.add_argument("--family-cm", required=True, type=Path)
    parser.add_argument("--cmalign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--input-pool-mode",
        choices=("canonical_deduplicated", "repository_raw_unique"),
        default="canonical_deduplicated",
        help="Upstream Figure 2 deduplication mode recorded as reconstruction provenance",
    )
    parser.add_argument(
        "--nltk-exhaustive-max-length",
        type=int,
        default=4,
        help="Exhaustively compare all RNA/dot-bracket strings through this length",
    )
    parser.add_argument(
        "--nltk-real-parity-records",
        type=int,
        default=8,
        help="Compare up to this many real folded family records with NLTK",
    )
    parser.add_argument(
        "--nltk-real-parity-max-length",
        type=int,
        default=256,
        help="Avoid cubic NLTK smoke checks above this residue length",
    )
    parser.add_argument(
        "--skip-nltk-parity",
        action="store_true",
        help="Explicitly skip exhaustive-short and real-record parity against NLTK",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only semantically complete batches after an interrupted run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not FAMILY_PATTERN.fullmatch(args.family):
        raise ValueError(f"invalid Rfam accession: {args.family!r}")
    if args.cpu < 1:
        raise ValueError("--cpu must be at least one")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least one")
    if args.nltk_exhaustive_max_length < 0:
        raise ValueError("--nltk-exhaustive-max-length must be non-negative")
    if args.nltk_real_parity_records < 0:
        raise ValueError("--nltk-real-parity-records must be non-negative")
    if args.nltk_real_parity_max_length < 1:
        raise ValueError("--nltk-real-parity-max-length must be positive")

    input_fasta = require_file(args.input_fasta, "length-filtered FASTA")
    family_cm = require_file(args.family_cm, "family CM")
    cmalign = require_file(args.cmalign, "cmalign executable")
    if not os.access(cmalign, os.X_OK):
        raise PermissionError(f"cmalign is not executable: {cmalign}")
    cm_info = inspect_family_cm(family_cm, args.family)
    version = cmalign_version(cmalign)
    if args.skip_nltk_parity:
        parity_cases = 0
        nltk_parser = None
    else:
        parity_cases = verify_direct_g6_against_nltk(
            args.nltk_exhaustive_max_length
        )
        nltk_parser = build_nltk_g6_parser()

    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.resume)
    run_spec = {
        "protocol": PROTOCOL,
        "protocol_scope": (
            "RfamGen Figure 2 paper-protocol reconstruction; not an official "
            "published record-ID split"
        ),
        "family": args.family,
        "input_pool_mode": args.input_pool_mode,
        "input_fasta": str(input_fasta),
        "family_cm": str(family_cm),
        "cm_info": cm_info,
        "cmalign": str(cmalign),
        "cmalign_version": version,
        "cmalign_mode": ["--notrunc", "--outformat", "Pfam"],
        "cpu": args.cpu,
        "batch_size": args.batch_size,
        "nltk_parity": {
            "skipped_explicitly": bool(args.skip_nltk_parity),
            "exhaustive_max_length": args.nltk_exhaustive_max_length,
            "exhaustive_cases_checked": parity_cases,
            "real_records_requested": args.nltk_real_parity_records,
            "real_record_max_length": args.nltk_real_parity_max_length,
        },
        "official_reference": {
            "fold": "rfamgen/src/preprocess.py (external reference implementation)",
            "grammar": "rfamgen/src/grammar.py (external reference implementation)",
        },
    }
    validate_or_write_run_spec(output_dir, run_spec, args.resume)

    batch_results: list[BatchResult] = []
    for batch_index, records in enumerate(
        iter_batches(iter_fasta(input_fasta, args.family), args.batch_size)
    ):
        batch_dir = output_dir / "batches" / f"batch_{batch_index:06d}"
        batch_dir.mkdir(exist_ok=True)
        marker_path = batch_dir / "_SUCCESS.json"
        if args.resume and marker_path.is_file():
            result = validate_completed_batch(
                batch_dir,
                batch_index,
                records,
                args.nltk_real_parity_max_length,
            )
            action = "reused"
        else:
            parity_already_checked = sum(
                previous.nltk_real_parity_checked for previous in batch_results
            )
            parity_remaining = max(
                0, args.nltk_real_parity_records - parity_already_checked
            )
            result = materialize_batch(
                batch_dir,
                batch_index,
                records,
                cmalign,
                family_cm,
                args.cpu,
                nltk_parser,
                parity_remaining,
                args.nltk_real_parity_max_length,
            )
            action = "processed"
        batch_results.append(result)
        print(
            json.dumps(
                {
                    "action": action,
                    "batch": batch_index,
                    "input": result.input_records,
                    "accepted": result.accepted_records,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if not batch_results:
        raise ValueError(f"input FASTA contains no records: {input_fasta}")
    records_path, accepted_path = consolidate_batches(output_dir, batch_results)
    input_records = sum(result.input_records for result in batch_results)
    accepted_records = sum(result.accepted_records for result in batch_results)
    rejection_reasons: Counter[str] = Counter()
    for result in batch_results:
        rejection_reasons.update(result.rejection_reasons)
    real_parity_eligible = sum(
        result.nltk_real_parity_eligible for result in batch_results
    )
    real_parity_checked = sum(
        result.nltk_real_parity_checked for result in batch_results
    )
    with accepted_path.open("r", encoding="utf-8") as handle:
        consolidated_accepted = count_fasta_records(handle)
    if consolidated_accepted != accepted_records:
        raise AssertionError(
            f"consolidated accepted FASTA has {consolidated_accepted}, expected "
            f"{accepted_records}"
        )
    manifest = {
        **run_spec,
        "status": "complete",
        "batches": len(batch_results),
        "input_records": input_records,
        "accepted_records": accepted_records,
        "rejected_records": input_records - accepted_records,
        "accepted_fraction": accepted_records / input_records,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "nltk_real_parity_eligible_records": real_parity_eligible,
        "nltk_real_parity_checked_records": real_parity_checked,
        "outputs": {
            "accepted_fasta": str(accepted_path),
            "record_manifest_jsonl": str(records_path),
            "batch_root": str(output_dir / "batches"),
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
