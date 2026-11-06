#!/usr/bin/env python3
"""Materialize canonical Figure 2 length-band FASTA pools on a worker.

The public RfamGen repository deduplicates exact raw full-family sequences,
removes seed sequences, filters ambiguous residues, and applies a family
length band.  Axis uses an additional canonical ACGU deduplication so records
that differ only through raw spelling (for example T versus U) cannot leak
across a later split.  This command materializes that explicitly reconstructed
pool for the 17 usable Figure 2 families.

RF00005 is intentionally unavailable here: the paper uses an RF00005_CCU
subset whose official record membership is not published.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import OrderedDict
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

from pipeline.benchmarks.rfamgen_figure2 import count_fasta_records


PROTOCOL = "axis-rfamgen-figure2-canonical-length-pools-v1-20260830"
LENGTH_REPORT_PROTOCOL = "rfamgen-figure2-length-frequency-audit-v1-20260830"
FORMAL_FAMILIES = (
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
EXCLUDED_FAMILY = "RF00005"
RNA_SOURCE_ALPHABET = frozenset("ACGUT")

DEFAULT_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / 'downloads_raw/AxisBiologicalSequenceBenchmarks-20260830/rfamgen_14.7')
)
DEFAULT_LENGTH_REPORT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / 'reports/rfam_figure2_length_filter_audit_20260830.json')
)


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


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    header: str | None = None
    pieces: list[str] = []
    with opener(path, "rt", encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(pieces)
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
                pieces = []
            else:
                if header is None:
                    raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
                pieces.append(line)
    if header is not None:
        yield header, "".join(pieces)


def normalize_full_sequence(raw_sequence: str) -> str | None:
    sequence = raw_sequence.upper().replace("-", "").replace(".", "")
    if not sequence or set(sequence).difference(RNA_SOURCE_ALPHABET):
        return None
    return sequence.replace("T", "U")


def parse_rfam_seed(path: Path, families: set[str]) -> dict[str, set[str]]:
    seeds = {family: set() for family in families}
    current_family: str | None = None
    aligned: OrderedDict[str, list[str]] = OrderedDict()

    def finish_block() -> None:
        nonlocal current_family, aligned
        if current_family in seeds:
            for identifier, pieces in aligned.items():
                sequence = "".join(pieces).upper().replace("-", "").replace(".", "")
                sequence = sequence.replace("T", "U")
                if not sequence:
                    raise ValueError(
                        f"{path}: empty normalized seed {current_family}/{identifier}"
                    )
                seeds[current_family].add(sequence)
        current_family = None
        aligned = OrderedDict()

    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line == "//":
                finish_block()
                continue
            if line.startswith("#=GF AC"):
                fields = line.split()
                if len(fields) != 3:
                    raise ValueError(f"{path}:{line_number}: malformed AC line")
                current_family = fields[2]
                continue
            if not line or line.startswith("#") or current_family not in seeds:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: malformed Stockholm sequence row"
                )
            identifier, aligned_piece = fields
            aligned.setdefault(identifier, []).append(aligned_piece)
    if current_family is not None or aligned:
        raise ValueError(f"{path}: unterminated Stockholm block")
    return seeds


def load_length_bounds(
    path: Path, families: Sequence[str]
) -> dict[str, tuple[int, int, int, int]]:
    with require_file(path, "length audit report").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("protocol") != LENGTH_REPORT_PROTOCOL:
        raise ValueError(
            f"{path}: unexpected length report protocol {report.get('protocol')!r}"
        )
    result: dict[str, tuple[int, int, int, int]] = {}
    for family in families:
        try:
            payload = report["families"][family]["modes"][
                "canonical_deduplicated_order"
            ]
            low, high = payload["frequent_length_bounds"]
            before_length = payload["records"]
            selected = payload["records_inside_inclusive_bounds"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: missing canonical length data for {family}") from error
        result[family] = (
            int(low),
            int(high),
            int(before_length),
            int(selected),
        )
    return result


def build_canonical_pool(
    fasta_path: Path,
    seed_sequences: set[str],
) -> tuple[list[tuple[str, str, int]], dict[str, int]]:
    raw_unique: OrderedDict[str, tuple[str, int]] = OrderedDict()
    raw_records = 0
    for header, raw_sequence in iter_fasta(fasta_path):
        # Assignment intentionally preserves the first sequence position while
        # matching the public repository's last-description-wins dictionary.
        raw_unique[raw_sequence] = (header, raw_records)
        raw_records += 1

    canonical_seen: set[str] = set()
    canonical_pool: list[tuple[str, str, int]] = []
    ambiguous_removed = 0
    seed_removed = 0
    canonical_collisions_removed = 0
    for raw_sequence, (header, source_index) in raw_unique.items():
        canonical = normalize_full_sequence(raw_sequence)
        if canonical is None:
            ambiguous_removed += 1
            continue
        if canonical in seed_sequences:
            seed_removed += 1
            continue
        if canonical in canonical_seen:
            canonical_collisions_removed += 1
            continue
        canonical_seen.add(canonical)
        canonical_pool.append((header, canonical, source_index))
    stats = {
        "raw_records": raw_records,
        "raw_exact_unique_records": len(raw_unique),
        "raw_exact_duplicates_removed": raw_records - len(raw_unique),
        "ambiguous_or_empty_removed": ambiguous_removed,
        "seed_records_removed": seed_removed,
        "canonical_collisions_removed": canonical_collisions_removed,
        "canonical_seed_removed_records": len(canonical_pool),
    }
    return canonical_pool, stats


def validate_completed_family(
    family_dir: Path,
    family: str,
    bounds: tuple[int, int, int, int],
) -> dict[str, Any]:
    manifest_path = require_file(family_dir / "manifest.json", "family manifest")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    low, high, expected_before, expected_selected = bounds
    expected = {
        "protocol": PROTOCOL,
        "family": family,
        "length_bounds_inclusive": [low, high],
        "canonical_seed_removed_records": expected_before,
        "selected_records": expected_selected,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{manifest_path}: {key}={manifest.get(key)!r}, expected {value!r}"
            )
    with require_file(family_dir / "length_filtered.fasta", "family FASTA").open(
        "r", encoding="utf-8"
    ) as handle:
        fasta_count = count_fasta_records(handle)
    records_path = require_file(family_dir / "records.jsonl", "family records")
    with records_path.open("r", encoding="utf-8") as handle:
        record_count = sum(1 for line in handle if line.strip())
    if fasta_count != expected_selected or record_count != expected_selected:
        raise ValueError(
            f"{family_dir}: completed counts FASTA={fasta_count}, JSONL={record_count}, "
            f"expected={expected_selected}"
        )
    return manifest


def materialize_family(
    family_dir: Path,
    family: str,
    fasta_path: Path,
    seed_sequences: set[str],
    bounds: tuple[int, int, int, int],
) -> dict[str, Any]:
    low, high, expected_before, expected_selected = bounds
    pool, stats = build_canonical_pool(fasta_path, seed_sequences)
    if len(pool) != expected_before:
        raise ValueError(
            f"{family}: canonical pool has {len(pool)}, length report expects "
            f"{expected_before}"
        )
    selected = [record for record in pool if low <= len(record[1]) <= high]
    if len(selected) != expected_selected:
        raise ValueError(
            f"{family}: length band selected {len(selected)}, report expects "
            f"{expected_selected}"
        )

    fasta_tmp = family_dir / "length_filtered.fasta.tmp"
    records_tmp = family_dir / "records.jsonl.tmp"
    with fasta_tmp.open("w", encoding="utf-8") as fasta_handle, records_tmp.open(
        "w", encoding="utf-8"
    ) as records_handle:
        for selected_index, (header, sequence, source_index) in enumerate(selected):
            output_id = f"{family}_length_{selected_index:09d}"
            fasta_handle.write(f">{output_id} {header}\n{sequence}\n")
            records_handle.write(
                json.dumps(
                    {
                        "output_id": output_id,
                        "selected_index": selected_index,
                        "raw_source_index": source_index,
                        "original_header": header,
                        "residue_length": len(sequence),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    fasta_output = family_dir / "length_filtered.fasta"
    records_output = family_dir / "records.jsonl"
    os.replace(fasta_tmp, fasta_output)
    os.replace(records_tmp, records_output)
    manifest = {
        "protocol": PROTOCOL,
        "protocol_scope": (
            "paper-protocol reconstruction with Axis canonical ACGU deduplication"
        ),
        "family": family,
        "source_fasta": str(fasta_path),
        "seed_sequence_count": len(seed_sequences),
        "length_bounds_inclusive": [low, high],
        **stats,
        "outside_length_band_removed": len(pool) - len(selected),
        "selected_records": len(selected),
        "outputs": {
            "fasta": str(fasta_output),
            "records_jsonl": str(records_output),
        },
    }
    atomic_write_json(family_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--length-report", type=Path, default=DEFAULT_LENGTH_REPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=list(FORMAL_FAMILIES))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only family outputs with a complete semantic manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = tuple(args.families)
    if len(set(families)) != len(families):
        raise ValueError("--families contains duplicates")
    unknown = sorted(set(families).difference(FORMAL_FAMILIES))
    if unknown:
        raise ValueError(
            f"unsupported formal families: {unknown}; {EXCLUDED_FAMILY} is explicitly "
            "excluded because official CCU membership is unavailable"
        )

    root = args.root.resolve()
    seed_path = require_file(root / "Rfam.seed.gz", "Rfam seed archive")
    length_report = require_file(args.length_report, "length report")
    fasta_paths = {
        family: require_file(
            root / "fasta_files" / f"{family}.fa.gz", f"{family} full FASTA"
        )
        for family in families
    }
    bounds = load_length_bounds(length_report, families)
    seeds = parse_rfam_seed(seed_path, set(families))

    output_dir = args.output_dir.resolve()
    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"--resume output directory does not exist: {output_dir}")
        existing_top = output_dir / "manifest.json"
        if existing_top.is_file():
            with existing_top.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            resume_expectations = {
                "protocol": PROTOCOL,
                "root": str(root),
                "length_report": str(length_report),
                "input_pool_mode": "canonical_deduplicated",
                "families": list(families),
            }
            for key, expected in resume_expectations.items():
                if previous.get(key) != expected:
                    raise ValueError(
                        f"{existing_top}: resume setting {key}={expected!r} differs "
                        f"from existing {previous.get(key)!r}"
                    )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    family_manifests: dict[str, Any] = {}
    for family in families:
        family_dir = output_dir / family
        family_dir.mkdir(exist_ok=True)
        if args.resume and (family_dir / "manifest.json").is_file():
            manifest = validate_completed_family(family_dir, family, bounds[family])
            action = "reused"
        else:
            manifest = materialize_family(
                family_dir,
                family,
                fasta_paths[family],
                seeds[family],
                bounds[family],
            )
            action = "processed"
        family_manifests[family] = manifest
        print(
            json.dumps(
                {
                    "action": action,
                    "family": family,
                    "selected_records": manifest["selected_records"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    total_selected = sum(
        int(manifest["selected_records"]) for manifest in family_manifests.values()
    )
    top_manifest = {
        "protocol": PROTOCOL,
        "protocol_scope": (
            "RfamGen Figure 2 paper-protocol reconstruction; exact official "
            "record membership and split IDs were not published"
        ),
        "root": str(root),
        "length_report": str(length_report),
        "input_pool_mode": "canonical_deduplicated",
        "families": list(families),
        "family_count": len(families),
        "selected_records": total_selected,
        "explicitly_excluded": {
            EXCLUDED_FAMILY: "paper uses RF00005_CCU; official subset IDs unavailable"
        },
        "family_manifests": {
            family: str(output_dir / family / "manifest.json") for family in families
        },
        "status": "complete",
    }
    atomic_write_json(output_dir / "manifest.json", top_manifest)
    print(json.dumps(top_manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
