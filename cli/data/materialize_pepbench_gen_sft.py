#!/usr/bin/env python3
"""Materialize the pretraining-exposed PepBench-Gen downstream task.

This worker-only command streams the complete PepBenchmark peptide source,
audits the source rows that the flagship graph pipeline could not encode,
freezes a length-stratified train/inner-validation/test split over only the
eligible canonical rows, and encodes each target with the frozen Axis
heavy-graph -> FDDFS -> BPE codec.  The condition is exactly the bare decimal
target length (for example ``"20"``).

PepBench-Gen is deliberately labelled *pretraining-exposed*.  The flagship
run configs and full-compact graph summaries are read as provenance: they show
that the flagships used the full compact corpus rather than the older sampled
corpus.  This is not SFT supervision leakage, but it is also not an unseen
generalization benchmark.

The expected non-canonical/out-of-range source rows are retained verbatim in
an explicit exclusion audit; they are never repaired or assigned to a split.
Any encoding failure among eligible rows, or any context length above 512,
makes the command fail non-zero.  No target is truncated or silently dropped.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.active_scope import FORMAL_TASK_DATASET_SCOPES  # noqa: E402
from pipeline.finetune.axis_b_data import build_axis_b_index  # noqa: E402
from pipeline.runtime_host import require_compute_host  # noqa: E402


ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / '')
)
DEFAULT_SOURCE = (
    ROOT / "data/clean_normv2/peptide/pepbenchmark_raw_normv2.jsonl.zst"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/processed/axis_b_biological_sft/pepbench_gen_pretraining_exposed_v1"
)
DEFAULT_GUIDE = ROOT / "data/fddfs_guide_recount_v1/guide_counts.json"
DEFAULT_VOCAB = ROOT / "data/fddfs_guide_recount_v1/vocab.json"
DEFAULT_MERGES = (
    ROOT
    / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1"
    / "codebook_fddfs_prefix70_mf80000_v1/merges.csv"
)
DEFAULT_TOKENIZER = ROOT / "data/pretrained_encoders/all-mpnet-base-v2/tokenizer.json"
DEFAULT_FULL_COMPACT_DONE = (
    ROOT
    / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1"
    / "encoded_compact_int_prefix70_mf80000_11w_v3/done.json"
)
DEFAULT_FLAGSHIP_SMALL_CONFIG = ROOT / "runs/flagship_v1/small_0p55b/run_config.json"
DEFAULT_FLAGSHIP_BIG_CONFIG = ROOT / "runs/flagship_v1/big_1p44b/run_config.json"
DEFAULT_GRAPH_ROOT = ROOT / "data/graphs_heavy_atom_normv2_compact_v1"

TASK = "pepbench_gen_length_conditioned"
DATASET_ID = "pepbench_gen_pretraining_exposed_v1"
PROTOCOL = "axis-pepbench-gen-pretraining-exposed-sft-v2-20260830"
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 5
MAX_LENGTH = 80
CONTEXT_BUDGET = 512
TARGET_FRAME_TOKENS = 3
EXPECTED_SOURCE_RECORDS = 71_137
EXPECTED_FULL_COMPACT_ENCODED = 70_117
EXPECTED_FULL_COMPACT_FAILED = 1_020
EXPECTED_ELIGIBLE_RECORDS = EXPECTED_FULL_COMPACT_ENCODED
EXPECTED_EXCLUDED_RECORDS = EXPECTED_FULL_COMPACT_FAILED
HISTORICAL_SAMPLED_PEPBENCH_RECORDS = 54_317
EXCLUSION_REASON_DEFINITIONS = {
    "empty_sequence": "normalized source sequence is empty",
    "contains_noncanonical_amino_acid": (
        "normalized source sequence contains at least one symbol outside the "
        "20 canonical amino-acid alphabet ACDEFGHIKLMNPQRSTVWY"
    ),
    "length_below_minimum": f"observed sequence length is below {MIN_LENGTH}",
    "length_above_maximum": f"observed sequence length is above {MAX_LENGTH}",
}
SPLITS = ("train", "validation", "test")
KNOWN_ZSTD = (
    
    "/usr/bin/zstd",
    "/usr/local/bin/zstd",
)


@dataclass(frozen=True)
class SourceIndexRecord:
    source_id: str
    record_id: str
    sequence: str
    length: int
    line_number: int = 0
    source_raw_sequence: str | None = None


@dataclass(frozen=True)
class ExcludedSourceRecord:
    source_id: str
    record_id: str
    original_sequence: str
    observed_length: int
    declared_length: int
    reasons: tuple[str, ...]
    noncanonical_symbols: tuple[str, ...]
    line_number: int
    source_raw_sequence: str | None

    def reference(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "source_id": self.source_id,
            "source_line_number": self.line_number,
            "original_sequence": self.original_sequence,
            "source_raw_sequence": self.source_raw_sequence,
            "observed_length": self.observed_length,
            "declared_length": self.declared_length,
            "reasons": list(self.reasons),
            "noncanonical_symbols": list(self.noncanonical_symbols),
            "eligibility_status": "excluded_expected_unencodable",
        }


@dataclass(frozen=True)
class ProvenancePathMapping:
    """One explicit old-prefix -> local-prefix comparison-only mapping."""

    old_prefix: Path
    local_prefix: Path

    def audit_record(self) -> dict[str, str]:
        return {
            "old_prefix": str(self.old_prefix),
            "local_prefix": str(self.local_prefix),
        }


@dataclass(frozen=True)
class SourceScan:
    raw_records: int
    eligible_records: tuple[SourceIndexRecord, ...]
    excluded_records: tuple[ExcludedSourceRecord, ...]
    exclusion_reason_counts: dict[str, int]


def require_worker() -> None:
    require_compute_host("PepBench-Gen materialization", socket.gethostname())


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: {label} root must be an object")
    return payload


def find_zstd() -> str:
    discovered = shutil.which("zstd")
    if discovered:
        return discovered
    for candidate in KNOWN_ZSTD:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("zstd executable is required to read .jsonl.zst input")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSON objects from plain JSONL or a zstd-compressed JSONL file."""

    if path.suffix != ".zst":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL line")
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number}: record is not an object")
                yield payload
        return

    command = [find_zstd(), "-q", "-dc", str(path)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    assert process.stderr is not None
    for line_number, raw_line in enumerate(process.stdout, start=1):
        if not raw_line.strip():
            process.kill()
            process.wait()
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            process.kill()
            process.wait()
            raise ValueError(f"{path}:{line_number}: record is not an object")
        yield payload
    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"zstd decompression failed for {path}: rc={return_code}, stderr={stderr.strip()}"
        )


def classify_source_row(
    payload: Mapping[str, Any], *, path: Path, line_number: int
) -> SourceIndexRecord | ExcludedSourceRecord:
    """Classify one source row without repairing its sequence.

    Missing identity/schema fields remain hard source-integrity errors.  Only
    sequence properties that make a row ineligible under the intended
    PepBenchmark/flagship graph contract are represented as audited
    exclusions.
    """

    source_id = str(payload.get("id") or "").strip()
    domain = str(payload.get("domain") or "").strip()
    source = str(payload.get("source") or "").strip()
    sequence_value = payload.get("normalized")
    declared_length = payload.get("length")
    if not source_id:
        raise ValueError(f"{path}:{line_number}: missing source id")
    if domain != "peptide" or source != "PepBenchmark":
        raise ValueError(
            f"{path}:{line_number}: expected peptide/PepBenchmark, got {domain}/{source}"
        )
    if not isinstance(sequence_value, str):
        raise ValueError(
            f"{path}:{line_number}: normalized sequence must be a string"
        )
    sequence = sequence_value
    if not isinstance(declared_length, int) or isinstance(declared_length, bool):
        raise ValueError(f"{path}:{line_number}: declared length must be an integer")
    if declared_length != len(sequence):
        raise ValueError(
            f"{path}:{line_number}: declared length {declared_length!r} != {len(sequence)}"
        )
    raw_value = payload.get("raw")
    if raw_value is not None and not isinstance(raw_value, str):
        raise ValueError(f"{path}:{line_number}: raw sequence must be a string or null")

    reasons: list[str] = []
    noncanonical = tuple(sorted(set(sequence).difference(AA_ALPHABET)))
    if not sequence:
        reasons.append("empty_sequence")
    if noncanonical:
        reasons.append("contains_noncanonical_amino_acid")
    if len(sequence) < MIN_LENGTH:
        reasons.append("length_below_minimum")
    if len(sequence) > MAX_LENGTH:
        reasons.append("length_above_maximum")
    record_id = f"pepbench-gen:{source_id}"
    if reasons:
        return ExcludedSourceRecord(
            source_id=source_id,
            record_id=record_id,
            original_sequence=sequence,
            observed_length=len(sequence),
            declared_length=declared_length,
            reasons=tuple(reasons),
            noncanonical_symbols=noncanonical,
            line_number=line_number,
            source_raw_sequence=raw_value,
        )
    return SourceIndexRecord(
        source_id=source_id,
        record_id=record_id,
        sequence=sequence,
        length=len(sequence),
        line_number=line_number,
        source_raw_sequence=raw_value,
    )


def scan_source_index(
    path: Path,
    *,
    expected_source_records: int = EXPECTED_SOURCE_RECORDS,
    expected_eligible_records: int = EXPECTED_ELIGIBLE_RECORDS,
    expected_excluded_records: int = EXPECTED_EXCLUDED_RECORDS,
) -> SourceScan:
    eligible: list[SourceIndexRecord] = []
    excluded: list[ExcludedSourceRecord] = []
    source_ids: set[str] = set()
    record_ids: set[str] = set()
    sequences: set[str] = set()
    for line_number, payload in enumerate(iter_jsonl(path), start=1):
        record = classify_source_row(payload, path=path, line_number=line_number)
        if record.source_id in source_ids:
            raise ValueError(f"{path}:{line_number}: duplicate source id {record.source_id}")
        if record.record_id in record_ids:
            raise ValueError(f"{path}:{line_number}: duplicate record id {record.record_id}")
        sequence = (
            record.sequence
            if isinstance(record, SourceIndexRecord)
            else record.original_sequence
        )
        if sequence in sequences:
            raise ValueError(f"{path}:{line_number}: duplicate source sequence")
        source_ids.add(record.source_id)
        record_ids.add(record.record_id)
        sequences.add(sequence)
        if isinstance(record, SourceIndexRecord):
            eligible.append(record)
        else:
            excluded.append(record)

    raw_records = len(eligible) + len(excluded)
    if raw_records != expected_source_records:
        raise ValueError(
            f"PepBenchmark raw source count changed: expected "
            f"{expected_source_records}, got {raw_records}"
        )
    if len(eligible) != expected_eligible_records:
        raise ValueError(
            f"PepBenchmark eligible canonical count changed: expected "
            f"{expected_eligible_records}, got {len(eligible)}"
        )
    if len(excluded) != expected_excluded_records:
        raise ValueError(
            f"PepBenchmark exclusion count changed: expected "
            f"{expected_excluded_records}, got {len(excluded)}"
        )
    reason_counts = Counter(
        reason for record in excluded for reason in record.reasons
    )
    return SourceScan(
        raw_records=raw_records,
        eligible_records=tuple(eligible),
        excluded_records=tuple(excluded),
        exclusion_reason_counts=dict(sorted(reason_counts.items())),
    )


def stable_score(seed: int, stage: str, length: int, record_id: str) -> bytes:
    """Deterministic split rank; this digest is not an acceptance gate."""

    text = f"{seed}\0{stage}\0{length}\0{record_id}"
    return hashlib.sha256(text.encode("utf-8")).digest()


def rounded_fraction_count(size: int, fraction: float) -> int:
    if size < 1:
        raise ValueError("cannot split an empty stratum")
    if not 0.0 < fraction < 1.0:
        raise ValueError("split fraction must be in (0, 1)")
    return int(math.floor(size * fraction + 0.5))


def exact_length_stratified_split(
    records: Sequence[SourceIndexRecord],
    *,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> tuple[dict[str, str], dict[int, dict[str, int]]]:
    """Freeze test first, then inner-validation, independently per length."""

    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("test + validation fractions must be below one")
    by_length: dict[int, list[SourceIndexRecord]] = defaultdict(list)
    for record in records:
        by_length[record.length].append(record)
    assignments: dict[str, str] = {}
    length_counts: dict[int, dict[str, int]] = {}
    for length in sorted(by_length):
        group = by_length[length]
        if len(group) < 3:
            raise ValueError(f"length {length} has fewer than three records")
        test_count = rounded_fraction_count(len(group), test_fraction)
        validation_count = rounded_fraction_count(len(group), validation_fraction)
        if test_count < 1 or validation_count < 1:
            raise ValueError(f"length {length} cannot provide non-empty held-out splits")
        if test_count + validation_count >= len(group):
            raise ValueError(f"length {length} split leaves no training records")

        test_order = sorted(
            group,
            key=lambda record: (
                stable_score(seed, "test", length, record.record_id),
                record.record_id,
            ),
        )
        test_ids = {record.record_id for record in test_order[:test_count]}
        training_candidates = [
            record for record in group if record.record_id not in test_ids
        ]
        validation_order = sorted(
            training_candidates,
            key=lambda record: (
                stable_score(seed, "inner-validation", length, record.record_id),
                record.record_id,
            ),
        )
        validation_ids = {
            record.record_id for record in validation_order[:validation_count]
        }
        counts = Counter()
        for record in group:
            if record.record_id in test_ids:
                split = "test"
            elif record.record_id in validation_ids:
                split = "validation"
            else:
                split = "train"
            assignments[record.record_id] = split
            counts[split] += 1
        length_counts[length] = {split: int(counts[split]) for split in SPLITS}
    if len(assignments) != len(records):
        raise AssertionError("split assignment count differs from source records")
    return assignments, length_counts


def _normalized_absolute_path(raw: str | Path, *, label: str) -> Path:
    """Normalize a provenance path without consulting the historical mount."""

    value = Path(raw).expanduser()
    if not value.is_absolute():
        raise ValueError(f"{label} must be an absolute path, got {raw!s}")
    if ".." in value.parts:
        raise ValueError(f"{label} is ambiguous because it contains '..': {raw!s}")
    return Path(os.path.normpath(str(value)))


def _path_is_within(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def parse_provenance_path_mappings(
    raw_mappings: Sequence[Sequence[str]] | None,
) -> tuple[ProvenancePathMapping, ...]:
    """Validate explicit comparison-only old-prefix -> local-prefix mappings.

    Overlapping old prefixes are rejected rather than applying longest-prefix
    or command-line-order precedence.  That keeps every historical path mapped
    by exactly one auditable rule.
    """

    mappings: list[ProvenancePathMapping] = []
    for index, raw_pair in enumerate(raw_mappings or ()):  # argparse uses pairs
        if len(raw_pair) != 2:
            raise ValueError(
                f"provenance path mapping {index} must contain OLD_PREFIX LOCAL_PREFIX"
            )
        old_prefix = _normalized_absolute_path(
            raw_pair[0], label=f"provenance path mapping {index} old prefix"
        )
        local_prefix = _normalized_absolute_path(
            raw_pair[1], label=f"provenance path mapping {index} local prefix"
        ).resolve(strict=False)
        if old_prefix == Path("/") or local_prefix == Path("/"):
            raise ValueError(
                f"provenance path mapping {index} is ambiguous/overbroad: "
                "root prefixes are forbidden"
            )
        candidate = ProvenancePathMapping(
            old_prefix=old_prefix,
            local_prefix=local_prefix,
        )
        for previous in mappings:
            if _path_is_within(
                candidate.old_prefix, previous.old_prefix
            ) or _path_is_within(previous.old_prefix, candidate.old_prefix):
                raise ValueError(
                    "provenance path mappings are ambiguous/multi-match because old "
                    f"prefixes overlap: {previous.old_prefix} and {candidate.old_prefix}"
                )
        mappings.append(candidate)
    return tuple(mappings)


def map_provenance_parent_for_comparison(
    original_parent: str | Path,
    *,
    mappings: Sequence[ProvenancePathMapping],
) -> tuple[Path, ProvenancePathMapping | None]:
    """Map one original sidecar path for comparison without rewriting it."""

    original = _normalized_absolute_path(
        original_parent, label="run_config.args.token_files parent"
    )
    if not mappings:
        return original.resolve(strict=False), None

    matches = [
        mapping
        for mapping in mappings
        if _path_is_within(original, mapping.old_prefix)
    ]
    if not matches:
        raise ValueError(
            f"flagship token source {original} is outside every configured old prefix"
        )
    if len(matches) != 1:
        # parse_provenance_path_mappings rejects overlap; retain a hard runtime
        # guard for callers constructing ProvenancePathMapping objects directly.
        raise ValueError(
            f"flagship token source {original} has multiple path mappings: "
            + ", ".join(str(item.old_prefix) for item in matches)
        )
    mapping = matches[0]
    suffix = original.relative_to(mapping.old_prefix)
    mapped_lexical = mapping.local_prefix / suffix
    mapped = mapped_lexical.resolve(strict=False)
    if not _path_is_within(mapped, mapping.local_prefix):
        raise ValueError(
            f"mapped flagship token source escapes local prefix {mapping.local_prefix}: {mapped}"
        )
    return mapped, mapping


def audit_flagship_pretraining(
    *,
    small_run_config: Path,
    big_run_config: Path,
    full_compact_done: Path,
    graph_root: Path,
    provenance_path_mappings: Sequence[ProvenancePathMapping] = (),
) -> dict[str, Any]:
    """Prove which pretraining corpus the two released flagships consumed."""

    compact_done = load_json(full_compact_done, "full compact BPE completion")
    compact_dir = full_compact_done.parent.resolve()
    if compact_done.get("format") != "compact_int_shards":
        raise ValueError(f"{full_compact_done}: unexpected compact format")

    flagship_configs: dict[str, dict[str, Any]] = {}
    for label, path in (
        ("small_0p55b", small_run_config),
        ("big_1p44b", big_run_config),
    ):
        payload = load_json(path, f"{label} run config")
        args = payload.get("args")
        if not isinstance(args, dict):
            raise ValueError(f"{path}: missing args object")
        patterns = args.get("token_files")
        if not isinstance(patterns, list) or len(patterns) != 1:
            raise ValueError(f"{path}: expected one full-compact token glob")
        original_pattern_parent = Path(str(patterns[0])).expanduser().parent
        pattern_parent, applied_mapping = map_provenance_parent_for_comparison(
            original_pattern_parent,
            mappings=provenance_path_mappings,
        )
        if pattern_parent != compact_dir:
            raise ValueError(
                f"{path}: flagship token source {pattern_parent} is not {compact_dir}"
            )
        if args.get("dataset_mode") != "online-pad":
            raise ValueError(f"{path}: unexpected flagship dataset mode")
        flagship_configs[label] = {
            "run_config": str(path.resolve()),
            "token_files": list(patterns),
            "token_file_parent_original": str(original_pattern_parent),
            "token_file_parent_compared": str(pattern_parent),
            "applied_provenance_path_mapping": (
                applied_mapping.audit_record() if applied_mapping is not None else None
            ),
            "dataset_mode": args["dataset_mode"],
            "compact_index_dir": args.get("compact_index_dir"),
            "val_fraction": args.get("val_fraction"),
            "split_strategy": args.get("split_strategy"),
        }

    upstream_split_counts: dict[str, dict[str, int]] = {}
    totals = Counter()
    evidence_files: list[str] = []
    for split in ("train", "valid", "test"):
        path = require_file(
            graph_root / "peptide" / split / "part-input017-00000.done.json",
            f"PepBenchmark full-graph {split} completion",
        )
        payload = load_json(path, f"PepBenchmark full-graph {split} completion")
        counts = payload.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"{path}: missing counts")
        seen = int(counts.get("seen", 0))
        encoded = int(counts.get("ok", 0))
        failed = int(counts.get("failed", 0))
        if seen != encoded + failed:
            raise ValueError(f"{path}: seen != encoded + failed")
        if payload.get("sources") != {"PepBenchmark": encoded}:
            raise ValueError(f"{path}: source accounting is not PepBenchmark-only")
        if "shards_normv2_all" not in str(payload.get("input") or ""):
            raise ValueError(f"{path}: not built from the full all-record shards")
        upstream_split_counts[split] = {
            "source_records_seen": seen,
            "graph_records_encoded": encoded,
            "graph_records_failed": failed,
        }
        totals.update(upstream_split_counts[split])
        evidence_files.append(str(path))

    expected = {
        "source_records_seen": EXPECTED_SOURCE_RECORDS,
        "graph_records_encoded": EXPECTED_FULL_COMPACT_ENCODED,
        "graph_records_failed": EXPECTED_FULL_COMPACT_FAILED,
    }
    if dict(totals) != expected:
        raise ValueError(f"PepBenchmark full-compact provenance changed: {dict(totals)}")

    return {
        "classification": "pretraining-exposed downstream benchmark",
        "sft_supervision_seen_during_pretraining": False,
        "unseen_generalization_claim_allowed": False,
        "flagship_input_selection": "full compact corpus",
        "historical_sampled_pepbench_records": HISTORICAL_SAMPLED_PEPBENCH_RECORDS,
        "historical_sampled_corpus_used_by_flagships": False,
        "provenance_path_semantic_comparison": {
            "mode": (
                "explicit_old_prefix_to_local_prefix"
                if provenance_path_mappings
                else "strict_path_identity"
            ),
            "comparison_only": True,
            "sidecar_content_rewritten": False,
            "mappings": [
                mapping.audit_record() for mapping in provenance_path_mappings
            ],
        },
        "full_compact": {
            "completion": str(full_compact_done.resolve()),
            "records_all_domains": int(compact_done["records"]),
            "tokens_all_domains": int(compact_done["tokens"]),
            "pepbenchmark_source_records_seen": totals["source_records_seen"],
            "pepbenchmark_graph_records_encoded": totals["graph_records_encoded"],
            "pepbenchmark_graph_records_failed": totals["graph_records_failed"],
            "upstream_shard_split_counts": upstream_split_counts,
            "graph_completion_evidence": evidence_files,
        },
        "flagships": flagship_configs,
        "exposure_interpretation": {
            "all_raw_source_records_entered_the_full_graph_pipeline": True,
            "eligible_benchmark_targets": EXPECTED_ELIGIBLE_RECORDS,
            "expected_unencodable_source_records": EXPECTED_EXCLUDED_RECORDS,
            "aggregate_artifacts_prove_successful_graph_encoding_for": EXPECTED_FULL_COMPACT_ENCODED,
            "aggregate_artifacts_record_graph_failures_for": EXPECTED_FULL_COMPACT_FAILED,
            "per_record_gradient_exposure": (
                "not asserted from aggregate completion files; the flagship also held "
                "out a stride validation fraction from the compact stream"
            ),
            "conservative_novelty_reference": (
                "all 71137 PepBenchmark source sequences are retained in the source "
                "reference, with 70117 eligible graph-encoded targets and 1020 "
                "explicit expected-unencodable exclusions"
            ),
        },
    }


def reconcile_source_scan_with_full_graph(
    scan: SourceScan, exposure: Mapping[str, Any]
) -> dict[str, int | bool]:
    """Require source eligibility counts to match flagship graph accounting."""

    full_compact = exposure.get("full_compact")
    if not isinstance(full_compact, Mapping):
        raise ValueError("pretraining exposure is missing full_compact accounting")
    graph_seen = int(full_compact.get("pepbenchmark_source_records_seen", -1))
    graph_encoded = int(full_compact.get("pepbenchmark_graph_records_encoded", -1))
    graph_failed = int(full_compact.get("pepbenchmark_graph_records_failed", -1))
    observed = {
        "raw_source_records": scan.raw_records,
        "eligible_canonical_records": len(scan.eligible_records),
        "excluded_expected_unencodable_records": len(scan.excluded_records),
    }
    expected = {
        "raw_source_records": graph_seen,
        "eligible_canonical_records": graph_encoded,
        "excluded_expected_unencodable_records": graph_failed,
    }
    if observed != expected:
        raise ValueError(
            "PepBenchmark source eligibility does not reconcile with full-graph "
            f"accounting: observed={observed}, graph={expected}"
        )
    return {**observed, "aggregate_counts_reconciled": True}


def quantile(values: Sequence[int], q: float) -> float:
    ordered = sorted(int(value) for value in values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (rank - low) * (ordered[high] - ordered[low]))


def summarize(values: Iterable[int]) -> dict[str, Any]:
    items = [int(value) for value in values]
    if not items:
        raise ValueError("cannot summarize an empty collection")
    return {
        "count": len(items),
        "min": min(items),
        "mean": sum(items) / len(items),
        "p50": quantile(items, 0.50),
        "p95": quantile(items, 0.95),
        "p99": quantile(items, 0.99),
        "max": max(items),
    }


def json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def make_sft_record(
    row: SourceIndexRecord,
    *,
    split: str,
    condition_text_tokens: int,
    metrics: Mapping[str, int],
    target_bpe: Sequence[int],
) -> dict[str, Any]:
    """Build the transport record without adding words to the condition."""

    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split!r}")
    if int(metrics["full_context_tokens"]) > CONTEXT_BUDGET:
        raise ValueError(
            f"{row.record_id}: full context {metrics['full_context_tokens']} exceeds "
            f"{CONTEXT_BUDGET}"
        )
    return {
        "id": row.record_id,
        "task": TASK,
        "target_type": "peptide",
        "condition": {
            "type": "text",
            "text": str(row.length),
            "molecule_bpe_ids": [],
        },
        "target_bpe_ids": [int(value) for value in target_bpe],
        "meta": {
            "benchmark": "PepBench-Gen reconstructed length-conditioned generation",
            "dataset_id": DATASET_ID,
            "split": split,
            "source_id": row.source_id,
            "target_length": row.length,
            "condition_semantics": "bare decimal target length",
            "text_prefix_tokens": int(condition_text_tokens),
            "target_fddfs_tokens": int(metrics["fddfs_tokens"]),
            "target_bpe_tokens": len(target_bpe),
            "full_context_tokens": int(metrics["full_context_tokens"]),
            "target_truncated": False,
            "pretraining_exposure": (
                "eligible PepBenchmark target; eligible count reconciles with the "
                "flagship full-compact graph-encoded aggregate"
            ),
            "pretraining_gradient_exposure": (
                "not resolved per record by aggregate flagship artifacts"
            ),
            "sft_supervision_seen_during_pretraining": False,
        },
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--full-compact-done", type=Path, default=DEFAULT_FULL_COMPACT_DONE
    )
    parser.add_argument(
        "--small-flagship-run-config", type=Path, default=DEFAULT_FLAGSHIP_SMALL_CONFIG
    )
    parser.add_argument(
        "--big-flagship-run-config", type=Path, default=DEFAULT_FLAGSHIP_BIG_CONFIG
    )
    parser.add_argument(
        "--provenance-path-map",
        action="append",
        nargs=2,
        default=[],
        metavar=("OLD_PREFIX", "LOCAL_PREFIX"),
        help=(
            "comparison-only mapping for absolute paths preserved inside flagship "
            "run_config sidecars; repeat for disjoint old prefixes"
        ),
    )
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    if FORMAL_TASK_DATASET_SCOPES.get(TASK) != (DATASET_ID,):
        raise RuntimeError("PepBench-Gen task registry and materializer disagree")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    source = require_file(args.source, "canonical PepBenchmark source")
    guide = require_file(args.guide, "FDDFS guide")
    vocab = require_file(args.vocab, "FDDFS vocabulary")
    merges = require_file(args.merges, "frozen BPE merges")
    tokenizer_path = require_file(args.tokenizer, "MPNet tokenizer")
    provenance_path_mappings = parse_provenance_path_mappings(
        args.provenance_path_map
    )

    exposure = audit_flagship_pretraining(
        small_run_config=args.small_flagship_run_config,
        big_run_config=args.big_flagship_run_config,
        full_compact_done=args.full_compact_done,
        graph_root=args.graph_root,
        provenance_path_mappings=provenance_path_mappings,
    )
    source_scan = scan_source_index(source)
    source_index = source_scan.eligible_records
    full_graph_reconciliation = reconcile_source_scan_with_full_graph(
        source_scan, exposure
    )
    assignments, length_split_counts = exact_length_stratified_split(
        source_index,
        seed=args.split_seed,
        test_fraction=args.test_fraction,
        validation_fraction=args.inner_validation_fraction,
    )

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    staging.mkdir(parents=True)

    exclusion_reference_path = staging / "excluded_source_references.jsonl"
    with exclusion_reference_path.open("x", encoding="utf-8") as handle:
        for record in source_scan.excluded_records:
            handle.write(json_line(record.reference()))
    exclusion_records = [
        record.reference() for record in source_scan.excluded_records
    ]
    exclusion_audit_path = staging / "source_exclusion_audit.json"
    atomic_json(
        exclusion_audit_path,
        {
            "complete": True,
            "policy": (
                "retain source sequence verbatim; exclude only non-canonical 20-AA "
                "or out-of-range [5,80] targets before splitting; never repair"
            ),
            "raw_source_records": source_scan.raw_records,
            "eligible_canonical_records": len(source_scan.eligible_records),
            "excluded_expected_unencodable_records": len(
                source_scan.excluded_records
            ),
            "exclusion_reason_counts": source_scan.exclusion_reason_counts,
            "exclusion_reason_definitions": EXCLUSION_REASON_DEFINITIONS,
            "full_graph_reconciliation": full_graph_reconciliation,
            "records": exclusion_records,
        },
    )

    from tokenizers import Tokenizer
    from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    condition_tokens = {
        length: int(sum(tokenizer.encode(str(length)).attention_mask))
        for length in sorted(length_split_counts)
    }
    if any(value <= 0 for value in condition_tokens.values()):
        raise ValueError(f"bare length encoded to zero text tokens: {condition_tokens}")
    codec = FrozenAxisCodec(guide, vocab, merges)

    output_paths = {split: staging / f"{split}.jsonl" for split in SPLITS}
    reference_paths = {
        split: staging / f"{split}_references.jsonl" for split in SPLITS
    }
    exposure_reference_path = (
        staging / "pepbenchmark_pretraining_source_references.jsonl"
    )
    handles = {
        split: output_paths[split].open("x", encoding="utf-8") for split in SPLITS
    }
    reference_handles = {
        split: reference_paths[split].open("x", encoding="utf-8") for split in SPLITS
    }
    with exposure_reference_path.open("x", encoding="utf-8") as exposure_handle:
        source_reference_records: list[
            tuple[int, SourceIndexRecord | ExcludedSourceRecord]
        ] = [
            (record.line_number, record) for record in source_scan.eligible_records
        ] + [
            (record.line_number, record) for record in source_scan.excluded_records
        ]
        for _line_number, record in sorted(
            source_reference_records, key=lambda item: item[0]
        ):
            if isinstance(record, SourceIndexRecord):
                exposure_handle.write(
                    json_line(
                        {
                            "id": record.record_id,
                            "source_id": record.source_id,
                            "source_line_number": record.line_number,
                            "sequence": record.sequence,
                            "source_raw_sequence": record.source_raw_sequence,
                            "target_length": record.length,
                            "split": assignments[record.record_id],
                            "eligibility_status": "eligible_canonical",
                            "entered_flagship_full_graph_pipeline": True,
                            "included_in_full_compact_graph_aggregate": True,
                            "exposure_scope": (
                                "flagship full-compact graph-encoded aggregate"
                            ),
                            "conservative_pretraining_source_exposed": True,
                        }
                    )
                )
            else:
                exposure_handle.write(
                    json_line(
                        {
                            **record.reference(),
                            "split": None,
                            "entered_flagship_full_graph_pipeline": True,
                            "included_in_full_compact_graph_aggregate": False,
                            "exposure_scope": (
                                "flagship full-graph source candidate; excluded before "
                                "downstream split and absent from graph-encoded aggregate"
                            ),
                            "conservative_pretraining_source_exposed": True,
                        }
                    )
                )
    counts = Counter()
    residue_lengths: list[int] = []
    target_bpe_lengths: list[int] = []
    context_lengths: list[int] = []
    context_overflows: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(source_index, start=1):
            split = assignments[row.record_id]
            condition_text = str(row.length)
            metrics, _fddfs, target_bpe = codec.encode_sequence(
                "peptide",
                row.sequence,
                row.record_id,
                text_condition_tokens=condition_tokens[row.length],
            )
            full_context = int(metrics["full_context_tokens"])
            if full_context > CONTEXT_BUDGET:
                context_overflows.append(
                    {
                        "id": row.record_id,
                        "source_id": row.source_id,
                        "split": split,
                        "residue_length": row.length,
                        "condition": condition_text,
                        "condition_text_tokens": condition_tokens[row.length],
                        "fddfs_tokens": int(metrics["fddfs_tokens"]),
                        "target_bpe_tokens": len(target_bpe),
                        "target_frame_tokens": TARGET_FRAME_TOKENS,
                        "full_context_tokens": full_context,
                        "context_budget": CONTEXT_BUDGET,
                    }
                )
            else:
                record = make_sft_record(
                    row,
                    split=split,
                    condition_text_tokens=condition_tokens[row.length],
                    metrics=metrics,
                    target_bpe=target_bpe,
                )
                handles[split].write(json_line(record))
                counts[split] += 1
                residue_lengths.append(row.length)
                target_bpe_lengths.append(len(target_bpe))
                context_lengths.append(full_context)

            reference = {
                "id": row.record_id,
                "source_id": row.source_id,
                "split": split,
                "sequence": row.sequence,
                "target_length": row.length,
            }
            reference_handles[split].write(json_line(reference))
            if index % args.progress_every == 0 or index == len(source_index):
                print(
                    json.dumps(
                        {
                            "encoded": index,
                            "total": len(source_index),
                            "context_overflows": len(context_overflows),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in (*handles.values(), *reference_handles.values()):
            handle.close()

    if context_overflows:
        audit_path = staging / "context_overflow_audit.json"
        atomic_json(
            audit_path,
            {
                "status": "failed",
                "reason": "one or more untruncated records exceed the Axis context budget",
                "context_budget": CONTEXT_BUDGET,
                "target_frame_tokens": TARGET_FRAME_TOKENS,
                "overflow_records": context_overflows,
            },
        )
        raise RuntimeError(
            f"{len(context_overflows)} PepBench-Gen records exceed context "
            f"{CONTEXT_BUDGET}; audit={audit_path}"
        )

    expected_counts = Counter(assignments.values())
    if counts != expected_counts:
        raise RuntimeError(
            f"materialized split counts differ from assignments: {counts} != {expected_counts}"
        )

    file_summaries = []
    for split in SPLITS:
        path = output_paths[split]
        file_summaries.append(
            {
                "dataset": DATASET_ID,
                "split": split,
                "relative_source": path.name,
                "records": int(counts[split]),
                "output_size": path.stat().st_size,
            }
        )
    manifest = {
        "complete": True,
        "task": TASK,
        "dataset_id": DATASET_ID,
        "records": sum(int(item["records"]) for item in file_summaries),
        "files": len(file_summaries),
        "file_summaries": file_summaries,
        "protocol": {
            "name": PROTOCOL,
            "benchmark_status": "reconstructed pretraining-exposed downstream benchmark",
            "serializer": "fddfs",
            "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
            "model_token_offset": 3,
            "condition": "bare decimal target length only",
            "target": "strict 20-AA peptide sequence of length 5 through 80",
            "source_eligibility": (
                "classify exact normalized sequence without case/whitespace repair; "
                "audit non-canonical/out-of-range source rows before splitting"
            ),
            "split": {
                "method": (
                    "exact per-length deterministic rank: freeze test first, then "
                    "inner-validation from remaining training candidates"
                ),
                "ranking_digest": "sha256 used only for deterministic assignment, not acceptance",
                "seed": args.split_seed,
                "test_fraction": args.test_fraction,
                "inner_validation_fraction_of_full_pool": args.inner_validation_fraction,
                "official_or_formal_test_used_for_early_stopping": False,
                "inner_validation_used_for_checkpoint_selection": True,
            },
            "truncation": "none; any context over 512 is a hard failure",
        },
        "source": str(source),
        "source_records": source_scan.raw_records,
        "eligible_records": len(source_scan.eligible_records),
        "excluded_records": len(source_scan.excluded_records),
        "source_eligibility": {
            "raw_source_records": source_scan.raw_records,
            "eligible_canonical_records": len(source_scan.eligible_records),
            "excluded_expected_unencodable_records": len(
                source_scan.excluded_records
            ),
            "exclusion_reason_counts": source_scan.exclusion_reason_counts,
            "exclusion_reason_definitions": EXCLUSION_REASON_DEFINITIONS,
            "exclusion_reference_output": exclusion_reference_path.name,
            "exclusion_audit_output": exclusion_audit_path.name,
            "full_graph_reconciliation": full_graph_reconciliation,
            "excluded_source_records": exclusion_records,
        },
        "sft_split_counts": {split: int(counts[split]) for split in SPLITS},
        "per_length_split_counts": {
            str(length): counts_by_split
            for length, counts_by_split in length_split_counts.items()
        },
        "reference_outputs": {
            split: reference_paths[split].name for split in SPLITS
        },
        "excluded_source_reference_output": exclusion_reference_path.name,
        "source_exclusion_audit_output": exclusion_audit_path.name,
        "pepbenchmark_pretraining_source_reference_output": exposure_reference_path.name,
        "pretraining_exposure": exposure,
        "novelty_reporting": {
            "relative_to_sft_train": "use train_references.jsonl",
            "relative_to_pepbenchmark_pretraining_source": (
                "use pepbenchmark_pretraining_source_references.jsonl as a conservative "
                "source-level exact/near-novelty reference; eligibility_status "
                "distinguishes the 70117 benchmark targets from 1020 expected "
                "unencodable source exclusions"
            ),
            "relative_to_complete_actual_flagship_pretraining_corpus": (
                "not computed by this materializer; the complete reference must also "
                "include the flagship's other peptide and protein sources and resolve "
                "compact train-versus-stride-validation membership before any full-corpus "
                "novelty claim"
            ),
            "pepbenchmark_subset_must_not_be_reported_as_complete_flagship_reference": True,
            "unseen_claim": "prohibited",
        },
        "condition_text_tokens_by_length": {
            str(length): count for length, count in condition_tokens.items()
        },
        "residue_length": summarize(residue_lengths),
        "target_bpe_tokens": summarize(target_bpe_lengths),
        "full_context_tokens": summarize(context_lengths),
        "context_budget": CONTEXT_BUDGET,
        "context_overflow_records": 0,
        "all_records_fit_without_truncation": True,
        "codec": codec.codebook,
    }
    atomic_json(staging / "manifest.json", manifest)
    index = build_axis_b_index(staging)
    expected_index_counts = {
        "train": counts["train"],
        "eval": counts["validation"],
        "test": counts["test"],
    }
    for split, expected in expected_index_counts.items():
        actual = int(index.task_counts(split).get(TASK, 0))
        if actual != expected:
            raise RuntimeError(
                f"indexed {split} count mismatch: expected={expected} actual={actual}"
            )
    index_manifest_path = staging / "axis_b_index_v1" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["data_root"] = str(output_root)
    atomic_json(index_manifest_path, index_manifest)
    os.replace(staging, output_root)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(output_root),
                "counts": {split: int(counts[split]) for split in SPLITS},
                "source_records": source_scan.raw_records,
                "eligible_records": len(source_scan.eligible_records),
                "excluded_records": len(source_scan.excluded_records),
                "pretraining_exposure": "full compact source; not unseen",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
