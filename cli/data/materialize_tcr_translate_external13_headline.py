#!/usr/bin/env python3
"""Materialize the TCR-TRANSLATE external-13 headline evaluation prompts.

The released external benchmark table contains fourteen pMHC groups.  The
paper reserves ``RVRAYTYSK_A*03:01`` for its deep-simulation experiment, so
the sparse headline benchmark contains the other thirteen groups.  This
adapter preserves every released reference in its original order and keeps
the external track separate from the released top-20 validation track.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Protocol, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from cli.data.materialize_tcr_translate_released_top20_auxiliary import (  # noqa: E402
    load_hla_pseudosequences,
)
from cli.data.materialize_tcr_translate_sft import (  # noqa: E402
    AA_ALPHABET,
    DATASET_ID,
    DEFAULT_GUIDE,
    DEFAULT_INPUT_ROOT,
    DEFAULT_MERGES,
    DEFAULT_VOCAB,
    TASK,
    TcrPairEncoder,
    require_file,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402


ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / '')
)
DEFAULT_HLA_MAP = DEFAULT_INPUT_ROOT.parent / "refs/hla_pseudo_seqs.csv"
DEFAULT_BENCHMARK_TABLE = DEFAULT_INPUT_ROOT / "benchmark_data_w_preds.csv"
DEFAULT_IMMPREP_SOURCE = DEFAULT_INPUT_ROOT / "IMMPREP_test_set.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data/processed/axis_b_biological_sft/tcr_translate_released_v1"
    / "external13_headline_eval_groups.jsonl"
)

TRACK = "external13_sparse_headline"
PROTOCOL = "axis-tcr-translate-external13-headline-groups-v1-20260830"
EXCLUDED_DEEP_SIMULATION_PMHC = "RVRAYTYSK_A*03:01"
EXPECTED_RELEASED_GROUPS = 14
EXPECTED_HEADLINE_GROUPS = 13
EXPECTED_RELEASED_REFERENCES = 1_315
EXPECTED_EXCLUDED_REFERENCES = 895
EXPECTED_HEADLINE_REFERENCES = 420
HEADLINE_K = 1_000
HEADLINE_NUM_BEAMS = 1_000
HEADLINE_MAX_NEW_TOKENS = 25
HEADLINE_NO_REPEAT_NGRAM_SIZE = 5
CONTEXT_BUDGET = 512


class AxisCodec(Protocol):
    codebook: Mapping[str, Any]

    def encode_sequence(self, *args: Any, **kwargs: Any) -> Any: ...

    def bpe_ids(self, tokens: Sequence[str], label: str) -> list[int]: ...


def require_worker() -> None:
    require_compute_host(
        "TCR-TRANSLATE external-13 materialization", socket.gethostname()
    )


def _reference_list(value: str, *, label: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{label}: reference_translations is not a Python list") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{label}: reference_translations must be a non-empty list")
    references: list[str] = []
    for index, item in enumerate(parsed):
        if (
            not isinstance(item, str)
            or not item
            or set(item).difference(AA_ALPHABET)
        ):
            raise ValueError(f"{label}: invalid CDR3beta reference at index {index}")
        references.append(item)
    return references


def load_external13_rows(
    path: Path,
    *,
    expected_released_groups: int = EXPECTED_RELEASED_GROUPS,
    expected_headline_groups: int = EXPECTED_HEADLINE_GROUPS,
    expected_released_references: int = EXPECTED_RELEASED_REFERENCES,
    expected_excluded_references: int = EXPECTED_EXCLUDED_REFERENCES,
    expected_headline_references: int = EXPECTED_HEADLINE_REFERENCES,
) -> list[dict[str, Any]]:
    path = require_file(path, "released external TCR benchmark table")
    rows: list[dict[str, Any]] = []
    released_references = 0
    excluded_references: int | None = None
    seen_pmhc: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Epitope",
            "Allele",
            "reference_translations",
            "Dataset",
            "Year",
            "pMHC",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            epitope = str(raw["Epitope"]).strip()
            allele = str(raw["Allele"]).strip()
            pmhc = str(raw["pMHC"]).strip()
            label = f"{path}:{line_number}"
            if not epitope or set(epitope).difference(AA_ALPHABET):
                raise ValueError(f"{label}: non-canonical epitope")
            if not allele.startswith("HLA-"):
                raise ValueError(f"{label}: allele must use the released HLA- prefix")
            expected_pmhc = f"{epitope}_{allele.removeprefix('HLA-')}"
            if pmhc != expected_pmhc:
                raise ValueError(
                    f"{label}: pMHC key {pmhc!r} != released fields {expected_pmhc!r}"
                )
            if pmhc in seen_pmhc:
                raise ValueError(f"{label}: duplicate external pMHC group {pmhc}")
            seen_pmhc.add(pmhc)
            references = _reference_list(
                str(raw["reference_translations"]), label=label
            )
            released_references += len(references)
            if pmhc == EXCLUDED_DEEP_SIMULATION_PMHC:
                excluded_references = len(references)
                continue
            rows.append(
                {
                    "source_line_number": line_number,
                    "epitope": epitope,
                    "allele": allele,
                    "pmhc": pmhc,
                    "dataset": str(raw["Dataset"]).strip(),
                    "year": str(raw["Year"]).strip(),
                    "closest_training": str(raw.get("closest_training", "")).strip(),
                    "references": references,
                }
            )
    if len(seen_pmhc) != expected_released_groups:
        raise ValueError(
            f"external table group count changed: expected={expected_released_groups} "
            f"actual={len(seen_pmhc)}"
        )
    if len(rows) != expected_headline_groups:
        raise ValueError(
            f"external headline group count changed: expected={expected_headline_groups} "
            f"actual={len(rows)}"
        )
    if released_references != expected_released_references:
        raise ValueError(
            "external released reference count changed: "
            f"expected={expected_released_references} actual={released_references}"
        )
    if excluded_references != expected_excluded_references:
        raise ValueError(
            "deep-simulation reference count changed: "
            f"expected={expected_excluded_references} actual={excluded_references}"
        )
    headline_references = sum(len(row["references"]) for row in rows)
    if headline_references != expected_headline_references:
        raise ValueError(
            "external-13 reference count changed: "
            f"expected={expected_headline_references} actual={headline_references}"
        )
    return rows


def materialize_external13_groups(
    *,
    benchmark_table: Path,
    immprep_source: Path,
    hla_map_path: Path,
    output_path: Path,
    codec: AxisCodec,
    expected_released_groups: int = EXPECTED_RELEASED_GROUPS,
    expected_headline_groups: int = EXPECTED_HEADLINE_GROUPS,
    expected_released_references: int = EXPECTED_RELEASED_REFERENCES,
    expected_excluded_references: int = EXPECTED_EXCLUDED_REFERENCES,
    expected_headline_references: int = EXPECTED_HEADLINE_REFERENCES,
) -> dict[str, Any]:
    rows = load_external13_rows(
        benchmark_table,
        expected_released_groups=expected_released_groups,
        expected_headline_groups=expected_headline_groups,
        expected_released_references=expected_released_references,
        expected_excluded_references=expected_excluded_references,
        expected_headline_references=expected_headline_references,
    )
    # This is a provenance source for the IMMREP-derived subset, not the
    # complete mixed-source sparse benchmark table.
    immprep_source = require_file(immprep_source, "released IMMREP source table")
    hla_map_path = require_file(hla_map_path, "released HLA pseudosequence map")
    hla_map = load_hla_pseudosequences(hla_map_path)
    output_path = output_path.expanduser().resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite external-13 output or manifest: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    encoder = TcrPairEncoder(codec)
    condition_lengths: list[int] = []
    reference_counts: list[int] = []
    with temporary.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            allele = row["allele"]
            hla = hla_map.get(allele)
            if hla is None:
                raise ValueError(f"external-13 allele absent from HLA map: {allele}")
            prompt_id = f"tcr-translate:external13:{index:02d}"
            condition_bpe = encoder.encode_condition(
                row["epitope"], hla, label=prompt_id
            )
            prompt_context = len(condition_bpe) + 2
            if prompt_context + HEADLINE_MAX_NEW_TOKENS > CONTEXT_BUDGET:
                raise ValueError(
                    f"{prompt_id}: prompt {prompt_context} plus max_new_tokens "
                    f"{HEADLINE_MAX_NEW_TOKENS} exceeds {CONTEXT_BUDGET}"
                )
            payload = {
                "id": prompt_id,
                "task": TASK,
                "dataset_id": DATASET_ID,
                "target_type": "peptide",
                "track": TRACK,
                "headline": True,
                "condition": {
                    "type": "molecule",
                    "molecule_bpe_ids": condition_bpe,
                },
                "pmhc": {
                    "official_key": row["pmhc"],
                    "epitope": row["epitope"],
                    "allele": allele,
                    "hla_pseudosequence": hla,
                    "source_dataset": row["dataset"],
                    "source_year": row["year"],
                    "closest_training": row["closest_training"],
                    "benchmark_table_line_number": row["source_line_number"],
                },
                # Preserve released order and multiplicity exactly.
                "reference_translations": list(row["references"]),
            }
            handle.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            condition_lengths.append(len(condition_bpe))
            reference_counts.append(len(row["references"]))
    os.replace(temporary, output_path)
    manifest = {
        "complete": True,
        "protocol": PROTOCOL,
        "track": TRACK,
        "headline": True,
        "task": TASK,
        "dataset_id": DATASET_ID,
        "output": output_path.name,
        "released_groups": expected_released_groups,
        "groups": len(rows),
        "references": sum(reference_counts),
        "excluded_deep_simulation_pmhc": EXCLUDED_DEEP_SIMULATION_PMHC,
        "excluded_deep_simulation_references": expected_excluded_references,
        "reference_count_min": min(reference_counts),
        "reference_count_max": max(reference_counts),
        "condition_bpe_min": min(condition_lengths),
        "condition_bpe_max": max(condition_lengths),
        "condition": (
            "FDDFS(epitope) + raw <SEP> + FDDFS(HLA-34), then one frozen-BPE "
            "encoding of the joined stream; no text prompt"
        ),
        "generation_contract": {
            "mode": "deterministic_beam",
            "k": HEADLINE_K,
            "num_beams": HEADLINE_NUM_BEAMS,
            "max_new_tokens": HEADLINE_MAX_NEW_TOKENS,
            "no_repeat_ngram_size": HEADLINE_NO_REPEAT_NGRAM_SIZE,
            "filter_invalid_or_duplicate_outputs": False,
            "replacement_sampling": False,
            "post_hoc_sequence_repair": False,
        },
        "headline_metrics": [
            "exact_reference_match_count",
            "sequence_recovery_ge_0_90_count",
            "giana_4_1_reference_clustered_count",
        ],
        "source_files": {
            "external_benchmark_table": str(benchmark_table.resolve()),
            "immprep_subset_provenance": str(immprep_source.resolve()),
            "hla_map": str(hla_map_path.resolve()),
        },
        "codec": dict(codec.codebook),
    }
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    with manifest_temporary.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(manifest_temporary, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-table", type=Path, default=DEFAULT_BENCHMARK_TABLE
    )
    parser.add_argument("--immprep-source", type=Path, default=DEFAULT_IMMPREP_SOURCE)
    parser.add_argument("--hla-map", type=Path, default=DEFAULT_HLA_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    from pipeline.finetune.frozen_axis_codec import (
        FrozenAxisCodec,
    )

    manifest = materialize_external13_groups(
        benchmark_table=args.benchmark_table,
        immprep_source=args.immprep_source,
        hla_map_path=args.hla_map,
        output_path=args.output,
        codec=FrozenAxisCodec(
            require_file(args.guide, "FDDFS guide"),
            require_file(args.vocab, "FDDFS vocabulary"),
            require_file(args.merges, "frozen BPE merges"),
        ),
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
