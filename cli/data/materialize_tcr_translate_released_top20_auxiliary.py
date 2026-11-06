#!/usr/bin/env python3
"""Build the lossless paired-validation intermediate for TCR top-20 prompts.

Do not score directly against this intermediate. The official CSV loader then
deduplicates references via to_seq2seq_dict. Apply
materialize_tcr_native_eval_references_worker.py using its saved native mapping
before evaluation; training records and generated slots are not deduplicated.

The released evaluator groups ``topk_val_df.csv`` by pMHC and evaluates one
100-candidate translation list against unique cognate CDR3beta references for
that pMHC.  The released parallel validation files are the exact model-input
view (92,602 rows, 20 conditions), so this adapter groups those rows without
deduplicating their targets and attaches the corresponding released allele
label from ``topk_val_df.csv``.

This is deliberately an auxiliary validation track.  The formal headline
test is the separate external-13 sparse benchmark; no result produced by
this module may be labelled as that headline result.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import Counter
import csv
from itertools import zip_longest
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Protocol, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from cli.data.materialize_tcr_translate_sft import (  # noqa: E402
    AA_ALPHABET,
    DATASET_ID,
    DEFAULT_GUIDE,
    DEFAULT_INPUT_ROOT,
    DEFAULT_MERGES,
    DEFAULT_VOCAB,
    RELEASED_VALIDATION_SOURCE_NAME,
    RELEASED_VALIDATION_TARGET_NAME,
    TASK,
    TcrPairEncoder,
    require_file,
)
from pipeline.runtime_host import require_compute_host  # noqa: E402


ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / '')
)
DEFAULT_HLA_MAP = DEFAULT_INPUT_ROOT.parent / "refs/hla_pseudo_seqs.csv"
DEFAULT_TOPK_TABLE = DEFAULT_INPUT_ROOT / "topk_val_df.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data/processed/axis_b_biological_sft/tcr_translate_released_v1"
    / "released_top20_auxiliary_eval_groups.jsonl"
)

TRACK = "released_top20_validation_auxiliary"
PROTOCOL = "axis-tcr-translate-released-top20-auxiliary-groups-v1-20260830"
EXPECTED_ROWS = 92_602
EXPECTED_GROUPS = 20
EXPECTED_DUPLICATE_SOURCE_TARGET_ROWS = 24_186
OFFICIAL_K = 100
OFFICIAL_NUM_BEAMS = 300
OFFICIAL_MAX_NEW_TOKENS = 25
OFFICIAL_NO_REPEAT_NGRAM_SIZE = 5
CONTEXT_BUDGET = 512


class AxisCodec(Protocol):
    codebook: Mapping[str, Any]

    def encode_sequence(self, *args: Any, **kwargs: Any) -> Any: ...

    def bpe_ids(self, tokens: Sequence[str], label: str) -> list[int]: ...


def require_worker() -> None:
    require_compute_host(
        "TCR-TRANSLATE released top-20 evaluation materialization",
        socket.gethostname(),
    )


def load_released_reference_groups(
    source_path: Path,
    target_path: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    expected_groups: int = EXPECTED_GROUPS,
    expected_duplicate_rows: int = EXPECTED_DUPLICATE_SOURCE_TARGET_ROWS,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    source_path = require_file(source_path, "released top-20 source")
    target_path = require_file(target_path, "released top-20 target")
    groups: dict[str, list[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_rows = 0
    rows = 0
    with source_path.open("r", encoding="utf-8") as source_handle, target_path.open(
        "r", encoding="utf-8"
    ) as target_handle:
        for line_number, pair in enumerate(
            zip_longest(source_handle, target_handle), start=1
        ):
            source_line, target_line = pair
            if source_line is None or target_line is None:
                raise ValueError(
                    f"unpaired released validation files at line {line_number}"
                )
            source = source_line.strip()
            target = target_line.strip()
            parts = source.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{source_path}:{line_number}: expected epitope and HLA-34"
                )
            epitope, hla = parts
            if not epitope or set(epitope).difference(AA_ALPHABET):
                raise ValueError(f"{source_path}:{line_number}: non-canonical epitope")
            if len(hla) != 34 or set(hla).difference(AA_ALPHABET):
                raise ValueError(f"{source_path}:{line_number}: invalid HLA pseudosequence")
            if not target or set(target).difference(AA_ALPHABET):
                raise ValueError(f"{target_path}:{line_number}: non-canonical CDR3beta")
            rows += 1
            groups.setdefault(source, []).append(target)
            source_target = (source, target)
            duplicate_rows += int(source_target in seen_pairs)
            seen_pairs.add(source_target)
    if rows != expected_rows:
        raise ValueError(
            f"released validation row count changed: expected={expected_rows} actual={rows}"
        )
    if len(groups) != expected_groups:
        raise ValueError(
            f"released validation pMHC count changed: expected={expected_groups} "
            f"actual={len(groups)}"
        )
    if duplicate_rows != expected_duplicate_rows:
        raise ValueError(
            "released validation duplicate source-target count changed: "
            f"expected={expected_duplicate_rows} actual={duplicate_rows}"
        )
    return groups, {
        "rows": rows,
        "groups": len(groups),
        "duplicate_source_target_rows": duplicate_rows,
    }


def load_hla_pseudosequences(path: Path) -> dict[str, str]:
    path = require_file(path, "released HLA pseudosequence map")
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"allele", "pseudo-sequence"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            allele = str(row["allele"]).strip()
            pseudo = str(row["pseudo-sequence"]).strip()
            if not allele or len(pseudo) != 34 or set(pseudo).difference(AA_ALPHABET):
                raise ValueError(f"{path}:{line_number}: invalid allele/pseudosequence")
            previous = mapping.get(allele)
            if previous is not None and previous != pseudo:
                raise ValueError(f"{path}:{line_number}: conflicting sequence for {allele}")
            mapping[allele] = pseudo
    return mapping


def map_released_conditions_to_alleles(
    topk_table: Path,
    hla_map: Mapping[str, str],
    source_keys: Sequence[str],
) -> dict[str, str]:
    """Recover the official ``epitope_allele`` labels for all 20 conditions."""

    topk_table = require_file(topk_table, "released grouped validation table")
    source_set = set(source_keys)
    matches: dict[str, set[str]] = {source: set() for source in source_keys}
    with topk_table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Epitope", "Allele"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{topk_table}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            epitope = str(row["Epitope"]).strip()
            allele = str(row["Allele"]).strip()
            if allele not in hla_map:
                raise ValueError(
                    f"{topk_table}:{line_number}: allele absent from released HLA map: "
                    f"{allele!r}"
                )
            source = f"{epitope} {hla_map[allele]}"
            if source not in source_set:
                raise ValueError(
                    f"{topk_table}:{line_number}: pMHC does not map to a released "
                    f"parallel condition: {epitope}_{allele}"
                )
            matches[source].add(allele)
    ambiguous = {
        source: sorted(alleles)
        for source, alleles in matches.items()
        if len(alleles) != 1
    }
    if ambiguous:
        raise ValueError(
            "released evaluator grouping cannot be represented uniquely by Axis "
            f"epitope+pseudosequence conditions: {ambiguous}"
        )
    return {source: next(iter(alleles)) for source, alleles in matches.items()}


def materialize_auxiliary_groups(
    *,
    source_path: Path,
    target_path: Path,
    topk_table: Path,
    hla_map_path: Path,
    output_path: Path,
    codec: AxisCodec,
    expected_rows: int = EXPECTED_ROWS,
    expected_groups: int = EXPECTED_GROUPS,
    expected_duplicate_rows: int = EXPECTED_DUPLICATE_SOURCE_TARGET_ROWS,
) -> dict[str, Any]:
    groups, audit = load_released_reference_groups(
        source_path,
        target_path,
        expected_rows=expected_rows,
        expected_groups=expected_groups,
        expected_duplicate_rows=expected_duplicate_rows,
    )
    hla_map = load_hla_pseudosequences(hla_map_path)
    alleles = map_released_conditions_to_alleles(
        topk_table, hla_map, list(groups)
    )
    output_path = output_path.expanduser().resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite evaluation output or manifest: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    encoder = TcrPairEncoder(codec)
    reference_counts: list[int] = []
    condition_lengths: list[int] = []
    with temporary.open("x", encoding="utf-8") as handle:
        for index, (source, references) in enumerate(groups.items()):
            epitope, hla = source.split()
            allele = alleles[source]
            prompt_id = f"tcr-translate:released-top20:{index:02d}"
            condition_bpe = encoder.encode_condition(
                epitope,
                hla,
                label=prompt_id,
            )
            prompt_context = len(condition_bpe) + 2  # target type + BOS
            if prompt_context + OFFICIAL_MAX_NEW_TOKENS > CONTEXT_BUDGET:
                raise ValueError(
                    f"{prompt_id}: prompt {prompt_context} plus official max_new_tokens "
                    f"{OFFICIAL_MAX_NEW_TOKENS} exceeds {CONTEXT_BUDGET}"
                )
            payload = {
                "id": prompt_id,
                "task": TASK,
                "dataset_id": DATASET_ID,
                "target_type": "peptide",
                "condition": {
                    "type": "molecule",
                    "molecule_bpe_ids": condition_bpe,
                },
                "pmhc": {
                    "official_key": f"{epitope}_{allele}",
                    "epitope": epitope,
                    "allele": allele,
                    "hla_pseudosequence": hla,
                    "released_source": source,
                },
                # Keep list order and duplicates exactly as released.
                "reference_translations": list(references),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            reference_counts.append(len(references))
            condition_lengths.append(len(condition_bpe))
    os.replace(temporary, output_path)
    if sum(reference_counts) != expected_rows:
        raise RuntimeError("grouped reference lists no longer preserve every released row")
    manifest = {
        "complete": True,
        "protocol": PROTOCOL,
        "track": TRACK,
        "headline": False,
        "task": TASK,
        "dataset_id": DATASET_ID,
        "output": output_path.name,
        "groups": len(groups),
        "references": sum(reference_counts),
        "reference_audit": audit,
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
            "k": OFFICIAL_K,
            "num_beams": OFFICIAL_NUM_BEAMS,
            "max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
            "no_repeat_ngram_size": OFFICIAL_NO_REPEAT_NGRAM_SIZE,
            "greedy_translation_required": True,
            "filter_invalid_or_duplicate_outputs": False,
            "replacement_sampling": False,
        },
        "source_files": {
            "parallel_source": str(source_path.resolve()),
            "parallel_target": str(target_path.resolve()),
            "topk_table": str(topk_table.resolve()),
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
        "--source",
        type=Path,
        default=DEFAULT_INPUT_ROOT / RELEASED_VALIDATION_SOURCE_NAME,
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_INPUT_ROOT / RELEASED_VALIDATION_TARGET_NAME,
    )
    parser.add_argument("--topk-table", type=Path, default=DEFAULT_TOPK_TABLE)
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

    manifest = materialize_auxiliary_groups(
        source_path=args.source,
        target_path=args.target,
        topk_table=args.topk_table,
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
