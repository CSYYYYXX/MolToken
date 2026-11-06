#!/usr/bin/env python3
"""Materialize one G6-filtered RfamGen family through the frozen Axis codec.

The paper-protocol reconstruction has already selected the family length band
and applied Infernal/G6 filtering.  This worker preserves the original G6
sequence as the evaluation reference, encodes the SFT target with the frozen
FDDFS/BPE codebook, and reproduces the repository's two-step 70/15/15 split.

Only RF01960 may use the explicitly approved context cap.  For an over-budget
RF01960 target, RNA residues are removed one at a time from the 3' end and the
heavy graph is rebuilt until the first sequence that fits.  FDDFS or BPE token
streams are never sliced directly.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / '')
)
DEFAULT_G6_ROOT = (
    ROOT / "data/processed/rfamgen_figure2_reconstruction_20260830/g6_pools"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data/processed/rfamgen_figure2_reconstruction_20260830/axis_context_pools"
)
DEFAULT_GUIDE = ROOT / "data/fddfs_guide_recount_v1/guide_counts.json"
DEFAULT_VOCAB = ROOT / "data/fddfs_guide_recount_v1/vocab.json"
DEFAULT_MERGES = (
    ROOT
    / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1"
    / "codebook_fddfs_prefix70_mf80000_v1/merges.csv"
)
DEFAULT_TOKENIZER = (
    ROOT / "data/pretrained_encoders/all-mpnet-base-v2/tokenizer.json"
)

PROTOCOL = "axis-rfamgen-figure2-post-g6-axis-v1-20260830"
G6_PROTOCOL = "axis-rfamgen-figure2-paper-g6-reconstruction-v1-20260830"
G6_AGGREGATE_PROTOCOL = "axis-rfamgen-figure2-paper-g6-formal17-v1-20260830"
CONTEXT_BUDGET = 512
TARGET_FRAME_TOKENS = 3
CAP_FAMILY = "RF01960"
FAMILY_PATTERN = re.compile(r"RF\d{5}")
RNA_ALPHABET = frozenset("ACGU")


class AxisCodec(Protocol):
    def encode_sequence(
        self,
        target_type: str,
        sequence: str,
        label: str,
        *,
        text_condition_tokens: int = 0,
        molecule_condition_bpe_tokens: int = 0,
    ) -> tuple[dict[str, int], list[str], list[int]]: ...


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    original_header: str
    sequence: str
    input_index: int


@dataclass(frozen=True)
class EncodedRecord:
    source: FastaRecord
    training_sequence: str
    target_bpe_ids: tuple[int, ...]
    original_metrics: Mapping[str, int]
    final_metrics: Mapping[str, int]
    deleted_3prime_suffix: str

    @property
    def target_truncated(self) -> bool:
        return bool(self.deleted_3prime_suffix)


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {resolved}")
    return resolved


def load_json(path: Path) -> Any:
    require_file(path, "JSON")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def iter_fasta(path: Path) -> Iterator[FastaRecord]:
    with require_file(path, "G6 accepted FASTA").open("r", encoding="utf-8") as handle:
        header: str | None = None
        pieces: list[str] = []
        index = 0
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield build_fasta_record(header, pieces, index, path)
                    index += 1
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
                pieces = []
            else:
                if header is None:
                    raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
                pieces.append(line)
        if header is not None:
            yield build_fasta_record(header, pieces, index, path)


def build_fasta_record(
    header: str, pieces: Sequence[str], index: int, source_path: Path
) -> FastaRecord:
    identifier, separator, original_header = header.partition(" ")
    sequence = "".join(pieces).upper()
    if not identifier or not sequence:
        raise ValueError(f"{source_path}: empty identifier or sequence at record {index}")
    invalid = sorted(set(sequence).difference(RNA_ALPHABET))
    if invalid:
        raise ValueError(
            f"{source_path}: {identifier} contains non-ACGU characters {invalid}"
        )
    return FastaRecord(
        identifier=identifier,
        original_header=original_header if separator else identifier,
        sequence=sequence,
        input_index=index,
    )


def encode_with_context_cap(
    codec: AxisCodec,
    record: FastaRecord,
    family: str,
    condition_tokens: int,
    context_budget: int = CONTEXT_BUDGET,
) -> EncodedRecord:
    metrics, _fddfs, bpe = codec.encode_sequence(
        "rna",
        record.sequence,
        record.identifier,
        text_condition_tokens=condition_tokens,
    )
    original_metrics = dict(metrics)
    if int(metrics["full_context_tokens"]) <= context_budget:
        return EncodedRecord(
            source=record,
            training_sequence=record.sequence,
            target_bpe_ids=tuple(int(value) for value in bpe),
            original_metrics=original_metrics,
            final_metrics=dict(metrics),
            deleted_3prime_suffix="",
        )
    if family != CAP_FAMILY:
        raise ValueError(
            f"{record.identifier}: full context {metrics['full_context_tokens']} exceeds "
            f"{context_budget}; context cap is allowed only for {CAP_FAMILY}"
        )

    for deleted_count in range(1, len(record.sequence)):
        candidate = record.sequence[:-deleted_count]
        candidate_metrics, _candidate_fddfs, candidate_bpe = codec.encode_sequence(
            "rna",
            candidate,
            f"{record.identifier}:cap3p:{deleted_count}",
            text_condition_tokens=condition_tokens,
        )
        if int(candidate_metrics["full_context_tokens"]) <= context_budget:
            return EncodedRecord(
                source=record,
                training_sequence=candidate,
                target_bpe_ids=tuple(int(value) for value in candidate_bpe),
                original_metrics=original_metrics,
                final_metrics=dict(candidate_metrics),
                deleted_3prime_suffix=record.sequence[-deleted_count:],
            )
    raise ValueError(
        f"{record.identifier}: no non-empty 3' residue prefix fits context {context_budget}"
    )


def resolve_post_cap_collisions(
    records: Sequence[EncodedRecord],
) -> tuple[list[EncodedRecord], list[dict[str, Any]]]:
    groups: dict[str, list[EncodedRecord]] = {}
    for record in records:
        groups.setdefault(record.training_sequence, []).append(record)

    retained: list[EncodedRecord] = []
    collisions: list[dict[str, Any]] = []
    for group in groups.values():
        if len(group) == 1:
            retained.append(group[0])
            continue
        if not any(item.target_truncated for item in group):
            identifiers = [item.source.identifier for item in group]
            raise ValueError(
                "unexpected duplicate G6 sequences before context capping: "
                f"{identifiers}"
            )
        winner = min(
            group,
            key=lambda item: (item.target_truncated, item.source.input_index),
        )
        retained.append(winner)
        for item in group:
            if item is winner:
                continue
            collisions.append(
                {
                    "excluded_id": item.source.identifier,
                    "retained_id": winner.source.identifier,
                    "excluded_was_truncated": item.target_truncated,
                    "retained_was_truncated": winner.target_truncated,
                    "training_sequence_length": len(item.training_sequence),
                }
            )
    retained.sort(key=lambda item: item.source.input_index)
    collisions.sort(key=lambda item: item["excluded_id"])
    return retained, collisions


def sklearn_70_15_15(
    identifiers: Sequence[str],
    *,
    seed: int = 42,
    splitter: Callable[..., tuple[list[str], list[str]]] | None = None,
) -> dict[str, list[str]]:
    if splitter is None:
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise ImportError(
                "RfamGen split reconstruction requires scikit-learn on the worker"
            ) from exc
        splitter = train_test_split
    train, validation_test = splitter(
        list(identifiers), test_size=0.30, random_state=int(seed)
    )
    validation, test = splitter(
        list(validation_test), test_size=0.50, random_state=int(seed)
    )
    outputs = {
        "train": list(train),
        "validation": list(validation),
        "test": list(test),
    }
    flattened = [item for values in outputs.values() for item in values]
    if len(flattened) != len(identifiers) or set(flattened) != set(identifiers):
        raise ValueError("70/15/15 split is not an exact partition of family identifiers")
    return outputs


def quantile(values: Sequence[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (rank - low) * (ordered[high] - ordered[low]))


def summarize(values: Iterable[int]) -> dict[str, Any]:
    items = [int(value) for value in values]
    return {
        "count": len(items),
        "min": min(items) if items else None,
        "mean": sum(items) / len(items) if items else None,
        "p50": quantile(items, 0.50),
        "p95": quantile(items, 0.95),
        "p99": quantile(items, 0.99),
        "max": max(items) if items else None,
    }


def axis_sft_record(
    record: EncodedRecord,
    *,
    family: str,
    split: str,
    condition_tokens: int,
    context_budget: int,
) -> dict[str, Any]:
    final = record.final_metrics
    return {
        "id": record.source.identifier,
        "task": "rfamgen_figure2_reconstruction17",
        "target_type": "rna",
        "condition": {
            "type": "text",
            "text": family,
            "molecule_bpe_ids": [],
        },
        "target_bpe_ids": list(record.target_bpe_ids),
        "meta": {
            "benchmark": "rfamgen_figure2_paper_protocol_reconstruction17",
            "split": split,
            "source_id": record.source.identifier,
            "source_header": record.source.original_header,
            "family": family,
            "sequence_length": len(record.training_sequence),
            "original_sequence_length": len(record.source.sequence),
            "text_tokens": int(condition_tokens),
            "text_prefix_tokens": int(condition_tokens),
            "molecule_prefix_tokens": 0,
            "target_bpe_tokens": len(record.target_bpe_ids),
            "original_target_bpe_tokens": int(record.original_metrics["bpe_tokens"]),
            "full_context_tokens": int(final["full_context_tokens"]),
            "original_full_context_tokens": int(
                record.original_metrics["full_context_tokens"]
            ),
            "context_budget": int(context_budget),
            "target_truncated": record.target_truncated,
            "truncation_side": "3prime" if record.target_truncated else None,
            "truncated_residue_count": len(record.deleted_3prime_suffix),
            "g6_parsable_before_axis_context_cap": True,
            "evaluation_reference_uses_original_sequence": True,
        },
    }


def canonical_record(record: EncodedRecord, family: str, split: str) -> dict[str, Any]:
    return {
        "id": record.source.identifier,
        "family": family,
        "split": split,
        "condition": family,
        "original_sequence": record.source.sequence,
        "training_sequence": record.training_sequence,
        "original_header": record.source.original_header,
        "target_truncated": record.target_truncated,
        "deleted_3prime_suffix": record.deleted_3prime_suffix,
        "original_sequence_length": len(record.source.sequence),
        "training_sequence_length": len(record.training_sequence),
        "original_target_bpe_tokens": int(record.original_metrics["bpe_tokens"]),
        "target_bpe_tokens": len(record.target_bpe_ids),
        "original_full_context_tokens": int(
            record.original_metrics["full_context_tokens"]
        ),
        "full_context_tokens": int(record.final_metrics["full_context_tokens"]),
    }


def prepare_output_dir(output_dir: Path, resume: bool) -> bool:
    manifest_path = output_dir / "manifest.json"
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"--resume output directory does not exist: {output_dir}")
        if not manifest_path.is_file():
            raise RuntimeError(
                f"{output_dir}: incomplete prior output has no manifest; archive it before rerun"
            )
        manifest = load_json(manifest_path)
        if manifest.get("protocol") != PROTOCOL or manifest.get("status") != "complete":
            raise ValueError(f"{manifest_path}: not a complete {PROTOCOL} output")
        return True
    output_dir.mkdir(parents=True, exist_ok=False)
    return False


def materialize_family(args: argparse.Namespace) -> dict[str, Any]:
    if not FAMILY_PATTERN.fullmatch(args.family):
        raise ValueError(f"invalid Rfam family accession: {args.family!r}")
    if args.context_budget < TARGET_FRAME_TOKENS + 1:
        raise ValueError("--context-budget is too small for a non-empty target")

    g6_root = args.g6_root.resolve()
    g6_aggregate = load_json(g6_root / "manifest.json")
    if (
        g6_aggregate.get("protocol") != G6_AGGREGATE_PROTOCOL
        or g6_aggregate.get("status") != "complete"
    ):
        raise ValueError(f"{g6_root / 'manifest.json'}: G6 aggregate is incomplete")
    g6_manifest_path = g6_root / args.family / "manifest.json"
    g6_manifest = load_json(g6_manifest_path)
    if (
        g6_manifest.get("protocol") != G6_PROTOCOL
        or g6_manifest.get("status") != "complete"
        or g6_manifest.get("family") != args.family
    ):
        raise ValueError(f"{g6_manifest_path}: unexpected family G6 manifest")

    output_dir = (args.output_root / args.family).resolve()
    if prepare_output_dir(output_dir, args.resume):
        manifest = load_json(output_dir / "manifest.json")
        if manifest.get("family") != args.family:
            raise ValueError(f"{output_dir / 'manifest.json'}: family mismatch")
        print(json.dumps({"family": args.family, "status": "already_complete"}))
        return manifest

    guide = require_file(args.guide, "FDDFS guide")
    vocab = require_file(args.vocab, "FDDFS vocabulary")
    merges = require_file(args.merges, "frozen BPE codebook")
    tokenizer_path = require_file(args.tokenizer, "MPNet tokenizer")
    accepted_fasta = require_file(g6_root / args.family / "accepted.fasta", "accepted FASTA")

    from tokenizers import Tokenizer
    from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    condition_tokens = int(sum(tokenizer.encode(args.family).attention_mask))
    if condition_tokens < 1:
        raise ValueError(f"{args.family}: family condition encoded to zero tokens")
    codec = FrozenAxisCodec(guide, vocab, merges)

    source_records = list(iter_fasta(accepted_fasta))
    if len(source_records) != int(g6_manifest["accepted_records"]):
        raise ValueError(
            f"{args.family}: accepted FASTA has {len(source_records)} records, "
            f"G6 manifest has {g6_manifest['accepted_records']}"
        )
    identifiers = [record.identifier for record in source_records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{args.family}: duplicate accepted FASTA identifiers")

    encoded: list[EncodedRecord] = []
    for index, source_record in enumerate(source_records, start=1):
        encoded.append(
            encode_with_context_cap(
                codec,
                source_record,
                args.family,
                condition_tokens,
                args.context_budget,
            )
        )
        if index % args.progress_every == 0 or index == len(source_records):
            print(
                json.dumps(
                    {
                        "family": args.family,
                        "encoded": index,
                        "total": len(source_records),
                    }
                ),
                flush=True,
            )

    retained, collisions = resolve_post_cap_collisions(encoded)
    split_ids = sklearn_70_15_15(
        [record.source.identifier for record in retained], seed=args.split_seed
    )
    by_id = {record.source.identifier: record for record in retained}
    membership = {
        identifier: split
        for split, values in split_ids.items()
        for identifier in values
    }

    paths = {
        "canonical_jsonl": output_dir / "canonical.jsonl",
        "records_jsonl": output_dir / "records.jsonl",
        "cap_audit_jsonl": output_dir / "cap_audit.jsonl",
        "collisions_jsonl": output_dir / "cap_collisions.jsonl",
    }
    split_paths: dict[str, dict[str, Path]] = {}
    for split in ("train", "validation", "test"):
        split_paths[split] = {
            "sft": output_dir / f"{split}.jsonl",
            "reference": output_dir / f"{split}_references.fasta",
            "ids": output_dir / f"{split}_ids.txt",
        }

    temporary_paths: list[tuple[Path, Path]] = []
    handles: dict[str, Any] = {}
    try:
        for key, final_path in paths.items():
            temporary = final_path.with_name(final_path.name + ".tmp")
            handles[key] = temporary.open("x", encoding="utf-8")
            temporary_paths.append((temporary, final_path))
        for split, outputs in split_paths.items():
            for kind, final_path in outputs.items():
                temporary = final_path.with_name(final_path.name + ".tmp")
                handles[f"{split}:{kind}"] = temporary.open("x", encoding="utf-8")
                temporary_paths.append((temporary, final_path))

        for record in retained:
            split = membership[record.source.identifier]
            canonical = canonical_record(record, args.family, split)
            sft = axis_sft_record(
                record,
                family=args.family,
                split=split,
                condition_tokens=condition_tokens,
                context_budget=args.context_budget,
            )
            handles["canonical_jsonl"].write(json_line(canonical))
            handles["records_jsonl"].write(json_line(sft))
            if record.target_truncated:
                handles["cap_audit_jsonl"].write(json_line(canonical))

        for collision in collisions:
            handles["collisions_jsonl"].write(json_line(collision))

        for split, split_identifiers in split_ids.items():
            for identifier in split_identifiers:
                record = by_id[identifier]
                handles[f"{split}:sft"].write(
                    json_line(
                        axis_sft_record(
                            record,
                            family=args.family,
                            split=split,
                            condition_tokens=condition_tokens,
                            context_budget=args.context_budget,
                        )
                    )
                )
                handles[f"{split}:reference"].write(
                    f">{identifier} {record.source.original_header}\n"
                    f"{record.source.sequence}\n"
                )
                handles[f"{split}:ids"].write(identifier + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    for temporary, final_path in temporary_paths:
        os.replace(temporary, final_path)

    truncated = [record for record in retained if record.target_truncated]
    manifest = {
        "protocol": PROTOCOL,
        "protocol_scope": (
            "RfamGen Figure 2 paper-protocol reconstruction; exact official "
            "membership and split IDs were not published"
        ),
        "status": "complete",
        "family": args.family,
        "condition": args.family,
        "condition_tokens": condition_tokens,
        "context_budget": args.context_budget,
        "target_frame_tokens": TARGET_FRAME_TOKENS,
        "g6_manifest": str(g6_manifest_path),
        "g6_accepted_records": len(source_records),
        "records_after_cap_collision_dedup": len(retained),
        "cap_policy": {
            "allowed_family": CAP_FAMILY,
            "operation": "remove the minimum RNA residue suffix from the 3prime end; rebuild heavy graph and re-encode after each removed residue",
            "direct_fddfs_or_bpe_token_slicing": False,
            "evaluation_reference_is_never_capped": True,
        },
        "truncated_records": len(truncated),
        "truncated_residue_count": summarize(
            len(record.deleted_3prime_suffix) for record in truncated
        ),
        "cap_collision_exclusions": len(collisions),
        "original_full_context_tokens": summarize(
            record.original_metrics["full_context_tokens"] for record in retained
        ),
        "final_full_context_tokens": summarize(
            record.final_metrics["full_context_tokens"] for record in retained
        ),
        "final_target_bpe_tokens": summarize(
            len(record.target_bpe_ids) for record in retained
        ),
        "split": {
            "method": "sklearn train_test_split twice per family: test_size=0.30 then 0.50",
            "random_state": args.split_seed,
            "counts": {key: len(value) for key, value in split_ids.items()},
        },
        "codec": codec.codebook,
        "tokenizer": str(tokenizer_path),
        "outputs": {
            **{key: str(value) for key, value in paths.items()},
            "splits": {
                split: {kind: str(path) for kind, path in values.items()}
                for split, values in split_paths.items()
            },
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--g6-root", type=Path, default=DEFAULT_G6_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    materialize_family(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
