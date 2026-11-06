#!/usr/bin/env python3
"""Audit domain-wise compact-corpus sizes without reading token payloads.

The compact random-access index stores one uint32 BPE-body length per record,
and the domain sidecar stores one aligned uint16 domain id.  This script scans
those two arrays, applies a deterministic record-level split, and reports both
raw and context-capped token counts.  It deliberately does not hash any file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_HF_DOWNLOAD_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_HF_DOWNLOAD_ROOT


DOMAIN_NAMES = ("small_molecule", "peptide", "protein", "dna", "rna")
SPLIT_NAMES = ("train", "validation", "test")
UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
SPLIT_MODULUS = 10_000


def splitmix64_array(values: np.ndarray) -> np.ndarray:
    """Vectorized form of the loader's scalar splitmix64 implementation."""
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return (x ^ (x >> np.uint64(31))) & UINT64_MASK


def quantile_from_histogram(histogram: np.ndarray, q: float) -> int:
    count = int(histogram.sum())
    if count <= 0:
        raise ValueError("cannot compute a quantile of an empty histogram")
    rank = int((count - 1) * q)
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    return int(np.searchsorted(cumulative, rank + 1, side="left"))


def main() -> None:
    default_root = DEFAULT_HF_DOWNLOAD_ROOT / "HoloMol-Pretrain-Compact"
    default_out = DEFAULT_DATA_ROOT / "reports" / "domainwise_compact_audit_seed20260902.json"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=default_root)
    ap.add_argument("--out-json", type=Path, default=default_out)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--split-seed", type=int, default=20260902)
    ap.add_argument("--split-modulus", type=int, default=SPLIT_MODULUS)
    ap.add_argument("--validation-buckets", type=int, default=50)
    ap.add_argument("--test-buckets", type=int, default=50)
    ap.add_argument("--chunk-records", type=int, default=2_000_000)
    args = ap.parse_args()

    root = args.dataset_root.expanduser().resolve()
    out_json = args.out_json.expanduser().resolve()
    if args.context_length < 2:
        raise ValueError("--context-length must be at least 2")
    if args.chunk_records <= 0:
        raise ValueError("--chunk-records must be positive")
    if args.split_modulus <= 0:
        raise ValueError("--split-modulus must be positive")
    heldout_buckets = args.validation_buckets + args.test_buckets
    if args.validation_buckets <= 0 or args.test_buckets <= 0:
        raise ValueError("validation and test bucket counts must both be positive")
    if heldout_buckets >= args.split_modulus:
        raise ValueError("validation plus test buckets must be below the modulus")

    compact_dir = root / "encoded_compact_int_prefix70_mf80000_11w_v3"
    index_dir = root / "encoded_compact_int_prefix70_mf80000_11w_v3_index"
    domain_dir = root / "encoded_compact_int_prefix70_mf80000_11w_v3_domain_ids"
    token_paths = sorted(compact_dir.glob("shard_*.bin"))
    if not token_paths:
        raise FileNotFoundError(f"no compact shards under {compact_dir}")

    index_summary_path = index_dir / "summary.json"
    domain_summary_path = domain_dir / "summary.json"
    index_summary = json.loads(index_summary_path.read_text(encoding="utf-8"))
    domain_summary = json.loads(domain_summary_path.read_text(encoding="utf-8"))
    histogram_width = int(index_summary["max_len"]) + 1
    if histogram_width <= 1:
        raise ValueError(f"invalid max length in {index_summary_path}")

    num_domains = len(DOMAIN_NAMES)
    num_splits = len(SPLIT_NAMES)
    record_counts = np.zeros((num_domains, num_splits), dtype=np.uint64)
    raw_body_tokens = np.zeros_like(record_counts)
    effective_body_tokens = np.zeros_like(record_counts)
    truncated_records = np.zeros_like(record_counts)
    length_histograms = np.zeros((num_domains, histogram_width), dtype=np.uint64)
    shard_rows: list[dict[str, Any]] = []

    body_cap = args.context_length - 1
    seed_u64 = np.uint64(args.split_seed & 0xFFFFFFFFFFFFFFFF)
    validation_end = args.validation_buckets
    test_end = validation_end + args.test_buckets

    for token_path in token_paths:
        stem = token_path.name
        lengths_path = index_dir / f"{stem}.lengths.u32"
        domains_path = domain_dir / f"{stem}.domain_ids.u16"
        meta_path = index_dir / f"{stem}.meta.json"
        for required in (lengths_path, domains_path, meta_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        shard_id_text = stem.removeprefix("shard_").removesuffix(".bin")
        shard_id = int(shard_id_text)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lengths = np.memmap(lengths_path, mode="r", dtype="<u4")
        domains = np.memmap(domains_path, mode="r", dtype="<u2")
        records = int(lengths.shape[0])
        if int(domains.shape[0]) != records:
            raise ValueError(f"sidecar length mismatch for {stem}")
        if int(meta["records"]) != records:
            raise ValueError(f"index meta record mismatch for {stem}")

        shard_body_tokens = 0
        shard_domain_records = np.zeros(num_domains, dtype=np.uint64)
        for start in range(0, records, args.chunk_records):
            stop = min(records, start + args.chunk_records)
            length_chunk = np.asarray(lengths[start:stop], dtype=np.uint64)
            domain_chunk = np.asarray(domains[start:stop], dtype=np.int64)
            if domain_chunk.size == 0:
                continue
            bad = (domain_chunk < 0) | (domain_chunk >= num_domains)
            if bool(bad.any()):
                bad_ids = np.unique(domain_chunk[bad]).tolist()
                raise ValueError(f"undefined domain ids in {stem}: {bad_ids}")
            max_observed = int(length_chunk.max())
            if max_observed >= histogram_width:
                raise ValueError(
                    f"observed length {max_observed} exceeds summary max {histogram_width - 1}"
                )

            shard_body_tokens += int(length_chunk.sum(dtype=np.uint64))
            shard_domain_records += np.bincount(domain_chunk, minlength=num_domains).astype(np.uint64)
            histogram_keys = domain_chunk * histogram_width + length_chunk.astype(np.int64)
            length_histograms += np.bincount(
                histogram_keys, minlength=num_domains * histogram_width
            ).reshape(num_domains, histogram_width).astype(np.uint64)

            raw_indices = np.arange(start, stop, dtype=np.uint64)
            tuple_keys = (np.uint64(shard_id) << np.uint64(32)) | raw_indices
            buckets = splitmix64_array(tuple_keys ^ seed_u64) % np.uint64(args.split_modulus)
            split_ids = np.zeros(domain_chunk.shape[0], dtype=np.int64)
            split_ids[buckets < np.uint64(validation_end)] = 1
            split_ids[
                (buckets >= np.uint64(validation_end)) & (buckets < np.uint64(test_end))
            ] = 2
            grouped = domain_chunk * num_splits + split_ids
            effective_chunk = np.minimum(length_chunk, np.uint64(body_cap))
            truncated_chunk = length_chunk > np.uint64(body_cap)

            record_counts += np.bincount(
                grouped, minlength=num_domains * num_splits
            ).reshape(num_domains, num_splits).astype(np.uint64)
            raw_body_tokens += np.rint(
                np.bincount(
                    grouped,
                    weights=length_chunk.astype(np.float64),
                    minlength=num_domains * num_splits,
                )
            ).astype(np.uint64).reshape(num_domains, num_splits)
            effective_body_tokens += np.rint(
                np.bincount(
                    grouped,
                    weights=effective_chunk.astype(np.float64),
                    minlength=num_domains * num_splits,
                )
            ).astype(np.uint64).reshape(num_domains, num_splits)
            truncated_records += np.bincount(
                grouped,
                weights=truncated_chunk.astype(np.uint8),
                minlength=num_domains * num_splits,
            ).astype(np.uint64).reshape(num_domains, num_splits)

        if shard_body_tokens != int(meta["tokens"]):
            raise ValueError(
                f"index token mismatch for {stem}: observed={shard_body_tokens} meta={meta['tokens']}"
            )
        shard_rows.append(
            {
                "shard_id": shard_id,
                "file": str(token_path),
                "records": records,
                "stored_body_tokens": shard_body_tokens,
                "domain_records": {
                    name: int(shard_domain_records[i]) for i, name in enumerate(DOMAIN_NAMES)
                },
            }
        )
        print(
            json.dumps(
                {
                    "event": "shard_audited",
                    "shard_id": shard_id,
                    "records": records,
                    "stored_body_tokens": shard_body_tokens,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    domain_rows: dict[str, Any] = {}
    for domain_id, domain_name in enumerate(DOMAIN_NAMES):
        histogram = length_histograms[domain_id]
        split_rows: dict[str, Any] = {}
        for split_id, split_name in enumerate(SPLIT_NAMES):
            records = int(record_counts[domain_id, split_id])
            raw_body = int(raw_body_tokens[domain_id, split_id])
            effective_body = int(effective_body_tokens[domain_id, split_id])
            trunc = int(truncated_records[domain_id, split_id])
            split_rows[split_name] = {
                "records": records,
                "stored_body_tokens": raw_body,
                "raw_input_tokens_with_eos": raw_body + records,
                "effective_body_tokens_context_capped": effective_body,
                "effective_input_tokens_with_eos": effective_body + records,
                "supervised_causal_loss_tokens": effective_body,
                "tokens_removed_by_context_cap": raw_body - effective_body,
                "truncated_records": trunc,
                "truncated_fraction": trunc / records if records else None,
            }

        records_total = int(record_counts[domain_id].sum())
        raw_body_total = int(raw_body_tokens[domain_id].sum())
        effective_body_total = int(effective_body_tokens[domain_id].sum())
        trunc_total = int(truncated_records[domain_id].sum())
        nonzero_lengths = np.flatnonzero(histogram)
        domain_rows[domain_name] = {
            "domain_id": domain_id,
            "records": records_total,
            "stored_body_tokens": raw_body_total,
            "raw_input_tokens_with_eos": raw_body_total + records_total,
            "effective_body_tokens_context_capped": effective_body_total,
            "effective_input_tokens_with_eos": effective_body_total + records_total,
            "supervised_causal_loss_tokens": effective_body_total,
            "tokens_removed_by_context_cap": raw_body_total - effective_body_total,
            "truncated_records": trunc_total,
            "truncated_fraction": trunc_total / records_total,
            "body_length": {
                "mean": raw_body_total / records_total,
                "min": int(nonzero_lengths[0]),
                "p50": quantile_from_histogram(histogram, 0.50),
                "p90": quantile_from_histogram(histogram, 0.90),
                "p95": quantile_from_histogram(histogram, 0.95),
                "p99": quantile_from_histogram(histogram, 0.99),
                "p999": quantile_from_histogram(histogram, 0.999),
                "max": int(nonzero_lengths[-1]),
            },
            "splits": split_rows,
            "four_train_epoch_effective_input_token_cap": 4
            * int(split_rows["train"]["effective_input_tokens_with_eos"]),
        }

    total_records = int(record_counts.sum())
    total_raw_body = int(raw_body_tokens.sum())
    total_effective_body = int(effective_body_tokens.sum())
    total_truncated = int(truncated_records.sum())
    declared_records = int(index_summary["records"])
    declared_tokens = int(index_summary["tokens"])
    if total_records != declared_records:
        raise ValueError(f"total record mismatch: observed={total_records} declared={declared_records}")
    if total_raw_body != declared_tokens:
        raise ValueError(f"total token mismatch: observed={total_raw_body} declared={declared_tokens}")
    declared_domain_records = domain_summary["domain_records"]
    for name in DOMAIN_NAMES:
        if int(domain_rows[name]["records"]) != int(declared_domain_records[name]):
            raise ValueError(f"domain record mismatch for {name}")

    report = {
        "schema_version": "domainwise_compact_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "inputs": {
            "compact_dir": str(compact_dir),
            "index_dir": str(index_dir),
            "domain_id_dir": str(domain_dir),
            "num_shards": len(token_paths),
        },
        "token_accounting": {
            "context_length": args.context_length,
            "body_token_cap": body_cap,
            "eos_tokens_per_record": 1,
            "padding_tokens_counted": False,
            "flops_token_definition": "min(stored_body_length, context_length - 1) + one EOS",
            "causal_loss_token_definition": "min(stored_body_length, context_length - 1)",
            "bpe_unfolding_assumed": "none",
        },
        "split_assignment": {
            "status": "proposed_freeze_candidate",
            "stable_key": "(uint64(shard_id) << 32) | uint64(raw_idx)",
            "hash": "splitmix64(stable_key XOR split_seed)",
            "seed": args.split_seed,
            "modulus": args.split_modulus,
            "validation_buckets": [0, validation_end - 1],
            "test_buckets": [validation_end, test_end - 1],
            "train_buckets": [test_end, args.split_modulus - 1],
            "target_fractions": {
                "train": (args.split_modulus - heldout_buckets) / args.split_modulus,
                "validation": args.validation_buckets / args.split_modulus,
                "test": args.test_buckets / args.split_modulus,
            },
        },
        "totals": {
            "records": total_records,
            "stored_body_tokens": total_raw_body,
            "raw_input_tokens_with_eos": total_raw_body + total_records,
            "effective_body_tokens_context_capped": total_effective_body,
            "effective_input_tokens_with_eos": total_effective_body + total_records,
            "tokens_removed_by_context_cap": total_raw_body - total_effective_body,
            "truncated_records": total_truncated,
            "truncated_fraction": total_truncated / total_records,
            "records_by_split": {
                name: int(record_counts[:, i].sum()) for i, name in enumerate(SPLIT_NAMES)
            },
            "effective_input_tokens_with_eos_by_split": {
                name: int((effective_body_tokens[:, i] + record_counts[:, i]).sum())
                for i, name in enumerate(SPLIT_NAMES)
            },
        },
        "domains": domain_rows,
        "shards": shard_rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_json)
    print(
        json.dumps(
            {
                "event": "domainwise_audit_complete",
                "out_json": str(out_json),
                "records": total_records,
                "effective_input_tokens_with_eos": total_effective_body + total_records,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
