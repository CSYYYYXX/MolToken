#!/usr/bin/env python3
"""Materialize frozen domain/split views as uint32 raw-record indices.

The compact token payload is not copied.  Each output index is relative to its
source compact shard and follows the exact splitmix64 rule frozen by the
domain-wise audit.  The top-level manifest is the only file consumed by the
training loader.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

try:
    from .audit_domainwise_compact import DOMAIN_NAMES, SPLIT_NAMES, splitmix64_array
    from .scaling_paths import DEFAULT_DATA_ROOT
except ImportError:  # direct script execution
    from audit_domainwise_compact import DOMAIN_NAMES, SPLIT_NAMES, splitmix64_array
    from scaling_paths import DEFAULT_DATA_ROOT


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def shard_id_from_name(name: str) -> int:
    prefix = "shard_"
    suffix = ".bin"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"unexpected compact shard name: {name}")
    return int(name[len(prefix) : -len(suffix)])


def selected_split_ids(
    shard_id: int,
    start: int,
    stop: int,
    seed: int,
    modulus: int,
    validation_end: int,
    test_end: int,
) -> np.ndarray:
    raw_indices = np.arange(start, stop, dtype=np.uint64)
    stable_keys = (np.uint64(shard_id) << np.uint64(32)) | raw_indices
    buckets = splitmix64_array(stable_keys ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF))
    buckets %= np.uint64(modulus)
    split_ids = np.zeros(stop - start, dtype=np.uint8)
    split_ids[buckets < np.uint64(validation_end)] = 1
    split_ids[
        (buckets >= np.uint64(validation_end)) & (buckets < np.uint64(test_end))
    ] = 2
    return split_ids


def output_index_path(root: Path, domain: str, split: str, shard_name: str) -> Path:
    return root / domain / split / f"{shard_name}.raw_indices.u32"


def materialize_shard(
    *,
    output_root: Path,
    compact_path: Path,
    lengths_path: Path,
    domains_path: Path,
    split_seed: int,
    split_modulus: int,
    validation_end: int,
    test_end: int,
    body_cap: int,
    chunk_records: int,
) -> list[dict[str, Any]]:
    shard_name = compact_path.name
    shard_id = shard_id_from_name(shard_name)
    lengths = np.memmap(lengths_path, mode="r", dtype="<u4")
    domains = np.memmap(domains_path, mode="r", dtype="<u2")
    records = int(lengths.shape[0])
    if int(domains.shape[0]) != records:
        raise ValueError(f"domain/length sidecar mismatch for {shard_name}")
    if records > int(np.iinfo(np.uint32).max):
        raise ValueError(f"one shard cannot be addressed by uint32: {shard_name}")

    final_paths: dict[int, Path] = {}
    tmp_paths: dict[int, Path] = {}
    handles: dict[int, BinaryIO] = {}
    counts = np.zeros(len(DOMAIN_NAMES) * len(SPLIT_NAMES), dtype=np.uint64)
    tokens = np.zeros_like(counts)
    try:
        for domain_id, domain_name in enumerate(DOMAIN_NAMES):
            for split_id, split_name in enumerate(SPLIT_NAMES):
                group_id = domain_id * len(SPLIT_NAMES) + split_id
                final_path = output_index_path(output_root, domain_name, split_name, shard_name)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
                tmp_path.unlink(missing_ok=True)
                final_paths[group_id] = final_path
                tmp_paths[group_id] = tmp_path
                handles[group_id] = tmp_path.open("wb")

        for start in range(0, records, chunk_records):
            stop = min(records, start + chunk_records)
            domain_chunk = np.asarray(domains[start:stop], dtype=np.int64)
            if bool(((domain_chunk < 0) | (domain_chunk >= len(DOMAIN_NAMES))).any()):
                raise ValueError(f"undefined domain id in {shard_name} [{start}:{stop}]")
            split_chunk = selected_split_ids(
                shard_id,
                start,
                stop,
                split_seed,
                split_modulus,
                validation_end,
                test_end,
            ).astype(np.int64, copy=False)
            grouped = domain_chunk * len(SPLIT_NAMES) + split_chunk
            raw_indices = np.arange(start, stop, dtype="<u4")
            effective_tokens = (
                np.minimum(np.asarray(lengths[start:stop], dtype=np.uint64), np.uint64(body_cap))
                + np.uint64(1)
            )

            # Sorting a 0..14 key once is substantially cheaper than scanning
            # the full chunk once for every view.
            order = np.argsort(grouped, kind="stable")
            sorted_groups = grouped[order]
            boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
            starts = np.r_[0, boundaries]
            stops = np.r_[boundaries, len(order)]
            for lo, hi in zip(starts.tolist(), stops.tolist()):
                group_id = int(sorted_groups[lo])
                chosen = order[lo:hi]
                raw_indices[chosen].tofile(handles[group_id])
                counts[group_id] += np.uint64(hi - lo)
                tokens[group_id] += effective_tokens[chosen].sum(dtype=np.uint64)
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in tmp_paths.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for group_id, tmp_path in tmp_paths.items():
            expected_bytes = int(counts[group_id]) * np.dtype("<u4").itemsize
            if tmp_path.stat().st_size != expected_bytes:
                raise RuntimeError(f"short raw-index output: {tmp_path}")
            os.replace(tmp_path, final_paths[group_id])

    rows: list[dict[str, Any]] = []
    for domain_id, domain_name in enumerate(DOMAIN_NAMES):
        for split_id, split_name in enumerate(SPLIT_NAMES):
            group_id = domain_id * len(SPLIT_NAMES) + split_id
            rows.append(
                {
                    "domain": domain_name,
                    "domain_id": domain_id,
                    "split": split_name,
                    "shard_id": shard_id,
                    "token_file": str(compact_path),
                    "raw_indices_file": str(final_paths[group_id]),
                    "records": int(counts[group_id]),
                    "effective_input_tokens_with_eos": int(tokens[group_id]),
                }
            )
    return rows


def main() -> None:
    default_audit = DEFAULT_DATA_ROOT / "reports" / "domainwise_compact_audit_seed20260902.json"
    default_output = DEFAULT_DATA_ROOT / "processed" / "domain_split_manifests_v1"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-json", type=Path, default=default_audit)
    ap.add_argument("--output-dir", type=Path, default=default_output)
    ap.add_argument("--chunk-records", type=int, default=2_000_000)
    args = ap.parse_args()
    if args.chunk_records <= 0:
        raise ValueError("--chunk-records must be positive")

    audit_path = args.audit_json.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assignment = audit["split_assignment"]
    if assignment.get("status") != "proposed_freeze_candidate":
        raise ValueError("audit split assignment is not the reviewed freeze candidate")
    if assignment.get("hash") != "splitmix64(stable_key XOR split_seed)":
        raise ValueError("unsupported split hash definition")

    compact_dir = Path(audit["inputs"]["compact_dir"]).resolve()
    index_dir = Path(audit["inputs"]["index_dir"]).resolve()
    domain_dir = Path(audit["inputs"]["domain_id_dir"]).resolve()
    body_cap = int(audit["token_accounting"]["body_token_cap"])
    split_seed = int(assignment["seed"])
    split_modulus = int(assignment["modulus"])
    validation_end = int(assignment["validation_buckets"][1]) + 1
    test_end = int(assignment["test_buckets"][1]) + 1
    token_paths = sorted(compact_dir.glob("shard_*.bin"))
    if not token_paths:
        raise FileNotFoundError(f"no compact shards under {compact_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for compact_path in token_paths:
        lengths_path = index_dir / f"{compact_path.name}.lengths.u32"
        domains_path = domain_dir / f"{compact_path.name}.domain_ids.u16"
        if not lengths_path.is_file() or not domains_path.is_file():
            raise FileNotFoundError(f"missing sidecar(s) for {compact_path.name}")
        shard_rows = materialize_shard(
            output_root=output_root,
            compact_path=compact_path.resolve(),
            lengths_path=lengths_path,
            domains_path=domains_path,
            split_seed=split_seed,
            split_modulus=split_modulus,
            validation_end=validation_end,
            test_end=test_end,
            body_cap=body_cap,
            chunk_records=args.chunk_records,
        )
        all_rows.extend(shard_rows)
        print(
            json.dumps(
                {
                    "event": "domain_split_shard_materialized",
                    "shard": compact_path.name,
                    "records": sum(row["records"] for row in shard_rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    views: dict[str, dict[str, Any]] = {}
    for domain_name in DOMAIN_NAMES:
        views[domain_name] = {}
        for split_name in SPLIT_NAMES:
            rows = [
                row for row in all_rows
                if row["domain"] == domain_name and row["split"] == split_name
            ]
            expected = audit["domains"][domain_name]["splits"][split_name]
            records = sum(int(row["records"]) for row in rows)
            tokens = sum(int(row["effective_input_tokens_with_eos"]) for row in rows)
            if records != int(expected["records"]):
                raise RuntimeError(f"record total differs from audit for {domain_name}/{split_name}")
            if tokens != int(expected["effective_input_tokens_with_eos"]):
                raise RuntimeError(f"token total differs from audit for {domain_name}/{split_name}")
            views[domain_name][split_name] = {
                "domain_id": int(audit["domains"][domain_name]["domain_id"]),
                "records": records,
                "effective_input_tokens_with_eos": tokens,
                "shards": rows,
            }

    manifest = {
        "schema_version": "domain_split_raw_indices_v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_json": str(audit_path),
        "dataset": {
            "compact_dir": str(compact_dir),
            "index_dir": str(index_dir),
            "domain_id_dir": str(domain_dir),
            "context_length": int(audit["token_accounting"]["context_length"]),
            "num_shards": len(token_paths),
        },
        "split_assignment": assignment,
        "views": views,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "domain_split_manifests_complete",
                "manifest": str(output_root / "manifest.json"),
                "index_bytes": sum(
                    Path(row["raw_indices_file"]).stat().st_size for row in all_rows
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
