#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
from array import array
from pathlib import Path


MAGIC = b"YTTMIBPE"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIQQi")
LEN_STRUCT = struct.Struct("<I")


def write_array(path: Path, typecode: str, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    arr = array(typecode, values)
    if arr and typecode == "H" and (min(arr) < 0 or max(arr) > 65535):
        raise ValueError(f"uint16 value out of range for {path}")
    if arr and typecode == "I" and (min(arr) < 0 or max(arr) > 0xFFFFFFFF):
        raise ValueError(f"uint32 value out of range for {path}")
    if arr and typecode == "Q" and min(arr) < 0:
        raise ValueError(f"uint64 value out of range for {path}")
    arr.tofile(tmp.open("wb"))
    os.replace(tmp, path)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_sequences(num_records: int, vocab_body: int, num_domains: int) -> tuple[list[list[int]], list[int]]:
    seqs: list[list[int]] = []
    domain_ids: list[int] = []
    for i in range(num_records):
        domain = i % num_domains
        length = 3 + (i % 6)
        base = 1 + domain * 7
        seq = [((base + i + j) % vocab_body) for j in range(length)]
        seqs.append(seq)
        domain_ids.append(domain)
    return seqs, domain_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a tiny compact-int dataset with domain sidecars for smoke tests.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-records", type=int, default=80)
    ap.add_argument("--vocab-body", type=int, default=32, help="Raw compact token ids are in [0, vocab_body).")
    ap.add_argument("--num-domains", type=int, default=5)
    args = ap.parse_args()

    if args.num_records <= 0:
        raise ValueError("--num-records must be positive")
    if args.vocab_body <= 1:
        raise ValueError("--vocab-body must be > 1")
    if args.num_domains <= 0 or args.num_domains > 65535:
        raise ValueError("--num-domains must be in [1, 65535]")

    compact_dir = args.out_dir / "compact"
    index_dir = args.out_dir / "index"
    domain_id_dir = args.out_dir / "domain_ids"
    compact_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    domain_id_dir.mkdir(parents=True, exist_ok=True)

    seqs, domain_ids = build_sequences(args.num_records, args.vocab_body, args.num_domains)
    token_path = compact_dir / "shard_00000.bin"
    offsets: list[int] = []
    lengths: list[int] = []
    ntokens = 0
    max_token = -1
    tmp = token_path.with_suffix(token_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, VERSION, len(seqs), 0, -1))
        for seq in seqs:
            offsets.append(f.tell() + LEN_STRUCT.size)
            lengths.append(len(seq))
            ntokens += len(seq)
            max_token = max(max_token, max(seq))
            f.write(LEN_STRUCT.pack(len(seq)))
            f.write(struct.pack(f"<{len(seq)}i", *seq))
        f.seek(0)
        f.write(HEADER_STRUCT.pack(MAGIC, VERSION, len(seqs), ntokens, max_token))
    os.replace(tmp, token_path)

    stem = token_path.name
    write_array(index_dir / f"{stem}.payload_offsets.u64", "Q", offsets)
    write_array(index_dir / f"{stem}.lengths.u32", "I", lengths)
    write_json(
        index_dir / f"{stem}.meta.json",
        {
            "input": str(token_path),
            "records": len(seqs),
            "tokens": ntokens,
            "max_token": max_token,
            "min_len": min(lengths),
            "max_len": max(lengths),
            "empty_samples": 0,
            "payload_offsets_path": str(index_dir / f"{stem}.payload_offsets.u64"),
            "lengths_path": str(index_dir / f"{stem}.lengths.u32"),
        },
    )
    write_array(domain_id_dir / f"{stem}.domain_ids.u16", "H", domain_ids)
    write_json(
        compact_dir / "done.json",
        {
            "mode": "tiny_domain_compact_smoke",
            "records": len(seqs),
            "tokens": ntokens,
            "max_token": max_token,
            "shards": [{"path": str(token_path), "records": len(seqs), "tokens": ntokens, "max_token": max_token}],
        },
    )
    write_json(
        args.out_dir / "manifest.json",
        {
            "compact_dir": str(compact_dir),
            "index_dir": str(index_dir),
            "domain_id_dir": str(domain_id_dir),
            "token_file": str(token_path),
            "records": len(seqs),
            "tokens": ntokens,
            "num_domains": int(args.num_domains),
            "domain_records": {str(i): int(domain_ids.count(i)) for i in range(args.num_domains)},
        },
    )
    print(json.dumps({"event": "tiny_domain_dataset_done", "out_dir": str(args.out_dir), "records": len(seqs), "tokens": ntokens}, ensure_ascii=False))


if __name__ == "__main__":
    main()
