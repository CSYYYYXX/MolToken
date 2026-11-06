#!/usr/bin/env python3
"""为 YTTMIBPE compact-int shard 构建在线读取索引。

输出每个 shard 对应的：
- *.payload_offsets.u64: 每条样本 token payload 的字节偏移
- *.lengths.u32: 每条样本的压缩后 token 长度
- *.meta.json: header 和统计信息

这个索引用于训练时在线读取单个样本、随机展开 BPE、再 pad 到固定长度。
"""

from __future__ import annotations

import argparse
from array import array
import glob
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import BinaryIO, Sequence


MAGIC = b"YTTMIBPE"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIQQi")
LEN_STRUCT = struct.Struct("<I")


def read_exact(f: BinaryIO, nbytes: int, what: str) -> bytes:
    data = f.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"unexpected EOF while reading {what}")
    return data


def read_header(f: BinaryIO, path: Path) -> tuple[int, int, int]:
    raw = read_exact(f, HEADER_STRUCT.size, f"header from {path}")
    magic, version, nseq, ntokens, max_token = HEADER_STRUCT.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"bad compact-int magic in {path}: {magic!r}")
    if int(version) != VERSION:
        raise ValueError(f"unsupported compact-int version={version} in {path}")
    return int(nseq), int(ntokens), int(max_token)


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    out = sorted(dict.fromkeys(paths))
    if not out:
        raise FileNotFoundError(f"no input shards matched: {patterns}")
    return out


def index_paths(index_dir: Path, compact_path: Path) -> tuple[Path, Path, Path]:
    stem = compact_path.name
    return (
        index_dir / f"{stem}.payload_offsets.u64",
        index_dir / f"{stem}.lengths.u32",
        index_dir / f"{stem}.meta.json",
    )


def write_array_values(f: BinaryIO, typecode: str, values: list[int]) -> None:
    if not values:
        return
    arr = array(typecode, values)
    if sys.byteorder != "little":
        arr.byteswap()
    arr.tofile(f)
    values.clear()


def atomic_replace(tmp: Path, final: Path) -> None:
    os.replace(tmp, final)


def build_one(path: Path, index_dir: Path, overwrite: bool, flush_records: int) -> dict[str, object]:
    offsets_path, lengths_path, meta_path = index_paths(index_dir, path)
    if not overwrite and offsets_path.exists() and lengths_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"input": str(path), "status": "exists", **meta}

    index_dir.mkdir(parents=True, exist_ok=True)
    offsets_tmp = offsets_path.with_suffix(offsets_path.suffix + ".tmp")
    lengths_tmp = lengths_path.with_suffix(lengths_path.suffix + ".tmp")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    started = time.time()
    records = 0
    observed_tokens = 0
    min_len: int | None = None
    max_len = 0
    empty = 0
    offset_buf: list[int] = []
    length_buf: list[int] = []

    with path.open("rb") as f, offsets_tmp.open("wb") as offsets_f, lengths_tmp.open("wb") as lengths_f:
        nseq, ntokens, max_token = read_header(f, path)
        for idx in range(nseq):
            (length,) = LEN_STRUCT.unpack(read_exact(f, LEN_STRUCT.size, f"length {idx} from {path}"))
            length = int(length)
            payload_offset = int(f.tell())
            offset_buf.append(payload_offset)
            length_buf.append(length)
            records += 1
            observed_tokens += length
            min_len = length if min_len is None else min(min_len, length)
            max_len = max(max_len, length)
            if length == 0:
                empty += 1
            f.seek(length * 4, 1)
            if len(offset_buf) >= flush_records:
                write_array_values(offsets_f, "Q", offset_buf)
                write_array_values(lengths_f, "I", length_buf)
        write_array_values(offsets_f, "Q", offset_buf)
        write_array_values(lengths_f, "I", length_buf)

    if records != nseq:
        raise RuntimeError(f"record count mismatch for {path}: header={nseq} observed={records}")
    if observed_tokens != ntokens:
        raise RuntimeError(f"token count mismatch for {path}: header={ntokens} observed={observed_tokens}")

    meta = {
        "input": str(path),
        "records": int(records),
        "tokens": int(observed_tokens),
        "max_token": int(max_token),
        "min_len": int(min_len or 0),
        "max_len": int(max_len),
        "empty_samples": int(empty),
        "payload_offsets_path": str(offsets_path),
        "lengths_path": str(lengths_path),
        "elapsed_sec": time.time() - started,
    }
    meta_tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    atomic_replace(offsets_tmp, offsets_path)
    atomic_replace(lengths_tmp, lengths_path)
    atomic_replace(meta_tmp, meta_path)
    return {"status": "built", **meta}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build compact-int random-access indices.")
    ap.add_argument("--input-shards", nargs="+", required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--flush-records", type=int, default=1_000_000)
    ap.add_argument("--summary-json", type=Path, default=None)
    args = ap.parse_args()

    paths = expand_inputs(args.input_shards)
    summaries = []
    for path in paths:
        row = build_one(path, args.index_dir, args.overwrite, args.flush_records)
        summaries.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    built = {
        "index_dir": str(args.index_dir),
        "num_shards": len(summaries),
        "records": int(sum(int(x.get("records", 0)) for x in summaries)),
        "tokens": int(sum(int(x.get("tokens", 0)) for x in summaries)),
        "max_len": int(max(int(x.get("max_len", 0)) for x in summaries)),
        "empty_samples": int(sum(int(x.get("empty_samples", 0)) for x in summaries)),
        "shards": summaries,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.summary_json.with_suffix(args.summary_json.suffix + ".tmp")
        tmp.write_text(json.dumps(built, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, args.summary_json)


if __name__ == "__main__":
    main()
