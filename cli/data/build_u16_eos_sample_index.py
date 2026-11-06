#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if matches:
            out.extend(matches)
            continue
        p = Path(pattern)
        if p.exists():
            out.append(p)
    out = sorted(dict.fromkeys(out))
    if not out:
        raise FileNotFoundError(f"no token files matched: {patterns}")
    return out


def index_paths(out_dir: Path, token_path: Path) -> tuple[Path, Path, Path]:
    stem = token_path.name
    return (
        out_dir / f"{stem}.starts.npy",
        out_dir / f"{stem}.lengths.npy",
        out_dir / f"{stem}.meta.json",
    )


def scan_one_file(
    token_path: Path,
    out_dir: Path,
    eos_token_id: int,
    chunk_tokens: int,
    force: bool,
) -> dict[str, Any]:
    starts_path, lengths_path, meta_path = index_paths(out_dir, token_path)
    if not force and starts_path.exists() and lengths_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "skipped_existing"
        return meta

    n_tokens = token_path.stat().st_size // np.dtype(np.uint16).itemsize
    arr = np.memmap(token_path, mode="r", dtype=np.uint16)
    start_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    next_start = 0
    max_len = 0
    min_len = None
    started = time.time()

    for offset in range(0, n_tokens, chunk_tokens):
        chunk = arr[offset: min(offset + chunk_tokens, n_tokens)]
        eos_local = np.flatnonzero(chunk == eos_token_id)
        if eos_local.size == 0:
            continue

        eos_global = eos_local.astype(np.uint64, copy=False) + np.uint64(offset)
        starts = np.empty(eos_global.shape[0], dtype=np.uint64)
        starts[0] = np.uint64(next_start)
        if eos_global.shape[0] > 1:
            starts[1:] = eos_global[:-1] + np.uint64(1)
        lengths_u64 = eos_global - starts + np.uint64(1)
        local_max = int(lengths_u64.max())
        local_min = int(lengths_u64.min())
        max_len = max(max_len, local_max)
        min_len = local_min if min_len is None else min(min_len, local_min)
        if local_max <= np.iinfo(np.uint16).max:
            lengths = lengths_u64.astype(np.uint16)
        else:
            lengths = lengths_u64.astype(np.uint32)
        start_parts.append(starts)
        length_parts.append(lengths)
        next_start = int(eos_global[-1]) + 1

    if next_start != n_tokens:
        raise RuntimeError(
            f"{token_path} has trailing tokens after the last EOS={eos_token_id}: "
            f"last_sample_end={next_start}, n_tokens={n_tokens}"
        )
    if not start_parts:
        raise RuntimeError(f"{token_path} contains no EOS={eos_token_id}")

    starts_all = np.concatenate(start_parts).astype(np.uint64, copy=False)
    if max_len <= np.iinfo(np.uint16).max:
        lengths_all = np.concatenate([x.astype(np.uint16, copy=False) for x in length_parts])
    else:
        lengths_all = np.concatenate([x.astype(np.uint32, copy=False) for x in length_parts])
    sample_count = int(lengths_all.shape[0])
    token_count = int(np.asarray(lengths_all, dtype=np.uint64).sum())
    if token_count != n_tokens:
        raise RuntimeError(f"index token count mismatch for {token_path}: index={token_count}, file={n_tokens}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(starts_path, starts_all)
    np.save(lengths_path, lengths_all)
    meta = {
        "source_path": str(token_path),
        "starts_path": str(starts_path),
        "lengths_path": str(lengths_path),
        "eos_token_id": int(eos_token_id),
        "token_count": n_tokens,
        "sample_count": sample_count,
        "min_sample_len": int(min_len if min_len is not None else 0),
        "max_sample_len": int(max_len),
        "length_dtype": str(lengths_all.dtype),
        "elapsed_sec": time.time() - started,
        "status": "built",
    }
    write_json(meta_path, meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build EOS-delimited sample indexes for uint16 token stream files.")
    ap.add_argument("--token-files", nargs="+", required=True, help="Glob(s) or paths for .u16 token files.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eos-token-id", type=int, default=2)
    ap.add_argument("--chunk-tokens", type=int, default=16_777_216)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    token_paths = expand_paths(args.token_files)
    all_meta = []
    started = time.time()
    for i, path in enumerate(token_paths, 1):
        meta = scan_one_file(path, args.out_dir, args.eos_token_id, args.chunk_tokens, args.force)
        all_meta.append(meta)
        print(json.dumps({
            "event": "indexed_file",
            "file_idx": i,
            "num_files": len(token_paths),
            "path": str(path),
            "status": meta.get("status"),
            "samples": meta.get("sample_count"),
            "tokens": meta.get("token_count"),
            "max_sample_len": meta.get("max_sample_len"),
            "elapsed_sec_file": meta.get("elapsed_sec"),
            "elapsed_sec_total": time.time() - started,
        }, ensure_ascii=False), flush=True)

    manifest = {
        "token_files": [str(p) for p in token_paths],
        "index_dir": str(args.out_dir),
        "eos_token_id": int(args.eos_token_id),
        "num_files": len(token_paths),
        "sample_count": int(sum(int(x.get("sample_count", 0)) for x in all_meta)),
        "token_count": int(sum(int(x.get("token_count", 0)) for x in all_meta)),
        "max_sample_len": int(max(int(x.get("max_sample_len", 0)) for x in all_meta)),
        "elapsed_sec": time.time() - started,
        "files": all_meta,
    }
    write_json(args.out_dir / "index_manifest.json", manifest)
    print(json.dumps({"event": "done", **{k: v for k, v in manifest.items() if k != "files"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
