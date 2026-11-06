#!/usr/bin/env python3
"""把 FDDFS compact-int token 文件重分片成 BPE worker 输入。

输入来自 ``serialize_fddfs_full.py`` 生成的 manifest。每个源文件已经是
YTTMIBPE compact-int 二进制，因此这里不解码 token，只按样本边界复制
``uint32 length + int32 token payload``，避免额外的 Python token 循环。
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import BinaryIO


MAGIC = b"YTTMIBPE"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIQQi")
LEN_STRUCT = struct.Struct("<I")
CACHE_VERSION = 1
CACHE_WAIT_TIMEOUT_SEC = 7200
CACHE_WAIT_POLL_SEC = 5


def log(message: str) -> None:
    """打印带时间戳的进度。"""

    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)


def read_exact(f: BinaryIO, nbytes: int, what: str) -> bytes:
    """读取固定字节数；文件损坏或截断时直接报错。"""

    data = f.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"unexpected EOF while reading {what}")
    return data


def read_header(f: BinaryIO, path: Path) -> dict[str, int]:
    """读取并校验 YTTMIBPE 文件头。"""

    magic, version, nseq, ntokens, max_token = HEADER_STRUCT.unpack(read_exact(f, HEADER_STRUCT.size, "header"))
    if magic != MAGIC:
        raise ValueError(f"bad compact-int magic: {path}")
    if int(version) != VERSION:
        raise ValueError(f"unsupported compact-int version={version}: {path}")
    return {"nseq": int(nseq), "ntokens": int(ntokens), "max_token": int(max_token)}


def write_header(f: BinaryIO, nseq: int, ntokens: int, max_token: int) -> None:
    """写 YTTMIBPE 文件头。"""

    f.seek(0)
    f.write(HEADER_STRUCT.pack(MAGIC, VERSION, int(nseq), int(ntokens), int(max_token)))


def infer_summary_path(token_path: Path, out_dir: Path, serializer: str) -> Path:
    """从 token 文件路径推断对应 summary JSON 路径。"""

    token_root = out_dir / "tokens"
    summary_root = out_dir / "summaries"
    try:
        rel = token_path.relative_to(token_root)
    except ValueError as exc:
        raise ValueError(f"token path is not under manifest token root: {token_path}") from exc
    name = rel.name
    suffix = f".{serializer}.bin"
    if not name.endswith(suffix):
        raise ValueError(f"unexpected {serializer} token filename: {token_path}")
    rel = rel.with_name(name[: -len(suffix)] + ".json")
    return summary_root / rel


def load_file_items(manifest_path: Path) -> list[dict[str, object]]:
    """读取全量 manifest，并补齐每个 token 文件的 summary 统计。"""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached = load_file_items_cache(manifest_path, manifest)
    if cached is not None:
        return cached
    return build_file_items_cache(manifest_path, manifest)


def cache_path_for_manifest(manifest_path: Path) -> Path:
    """返回当前 manifest 对应的文件统计缓存路径。"""

    return manifest_path.with_suffix(".file_items_cache.json")


def cache_meta_for_manifest(manifest_path: Path, manifest: dict) -> dict[str, object]:
    """生成用于判断缓存是否匹配当前 manifest 的元信息。"""

    stat = manifest_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "manifest": str(manifest_path),
        "manifest_size": int(stat.st_size),
        "manifest_mtime_ns": int(stat.st_mtime_ns),
        "serializer": str(manifest.get("serializer") or "fddfs").lower(),
        "token_files_count": int(len(manifest.get("token_files") or [])),
    }


def cache_matches(payload: dict, expected_meta: dict[str, object]) -> bool:
    """判断缓存文件是否来自同一份 manifest。"""

    meta = payload.get("meta") or {}
    return all(meta.get(key) == value for key, value in expected_meta.items())


def read_file_items_cache(cache_path: Path, expected_meta: dict[str, object]) -> list[dict[str, object]] | None:
    """读取可复用的文件统计缓存；缓存不匹配时返回 None。"""

    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log(f"ignore unreadable file item cache path={cache_path} error={exc}")
        return None
    if not cache_matches(payload, expected_meta):
        log(f"ignore stale file item cache path={cache_path}")
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        log(f"ignore malformed file item cache path={cache_path}")
        return None
    log(f"file item cache hit path={cache_path} items={len(items)}")
    return items


def load_file_items_cache(manifest_path: Path, manifest: dict) -> list[dict[str, object]] | None:
    """优先从共享缓存读取 token 文件统计；其他 rank 会等待正在生成的缓存。"""

    cache_path = cache_path_for_manifest(manifest_path)
    expected_meta = cache_meta_for_manifest(manifest_path, manifest)
    cached = read_file_items_cache(cache_path, expected_meta)
    if cached is not None:
        return cached

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    try:
        os.mkdir(lock_path)
    except FileExistsError:
        start = time.time()
        log(f"waiting file item cache path={cache_path}")
        while time.time() - start < CACHE_WAIT_TIMEOUT_SEC:
            cached = read_file_items_cache(cache_path, expected_meta)
            if cached is not None:
                return cached
            if not lock_path.exists():
                break
            time.sleep(CACHE_WAIT_POLL_SEC)
        log(f"file item cache wait expired; building locally path={cache_path}")
        return None
    except OSError as exc:
        log(f"file item cache lock unavailable path={lock_path} error={exc}; building locally")
        return None
    return None


def build_file_items_cache(manifest_path: Path, manifest: dict) -> list[dict[str, object]]:
    """读取 summary 生成文件统计，并在持有锁时写入共享缓存。"""

    cache_path = cache_path_for_manifest(manifest_path)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    expected_meta = cache_meta_for_manifest(manifest_path, manifest)
    out_dir = Path(str(manifest["out_dir"]))
    serializer = str(manifest.get("serializer") or "fddfs").lower()
    raw_files = list(manifest.get("token_files") or [])
    if not raw_files:
        raise ValueError(f"manifest has no token_files: {manifest_path}")

    items: list[dict[str, object]] = []
    start = time.time()
    for raw in raw_files:
        token_path = Path(str(raw))
        summary_path = infer_summary_path(token_path, out_dir, serializer)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        items.append(
            {
                "path": str(token_path),
                "summary": str(summary_path),
                "domain": str(summary.get("domain") or "unknown"),
                "records": int(summary.get("records") or 0),
                "tokens": int(summary.get("tokens") or 0),
                "max_token": int(summary.get("max_token") or manifest.get("max_token") or -1),
                "max_sample_tokens": int(summary.get("max_sample_tokens") or 0),
            }
        )
        if len(items) % 10000 == 0:
            log(f"file item cache build progress items={len(items)}/{len(raw_files)} elapsed_sec={time.time() - start:.1f}")
    if lock_path.exists():
        payload = {"meta": expected_meta, "items": items}
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        tmp_path.replace(cache_path)
        try:
            os.rmdir(lock_path)
        except OSError as exc:
            log(f"warning could not remove file item cache lock path={lock_path} error={exc}")
        log(f"file item cache written path={cache_path} items={len(items)} elapsed_sec={time.time() - start:.1f}")
    return items


def select_files(items: list[dict[str, object]], token_budget: int, seed: int) -> tuple[list[dict[str, object]], int]:
    """按 token budget 选择输入文件；token_budget <= 0 表示全量。"""

    files = list(items)
    rng = random.Random(seed)
    rng.shuffle(files)
    if token_budget <= 0:
        return files, sum(int(item["tokens"]) for item in files)
    selected: list[dict[str, object]] = []
    selected_tokens = 0
    for item in files:
        selected.append(item)
        selected_tokens += int(item["tokens"])
        if selected_tokens >= token_budget:
            break
    return selected, selected_tokens


def assign_files(files: list[dict[str, object]], num_shards: int) -> tuple[list[list[dict[str, object]]], list[int]]:
    """按文件 token 数做 token-balanced 分配。"""

    loads = [(0, i) for i in range(num_shards)]
    heapq.heapify(loads)
    assigned: list[list[dict[str, object]]] = [[] for _ in range(num_shards)]
    assigned_tokens = [0 for _ in range(num_shards)]
    for item in sorted(files, key=lambda x: int(x["tokens"]), reverse=True):
        load, shard = heapq.heappop(loads)
        assigned[shard].append(item)
        tokens = int(item["tokens"])
        assigned_tokens[shard] += tokens
        heapq.heappush(loads, (load + tokens, shard))
    for shard_files in assigned:
        shard_files.sort(key=lambda x: str(x["path"]))
    return assigned, assigned_tokens


def copy_file_records(src_path: Path, out: BinaryIO) -> dict[str, int]:
    """把一个 compact-int 文件里的所有样本复制到当前输出 shard。"""

    records = 0
    tokens = 0
    max_sample_tokens = 0
    with src_path.open("rb") as src:
        header = read_header(src, src_path)
        local_tokens = 0
        for idx in range(int(header["nseq"])):
            raw_len = read_exact(src, LEN_STRUCT.size, f"sequence length #{idx} in {src_path}")
            (length,) = LEN_STRUCT.unpack(raw_len)
            payload = read_exact(src, int(length) * 4, f"sequence payload #{idx} in {src_path}")
            out.write(raw_len)
            out.write(payload)
            records += 1
            tokens += int(length)
            local_tokens += int(length)
            if int(length) > max_sample_tokens:
                max_sample_tokens = int(length)
        if local_tokens != int(header["ntokens"]):
            raise ValueError(
                f"token mismatch in {src_path}: header={header['ntokens']} observed={local_tokens}"
            )
    return {"records": records, "tokens": tokens, "max_sample_tokens": max_sample_tokens}


def write_records_to_shard(path: Path, files: list[dict[str, object]], args: argparse.Namespace, rank: int | None = None) -> dict:
    """把分配给一个 rank 的源文件复制成一个 compact-int shard。"""

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    records = 0
    tokens = 0
    max_token = int(args.max_token) if args.max_token is not None else -1
    max_sample_tokens = 0
    domain_records: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    start = time.time()
    with tmp.open("w+b") as out:
        write_header(out, 0, 0, max_token)
        for file_idx, item in enumerate(files, 1):
            src_path = Path(str(item["path"]))
            copied = copy_file_records(src_path, out)
            records += int(copied["records"])
            tokens += int(copied["tokens"])
            max_sample_tokens = max(max_sample_tokens, int(copied["max_sample_tokens"]), int(item["max_sample_tokens"]))
            if args.max_token is None:
                max_token = max(max_token, int(item["max_token"]))
            domain = str(item["domain"])
            domain_records[domain] += int(copied["records"])
            domain_tokens[domain] += int(copied["tokens"])
            if args.log_every and file_idx % int(args.log_every) == 0:
                rank_text = f" rank={rank}" if rank is not None else ""
                log(
                    f"fddfs shard progress{rank_text} files={file_idx}/{len(files)} "
                    f"records={records} tokens={tokens} tokens_b={tokens / 1e9:.3f} "
                    f"elapsed_sec={time.time() - start:.1f}"
                )
        write_header(out, records, tokens, max_token)
    tmp.replace(path)
    return {
        "path": str(path),
        "records": int(records),
        "tokens": int(tokens),
        "tokens_b": float(tokens) / 1e9,
        "max_token": int(max_token),
        "max_sample_tokens": int(max_sample_tokens),
        "domain_records": dict(sorted(domain_records.items())),
        "domain_tokens": dict(sorted(domain_tokens.items())),
        "files_read": int(len(files)),
        "elapsed_sec": time.time() - start,
    }


def build_rank_shard(args: argparse.Namespace) -> dict:
    """构建当前 rank 的一个 shard。"""

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = int(args.num_shards)
    rank = int(args.rank)
    if rank < 0 or rank >= num_shards:
        raise ValueError("--rank must be in [0, num_shards)")
    shard_path = out_dir / f"shard_{rank:05d}.bin"
    done_path = Path(args.summary) if args.summary else out_dir / f"rank_{rank:05d}.done.json"
    if done_path.exists() and shard_path.exists() and shard_path.stat().st_size > 0 and not args.overwrite:
        log(f"existing rank shard found rank={rank} path={shard_path}")
        return json.loads(done_path.read_text(encoding="utf-8"))

    items = load_file_items(Path(args.manifest))
    selected, selected_tokens = select_files(items, int(args.token_budget), int(args.seed))
    assigned, assigned_tokens = assign_files(selected, num_shards)
    rank_files = assigned[rank]
    log(
        f"fddfs rank shard start rank={rank}/{num_shards} selected_files={len(selected)} "
        f"selected_tokens={selected_tokens} assigned_files={len(rank_files)} "
        f"assigned_tokens={assigned_tokens[rank]}"
    )
    shard_summary = write_records_to_shard(shard_path, rank_files, args, rank=rank)
    summary = {
        "mode": "fddfs_rank_local",
        "manifest": str(args.manifest),
        "out_dir": str(out_dir),
        "rank": int(rank),
        "num_shards": int(num_shards),
        "token_budget": int(args.token_budget),
        "seed": int(args.seed),
        "selected_files": int(len(selected)),
        "selected_manifest_tokens": int(selected_tokens),
        "assigned_manifest_tokens": int(assigned_tokens[rank]),
        **shard_summary,
    }
    tmp_summary = done_path.with_suffix(done_path.suffix + ".tmp")
    tmp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_summary.replace(done_path)
    log(f"fddfs rank shard done rank={rank} tokens={summary['tokens']} elapsed_sec={summary['elapsed_sec']:.1f}")
    return summary


def aggregate_rank_shards(args: argparse.Namespace) -> dict:
    """聚合所有 rank-local shard summary。"""

    out_dir = Path(args.out_dir)
    num_shards = int(args.num_shards)
    done_path = Path(args.summary) if args.summary else out_dir / "done.json"
    rank_summaries = []
    for rank in range(num_shards):
        path = out_dir / f"rank_{rank:05d}.done.json"
        if not path.exists():
            raise FileNotFoundError(f"missing rank summary: {path}")
        rank_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    total_records = sum(int(s.get("records") or 0) for s in rank_summaries)
    total_tokens = sum(int(s.get("tokens") or 0) for s in rank_summaries)
    domain_records: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    for summary in rank_summaries:
        domain_records.update({k: int(v) for k, v in summary.get("domain_records", {}).items()})
        domain_tokens.update({k: int(v) for k, v in summary.get("domain_tokens", {}).items()})
    summary = {
        "mode": "fddfs_rank_local_aggregate",
        "manifest": str(args.manifest),
        "out_dir": str(out_dir),
        "num_shards": int(num_shards),
        "token_budget": int(args.token_budget),
        "seed": int(args.seed),
        "records": int(total_records),
        "tokens": int(total_tokens),
        "tokens_b": float(total_tokens) / 1e9,
        "max_token": max(int(s.get("max_token") or -1) for s in rank_summaries),
        "max_sample_tokens": max(int(s.get("max_sample_tokens") or 0) for s in rank_summaries),
        "domain_records": dict(sorted(domain_records.items())),
        "domain_tokens": dict(sorted(domain_tokens.items())),
        "shards": [
            {
                "rank": int(s["rank"]),
                "path": str(s["path"]),
                "records": int(s["records"]),
                "tokens": int(s["tokens"]),
                "tokens_b": float(s["tokens"]) / 1e9,
                "size_gb": Path(str(s["path"])).stat().st_size / (1024**3),
                "elapsed_sec": float(s["elapsed_sec"]),
            }
            for s in rank_summaries
        ],
        "elapsed_sec_max": max(float(s.get("elapsed_sec") or 0.0) for s in rank_summaries),
    }
    tmp_summary = done_path.with_suffix(done_path.suffix + ".tmp")
    tmp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_summary.replace(done_path)
    log(f"fddfs aggregate done out={done_path} tokens={total_tokens} records={total_records}")
    return summary


def main() -> int:
    """命令行入口。"""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--token-budget", type=int, default=0, help="<=0 表示使用 manifest 全量 token 文件。")
    ap.add_argument("--seed", type=int, default=20260525)
    ap.add_argument("--max-token", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=1000, help="按输入文件数打印进度。")
    ap.add_argument("--rank-local", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate_rank_shards(args)
    elif args.rank_local:
        if args.rank is None:
            raise ValueError("--rank is required with --rank-local")
        build_rank_shard(args)
    else:
        raise ValueError("compact-int shard builder currently expects --rank-local or --aggregate-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
