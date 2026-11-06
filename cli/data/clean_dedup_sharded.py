#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterator

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from cli.data.clean_dedup_audit import (  # noqa: E402
    REQUIRED_FIELDS,
    bin_heavy,
    bin_length,
    bytes_label,
    derived_output_path,
    find_zstd,
    load_rdkit,
    load_yaml,
    row_size,
    sha1_text,
    utc_now,
    validate_row,
)


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_CFG: dict[str, Any] = {}
_DOMAIN = ""
_CHEM: Any = None


def log(message: str) -> None:
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def bucket_for(key: str, buckets: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def is_zstd_file(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(4) == ZSTD_MAGIC


def iter_jsonl_lines_auto(path: Path) -> Iterator[bytes]:
    if path.suffix == ".zst" and is_zstd_file(path):
        zstd = find_zstd()
        if zstd is None:
            raise RuntimeError(f"zstd-compressed file requires zstd executable: {path}")
        proc = subprocess.Popen([zstd, "-dc", str(path)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if line.strip():
                    yield line if line.endswith(b"\n") else line + b"\n"
        finally:
            proc.stdout.close()
            rc = proc.wait()
            if rc not in {0, -13}:
                raise RuntimeError(f"zstd -dc failed for {path} rc={rc}")
    else:
        with path.open("rb") as f:
            for line in f:
                if line.strip():
                    yield line if line.endswith(b"\n") else line + b"\n"


def zstd_writer(path: Path, threads: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    zstd = find_zstd()
    if zstd is None:
        raise RuntimeError(f"zstd executable is required for compressed output: {path}")
    level = os.environ.get("MOLSCALING_ZSTD_LEVEL", "1")
    proc = subprocess.Popen([zstd, f"-T{threads}", f"-{level}", "-f", "-o", str(path)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    return proc, proc.stdin


def finish_writer(proc: Any, stream: Any) -> None:
    stream.close()
    if proc is not None:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"zstd failed rc={rc}")


def init_validate_worker(cfg: dict[str, Any], domain: str) -> None:
    global _CFG, _DOMAIN, _CHEM
    _CFG = cfg
    _DOMAIN = domain
    _CHEM = load_rdkit() if domain == "small_molecule" and cfg.get("checks", {}).get("rdkit_small_molecule", True) else None


def iter_line_chunks(path: Path, max_records: int | None, chunk_size: int) -> Iterator[list[bytes]]:
    batch: list[bytes] = []
    for i, line in enumerate(iter_jsonl_lines_auto(path), 1):
        if max_records is not None and i > max_records:
            break
        batch.append(line)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_chunk(lines: list[bytes]) -> tuple[dict[str, int], int, int, list[tuple[bytes, str]]]:
    rejects: Counter[str] = Counter()
    valid: list[tuple[bytes, str]] = []
    total_size = 0
    for line in lines:
        total_size += len(line)
        try:
            row = json.loads(line)
        except Exception:
            rejects["json_decode_error"] += 1
            continue
        ok, reason = validate_row(row, _DOMAIN, _CFG, _CHEM)
        if not ok:
            rejects[reason] += 1
            continue
        valid.append((line, str(row["sha1"])))
    return dict(rejects), len(lines), total_size, valid


class BucketWriters:
    def __init__(self, input_dir: Path, buckets: int, zstd_threads: int) -> None:
        self.items: list[tuple[Any, Any]] = []
        input_dir.mkdir(parents=True, exist_ok=True)
        for bucket in range(buckets):
            path = input_dir / f"bucket-{bucket:05d}.jsonl.zst"
            self.items.append(zstd_writer(path, zstd_threads))

    def write(self, bucket: int, line: bytes) -> None:
        self.items[bucket][1].write(line)

    def close(self) -> None:
        first_error: BaseException | None = None
        for proc, stream in self.items:
            try:
                finish_writer(proc, stream)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def terminate(self) -> None:
        for proc, stream in self.items:
            try:
                stream.close()
            except Exception:
                pass
            if proc is not None and proc.poll() is None:
                proc.terminate()


def initial_stats(item: dict[str, Any], root: Path, output_root: str) -> dict[str, Any]:
    input_rel = str(item["path"])
    domain = str(item["domain"])
    return {
        "input": str(root / input_rel),
        "output": str(derived_output_path(root, output_root, input_rel, domain)),
        "domain": domain,
        "exists": (root / input_rel).exists(),
        "records_seen": 0,
        "records_kept": 0,
        "input_uncompressed_bytes": 0,
        "output_uncompressed_bytes": 0,
        "output_compressed_bytes": 0,
        "reject_reasons": {},
        "length_bins": {},
        "heavy_atom_bins": {},
        "sources": {},
        "duplicate_examples": [],
    }


def spool_input(
    item: dict[str, Any],
    input_index: int,
    root: Path,
    spool_root: Path,
    cfg: dict[str, Any],
    buckets: int,
    max_records: int | None,
    workers: int,
    chunk_size: int,
    zstd_threads: int,
    output_root: str,
) -> dict[str, Any]:
    input_rel = str(item["path"])
    domain = str(item["domain"])
    in_path = root / input_rel
    stats = initial_stats(item, root, output_root)
    rejects: Counter[str] = Counter()
    if not in_path.exists():
        rejects["missing_input"] += 1
        stats["reject_reasons"] = dict(rejects)
        log(f"sharded clean spool skip missing input={in_path}")
        return stats

    input_dir = spool_root / f"input-{input_index:03d}"
    writers = BucketWriters(input_dir, buckets, zstd_threads)
    log(f"sharded clean spool start input={in_path} domain={domain} workers={workers} buckets={buckets}")
    try:
        ctx = get_context("fork")
        with ctx.Pool(
            processes=workers,
            initializer=init_validate_worker,
            initargs=(cfg, domain),
            maxtasksperchild=300,
        ) as pool:
            next_log = 1_000_000
            for chunk_rejects, rows_seen, input_bytes, valid_rows in pool.imap_unordered(
                validate_chunk,
                iter_line_chunks(in_path, max_records, chunk_size),
                chunksize=1,
            ):
                rejects.update(chunk_rejects)
                stats["records_seen"] += rows_seen
                stats["input_uncompressed_bytes"] += input_bytes
                for line, key in valid_rows:
                    writers.write(bucket_for(key, buckets), line)
                if int(stats["records_seen"]) >= next_log:
                    log(
                        f"sharded clean spool progress input={in_path.name} "
                        f"seen={stats['records_seen']} valid_spooled~={stats['records_seen'] - sum(rejects.values())}"
                    )
                    next_log += 1_000_000
        writers.close()
    except BaseException:
        writers.terminate()
        raise
    stats["reject_reasons"] = dict(rejects)
    log(
        f"sharded clean spool done input={in_path} "
        f"seen={stats['records_seen']} input_uncompressed={bytes_label(int(stats['input_uncompressed_bytes']))}"
    )
    return stats


def dedup_bucket(task: tuple[int, list[dict[str, Any]], str, str, int, int]) -> dict[str, Any]:
    bucket, inputs, spool_root_s, chunk_root_s, buckets, zstd_threads = task
    spool_root = Path(spool_root_s)
    chunk_root = Path(chunk_root_s)
    owner: dict[str, int] = {}
    per_input: dict[int, dict[str, Any]] = {}
    writers: dict[int, tuple[Any, Any, Path]] = {}
    try:
        for input_index, item in enumerate(inputs):
            bucket_path = spool_root / f"input-{input_index:03d}" / f"bucket-{bucket:05d}.jsonl.zst"
            if not bucket_path.exists() or bucket_path.stat().st_size == 0:
                continue
            input_stats = per_input.setdefault(input_index, {
                "records_kept": 0,
                "output_uncompressed_bytes": 0,
                "reject_reasons": Counter(),
                "length_bins": Counter(),
                "heavy_atom_bins": Counter(),
                "sources": Counter(),
                "duplicate_examples": [],
                "chunks": [],
            })
            for line in iter_jsonl_lines_auto(bucket_path):
                row = json.loads(line)
                key = str(row["sha1"])
                first_index = owner.get(key)
                if first_index is not None:
                    reason = "duplicate_within_input" if first_index == input_index else "duplicate_cross_source"
                    input_stats["reject_reasons"][reason] += 1
                    examples = input_stats["duplicate_examples"]
                    if isinstance(examples, list) and len(examples) < 5:
                        examples.append({
                            "sha1": key,
                            "current_input": str(item["path"]),
                            "first_input": str(inputs[first_index]["path"]),
                            "domain": str(item["domain"]),
                            "source": row.get("source"),
                        })
                    continue
                owner[key] = input_index
                writer = writers.get(input_index)
                if writer is None:
                    chunk_path = chunk_root / f"input-{input_index:03d}" / f"bucket-{bucket:05d}.jsonl.zst"
                    proc, stream = zstd_writer(chunk_path, zstd_threads)
                    writer = (proc, stream, chunk_path)
                    writers[input_index] = writer
                    input_stats["chunks"].append(str(chunk_path))
                proc, stream, chunk_path = writer
                stream.write(line)
                input_stats["records_kept"] += 1
                input_stats["output_uncompressed_bytes"] += len(line)
                input_stats["length_bins"][bin_length(int(row["length"]))] += 1
                input_stats["heavy_atom_bins"][bin_heavy(int(row["estimated_heavy_atoms"]))] += 1
                input_stats["sources"][str(row.get("source", "unknown"))] += 1
    finally:
        for proc, stream, _ in writers.values():
            finish_writer(proc, stream)

    compact: dict[str, Any] = {"bucket": bucket, "unique_keys": len(owner), "inputs": {}}
    for input_index, stats in per_input.items():
        compact["inputs"][str(input_index)] = {
            "records_kept": int(stats["records_kept"]),
            "output_uncompressed_bytes": int(stats["output_uncompressed_bytes"]),
            "reject_reasons": dict(stats["reject_reasons"]),
            "length_bins": dict(stats["length_bins"]),
            "heavy_atom_bins": dict(stats["heavy_atom_bins"]),
            "sources": dict(stats["sources"]),
            "duplicate_examples": stats["duplicate_examples"],
            "chunks": stats["chunks"],
        }
    return compact


def merge_zstd_frames(chunks: list[Path], out_path: Path, overwrite: bool) -> int:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    total = 0
    with tmp_path.open("wb") as out:
        for chunk in sorted(chunks):
            if not chunk.exists() or chunk.stat().st_size == 0:
                continue
            with chunk.open("rb") as f:
                shutil.copyfileobj(f, out, length=16 * 1024 * 1024)
            total += chunk.stat().st_size
    tmp_path.replace(out_path)
    return total


def add_bucket_result(
    results: list[dict[str, Any]],
    bucket_result: dict[str, Any],
) -> dict[int, list[Path]]:
    chunks_by_input: dict[int, list[Path]] = defaultdict(list)
    for input_index_s, stats in (bucket_result.get("inputs") or {}).items():
        input_index = int(input_index_s)
        result = results[input_index]
        result["records_kept"] = int(result.get("records_kept") or 0) + int(stats.get("records_kept") or 0)
        result["output_uncompressed_bytes"] = (
            int(result.get("output_uncompressed_bytes") or 0) + int(stats.get("output_uncompressed_bytes") or 0)
        )
        for key in ("reject_reasons", "length_bins", "heavy_atom_bins", "sources"):
            merged = Counter(result.get(key) or {})
            merged.update(stats.get(key) or {})
            result[key] = dict(merged)
        examples = result.get("duplicate_examples")
        if isinstance(examples, list):
            for example in stats.get("duplicate_examples") or []:
                if len(examples) >= 20:
                    break
                examples.append(example)
        for chunk in stats.get("chunks") or []:
            chunks_by_input[input_index].append(Path(str(chunk)))
    return chunks_by_input


def summarize_report(
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    stamp: str,
    spool_root: Path,
) -> dict[str, Any]:
    totals = Counter()
    domains: dict[str, Counter[str]] = defaultdict(Counter)
    reject_reasons = Counter()
    for item in results:
        domain = str(item["domain"])
        totals["records_seen"] += int(item.get("records_seen") or 0)
        totals["records_kept"] += int(item.get("records_kept") or 0)
        totals["input_uncompressed_bytes"] += int(item.get("input_uncompressed_bytes") or 0)
        totals["output_uncompressed_bytes"] += int(item.get("output_uncompressed_bytes") or 0)
        totals["output_compressed_bytes"] += int(item.get("output_compressed_bytes") or 0)
        domains[domain]["records_seen"] += int(item.get("records_seen") or 0)
        domains[domain]["records_kept"] += int(item.get("records_kept") or 0)
        domains[domain]["output_uncompressed_bytes"] += int(item.get("output_uncompressed_bytes") or 0)
        domains[domain]["output_compressed_bytes"] += int(item.get("output_compressed_bytes") or 0)
        for reason, count in (item.get("reject_reasons") or {}).items():
            reject_reasons[str(reason)] += int(count)
            domains[domain][f"reject:{reason}"] += int(count)
    return {
        "created_utc": utc_now(),
        "stamp": stamp,
        "config": str(args.config),
        "root": str(Path(cfg["root"]).resolve()),
        "output_root": str(args.output_root or cfg.get("output_root") or "clean_sharded"),
        "max_records_per_input": args.max_records,
        "dedup_backend": "sha1_hash_bucket_sharded_in_memory",
        "sharded_buckets": args.buckets,
        "sharded_workers": args.workers,
        "chunk_size": args.chunk_size,
        "spool_root": str(spool_root),
        "totals": dict(totals),
        "reject_reasons": dict(reject_reasons),
        "domains": {k: dict(v) for k, v in sorted(domains.items())},
        "inputs": results,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    totals = report.get("totals") or {}
    lines = [
        "# Graph 前分片并行清洗去重审计",
        "",
        f"生成时间：{report.get('created_utc')}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 输入记录 | {int(totals.get('records_seen') or 0):,} |",
        f"| 清洗去重后保留记录 | {int(totals.get('records_kept') or 0):,} |",
        f"| 输入未压缩规模 | {bytes_label(int(totals.get('input_uncompressed_bytes') or 0))} |",
        f"| 输出未压缩规模 | {bytes_label(int(totals.get('output_uncompressed_bytes') or 0))} |",
        f"| 输出压缩文件占用 | {bytes_label(int(totals.get('output_compressed_bytes') or 0))} |",
        f"| hash 分桶数 | {int(report.get('sharded_buckets') or 0):,} |",
        f"| 并行 worker | {int(report.get('sharded_workers') or 0):,} |",
        "",
        "## 按领域统计",
        "",
        "| 领域 | 输入记录 | 保留记录 | 输出未压缩 | 输出压缩占用 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for domain, stats in sorted((report.get("domains") or {}).items()):
        lines.append(
            f"| {domain} | {int(stats.get('records_seen') or 0):,} | {int(stats.get('records_kept') or 0):,} | "
            f"{bytes_label(int(stats.get('output_uncompressed_bytes') or 0))} | "
            f"{bytes_label(int(stats.get('output_compressed_bytes') or 0))} |"
        )
    lines.extend(["", "## 过滤和去重原因", "", "| 原因 | 数量 |", "| --- | ---: |"])
    for reason, count in sorted((report.get("reject_reasons") or {}).items()):
        lines.append(f"| {reason} | {int(count):,} |")
    lines.extend([
        "",
        "## 输入文件明细",
        "",
        "| 输入 | 输出 | 领域 | 输入记录 | 保留记录 | 输出压缩占用 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for item in report.get("inputs") or []:
        lines.append(
            f"| `{item.get('input')}` | `{item.get('output')}` | {item.get('domain')} | "
            f"{int(item.get('records_seen') or 0):,} | {int(item.get('records_kept') or 0):,} | "
            f"{bytes_label(int(item.get('output_compressed_bytes') or 0))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sharded parallel clean, validate, and globally deduplicate molscaling JSONL.zst files.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append", help="Only process input paths whose basename or path contains this value.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(64, os.cpu_count() or 2)))
    parser.add_argument("--buckets", type=int, default=max(1, min(128, os.cpu_count() or 2)))
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--spool-dir", type=Path, default=Path(tempfile.gettempdir()) / "moltoken_clean_sharded")
    parser.add_argument("--zstd-threads", type=int, default=int(os.environ.get("MOLSCALING_SHARDED_CLEAN_ZSTD_THREADS", "1")))
    parser.add_argument("--keep-spool", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["root"]).resolve()
    output_root = str(args.output_root or cfg.get("output_root") or "clean_sharded")
    lock_path = root / "reports" / f"clean_dedup_sharded_{output_root}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another clean_dedup_sharded is already running lock={lock_path}", flush=True)
        return 75

    inputs = list(cfg.get("inputs") or [])
    if args.source:
        wanted = list(args.source)
        inputs = [
            item for item in inputs
            if any(token in str(item.get("path", "")) or token in Path(str(item.get("path", ""))).name for token in wanted)
        ]

    missing_inputs = [str(root / str(item["path"])) for item in inputs if not (root / str(item["path"])).exists()]
    if missing_inputs and not args.allow_missing:
        print("clean_dedup_sharded refusing to run because configured inputs are missing:", flush=True)
        for path in missing_inputs:
            print(f"  missing input={path}", flush=True)
        return 2

    for item in inputs:
        input_rel = str(item["path"])
        domain = str(item["domain"])
        out_path = derived_output_path(root, output_root, input_rel, domain)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.spool_dir.resolve() / f"clean_sharded_{output_root}_{stamp}_{os.getpid()}"
    spool_root = run_root / "spool"
    chunk_root = run_root / "chunks"
    spool_root.mkdir(parents=True, exist_ok=True)
    chunk_root.mkdir(parents=True, exist_ok=True)

    log(
        f"sharded clean plan inputs={len(inputs)} workers={args.workers} "
        f"buckets={args.buckets} chunk_size={args.chunk_size} spool={run_root}"
    )
    results = [
        spool_input(
            item,
            input_index,
            root,
            spool_root,
            cfg,
            args.buckets,
            args.max_records,
            args.workers,
            args.chunk_size,
            args.zstd_threads,
            output_root,
        )
        for input_index, item in enumerate(inputs)
    ]

    tasks = [
        (bucket, inputs, str(spool_root), str(chunk_root), args.buckets, args.zstd_threads)
        for bucket in range(args.buckets)
    ]
    all_chunks: dict[int, list[Path]] = defaultdict(list)
    log(f"sharded clean dedup start buckets={args.buckets} workers={args.workers}")
    ctx = get_context("fork")
    with ctx.Pool(processes=args.workers, maxtasksperchild=20) as pool:
        completed = 0
        for bucket_result in pool.imap_unordered(dedup_bucket, tasks, chunksize=1):
            completed += 1
            chunks = add_bucket_result(results, bucket_result)
            for input_index, paths in chunks.items():
                all_chunks[input_index].extend(paths)
            if completed % 4 == 0 or completed == args.buckets:
                kept = sum(int(item.get("records_kept") or 0) for item in results)
                log(f"sharded clean dedup progress buckets_done={completed}/{args.buckets} kept={kept}")

    log("sharded clean merge start")
    for input_index, item in enumerate(inputs):
        input_rel = str(item["path"])
        domain = str(item["domain"])
        out_path = derived_output_path(root, output_root, input_rel, domain)
        compressed = merge_zstd_frames(all_chunks.get(input_index, []), out_path, args.overwrite)
        results[input_index]["output_compressed_bytes"] = compressed
        log(f"sharded clean merge done input={Path(input_rel).name} out={out_path} compressed={bytes_label(compressed)}")

    report = summarize_report(results, cfg, args, stamp, run_root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_report = reports / f"clean_dedup_sharded_{output_root}_{stamp}.json"
    md_report = reports / f"clean_dedup_sharded_{output_root}_{stamp}.md"
    latest_json = reports / f"clean_dedup_sharded_{output_root}.latest.json"
    latest_md = reports / f"clean_dedup_sharded_{output_root}.latest.md"
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    json_report.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    write_markdown(report, md_report)
    write_markdown(report, latest_md)
    print(text, end="")

    if not args.keep_spool:
        shutil.rmtree(run_root, ignore_errors=True)
        log(f"sharded clean removed spool={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
