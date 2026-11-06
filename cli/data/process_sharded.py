#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
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

from cli.data.process_all import (  # noqa: E402
    AA_HEAVY_ATOMS,
    DNA_HEAVY_ATOMS,
    RNA_HEAVY_ATOMS,
    annotate_nonstandard,
    canonicalize_smiles,
    find_zstd,
    iter_fasta,
    iter_pubchem_cid_smiles,
    load_yaml,
    output_lock,
    passes_input_sampling,
    passes_source_sampling,
    reject_low_complexity,
    sequence_heavy_atoms,
    sequence_windows,
    sha1_text,
    source_tmp_output,
    split_ints,
    strict_alphabet_filter_enabled,
    utc_now,
    write_source_done,
)


def log(message: str) -> None:
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def bucket_for(text: str, buckets: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def json_line(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"


def row_line(row: dict[str, Any]) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"


def _variant_source(source: dict[str, Any], variant: str) -> dict[str, Any]:
    src = dict(source)
    src["name"] = f"{source['name']}_{variant}"
    out = Path(str(source["output"]))
    if "".join(out.suffixes[-2:]) == ".jsonl.zst":
        stem = out.name[:-len(".jsonl.zst")]
        src["output"] = str(out.with_name(f"{stem}_{variant}.jsonl.zst"))
    else:
        src["output"] = str(out.with_name(f"{out.stem}_{variant}{out.suffix}"))
    src["skip_if_output_exists"] = True
    return src


def _fast_text_reader(path: Path, read_threads: int):
    if path.suffix != ".gz":
        return path.open("rt", encoding="utf-8", errors="replace")
    pigz = shutil.which("pigz")
    if pigz:
        proc = subprocess.Popen(
            [pigz, "-dc", "-p", str(read_threads), str(path)],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1024 * 1024,
        )
        assert proc.stdout is not None
        return _ProcessTextStream(proc)
    gzip_bin = shutil.which("gzip")
    if gzip_bin:
        proc = subprocess.Popen(
            [gzip_bin, "-dc", str(path)],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1024 * 1024,
        )
        assert proc.stdout is not None
        return _ProcessTextStream(proc)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace")


class _ProcessTextStream:
    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc
        self.stream = proc.stdout

    def __iter__(self):
        assert self.stream is not None
        return iter(self.stream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stream is not None:
            self.stream.close()
        rc = self.proc.wait()
        if rc not in {0, -13} and exc_type is None:
            raise RuntimeError(f"input decompressor failed rc={rc}")


def iter_fasta_fast(path: Path, read_threads: int) -> Iterator[tuple[str, str, str]]:
    import re
    from cli.data.process_all import parse_header_id

    header: str | None = None
    seq_parts: list[str] = []
    with _fast_text_reader(path, read_threads) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield parse_header_id(header), header, re.sub(r"\s+", "", "".join(seq_parts)).upper()
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield parse_header_id(header), header, re.sub(r"\s+", "", "".join(seq_parts)).upper()


def iter_pubchem_fast(path: Path, read_threads: int) -> Iterator[tuple[str, str]]:
    with _fast_text_reader(path, read_threads) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                yield parts[0], parts[1]


def zstd_writer(path: Path, threads: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    zstd = find_zstd()
    if zstd is None:
        raise RuntimeError(f"zstd executable is required for compressed output: {path}")
    level = os.environ.get("MOLSCALING_ZSTD_LEVEL", "3")
    proc = subprocess.Popen([zstd, f"-T{threads}", f"-{level}", "-f", "-o", str(path)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    return proc, proc.stdin


def finish_writer(proc: Any, stream: Any) -> None:
    stream.close()
    if proc is not None:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"zstd failed rc={rc}")


def ensure_output_available(root: Path, sources: list[dict[str, Any]], overwrite: bool) -> None:
    for source in sources:
        out_path = root / source["output"]
        done = root / "reports" / f"process_report_{source['name']}.done.json"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing output: {out_path}")
        if done.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing done report: {done}")


def merge_zstd_frames(chunks: list[Path], out_path: Path) -> int:
    tmp_path = source_tmp_output(out_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
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


def spool_pubchem(
    root: Path,
    sources: list[dict[str, Any]],
    spool_dir: Path,
    buckets: int,
    max_records: int | None,
    read_threads: int,
) -> dict[str, Any]:
    if len({src["input"] for src in sources}) != 1:
        raise ValueError("PubChem sharded mode requires one shared input")
    input_path = root / sources[0]["input"]
    spool_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[tuple[int, int], Any] = {}
    counts: dict[str, Counter[str]] = {src["name"]: Counter() for src in sources}
    spooled: Counter[str] = Counter()
    try:
        for source_index, source in enumerate(sources):
            source_dir = spool_dir / source["name"]
            source_dir.mkdir(parents=True, exist_ok=True)
            for bucket in range(buckets):
                handles[(source_index, bucket)] = (source_dir / f"bucket-{bucket:05d}.jsonl").open(
                    "wb", buffering=1024 * 1024
                )
        log(f"sharded spool start kind=pubchem input={input_path} sources={','.join(src['name'] for src in sources)}")
        for seen, (original_id, raw_smiles) in enumerate(iter_pubchem_fast(input_path, read_threads), 1):
            if max_records is not None and seen > max_records:
                break
            key = f"{original_id}|{raw_smiles}"
            bucket = bucket_for(key, buckets)
            payload = json_line([original_id, raw_smiles])
            for source_index, source in enumerate(sources):
                if passes_input_sampling(key, source):
                    handles[(source_index, bucket)].write(payload)
                    spooled[source["name"]] += 1
                else:
                    counts[source["name"]]["input_sampled_out"] += 1
            if seen % 1000000 == 0:
                log(
                    "sharded spool progress kind=pubchem "
                    f"seen={seen} " + " ".join(f"{src['name']}={spooled[src['name']]}" for src in sources)
                )
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "input": str(input_path),
        "counts": {name: dict(counter) for name, counter in counts.items()},
        "spooled": dict(spooled),
    }


def pubchem_worker(task: tuple[dict[str, Any], dict[str, Any], Path, Path, int]) -> dict[str, Any]:
    source, cfg, bucket_path, chunk_path, zstd_threads = task
    counts: Counter[str] = Counter()
    kept = 0
    out_bytes = 0
    proc, stream = zstd_writer(chunk_path, zstd_threads)
    try:
        small_cfg = cfg["small_molecule"]
        with bucket_path.open("rb") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                original_id, raw_smiles = json.loads(raw_line)
                normalized, heavy, status = canonicalize_smiles(str(raw_smiles), small_cfg)
                counts[status] += 1
                if status != "ok" or normalized is None or heavy is None:
                    continue
                row_hash = sha1_text(f"small_molecule|{normalized}")
                if not passes_source_sampling(row_hash, source):
                    counts["sampled_out"] += 1
                    continue
                row = {
                    "id": f"{source['source']}:{original_id}",
                    "domain": "small_molecule",
                    "source": source["source"],
                    "source_version": source["source_version"],
                    "original_id": original_id,
                    "raw": raw_smiles,
                    "normalized": normalized,
                    "length": len(normalized),
                    "estimated_heavy_atoms": heavy,
                    "split_group": normalized,
                    "metadata": {},
                    "filter_status": "ok",
                    "sha1": row_hash,
                }
                line = row_line(row)
                stream.write(line)
                out_bytes += len(line)
                kept += 1
        finish_writer(proc, stream)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        raise
    return {
        "source_name": source["name"],
        "bucket": bucket_path.name,
        "chunk": str(chunk_path),
        "counts": dict(counts),
        "kept": kept,
        "uncompressed_bytes": out_bytes,
        "compressed_bytes": chunk_path.stat().st_size if chunk_path.exists() else 0,
    }


def process_pubchem(
    root: Path,
    sources: list[dict[str, Any]],
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    locks = [output_lock(root / src["output"]) for src in sources]
    with contextlib.ExitStack() as stack:
        for lock in locks:
            stack.enter_context(lock)
        ensure_output_available(root, sources, args.overwrite)
        run_dir = Path(args.spool_dir).resolve() / f"pubchem_{args.variant}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        spool_dir = run_dir / "spool"
        chunk_dir = run_dir / "chunks"
        spool_report = spool_pubchem(root, sources, spool_dir, args.buckets, args.max_records, args.read_threads)
        tasks = []
        for source in sources:
            for bucket in range(args.buckets):
                bucket_path = spool_dir / source["name"] / f"bucket-{bucket:05d}.jsonl"
                chunk_path = chunk_dir / source["name"] / f"{source['name']}.part-{bucket:05d}.jsonl.zst"
                tasks.append((source, cfg, bucket_path, chunk_path, args.zstd_threads))
        log(f"sharded worker start kind=pubchem tasks={len(tasks)} workers={args.workers}")
        aggregate: dict[str, Counter[str]] = {src["name"]: Counter(spool_report["counts"].get(src["name"], {})) for src in sources}
        kept: Counter[str] = Counter()
        uncompressed: Counter[str] = Counter()
        chunks_by_source: dict[str, list[Path]] = defaultdict(list)
        ctx = get_context("fork")
        with ctx.Pool(processes=args.workers, maxtasksperchild=100) as pool:
            for result in pool.imap_unordered(pubchem_worker, tasks, chunksize=1):
                name = str(result["source_name"])
                aggregate[name].update(result["counts"])
                kept[name] += int(result["kept"])
                uncompressed[name] += int(result["uncompressed_bytes"])
                chunks_by_source[name].append(Path(str(result["chunk"])))
                if kept[name] and kept[name] % 1000000 < int(result["kept"]):
                    log(f"sharded progress kind=pubchem source={name} kept={kept[name]}")
        results = []
        for source in sources:
            out_path = root / source["output"]
            compressed = merge_zstd_frames(chunks_by_source[source["name"]], out_path)
            result = {
                "name": source["name"],
                "domain": "small_molecule",
                "input": str(root / source["input"]),
                "output": str(out_path),
                "counts": dict(aggregate[source["name"]]),
                "kept": int(kept[source["name"]]),
                "uncompressed_bytes": int(uncompressed[source["name"]]),
                "compressed_bytes": int(compressed),
                "created_utc": utc_now(),
                "sharded_workers": args.workers,
                "sharded_buckets": args.buckets,
                "dedup_stage": "deferred_to_clean_dedup",
            }
            write_source_done(root, result)
            log(
                f"sharded done kind=pubchem source={source['name']} kept={result['kept']} "
                f"out_compressed={compressed}"
            )
            results.append(result)
        return results


def spool_fasta(
    root: Path,
    source: dict[str, Any],
    spool_dir: Path,
    buckets: int,
    max_records: int | None,
    read_threads: int,
) -> dict[str, Any]:
    input_path = root / source["input"]
    source_dir = spool_dir / source["name"]
    source_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (source_dir / f"bucket-{bucket:05d}.jsonl").open("wb", buffering=1024 * 1024)
        for bucket in range(buckets)
    ]
    counts: Counter[str] = Counter()
    spooled = 0
    try:
        log(f"sharded spool start kind=fasta source={source['name']} input={input_path}")
        for seen, (original_id, header, raw_seq) in enumerate(iter_fasta_fast(input_path, read_threads), 1):
            if max_records is not None and seen > max_records:
                break
            key = f"{original_id}|{raw_seq[:80]}|{len(raw_seq)}"
            if passes_input_sampling(key, source):
                bucket = bucket_for(key, buckets)
                handles[bucket].write(json_line([original_id, header, raw_seq]))
                spooled += 1
            else:
                counts["input_sampled_out"] += 1
            if seen % 250000 == 0:
                log(f"sharded spool progress kind=fasta source={source['name']} seen={seen} spooled={spooled}")
    finally:
        for handle in handles:
            handle.close()
    return {"input": str(input_path), "counts": dict(counts), "spooled": spooled}


def fasta_worker(task: tuple[dict[str, Any], dict[str, Any], Path, Path, int]) -> dict[str, Any]:
    source, cfg, bucket_path, chunk_path, zstd_threads = task
    domain = str(source["domain"])
    domain_cfg = dict(cfg.get(domain) or {})
    alphabet = set(str(domain_cfg.get("alphabet", "")).strip())
    strict_alphabet_filter = strict_alphabet_filter_enabled(domain_cfg)
    min_length = int(domain_cfg.get("min_length", 1))
    max_length = int(domain_cfg.get("max_length", 10**9))
    window_sizes = split_ints(domain_cfg.get("window_sizes"), [256])
    stride_value = domain_cfg.get("stride")
    stride = None if stride_value in {None, "", "half"} else int(stride_value)
    max_windows = int(domain_cfg.get("max_windows_per_record", 0))
    max_single_fraction = float(domain_cfg.get("max_single_base_fraction", 1.1))
    allow_t_to_u = bool(domain_cfg.get("t_to_u", False))
    if domain in {"protein", "peptide"}:
        heavy_table = AA_HEAVY_ATOMS
    elif domain == "dna":
        heavy_table = DNA_HEAVY_ATOMS
    elif domain == "rna":
        heavy_table = RNA_HEAVY_ATOMS
    else:
        raise ValueError(f"Unsupported FASTA domain: {domain}")

    counts: Counter[str] = Counter()
    kept = 0
    out_bytes = 0
    proc, stream = zstd_writer(chunk_path, zstd_threads)
    try:
        with bucket_path.open("rb") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                original_id, header, raw_seq = json.loads(raw_line)
                seq = str(raw_seq)
                if domain == "rna" and allow_t_to_u:
                    seq = seq.replace("T", "U")
                if len(seq) < min_length:
                    counts["too_short_record"] += 1
                    continue
                if len(seq) > max_length:
                    counts["too_long_record"] += 1
                    continue
                if alphabet and any(ch not in alphabet for ch in seq):
                    counts["nonstandard_record"] += 1
                    if strict_alphabet_filter:
                        continue
                window_count = 0
                for start, end, window in sequence_windows(seq, window_sizes, stride, max_windows):
                    if len(window) < min_length:
                        counts["too_short_window"] += 1
                        continue
                    if alphabet and any(ch not in alphabet for ch in window):
                        counts["nonstandard_window"] += 1
                        if strict_alphabet_filter:
                            continue
                    if domain in {"dna", "rna"} and reject_low_complexity(window, max_single_fraction):
                        counts["low_complexity_window"] += 1
                        continue
                    row_hash = sha1_text(f"{domain}|{window}")
                    if not passes_source_sampling(row_hash, source):
                        counts["sampled_out"] += 1
                        continue
                    row = {
                        "id": f"{source['source']}:{original_id}:{start}:{end}",
                        "domain": domain,
                        "source": source["source"],
                        "source_version": source["source_version"],
                        "original_id": original_id,
                        "raw": window,
                        "normalized": window,
                        "length": len(window),
                        "estimated_heavy_atoms": sequence_heavy_atoms(window, heavy_table),
                        "split_group": original_id,
                        "metadata": annotate_nonstandard({
                            "header": header,
                            "start": start,
                            "end": end,
                            "source_record_length": len(seq),
                        }, window, alphabet),
                        "filter_status": "ok",
                        "sha1": row_hash,
                    }
                    line = row_line(row)
                    stream.write(line)
                    out_bytes += len(line)
                    kept += 1
                    window_count += 1
                if window_count:
                    counts["ok"] += window_count
                else:
                    counts["no_windows_kept"] += 1
        finish_writer(proc, stream)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        raise
    return {
        "source_name": source["name"],
        "bucket": bucket_path.name,
        "chunk": str(chunk_path),
        "counts": dict(counts),
        "kept": kept,
        "uncompressed_bytes": out_bytes,
        "compressed_bytes": chunk_path.stat().st_size if chunk_path.exists() else 0,
    }


def process_fasta(
    root: Path,
    source: dict[str, Any],
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_path = root / source["output"]
    with output_lock(out_path):
        ensure_output_available(root, [source], args.overwrite)
        run_dir = Path(args.spool_dir).resolve() / f"{source['name']}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        spool_dir = run_dir / "spool"
        chunk_dir = run_dir / "chunks"
        spool_report = spool_fasta(root, source, spool_dir, args.buckets, args.max_records, args.read_threads)
        tasks = []
        for bucket in range(args.buckets):
            bucket_path = spool_dir / source["name"] / f"bucket-{bucket:05d}.jsonl"
            chunk_path = chunk_dir / source["name"] / f"{source['name']}.part-{bucket:05d}.jsonl.zst"
            tasks.append((source, cfg, bucket_path, chunk_path, args.zstd_threads))
        log(f"sharded worker start kind=fasta source={source['name']} tasks={len(tasks)} workers={args.workers}")
        aggregate = Counter(spool_report["counts"])
        kept = 0
        uncompressed = 0
        chunks: list[Path] = []
        ctx = get_context("fork")
        with ctx.Pool(processes=args.workers, maxtasksperchild=50) as pool:
            for result in pool.imap_unordered(fasta_worker, tasks, chunksize=1):
                aggregate.update(result["counts"])
                kept += int(result["kept"])
                uncompressed += int(result["uncompressed_bytes"])
                chunks.append(Path(str(result["chunk"])))
                if kept and kept % 1000000 < int(result["kept"]):
                    log(f"sharded progress kind=fasta source={source['name']} kept={kept}")
        compressed = merge_zstd_frames(chunks, out_path)
        result = {
            "name": source["name"],
            "domain": source["domain"],
            "input": str(root / source["input"]),
            "output": str(out_path),
            "counts": dict(aggregate),
            "kept": int(kept),
            "uncompressed_bytes": int(uncompressed),
            "compressed_bytes": int(compressed),
            "created_utc": utc_now(),
            "sharded_workers": args.workers,
            "sharded_buckets": args.buckets,
            "dedup_stage": "deferred_to_clean_dedup",
        }
        write_source_done(root, result)
        log(f"sharded done kind=fasta source={source['name']} kept={kept} out_compressed={compressed}")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Sharded high-throughput processor for large molscaling sources.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--variant", default="bulk")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--buckets", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--spool-dir", type=Path, default=Path("./outputs/tmp"))
    parser.add_argument("--read-threads", type=int, default=int(os.environ.get("MOLSCALING_READ_THREADS", "8")))
    parser.add_argument("--zstd-threads", type=int, default=int(os.environ.get("MOLSCALING_SHARDED_ZSTD_THREADS", "1")))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["root"]).resolve()
    wanted = set(args.source)
    base_sources = [src for src in (cfg.get("sources") or []) if src.get("name") in wanted]
    if len(base_sources) != len(wanted):
        found = {src.get("name") for src in base_sources}
        raise SystemExit(f"missing configured sources: {sorted(wanted - found)}")
    sources = [_variant_source(src, args.variant) for src in base_sources]
    formats = {src.get("format") for src in sources}
    plan = {
        "event": "sharded_process_plan",
        "root": str(root),
        "sources": [src["name"] for src in sources],
        "formats": sorted(str(fmt) for fmt in formats),
        "workers": args.workers,
        "buckets": args.buckets,
        "spool_dir": str(args.spool_dir),
        "read_threads": args.read_threads,
        "zstd_threads": args.zstd_threads,
        "max_records": args.max_records,
        "dedup_stage": "deferred_to_clean_dedup",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    args.spool_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if formats == {"pubchem_cid_smiles_gz"}:
        results.extend(process_pubchem(root, sources, cfg, args))
    elif len(sources) == 1 and formats == {"fasta_gz"}:
        results.append(process_fasta(root, sources[0], cfg, args))
    else:
        raise SystemExit("sharded mode supports grouped PubChem sources or one FASTA source per invocation")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"sharded_process_report_{args.variant}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "sharded_process_report", "path": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
