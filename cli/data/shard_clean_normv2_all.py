#!/usr/bin/env python3
"""把 clean_normv2 全量流式重分片为 graph 阶段输入。

这个脚本不做抽样；它只做三件事：

1. 逐行读取 clean JSONL.zst；
2. 用稳定哈希补齐 train/valid/test split 字段；
3. 按未压缩字节数切成较大的 JSONL.zst shard。

必须在 worker 上运行，不要在跳板机上处理全量数据。
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from cli.data import sample_corpus as base  # noqa: E402


def parse_inputs(raw: str) -> list[dict[str, str]]:
    """解析输入清单；每项格式为 ``domain:relative/path``。"""

    items: list[dict[str, str]] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"bad input item, expected domain:path: {text}")
        domain, path = text.split(":", 1)
        domain = domain.strip()
        path = path.strip()
        if not domain or not path:
            raise ValueError(f"bad input item, expected domain:path: {text}")
        items.append({"domain": domain, "path": path})
    return items


def default_inputs() -> list[dict[str, str]]:
    """返回 normv2 全量 clean 文件清单。"""

    return [
        {"domain": "small_molecule", "path": "clean_normv2/small_molecule/chembl_36_chemreps.jsonl.zst"},
        {"domain": "small_molecule", "path": "clean_normv2/small_molecule/pubchem_cid_smiles.jsonl.zst"},
        {"domain": "small_molecule", "path": "clean_normv2/small_molecule/pubchem_cid_smiles_extra.jsonl.zst"},
        {"domain": "small_molecule", "path": "clean_normv2/small_molecule/pubchem_cid_smiles_extra2.jsonl.zst"},
        {"domain": "small_molecule", "path": "clean_normv2/small_molecule/pubchem_cid_smiles_extra3_bulk.jsonl.zst"},
        {"domain": "small_molecule", "path": "clean_normv2/small_molecule/pubchem_cid_smiles_extra4_bulk.jsonl.zst"},
        {"domain": "protein", "path": "clean_normv2/protein/swissprot_fasta_normv2.jsonl.zst"},
        {"domain": "protein", "path": "clean_normv2/protein/uniref50_fasta_normv2.jsonl.zst"},
        {"domain": "dna", "path": "clean_normv2/dna/refseq_viral_genomic_fna_normv2.jsonl.zst"},
        {"domain": "dna", "path": "clean_normv2/dna/refseq_archaea_genomic_fna_normv2.jsonl.zst"},
        {"domain": "dna", "path": "clean_normv2/dna/refseq_fungi_genomic_fna_normv2.jsonl.zst"},
        {"domain": "dna", "path": "clean_normv2/dna/refseq_bacteria_genomic_fna_normv2.jsonl.zst"},
        {"domain": "rna", "path": "clean_normv2/rna/rnacentral_active_fasta_normv2.jsonl.zst"},
        {"domain": "rna", "path": "clean_normv2/rna/rfam_RF00001_fasta_normv2.jsonl.zst"},
        {"domain": "rna", "path": "clean_normv2/rna/rfam_RF00005_fasta_normv2.jsonl.zst"},
        {"domain": "rna", "path": "clean_normv2/rna/rfam_RF00017_fasta_normv2.jsonl.zst"},
        {"domain": "rna", "path": "clean_normv2/rna/rfam_RF00026_fasta_normv2.jsonl.zst"},
        {"domain": "peptide", "path": "clean_normv2/peptide/pepbenchmark_raw_normv2.jsonl.zst"},
        {"domain": "peptide", "path": "clean_normv2/peptide/protein_derived_peptide_swissprot_normv2.jsonl.zst"},
    ]


def zstd_writer(path: Path, threads: int, level: int):
    """打开一个 zstd 压缩写入流。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    zstd = base.find_zstd()
    if zstd is None:
        raise RuntimeError("zstd executable is required")
    proc = subprocess.Popen(
        [zstd, f"-T{max(1, threads)}", f"-{level}", "-f", "-o", str(path)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    return proc, proc.stdin


def finish_writer(proc: subprocess.Popen[bytes], stream: Any) -> None:
    """关闭压缩写入流，并检查 zstd 返回码。"""

    stream.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"zstd failed with rc={rc}")


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """流式读取 JSONL.zst 行。"""

    yield from base.iter_jsonl_zst(path)


def output_path(root: Path, output_root: str, domain: str, split: str, input_index: int, shard_id: int) -> Path:
    """返回当前 shard 的输出路径。"""

    return root / output_root / domain / split / f"part-input{input_index:03d}-{shard_id:05d}.jsonl.zst"


def shard_one(
    root_text: str,
    output_root: str,
    item: dict[str, str],
    input_index: int,
    seed: int,
    valid_fraction: float,
    test_fraction: float,
    shard_bytes: int,
    zstd_threads: int,
    zstd_level: int,
    log_every: int,
) -> dict[str, Any]:
    """处理一个 clean 文件，写出它的全量 train/valid/test shards。"""

    root = Path(root_text)
    domain = item["domain"]
    input_path = root / item["path"]
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    stats = Counter()
    split_stats: dict[str, Counter[str]] = {}
    outputs: list[dict[str, Any]] = []
    handles: dict[str, tuple[subprocess.Popen[bytes], Any, int, int, int]] = {}

    def open_split(split: str) -> tuple[subprocess.Popen[bytes], Any, int, int, int]:
        shard_id = int(split_stats.setdefault(split, Counter()).get("shards_opened", 0))
        split_stats[split]["shards_opened"] += 1
        out = output_path(root, output_root, domain, split, input_index, shard_id)
        proc, stream = zstd_writer(out, zstd_threads, zstd_level)
        outputs.append(
            {
                "domain": domain,
                "split": split,
                "path": str(out),
                "records": 0,
                "uncompressed_bytes": 0,
                "compressed_bytes": 0,
            }
        )
        return proc, stream, 0, shard_id, len(outputs) - 1

    try:
        for row_idx, row in enumerate(iter_rows(input_path), 1):
            split = base.split_for(row, seed, valid_fraction, test_fraction)
            row["split"] = split
            clean = {k: v for k, v in row.items() if not k.startswith("_sample_")}
            line = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"

            handle = handles.get(split)
            if handle is None:
                handle = open_split(split)
                handles[split] = handle
            proc, stream, used, shard_id, output_index = handle
            if used >= shard_bytes and outputs[output_index]["records"] > 0:
                finish_writer(proc, stream)
                handle = open_split(split)
                handles[split] = handle
                proc, stream, used, shard_id, output_index = handle

            stream.write(line)
            used += len(line)
            handles[split] = (proc, stream, used, shard_id, output_index)

            outputs[output_index]["records"] += 1
            outputs[output_index]["uncompressed_bytes"] += len(line)
            split_stats.setdefault(split, Counter())["records"] += 1
            split_stats[split]["uncompressed_bytes"] += len(line)
            stats["records"] += 1
            stats["uncompressed_bytes"] += len(line)
            stats["estimated_heavy_atoms"] += int(row.get("estimated_heavy_atoms") or 0)
            if log_every > 0 and row_idx % log_every == 0:
                print(
                    f"{dt.datetime.now().isoformat(timespec='seconds')} shard_progress "
                    f"domain={domain} input={input_path.name} records={row_idx}",
                    flush=True,
                )
    finally:
        for proc, stream, _used, _shard_id, _output_index in handles.values():
            finish_writer(proc, stream)

    compressed_total = 0
    for out in outputs:
        path = Path(str(out["path"]))
        compressed = path.stat().st_size if path.exists() else 0
        out["compressed_bytes"] = compressed
        compressed_total += compressed

    return {
        "index": input_index,
        "domain": domain,
        "input": str(input_path),
        "records": int(stats["records"]),
        "uncompressed_bytes": int(stats["uncompressed_bytes"]),
        "estimated_heavy_atoms": int(stats["estimated_heavy_atoms"]),
        "compressed_output_bytes": int(compressed_total),
        "splits": {
            split: {key: int(value) for key, value in split_counter.items()}
            for split, split_counter in sorted(split_stats.items())
        },
        "outputs": outputs,
    }


def write_report(root: Path, output_root: str, report: dict[str, Any]) -> None:
    """写 reports 和 manifests，便于后续溯源。"""

    reports = root / "reports"
    manifests = root / "manifests"
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (reports / f"shard_clean_normv2_all_{stamp}.json").write_text(text, encoding="utf-8")
    (reports / "shard_clean_normv2_all.latest.json").write_text(text, encoding="utf-8")
    (manifests / "dataset_manifest_normv2_all.json").write_text(text, encoding="utf-8")
    (root / output_root / "manifest.json").write_text(text, encoding="utf-8")


def main() -> int:
    """命令行入口。"""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            str(_moltoken_paths.LOCAL_DATA_ROOT / '')
        ),
    )
    ap.add_argument("--output-root", default="shards_normv2_all")
    ap.add_argument("--inputs", default="", help="Optional comma-separated domain:path list.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260518)
    ap.add_argument("--valid-fraction", type=float, default=0.005)
    ap.add_argument("--test-fraction", type=float, default=0.005)
    ap.add_argument("--shard-uncompressed-bytes", type=int, default=1_073_741_824)
    ap.add_argument("--zstd-threads", type=int, default=1)
    ap.add_argument("--zstd-level", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=2_000_000)
    args = ap.parse_args()

    if not __import__("pipeline.runtime_host", fromlist=["is_compute_host"]).is_compute_host(os.uname().nodename):
        raise SystemExit(f"refusing to run outside a worker: hostname={os.uname().nodename}")
    root = args.root.resolve()
    output_path_root = root / args.output_root
    if output_path_root.exists() and any(output_path_root.iterdir()):
        raise SystemExit(f"output root exists and is not empty: {output_path_root}")
    output_path_root.mkdir(parents=True, exist_ok=True)

    inputs = parse_inputs(args.inputs) if args.inputs.strip() else default_inputs()
    workers = max(1, min(int(args.workers), len(inputs)))

    lock_path = root / "reports" / "shard_clean_normv2_all.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another all-shard job is running lock={lock_path}", flush=True)
        return 75

    print(
        f"{dt.datetime.now().isoformat(timespec='seconds')} shard_clean_normv2_all_start "
        f"inputs={len(inputs)} workers={workers} output={output_path_root}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                shard_one,
                str(root),
                str(args.output_root),
                item,
                idx,
                int(args.seed),
                float(args.valid_fraction),
                float(args.test_fraction),
                int(args.shard_uncompressed_bytes),
                int(args.zstd_threads),
                int(args.zstd_level),
                int(args.log_every),
            )
            for idx, item in enumerate(inputs)
        ]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"{dt.datetime.now().isoformat(timespec='seconds')} shard_clean_normv2_all_done "
                f"done={done}/{len(futures)} domain={result['domain']} records={result['records']}",
                flush=True,
            )

    domains: dict[str, Counter[str]] = {}
    splits: dict[str, Counter[str]] = {}
    outputs: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda x: int(x["index"])):
        dom = str(result["domain"])
        domains.setdefault(dom, Counter())
        for key in ("records", "uncompressed_bytes", "estimated_heavy_atoms", "compressed_output_bytes"):
            domains[dom][key] += int(result.get(key) or 0)
        for split, split_stats in (result.get("splits") or {}).items():
            splits.setdefault(split, Counter())
            for key, value in split_stats.items():
                if key != "shards_opened":
                    splits[split][key] += int(value)
        outputs.extend(result.get("outputs") or [])

    report = {
        "status": "ok",
        "created_utc": base.utc_now(),
        "root": str(root),
        "output_root": str(args.output_root),
        "inputs": results,
        "outputs": outputs,
        "domains": {
            domain: {key: int(value) for key, value in counter.items()}
            for domain, counter in sorted(domains.items())
        },
        "splits": {
            split: {key: int(value) for key, value in counter.items()}
            for split, counter in sorted(splits.items())
        },
        "total_records": int(sum(counter["records"] for counter in domains.values())),
        "total_uncompressed_bytes": int(sum(counter["uncompressed_bytes"] for counter in domains.values())),
        "total_compressed_output_bytes": int(sum(counter["compressed_output_bytes"] for counter in domains.values())),
        "workers": workers,
        "seed": int(args.seed),
        "valid_fraction": float(args.valid_fraction),
        "test_fraction": float(args.test_fraction),
        "shard_uncompressed_bytes": int(args.shard_uncompressed_bytes),
        "zstd_threads": int(args.zstd_threads),
        "zstd_level": int(args.zstd_level),
    }
    write_report(root, str(args.output_root), report)
    print("shard_clean_normv2_all_complete " + json.dumps({k: report[k] for k in ("total_records", "total_uncompressed_bytes", "total_compressed_output_bytes")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
