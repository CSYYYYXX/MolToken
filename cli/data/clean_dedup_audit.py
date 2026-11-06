#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import yaml


REQUIRED_FIELDS = {
    "id",
    "domain",
    "source",
    "source_version",
    "original_id",
    "raw",
    "normalized",
    "length",
    "estimated_heavy_atoms",
    "split_group",
    "metadata",
    "filter_status",
    "sha1",
}

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
KNOWN_ZSTD = [
    
    "/usr/bin/zstd",
    "/usr/local/bin/zstd",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def find_zstd() -> str | None:
    found = shutil.which("zstd")
    if found:
        return found
    for candidate in KNOWN_ZSTD:
        if Path(candidate).exists():
            return candidate
    return None


def is_zstd_file(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(4) == ZSTD_MAGIC


def iter_jsonl_auto(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".zst" and is_zstd_file(path):
        zstd = find_zstd()
        if zstd is None:
            raise RuntimeError(f"zstd-compressed file requires zstd executable: {path}")
        proc = subprocess.Popen([zstd, "-dc", str(path)], stdout=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if line.strip():
                    yield json.loads(line)
        finally:
            proc.stdout.close()
            rc = proc.wait()
            if rc not in {0, -13}:
                raise RuntimeError(f"zstd -dc failed for {path} rc={rc}")
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def zstd_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    zstd = find_zstd()
    if zstd is None:
        raise RuntimeError(f"zstd executable is required for compressed output: {path}")
    level = os.environ.get("MOLSCALING_ZSTD_LEVEL", "3")
    proc = subprocess.Popen([zstd, "-T0", f"-{level}", "-f", "-o", str(path)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    return proc, proc.stdin


def finish_writer(proc, stream) -> None:
    stream.close()
    if proc is not None:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"zstd failed with rc={rc}")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def row_size(row: dict[str, Any]) -> int:
    return len(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")) + 1


def bytes_label(n: int | None) -> str:
    if n is None:
        return "-"
    value = float(n)
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GB"
    return f"{value / 1024**2:.2f} MB"


def bin_length(value: int) -> str:
    if value < 16:
        return "lt016"
    if value < 64:
        return "016_063"
    if value < 128:
        return "064_127"
    if value < 256:
        return "128_255"
    if value < 512:
        return "256_511"
    if value < 1024:
        return "512_1023"
    return "ge1024"


def bin_heavy(value: int) -> str:
    if value < 64:
        return "lt064"
    if value < 256:
        return "064_255"
    if value < 1024:
        return "256_1023"
    if value < 4096:
        return "1024_4095"
    return "ge4096"


class GlobalDeduper:
    def __init__(self, cfg: dict[str, Any], stamp: str):
        dedup_cfg = cfg.get("dedup") or {}
        dedup_dir = Path(str(dedup_cfg.get("dir") or Path(tempfile.gettempdir()) / "moltoken_clean_dedup"))
        dedup_dir.mkdir(parents=True, exist_ok=True)
        self.path = dedup_dir / f"seen_{stamp}_{os.getpid()}.sqlite"
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute(
            "CREATE TABLE seen (sha1 TEXT PRIMARY KEY, domain TEXT, input_path TEXT, source TEXT)"
        )
        self.pending = 0

    def add(self, key: str, domain: str, input_path: str, source: str) -> tuple[bool, str | None]:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO seen (sha1, domain, input_path, source) VALUES (?, ?, ?, ?)",
            (key, domain, input_path, source),
        )
        self.pending += 1
        if self.pending >= 100000:
            self.conn.commit()
            self.pending = 0
        if cur.rowcount:
            return True, None
        owner = self.conn.execute("SELECT input_path FROM seen WHERE sha1 = ?", (key,)).fetchone()
        return False, str(owner[0]) if owner else None

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def load_rdkit():
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError as exc:
        raise RuntimeError("RDKit is required for small-molecule clean/dedup validation") from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem


def validate_row(
    row: dict[str, Any],
    expected_domain: str,
    cfg: dict[str, Any],
    chem: Any,
) -> tuple[bool, str]:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        return False, "missing_required_fields"
    domain = str(row.get("domain"))
    if domain != expected_domain:
        return False, "domain_mismatch"
    normalized = row.get("normalized")
    if not isinstance(normalized, str) or not normalized:
        return False, "empty_normalized"
    if cfg.get("checks", {}).get("strict_filter_status_ok", True) and row.get("filter_status") != "ok":
        return False, "filter_status_not_ok"
    if cfg.get("checks", {}).get("strict_length", True):
        if not isinstance(row.get("length"), int) or int(row["length"]) != len(normalized):
            return False, "length_mismatch"
    if not isinstance(row.get("estimated_heavy_atoms"), int) or int(row["estimated_heavy_atoms"]) <= 0:
        return False, "bad_estimated_heavy_atoms"
    if cfg.get("checks", {}).get("recompute_sha1", True):
        expected_sha1 = sha1_text(f"{domain}|{normalized}")
        if row.get("sha1") != expected_sha1:
            return False, "sha1_mismatch"
    domain_cfg = (cfg.get("domains") or {}).get(domain) or {}
    alphabet = str(domain_cfg.get("alphabet") or "")
    strict_alphabet = bool(domain_cfg.get("strict_alphabet_filter", False))
    if strict_alphabet and alphabet and any(ch not in alphabet for ch in normalized):
        return False, "invalid_alphabet"
    if domain == "small_molecule" and cfg.get("checks", {}).get("rdkit_small_molecule", True):
        if chem is None:
            return False, "rdkit_unavailable"
        if chem.MolFromSmiles(normalized) is None:
            return False, "rdkit_parse_failed"
    return True, "ok"


def derived_output_path(root: Path, output_root: str, input_rel: str, domain: str) -> Path:
    name = Path(input_rel).name
    return root / output_root / domain / name


def clean_one_input(
    item: dict[str, Any],
    root: Path,
    output_root: str,
    cfg: dict[str, Any],
    deduper: GlobalDeduper,
    max_records: int | None,
    overwrite: bool,
    chem: Any,
    dry_run: bool,
) -> dict[str, Any]:
    input_rel = str(item["path"])
    expected_domain = str(item["domain"])
    in_path = root / input_rel
    out_path = derived_output_path(root, output_root, input_rel, expected_domain)
    stats: dict[str, Any] = {
        "input": str(in_path),
        "output": str(out_path),
        "domain": expected_domain,
        "exists": in_path.exists(),
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
    reject_reasons: Counter[str] = Counter()
    length_bins: Counter[str] = Counter()
    heavy_bins: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    if not in_path.exists():
        reject_reasons["missing_input"] += 1
        stats["reject_reasons"] = dict(reject_reasons)
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} clean skip missing input={in_path}", flush=True)
        return stats
    if dry_run:
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} clean dry-run input={in_path} output={out_path}", flush=True)
        return stats
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")

    tmp_dir = out_path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{out_path.name}.", suffix=".tmp", dir=str(tmp_dir), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    tmp_path.unlink(missing_ok=True)

    proc, stream = zstd_writer(tmp_path)
    try:
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} clean start input={in_path} output={out_path}", flush=True)
        for i, row in enumerate(iter_jsonl_auto(in_path), 1):
            if max_records is not None and i > max_records:
                break
            stats["records_seen"] += 1
            stats["input_uncompressed_bytes"] += row_size(row)
            ok, reason = validate_row(row, expected_domain, cfg, chem)
            if not ok:
                reject_reasons[reason] += 1
                continue
            domain_cfg = (cfg.get("domains") or {}).get(expected_domain) or {}
            alphabet = str(domain_cfg.get("alphabet") or "")
            normalized = str(row.get("normalized") or "")
            if alphabet and any(ch not in alphabet for ch in normalized):
                reject_reasons["observed_nonstandard_alphabet"] += 1
            key = str(row["sha1"])
            added, owner = deduper.add(key, expected_domain, input_rel, str(row.get("source", "")))
            if not added:
                reason = "duplicate_within_input" if owner == input_rel else "duplicate_cross_source"
                reject_reasons[reason] += 1
                examples = stats["duplicate_examples"]
                if isinstance(examples, list) and len(examples) < 20:
                    examples.append({
                        "sha1": key,
                        "current_input": input_rel,
                        "first_input": owner,
                        "domain": expected_domain,
                        "source": row.get("source"),
                    })
                continue
            line = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            stream.write(line)
            stats["records_kept"] += 1
            stats["output_uncompressed_bytes"] += len(line)
            length_bins[bin_length(int(row["length"]))] += 1
            heavy_bins[bin_heavy(int(row["estimated_heavy_atoms"]))] += 1
            sources[str(row.get("source", "unknown"))] += 1
            if i % 500000 == 0:
                print(
                    f"{dt.datetime.now().isoformat(timespec='seconds')} clean progress "
                    f"input={in_path.name} seen={stats['records_seen']} kept={stats['records_kept']} "
                    f"out_uncompressed={stats['output_uncompressed_bytes']}",
                    flush=True,
                )
    finally:
        finish_writer(proc, stream)
    tmp_path.replace(out_path)
    stats["output_compressed_bytes"] = out_path.stat().st_size if out_path.exists() else 0
    stats["reject_reasons"] = dict(reject_reasons)
    stats["length_bins"] = dict(length_bins)
    stats["heavy_atom_bins"] = dict(heavy_bins)
    stats["sources"] = dict(sources)
    print(
        f"{dt.datetime.now().isoformat(timespec='seconds')} clean done input={in_path} "
        f"seen={stats['records_seen']} kept={stats['records_kept']} "
        f"out_compressed={stats['output_compressed_bytes']}",
        flush=True,
    )
    return stats


def summarize(results: list[dict[str, Any]], cfg: dict[str, Any], args: argparse.Namespace, stamp: str, db_path: Path) -> dict[str, Any]:
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
        "output_root": str(args.output_root or cfg.get("output_root") or "clean"),
        "max_records_per_input": args.max_records,
        "dry_run": bool(args.dry_run),
        "dedup_db": str(db_path),
        "totals": dict(totals),
        "reject_reasons": dict(reject_reasons),
        "domains": {k: dict(v) for k, v in sorted(domains.items())},
        "inputs": results,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    totals = report["totals"]
    lines = [
        "# Graph 前清洗去重审计",
        "",
        f"生成时间：{report['created_utc']}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 输入记录 | {totals.get('records_seen', 0):,} |",
        f"| 清洗去重后保留记录 | {totals.get('records_kept', 0):,} |",
        f"| 输入未压缩规模 | {bytes_label(totals.get('input_uncompressed_bytes', 0))} |",
        f"| 输出未压缩规模 | {bytes_label(totals.get('output_uncompressed_bytes', 0))} |",
        f"| 输出压缩文件占用 | {bytes_label(totals.get('output_compressed_bytes', 0))} |",
        "",
        "## 按领域统计",
        "",
        "| 领域 | 输入记录 | 保留记录 | 输出未压缩 | 输出压缩占用 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for domain, stats in sorted((report.get("domains") or {}).items()):
        lines.append(
            f"| {domain} | {stats.get('records_seen', 0):,} | {stats.get('records_kept', 0):,} | "
            f"{bytes_label(stats.get('output_uncompressed_bytes', 0))} | {bytes_label(stats.get('output_compressed_bytes', 0))} |"
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
    lines.extend([
        "",
        "说明：该阶段在 heavy-atom graph 展开之前运行，只保留通过 schema、hash、长度、alphabet/RDKit 检查且未发生单源或跨源重复的 JSONL 记录。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean, globally deduplicate, and audit processed molscaling JSONL files before graph expansion.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--source", action="append", help="Only process input paths whose basename or path contains this value.")
    ap.add_argument("--output-root", default=None, help="Override clean output root under data root, e.g. clean_smoke.")
    ap.add_argument("--max-records", type=int, default=None, help="Maximum records to inspect per input.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="Continue and write an audit report even if configured inputs are missing.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["root"]).resolve()
    lock_path = root / "reports" / "clean_dedup_audit.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another clean_dedup_audit is already running lock={lock_path}", flush=True)
        return 75
    output_root = str(args.output_root or cfg.get("output_root") or "clean")
    inputs = list(cfg.get("inputs") or [])
    if args.source:
        wanted = list(args.source)
        inputs = [
            item for item in inputs
            if any(token in str(item.get("path", "")) or token in Path(str(item.get("path", ""))).name for token in wanted)
        ]
    missing_inputs = [str(root / str(item["path"])) for item in inputs if not (root / str(item["path"])).exists()]
    if missing_inputs and not args.allow_missing:
        print("clean_dedup_audit refusing to run because configured inputs are missing:", flush=True)
        for path in missing_inputs:
            print(f"  missing input={path}", flush=True)
        print("Pass --allow-missing only for explicit smoke/debug audits.", flush=True)
        return 2
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    chem = load_rdkit() if cfg.get("checks", {}).get("rdkit_small_molecule", True) else None
    deduper = GlobalDeduper(cfg, stamp)
    results: list[dict[str, Any]] = []
    try:
        for item in inputs:
            results.append(clean_one_input(item, root, output_root, cfg, deduper, args.max_records, args.overwrite, chem, args.dry_run))
    finally:
        deduper.close()

    report = summarize(results, cfg, argparse.Namespace(**{**vars(args), "output_root": output_root}), stamp, deduper.path)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_report = reports / f"clean_dedup_audit_{stamp}.json"
    md_report = reports / f"clean_dedup_audit_{stamp}.md"
    latest_json = reports / "clean_dedup_audit.latest.json"
    latest_md = reports / "clean_dedup_audit.latest.md"
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    json_report.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    write_markdown(report, md_report)
    write_markdown(report, latest_md)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
