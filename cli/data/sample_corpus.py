#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
KNOWN_ZSTD = [
    
    "/usr/bin/zstd",
    "/usr/local/bin/zstd",
]


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


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
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


def stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}|{text}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64)


def split_for(row: dict[str, Any], seed: int, valid_fraction: float, test_fraction: float) -> str:
    key = str(row.get("split_group") or row.get("sha1") or row.get("normalized"))
    x = stable_unit_interval(key, seed)
    if x < test_fraction:
        return "test"
    if x < test_fraction + valid_fraction:
        return "valid"
    return "train"


def row_size(row: dict[str, Any]) -> int:
    return len(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")) + 1


def domain_bin(row: dict[str, Any]) -> str:
    length = int(row.get("length") or 0)
    heavy = int(row.get("estimated_heavy_atoms") or 0)
    if length < 64:
        lbin = "lt064"
    elif length < 128:
        lbin = "064_127"
    elif length < 256:
        lbin = "128_255"
    elif length < 512:
        lbin = "256_511"
    else:
        lbin = "ge512"
    if heavy < 64:
        hbin = "h_lt064"
    elif heavy < 256:
        hbin = "h_064_255"
    elif heavy < 1024:
        hbin = "h_256_1023"
    else:
        hbin = "h_ge1024"
    return f"{lbin}:{hbin}"


def collect_inventory(config: dict[str, Any], root: Path, target_scale: float) -> dict[str, Any]:
    budgets = config.get("budgets") or {}
    stats: dict[str, Any] = {
        "inputs": [],
        "missing": [],
        "streaming": True,
        "selection_strategy": "two_pass_hash_stratified_by_length_heavy_atom_bin",
    }
    for item in config.get("inputs") or []:
        path = root / str(item["path"])
        domain = str(item["domain"])
        if not path.exists():
            stats["missing"].append(str(path))
            continue
        count = 0
        bytes_seen = 0
        domain_stats = stats.setdefault("domains", {}).setdefault(domain, {
            "candidate_records": 0,
            "candidate_uncompressed_bytes": 0,
            "selected_records": 0,
            "selected_uncompressed_bytes": 0,
            "selected_estimated_heavy_atoms": 0,
            "bins": {},
        })
        for row in iter_jsonl_zst(path):
            size = row_size(row)
            sample_bin = domain_bin(row)
            bytes_seen += size
            count += 1
            domain_stats["candidate_records"] += 1
            domain_stats["candidate_uncompressed_bytes"] += size
            bin_stats = domain_stats["bins"].setdefault(sample_bin, {"records": 0, "uncompressed_bytes": 0})
            bin_stats["records"] += 1
            bin_stats["uncompressed_bytes"] += size
        stats["inputs"].append({"path": str(path), "domain": domain, "records": count, "uncompressed_bytes": bytes_seen})
    for domain, domain_stats in (stats.get("domains") or {}).items():
        target = int((budgets.get(domain) or {}).get("target_uncompressed_bytes", 0) * target_scale)
        domain_stats["target_uncompressed_bytes"] = target
    return stats


def add_required_report_status(config: dict[str, Any], root: Path, stats: dict[str, Any]) -> None:
    missing = []
    existing = []
    for rel in config.get("required_reports") or []:
        path = root / str(rel)
        item = {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        if path.exists() and path.stat().st_size > 0:
            existing.append(item)
        else:
            missing.append(item)
    stats["required_reports"] = existing
    stats["missing_required_reports"] = missing


def add_bin_targets(stats: dict[str, Any]) -> None:
    for domain_stats in (stats.get("domains") or {}).values():
        candidate_bytes = int(domain_stats.get("candidate_uncompressed_bytes") or 0)
        target_bytes = int(domain_stats.get("target_uncompressed_bytes") or 0)
        effective_target = min(target_bytes, candidate_bytes)
        domain_stats["effective_target_uncompressed_bytes"] = effective_target
        bins = domain_stats.get("bins") or {}
        for bin_stats in bins.values():
            bin_bytes = int(bin_stats.get("uncompressed_bytes") or 0)
            if candidate_bytes <= 0 or effective_target <= 0:
                bin_target = 0
            elif effective_target >= candidate_bytes:
                bin_target = bin_bytes
            else:
                bin_target = int(effective_target * (bin_bytes / candidate_bytes))
            bin_stats["target_uncompressed_bytes"] = bin_target
            bin_stats["selection_probability"] = min(1.0, (bin_target / bin_bytes) if bin_bytes else 0.0)
            bin_stats["selected_records"] = 0
            bin_stats["selected_uncompressed_bytes"] = 0


def projected_selected_uncompressed(stats: dict[str, Any]) -> int:
    return sum(
        int(domain_stats.get("effective_target_uncompressed_bytes") or 0)
        for domain_stats in (stats.get("domains") or {}).values()
    )


def total_candidate_uncompressed(stats: dict[str, Any]) -> int:
    return sum(
        int(domain_stats.get("candidate_uncompressed_bytes") or 0)
        for domain_stats in (stats.get("domains") or {}).values()
    )


def compressed_bytes_for_inputs(stats: dict[str, Any]) -> int:
    total = 0
    for item in stats.get("inputs") or []:
        path = Path(str(item["path"]))
        if path.exists():
            total += path.stat().st_size
    return total


def apply_domain_targets(config: dict[str, Any], stats: dict[str, Any], scale: float) -> None:
    budgets = config.get("budgets") or {}
    for domain, domain_stats in (stats.get("domains") or {}).items():
        domain_stats["target_uncompressed_bytes"] = int(
            (budgets.get(domain) or {}).get("target_uncompressed_bytes", 0) * scale
        )


def projected_uncompressed_for_scale(config: dict[str, Any], stats: dict[str, Any], scale: float) -> int:
    budgets = config.get("budgets") or {}
    total = 0
    for domain, domain_stats in (stats.get("domains") or {}).items():
        target = int((budgets.get(domain) or {}).get("target_uncompressed_bytes", 0) * scale)
        candidate = int(domain_stats.get("candidate_uncompressed_bytes") or 0)
        total += min(target, candidate)
    return total


def estimate_scale_for_compressed_target(
    config: dict[str, Any],
    root: Path,
    configured_total: int,
    target_gb: float,
) -> tuple[float, dict[str, Any]]:
    target_bytes = int(target_gb * 1024**3)
    baseline = collect_inventory(config, root, 1.0)
    add_bin_targets(baseline)
    compressed_inputs = compressed_bytes_for_inputs(baseline)
    candidate_uncompressed = total_candidate_uncompressed(baseline)
    selected_uncompressed = projected_selected_uncompressed(baseline)
    if compressed_inputs <= 0 or candidate_uncompressed <= 0 or selected_uncompressed <= 0:
        baseline["target_metric"] = "compressed_storage_estimate"
        baseline["target_compressed_bytes"] = target_bytes
        baseline["estimated_input_compression_ratio"] = None
        return 1.0, baseline

    compression_ratio = compressed_inputs / candidate_uncompressed
    desired_uncompressed = int(target_bytes / compression_ratio)
    max_scale = candidate_uncompressed / configured_total if configured_total else scale
    if desired_uncompressed >= candidate_uncompressed:
        scale = max_scale
    else:
        lo, hi = 0.0, max(max_scale, 1.0)
        for _ in range(48):
            mid = (lo + hi) / 2.0
            if projected_uncompressed_for_scale(config, baseline, mid) < desired_uncompressed:
                lo = mid
            else:
                hi = mid
        scale = hi

    scaled = copy.deepcopy(baseline)
    apply_domain_targets(config, scaled, scale)
    add_bin_targets(scaled)
    scaled["target_metric"] = "compressed_storage_estimate"
    scaled["target_compressed_bytes"] = target_bytes
    scaled["estimated_input_compression_ratio"] = compression_ratio
    scaled["estimated_selected_compressed_bytes"] = int(projected_selected_uncompressed(scaled) * compression_ratio)
    scaled["compressed_input_bytes"] = compressed_inputs
    scaled["candidate_uncompressed_bytes_total"] = candidate_uncompressed
    return scale, scaled


def selected_rows(
    config: dict[str, Any],
    root: Path,
    stats: dict[str, Any],
    seed: int,
    valid_fraction: float,
    test_fraction: float,
) -> Iterator[dict[str, Any]]:
    for item in config.get("inputs") or []:
        path = root / str(item["path"])
        domain = str(item["domain"])
        if not path.exists():
            continue
        domain_stats = (stats.get("domains") or {}).get(domain) or {}
        bins = domain_stats.get("bins") or {}
        for row in iter_jsonl_zst(path):
            size = row_size(row)
            sample_bin = domain_bin(row)
            bin_stats = bins.get(sample_bin) or {}
            probability = float(bin_stats.get("selection_probability") or 0.0)
            if probability <= 0.0:
                continue
            key = str(row.get("sha1") or f"{domain}|{row.get('normalized')}")
            if probability < 1.0 and stable_unit_interval(f"select|{domain}|{sample_bin}|{key}", seed) >= probability:
                continue
            row["_sample_size"] = size
            row["_sample_bin"] = sample_bin
            row["split"] = split_for(row, seed, valid_fraction, test_fraction)
            domain_stats["selected_records"] = int(domain_stats.get("selected_records") or 0) + 1
            domain_stats["selected_uncompressed_bytes"] = int(domain_stats.get("selected_uncompressed_bytes") or 0) + size
            domain_stats["selected_estimated_heavy_atoms"] = (
                int(domain_stats.get("selected_estimated_heavy_atoms") or 0)
                + int(row.get("estimated_heavy_atoms") or 0)
            )
            bin_stats["selected_records"] = int(bin_stats.get("selected_records") or 0) + 1
            bin_stats["selected_uncompressed_bytes"] = int(bin_stats.get("selected_uncompressed_bytes") or 0) + size
            yield row


def write_shards(rows: Iterable[dict[str, Any]], root: Path, shard_limit: int, output_root: str) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    shard_rows: Counter[str] = Counter()
    shard_bytes: Counter[str] = Counter()
    open_handles: dict[tuple[str, str, int], tuple[Any, Any, int]] = {}
    output_index: dict[tuple[str, str, int], int] = {}
    outputs: list[dict[str, Any]] = []
    try:
        for row in rows:
            domain = str(row["domain"])
            split = str(row["split"])
            shard_id = counters[f"{domain}:{split}"]
            handle_key = (domain, split, shard_id)
            proc_stream_size = open_handles.get(handle_key)
            if proc_stream_size is None:
                out = root / output_root / domain / split / f"part-{shard_id:05d}.jsonl.zst"
                proc, stream = zstd_writer(out)
                proc_stream_size = (proc, stream, 0)
                open_handles[handle_key] = proc_stream_size
                outputs.append({
                    "domain": domain,
                    "split": split,
                    "path": str(out),
                    "records": 0,
                    "uncompressed_bytes": 0,
                    "compressed_bytes": 0,
                })
                output_index[handle_key] = len(outputs) - 1
            proc, stream, used = proc_stream_size
            clean = {k: v for k, v in row.items() if not k.startswith("_sample_")}
            line = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            if used >= shard_limit and shard_rows[f"{domain}:{split}:{shard_id}"] > 0:
                finish_writer(proc, stream)
                del open_handles[handle_key]
                del output_index[handle_key]
                counters[f"{domain}:{split}"] += 1
                shard_id = counters[f"{domain}:{split}"]
                out = root / output_root / domain / split / f"part-{shard_id:05d}.jsonl.zst"
                proc, stream = zstd_writer(out)
                used = 0
                handle_key = (domain, split, shard_id)
                open_handles[handle_key] = (proc, stream, used)
                outputs.append({
                    "domain": domain,
                    "split": split,
                    "path": str(out),
                    "records": 0,
                    "uncompressed_bytes": 0,
                    "compressed_bytes": 0,
                })
                output_index[handle_key] = len(outputs) - 1
            stream.write(line)
            used += len(line)
            open_handles[handle_key] = (proc, stream, used)
            shard_rows[f"{domain}:{split}:{shard_id}"] += 1
            shard_bytes[f"{domain}:{split}:{shard_id}"] += len(line)
            outputs[output_index[handle_key]]["records"] += 1
            outputs[output_index[handle_key]]["uncompressed_bytes"] += len(line)
    finally:
        for proc, stream, _ in open_handles.values():
            finish_writer(proc, stream)
    total_compressed = 0
    for item in outputs:
        path = Path(str(item["path"]))
        compressed = path.stat().st_size if path.exists() else 0
        item["compressed_bytes"] = compressed
        total_compressed += compressed
    return {"outputs": outputs, "output_compressed_bytes": total_compressed}


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample and shard processed molscaling JSONL.zst files.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target-size-gb", type=float, default=None, help="Scale configured budgets to this total uncompressed size.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="Return success even when configured input files are missing.")
    args = ap.parse_args()

    config = load_yaml(args.config)
    root = Path(config["root"]).resolve()
    output_root = str(config.get("output_root") or "shards")
    lock_path = root / "reports" / "sample_corpus.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another sample_corpus is already running lock={lock_path}", flush=True)
        return 75
    budgets = config.get("budgets") or {}
    configured_total = sum(int(v.get("target_uncompressed_bytes", 0)) for v in budgets.values())
    target_scale = 1.0
    stats: dict[str, Any] | None = None
    if args.target_size_gb is not None and configured_total:
        if str(config.get("target_size_metric") or "uncompressed").lower() in {
            "compressed",
            "compressed_storage",
            "compressed_storage_estimate",
        }:
            target_scale, stats = estimate_scale_for_compressed_target(config, root, configured_total, args.target_size_gb)
        else:
            target_scale = (args.target_size_gb * 1024**3) / configured_total
    seed = int(config.get("seed", 0))
    if stats is None:
        stats = collect_inventory(config, root, target_scale)
        add_bin_targets(stats)
        stats["target_metric"] = str(config.get("target_size_metric") or "uncompressed")
    add_required_report_status(config, root, stats)
    rows = selected_rows(
        config,
        root,
        stats,
        seed,
        float(config.get("valid_fraction", 0.005)),
        float(config.get("test_fraction", 0.005)),
    )
    if not args.dry_run:
        shard_stats = write_shards(rows, root, int(config.get("shard_uncompressed_bytes", 1024**3)), output_root)
        stats.update(shard_stats)
    else:
        for _ in rows:
            pass
    missing_inputs = list(stats.get("missing") or [])
    missing_reports = list(stats.get("missing_required_reports") or [])
    if missing_inputs or missing_reports:
        stats["status"] = "missing_inputs"
        stats["missing_input_count"] = len(missing_inputs)
        stats["missing_required_report_count"] = len(missing_reports)
    else:
        stats["status"] = "ok"
        stats["missing_input_count"] = 0
        stats["missing_required_report_count"] = 0
    stats["created_utc"] = utc_now()
    stats["target_scale"] = target_scale
    stats["dry_run"] = bool(args.dry_run)
    stats["output_root"] = output_root

    reports = root / "reports"
    manifests = root / "manifests"
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = str(config.get("report_prefix") or "sample_report")
    latest_report_name = str(config.get("latest_report_name") or "sample_report.latest.json")
    manifest_name = str(config.get("manifest_name") or "dataset_manifest.json")
    report = reports / f"{report_prefix}_{stamp}.json"
    latest_report = reports / latest_report_name
    manifest = manifests / manifest_name
    text = json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    report.write_text(text, encoding="utf-8")
    latest_report.write_text(text, encoding="utf-8")
    if not args.dry_run:
        manifest.write_text(text, encoding="utf-8")
    print(text, end="")
    if (missing_inputs or missing_reports) and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
