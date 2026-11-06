#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from cli.data import sample_corpus as base  # noqa: E402


def _empty_domain_stats() -> dict[str, Any]:
    return {
        "candidate_records": 0,
        "candidate_uncompressed_bytes": 0,
        "selected_records": 0,
        "selected_uncompressed_bytes": 0,
        "selected_estimated_heavy_atoms": 0,
        "bins": {},
    }


def _inventory_one(root_text: str, item: dict[str, Any], index: int) -> dict[str, Any]:
    root = Path(root_text)
    path = root / str(item["path"])
    domain = str(item["domain"])
    if not path.exists():
        return {"index": index, "path": str(path), "domain": domain, "missing": True}

    count = 0
    bytes_seen = 0
    domain_stats = _empty_domain_stats()
    for row in base.iter_jsonl_zst(path):
        size = base.row_size(row)
        sample_bin = base.domain_bin(row)
        count += 1
        bytes_seen += size
        domain_stats["candidate_records"] += 1
        domain_stats["candidate_uncompressed_bytes"] += size
        bin_stats = domain_stats["bins"].setdefault(sample_bin, {"records": 0, "uncompressed_bytes": 0})
        bin_stats["records"] += 1
        bin_stats["uncompressed_bytes"] += size

    return {
        "index": index,
        "path": str(path),
        "domain": domain,
        "missing": False,
        "records": count,
        "uncompressed_bytes": bytes_seen,
        "compressed_bytes": path.stat().st_size,
        "domain_stats": domain_stats,
    }


def _merge_inventory(config: dict[str, Any], root: Path, inventories: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "inputs": [],
        "missing": [],
        "streaming": True,
        "selection_strategy": "parallel_two_pass_hash_stratified_by_length_heavy_atom_bin",
        "domains": {},
    }
    for inv in sorted(inventories, key=lambda x: int(x["index"])):
        if inv.get("missing"):
            stats["missing"].append(str(inv["path"]))
            continue
        domain = str(inv["domain"])
        stats["inputs"].append(
            {
                "path": str(inv["path"]),
                "domain": domain,
                "records": int(inv["records"]),
                "uncompressed_bytes": int(inv["uncompressed_bytes"]),
                "compressed_bytes": int(inv["compressed_bytes"]),
            }
        )
        merged = stats["domains"].setdefault(domain, _empty_domain_stats())
        src = inv["domain_stats"]
        merged["candidate_records"] += int(src["candidate_records"])
        merged["candidate_uncompressed_bytes"] += int(src["candidate_uncompressed_bytes"])
        for bin_name, bin_stats in (src.get("bins") or {}).items():
            out_bin = merged["bins"].setdefault(str(bin_name), {"records": 0, "uncompressed_bytes": 0})
            out_bin["records"] += int(bin_stats.get("records") or 0)
            out_bin["uncompressed_bytes"] += int(bin_stats.get("uncompressed_bytes") or 0)
    for domain, domain_stats in stats["domains"].items():
        target = int((config.get("budgets") or {}).get(domain, {}).get("target_uncompressed_bytes", 0))
        domain_stats["target_uncompressed_bytes"] = target
    return stats


def _estimate_scaled_stats(
    config: dict[str, Any],
    baseline: dict[str, Any],
    configured_total: int,
    target_gb: float | None,
) -> tuple[float, dict[str, Any]]:
    target_scale = 1.0
    stats = copy.deepcopy(baseline)
    if target_gb is None or not configured_total:
        base.add_bin_targets(stats)
        stats["target_metric"] = str(config.get("target_size_metric") or "uncompressed")
        return target_scale, stats

    metric = str(config.get("target_size_metric") or "uncompressed").lower()
    if metric in {"compressed", "compressed_storage", "compressed_storage_estimate"}:
        target_bytes = int(target_gb * 1024**3)
        base.add_bin_targets(stats)
        compressed_inputs = sum(int(item.get("compressed_bytes") or 0) for item in stats.get("inputs") or [])
        candidate_uncompressed = base.total_candidate_uncompressed(stats)
        selected_uncompressed = base.projected_selected_uncompressed(stats)
        if compressed_inputs <= 0 or candidate_uncompressed <= 0 or selected_uncompressed <= 0:
            stats["target_metric"] = "compressed_storage_estimate"
            stats["target_compressed_bytes"] = target_bytes
            stats["estimated_input_compression_ratio"] = None
            return 1.0, stats

        compression_ratio = compressed_inputs / candidate_uncompressed
        desired_uncompressed = int(target_bytes / compression_ratio)
        max_scale = candidate_uncompressed / configured_total
        if desired_uncompressed >= candidate_uncompressed:
            target_scale = max_scale
        else:
            lo, hi = 0.0, max(max_scale, 1.0)
            for _ in range(48):
                mid = (lo + hi) / 2.0
                if base.projected_uncompressed_for_scale(config, stats, mid) < desired_uncompressed:
                    lo = mid
                else:
                    hi = mid
            target_scale = hi

        scaled = copy.deepcopy(baseline)
        base.apply_domain_targets(config, scaled, target_scale)
        base.add_bin_targets(scaled)
        scaled["target_metric"] = "compressed_storage_estimate"
        scaled["target_compressed_bytes"] = target_bytes
        scaled["estimated_input_compression_ratio"] = compression_ratio
        scaled["estimated_selected_compressed_bytes"] = int(
            base.projected_selected_uncompressed(scaled) * compression_ratio
        )
        scaled["compressed_input_bytes"] = compressed_inputs
        scaled["candidate_uncompressed_bytes_total"] = candidate_uncompressed
        return target_scale, scaled

    target_scale = (target_gb * 1024**3) / configured_total
    base.apply_domain_targets(config, stats, target_scale)
    base.add_bin_targets(stats)
    stats["target_metric"] = str(config.get("target_size_metric") or "uncompressed")
    return target_scale, stats


def _zstd_writer(path: Path, zstd_threads: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    zstd = base.find_zstd()
    if zstd is None:
        raise RuntimeError(f"zstd executable is required for compressed output: {path}")
    level = os.environ.get("MOLSCALING_ZSTD_LEVEL", "3")
    proc = subprocess.Popen(
        [zstd, f"-T{max(1, zstd_threads)}", f"-{level}", "-f", "-o", str(path)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    return proc, proc.stdin


def _finish_writer(proc: subprocess.Popen[bytes] | None, stream: Any) -> None:
    stream.close()
    if proc is not None:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"zstd failed with rc={rc}")


def _write_selected_one(
    root_text: str,
    output_root: str,
    item: dict[str, Any],
    index: int,
    domain_stats: dict[str, Any],
    seed: int,
    valid_fraction: float,
    test_fraction: float,
    shard_limit: int,
    zstd_threads: int,
) -> dict[str, Any]:
    root = Path(root_text)
    path = root / str(item["path"])
    domain = str(item["domain"])
    bins = domain_stats.get("bins") or {}
    selected_records = 0
    selected_uncompressed = 0
    selected_heavy = 0
    bin_selected: dict[str, dict[str, int]] = {}
    outputs: list[dict[str, Any]] = []

    counters: Counter[str] = Counter()
    shard_rows: Counter[str] = Counter()
    handles: dict[str, tuple[Any, Any, int, int, int]] = {}

    def open_for(split: str) -> tuple[Any, Any, int, int, int]:
        shard_id = counters[split]
        out = root / output_root / domain / split / f"part-input{index:03d}-{shard_id:05d}.jsonl.zst"
        proc, stream = _zstd_writer(out, zstd_threads)
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
        for row in base.iter_jsonl_zst(path):
            size = base.row_size(row)
            sample_bin = base.domain_bin(row)
            bin_stats = bins.get(sample_bin) or {}
            probability = float(bin_stats.get("selection_probability") or 0.0)
            if probability <= 0.0:
                continue
            key = str(row.get("sha1") or f"{domain}|{row.get('normalized')}")
            draw_key = f"select|{domain}|{sample_bin}|{key}"
            if probability < 1.0 and base.stable_unit_interval(draw_key, seed) >= probability:
                continue

            split = base.split_for(row, seed, valid_fraction, test_fraction)
            handle = handles.get(split)
            if handle is None:
                handle = open_for(split)
                handles[split] = handle
            proc, stream, used, shard_id, out_index = handle
            if used >= shard_limit and shard_rows[f"{split}:{shard_id}"] > 0:
                _finish_writer(proc, stream)
                counters[split] += 1
                handle = open_for(split)
                handles[split] = handle
                proc, stream, used, shard_id, out_index = handle

            row["split"] = split
            clean = {k: v for k, v in row.items() if not k.startswith("_sample_")}
            line = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            stream.write(line)
            used += len(line)
            handles[split] = (proc, stream, used, shard_id, out_index)
            shard_rows[f"{split}:{shard_id}"] += 1

            selected_records += 1
            selected_uncompressed += size
            selected_heavy += int(row.get("estimated_heavy_atoms") or 0)
            out_item = outputs[out_index]
            out_item["records"] += 1
            out_item["uncompressed_bytes"] += len(line)
            bin_out = bin_selected.setdefault(sample_bin, {"selected_records": 0, "selected_uncompressed_bytes": 0})
            bin_out["selected_records"] += 1
            bin_out["selected_uncompressed_bytes"] += size
    finally:
        for proc, stream, _, _, _ in handles.values():
            _finish_writer(proc, stream)

    compressed_total = 0
    for item_out in outputs:
        out_path = Path(str(item_out["path"]))
        compressed = out_path.stat().st_size if out_path.exists() else 0
        item_out["compressed_bytes"] = compressed
        compressed_total += compressed

    return {
        "index": index,
        "domain": domain,
        "selected_records": selected_records,
        "selected_uncompressed_bytes": selected_uncompressed,
        "selected_estimated_heavy_atoms": selected_heavy,
        "bins": bin_selected,
        "outputs": outputs,
        "output_compressed_bytes": compressed_total,
    }


def _merge_selection(stats: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    output_compressed = 0
    for result in sorted(results, key=lambda x: int(x["index"])):
        domain = str(result["domain"])
        domain_stats = stats["domains"][domain]
        domain_stats["selected_records"] += int(result["selected_records"])
        domain_stats["selected_uncompressed_bytes"] += int(result["selected_uncompressed_bytes"])
        domain_stats["selected_estimated_heavy_atoms"] += int(result["selected_estimated_heavy_atoms"])
        for bin_name, bin_result in (result.get("bins") or {}).items():
            bin_stats = domain_stats["bins"].setdefault(str(bin_name), {})
            bin_stats["selected_records"] = int(bin_stats.get("selected_records") or 0) + int(
                bin_result.get("selected_records") or 0
            )
            bin_stats["selected_uncompressed_bytes"] = int(
                bin_stats.get("selected_uncompressed_bytes") or 0
            ) + int(bin_result.get("selected_uncompressed_bytes") or 0)
        outputs.extend(result.get("outputs") or [])
        output_compressed += int(result.get("output_compressed_bytes") or 0)
    stats["outputs"] = outputs
    stats["output_compressed_bytes"] = output_compressed
    return stats


def _replace_output_root_in_paths(stats: dict[str, Any], old: Path, new: Path) -> None:
    old_text = str(old)
    new_text = str(new)
    for item in stats.get("outputs") or []:
        path = str(item.get("path") or "")
        if path.startswith(old_text):
            item["path"] = new_text + path[len(old_text) :]


def _write_reports(config: dict[str, Any], root: Path, stats: dict[str, Any]) -> None:
    reports = root / "reports"
    manifests = root / "manifests"
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = str(config.get("report_prefix") or "sample_report")
    latest_report_name = str(config.get("latest_report_name") or "sample_report.latest.json")
    manifest_name = str(config.get("manifest_name") or "dataset_manifest.json")
    text = json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (reports / f"{report_prefix}_{stamp}.json").write_text(text, encoding="utf-8")
    (reports / latest_report_name).write_text(text, encoding="utf-8")
    (manifests / manifest_name).write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel sample and shard molscaling JSONL.zst files.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-size-gb", type=float, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--temp-output-root", type=str, default=None)
    parser.add_argument("--activate-temp-output", action="store_true")
    parser.add_argument("--zstd-threads", type=int, default=1)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    config = base.load_yaml(args.config)
    root = Path(config["root"]).resolve()
    output_root = args.output_root or str(config.get("output_root") or "shards")
    active_output_root = args.temp_output_root or output_root
    final_output_path = root / output_root
    active_output_path = root / active_output_root
    inputs = list(config.get("inputs") or [])
    workers = args.workers or min(os.cpu_count() or 1, len(inputs) or 1)
    workers = max(1, min(workers, len(inputs) or workers))

    if active_output_path.exists() and any(active_output_path.iterdir()):
        raise RuntimeError(f"active output root already exists and is not empty: {active_output_path}")
    if args.activate_temp_output and active_output_path == final_output_path:
        raise RuntimeError("--activate-temp-output requires --temp-output-root different from --output-root")
    if args.activate_temp_output and final_output_path.exists() and any(final_output_path.iterdir()):
        raise RuntimeError(f"final output root already exists and is not empty: {final_output_path}")

    lock_path = root / "reports" / "sample_corpus.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another sample_corpus is already running lock={lock_path}", flush=True)
        return 75

    print(f"{dt.datetime.now().isoformat(timespec='seconds')} parallel inventory start workers={workers}", flush=True)
    inventories: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_inventory_one, str(root), item, index)
            for index, item in enumerate(inputs)
        ]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            inventories.append(result)
            print(
                f"{dt.datetime.now().isoformat(timespec='seconds')} inventory progress "
                f"done={done}/{len(futures)} domain={result.get('domain')} path={Path(str(result.get('path'))).name}",
                flush=True,
            )

    baseline = _merge_inventory(config, root, inventories)
    configured_total = sum(int(v.get("target_uncompressed_bytes", 0)) for v in (config.get("budgets") or {}).values())
    target_scale, stats = _estimate_scaled_stats(config, baseline, configured_total, args.target_size_gb)
    base.add_required_report_status(config, root, stats)

    missing_inputs = list(stats.get("missing") or [])
    missing_reports = list(stats.get("missing_required_reports") or [])
    if missing_inputs or missing_reports:
        stats["status"] = "missing_inputs"
        stats["missing_input_count"] = len(missing_inputs)
        stats["missing_required_report_count"] = len(missing_reports)
        stats["created_utc"] = base.utc_now()
        stats["target_scale"] = target_scale
        stats["output_root"] = output_root
        _write_reports(config, root, stats)
        if not args.allow_missing:
            return 2

    print(f"{dt.datetime.now().isoformat(timespec='seconds')} parallel selection start workers={workers}", flush=True)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _write_selected_one,
                str(root),
                active_output_root,
                item,
                index,
                copy.deepcopy((stats.get("domains") or {}).get(str(item["domain"]), {})),
                int(config.get("seed", 0)),
                float(config.get("valid_fraction", 0.005)),
                float(config.get("test_fraction", 0.005)),
                int(config.get("shard_uncompressed_bytes", 1024**3)),
                int(args.zstd_threads),
            )
            for index, item in enumerate(inputs)
            if (root / str(item["path"])).exists()
        ]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"{dt.datetime.now().isoformat(timespec='seconds')} selection progress "
                f"done={done}/{len(futures)} domain={result.get('domain')} "
                f"records={result.get('selected_records')}",
                flush=True,
            )

    _merge_selection(stats, results)
    if args.activate_temp_output:
        if final_output_path.exists():
            if any(final_output_path.iterdir()):
                raise RuntimeError(f"final output root became non-empty: {final_output_path}")
            final_output_path.rmdir()
        active_output_path.rename(final_output_path)
        _replace_output_root_in_paths(stats, active_output_path, final_output_path)

    stats["status"] = "ok"
    stats["missing_input_count"] = 0
    stats["missing_required_report_count"] = 0
    stats["created_utc"] = base.utc_now()
    stats["target_scale"] = target_scale
    stats["dry_run"] = False
    stats["parallel_workers"] = workers
    stats["zstd_threads"] = int(args.zstd_threads)
    stats["output_root"] = output_root
    if args.temp_output_root:
        stats["temp_output_root"] = active_output_root
        stats["activated_temp_output"] = bool(args.activate_temp_output)

    _write_reports(config, root, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
