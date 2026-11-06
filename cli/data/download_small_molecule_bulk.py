#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("./data")
PUBCHEM_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
ZINC_BASE = "https://files.docking.org/zinc20-ML/smiles"
ZINC_CHUNKS = [f"ZINC20_smiles_chunk_{i}.tar.gz" for i in range(1, 21)]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_tasks(root: Path, include_pubchem: bool, include_zinc: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if include_pubchem:
        tasks.append(
            {
                "name": "pubchem_cid_smiles_full_20260601",
                "domain": "small_molecule",
                "source": "PubChem",
                "source_version": "compound_extras_current_20260601",
                "url": PUBCHEM_URL,
                "path": str(root / "raw/small_molecule/pubchem_full_20260601/CID-SMILES.gz"),
            }
        )
    if include_zinc:
        for chunk in ZINC_CHUNKS:
            tasks.append(
                {
                    "name": f"zinc20_ml_smiles_{chunk.removesuffix('.tar.gz')}",
                    "domain": "small_molecule",
                    "source": "ZINC",
                    "source_version": "ZINC20-ML_smiles_2021_10",
                    "url": f"{ZINC_BASE}/{chunk}",
                    "path": str(root / "raw/small_molecule/zinc20_ml_smiles" / chunk),
                }
            )
    return tasks


def download_one(task: dict[str, Any], logs_root: Path, force: bool) -> dict[str, Any]:
    out = Path(task["path"])
    tmp = out.with_suffix(out.suffix + ".part")
    log_path = logs_root / f"{task['name']}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(task)
    row["started_utc"] = utc_now()
    if out.exists() and out.stat().st_size > 0 and not force:
        row["status"] = "skipped_existing"
        row["size_bytes"] = out.stat().st_size
        row["sha256"] = sha256_file(out)
        row["finished_utc"] = utc_now()
        return row

    cmd = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-delay",
        "10",
        "--retry-all-errors",
        "--connect-timeout",
        "60",
        "--continue-at",
        "-",
        "--output",
        str(tmp),
        str(task["url"]),
    ]
    start = time.time()
    with log_path.open("ab") as log:
        log.write(f"\n=== {utc_now()} ===\n".encode())
        log.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, stdout=log, stderr=log)
    row["returncode"] = int(proc.returncode)
    row["elapsed_sec"] = time.time() - start
    if proc.returncode == 0:
        tmp.replace(out)
        row["status"] = "downloaded"
        row["size_bytes"] = out.stat().st_size
        row["sha256"] = sha256_file(out)
    else:
        row["status"] = "failed"
        row["partial_size_bytes"] = tmp.stat().st_size if tmp.exists() else 0
    row["finished_utc"] = utc_now()
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--logs-root", default="")
    parser.add_argument("--no-pubchem", action="store_true")
    parser.add_argument("--no-zinc", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = Path(args.manifest) if args.manifest else root / "manifests" / f"download_small_molecule_bulk_{stamp}.json"
    logs_root = Path(args.logs_root) if args.logs_root else root / "logs" / f"download_small_molecule_bulk_{stamp}"
    tasks = build_tasks(root, include_pubchem=not args.no_pubchem, include_zinc=not args.no_zinc)

    print(f"{utc_now()} tasks={len(tasks)} parallel={args.parallel}", flush=True)
    rows: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futs = [pool.submit(download_one, task, logs_root, args.force) for task in tasks]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(
                f"{utc_now()} done {i}/{len(tasks)} {row['name']} {row['status']} size={row.get('size_bytes') or row.get('partial_size_bytes')}",
                flush=True,
            )

    status_counts: dict[str, int] = {}
    total_bytes = 0
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        total_bytes += int(row.get("size_bytes") or 0)
    result = {
        "created_utc": utc_now(),
        "root": str(root),
        "task_count": len(tasks),
        "parallel": args.parallel,
        "status_counts": status_counts,
        "total_size_bytes": total_bytes,
        "total_size_gb": total_bytes / 1e9,
        "logs_root": str(logs_root),
        "downloads": sorted(rows, key=lambda item: item["path"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_manifest.replace(manifest_path)
    latest = manifest_path.parent / "download_small_molecule_bulk.latest.json"
    latest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{utc_now()} wrote {manifest_path}", flush=True)
    return 0 if not status_counts.get("failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
