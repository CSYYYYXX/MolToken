#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT = Path("./data")
REPO = "introvoyz041/ZINC20"
REVISION = "719b18a4d149ebcb38bd45553edfbb7d95b31e4b"
API_URL = f"https://huggingface.co/api/datasets/{REPO}?full=1"


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


def run_curl(url: str, out: Path, log: Path, *, resume: bool = True) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
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
        "--output",
        str(out),
        url,
    ]
    if resume:
        cmd[1:1] = ["--continue-at", "-"]
    with log.open("ab") as f:
        f.write(f"\n=== {utc_now()} ===\n".encode())
        f.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, stdout=f, stderr=f)
    return int(proc.returncode)


def fetch_api_snapshot(root: Path, logs_root: Path) -> dict[str, Any]:
    out_dir = root / "raw/small_molecule/zinc20_hf_parquet"
    api_path = out_dir / "hf_api_snapshot.json"
    tmp = api_path.with_suffix(".json.tmp")
    rc = run_curl(API_URL, tmp, logs_root / "hf_api_snapshot.log", resume=False)
    if rc != 0:
        raise RuntimeError(f"failed to fetch HF API snapshot rc={rc}")
    tmp.replace(api_path)
    return json.loads(api_path.read_text(encoding="utf-8"))


def build_tasks(root: Path, snapshot: dict[str, Any], include_sidecars: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for sibling in snapshot.get("siblings") or []:
        name = str(sibling.get("rfilename") or "")
        if not name:
            continue
        if name.endswith(".parquet") or (include_sidecars and name in {"README.md", ".gitattributes"}):
            encoded = quote(name, safe="/")
            tasks.append(
                {
                    "name": name,
                    "domain": "small_molecule",
                    "source": "ZINC",
                    "source_version": "ZINC20_HuggingFace_mirror_introvoyz041_719b18a",
                    "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{encoded}?download=true",
                    "path": str(root / "raw/small_molecule/zinc20_hf_parquet" / name),
                }
            )
    return sorted(tasks, key=lambda item: item["path"])


def download_one(task: dict[str, Any], logs_root: Path, force: bool, do_sha256: bool) -> dict[str, Any]:
    out = Path(task["path"])
    tmp = out.with_suffix(out.suffix + ".part")
    row = dict(task)
    row["started_utc"] = utc_now()
    if out.exists() and out.stat().st_size > 0 and not force:
        row["status"] = "skipped_existing"
        row["size_bytes"] = out.stat().st_size
        if do_sha256:
            row["sha256"] = sha256_file(out)
        row["finished_utc"] = utc_now()
        return row
    start = time.time()
    rc = run_curl(str(task["url"]), tmp, logs_root / (out.name + ".log"), resume=True)
    row["returncode"] = rc
    row["elapsed_sec"] = time.time() - start
    if rc == 0:
        tmp.replace(out)
        row["status"] = "downloaded"
        row["size_bytes"] = out.stat().st_size
        if do_sha256:
            row["sha256"] = sha256_file(out)
    else:
        row["status"] = "failed"
        row["partial_size_bytes"] = tmp.stat().st_size if tmp.exists() else 0
    row["finished_utc"] = utc_now()
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--logs-root", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--no-sidecars", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_root = Path(args.logs_root) if args.logs_root else root / "logs" / f"download_zinc20_hf_parquet_{stamp}"
    manifest_path = Path(args.manifest) if args.manifest else root / "manifests" / f"download_zinc20_hf_parquet_{stamp}.json"
    snapshot = fetch_api_snapshot(root, logs_root)
    tasks = build_tasks(root, snapshot, include_sidecars=not args.no_sidecars)
    print(f"{utc_now()} tasks={len(tasks)} parallel={args.parallel}", flush=True)

    rows: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futs = [pool.submit(download_one, task, logs_root, args.force, args.sha256) for task in tasks]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if i % 10 == 0 or row["status"] == "failed" or i == len(tasks):
                print(
                    f"{utc_now()} done {i}/{len(tasks)} {row['status']} {row['name']} size={row.get('size_bytes') or row.get('partial_size_bytes')}",
                    flush=True,
                )

    status_counts: dict[str, int] = {}
    total_bytes = 0
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        total_bytes += int(row.get("size_bytes") or 0)
    result = {
        "created_utc": utc_now(),
        "repo": REPO,
        "revision": REVISION,
        "api_url": API_URL,
        "root": str(root),
        "task_count": len(tasks),
        "parallel": args.parallel,
        "status_counts": status_counts,
        "total_size_bytes": total_bytes,
        "total_size_gb": total_bytes / 1e9,
        "logs_root": str(logs_root),
        "official_zinc_direct_status": "files.docking.org returned HTTP 403 from this cluster worker/proxy; using this HuggingFace mirror as the downloadable raw source.",
        "downloads": sorted(rows, key=lambda item: item["path"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_manifest.replace(manifest_path)
    latest = manifest_path.parent / "download_zinc20_hf_parquet.latest.json"
    latest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{utc_now()} wrote {manifest_path}", flush=True)
    return 0 if not status_counts.get("failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
