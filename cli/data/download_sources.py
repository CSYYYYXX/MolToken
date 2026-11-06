#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"Cannot infer filename from URL: {url}")
    return name


def run_download(url: str, out_path: Path, log_path: Path, *, max_bytes: int | None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    cmd = [
        "curl",
        "-L",
        "--retry",
        "5",
        "--retry-delay",
        "10",
        "--continue-at",
        "-",
        "--output",
        str(tmp_path),
        url,
    ]
    if max_bytes is not None:
        cmd[1:1] = ["--range", f"0-{max_bytes - 1}"]
        # Range downloads are intentionally not resumable in smoke mode.
        cmd = [x for x in cmd if x not in {"--continue-at", "-"}]
        if tmp_path.exists():
            tmp_path.unlink()
    with log_path.open("ab") as log:
        log.write(f"\n=== DOWNLOAD {utc_now()} ===\n".encode())
        log.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, stdout=log, stderr=log)
    if proc.returncode == 0:
        tmp_path.replace(out_path)
    return int(proc.returncode)


def select_sources(config: dict[str, Any], names: set[str] | None, smoke: bool) -> list[dict[str, Any]]:
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Config missing list field: sources")
    if smoke:
        smoke_names = set(config.get("smoke_sources") or [])
        sources = [s for s in sources if s.get("name") in smoke_names]
    if names:
        sources = [s for s in sources if s.get("name") in names]
    return sources


def download_source(
    src: dict[str, Any],
    raw_root: Path,
    logs_root: Path,
    *,
    dry_run: bool,
    max_bytes: int | None,
) -> dict[str, Any]:
    if str(src.get("download_method") or "").strip() == "hf_dataset_snapshot":
        return download_hf_snapshot_source(src, raw_root, logs_root, dry_run=dry_run, max_bytes=max_bytes)

    name = str(src["name"])
    url = str(src["url"])
    rel = Path(str(src["raw_subdir"])) / filename_from_url(url)
    out_path = raw_root / rel
    log_path = logs_root / f"{name}.log"
    checksum_path = None
    row: dict[str, Any] = {
        "name": name,
        "domain": src.get("domain"),
        "source": src.get("source"),
        "source_version": src.get("source_version"),
        "url": url,
        "raw_path": str(out_path),
        "download_date_utc": utc_now(),
        "license_note": src.get("license_note"),
        "status": "dry_run" if dry_run else "pending",
    }
    if src.get("checksum_url"):
        checksum_url = str(src["checksum_url"])
        checksum_path = raw_root / Path(str(src["raw_subdir"])) / filename_from_url(checksum_url)
        row["checksum_url"] = checksum_url
        row["checksum_path"] = str(checksum_path)

    if dry_run:
        return row

    start = time.time()
    if checksum_path is not None and not checksum_path.exists():
        rc = run_download(str(src["checksum_url"]), checksum_path, logs_root / f"{name}.checksum.log", max_bytes=None)
        if rc != 0:
            row["checksum_download_rc"] = rc

    rc = run_download(url, out_path, log_path, max_bytes=max_bytes)
    row["status"] = "downloaded" if rc == 0 else "failed"
    row["returncode"] = rc
    row["elapsed_sec"] = time.time() - start
    row["size_bytes"] = file_size(out_path)
    if out_path.exists() and rc == 0:
        row["sha256"] = file_sha256(out_path)
    return row


def _hf_selected_files(snapshot: dict[str, Any], src: dict[str, Any]) -> list[str]:
    include_regex = str(src.get("hf_include_regex") or "").strip()
    exclude_regex = str(src.get("hf_exclude_regex") or "").strip()
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    files: list[str] = []
    for sibling in snapshot.get("siblings") or []:
        filename = str(sibling.get("rfilename") or "")
        if not filename:
            continue
        if include_re is not None and include_re.search(filename) is None:
            continue
        if exclude_re is not None and exclude_re.search(filename) is not None:
            continue
        files.append(filename)
    return sorted(set(files))


def _hf_files_from_config(src: dict[str, Any]) -> list[str]:
    raw_files = src.get("hf_files")
    if raw_files is None:
        return []
    if not isinstance(raw_files, list):
        raise ValueError(f"hf_files must be a list for source {src.get('name')}")
    return sorted(set(str(item) for item in raw_files if str(item).strip()))


def download_hf_snapshot_source(
    src: dict[str, Any],
    raw_root: Path,
    logs_root: Path,
    *,
    dry_run: bool,
    max_bytes: int | None,
) -> dict[str, Any]:
    name = str(src["name"])
    repo = str(src["hf_repo"])
    revision = str(src.get("hf_revision") or "main")
    api_url = str(src.get("url") or f"https://huggingface.co/api/datasets/{repo}?full=1")
    out_dir = raw_root / str(src["raw_subdir"])
    api_path = out_dir / "hf_api_snapshot.json"
    row: dict[str, Any] = {
        "name": name,
        "domain": src.get("domain"),
        "source": src.get("source"),
        "source_version": src.get("source_version"),
        "url": api_url,
        "hf_repo": repo,
        "hf_revision": revision,
        "raw_path": str(out_dir),
        "download_date_utc": utc_now(),
        "license_note": src.get("license_note"),
        "status": "dry_run" if dry_run else "pending",
    }
    if dry_run:
        return row

    out_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    selected = _hf_files_from_config(src)
    snapshot: dict[str, Any] = {}
    if selected:
        row["hf_files_from_config"] = True
    else:
        api_rc = run_download(api_url, api_path, logs_root / f"{name}.api.log", max_bytes=None)
        row["api_returncode"] = api_rc
        if api_rc != 0:
            row["status"] = "failed"
            return row

        snapshot = json.loads(api_path.read_text(encoding="utf-8"))
        if snapshot.get("error"):
            row["status"] = "failed"
            row["error"] = snapshot.get("error")
            return row
        row["hf_sha"] = snapshot.get("sha")
        row["hf_last_modified"] = snapshot.get("lastModified")
        selected = _hf_selected_files(snapshot, src)
    row["selected_file_count"] = len(selected)
    max_files = int(src.get("hf_max_files") or 0)
    if max_files > 0:
        selected = selected[:max_files]
        row["selected_file_count_after_limit"] = len(selected)

    file_rows: list[dict[str, Any]] = []
    started = time.time()
    for filename in selected:
        file_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(filename)}"
        out_path = out_dir / filename
        rc = run_download(file_url, out_path, logs_root / f"{name}.files.log", max_bytes=max_bytes)
        file_row: dict[str, Any] = {
            "rfilename": filename,
            "path": str(out_path),
            "returncode": rc,
            "size_bytes": file_size(out_path),
        }
        if rc == 0 and out_path.exists():
            file_row["sha256"] = file_sha256(out_path)
        file_rows.append(file_row)

    failed = [f for f in file_rows if f.get("returncode") != 0]
    row["status"] = "failed" if failed else "downloaded"
    row["returncode"] = 1 if failed else 0
    row["elapsed_sec"] = time.time() - started
    row["file_count"] = len(file_rows)
    row["size_bytes"] = sum(int(f.get("size_bytes") or 0) for f in file_rows)
    row["files"] = file_rows
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Download raw molscaling sources with manifests and logs.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", action="append", help="Download only this source name; can be repeated.")
    ap.add_argument("--smoke", action="store_true", help="Use config smoke_sources.")
    ap.add_argument("--max-bytes", type=int, default=None, help="Range-download at most N bytes per source for smoke tests.")
    ap.add_argument("--parallel", type=int, default=1, help="Download up to N selected sources concurrently.")
    args = ap.parse_args()

    config = load_yaml(args.config)
    root = Path(config["root"]).resolve()
    raw_root = root / "raw"
    manifests_root = root / "manifests"
    logs_root = root.parent / "logs" / "download"
    selected = select_sources(config, set(args.source or []) or None, args.smoke)
    if not selected:
        raise SystemExit("No sources selected.")

    print(json.dumps({
        "event": "download_plan",
        "root": str(root),
        "count": len(selected),
        "dry_run": args.dry_run,
        "smoke": args.smoke,
        "max_bytes": args.max_bytes,
        "parallel": max(1, args.parallel),
        "sources": [s.get("name") for s in selected],
    }, indent=2, ensure_ascii=False))

    manifest_rows: list[dict[str, Any]] = []
    parallel = max(1, args.parallel)
    if parallel == 1 or len(selected) == 1:
        for src in selected:
            manifest_rows.append(download_source(src, raw_root, logs_root, dry_run=args.dry_run, max_bytes=args.max_bytes))
    else:
        with futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            pending = [
                executor.submit(download_source, src, raw_root, logs_root, dry_run=args.dry_run, max_bytes=args.max_bytes)
                for src in selected
            ]
            for item in futures.as_completed(pending):
                manifest_rows.append(item.result())
        manifest_rows.sort(key=lambda row: str(row.get("name", "")))

    manifests_root.mkdir(parents=True, exist_ok=True)
    out_manifest = manifests_root / f"download_manifest_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest = manifests_root / "download_manifest.latest.json"
    payload = {
        "created_utc": utc_now(),
        "config": str(args.config),
        "dry_run": args.dry_run,
        "smoke": args.smoke,
        "max_bytes": args.max_bytes,
        "parallel": parallel,
        "sources": manifest_rows,
    }
    out_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "manifest_written", "path": str(out_manifest), "latest": str(latest)}, indent=2))
    failed = [r for r in manifest_rows if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
