#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


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


def iter_jsonl_zst(path: Path) -> Iterable[dict]:
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


def iter_input_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*.jsonl.zst") if p.is_file())
    return [path]


def validate_one(path: Path, max_records: int | None, seen_global: set[str] | None = None) -> tuple[dict, int]:
    counts: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    seen_normalized: set[str] = set()
    duplicates = 0
    duplicates_cross_file = 0
    missing_fields = 0
    rdkit_ok = 0
    rdkit_fail = 0
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for corpus validation") from exc
    lengths = []
    heavy_atoms = []
    for i, row in enumerate(iter_jsonl_zst(path), 1):
        if max_records is not None and i > max_records:
            break
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            missing_fields += 1
        norm = str(row.get("normalized", ""))
        seen_in_current_file = norm in seen_normalized
        if seen_in_current_file:
            duplicates += 1
        if seen_global is not None:
            if norm in seen_global and not seen_in_current_file:
                duplicates_cross_file += 1
            seen_global.add(norm)
        seen_normalized.add(norm)
        domain = str(row.get("domain", "unknown"))
        source = str(row.get("source", "unknown"))
        domains[domain] += 1
        sources[source] += 1
        counts[str(row.get("filter_status", "unknown"))] += 1
        if isinstance(row.get("length"), int):
            lengths.append(row["length"])
        if isinstance(row.get("estimated_heavy_atoms"), int):
            heavy_atoms.append(row["estimated_heavy_atoms"])
        if domain == "small_molecule":
            mol = Chem.MolFromSmiles(norm)
            if mol is None:
                rdkit_fail += 1
            else:
                rdkit_ok += 1
    total = sum(domains.values())
    return {
        "input": str(path),
        "records_checked": total,
        "domains": dict(domains),
        "sources": dict(sources),
        "filter_status": dict(counts),
        "duplicates_in_checked": duplicates,
        "duplicates_cross_files_in_checked": duplicates_cross_file,
        "missing_required_fields": missing_fields,
        "rdkit_ok": rdkit_ok,
        "rdkit_fail": rdkit_fail,
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "heavy_atoms_min": min(heavy_atoms) if heavy_atoms else None,
        "heavy_atoms_max": max(heavy_atoms) if heavy_atoms else None,
    }, total


def validate(path: Path, max_records: int | None) -> dict:
    paths = iter_input_paths(path)
    if not paths:
        return {
            "input": str(path),
            "input_kind": "directory" if path.is_dir() else "file",
            "files_checked": 0,
            "records_checked": 0,
            "missing_required_fields": 0,
        "duplicates_in_checked": 0,
        "duplicates_cross_files_in_checked": 0,
        "rdkit_fail": 0,
        "error": "no_jsonl_zst_files_found",
        }

    aggregate: dict = {
        "input": str(path),
        "input_kind": "directory" if path.is_dir() else "file",
        "files_checked": 0,
        "records_checked": 0,
        "domains": {},
        "sources": {},
        "filter_status": {},
        "duplicates_in_checked": 0,
        "duplicates_cross_files_in_checked": 0,
        "missing_required_fields": 0,
        "rdkit_ok": 0,
        "rdkit_fail": 0,
        "length_min": None,
        "length_max": None,
        "heavy_atoms_min": None,
        "heavy_atoms_max": None,
        "files": [],
    }
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    remaining = max_records
    seen_global: set[str] | None = set() if path.is_dir() else None
    for file_path in paths:
        if remaining is not None and remaining <= 0:
            break
        file_result, checked = validate_one(file_path, remaining, seen_global)
        aggregate["files_checked"] += 1
        aggregate["records_checked"] += int(file_result.get("records_checked") or 0)
        aggregate["duplicates_in_checked"] += int(file_result.get("duplicates_in_checked") or 0)
        aggregate["duplicates_cross_files_in_checked"] += int(file_result.get("duplicates_cross_files_in_checked") or 0)
        aggregate["missing_required_fields"] += int(file_result.get("missing_required_fields") or 0)
        aggregate["rdkit_ok"] += int(file_result.get("rdkit_ok") or 0)
        aggregate["rdkit_fail"] += int(file_result.get("rdkit_fail") or 0)
        for key, value in (file_result.get("domains") or {}).items():
            domain_counts[str(key)] += int(value)
        for key, value in (file_result.get("sources") or {}).items():
            source_counts[str(key)] += int(value)
        for key, value in (file_result.get("filter_status") or {}).items():
            status_counts[str(key)] += int(value)
        for field in ["length_min", "heavy_atoms_min"]:
            value = file_result.get(field)
            if value is not None:
                aggregate[field] = value if aggregate[field] is None else min(aggregate[field], value)
        for field in ["length_max", "heavy_atoms_max"]:
            value = file_result.get(field)
            if value is not None:
                aggregate[field] = value if aggregate[field] is None else max(aggregate[field], value)
        aggregate["files"].append(file_result)
        if remaining is not None:
            remaining -= checked
    aggregate["domains"] = dict(domain_counts)
    aggregate["sources"] = dict(source_counts)
    aggregate["filter_status"] = dict(status_counts)
    return aggregate


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate normalized molscaling JSONL.zst corpus shards.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Do not fail on duplicate normalized records. Use only before the clean/dedup stage.",
    )
    args = ap.parse_args()
    result = validate(args.input, args.max_records)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    duplicate_failure = (
        int(result.get("duplicates_in_checked") or 0)
        or int(result.get("duplicates_cross_files_in_checked") or 0)
    )
    return 1 if (
        result["missing_required_fields"]
        or (duplicate_failure and not args.allow_duplicates)
        or result["rdkit_fail"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
