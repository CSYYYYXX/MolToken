#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


AA_HEAVY_ATOMS = {
    "A": 5, "R": 11, "N": 8, "D": 8, "C": 6, "Q": 9, "E": 9, "G": 4,
    "H": 10, "I": 8, "L": 8, "K": 9, "M": 8, "F": 11, "P": 7, "S": 6,
    "T": 7, "W": 14, "Y": 12, "V": 7,
}

# Coarse nucleotide-template estimates for sampling balance. Exact graph expansion
# will be done later from templates.
DNA_HEAVY_ATOMS = {"A": 21, "C": 19, "G": 22, "T": 20}
RNA_HEAVY_ATOMS = {"A": 22, "C": 20, "G": 23, "U": 20}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


KNOWN_ZSTD = [
    
    "/usr/bin/zstd",
    "/usr/local/bin/zstd",
]
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def find_zstd() -> str | None:
    found = shutil.which("zstd")
    if found:
        return found
    for candidate in KNOWN_ZSTD:
        if Path(candidate).exists():
            return candidate
    return None


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_fraction(text: str) -> float:
    value = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)
    return value / float(16**16)


class Deduper:
    def __init__(self, cfg: dict[str, Any], source_name: str):
        dedup_cfg = cfg.get("dedup") or {}
        self.backend = str(dedup_cfg.get("backend") or "memory")
        self.count = 0
        self._seen: set[str] | None = set() if self.backend == "memory" else None
        self._conn: sqlite3.Connection | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._pending = 0
        if self.backend == "sqlite":
            dedup_dir = Path(str(dedup_cfg.get("dir") or str(Path(tempfile.gettempdir()) / "moltoken_dedup")))
            dedup_dir.mkdir(parents=True, exist_ok=True)
            self._tmpdir = tempfile.TemporaryDirectory(prefix=f"{source_name}.", dir=str(dedup_dir))
            db_path = Path(self._tmpdir.name) / "seen.sqlite"
            self._conn = sqlite3.connect(str(db_path))
            self._conn.execute("PRAGMA journal_mode=OFF")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute("CREATE TABLE seen (sha1 TEXT PRIMARY KEY)")
        elif self.backend != "memory":
            raise ValueError(f"Unsupported dedup backend: {self.backend}")

    def add(self, key: str) -> bool:
        if self._seen is not None:
            if key in self._seen:
                return False
            self._seen.add(key)
            self.count += 1
            return True
        assert self._conn is not None
        cur = self._conn.execute("INSERT OR IGNORE INTO seen (sha1) VALUES (?)", (key,))
        self._pending += 1
        if self._pending >= 100000:
            self._conn.commit()
            self._pending = 0
        if cur.rowcount == 0:
            return False
        self.count += 1
        return True

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


def open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


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


@contextlib.contextmanager
def output_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def process_all_global_lock(root: Path):
    if os.environ.get("MOLSCALING_DISABLE_PROCESS_ALL_GLOBAL_LOCK"):
        yield
        return
    lock_path = root / "reports" / "process_all.global.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{dt.datetime.now().isoformat(timespec='seconds')} waiting process_all global lock={lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} acquired process_all global lock={lock_path}", flush=True)
        try:
            yield
        finally:
            print(f"{dt.datetime.now().isoformat(timespec='seconds')} releasing process_all global lock={lock_path}", flush=True)


def skipped_existing_result(source: dict[str, Any], root: Path, domain: str, out_path: Path) -> dict[str, Any] | None:
    done_path = root / "reports" / f"process_report_{source['name']}.done.json"
    if (
        source.get("skip_if_output_exists")
        and out_path.exists()
        and out_path.stat().st_size > 0
        and done_path.exists()
        and done_path.stat().st_size > 0
    ):
        return {
            "name": source["name"],
            "domain": domain,
            "input": str(root / source["input"]),
            "output": str(out_path),
            "counts": {"skipped_existing_output": 1},
            "kept": 0,
            "uncompressed_bytes": 0,
            "created_utc": utc_now(),
        }
    return None


def source_tmp_output(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def finalize_source_output(tmp_path: Path, out_path: Path) -> None:
    tmp_path.replace(out_path)


def write_source_done(root: Path, result: dict[str, Any]) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    done = reports / f"process_report_{result['name']}.done.json"
    tmp = done.with_suffix(done.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(done)


def split_ints(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_header_id(header: str) -> str:
    token = header.strip().split()[0] if header.strip() else "unknown"
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return token.lstrip(">")


def iter_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    header: str | None = None
    seq_parts: list[str] = []
    with open_text_auto(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    seq = re.sub(r"\s+", "", "".join(seq_parts)).upper()
                    yield parse_header_id(header), header, seq
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            seq = re.sub(r"\s+", "", "".join(seq_parts)).upper()
            yield parse_header_id(header), header, seq


def iter_chembl_chemreps(path: Path) -> Iterable[tuple[str, str]]:
    with open_text_auto(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        lower = [h.lower() for h in header]
        try:
            id_idx = lower.index("chembl_id")
        except ValueError:
            id_idx = 0
        smiles_idx = lower.index("canonical_smiles") if "canonical_smiles" in lower else 1
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_idx, smiles_idx):
                continue
            yield parts[id_idx], parts[smiles_idx]


def iter_pubchem_cid_smiles(path: Path) -> Iterable[tuple[str, str]]:
    with open_text_auto(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]


def iter_pepbenchmark_raw(input_dir: Path) -> Iterator[tuple[str, str, dict[str, Any]]]:
    base_filenames = {"pos.csv", "neg.csv", "raw.csv", "all.csv", "pos_without_X.csv"}
    sequence_columns = [
        "sequence",
        "seq",
        "peptide",
        "peptide_sequence",
        "peptide_seq",
        "pep_seq",
        "fasta",
    ]
    for csv_path in sorted(input_dir.rglob("*.csv")):
        if csv_path.name not in base_filenames:
            continue
        rel = csv_path.relative_to(input_dir).as_posix()
        parts = rel.split("/")
        task_group = parts[1] if len(parts) > 1 else ""
        dataset_kind = parts[2] if len(parts) > 2 else ""
        dataset_name = parts[3] if len(parts) > 3 else (parts[-2] if len(parts) > 1 else "")
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            lower_to_field = {field.strip().lower(): field for field in reader.fieldnames}
            selected_fields = [lower_to_field[col] for col in sequence_columns if col in lower_to_field]
            if "prot_seq" in lower_to_field:
                selected_fields = [field for field in selected_fields if field != lower_to_field["prot_seq"]]
            if not selected_fields:
                continue
            for row_index, row in enumerate(reader):
                for field in selected_fields:
                    seq = str(row.get(field) or "").strip()
                    if not seq:
                        continue
                    original_id = f"{rel}:{row_index}:{field}"
                    metadata = {
                        "raw_file": rel,
                        "csv_column": field,
                        "csv_row": row_index,
                        "task_group": task_group,
                        "dataset_kind": dataset_kind,
                        "dataset_name": dataset_name,
                    }
                    if "label" in row:
                        metadata["label"] = row.get("label")
                    yield original_id, seq, metadata


def canonicalize_smiles(raw: str, cfg: dict[str, Any]) -> tuple[str | None, int | None, str]:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except Exception as exc:
        raise SystemExit(f"RDKit is required for small molecule processing: {exc}") from exc
    RDLogger.DisableLog("rdApp.*")
    raw = raw.strip()
    if not raw:
        return None, None, "empty"
    if "." in raw and cfg.get("mixture_policy", "reject") == "reject":
        return None, None, "mixture"
    mol = Chem.MolFromSmiles(raw, sanitize=True)
    if mol is None:
        return None, None, "rdkit_parse_failed"
    allowed = set(cfg.get("allowed_elements") or [])
    if allowed:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in allowed:
                return None, None, f"element_{atom.GetSymbol()}"
    heavy = int(mol.GetNumHeavyAtoms())
    if heavy < int(cfg.get("heavy_atoms_min", 1)):
        return None, heavy, "too_few_heavy_atoms"
    if heavy > int(cfg.get("heavy_atoms_max", 10**9)):
        return None, heavy, "too_many_heavy_atoms"
    normalized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=bool(cfg.get("isomeric_smiles", True)))
    return normalized, heavy, "ok"


def write_row(stream, row: dict[str, Any]) -> int:
    line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    stream.write(line)
    return len(line)


def log_progress(kind: str, source: dict[str, Any], seen: int, kept: int, out_bytes: int) -> None:
    print(
        f"{dt.datetime.now().isoformat(timespec='seconds')} process progress "
        f"kind={kind} source={source.get('name')} seen={seen} kept={kept} "
        f"out_uncompressed={out_bytes}",
        flush=True,
    )


def passes_source_sampling(row_hash: str, source: dict[str, Any]) -> bool:
    upper = float(source.get("keep_fraction") or 1.0)
    lower = float(source.get("keep_min_fraction") or 0.0)
    if upper <= lower or upper <= 0.0:
        return False
    if lower <= 0.0 and upper >= 1.0:
        return True
    salt = str(source.get("sample_seed") or source.get("name") or "")
    value = stable_fraction(f"{salt}|{row_hash}")
    return lower <= value < upper


def passes_input_sampling(key: str, source: dict[str, Any]) -> bool:
    upper = float(source.get("input_keep_fraction") or 1.0)
    lower = float(source.get("input_keep_min_fraction") or 0.0)
    if upper <= lower or upper <= 0.0:
        return False
    if lower <= 0.0 and upper >= 1.0:
        return True
    salt = str(source.get("sample_seed") or source.get("name") or "")
    value = stable_fraction(f"input|{salt}|{key}")
    return lower <= value < upper


def process_smiles_source(
    source: dict[str, Any],
    root: Path,
    cfg: dict[str, Any],
    max_records: int | None,
    iterator: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    out_path = root / source["output"]
    with output_lock(out_path):
        skipped = skipped_existing_result(source, root, "small_molecule", out_path)
        if skipped is not None:
            return skipped
        return _process_smiles_source_locked(source, root, cfg, max_records, iterator, out_path)


def _process_smiles_source_locked(
    source: dict[str, Any],
    root: Path,
    cfg: dict[str, Any],
    max_records: int | None,
    iterator: Iterable[tuple[str, str]],
    out_path: Path,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    deduper = Deduper(cfg, str(source["name"]))
    max_kept = int(source.get("max_kept") or 0)
    out_uncompressed_bytes = 0
    tmp_out = source_tmp_output(out_path)
    proc, stream = zstd_writer(tmp_out)
    try:
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} process start kind=smiles source={source.get('name')} output={out_path}", flush=True)
        for n, (original_id, raw_smiles) in enumerate(iterator, 1):
            if max_records is not None and n > max_records:
                break
            if n % 1000000 == 0:
                log_progress("smiles", source, n, deduper.count, out_uncompressed_bytes)
            if not passes_input_sampling(f"{original_id}|{raw_smiles}", source):
                counts["input_sampled_out"] += 1
                continue
            normalized, heavy, status = canonicalize_smiles(raw_smiles, cfg["small_molecule"])
            counts[status] += 1
            if status != "ok" or normalized is None or heavy is None:
                continue
            row_hash = sha1_text(f"small_molecule|{normalized}")
            if not passes_source_sampling(row_hash, source):
                counts["sampled_out"] += 1
                continue
            if not deduper.add(row_hash):
                counts["duplicate"] += 1
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
            out_uncompressed_bytes += write_row(stream, row)
            if max_kept > 0 and deduper.count >= max_kept:
                counts["max_kept_reached"] += 1
                break
    finally:
        finish_writer(proc, stream)
        deduper.close()
    finalize_source_output(tmp_out, out_path)
    print(
        f"{dt.datetime.now().isoformat(timespec='seconds')} process done kind=smiles "
        f"source={source.get('name')} kept={deduper.count} out_uncompressed={out_uncompressed_bytes}",
        flush=True,
    )
    return {
        "name": source["name"],
        "domain": "small_molecule",
        "input": str(root / source["input"]),
        "output": str(out_path),
        "counts": dict(counts),
        "kept": deduper.count,
        "uncompressed_bytes": out_uncompressed_bytes,
        "created_utc": utc_now(),
    }


def process_chembl(source: dict[str, Any], root: Path, cfg: dict[str, Any], max_records: int | None) -> dict[str, Any]:
    in_path = root / source["input"]
    return process_smiles_source(source, root, cfg, max_records, iter_chembl_chemreps(in_path))


def process_pubchem(source: dict[str, Any], root: Path, cfg: dict[str, Any], max_records: int | None) -> dict[str, Any]:
    in_path = root / source["input"]
    return process_smiles_source(source, root, cfg, max_records, iter_pubchem_cid_smiles(in_path))


def sequence_windows(seq: str, window_sizes: list[int], stride: int | None, max_windows: int) -> Iterator[tuple[int, int, str]]:
    emitted = 0
    max_size = max(window_sizes)
    if len(seq) <= max_size:
        yield 0, len(seq), seq
        return
    for size in window_sizes:
        step = stride if stride is not None else max(1, size // 2)
        for start in range(0, len(seq) - size + 1, step):
            yield start, start + size, seq[start:start + size]
            emitted += 1
            if max_windows > 0 and emitted >= max_windows:
                return


def reject_low_complexity(seq: str, max_single_fraction: float) -> bool:
    if not seq:
        return True
    counts = Counter(seq)
    return (max(counts.values()) / len(seq)) > max_single_fraction


def strict_alphabet_filter_enabled(domain_cfg: dict[str, Any]) -> bool:
    """Whether to reject records before graph construction for alphabet mismatch."""

    return bool(domain_cfg.get("strict_alphabet_filter", False))


def nonstandard_symbols(seq: str, alphabet: set[str]) -> list[str]:
    if not alphabet:
        return []
    return sorted({ch for ch in seq if ch not in alphabet})


def annotate_nonstandard(metadata: dict[str, Any], seq: str, alphabet: set[str]) -> dict[str, Any]:
    symbols = nonstandard_symbols(seq, alphabet)
    if symbols:
        counts = Counter(ch for ch in seq if ch in set(symbols))
        metadata["nonstandard_symbols"] = symbols
        metadata["nonstandard_symbol_counts"] = dict(sorted(counts.items()))
    return metadata


def sequence_heavy_atoms(seq: str, table: dict[str, int]) -> int:
    if not table:
        return len(seq)
    fallback = max(1, round(sum(table.values()) / len(table)))
    return sum(table.get(ch, fallback) for ch in seq)


def process_fasta(source: dict[str, Any], root: Path, cfg: dict[str, Any], max_records: int | None) -> dict[str, Any]:
    in_path = root / source["input"]
    out_path = root / source["output"]
    domain = str(source["domain"])
    with output_lock(out_path):
        skipped = skipped_existing_result(source, root, domain, out_path)
        if skipped is not None:
            return skipped
        return _process_fasta_locked(source, root, cfg, max_records, in_path, out_path, domain)


def _process_fasta_locked(
    source: dict[str, Any],
    root: Path,
    cfg: dict[str, Any],
    max_records: int | None,
    in_path: Path,
    out_path: Path,
    domain: str,
) -> dict[str, Any]:
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
    deduper = Deduper(cfg, str(source["name"]))
    max_kept = int(source.get("max_kept") or 0)
    out_uncompressed_bytes = 0
    stop = False
    tmp_out = source_tmp_output(out_path)
    proc, stream = zstd_writer(tmp_out)
    try:
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} process start kind=fasta source={source.get('name')} output={out_path}", flush=True)
        for n, (original_id, header, raw_seq) in enumerate(iter_fasta(in_path), 1):
            if stop:
                break
            if max_records is not None and n > max_records:
                break
            if n % 250000 == 0:
                log_progress("fasta", source, n, deduper.count, out_uncompressed_bytes)
            if not passes_input_sampling(f"{original_id}|{raw_seq[:80]}|{len(raw_seq)}", source):
                counts["input_sampled_out"] += 1
                continue
            seq = raw_seq
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
                if not deduper.add(row_hash):
                    counts["duplicate"] += 1
                    continue
                window_count += 1
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
                out_uncompressed_bytes += write_row(stream, row)
                if max_kept > 0 and deduper.count >= max_kept:
                    counts["max_kept_reached"] += 1
                    stop = True
                    break
            if window_count:
                counts["ok"] += window_count
            else:
                counts["no_windows_kept"] += 1
    finally:
        finish_writer(proc, stream)
        deduper.close()
    finalize_source_output(tmp_out, out_path)
    print(
        f"{dt.datetime.now().isoformat(timespec='seconds')} process done kind=fasta "
        f"source={source.get('name')} kept={deduper.count} out_uncompressed={out_uncompressed_bytes}",
        flush=True,
    )
    return {
        "name": source["name"],
        "domain": domain,
        "input": str(in_path),
        "output": str(out_path),
        "counts": dict(counts),
        "kept": deduper.count,
        "uncompressed_bytes": out_uncompressed_bytes,
        "created_utc": utc_now(),
    }


def normalize_peptide_sequence(raw: str) -> str:
    return "".join(str(raw).split()).upper()


def process_pepbenchmark_raw(source: dict[str, Any], root: Path, cfg: dict[str, Any], max_records: int | None) -> dict[str, Any]:
    in_path = root / source["input"]
    out_path = root / source["output"]
    with output_lock(out_path):
        skipped = skipped_existing_result(source, root, "peptide", out_path)
        if skipped is not None:
            return skipped
        return _process_pepbenchmark_raw_locked(source, root, cfg, max_records, in_path, out_path)


def _process_pepbenchmark_raw_locked(
    source: dict[str, Any],
    root: Path,
    cfg: dict[str, Any],
    max_records: int | None,
    in_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    domain_cfg = dict(cfg.get("peptide") or {})
    alphabet = set(str(domain_cfg.get("alphabet", "")).strip())
    strict_alphabet_filter = strict_alphabet_filter_enabled(domain_cfg)
    min_length = int(domain_cfg.get("min_length", 5))
    max_length = int(domain_cfg.get("max_length", 80))

    counts: Counter[str] = Counter()
    deduper = Deduper(cfg, str(source["name"]))
    max_kept = int(source.get("max_kept") or 0)
    out_uncompressed_bytes = 0
    tmp_out = source_tmp_output(out_path)
    proc, stream = zstd_writer(tmp_out)
    try:
        for n, (original_id, raw_seq, metadata) in enumerate(iter_pepbenchmark_raw(in_path), 1):
            if max_records is not None and n > max_records:
                break
            seq = normalize_peptide_sequence(raw_seq)
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
            row_hash = sha1_text(f"peptide|{seq}")
            if not passes_source_sampling(row_hash, source):
                counts["sampled_out"] += 1
                continue
            if not deduper.add(row_hash):
                counts["duplicate"] += 1
                continue
            row = {
                "id": f"{source['source']}:{original_id}",
                "domain": "peptide",
                "source": source["source"],
                "source_version": source["source_version"],
                "original_id": original_id,
                "raw": raw_seq,
                "normalized": seq,
                "length": len(seq),
                "estimated_heavy_atoms": sequence_heavy_atoms(seq, AA_HEAVY_ATOMS),
                "split_group": seq,
                "metadata": annotate_nonstandard(dict(metadata), seq, alphabet),
                "filter_status": "ok",
                "sha1": row_hash,
            }
            counts["ok"] += 1
            out_uncompressed_bytes += write_row(stream, row)
            if max_kept > 0 and deduper.count >= max_kept:
                counts["max_kept_reached"] += 1
                break
    finally:
        finish_writer(proc, stream)
        deduper.close()
    finalize_source_output(tmp_out, out_path)
    return {
        "name": source["name"],
        "domain": "peptide",
        "input": str(in_path),
        "output": str(out_path),
        "counts": dict(counts),
        "kept": deduper.count,
        "uncompressed_bytes": out_uncompressed_bytes,
        "created_utc": utc_now(),
    }


def peptide_fragment_for(seq: str, key: str, lengths: list[int], min_length: int, max_length: int) -> tuple[int, int, str] | None:
    allowed_lengths = [length for length in lengths if min_length <= length <= max_length and length <= len(seq)]
    if not allowed_lengths:
        return None
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    length = allowed_lengths[int.from_bytes(digest[:4], "big") % len(allowed_lengths)]
    max_start = len(seq) - length
    start = 0 if max_start <= 0 else int.from_bytes(digest[4:12], "big") % (max_start + 1)
    return start, start + length, seq[start:start + length]


def process_protein_derived_peptides(source: dict[str, Any], root: Path, cfg: dict[str, Any], max_records: int | None) -> dict[str, Any]:
    in_path = root / source["input"]
    out_path = root / source["output"]
    with output_lock(out_path):
        skipped = skipped_existing_result(source, root, "peptide", out_path)
        if skipped is not None:
            return skipped
        return _process_protein_derived_peptides_locked(source, root, cfg, max_records, in_path, out_path)


def _process_protein_derived_peptides_locked(
    source: dict[str, Any],
    root: Path,
    cfg: dict[str, Any],
    max_records: int | None,
    in_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    domain_cfg = dict(cfg.get("peptide") or {})
    alphabet = set(str(domain_cfg.get("alphabet", "")).strip())
    strict_alphabet_filter = strict_alphabet_filter_enabled(domain_cfg)
    min_length = int(domain_cfg.get("min_length", 5))
    max_length = int(domain_cfg.get("max_length", 80))
    fragment_lengths = split_ints(source.get("fragment_lengths"), [8, 16, 32, 64])
    fragments_per_record = max(1, int(source.get("fragments_per_record") or 1))

    counts: Counter[str] = Counter()
    deduper = Deduper(cfg, str(source["name"]))
    max_kept = int(source.get("max_kept") or 0)
    out_uncompressed_bytes = 0
    stop = False
    tmp_out = source_tmp_output(out_path)
    proc, stream = zstd_writer(tmp_out)
    try:
        for n, row in enumerate(iter_jsonl_auto(in_path), 1):
            if stop:
                break
            if max_records is not None and n > max_records:
                break
            if str(row.get("domain")) != "protein":
                counts["non_protein_parent"] += 1
                continue
            parent_seq = str(row.get("normalized") or "").upper()
            parent_key = str(row.get("sha1") or row.get("id") or n)
            if not passes_input_sampling(parent_key, source):
                counts["input_sampled_out"] += 1
                continue
            if alphabet and any(ch not in alphabet for ch in parent_seq):
                counts["nonstandard_parent"] += 1
                if strict_alphabet_filter:
                    continue
            for frag_index in range(fragments_per_record):
                fragment = peptide_fragment_for(
                    parent_seq,
                    f"{source['name']}|{parent_key}|{frag_index}",
                    fragment_lengths,
                    min_length,
                    max_length,
                )
                if fragment is None:
                    counts["too_short_parent"] += 1
                    continue
                start, end, seq = fragment
                if alphabet and any(ch not in alphabet for ch in seq):
                    counts["nonstandard_fragment"] += 1
                    if strict_alphabet_filter:
                        continue
                row_hash = sha1_text(f"peptide|{seq}")
                if not passes_source_sampling(row_hash, source):
                    counts["sampled_out"] += 1
                    continue
                if not deduper.add(row_hash):
                    counts["duplicate"] += 1
                    continue
                original_id = f"{row.get('id', parent_key)}:{start}:{end}:{frag_index}"
                out_row = {
                    "id": f"{source['source']}:{original_id}",
                    "domain": "peptide",
                    "source": source["source"],
                    "source_version": source["source_version"],
                    "original_id": original_id,
                    "raw": seq,
                    "normalized": seq,
                    "length": len(seq),
                    "estimated_heavy_atoms": sequence_heavy_atoms(seq, AA_HEAVY_ATOMS),
                    "split_group": str(row.get("split_group") or row.get("original_id") or parent_key),
                    "metadata": annotate_nonstandard({
                        "derivation": "protein_fragment",
                        "parent_id": row.get("id"),
                        "parent_source": row.get("source"),
                        "parent_source_version": row.get("source_version"),
                        "parent_original_id": row.get("original_id"),
                        "parent_length": row.get("length"),
                        "parent_start": start,
                        "parent_end": end,
                        "fragment_index": frag_index,
                    }, seq, alphabet),
                    "filter_status": "ok",
                    "sha1": row_hash,
                }
                counts["ok"] += 1
                out_uncompressed_bytes += write_row(stream, out_row)
                if max_kept > 0 and deduper.count >= max_kept:
                    counts["max_kept_reached"] += 1
                    stop = True
                    break
    finally:
        finish_writer(proc, stream)
        deduper.close()
    finalize_source_output(tmp_out, out_path)
    return {
        "name": source["name"],
        "domain": "peptide",
        "input": str(in_path),
        "output": str(out_path),
        "counts": dict(counts),
        "kept": deduper.count,
        "uncompressed_bytes": out_uncompressed_bytes,
        "created_utc": utc_now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize raw molscaling sources into JSONL.zst.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--source", action="append")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = Path(cfg["root"]).resolve()
    sources = cfg.get("sources") or []
    if args.source:
        wanted = set(args.source)
        sources = [s for s in sources if s.get("name") in wanted]
    print(json.dumps({
        "event": "process_plan",
        "root": str(root),
        "sources": [s["name"] for s in sources],
        "max_records": args.max_records,
        "dry_run": args.dry_run,
    }, indent=2))
    if args.dry_run:
        return 0
    results = []
    with process_all_global_lock(root):
        for src in sources:
            fmt = src.get("format")
            if fmt == "chembl_chemreps_tsv_gz":
                results.append(process_chembl(src, root, cfg, args.max_records))
            elif fmt == "pubchem_cid_smiles_gz":
                results.append(process_pubchem(src, root, cfg, args.max_records))
            elif fmt == "fasta_gz":
                results.append(process_fasta(src, root, cfg, args.max_records))
            elif fmt == "pepbenchmark_raw_snapshot":
                results.append(process_pepbenchmark_raw(src, root, cfg, args.max_records))
            elif fmt == "protein_derived_peptide_jsonl":
                results.append(process_protein_derived_peptides(src, root, cfg, args.max_records))
            else:
                raise ValueError(f"Unsupported source format: {fmt}")
            write_source_done(root, results[-1])
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"process_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest = reports / "process_report.latest.json"
    payload = {"created_utc": utc_now(), "results": results}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "process_report_written", "path": str(out), "latest": str(latest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
