#!/usr/bin/env python3
from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import multiprocessing as mp
import os
import pickle
import socket
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.graph.io import write_compact_graph_zst
from pipeline.graph.chem_graph_builder import (
    is_standard_nucleotide_sequence,
    is_standard_protein_sequence,
    nucleotide_sequence_to_graph_data_fast,
    protein_sequence_to_graph_data_fast,
)


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
KNOWN_ZSTD = [
    
    "/usr/bin/zstd",
    "/usr/local/bin/zstd",
]
DEFAULT_ROOT = Path(str(_moltoken_paths.LOCAL_DATA_ROOT / ''))
DEFAULT_CONFIG = PIPELINE_ROOT / "configs/data/current/graph_heavy_atom_normv2_compact_v1.yaml"
BOND_TYPE_TO_ID = {
    "UNSPECIFIED": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DNA_ALPHABET = set("ACGT")
RNA_ALPHABET = set("ACGU")
PARSER_VERSION = "heavy_atom_graph_v1_rdkit_2d"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


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
        proc = subprocess.Popen([zstd, "-q", "-dc", str(path)], stdout=subprocess.PIPE, text=True)
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


def zstd_pickle_dump(path: Path, obj: Any, level: int, threads: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    zstd = find_zstd()
    if zstd is None:
        raise RuntimeError("zstd executable is required for .pkl.zst output")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    proc = subprocess.Popen(
        [zstd, "-q", f"-T{max(1, int(threads))}", f"-{int(level)}", "-f", "-o", str(tmp)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        pickle.dump(obj, proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"zstd failed for {path} rc={rc}")
    tmp.replace(path)
    return path.stat().st_size


def zstd_pickle_load(path: Path) -> Any:
    zstd = find_zstd()
    if zstd is None:
        raise RuntimeError("zstd executable is required for .pkl.zst input")
    proc = subprocess.Popen([zstd, "-q", "-dc", str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        return pickle.load(proc.stdout)
    finally:
        proc.stdout.close()
        rc = proc.wait()
        if rc not in {0, -13}:
            raise RuntimeError(f"zstd -dc failed for {path} rc={rc}")


def resolve_input_paths(input_root: Path, domains: set[str] | None, splits: set[str] | None) -> list[Path]:
    paths = []
    for path in sorted(input_root.rglob("*.jsonl.zst")):
        rel = path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue
        domain, split = rel.parts[0], rel.parts[1]
        if domains is not None and domain not in domains:
            continue
        if splits is not None and split not in splits:
            continue
        paths.append(path)
    return paths


def read_selected_relatives(path: Path) -> set[str]:
    selected: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.endswith(".jsonl.zst"):
            item = item[: -len(".jsonl.zst")]
        selected.add(item)
    return selected


def output_stem(input_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_path.relative_to(input_root)
    name = rel.name
    if name.endswith(".jsonl.zst"):
        name = name[: -len(".jsonl.zst")]
    return output_root / rel.parent / name


def done_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_stem(input_path, input_root, output_root).with_suffix(".done.json")


def lock_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_stem(input_path, input_root, output_root).with_suffix(".lock")


def cleanup_graph_outputs(stem: Path) -> None:
    for old in stem.parent.glob(f"{stem.name}-graph-*.pkl.zst"):
        old.unlink()
    for old in stem.parent.glob(f"{stem.name}-graph-*.cgraph.zst"):
        old.unlink()
    for old in stem.parent.glob(f"{stem.name}-graph-*.pkl.zst.tmp"):
        old.unlink()
    for old in stem.parent.glob(f"{stem.name}-graph-*.cgraph.zst.tmp"):
        old.unlink()


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def _remove_stale_lock_if_safe(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8").splitlines()[0])
        except Exception:
            return
        pid = int(payload.get("pid") or -1)
        host = str(payload.get("host") or "")
        current_host = socket.gethostname()
        if host and host != current_host:
            return
        if pid <= 0:
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.path.unlink()
            return
        except PermissionError:
            return

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            self._remove_stale_lock_if_safe()
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                raise RuntimeError(f"active graph build lock exists: {self.path}") from exc
            except FileNotFoundError:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileNotFoundError:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        if self.fd is None:
            raise RuntimeError(f"failed to acquire graph build lock: {self.path}")
        payload = json.dumps(
            {"pid": os.getpid(), "host": socket.gethostname(), "created_utc": utc_now()},
            ensure_ascii=False,
        ) + "\n"
        os.write(self.fd, payload.encode("utf-8"))
        os.close(self.fd)
        self.fd = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _bond_type_id(bond: Any) -> int:
    name = str(bond.GetBondType()).upper()
    if bond.GetIsAromatic():
        return BOND_TYPE_TO_ID["AROMATIC"]
    return BOND_TYPE_TO_ID.get(name, 0)


def mol_to_graph_data(
    mol: Any,
    sample_id: str,
    properties: dict[str, Any],
    *,
    sequence: str | None = None,
    smiles: str | None = None,
) -> dict[str, Any]:
    import numpy as np
    from rdkit import Chem

    mol = Chem.RemoveHs(mol, sanitize=False)
    Chem.SanitizeMol(mol)

    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    old_to_new = {atom.GetIdx(): i for i, atom in enumerate(atoms)}
    src: list[int] = []
    dst: list[int] = []
    edge_type_ids: list[int] = []
    edge_features: list[list[int]] = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in old_to_new or end not in old_to_new:
            continue
        bond_type = _bond_type_id(bond)
        stereo = int(bond.GetStereo())
        aromatic = int(bond.GetIsAromatic())
        for u, v in ((old_to_new[begin], old_to_new[end]), (old_to_new[end], old_to_new[begin])):
            src.append(u)
            dst.append(v)
            edge_type_ids.append(bond_type)
            edge_features.append([bond_type, aromatic, stereo])

    node_type_ids: list[int] = []
    node_features: list[list[int]] = []
    for atom in atoms:
        atomic_num = int(atom.GetAtomicNum())
        node_type_ids.append(atomic_num)
        node_features.append(
            [
                atomic_num,
                int(atom.GetFormalCharge()),
                int(atom.GetIsAromatic()),
                int(atom.GetChiralTag()),
                int(atom.GetDegree()),
                int(atom.GetTotalValence()),
            ]
        )

    node_type_arr = np.asarray(node_type_ids, dtype=np.int64)
    edge_type_arr = np.asarray(edge_type_ids, dtype=np.int64)
    graph_data: dict[str, Any] = {
        "sample_id": sample_id,
        "num_nodes": int(len(node_type_ids)),
        "edges": (src, dst),
        "node_features": np.asarray(node_features, dtype=np.int64).reshape((-1, 6)),
        "edge_features": np.asarray(edge_features, dtype=np.int64).reshape((-1, 3)),
        "node_type_ids": node_type_arr,
        "edge_type_ids": edge_type_arr,
        "node_token_ids": (node_type_arr * 2 + 1).reshape((-1, 1)),
        "edge_token_ids": (edge_type_arr * 2).reshape((-1, 1)),
        "properties": properties,
    }
    if sequence is not None:
        graph_data["sequence"] = sequence
    if smiles is not None:
        graph_data["smiles"] = smiles
    return graph_data


def make_meta(row: dict[str, Any], input_path: str, split: str, parser_version: str) -> dict[str, Any]:
    domain = str(row.get("domain") or "unknown")
    source = str(row.get("source") or "unknown")
    return {
        "sample_id": str(row.get("id") or ""),
        "source_domain": domain,
        "source_dataset": source,
        "domain": domain,
        "source": source,
        "source_version": row.get("source_version"),
        "original_id": row.get("original_id"),
        "split": split,
        "sha1": row.get("sha1"),
        "length": row.get("length"),
        "estimated_heavy_atoms": row.get("estimated_heavy_atoms"),
        "normalized": row.get("normalized"),
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "input_path": input_path,
        "parser_version": parser_version,
    }


def build_record(task: tuple[dict[str, Any], str, str, str, dict[str, int]]) -> tuple[bool, Any]:
    row, input_path, split, parser_version, flavors = task
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    domain = str(row.get("domain") or "")
    normalized = str(row.get("normalized") or "").strip()
    sample_id = str(row.get("id") or row.get("sha1") or "")
    if not normalized:
        return False, {
            "reason": "empty_normalized",
            "sample_id": sample_id,
            "domain": domain,
        }

    meta = make_meta(row, input_path, split, parser_version)
    if domain == "small_molecule":
        meta["graph_builder"] = "rdkit_smiles"
        mol = Chem.MolFromSmiles(normalized)
        if mol is None:
            return False, {
                "reason": "rdkit_smiles_parse_failed",
                "sample_id": sample_id,
                "domain": domain,
            }
        graph = mol_to_graph_data(
            mol, sample_id, dict(meta), smiles=normalized
        )
    elif domain in {"peptide", "protein"}:
        seq = normalized.upper()
        if is_standard_protein_sequence(seq):
            meta["graph_builder"] = "template_fast_protein"
            graph = protein_sequence_to_graph_data_fast(seq, sample_id, dict(meta))
        else:
            meta["graph_builder"] = "rdkit_fasta"
            mol = Chem.MolFromFASTA(seq, flavor=int(flavors.get(domain, 0)))
            if mol is None:
                return False, {
                    "reason": "rdkit_fasta_parse_failed",
                    "sample_id": sample_id,
                    "domain": domain,
                }
            graph = mol_to_graph_data(mol, sample_id, dict(meta), sequence=seq)
    elif domain == "dna":
        seq = normalized.upper()
        if is_standard_nucleotide_sequence(seq, "dna"):
            meta["graph_builder"] = "template_fast_dna"
            graph = nucleotide_sequence_to_graph_data_fast(
                seq, "dna", sample_id, dict(meta)
            )
        else:
            meta["graph_builder"] = "rdkit_dna"
            mol = Chem.MolFromFASTA(seq, flavor=int(flavors.get(domain, 6)))
            if mol is None:
                return False, {
                    "reason": "rdkit_dna_parse_failed",
                    "sample_id": sample_id,
                    "domain": domain,
                }
            graph = mol_to_graph_data(mol, sample_id, dict(meta), sequence=seq)
    elif domain == "rna":
        seq = normalized.upper()
        if is_standard_nucleotide_sequence(seq, "rna"):
            meta["graph_builder"] = "template_fast_rna"
            graph = nucleotide_sequence_to_graph_data_fast(
                seq, "rna", sample_id, dict(meta)
            )
        else:
            meta["graph_builder"] = "rdkit_rna"
            mol = Chem.MolFromFASTA(seq, flavor=int(flavors.get(domain, 2)))
            if mol is None:
                return False, {
                    "reason": "rdkit_rna_parse_failed",
                    "sample_id": sample_id,
                    "domain": domain,
                }
            graph = mol_to_graph_data(mol, sample_id, dict(meta), sequence=seq)
    else:
        return False, {
            "reason": "unsupported_domain",
            "sample_id": sample_id,
            "domain": domain,
        }

    if int(graph["num_nodes"]) <= 0:
        return False, {
            "reason": "empty_graph",
            "sample_id": sample_id,
            "domain": domain,
        }
    return True, (graph, meta)


def validate_graph_record(record: tuple[dict[str, Any], dict[str, Any]]) -> str | None:
    graph, _meta = record
    required = {
        "sample_id",
        "num_nodes",
        "edges",
        "node_features",
        "edge_features",
        "node_type_ids",
        "edge_type_ids",
        "node_token_ids",
        "edge_token_ids",
        "properties",
    }
    missing = required - set(graph)
    if missing:
        return f"missing:{','.join(sorted(missing))}"
    num_nodes = int(graph["num_nodes"])
    if len(graph["node_type_ids"]) != num_nodes:
        return "node_type_len_mismatch"
    src, dst = graph["edges"]
    if len(src) != len(dst):
        return "edge_src_dst_len_mismatch"
    if len(graph["edge_type_ids"]) != len(src):
        return "edge_type_len_mismatch"
    if graph["node_features"].shape != (num_nodes, 6):
        return "node_feature_shape_mismatch"
    if graph["edge_features"].shape != (len(src), 3):
        return "edge_feature_shape_mismatch"
    return None


class ChunkWriter:
    def __init__(
        self,
        stem: Path,
        zstd_level: int,
        zstd_threads: int,
        max_graphs: int,
        max_nodes: int,
        validate_records: bool,
        graph_format: str,
    ) -> None:
        self.stem = stem
        self.zstd_level = zstd_level
        self.zstd_threads = zstd_threads
        self.max_graphs = max_graphs
        self.max_nodes = max_nodes
        self.validate_records = validate_records
        self.graph_format = graph_format
        self.chunk: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.chunk_nodes = 0
        self.index = 0
        self.outputs: list[dict[str, Any]] = []
        self.validation_errors: Counter[str] = Counter()

    def add(self, record: tuple[dict[str, Any], dict[str, Any]]) -> None:
        if self.validate_records:
            reason = validate_graph_record(record)
            if reason is not None:
                self.validation_errors[reason] += 1
                return False
        self.chunk.append(record)
        self.chunk_nodes += int(record[0]["num_nodes"])
        if len(self.chunk) >= self.max_graphs or self.chunk_nodes >= self.max_nodes:
            self.flush()
        return True

    def flush(self) -> None:
        if not self.chunk:
            return
        if self.graph_format == "compact":
            out = self.stem.with_name(f"{self.stem.name}-graph-{self.index:05d}.cgraph.zst")
        else:
            out = self.stem.with_name(f"{self.stem.name}-graph-{self.index:05d}.pkl.zst")
        records = len(self.chunk)
        nodes = sum(int(item[0]["num_nodes"]) for item in self.chunk)
        edges = sum(len(item[0]["edges"][0]) for item in self.chunk)
        if self.graph_format == "compact":
            compressed = write_compact_graph_zst(
                out,
                self.chunk,
                zstd_level=self.zstd_level,
                zstd_threads=self.zstd_threads,
            )
        else:
            compressed = zstd_pickle_dump(out, self.chunk, self.zstd_level, self.zstd_threads)
        self.outputs.append(
            {
                "path": str(out),
                "records": records,
                "nodes": nodes,
                "edges": edges,
                "compressed_bytes": compressed,
                "compressed_gb": compressed / 1e9,
            }
        )
        self.index += 1
        self.chunk = []
        self.chunk_nodes = 0

    def close(self) -> None:
        self.flush()


def process_input_file(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    flavors: dict[str, int],
    parser_version: str,
) -> dict[str, Any]:
    rel = input_path.relative_to(input_root)
    domain = rel.parts[0]
    split = rel.parts[1]
    done_path = done_path_for(input_path, input_root, output_root)
    if done_path.exists() and not args.overwrite:
        return {
            "input": str(input_path),
            "domain": domain,
            "split": split,
            "status": "skipped_existing_done",
            "done_path": str(done_path),
        }

    stem = output_stem(input_path, input_root, output_root)
    with FileLock(lock_path_for(input_path, input_root, output_root)):
        # Another file worker may have completed this shard while we were waiting
        # to acquire the lock.
        if done_path.exists() and not args.overwrite:
            return {
                "input": str(input_path),
                "domain": domain,
                "split": split,
                "status": "skipped_existing_done",
                "done_path": str(done_path),
            }
        if args.overwrite or not done_path.exists():
            cleanup_graph_outputs(stem)
        return _process_input_file_locked(input_path, input_root, output_root, args, flavors, parser_version)


def _process_input_file_locked(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    flavors: dict[str, int],
    parser_version: str,
) -> dict[str, Any]:
    rel = input_path.relative_to(input_root)
    domain = rel.parts[0]
    split = rel.parts[1]
    done_path = done_path_for(input_path, input_root, output_root)
    stem = output_stem(input_path, input_root, output_root)
    writer = ChunkWriter(
        stem=stem,
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        max_graphs=args.max_graphs_per_chunk,
        max_nodes=args.max_nodes_per_chunk,
        validate_records=not args.no_validate_records,
        graph_format=args.graph_format,
    )
    pending: set[futures.Future] = set()
    max_pending = max(1, int(args.workers) * int(args.pending_multiplier))
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    node_min: int | None = None
    node_max: int | None = None
    edge_min: int | None = None
    edge_max: int | None = None
    node_sum = 0
    edge_sum = 0
    start = time.time()

    def consume_done(done: Iterable[futures.Future]) -> None:
        nonlocal node_min, node_max, edge_min, edge_max, node_sum, edge_sum
        for fut in done:
            ok, payload = fut.result()
            if not ok:
                reason = str(payload.get("reason") or "unknown")
                failures[reason] += 1
                counts["failed"] += 1
                if len(failure_examples) < int(args.max_failure_examples):
                    failure_examples.append(payload)
                continue
            record = payload
            graph, meta = record
            if not writer.add(record):
                counts["failed"] += 1
                continue
            nodes = int(graph["num_nodes"])
            edges = len(graph["edges"][0])
            node_sum += nodes
            edge_sum += edges
            node_min = nodes if node_min is None else min(node_min, nodes)
            node_max = nodes if node_max is None else max(node_max, nodes)
            edge_min = edges if edge_min is None else min(edge_min, edges)
            edge_max = edges if edge_max is None else max(edge_max, edges)
            sources[str(meta.get("source") or "unknown")] += 1
            counts["ok"] += 1

    log(f"graph start input={input_path} domain={domain} split={split}")
    if int(args.workers) <= 1:
        for row in iter_jsonl_zst(input_path):
            if args.max_records_per_shard is not None and counts["seen"] >= args.max_records_per_shard:
                break
            counts["seen"] += 1
            task = (row, str(input_path), split, parser_version, flavors)
            ok, payload = build_record(task)
            if not ok:
                reason = str(payload.get("reason") or "unknown")
                failures[reason] += 1
                counts["failed"] += 1
                if len(failure_examples) < int(args.max_failure_examples):
                    failure_examples.append(payload)
                continue
            record = payload
            graph, meta = record
            if not writer.add(record):
                counts["failed"] += 1
                continue
            nodes = int(graph["num_nodes"])
            edges = len(graph["edges"][0])
            node_sum += nodes
            edge_sum += edges
            node_min = nodes if node_min is None else min(node_min, nodes)
            node_max = nodes if node_max is None else max(node_max, nodes)
            edge_min = edges if edge_min is None else min(edge_min, edges)
            edge_max = edges if edge_max is None else max(edge_max, edges)
            sources[str(meta.get("source") or "unknown")] += 1
            counts["ok"] += 1
            if counts["seen"] and counts["seen"] % int(args.log_every) == 0:
                log(f"graph progress input={input_path.name} seen={counts['seen']} ok={counts['ok']} failed={counts['failed']}")
    else:
        max_tasks_per_child = None if int(args.maxtasksperchild) <= 0 else int(args.maxtasksperchild)
        start_method = str(getattr(args, "record_start_method", "auto") or "auto")
        if start_method == "auto":
            if int(getattr(args, "file_workers", 1) or 1) > 1:
                start_method = "forkserver"
            elif max_tasks_per_child is None:
                start_method = "fork"
            else:
                start_method = "spawn"
        if start_method == "fork" and max_tasks_per_child is not None:
            raise RuntimeError("--record-start-method fork is incompatible with positive --maxtasksperchild")
        mp_context = mp.get_context(start_method)
        with futures.ProcessPoolExecutor(
            max_workers=args.workers,
            max_tasks_per_child=max_tasks_per_child,
            mp_context=mp_context,
        ) as pool:
            for row in iter_jsonl_zst(input_path):
                if args.max_records_per_shard is not None and counts["seen"] >= args.max_records_per_shard:
                    break
                counts["seen"] += 1
                task = (row, str(input_path), split, parser_version, flavors)
                pending.add(pool.submit(build_record, task))
                if len(pending) >= max_pending:
                    done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                    consume_done(done)
                if counts["seen"] and counts["seen"] % int(args.log_every) == 0:
                    log(f"graph progress input={input_path.name} seen={counts['seen']} ok={counts['ok']} failed={counts['failed']}")
            if pending:
                done, pending = futures.wait(pending)
                consume_done(done)

    writer.close()
    if writer.validation_errors:
        failures.update({f"validation:{k}": v for k, v in writer.validation_errors.items()})

    elapsed = time.time() - start
    report = {
        "created_utc": utc_now(),
        "parser_version": parser_version,
        "input": str(input_path),
        "input_relative": str(rel),
        "domain": domain,
        "split": split,
        "status": "done",
        "counts": dict(counts),
        "failures": dict(failures),
        "failure_examples": failure_examples,
        "sources": dict(sources),
        "outputs": writer.outputs,
        "output_records": int(sum(item["records"] for item in writer.outputs)),
        "output_nodes": int(sum(item["nodes"] for item in writer.outputs)),
        "output_edges": int(sum(item["edges"] for item in writer.outputs)),
        "output_compressed_bytes": int(sum(item["compressed_bytes"] for item in writer.outputs)),
        "node_min": node_min,
        "node_max": node_max,
        "node_avg": (node_sum / counts["ok"]) if counts["ok"] else None,
        "edge_min": edge_min,
        "edge_max": edge_max,
        "edge_avg": (edge_sum / counts["ok"]) if counts["ok"] else None,
        "elapsed_sec": elapsed,
        "records_per_sec": (counts["seen"] / elapsed) if elapsed > 0 else None,
        "done_path": str(done_path),
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = done_path.with_suffix(done_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(done_path)
    log(
        f"graph done input={input_path.name} seen={counts['seen']} ok={counts['ok']} "
        f"failed={counts['failed']} outputs={len(writer.outputs)} elapsed={elapsed:.1f}s"
    )
    return report


def process_input_file_task(task: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(str(task["input_path"]))
    try:
        args = argparse.Namespace(**task["args"])
        return process_input_file(
            input_path,
            Path(str(task["input_root"])),
            Path(str(task["output_root"])),
            args,
            {str(k): int(v) for k, v in dict(task["flavors"]).items()},
            str(task["parser_version"]),
        )
    except Exception as exc:  # noqa: BLE001
        rel = Path(str(task.get("relative") or input_path.name))
        parts = rel.parts
        domain = parts[0] if len(parts) >= 3 else "unknown"
        split = parts[1] if len(parts) >= 3 else "unknown"
        return {
            "input": str(input_path),
            "input_relative": str(rel),
            "domain": domain,
            "split": split,
            "status": "failed_exception",
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=12),
            "counts": {"seen": 0, "ok": 0, "failed": 0},
        }


def write_run_report(root: Path, output_root: Path, report_prefix: str, reports: list[dict[str, Any]], dry_run: bool) -> Path:
    aggregate: dict[str, Any] = {
        "created_utc": utc_now(),
        "dry_run": dry_run,
        "output_root": str(output_root),
        "files": reports,
        "domains": {},
        "totals": {
            "input_files": len(reports),
            "seen": 0,
            "ok": 0,
            "failed": 0,
            "output_records": 0,
            "output_nodes": 0,
            "output_edges": 0,
            "output_compressed_bytes": 0,
        },
    }
    for report in reports:
        domain = str(report.get("domain") or "unknown")
        d = aggregate["domains"].setdefault(
            domain,
            {"input_files": 0, "seen": 0, "ok": 0, "failed": 0, "output_records": 0, "output_nodes": 0, "output_edges": 0},
        )
        counts = report.get("counts") or {}
        d["input_files"] += 1
        for key in ("seen", "ok", "failed"):
            d[key] += int(counts.get(key) or 0)
            aggregate["totals"][key] += int(counts.get(key) or 0)
        for key in ("output_records", "output_nodes", "output_edges", "output_compressed_bytes"):
            value = int(report.get(key) or 0)
            d[key] = int(d.get(key) or 0) + value
            aggregate["totals"][key] += value
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = reports_dir / f"{report_prefix}_{stamp}.json"
    latest = reports_dir / f"{report_prefix}.latest.json"
    text = json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build molscaling 2D heavy-atom graph shards from normalized JSONL.zst shards.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--input-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--report-prefix", default=None)
    ap.add_argument("--domains", default=None, help="Comma-separated domains to process.")
    ap.add_argument("--splits", default=None, help="Comma-separated splits to process.")
    ap.add_argument(
        "--input-relatives-file",
        type=Path,
        default=None,
        help="Optional newline file of input paths relative to input-root, with or without .jsonl.zst.",
    )
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--file-index", type=int, default=None, help="Process only files where ordinal %% file_stride == file_index.")
    ap.add_argument("--file-stride", type=int, default=None, help="Total number of file partitions for --file-index.")
    ap.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Number of input shards to process concurrently. Total CPU is roughly file-workers * workers.",
    )
    ap.add_argument(
        "--file-executor",
        choices=["thread", "process"],
        default="process",
        help=(
            "Executor for shard-level parallelism. process is the default for the preferred "
            "FILE_WORKERS x WORKERS=1 mode; thread is only useful when each shard owns an "
            "inner record process pool."
        ),
    )
    ap.add_argument("--file-chunksize", type=int, default=1, help="Chunksize for shard-level worker pool.")
    ap.add_argument("--max-records-per-shard", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--record-start-method",
        choices=["auto", "fork", "spawn", "forkserver"],
        default="auto",
        help=(
            "Multiprocessing start method for record-level RDKit workers. auto uses fork "
            "for one shard at a time and forkserver when --file-workers > 1 to avoid "
            "forking child processes from shard-level threads."
        ),
    )
    ap.add_argument("--pending-multiplier", type=int, default=None)
    ap.add_argument("--maxtasksperchild", type=int, default=None)
    ap.add_argument("--max-graphs-per-chunk", type=int, default=None)
    ap.add_argument("--max-nodes-per-chunk", type=int, default=None)
    ap.add_argument("--zstd-level", type=int, default=None)
    ap.add_argument("--zstd-threads", type=int, default=None)
    ap.add_argument("--max-failure-examples", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=100000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-validate-records", action="store_true")
    ap.add_argument(
        "--graph-format",
        choices=["pickle", "compact"],
        default=None,
        help=(
            "中间 graph 落盘格式。默认读取 config.output.graph_format；"
            "未配置时使用 pickle 兼容旧流程。"
        ),
    )
    args = ap.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"graph config does not exist: {args.config}")
    cfg = load_yaml(args.config)
    cfg_root = Path(str(cfg.get("root") or DEFAULT_ROOT))
    root = args.root or cfg_root
    input_root = args.input_root or (root / str(cfg.get("input_root") or "shards_bulk"))
    output_root = args.output_root or (root / str(cfg.get("output_root") or "graphs_heavy_atom_v1"))
    report_prefix = str(args.report_prefix or cfg.get("report_prefix") or "graph_heavy_atom_v1")
    parser_version = str(cfg.get("parser_version") or PARSER_VERSION)

    runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    args.workers = int(args.workers or runtime.get("workers") or 16)
    args.pending_multiplier = int(args.pending_multiplier or runtime.get("pending_multiplier") or 4)
    args.maxtasksperchild = int(
        args.maxtasksperchild
        if args.maxtasksperchild is not None
        else runtime.get("maxtasksperchild", 200)
    )
    args.max_graphs_per_chunk = int(args.max_graphs_per_chunk or output_cfg.get("max_graphs_per_chunk") or 20000)
    args.max_nodes_per_chunk = int(args.max_nodes_per_chunk or output_cfg.get("max_nodes_per_chunk") or 2_000_000)
    args.zstd_level = int(args.zstd_level or output_cfg.get("zstd_level") or 3)
    args.zstd_threads = int(args.zstd_threads or output_cfg.get("zstd_threads") or 2)
    args.graph_format = str(args.graph_format or output_cfg.get("graph_format") or "pickle")

    domain_cfg = cfg.get("domains") if isinstance(cfg.get("domains"), dict) else {}
    flavors: dict[str, int] = {}
    for domain, spec in domain_cfg.items():
        if isinstance(spec, dict) and "rdkit_fasta_flavor" in spec:
            flavors[str(domain)] = int(spec["rdkit_fasta_flavor"])
    flavors.setdefault("peptide", 0)
    flavors.setdefault("protein", 0)
    flavors.setdefault("rna", 2)
    flavors.setdefault("dna", 6)

    domains = {x.strip() for x in args.domains.split(",") if x.strip()} if args.domains else None
    splits = {x.strip() for x in args.splits.split(",") if x.strip()} if args.splits else None
    paths = resolve_input_paths(input_root, domains, splits)
    if args.input_relatives_file is not None:
        selected_relatives = read_selected_relatives(args.input_relatives_file)
        paths = [
            p
            for p in paths
            if str(p.relative_to(input_root)) in selected_relatives
            or str(p.relative_to(input_root))[: -len(".jsonl.zst")] in selected_relatives
        ]
    if args.file_stride is not None or args.file_index is not None:
        if args.file_stride is None or args.file_index is None:
            raise SystemExit("--file-index and --file-stride must be provided together")
        if args.file_stride <= 0:
            raise SystemExit("--file-stride must be positive")
        if args.file_index < 0 or args.file_index >= args.file_stride:
            raise SystemExit("--file-index must satisfy 0 <= file_index < file_stride")
        paths = [path for idx, path in enumerate(paths) if idx % int(args.file_stride) == int(args.file_index)]
    if args.max_files is not None:
        paths = paths[: args.max_files]

    plan = {
        "config": str(args.config),
        "root": str(root),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "domains": sorted(domains) if domains else "all",
        "splits": sorted(splits) if splits else "all",
        "input_files": len(paths),
        "file_index": args.file_index,
        "file_stride": args.file_stride,
        "file_workers": args.file_workers,
        "file_executor": args.file_executor,
        "file_chunksize": args.file_chunksize,
        "max_records_per_shard": args.max_records_per_shard,
        "workers": args.workers,
        "record_start_method": args.record_start_method,
        "maxtasksperchild": args.maxtasksperchild,
        "max_graphs_per_chunk": args.max_graphs_per_chunk,
        "max_nodes_per_chunk": args.max_nodes_per_chunk,
        "graph_format": args.graph_format,
        "dry_run": args.dry_run,
    }
    print(json.dumps({"event": "graph_build_plan", **plan}, indent=2, ensure_ascii=False), flush=True)
    if int(args.file_workers) > 1 and args.file_executor == "thread" and int(args.workers) <= 1:
        log(
            "warning: --file-executor thread with --workers <= 1 may serialize Python-side "
            "work under the GIL; use --file-executor process for shard-level CPU parallelism"
        )
    if int(args.file_workers) > 1 and args.file_executor == "process" and int(args.workers) > 1:
        log(
            "warning: process shard workers plus record workers creates nested process pools; "
            "prefer --workers 1 unless a single shard is the bottleneck"
        )
    if args.dry_run:
        for path in paths[:50]:
            print(path)
        return 0
    if not paths:
        raise SystemExit("no input shards found")

    reports = []
    if int(args.file_workers) <= 1:
        for path in paths:
            reports.append(process_input_file(path, input_root, output_root, args, flavors, parser_version))
    else:
        args_payload = vars(args).copy()
        for key in ("config", "root", "input_root", "output_root", "input_relatives_file"):
            if args_payload.get(key) is not None:
                args_payload[key] = str(args_payload[key])
        tasks = [
            {
                "input_path": str(path),
                "relative": str(path.relative_to(input_root)),
                "input_root": str(input_root),
                "output_root": str(output_root),
                "args": args_payload,
                "flavors": flavors,
                "parser_version": parser_version,
            }
            for path in paths
        ]
        started = time.time()
        completed = 0
        failed = 0
        executor_cls = futures.ThreadPoolExecutor if args.file_executor == "thread" else futures.ProcessPoolExecutor
        with executor_cls(max_workers=int(args.file_workers)) as pool:
            for report in pool.map(process_input_file_task, tasks, chunksize=max(1, int(args.file_chunksize))):
                reports.append(report)
                completed += 1
                if str(report.get("status") or "") == "failed_exception":
                    failed += 1
                if completed == 1 or completed % max(1, int(args.file_workers)) == 0 or completed == len(tasks):
                    elapsed = time.time() - started
                    log(
                        f"graph file-progress completed={completed}/{len(tasks)} failed={failed} "
                        f"elapsed={elapsed:.1f}s"
                    )
        if failed:
            raise RuntimeError(f"{failed} input shard(s) failed; inspect reports for failed_exception")
    report_path = write_run_report(root, output_root, report_prefix, reports, dry_run=False)
    print(json.dumps({"event": "graph_report_written", "path": str(report_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
