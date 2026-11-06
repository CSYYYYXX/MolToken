"""Heavy atom graph 分片 I/O 和签名工具。"""

from __future__ import annotations

import io
import json
import pickle
import shutil
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from pipeline.paths import PipelinePaths


DEFAULT_DATA_ROOT = PipelinePaths().data_root
DEFAULT_GRAPH_ROOT = PipelinePaths().graph_root
GRAPH_STREAM_MARKER = {"__graph_shard_format__": "pickle_stream_v1"}
COMPACT_GRAPH_MAGIC = b"CSYCGPH1"
COMPACT_GRAPH_VERSION = 1
COMPACT_HEADER_LEN_STRUCT = struct.Struct("<Q")


def _numpy_dtype_for_signed(min_value: int, max_value: int) -> np.dtype:
    """根据实际范围选择无损 signed dtype。"""

    if np.iinfo(np.int8).min <= min_value and max_value <= np.iinfo(np.int8).max:
        return np.dtype("<i1")
    if np.iinfo(np.int16).min <= min_value and max_value <= np.iinfo(np.int16).max:
        return np.dtype("<i2")
    if np.iinfo(np.int32).min <= min_value and max_value <= np.iinfo(np.int32).max:
        return np.dtype("<i4")
    return np.dtype("<i8")


def _numpy_dtype_for_unsigned(max_value: int) -> np.dtype:
    """根据实际范围选择无损 unsigned dtype。"""

    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype("<u1")
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype("<u2")
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype("<u4")
    return np.dtype("<u8")


def _compact_signed_array(rows: list[np.ndarray], columns: int) -> np.ndarray:
    """把多段 feature array 拼接后降到最小无损 signed dtype。"""

    if rows:
        arr64 = np.concatenate([np.asarray(row, dtype=np.int64).reshape((-1, columns)) for row in rows], axis=0)
    else:
        arr64 = np.zeros((0, columns), dtype=np.int64)
    if arr64.size == 0:
        return arr64.astype(np.dtype("<i1"), copy=False)
    dtype = _numpy_dtype_for_signed(int(arr64.min()), int(arr64.max()))
    return arr64.astype(dtype, copy=False)


def _compact_unsigned_array(values: list[np.ndarray]) -> np.ndarray:
    """把多段 index array 拼接后降到最小无损 unsigned dtype。"""

    if values:
        arr64 = np.concatenate([np.asarray(row, dtype=np.uint64).reshape((-1,)) for row in values], axis=0)
    else:
        arr64 = np.zeros((0,), dtype=np.uint64)
    max_value = int(arr64.max()) if arr64.size else 0
    dtype = _numpy_dtype_for_unsigned(max_value)
    return arr64.astype(dtype, copy=False)


def find_zstd() -> str:
    """寻找可用的 zstd 命令，用于读取 ``*.pkl.zst`` graph 分片。"""

    found = shutil.which("zstd")
    if found:
        return found
    for candidate in (
        
        "/usr/bin/zstd",
        "/usr/local/bin/zstd",
    ):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("zstd executable not found")


def _zstd_write(path: Path, level: int, threads: int):
    """打开一个写入 zstd 压缩流的进程。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    zstd = find_zstd()
    tmp = path.with_suffix(path.suffix + ".tmp")
    proc = subprocess.Popen(
        [zstd, "-q", f"-T{max(1, int(threads))}", f"-{int(level)}", "-f", "-o", str(tmp)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    return proc, tmp


def _finish_zstd_write(proc: subprocess.Popen, tmp: Path, path: Path) -> int:
    """关闭 zstd 写入进程并原子替换目标文件。"""

    assert proc.stdin is not None
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"zstd failed for {path} rc={rc}")
    tmp.replace(path)
    return path.stat().st_size


def _metadata_for_graph(graph: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """提取可 JSON 化的 per-graph metadata，不丢弃 properties/sequence/smiles。"""

    payload: dict[str, Any] = {
        "sample_id": str(graph.get("sample_id") or meta.get("sample_id") or ""),
        "num_nodes": int(graph.get("num_nodes") or 0),
        "properties": graph.get("properties") if isinstance(graph.get("properties"), dict) else {},
        "meta": meta,
    }
    if "sequence" in graph:
        payload["sequence"] = graph["sequence"]
    if "smiles" in graph:
        payload["smiles"] = graph["smiles"]
    return payload


def _restore_graph(
    item: dict[str, Any],
    node_features: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_features: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """从 compact arrays 恢复旧 graph dict 结构。"""

    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    graph: dict[str, Any] = {
        "sample_id": str(item.get("sample_id") or meta.get("sample_id") or ""),
        "num_nodes": int(item.get("num_nodes") or len(node_features)),
        "edges": (
            [int(x) for x in np.asarray(edge_src, dtype=np.uint64).tolist()],
            [int(x) for x in np.asarray(edge_dst, dtype=np.uint64).tolist()],
        ),
        "node_features": np.asarray(node_features, dtype=np.int64).reshape((-1, 6)),
        "edge_features": np.asarray(edge_features, dtype=np.int64).reshape((-1, 3)),
        "properties": item.get("properties") if isinstance(item.get("properties"), dict) else {},
    }
    node_type_arr = graph["node_features"][:, 0].astype(np.int64, copy=False)
    edge_type_arr = graph["edge_features"][:, 0].astype(np.int64, copy=False)
    graph["node_type_ids"] = node_type_arr
    graph["edge_type_ids"] = edge_type_arr
    graph["node_token_ids"] = (node_type_arr * 2 + 1).reshape((-1, 1))
    graph["edge_token_ids"] = (edge_type_arr * 2).reshape((-1, 1))
    if "sequence" in item:
        graph["sequence"] = item["sequence"]
    if "smiles" in item:
        graph["smiles"] = item["smiles"]
    return graph, meta


def write_compact_graph_zst(
    path: Path,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    zstd_level: int = 1,
    zstd_threads: int = 1,
) -> int:
    """写无损 compact graph shard。

    格式特点：

    - 每个 graph 的 metadata 完整写入 JSON header；
    - 节点/边 feature 和边 index 拼成连续数组，按实际范围降 dtype；
    - ``node_type_ids`` / ``edge_type_ids`` / token ids 不落盘，读取时无损重建；
    - 输出整体再用 zstd 压缩。
    """

    node_chunks: list[np.ndarray] = []
    edge_feature_chunks: list[np.ndarray] = []
    edge_src_chunks: list[np.ndarray] = []
    edge_dst_chunks: list[np.ndarray] = []
    graph_items: list[dict[str, Any]] = []
    node_offsets = [0]
    edge_offsets = [0]
    for graph, meta in records:
        nfeat = np.asarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
        efeat = np.asarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
        src_raw, dst_raw = graph["edges"]
        src = np.asarray(src_raw, dtype=np.uint64).reshape((-1,))
        dst = np.asarray(dst_raw, dtype=np.uint64).reshape((-1,))
        if len(src) != len(dst) or len(src) != len(efeat):
            raise ValueError(f"edge length mismatch for sample_id={graph.get('sample_id')}")
        if int(graph["num_nodes"]) != len(nfeat):
            raise ValueError(f"node length mismatch for sample_id={graph.get('sample_id')}")
        node_chunks.append(nfeat)
        edge_feature_chunks.append(efeat)
        edge_src_chunks.append(src)
        edge_dst_chunks.append(dst)
        graph_items.append(_metadata_for_graph(graph, meta))
        node_offsets.append(node_offsets[-1] + len(nfeat))
        edge_offsets.append(edge_offsets[-1] + len(src))

    arrays = {
        "node_features": _compact_signed_array(node_chunks, 6),
        "edge_features": _compact_signed_array(edge_feature_chunks, 3),
        "edge_src": _compact_unsigned_array(edge_src_chunks),
        "edge_dst": _compact_unsigned_array(edge_dst_chunks),
        "node_offsets": np.asarray(node_offsets, dtype=np.dtype("<u8")),
        "edge_offsets": np.asarray(edge_offsets, dtype=np.dtype("<u8")),
    }
    header = {
        "format": "compact_heavy_atom_graph",
        "version": COMPACT_GRAPH_VERSION,
        "records": len(records),
        "graphs": graph_items,
        "arrays": {
            name: {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "nbytes": int(arr.nbytes),
            }
            for name, arr in arrays.items()
        },
    }
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    proc, tmp = _zstd_write(path, zstd_level, zstd_threads)
    try:
        assert proc.stdin is not None
        proc.stdin.write(COMPACT_GRAPH_MAGIC)
        proc.stdin.write(COMPACT_HEADER_LEN_STRUCT.pack(len(header_bytes)))
        proc.stdin.write(header_bytes)
        for name in ("node_features", "edge_features", "edge_src", "edge_dst", "node_offsets", "edge_offsets"):
            proc.stdin.write(np.ascontiguousarray(arrays[name]).tobytes(order="C"))
    except Exception:
        proc.kill()
        proc.wait()
        if tmp.exists():
            tmp.unlink()
        raise
    return _finish_zstd_write(proc, tmp, path)


def _read_exact(stream: io.BufferedReader, size: int) -> bytes:
    """从 stream 读取指定字节数。"""

    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"unexpected EOF while reading compact graph: need={size} got={len(data)}")
    return data


def iter_compact_graph_zst(path: Path, zstd: str | None = None) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """流式读取 ``write_compact_graph_zst`` 写出的 compact graph shard。"""

    zstd = zstd or find_zstd()
    proc = subprocess.Popen([zstd, "-q", "-dc", str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        magic = _read_exact(proc.stdout, len(COMPACT_GRAPH_MAGIC))
        if magic != COMPACT_GRAPH_MAGIC:
            raise ValueError(f"not a compact graph shard: {path}")
        header_len = COMPACT_HEADER_LEN_STRUCT.unpack(_read_exact(proc.stdout, COMPACT_HEADER_LEN_STRUCT.size))[0]
        header = json.loads(_read_exact(proc.stdout, int(header_len)).decode("utf-8"))
        arrays: dict[str, np.ndarray] = {}
        for name in ("node_features", "edge_features", "edge_src", "edge_dst", "node_offsets", "edge_offsets"):
            spec = header["arrays"][name]
            dtype = np.dtype(spec["dtype"])
            shape = tuple(int(x) for x in spec["shape"])
            raw = _read_exact(proc.stdout, int(spec["nbytes"]))
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape)
        node_offsets = np.asarray(arrays["node_offsets"], dtype=np.uint64)
        edge_offsets = np.asarray(arrays["edge_offsets"], dtype=np.uint64)
        for idx, item in enumerate(header["graphs"]):
            ns, ne = int(node_offsets[idx]), int(node_offsets[idx + 1])
            es, ee = int(edge_offsets[idx]), int(edge_offsets[idx + 1])
            yield _restore_graph(
                item,
                arrays["node_features"][ns:ne],
                arrays["edge_src"][es:ee],
                arrays["edge_dst"][es:ee],
                arrays["edge_features"][es:ee],
            )
    finally:
        proc.stdout.close()
        rc = proc.wait()
        if rc not in (0, -13):
            raise RuntimeError(f"zstd -dc failed for {path} rc={rc}")


def iter_pickle_zst(path: Path, zstd: str | None = None) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """流式读取一个由 pickle 再 zstd 压缩的 graph 分片。

    兼容两种格式：

    - 旧格式：整个 shard 是一个 ``list[(graph, meta)]`` pickle 对象；
    - 新格式：第一个 pickle 对象是 ``GRAPH_STREAM_MARKER``，后续每个对象是一条
      ``(graph, meta)``。新格式用于超大蛋白/DNA shard，避免写入端把整个 shard
      的 graph 同时放在内存里。
    """

    zstd = zstd or find_zstd()
    proc = subprocess.Popen([zstd, "-q", "-dc", str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        first = pickle.load(proc.stdout)
        if isinstance(first, dict) and first == GRAPH_STREAM_MARKER:
            while True:
                try:
                    yield pickle.load(proc.stdout)
                except EOFError:
                    break
        else:
            for record in first:
                yield record
    finally:
        proc.stdout.close()
        rc = proc.wait()
        if rc not in (0, -13):
            raise RuntimeError(f"zstd -dc failed for {path} rc={rc}")


def iter_graph_records(path: Path, zstd: str | None = None) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """按文件后缀自动读取 pickle graph 或 compact graph。"""

    name = path.name
    if name.endswith(".cgraph.zst"):
        yield from iter_compact_graph_zst(path, zstd)
    elif name.endswith(".pkl.zst"):
        yield from iter_pickle_zst(path, zstd)
    else:
        raise ValueError(f"unsupported graph shard suffix: {path}")


def pick_graph_records(
    graph_root: Path,
    domains: list[str],
    split: str,
    max_files_per_domain: int,
    max_records_per_domain: int,
    max_nodes_per_record: int,
) -> list[tuple[str, Path, int, dict[str, Any]]]:
    """从每个 domain 抽少量 graph record，用于 smoke test 或小规模实验。"""

    zstd = find_zstd()
    records: list[tuple[str, Path, int, dict[str, Any]]] = []
    for domain in domains:
        root = graph_root / domain / split
        files = sorted(root.glob("*.pkl.zst"))
        if not files:
            continue
        domain_count = 0
        for graph_file in files[: max(1, int(max_files_per_domain))]:
            for record_idx, (graph, _meta) in enumerate(iter_pickle_zst(graph_file, zstd)):
                if max_nodes_per_record > 0 and int(graph["num_nodes"]) > max_nodes_per_record:
                    continue
                records.append((domain, graph_file, record_idx, graph))
                domain_count += 1
                if domain_count >= max_records_per_domain:
                    break
            if domain_count >= max_records_per_domain:
                break
    return records


def graph_signature(graph_data: dict[str, Any], remap: dict[int, int] | None = None) -> dict[str, Any]:
    """生成 graph 的节点/有向边签名，用于反序列化一致性检查。"""

    node_features = np.asarray(graph_data["node_features"], dtype=np.int64).reshape((-1, 6))
    edge_features = np.asarray(graph_data["edge_features"], dtype=np.int64).reshape((-1, 3))
    if remap is None:
        node_sig = [tuple(int(x) for x in row.tolist()) for row in node_features]
        src_raw, dst_raw = graph_data["edges"]
        edge_sig = Counter(
            (
                int(u),
                int(v),
                tuple(int(x) for x in edge_features[i].tolist()),
            )
            for i, (u, v) in enumerate(zip(src_raw, dst_raw))
        )
    else:
        node_sig = [None] * len(remap)
        for old_id, new_id in remap.items():
            node_sig[new_id] = tuple(int(x) for x in node_features[old_id].tolist())
        src_raw, dst_raw = graph_data["edges"]
        edge_sig = Counter(
            (
                remap[int(u)],
                remap[int(v)],
                tuple(int(x) for x in edge_features[i].tolist()),
            )
            for i, (u, v) in enumerate(zip(src_raw, dst_raw))
        )
    return {"nodes": node_sig, "edges": edge_sig}
