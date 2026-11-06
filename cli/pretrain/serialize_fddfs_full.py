#!/usr/bin/env python3
"""并行执行 full heavy atom graph -> compact-int 序列化。

流程：

1. count：流式读取 graph 分片，统计 node/edge/triplet 频率，构造 frequency guide；
2. vocab：用统计到的 node/edge feature 建基础 vocab，动态编码 R<num>；
3. serialize：再次流式读取 graph 分片，按 FDDFS 序列化并写 YTTM compact-int binary。

这个脚本必须在 worker 上运行。不要在跳板机上跑全量。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.graph.io import iter_graph_records  # noqa: E402
from pipeline.paths import PipelinePaths  # noqa: E402
from pipeline.serialization.freq_dfs_core import (  # noqa: E402
    FrequencyGuide,
    build_undirected_edges,
    decode_feature,
    edge_features,
    encode_feature,
    node_features,
)
from pipeline.serialization.fddfs import (
    build_vocab as build_fddfs_vocab,
    encode_fddfs_token,
    serialize_fddfs,
)
try:
    from pipeline.serialization.stackk_fast import (  # noqa: E402
        FAST_FDDFS_AVAILABLE,
        FAST_FDDFS_ERROR,
        FAST_STACKK_COUNT_AVAILABLE,
        FAST_STACKK_COUNT_ERROR,
        FastFDDFSEncoder,
        FastStackKCounter,
    )
except Exception as exc:  # noqa: BLE001, E722
    FAST_FDDFS_AVAILABLE = False
    FAST_FDDFS_ERROR = exc
    FAST_STACKK_COUNT_AVAILABLE = False
    FAST_STACKK_COUNT_ERROR = exc
    FastFDDFSEncoder = None  # type: ignore[assignment]
    FastStackKCounter = None  # type: ignore[assignment]


SAMPLE_MAGIC = b"YTTMIBPE"
SAMPLE_VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIQQi")
LEN_STRUCT = struct.Struct("<I")
GLOBAL_GUIDE: FrequencyGuide | None = None
GLOBAL_VOCAB: dict[str, Any] | None = None
GLOBAL_FAST_ENCODER: Any | None = None
GLOBAL_SERIALIZER = "fddfs"


def parse_domains(raw: str) -> list[str]:
    """解析逗号分隔的 domain 列表。"""

    return [item.strip() for item in raw.split(",") if item.strip()]


def tuple_from_token(token: str) -> tuple[int, ...]:
    """把 ``N:1,2`` / ``E:1,2`` 这种 token 的 feature 部分转成 tuple。"""

    _prefix, raw = token.split(":", 1)
    return tuple(int(x) for x in raw.split(",") if x)


def collect_graph_files(
    graph_root: Path,
    domains: list[str],
    splits: list[str],
    max_files_per_domain: int,
) -> list[tuple[str, str, Path]]:
    """收集需要处理的 graph 分片。"""

    files: list[tuple[str, str, Path]] = []
    for domain in domains:
        for split in splits:
            root = graph_root / domain / split
            domain_files = sorted(root.glob("*.pkl.zst")) + sorted(root.glob("*.cgraph.zst"))
            if max_files_per_domain > 0:
                domain_files = domain_files[:max_files_per_domain]
            files.extend((domain, split, path) for path in domain_files)
    if not files:
        raise FileNotFoundError(f"no graph shards found under {graph_root}")
    return files


def count_graph_file(task: tuple[str, str, str, bool]) -> dict[str, Any]:
    """统计一个 graph 分片里的 DFS guide 频率。"""

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))
    domain, split, raw_path, use_cpp_count = task
    path = Path(raw_path)
    started = time.time()
    if use_cpp_count:
        if not FAST_STACKK_COUNT_AVAILABLE or FastStackKCounter is None:
            raise RuntimeError(f"C++ stackK count backend requested but unavailable: {FAST_STACKK_COUNT_ERROR}")
        counter = FastStackKCounter()
        for graph, _meta in iter_graph_records(path):
            counter.add_graph(graph)
        payload = counter.result()
        return {
            "domain": domain,
            "split": split,
            "path": str(path),
            "stats": payload["stats"],
            "node_counts": payload["node_counts"],
            "edge_counts": payload["edge_counts"],
            "triplet_counts": payload["triplet_counts"],
            "elapsed_sec": time.time() - started,
            "count_backend": "cpp",
        }

    node_counts: Counter[tuple[int, ...]] = Counter()
    edge_counts: Counter[tuple[int, ...]] = Counter()
    triplet_counts: Counter[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = Counter()
    stats: Counter[str] = Counter()
    for graph, _meta in iter_graph_records(path):
        stats["records"] += 1
        stats["nodes"] += int(graph["num_nodes"])
        src, _dst = graph["edges"]
        stats["directed_edges"] += len(src)
        nfeat = node_features(graph)
        for feat in nfeat:
            node_counts[feat] += 1
        edges, _adj = build_undirected_edges(graph)
        stats["undirected_edges"] += len(edges)
        for edge in edges:
            edge_counts[edge.feat] += 1
            a = nfeat[edge.u]
            b = nfeat[edge.v]
            triplet_counts[(a, edge.feat, b)] += 1
            triplet_counts[(b, edge.feat, a)] += 1
    return {
        "domain": domain,
        "split": split,
        "path": str(path),
        "stats": dict(stats),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "triplet_counts": triplet_counts,
        "elapsed_sec": time.time() - started,
        "count_backend": "python",
    }


def merge_count_results(results: Iterable[dict[str, Any]], total: int, log_every: int) -> dict[str, Any]:
    """合并所有 count worker 的结果。"""

    merged_node: Counter[tuple[int, ...]] = Counter()
    merged_edge: Counter[tuple[int, ...]] = Counter()
    merged_triplet: Counter[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = Counter()
    merged_stats: Counter[str] = Counter()
    domain_stats: dict[str, Counter[str]] = {}
    split_stats: dict[str, Counter[str]] = {}
    started = time.time()
    for idx, result in enumerate(results, 1):
        domain = result["domain"]
        split = result["split"]
        domain_stats.setdefault(domain, Counter())
        domain_stats[domain].update(result["stats"])
        split_stats.setdefault(split, Counter())
        split_stats[split].update(result["stats"])
        merged_stats.update(result["stats"])
        merged_node.update(result["node_counts"])
        merged_edge.update(result["edge_counts"])
        merged_triplet.update(result["triplet_counts"])
        if log_every > 0 and (idx == 1 or idx % log_every == 0 or idx == total):
            elapsed = time.time() - started
            print(
                "count_progress "
                f"files={idx}/{total} records={int(merged_stats['records'])} "
                f"nodes={int(merged_stats['nodes'])} elapsed_sec={elapsed:.2f}",
                flush=True,
            )
    default_edge = merged_edge.most_common(1)[0][0] if merged_edge else (1, 0, 0)
    return {
        "node_counts": merged_node,
        "edge_counts": merged_edge,
        "triplet_counts": merged_triplet,
        "stats": merged_stats,
        "domain_stats": domain_stats,
        "split_stats": split_stats,
        "default_edge_feature": default_edge,
    }


def load_count_payload(path: Path) -> dict[str, Any]:
    """从落盘的 guide_counts.json 反解出 build_guide 需要的 payload。

    落盘时 key 被 ``encode_feature`` 编成字符串，内存里的 payload 用的是 tuple，
    这里负责转回去。下游微调预处理靠这个函数复用语料级 guide，而不是拿自己那点
    数据现场重建——后者会让 DFS 顺序与预训练不一致。
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if "triplet_counts" not in raw:
        raise ValueError(
            f"{path} 里没有 triplet_counts，这是 2026-08 之前的旧格式（只存了 "
            f"triplet_unique={raw.get('triplet_unique')} 这个计数）。"
            "triplet_counts 是邻居排序的主键，无法从其它字段推出来，"
            "必须重跑 --phase count 生成新的 guide_counts.json。"
        )

    triplet_counts: Counter[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = Counter()
    for key, value in raw["triplet_counts"].items():
        src_tok, edge_tok, dst_tok = key.split("|")
        triplet_counts[
            (
                decode_feature(src_tok, "N"),
                decode_feature(edge_tok, "E"),
                decode_feature(dst_tok, "N"),
            )
        ] = int(value)

    return {
        "node_counts": Counter(
            {decode_feature(k, "N"): int(v) for k, v in raw["node_counts"].items()}
        ),
        "edge_counts": Counter(
            {decode_feature(k, "E"): int(v) for k, v in raw["edge_counts"].items()}
        ),
        "triplet_counts": triplet_counts,
        "default_edge_feature": tuple(int(x) for x in raw["default_edge_feature"]),
    }


def build_guide(payload: dict[str, Any]) -> FrequencyGuide:
    """从 count payload 构造 FrequencyGuide，不把 graph 载入内存。"""

    guide = FrequencyGuide.__new__(FrequencyGuide)
    guide.node_counts = Counter(payload["node_counts"])
    guide.edge_counts = Counter(payload["edge_counts"])
    guide.triplet_counts = Counter(payload["triplet_counts"])
    guide.default_edge_feature = tuple(int(x) for x in payload["default_edge_feature"])
    return guide


def write_header(f, nseq: int, ntokens: int, max_token: int) -> None:
    """写 compact-int 文件头。"""

    f.seek(0)
    f.write(HEADER_STRUCT.pack(SAMPLE_MAGIC, SAMPLE_VERSION, int(nseq), int(ntokens), int(max_token)))


def write_int_sequence(f, ids: list[int]) -> None:
    """写一条 int32 序列。"""

    f.write(LEN_STRUCT.pack(len(ids)))
    arr = array("i", ids)
    if sys.byteorder != "little":
        arr.byteswap()
    arr.tofile(f)


def output_paths(out_dir: Path, domain: str, source_path: Path, split: str) -> tuple[Path, Path]:
    """根据输入分片生成输出 binary 和 summary 路径。"""

    rel_name = source_path.name.removesuffix(".pkl.zst")
    rel_name = rel_name.removesuffix(".cgraph.zst")
    token_path = out_dir / "tokens" / domain / split / f"{rel_name}.fddfs.bin"
    summary_path = out_dir / "summaries" / domain / split / f"{rel_name}.json"
    return token_path, summary_path


def serialize_graph_file(task: dict[str, Any]) -> dict[str, Any]:
    """序列化一个 graph 分片并写 compact-int binary。"""

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))
    if GLOBAL_GUIDE is None or GLOBAL_VOCAB is None:
        raise RuntimeError("serialize worker was not initialized")
    domain = task["domain"]
    split = task["split"]
    path = Path(task["path"])
    out_dir = Path(task["out_dir"])
    overwrite = bool(task["overwrite"])
    guide = GLOBAL_GUIDE
    vocab = GLOBAL_VOCAB
    fast_encoder = GLOBAL_FAST_ENCODER
    omit_default_edge = bool(vocab["omit_default_edge"])
    token_path, summary_path = output_paths(out_dir, domain, path, split)
    if token_path.exists() and summary_path.exists() and not overwrite:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["skipped_existing"] = True
        return data

    token_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_token = token_path.with_suffix(token_path.suffix + ".tmp")
    stats: Counter[str] = Counter()
    max_token = -1
    max_k = 0
    max_ring_id = 0
    max_active_ring_ids = 0
    max_sample_tokens = 0
    started = time.time()
    with tmp_token.open("wb") as f:
        write_header(f, 0, 0, 0)
        for graph, _meta in iter_graph_records(path):
            if fast_encoder is not None:
                encoded = fast_encoder.serialize_ids(graph)
                ids = [int(x) for x in encoded["ids"]]
                result_stats = encoded["stats"]
            else:
                result = serialize_fddfs(graph, guide, omit_default_edge=omit_default_edge)
                ids = [encode_fddfs_token(token, vocab) for token in result.sequence]
                result_stats = result.stats
            sample_tokens = len(ids)
            if ids:
                max_token = max(max_token, max(ids))
            max_k = max(max_k, int(result_stats.get("max_stackk_cycle_len", 0)))
            max_ring_id = max(max_ring_id, int(result_stats.get("max_ring_id", 0)))
            max_active_ring_ids = max(max_active_ring_ids, int(result_stats.get("max_active_ring_ids", 0)))
            max_sample_tokens = max(max_sample_tokens, sample_tokens)
            write_int_sequence(f, ids)
            stats["records"] += 1
            stats["input_nodes"] += int(graph["num_nodes"])
            src, _dst = graph["edges"]
            stats["input_directed_edges"] += len(src)
            stats["tokens"] += sample_tokens
            stats["tree_edges"] += int(result_stats.get("tree_edges", 0))
            stats["non_tree_edges"] += int(result_stats.get("non_tree_edges", 0))
            stats["branch_pairs"] += int(result_stats.get("branch_pairs", 0))
            stats["sum_stackk_cycle_len"] += int(result_stats.get("sum_stackk_cycle_len", 0))
            stats["ring_tokens"] += int(result_stats.get("ring_tokens", 0))
            stats["ring_open_tokens"] += int(result_stats.get("ring_open_tokens", 0))
            stats["ring_close_tokens"] += int(result_stats.get("ring_close_tokens", 0))
        if max_token < 0:
            max_token = 0
        write_header(f, int(stats["records"]), int(stats["tokens"]), int(max_token))
    os.replace(tmp_token, token_path)
    summary = {
        "domain": domain,
        "split": split,
        "serializer": "fddfs",
        "source": str(path),
        "output": str(token_path),
        "records": int(stats["records"]),
        "input_nodes": int(stats["input_nodes"]),
        "input_directed_edges": int(stats["input_directed_edges"]),
        "tokens": int(stats["tokens"]),
        "tree_edges": int(stats["tree_edges"]),
        "non_tree_edges": int(stats["non_tree_edges"]),
        "branch_pairs": int(stats["branch_pairs"]),
        "sum_stackk_cycle_len": int(stats["sum_stackk_cycle_len"]),
        "ring_tokens": int(stats["ring_tokens"]),
        "ring_open_tokens": int(stats["ring_open_tokens"]),
        "ring_close_tokens": int(stats["ring_close_tokens"]),
        "max_k": int(max_k),
        "max_ring_id": int(max_ring_id),
        "max_active_ring_ids": int(max_active_ring_ids),
        "max_sample_tokens": int(max_sample_tokens),
        "max_token": int(max_token),
        "elapsed_sec": time.time() - started,
        "tokens_per_sec": float(stats["tokens"] / max(time.time() - started, 1e-9)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def init_serialize_worker(guide_payload: dict[str, Any], vocab: dict[str, Any], use_cpp_serializer: bool) -> None:
    """在每个 serialize worker 内初始化只读 guide/vocab。"""

    global GLOBAL_GUIDE, GLOBAL_VOCAB, GLOBAL_FAST_ENCODER, GLOBAL_SERIALIZER
    GLOBAL_GUIDE = build_guide(guide_payload)
    GLOBAL_VOCAB = vocab
    GLOBAL_SERIALIZER = "fddfs"
    if use_cpp_serializer:
        if not FAST_FDDFS_AVAILABLE or FastFDDFSEncoder is None:
            raise RuntimeError(f"C++ FDDFS backend requested but unavailable: {FAST_FDDFS_ERROR}")
        GLOBAL_FAST_ENCODER = FastFDDFSEncoder(GLOBAL_GUIDE, vocab)
    else:
        GLOBAL_FAST_ENCODER = None
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))


def json_counter(counter: Counter[Any]) -> dict[str, int]:
    """把 Counter 转成 JSON 可写的 dict。"""

    return {str(key): int(value) for key, value in counter.items()}


def save_count_payload(out_dir: Path, payload: dict[str, Any], vocab: dict[str, Any]) -> None:
    """保存 guide 统计和 vocab。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    guide_json = {
        "default_edge_feature": list(int(x) for x in payload["default_edge_feature"]),
        "node_counts": {encode_feature("N", key): int(value) for key, value in payload["node_counts"].items()},
        "edge_counts": {encode_feature("E", key): int(value) for key, value in payload["edge_counts"].items()},
        "triplet_unique": len(payload["triplet_counts"]),
        # triplet_counts 是 transition_key_from_features 的排序主键，缺了它无法复现
        # guide。2026-08 之前只存了 triplet_unique 计数，导致下游微调数据只能现场
        # 重建 guide，与预训练不一致。key 格式 "N:...|E:...|N:..."，由
        # load_count_payload() 反解。
        "triplet_counts": {
            "|".join(
                (
                    encode_feature("N", src),
                    encode_feature("E", edge),
                    encode_feature("N", dst),
                )
            ): int(value)
            for (src, edge, dst), value in payload["triplet_counts"].items()
        },
        "stats": {key: int(value) for key, value in payload["stats"].items()},
        "domain_stats": {
            domain: {key: int(value) for key, value in stats.items()}
            for domain, stats in payload["domain_stats"].items()
        },
        "split_stats": {
            split: {key: int(value) for key, value in stats.items()}
            for split, stats in payload.get("split_stats", {}).items()
        },
    }
    (out_dir / "guide_counts.json").write_text(json.dumps(guide_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate_serialize_results(results: Iterable[dict[str, Any]], total: int, log_every: int) -> dict[str, Any]:
    """合并序列化结果。"""

    stats: Counter[str] = Counter()
    domain_stats: dict[str, Counter[str]] = {}
    split_stats: dict[str, Counter[str]] = {}
    max_k = 0
    max_ring_id = 0
    max_active_ring_ids = 0
    max_token = 0
    max_sample_tokens = 0
    outputs: list[str] = []
    started = time.time()
    for idx, item in enumerate(results, 1):
        domain = item["domain"]
        split = item["split"]
        domain_stats.setdefault(domain, Counter())
        split_stats.setdefault(split, Counter())
        for key in (
            "records",
            "input_nodes",
            "input_directed_edges",
            "tokens",
            "tree_edges",
            "non_tree_edges",
            "branch_pairs",
            "sum_stackk_cycle_len",
            "ring_tokens",
            "ring_open_tokens",
            "ring_close_tokens",
        ):
            stats[key] += int(item.get(key, 0))
            domain_stats[domain][key] += int(item.get(key, 0))
            split_stats[split][key] += int(item.get(key, 0))
        max_k = max(max_k, int(item.get("max_k", 0)))
        max_ring_id = max(max_ring_id, int(item.get("max_ring_id", 0)))
        max_active_ring_ids = max(max_active_ring_ids, int(item.get("max_active_ring_ids", 0)))
        max_token = max(max_token, int(item.get("max_token", 0)))
        item_max_sample_tokens = int(item.get("max_sample_tokens", 0))
        max_sample_tokens = max(max_sample_tokens, item_max_sample_tokens)
        domain_stats[domain]["max_sample_tokens"] = max(
            int(domain_stats[domain].get("max_sample_tokens", 0)),
            item_max_sample_tokens,
        )
        split_stats[split]["max_sample_tokens"] = max(
            int(split_stats[split].get("max_sample_tokens", 0)),
            item_max_sample_tokens,
        )
        outputs.append(item["output"])
        if log_every > 0 and (idx == 1 or idx % log_every == 0 or idx == total):
            elapsed = time.time() - started
            print(
                "serialize_progress "
                f"files={idx}/{total} records={int(stats['records'])} tokens={int(stats['tokens'])} "
                f"elapsed_sec={elapsed:.2f} tokens_per_sec={stats['tokens'] / max(elapsed, 1e-9):.2f}",
                flush=True,
            )
    elapsed = time.time() - started
    return {
        "records": int(stats["records"]),
        "input_nodes": int(stats["input_nodes"]),
        "input_directed_edges": int(stats["input_directed_edges"]),
        "input_undirected_edges": int(stats["input_directed_edges"] // 2),
        "tokens": int(stats["tokens"]),
        "tree_edges": int(stats["tree_edges"]),
        "non_tree_edges": int(stats["non_tree_edges"]),
        "branch_pairs": int(stats["branch_pairs"]),
        "sum_stackk_cycle_len": int(stats["sum_stackk_cycle_len"]),
        "ring_tokens": int(stats["ring_tokens"]),
        "ring_open_tokens": int(stats["ring_open_tokens"]),
        "ring_close_tokens": int(stats["ring_close_tokens"]),
        "max_k": int(max_k),
        "max_ring_id": int(max_ring_id),
        "max_active_ring_ids": int(max_active_ring_ids),
        "max_sample_tokens": int(max_sample_tokens),
        "max_token": int(max_token),
        "observed_vocab_size": int(max_token + 1),
        "elapsed_sec": elapsed,
        "tokens_per_sec": float(stats["tokens"] / max(elapsed, 1e-9)),
        "domain_stats": {
            domain: {key: int(value) for key, value in values.items()}
            for domain, values in domain_stats.items()
        },
        "split_stats": {
            split: {key: int(value) for key, value in values.items()}
            for split, values in split_stats.items()
        },
        "token_files": outputs,
    }


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=PipelinePaths().graph_root)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--domains", default="small_molecule,peptide,protein,dna,rna")
    parser.add_argument("--split", default="train", help="逗号分隔 split 列表；例如 train 或 train,valid,test。")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--max-files-per-domain", type=int, default=0)
    parser.add_argument("--phase", choices=["all", "count", "serialize"], default="all")
    parser.add_argument("--omit-default-edge", action="store_true", default=True)
    parser.add_argument("--keep-default-edge", action="store_false", dest="omit_default_edge")
    parser.add_argument("--python-count", action="store_true", help="禁用 C++ count hot path，使用纯 Python 频率统计。")
    parser.add_argument("--python-serializer", action="store_true", help="禁用 C++ serializer hot path，使用纯 Python 序列化。")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every-files", type=int, default=200)
    args = parser.parse_args()

    domains = parse_domains(args.domains)
    splits = parse_domains(args.split)
    if not splits:
        raise ValueError("--split did not contain any split name")
    files = collect_graph_files(args.graph_root, domains, splits, args.max_files_per_domain)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_cpp_count = bool(FAST_STACKK_COUNT_AVAILABLE and not args.python_count)
    disable_cpp_serializer = bool(args.python_serializer)
    use_cpp_serializer = bool(FAST_FDDFS_AVAILABLE and not disable_cpp_serializer)
    serializer_backend_name = "cpp" if use_cpp_serializer else "python"
    print(
        "fddfs_full_start "
        f"phase={args.phase} graph_root={args.graph_root} out_dir={args.out_dir} "
        f"splits={','.join(splits)} files={len(files)} workers={args.workers} "
        f"cpp_count={use_cpp_count} serializer_backend={serializer_backend_name}",
        flush=True,
    )
    if args.python_count:
        print("count_backend python", flush=True)
    elif FAST_STACKK_COUNT_AVAILABLE:
        print("count_backend cpp", flush=True)
    else:
        raise RuntimeError(f"C++ frequency-count backend is not available; build it or pass --python-count. import_error={FAST_STACKK_COUNT_ERROR}")
    if disable_cpp_serializer:
        print("fddfs_backend python", flush=True)
    elif FAST_FDDFS_AVAILABLE:
        print("fddfs_backend cpp", flush=True)
    else:
        raise RuntimeError(f"C++ FDDFS backend is not available; build it or pass --python-serializer. import_error={FAST_FDDFS_ERROR}")

    count_payload: dict[str, Any]
    vocab: dict[str, Any]
    if args.phase in {"all", "count"}:
        count_started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            count_results = pool.map(
                count_graph_file,
                [(domain, split, str(path), use_cpp_count) for domain, split, path in files],
                chunksize=max(1, args.chunksize),
            )
            count_payload = merge_count_results(count_results, len(files), args.log_every_files)
        vocab = build_fddfs_vocab(
            count_payload["node_counts"],
            count_payload["edge_counts"],
            count_payload["default_edge_feature"],
            args.omit_default_edge,
        )
        vocab["serializer"] = "fddfs"
        save_count_payload(args.out_dir, count_payload, vocab)
        count_summary = {
            "serializer": "fddfs",
            "elapsed_sec": time.time() - count_started,
            "files": len(files),
            "workers": args.workers,
            "count_backend": "cpp" if use_cpp_count else "python",
            "stats": {key: int(value) for key, value in count_payload["stats"].items()},
            "split_stats": {
                split: {key: int(value) for key, value in stats.items()}
                for split, stats in count_payload.get("split_stats", {}).items()
            },
            "default_edge_feature": list(int(x) for x in count_payload["default_edge_feature"]),
            "base_vocab_size": len(vocab["token_to_id"]),
            "k_base": vocab.get("k_base"),
            "ring_base": vocab.get("ring_base"),
        }
        (args.out_dir / "count_summary.json").write_text(json.dumps(count_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("count_done " + json.dumps(count_summary, ensure_ascii=False), flush=True)
        if args.phase == "count":
            return 0
    else:
        # 以前这里只能读回 node/edge counts，triplet_counts 拿不到（落盘时被丢了），
        # 所以直接 raise 要求跑 phase=all。现在 save_count_payload 会落盘
        # triplet_counts，load_count_payload 能完整反解，phase=serialize 可用了。
        count_payload = load_count_payload(args.out_dir / "guide_counts.json")
        count_payload["split_stats"] = {}
        vocab = json.loads((args.out_dir / "vocab.json").read_text(encoding="utf-8"))
        if vocab.get("serializer") != "fddfs":
            raise ValueError(f"vocab is not FDDFS: {args.out_dir / 'vocab.json'}")

    serialize_started = time.time()
    guide_payload = {
        "node_counts": count_payload["node_counts"],
        "edge_counts": count_payload["edge_counts"],
        "triplet_counts": count_payload["triplet_counts"],
        "default_edge_feature": count_payload["default_edge_feature"],
    }
    serialize_tasks = [
        {
            "domain": domain,
            "split": split,
            "path": str(path),
            "out_dir": str(args.out_dir),
            "overwrite": args.overwrite,
        }
        for domain, split, path in files
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_serialize_worker,
        initargs=(guide_payload, vocab, use_cpp_serializer),
    ) as pool:
        serialize_results = pool.map(serialize_graph_file, serialize_tasks, chunksize=max(1, args.chunksize))
        summary = aggregate_serialize_results(serialize_results, len(files), args.log_every_files)
    summary.update(
        {
            "mode": "fddfs_compact_int",
            "serializer": "fddfs",
            "graph_root": str(args.graph_root),
            "out_dir": str(args.out_dir),
            "splits": splits,
            "files": len(files),
            "count_backend": "cpp" if use_cpp_count else "python",
            "serializer_backend": serializer_backend_name,
            "fddfs_backend": serializer_backend_name,
            "workers": args.workers,
            "count_elapsed_sec": float(serialize_started - count_started) if "count_started" in locals() else None,
            "serialize_elapsed_sec": time.time() - serialize_started,
            "default_edge_feature": list(int(x) for x in count_payload["default_edge_feature"]),
            "omit_default_edge": bool(args.omit_default_edge),
            "base_vocab_size": len(vocab["token_to_id"]),
            "k_base": int(vocab["k_base"]) if "k_base" in vocab else None,
            "ring_base": int(vocab["ring_base"]) if "ring_base" in vocab else None,
        }
    )
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("serialize_done " + json.dumps({k: v for k, v in summary.items() if k != "token_files"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
