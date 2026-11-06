#!/usr/bin/env python3
"""校验 C++ FDDFS hot path 与 Python FDDFS 等价。

这个脚本默认只跑手写小图，不读取真实数据；用于确认 C++ 扩展编译后，
``FastFDDFSEncoder`` 输出的 int token id 与 Python ``serialize_fddfs`` +
``encode_fddfs_token`` 完全一致。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.serialization.fddfs import build_vocab, encode_fddfs_token, serialize_fddfs, verify_roundtrip  # noqa: E402
from pipeline.serialization.fddfs import build_frequency_guide  # noqa: E402
from pipeline.serialization.stackk_fast import (  # noqa: E402
    FAST_FDDFS_AVAILABLE,
    FAST_FDDFS_ERROR,
    FAST_STACKK_AVAILABLE,
    FAST_STACKK_IMPORT_ERROR,
    FastFDDFSEncoder,
)


CARBON = (6, 0, 0, 0, 0, 0)
NITROGEN = (7, 0, 0, 0, 0, 0)
OXYGEN = (8, 0, 0, 0, 0, 0)
SINGLE = (1, 0, 0)
DOUBLE = (2, 0, 0)
AROMATIC = (4, 0, 0)


def make_graph(
    node_features: list[tuple[int, int, int, int, int, int]],
    undirected_edges: list[tuple[int, int, tuple[int, int, int]]],
) -> dict[str, Any]:
    """构造最小 heavy atom graph；每条无向边展开成双向 directed edge。"""

    src: list[int] = []
    dst: list[int] = []
    edge_features: list[tuple[int, int, int]] = []
    for u, v, feat in undirected_edges:
        src.extend([u, v])
        dst.extend([v, u])
        edge_features.extend([feat, feat])
    node_arr = np.asarray(node_features, dtype=np.int64)
    edge_arr = np.asarray(edge_features, dtype=np.int64)
    return {
        "num_nodes": int(len(node_features)),
        "edges": (src, dst),
        "node_features": node_arr,
        "edge_features": edge_arr,
        "node_type_ids": node_arr[:, 0],
        "edge_type_ids": edge_arr[:, 0] if len(edge_arr) else np.zeros((0,), dtype=np.int64),
    }


def synthetic_graphs() -> dict[str, dict[str, Any]]:
    """返回覆盖单环、支链、ring id 复用和多活跃环的小图集合。"""

    return {
        "single_cycle_6": make_graph(
            [CARBON] * 6,
            [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 5, SINGLE), (5, 0, SINGLE)],
        ),
        "branched_hetero_cycle": make_graph(
            [CARBON, NITROGEN, CARBON, OXYGEN, CARBON, CARBON, OXYGEN],
            [
                (0, 1, SINGLE),
                (1, 2, DOUBLE),
                (2, 3, SINGLE),
                (3, 4, SINGLE),
                (4, 0, SINGLE),
                (2, 5, SINGLE),
                (5, 6, DOUBLE),
            ],
        ),
        "two_disjoint_cycles_reuse_r1": make_graph(
            [CARBON] * 12,
            [
                (0, 1, AROMATIC),
                (1, 2, AROMATIC),
                (2, 3, AROMATIC),
                (3, 4, AROMATIC),
                (4, 5, AROMATIC),
                (5, 0, AROMATIC),
                (6, 7, SINGLE),
                (7, 8, SINGLE),
                (8, 9, SINGLE),
                (9, 10, SINGLE),
                (10, 11, SINGLE),
                (11, 6, SINGLE),
            ],
        ),
        "fused_two_cycles_need_r2": make_graph(
            [CARBON] * 10,
            [
                (0, 1, SINGLE),
                (1, 2, SINGLE),
                (2, 3, SINGLE),
                (3, 4, SINGLE),
                (4, 5, SINGLE),
                (5, 0, SINGLE),
                (4, 6, SINGLE),
                (6, 7, SINGLE),
                (7, 8, SINGLE),
                (8, 9, SINGLE),
                (9, 5, SINGLE),
            ],
        ),
    }


def main() -> int:
    """执行 C++/Python FDDFS 等价检查。"""

    if not FAST_STACKK_AVAILABLE:
        raise RuntimeError(f"C++ extension unavailable: {FAST_STACKK_IMPORT_ERROR}")
    if not FAST_FDDFS_AVAILABLE:
        raise RuntimeError(f"C++ FDDFS backend unavailable: {FAST_FDDFS_ERROR}")

    graphs_by_name = synthetic_graphs()
    graphs = list(graphs_by_name.values())
    guide = build_frequency_guide(graphs)
    vocab = build_vocab(
        guide.node_counts,
        guide.edge_counts,
        guide.default_edge_feature,
        omit_default_edge=True,
    )
    fast = FastFDDFSEncoder(guide, vocab)

    started = time.time()
    checked = 0
    total_tokens = 0
    max_ring_id = 0
    ring_hist: Counter[str] = Counter()
    for name, graph in graphs_by_name.items():
        py_result = serialize_fddfs(graph, guide, omit_default_edge=True)
        py_ids = [encode_fddfs_token(token, vocab) for token in py_result.sequence]
        cpp_result = fast.serialize_ids(graph)
        cpp_ids = [int(x) for x in cpp_result["ids"]]

        if py_ids != cpp_ids:
            for pos, (left, right) in enumerate(zip(py_ids, cpp_ids)):
                if left != right:
                    break
            else:
                pos = min(len(py_ids), len(cpp_ids))
            raise AssertionError(
                f"FDDFS C++ mismatch case={name} pos={pos} "
                f"py_len={len(py_ids)} cpp_len={len(cpp_ids)} "
                f"py_window={py_ids[max(0, pos - 8):pos + 8]} "
                f"cpp_window={cpp_ids[max(0, pos - 8):pos + 8]}"
            )
        if not verify_roundtrip(graph, py_result):
            raise AssertionError(f"FDDFS Python roundtrip failed: {name}")
        total_tokens += len(py_ids)
        max_ring_id = max(max_ring_id, int(py_result.stats.get("max_ring_id", 0)))
        ring_hist.update(token for token in py_result.sequence if token.startswith("R") and token[1:].isdigit())
        checked += 1

    report = {
        "status": "FDDFS_CPP_EQUIV_OK",
        "checked_records": checked,
        "elapsed_sec": round(time.time() - started, 6),
        "total_tokens": total_tokens,
        "max_ring_id": max_ring_id,
        "ring_hist": dict(sorted(ring_hist.items())),
        "base_vocab_size": len(vocab["token_to_id"]),
        "ring_base": int(vocab["ring_base"]),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
