"""stackK / FDDFS C++ 热区包装。

Python 继续负责 pickle/zstd I/O 和 multiprocessing；这个模块只把单个 graph 的
DFS 规划、serializer emit、token id 映射交给 C++，以便和 Python 版本做逐样本
等价校验。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from pipeline.serialization.freq_dfs_core import decode_feature

try:
    from pipeline.serialization import _stackk_cpp

    FAST_STACKK_AVAILABLE = True
    FAST_STACKK_IMPORT_ERROR: Exception | None = None
    FAST_STACKK_COUNT_AVAILABLE = hasattr(_stackk_cpp, "FastStackKCounter")
    FAST_STACKK_COUNT_ERROR: Exception | None = None
    FAST_FDDFS_AVAILABLE = hasattr(_stackk_cpp.FastStackKGuide, "serialize_fddfs")
    FAST_FDDFS_ERROR: Exception | None = None
    if not FAST_STACKK_COUNT_AVAILABLE:
        FAST_STACKK_COUNT_ERROR = RuntimeError("pipeline.serialization._stackk_cpp has no FastStackKCounter")
    if not FAST_FDDFS_AVAILABLE:
        FAST_FDDFS_ERROR = RuntimeError("pipeline.serialization._stackk_cpp.FastStackKGuide has no serialize_fddfs")
except Exception as exc:  # noqa: BLE001
    _stackk_cpp = None
    FAST_STACKK_AVAILABLE = False
    FAST_STACKK_IMPORT_ERROR = exc
    FAST_STACKK_COUNT_AVAILABLE = False
    FAST_STACKK_COUNT_ERROR = exc
    FAST_FDDFS_AVAILABLE = False
    FAST_FDDFS_ERROR = exc


def _items(counter: Counter[Any]) -> list[tuple[Any, int]]:
    """把 Counter 变成 C++ 构造器接受的稳定 list。"""

    return [(key, int(value)) for key, value in counter.items()]


def _triplet_items(counter: Counter[Any]) -> list[tuple[Any, Any, Any, int]]:
    """把 triplet Counter 展成 ``(src_feat, edge_feat, dst_feat, count)``。"""

    out: list[tuple[Any, Any, Any, int]] = []
    for key, value in counter.items():
        src_feat, edge_feat, dst_feat = key
        out.append((src_feat, edge_feat, dst_feat, int(value)))
    return out


def _vocab_feature_items(vocab: dict[str, Any], prefix: str) -> list[tuple[tuple[int, ...], int]]:
    """从 string vocab 抽取 node/edge feature 到 id 的映射。"""

    head = f"{prefix}:"
    items: list[tuple[tuple[int, ...], int]] = []
    token_to_id = vocab["token_to_id"]
    for token, token_id in token_to_id.items():
        if token.startswith(head):
            items.append((decode_feature(token, prefix), int(token_id)))
    return items


def _tuple_counter(items: list[tuple[Any, int]]) -> Counter[tuple[int, ...]]:
    """把 C++ 返回的 ``[(tuple, count), ...]`` 转成 Python Counter。"""

    return Counter({tuple(int(x) for x in key): int(value) for key, value in items})


def _triplet_counter(items: list[tuple[Any, Any, Any, int]]) -> Counter[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """把 C++ 返回的 triplet 列表转成 Python Counter。"""

    out: Counter[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = Counter()
    for src_feat, edge_feat, dst_feat, value in items:
        out[
            (
                tuple(int(x) for x in src_feat),
                tuple(int(x) for x in edge_feat),
                tuple(int(x) for x in dst_feat),
            )
        ] = int(value)
    return out


class FastStackKEncoder:
    """C++ stackK 编码器。

    构造阶段把 guide/vocab 转成 C++ 哈希表；每个 worker 初始化一次，之后每个
    graph 只传 numpy 数组，避免在热循环里反复处理 Python string token。
    """

    def __init__(self, guide: Any, vocab: dict[str, Any]) -> None:
        if not FAST_STACKK_AVAILABLE:
            raise ImportError("pipeline.serialization._stackk_cpp is not available") from FAST_STACKK_IMPORT_ERROR
        token_to_id = vocab["token_to_id"]
        self._encoder = _stackk_cpp.FastStackKGuide(
            _items(Counter(guide.node_counts)),
            _items(Counter(guide.edge_counts)),
            _triplet_items(Counter(guide.triplet_counts)),
            tuple(int(x) for x in guide.default_edge_feature),
            _vocab_feature_items(vocab, "N"),
            _vocab_feature_items(vocab, "E"),
            int(token_to_id["("]),
            int(token_to_id[")"]),
            int(token_to_id["<SEP>"]),
            int(vocab["k_base"]),
            int(vocab["k_min"]),
            bool(vocab["omit_default_edge"]),
        )

    def serialize_ids(self, graph: dict[str, Any]) -> dict[str, Any]:
        """把单个 graph 序列化成 int token id 序列。"""

        src_raw, dst_raw = graph["edges"]
        src = np.ascontiguousarray(src_raw, dtype=np.int64).reshape((-1,))
        dst = np.ascontiguousarray(dst_raw, dtype=np.int64).reshape((-1,))
        node_features = np.ascontiguousarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
        edge_features = np.ascontiguousarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
        return self._encoder.serialize(
            int(graph["num_nodes"]),
            src,
            dst,
            node_features,
            edge_features,
        )


class FastFDDFSEncoder:
    """C++ FDDFS 编码器。

    这里复用 stackK C++ 扩展里的无向边构建、频率引导 DFS 计划和树遍历代码；
    只把非树边 emit 从 ``K<num>`` 换成两端同号的 ``R<num>``，并在闭环后复用
    ring id。
    """

    def __init__(self, guide: Any, vocab: dict[str, Any]) -> None:
        if not FAST_STACKK_AVAILABLE:
            raise ImportError("pipeline.serialization._stackk_cpp is not available") from FAST_STACKK_IMPORT_ERROR
        if not FAST_FDDFS_AVAILABLE:
            raise ImportError("pipeline.serialization._stackk_cpp Fast FDDFS is not available") from FAST_FDDFS_ERROR
        token_to_id = vocab["token_to_id"]
        self.ring_base = int(vocab["ring_base"])
        self.ring_min = int(vocab.get("ring_min", 1))
        # C++ 构造器历史上按 stackK 命名，需要 k_base/k_min。FDDFS 不使用这两个
        # 字段，传 ring 参数作为无害占位，真正映射在 serialize_fddfs() 参数里。
        self._encoder = _stackk_cpp.FastStackKGuide(
            _items(Counter(guide.node_counts)),
            _items(Counter(guide.edge_counts)),
            _triplet_items(Counter(guide.triplet_counts)),
            tuple(int(x) for x in guide.default_edge_feature),
            _vocab_feature_items(vocab, "N"),
            _vocab_feature_items(vocab, "E"),
            int(token_to_id["("]),
            int(token_to_id[")"]),
            int(token_to_id["<SEP>"]),
            self.ring_base,
            self.ring_min,
            bool(vocab["omit_default_edge"]),
        )

    def serialize_ids(self, graph: dict[str, Any]) -> dict[str, Any]:
        """把单个 graph 序列化成 FDDFS int token id 序列。"""

        src_raw, dst_raw = graph["edges"]
        src = np.ascontiguousarray(src_raw, dtype=np.int64).reshape((-1,))
        dst = np.ascontiguousarray(dst_raw, dtype=np.int64).reshape((-1,))
        node_features = np.ascontiguousarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
        edge_features = np.ascontiguousarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
        return self._encoder.serialize_fddfs(
            int(graph["num_nodes"]),
            src,
            dst,
            node_features,
            edge_features,
            self.ring_base,
            self.ring_min,
        )


class FastStackKCounter:
    """C++ DFS guide 频率统计器。

    每个 count worker 持有一个实例，流式加入当前分片内的 graph。C++ 负责扫
    node/edge 数组和统计 node、edge、双向 triplet 频率；Python 只在分片结尾
    把结果转回 Counter，供后续合并和构造 guide 使用。
    """

    def __init__(self) -> None:
        if not FAST_STACKK_COUNT_AVAILABLE:
            raise ImportError("pipeline.serialization._stackk_cpp.FastStackKCounter is not available") from FAST_STACKK_COUNT_ERROR
        self._counter = _stackk_cpp.FastStackKCounter()

    def add_graph(self, graph: dict[str, Any]) -> None:
        """把单个 graph 加入频率统计。"""

        src_raw, dst_raw = graph["edges"]
        src = np.ascontiguousarray(src_raw, dtype=np.int64).reshape((-1,))
        dst = np.ascontiguousarray(dst_raw, dtype=np.int64).reshape((-1,))
        node_features = np.ascontiguousarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
        edge_features = np.ascontiguousarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
        self._counter.add_graph(
            int(graph["num_nodes"]),
            src,
            dst,
            node_features,
            edge_features,
        )

    def result(self) -> dict[str, Any]:
        """返回和 Python count 阶段同形状的 Counter/stat payload。"""

        raw = self._counter.result()
        return {
            "node_counts": _tuple_counter(raw["node_counts"]),
            "edge_counts": _tuple_counter(raw["edge_counts"]),
            "triplet_counts": _triplet_counter(raw["triplet_counts"]),
            "stats": {str(key): int(value) for key, value in raw["stats"].items()},
        }
