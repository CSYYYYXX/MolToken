"""Frequency-Double-DFS graph serialization.

FDDFS 是当前 ring-id 版本的频率引导 DFS 序列化：

- 第一次 DFS 只负责确定 spanning forest 和非树边；
- 第二次按这个 DFS 计划输出节点、树边、括号和非树边 ring id；
- 非树边在两个端点都写同一个 ``R<num>``；
- ring id 只在当前未闭合的环里保持占用，闭合后放回空闲池复用；
- 同一个节点上如果同时有开环和闭环，先输出开环，再输出闭环。

这个表示和 SMILES 的 ring number 思路一致，但 DFS 顺序不是化学 canonical
排序，而是由预训练数据里的节点/边/邻接三元组频率决定。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from pipeline.serialization.freq_dfs_core import (
    FrequencyGuide,
    decode_feature,
    deserialize,
    edge_token,
    encode_feature,
    graph_signature,
    is_ring_token,
    make_dfs_plan,
    node_features,
)


def is_fddfs_ring_token(token: str) -> bool:
    """判断 token 是否为 FDDFS 的 ``R<num>`` 环编号。"""

    return is_ring_token(token)


@dataclass
class FDDFSResult:
    """一次 FDDFS 序列化的结果。"""

    sequence: list[str]
    old_to_new: dict[int, int]
    stats: dict[str, int]
    default_edge_feature: tuple[int, ...]


def build_frequency_guide(graphs: list[dict[str, Any]]) -> FrequencyGuide:
    """基于一批 graph 构建 DFS root/neighbor 频率引导器。"""

    return FrequencyGuide(graphs)


def serialize_fddfs(
    graph: dict[str, Any],
    guide: FrequencyGuide,
    *,
    omit_default_edge: bool = True,
) -> FDDFSResult:
    """把一个 heavy atom graph 序列化成 FDDFS token 序列。

    非树边的两个端点会各写一次同号 ``R<num>``。第一次遇到这个 token 时，
    decoder 记录当前节点；第二次遇到时，把当前节点和之前记录的节点连回去。
    编号分配使用最小可用正整数，闭环后释放，因此 ``R1`` 可以在同一条序列
    后面再次出现并表示另一个已经独立闭合的环。
    """

    plan, edges, _adjacency = make_dfs_plan(graph, guide)
    nfeat = node_features(graph)

    ring_events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for edge_id in plan.non_tree_edges:
        edge = edges[edge_id]
        if plan.entry_order[edge.u] <= plan.entry_order[edge.v]:
            opener, closer = edge.u, edge.v
        else:
            opener, closer = edge.v, edge.u
        ring_events[opener].append(("open", edge_id))
        ring_events[closer].append(("close", edge_id))

    ordered_events: dict[int, list[tuple[str, int]]] = {}
    for node, events in ring_events.items():
        # 同一节点先开环后闭环；每一类内部仍按频率引导排序。
        open_events = [item for item in events if item[0] == "open"]
        close_events = [item for item in events if item[0] == "close"]
        sort_key = lambda item: guide.transition_key_from_features(
            nfeat[node],
            edges[item[1]].feat,
            nfeat[edges[item[1]].other(node)],
            edges[item[1]].other(node),
        )
        ordered_events[node] = sorted(open_events, key=sort_key, reverse=True) + sorted(
            close_events,
            key=sort_key,
            reverse=True,
        )

    sequence: list[str] = []
    stats = Counter()
    active: dict[int, int] = {}
    free_ids: list[int] = []
    next_ring_id = 1
    max_active = 0
    max_ring_id = 0

    def allocate_ring_id() -> int:
        """返回最小可用 ring id；没有空闲编号时才增长编号空间。"""

        nonlocal next_ring_id, max_ring_id
        if free_ids:
            rid = min(free_ids)
            free_ids.remove(rid)
        else:
            rid = next_ring_id
            next_ring_id += 1
        max_ring_id = max(max_ring_id, rid)
        return rid

    def release_ring_id(rid: int) -> None:
        """闭环后释放 ring id，允许后续非树边复用。"""

        free_ids.append(int(rid))

    def emit_node(node: int) -> None:
        """输出一个节点及挂在该节点上的非树边 ring 事件。"""

        nonlocal max_active
        sequence.append(encode_feature("N", nfeat[node]))
        stats["node_tokens"] += 1
        for kind, edge_id in ordered_events.get(node, []):
            edge = edges[edge_id]
            if kind == "open":
                rid = allocate_ring_id()
                active[edge_id] = rid
                sequence.extend(edge_token(edge, guide.default_edge_feature, omit_default_edge))
                sequence.append(f"R{rid}")
                max_active = max(max_active, len(active))
                stats["ring_open_tokens"] += 1
                stats["ring_tokens"] += 1
                continue

            if edge_id not in active:
                raise ValueError(f"ring edge closed before open: edge_id={edge_id}")
            rid = active.pop(edge_id)
            sequence.append(f"R{rid}")
            release_ring_id(rid)
            stats["ring_close_tokens"] += 1
            stats["ring_tokens"] += 1

    def emit_subtree(node: int) -> None:
        """按 DFS 计划输出当前子树；主链 child 不加括号。"""

        emit_node(node)
        children = plan.tree_children.get(node, [])
        if not children:
            return
        main_child, main_edge_id = children[-1]
        for child, edge_id in children[:-1]:
            sequence.append("(")
            sequence.extend(edge_token(edges[edge_id], guide.default_edge_feature, omit_default_edge))
            emit_subtree(child)
            sequence.append(")")
            stats["branch_pairs"] += 1
        sequence.extend(edge_token(edges[main_edge_id], guide.default_edge_feature, omit_default_edge))
        emit_subtree(main_child)

    for root_idx, root in enumerate(plan.roots):
        if root_idx:
            sequence.append("<SEP>")
            stats["component_sep"] += 1
        emit_subtree(root)

    if active:
        raise ValueError(f"unclosed ring ids: {active}")

    stats["tokens"] = len(sequence)
    stats["tree_edges"] = sum(len(v) for v in plan.tree_children.values())
    stats["non_tree_edges"] = len(plan.non_tree_edges)
    stats["max_active_ring_ids"] = max_active
    stats["max_ring_id"] = max_ring_id
    return FDDFSResult(
        sequence=sequence,
        old_to_new=plan.old_to_new,
        stats={k: int(v) for k, v in stats.items()},
        default_edge_feature=tuple(int(x) for x in guide.default_edge_feature),
    )


def deserialize_fddfs(sequence: list[str], default_edge: tuple[int, ...]) -> dict[str, Any]:
    """把 FDDFS token 序列反序列化成 heavy atom graph。"""

    return deserialize(sequence, mode="ring", default_edge=default_edge)


def verify_roundtrip(graph: dict[str, Any], result: FDDFSResult) -> bool:
    """检查 FDDFS 序列能否精确恢复当前 heavy atom graph。"""

    decoded = deserialize_fddfs(result.sequence, result.default_edge_feature)
    original_sig = graph_signature(graph, result.old_to_new)
    decoded_sig = graph_signature(decoded)
    return original_sig["nodes"] == decoded_sig["nodes"] and original_sig["edges"] == decoded_sig["edges"]


def build_vocab(
    node_counts: Counter[tuple[int, ...]],
    edge_counts: Counter[tuple[int, ...]],
    default_edge: tuple[int, ...],
    omit_default_edge: bool,
) -> dict[str, Any]:
    """构造 FDDFS 基础 vocab；``R<num>`` 由 ``ring_base + num - 1`` 动态映射。"""

    token_to_id: dict[str, int] = {"(": 0, ")": 1, "<SEP>": 2}
    for feat in sorted(node_counts):
        token_to_id[encode_feature("N", feat)] = len(token_to_id)
    for feat in sorted(edge_counts):
        if omit_default_edge and tuple(feat) == tuple(default_edge):
            continue
        token_to_id[encode_feature("E", feat)] = len(token_to_id)
    return {
        "token_to_id": token_to_id,
        "ring_base": len(token_to_id),
        "ring_min": 1,
        "default_edge_feature": list(int(x) for x in default_edge),
        "omit_default_edge": bool(omit_default_edge),
        "serializer": "FDDFS",
    }


def encode_fddfs_token(token: str, vocab: dict[str, Any]) -> int:
    """把一个 FDDFS string token 转成 int id。"""

    if is_fddfs_ring_token(token):
        ring_id = int(token[1:])
        if ring_id < int(vocab["ring_min"]):
            raise ValueError(f"bad ring token: {token}")
        return int(vocab["ring_base"]) + ring_id - int(vocab["ring_min"])
    try:
        return int(vocab["token_to_id"][token])
    except KeyError as exc:
        raise KeyError(f"token missing from FDDFS vocab: {token}") from exc
