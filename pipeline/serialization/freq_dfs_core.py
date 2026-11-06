#!/usr/bin/env python3
"""频率引导 DFS 可逆序列化的共享核心。

这个脚本用于验证一个想法：不使用 RDKit/SMILES 的化学 canonical 排序，
而是先在样本图上统计局部频率，再用这些频率决定 DFS 的 root、邻居顺序、
主链 child 和支链 child。非树边同步实现两种表示：

1. ring id：类似 SMILES 的环编号，在较早出现的端点打开，在较晚出现的
   端点关闭。
2. @k：只在较晚出现的端点写一个回指，k 表示当前原子往前数第 k 个原子
   occurrence。

节点和边 token 保留完整 heavy atom feature tuple，因此可以反序列化回当前
heavy atom graph。节点 token 的可读形式会把 RDKit chiral tag 显示成
SMILES 风格的 @ / @@。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.graph.io import DEFAULT_DATA_ROOT, DEFAULT_GRAPH_ROOT, find_zstd, graph_signature, iter_pickle_zst


DEFAULT_OUT = DEFAULT_DATA_ROOT / "reports/fddfs_diagnostics/freq_dfs_roundtrip.json"
MOST_FREQUENT_EDGE_TYPE = 1


ATOM_SYMBOL = {
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

BOND_SYMBOL = {
    0: "~",
    1: "-",
    2: "=",
    3: "#",
    4: ":",
}


def feature_tuple(row: Any) -> tuple[int, ...]:
    """把 numpy/list feature 行规范成 int tuple。"""
    return tuple(int(x) for x in np.asarray(row, dtype=np.int64).reshape((-1,)).tolist())


def node_features(graph: dict[str, Any]) -> list[tuple[int, ...]]:
    """返回当前图的完整节点 feature tuple。"""
    return [feature_tuple(row) for row in np.asarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))]


def edge_features(graph: dict[str, Any]) -> list[tuple[int, ...]]:
    """返回当前图的完整边 feature tuple。"""
    return [feature_tuple(row) for row in np.asarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))]


def chiral_suffix(chiral_tag: int) -> str:
    """把 RDKit chiral tag 显示成 SMILES 风格的手性标记。"""
    if int(chiral_tag) == 1:
        return "@@"
    if int(chiral_tag) == 2:
        return "@"
    if int(chiral_tag) > 2:
        return f"@{int(chiral_tag)}"
    return ""


def compact_node_label(feat: tuple[int, ...]) -> str:
    """生成节点的紧凑可读标签；完整 feature 仍写在 token 中。"""
    atomic_num = int(feat[0])
    charge = int(feat[1]) if len(feat) > 1 else 0
    aromatic = int(feat[2]) if len(feat) > 2 else 0
    chiral = int(feat[3]) if len(feat) > 3 else 0
    symbol = ATOM_SYMBOL.get(atomic_num, f"X{atomic_num}")
    if aromatic and symbol in {"B", "C", "N", "O", "P", "S"}:
        symbol = symbol.lower()
    if charge > 0:
        symbol += "+" if charge == 1 else f"+{charge}"
    elif charge < 0:
        symbol += "-" if charge == -1 else str(charge)
    symbol += chiral_suffix(chiral)
    return symbol


def compact_edge_label(feat: tuple[int, ...]) -> str:
    """生成边的紧凑可读标签；完整 feature 仍写在 token 中。"""
    stereo = int(feat[2]) if len(feat) > 2 else 0
    base = BOND_SYMBOL.get(int(feat[0]), f"E{int(feat[0])}")
    if stereo:
        base += f"/{stereo}"
    return base


def encode_feature(prefix: str, feat: tuple[int, ...]) -> str:
    """把 feature tuple 编成一个可反解析的 token。"""
    return f"{prefix}:{','.join(str(int(x)) for x in feat)}"


def decode_feature(token: str, prefix: str) -> tuple[int, ...]:
    """从 token 解析 feature tuple。"""
    head = f"{prefix}:"
    if not token.startswith(head):
        raise ValueError(f"expected {head} token, got {token}")
    return tuple(int(x) for x in token[len(head) :].split(",") if x != "")


def is_node_token(token: str) -> bool:
    return token.startswith("N:")


def is_edge_token(token: str) -> bool:
    return token.startswith("E:")


def is_ring_token(token: str) -> bool:
    return token.startswith("R") and token[1:].isdigit()


def is_atk_token(token: str) -> bool:
    return token.startswith("@") and token[1:].isdigit()


@dataclass(frozen=True)
class UndirectedEdge:
    """无向边视角的一条 heavy atom bond。"""

    edge_id: int
    u: int
    v: int
    feat: tuple[int, ...]

    def other(self, node: int) -> int:
        if node == self.u:
            return self.v
        if node == self.v:
            return self.u
        raise ValueError(f"node={node} is not incident to edge={self.edge_id}")


class FrequencyGuide:
    """局部频率表，用于决定 DFS root 和分支顺序。"""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.node_counts: Counter[tuple[int, ...]] = Counter()
        self.edge_counts: Counter[tuple[int, ...]] = Counter()
        self.triplet_counts: Counter[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = Counter()
        for graph in records:
            nfeat = node_features(graph)
            for feat in nfeat:
                self.node_counts[feat] += 1
            for edge in build_undirected_edges(graph)[0]:
                self.edge_counts[edge.feat] += 1
                a = nfeat[edge.u]
                b = nfeat[edge.v]
                self.triplet_counts[(a, edge.feat, b)] += 1
                self.triplet_counts[(b, edge.feat, a)] += 1
        self.default_edge_feature = (
            self.edge_counts.most_common(1)[0][0] if self.edge_counts else (MOST_FREQUENT_EDGE_TYPE, 0, 0)
        )

    def root_key(self, graph: dict[str, Any], node: int, degree: int) -> tuple[int, int, str, int]:
        """root 选择：高频节点优先，高度数优先，最后用稳定文本和原 index 打破并列。"""
        feat = node_features(graph)[int(node)]
        return (
            int(self.node_counts.get(feat, 0)),
            int(degree),
            compact_node_label(feat),
            -int(node),
        )

    def transition_key(self, graph: dict[str, Any], src: int, edge: UndirectedEdge, dst: int) -> tuple[int, int, int, str, int]:
        """邻居选择：高频 atom-edge-atom 优先，然后看 edge/node 频率。"""
        nfeat = node_features(graph)
        sfeat = nfeat[int(src)]
        dfeat = nfeat[int(dst)]
        return self.transition_key_from_features(sfeat, edge.feat, dfeat, dst)

    def transition_key_from_features(
        self,
        src_feat: tuple[int, ...],
        edge_feat: tuple[int, ...],
        dst_feat: tuple[int, ...],
        dst: int,
    ) -> tuple[int, int, int, str, int]:
        """邻居选择的低开销版本，调用方已缓存当前图的 feature。"""
        return (
            int(self.triplet_counts.get((src_feat, edge_feat, dst_feat), 0)),
            int(self.edge_counts.get(edge_feat, 0)),
            int(self.node_counts.get(dst_feat, 0)),
            compact_node_label(dst_feat),
            -int(dst),
        )


def build_undirected_edges(graph: dict[str, Any]) -> tuple[list[UndirectedEdge], dict[int, list[tuple[int, int]]]]:
    """把双向 directed edge 压成无向边，并保留邻接表。

    当前 heavy atom graph 的常规格式是每条化学键存两条方向相反的边。这个
    原型按无向键序列化；如果同一无向边的两个方向 feature 不一致，直接报错，
    避免悄悄丢失信息。
    """
    src_raw, dst_raw = graph["edges"]
    efeat = edge_features(graph)
    by_pair: dict[tuple[int, int], tuple[int, ...]] = {}
    first_idx: dict[tuple[int, int], int] = {}
    for edge_idx, (u_raw, v_raw) in enumerate(zip(src_raw, dst_raw)):
        u = int(u_raw)
        v = int(v_raw)
        if u == v:
            raise ValueError("self-loop is not supported by this prototype")
        key = (u, v) if u < v else (v, u)
        feat = efeat[edge_idx]
        if key in by_pair and by_pair[key] != feat:
            raise ValueError(f"asymmetric directed edge feature for pair={key}: {by_pair[key]} vs {feat}")
        by_pair[key] = feat
        first_idx.setdefault(key, edge_idx)

    edges: list[UndirectedEdge] = []
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id, key in enumerate(sorted(by_pair)):
        u, v = key
        edge = UndirectedEdge(edge_id=edge_id, u=u, v=v, feat=by_pair[key])
        edges.append(edge)
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
    return edges, adjacency


def connected_components_from_adjacency(num_nodes: int, adjacency: dict[int, list[tuple[int, int]]]) -> list[list[int]]:
    """根据无向邻接表计算连通分量。"""
    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in range(int(num_nodes)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr, _edge_id in adjacency.get(node, []):
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        comps.append(sorted(comp))
    return comps


@dataclass
class DfsPlan:
    """第一次 DFS 得到的树边和非树边计划。"""

    roots: list[int]
    tree_children: dict[int, list[tuple[int, int]]]
    non_tree_edges: list[int]
    entry_order: dict[int, int]
    old_to_new: dict[int, int]


def make_dfs_plan(graph: dict[str, Any], guide: FrequencyGuide) -> tuple[DfsPlan, list[UndirectedEdge], dict[int, list[tuple[int, int]]]]:
    """按频率引导做第一次 DFS，确定 spanning forest 和非树边。"""
    num_nodes = int(graph["num_nodes"])
    nfeat = node_features(graph)
    edges, adjacency = build_undirected_edges(graph)
    comps = connected_components_from_adjacency(num_nodes, adjacency)

    degree = {node: len(adjacency.get(node, [])) for node in range(num_nodes)}
    roots: list[int] = []
    colors = [0] * num_nodes
    entry_order: dict[int, int] = {}
    tree_children: dict[int, list[tuple[int, int]]] = defaultdict(list)
    tree_edge_ids: set[int] = set()
    non_tree_edge_ids: set[int] = set()
    old_to_new: dict[int, int] = {}

    def sorted_neighbors(node: int) -> list[tuple[int, int]]:
        items = adjacency.get(node, [])
        return sorted(
            items,
            key=lambda item: guide.transition_key_from_features(
                nfeat[node],
                edges[item[1]].feat,
                nfeat[item[0]],
                item[0],
            ),
            reverse=True,
        )

    def dfs(node: int, parent_edge: int | None) -> None:
        colors[node] = 1
        entry_order[node] = len(entry_order)
        old_to_new[node] = len(old_to_new)
        # 最高频的 child 作为主链，SMILES 风格下主链 child 放在最后、
        # 不加括号；因此第一次 DFS 也按“分支先、主链后”的输出顺序进入，
        # 让 entry_order / old_to_new 与最终序列里的节点出现顺序一致。
        neighbors = sorted_neighbors(node)
        if len(neighbors) > 1:
            neighbors = neighbors[1:] + neighbors[:1]
        for nbr, edge_id in neighbors:
            if parent_edge is not None and edge_id == parent_edge:
                continue
            if colors[nbr] == 0:
                tree_edge_ids.add(edge_id)
                tree_children[node].append((nbr, edge_id))
                dfs(nbr, edge_id)
            elif edge_id not in tree_edge_ids:
                non_tree_edge_ids.add(edge_id)
        colors[node] = 2

    comp_roots = []
    for comp in comps:
        root = max(comp, key=lambda node: guide.root_key(graph, node, degree[node]))
        comp_roots.append(root)
    comp_roots.sort(key=lambda node: guide.root_key(graph, node, degree[node]), reverse=True)

    for root in comp_roots:
        if colors[root] == 0:
            roots.append(root)
            dfs(root, None)
    for node in range(num_nodes):
        if colors[node] == 0:
            roots.append(node)
            dfs(node, None)

    non_tree_edges = sorted(non_tree_edge_ids, key=lambda edge_id: (entry_order[edges[edge_id].u], entry_order[edges[edge_id].v], edge_id))
    return DfsPlan(roots, dict(tree_children), non_tree_edges, entry_order, old_to_new), edges, adjacency


def edge_token(edge: UndirectedEdge, default_edge: tuple[int, ...], omit_default_edge: bool) -> list[str]:
    """按配置返回边 token；最常见边可以省略。"""
    if omit_default_edge and edge.feat == default_edge:
        return []
    return [encode_feature("E", edge.feat)]


def serialize_atk(graph: dict[str, Any], guide: FrequencyGuide, omit_default_edge: bool) -> tuple[list[str], dict[str, Any]]:
    """用 @k 表示非树边的频率引导 DFS 序列。"""
    plan, edges, _adjacency = make_dfs_plan(graph, guide)
    nfeat = node_features(graph)
    close_events: dict[int, list[int]] = defaultdict(list)
    for edge_id in plan.non_tree_edges:
        edge = edges[edge_id]
        later = edge.u if plan.entry_order[edge.u] > plan.entry_order[edge.v] else edge.v
        close_events[later].append(edge_id)
    for node in close_events:
        close_events[node].sort(
            key=lambda eid: guide.transition_key_from_features(
                nfeat[node],
                edges[eid].feat,
                nfeat[edges[eid].other(node)],
                edges[eid].other(node),
            ),
            reverse=True,
        )

    sequence: list[str] = []
    atom_occurrence: list[int] = []
    stats = Counter()
    sum_atk = 0
    max_atk = 0

    def emit_node(node: int) -> None:
        nonlocal sum_atk, max_atk
        sequence.append(encode_feature("N", nfeat[node]))
        atom_occurrence.append(node)
        stats["node_tokens"] += 1
        current_occ = len(atom_occurrence) - 1
        for edge_id in close_events.get(node, []):
            edge = edges[edge_id]
            target = edge.other(node)
            if target not in plan.old_to_new:
                raise ValueError(f"target node was not visited before: edge={edge_id}")
            try:
                target_occ = atom_occurrence.index(target)
            except ValueError as exc:
                raise ValueError(f"@k target not in previous atom occurrences: target={target}") from exc
            distance = current_occ - target_occ
            if distance <= 0:
                raise ValueError(f"bad @k distance={distance} for edge={edge_id}")
            sequence.extend(edge_token(edge, guide.default_edge_feature, omit_default_edge))
            sequence.append(f"@{distance}")
            stats["non_tree_edges"] += 1
            stats["atk_tokens"] += 1
            sum_atk += distance
            max_atk = max(max_atk, distance)

    def emit_subtree(node: int) -> None:
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

    stats["tokens"] = len(sequence)
    stats["tree_edges"] = sum(len(v) for v in plan.tree_children.values())
    stats["sum_atk_distance"] = sum_atk
    stats["max_atk_distance"] = max_atk
    stats["mean_atk_distance_x1000"] = int(round((sum_atk / stats["atk_tokens"]) * 1000)) if stats["atk_tokens"] else 0
    return sequence, {"old_to_new": plan.old_to_new, "stats": dict(stats)}


def serialize_ring(graph: dict[str, Any], guide: FrequencyGuide, omit_default_edge: bool) -> tuple[list[str], dict[str, Any]]:
    """用 ring id 表示非树边的频率引导 DFS 序列。"""
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
    for node in ring_events:
        ring_events[node].sort(
            key=lambda item: (
                0 if item[0] == "close" else 1,
                guide.transition_key_from_features(
                    nfeat[node],
                    edges[item[1]].feat,
                    nfeat[edges[item[1]].other(node)],
                    edges[item[1]].other(node),
                ),
            ),
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
        nonlocal next_ring_id, max_ring_id
        if free_ids:
            rid = min(free_ids)
            free_ids.remove(rid)
        else:
            rid = next_ring_id
            next_ring_id += 1
        max_ring_id = max(max_ring_id, rid)
        return rid

    def emit_node(node: int) -> None:
        nonlocal max_active
        sequence.append(encode_feature("N", nfeat[node]))
        stats["node_tokens"] += 1
        for kind, edge_id in ring_events.get(node, []):
            edge = edges[edge_id]
            if kind == "open":
                rid = allocate_ring_id()
                active[edge_id] = rid
                sequence.extend(edge_token(edge, guide.default_edge_feature, omit_default_edge))
                sequence.append(f"R{rid}")
                max_active = max(max_active, len(active))
                stats["ring_open_tokens"] += 1
            else:
                if edge_id not in active:
                    raise ValueError(f"ring edge closed before open: edge={edge_id}")
                rid = active.pop(edge_id)
                sequence.append(f"R{rid}")
                free_ids.append(rid)
                stats["ring_close_tokens"] += 1

    def emit_subtree(node: int) -> None:
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
    return sequence, {"old_to_new": plan.old_to_new, "stats": dict(stats)}


def add_undirected_edge(
    src: list[int],
    dst: list[int],
    efeats: list[tuple[int, ...]],
    u: int,
    v: int,
    feat: tuple[int, ...],
) -> None:
    """反序列化时把一条无向键恢复成两条 directed edge。"""
    src.extend([u, v])
    dst.extend([v, u])
    efeats.extend([feat, feat])


class SequenceParser:
    """解析 ring id 或 @k 序列，并恢复 heavy atom graph。"""

    def __init__(self, sequence: list[str], mode: str, default_edge: tuple[int, ...]) -> None:
        self.sequence = sequence
        self.mode = mode
        self.default_edge = default_edge
        self.pos = 0
        self.node_feats: list[tuple[int, ...]] = []
        self.src: list[int] = []
        self.dst: list[int] = []
        self.edge_feats: list[tuple[int, ...]] = []
        self.atom_occurrence: list[int] = []
        self.open_rings: dict[str, tuple[int, tuple[int, ...]]] = {}

    def parse(self) -> dict[str, Any]:
        while self.pos < len(self.sequence):
            if self.sequence[self.pos] == "<SEP>":
                self.pos += 1
                continue
            self._parse_tree(parent=None, incoming_edge=None)
        if self.open_rings:
            raise ValueError(f"unclosed ring markers: {sorted(self.open_rings)}")
        node_arr = np.asarray(self.node_feats, dtype=np.int64).reshape((-1, 6))
        edge_arr = np.asarray(self.edge_feats, dtype=np.int64).reshape((-1, 3))
        return {
            "num_nodes": len(self.node_feats),
            "edges": (self.src, self.dst),
            "node_features": node_arr,
            "edge_features": edge_arr,
            "node_type_ids": node_arr[:, 0] if len(node_arr) else np.zeros((0,), dtype=np.int64),
            "edge_type_ids": edge_arr[:, 0] if len(edge_arr) else np.zeros((0,), dtype=np.int64),
        }

    def _read_optional_edge(self) -> tuple[int, ...]:
        if self.pos < len(self.sequence) and is_edge_token(self.sequence[self.pos]):
            feat = decode_feature(self.sequence[self.pos], "E")
            self.pos += 1
            return feat
        return self.default_edge

    def _read_node(
        self,
        parent: int | None,
        incoming_edge: tuple[int, ...] | None,
    ) -> int:
        """Consume one node and attach it to ``parent`` when present."""

        if self.pos >= len(self.sequence) or not is_node_token(self.sequence[self.pos]):
            got = self.sequence[self.pos] if self.pos < len(self.sequence) else "<END>"
            raise ValueError(f"expected node token at pos={self.pos}, got={got}")
        node_id = len(self.node_feats)
        self.node_feats.append(decode_feature(self.sequence[self.pos], "N"))
        self.atom_occurrence.append(node_id)
        self.pos += 1
        if parent is not None:
            add_undirected_edge(
                self.src,
                self.dst,
                self.edge_feats,
                parent,
                node_id,
                incoming_edge or self.default_edge,
            )
        return node_id

    def _parse_tree(self, parent: int | None, incoming_edge: tuple[int, ...] | None) -> int:
        node_id = self._read_node(parent, incoming_edge)
        # Only a completed parenthesized branch needs a return continuation.
        # Main-chain recursion in the old parser was a tail call, so advancing
        # ``node_id`` is equivalent while avoiding one Python/C stack frame per
        # atom in long proteins and nucleic acids.
        branch_parents: list[int] = []

        while True:
            if self.pos >= len(self.sequence):
                if branch_parents:
                    raise ValueError(f"missing branch close at pos={self.pos}")
                return node_id
            token = self.sequence[self.pos]
            if token in {")", "<SEP>"}:
                if branch_parents:
                    if token != ")":
                        raise ValueError(f"missing branch close at pos={self.pos}")
                    self.pos += 1
                    node_id = branch_parents.pop()
                    continue
                return node_id
            if token == "(":
                self.pos += 1
                branch_edge = self._read_optional_edge()
                branch_parents.append(node_id)
                node_id = self._read_node(parent=node_id, incoming_edge=branch_edge)
                continue
            if self.mode == "ring" and is_ring_token(token):
                self._handle_ring(node_id, self.default_edge)
                continue
            if self.mode == "atk" and is_atk_token(token):
                self._handle_atk(node_id, self.default_edge)
                continue
            if is_edge_token(token):
                edge = self._read_optional_edge()
                if self.pos < len(self.sequence) and self.mode == "ring" and is_ring_token(self.sequence[self.pos]):
                    self._handle_ring(node_id, edge)
                    continue
                if self.pos < len(self.sequence) and self.mode == "atk" and is_atk_token(self.sequence[self.pos]):
                    self._handle_atk(node_id, edge)
                    continue
                node_id = self._read_node(parent=node_id, incoming_edge=edge)
                continue
            if is_node_token(token):
                node_id = self._read_node(parent=node_id, incoming_edge=self.default_edge)
                continue
            raise ValueError(f"unexpected token at pos={self.pos}: {token}")

    def _handle_ring(self, current_node: int, edge: tuple[int, ...]) -> None:
        token = self.sequence[self.pos]
        if token in self.open_rings:
            other_node, stored_edge = self.open_rings.pop(token)
            add_undirected_edge(self.src, self.dst, self.edge_feats, other_node, current_node, stored_edge)
        else:
            self.open_rings[token] = (current_node, edge)
        self.pos += 1

    def _handle_atk(self, current_node: int, edge: tuple[int, ...]) -> None:
        token = self.sequence[self.pos]
        distance = int(token[1:])
        current_occ = len(self.atom_occurrence) - 1
        target_occ = current_occ - distance
        if target_occ < 0:
            raise ValueError(f"bad @k token {token} at atom occurrence {current_occ}")
        target_node = self.atom_occurrence[target_occ]
        add_undirected_edge(self.src, self.dst, self.edge_feats, target_node, current_node, edge)
        self.pos += 1


def deserialize(sequence: list[str], mode: str, default_edge: tuple[int, ...]) -> dict[str, Any]:
    """按指定模式反序列化 token 序列。"""
    return SequenceParser(sequence, mode, default_edge).parse()


def readable_sequence(sequence: list[str], limit: int) -> list[str]:
    """把完整 feature token 压成便于人工检查的短 token。"""
    out: list[str] = []
    for token in sequence[:limit]:
        if is_node_token(token):
            out.append(compact_node_label(decode_feature(token, "N")))
        elif is_edge_token(token):
            out.append(compact_edge_label(decode_feature(token, "E")))
        else:
            out.append(token)
    return out


def pick_records(
    graph_root: Path,
    domains: list[str],
    split: str,
    max_files_per_domain: int,
    max_records_per_domain: int,
    max_nodes_per_record: int,
) -> list[tuple[str, Path, int, dict[str, Any]]]:
    """从每个 domain 取少量 graph records 做原型实验。"""
    zstd = find_zstd()
    records: list[tuple[str, Path, int, dict[str, Any]]] = []
    for domain in domains:
        root = graph_root / domain / split
        files = sorted(root.glob("*.pkl.zst"))
        if not files:
            print(f"skip missing domain/split: {root}", file=sys.stderr)
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


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    """返回简单长度统计。"""
    if not values:
        return {"count": 0, "mean": 0.0, "max": 0}
    arr = sorted(int(x) for x in values)
    return {
        "count": len(arr),
        "mean": sum(arr) / len(arr),
        "min": arr[0],
        "p50": arr[len(arr) // 2],
        "p95": arr[min(len(arr) - 1, int(len(arr) * 0.95))],
        "max": arr[-1],
    }


def summarize_mode(counter: Counter[str], lengths: list[int], vocab: Counter[str]) -> dict[str, Any]:
    """汇总单个序列化模式的计数和派生指标。"""
    records = int(counter.get("records", 0))
    input_nodes = int(counter.get("input_nodes", 0))
    input_directed_edges = int(counter.get("input_directed_edges", 0))
    output_tokens = int(counter.get("output_tokens", 0))
    atk_tokens = int(counter.get("atk_tokens", 0))
    sum_atk = int(counter.get("sum_atk_distance", 0))
    return {
        "counts": dict(counter),
        "lengths": summarize_lengths(lengths),
        "vocab_size": len(vocab),
        "top_tokens": vocab.most_common(20),
        "derived": {
            "mean_tokens_per_record": (output_tokens / records) if records else 0.0,
            "mean_tokens_per_input_node": (output_tokens / input_nodes) if input_nodes else 0.0,
            "mean_tokens_per_input_directed_edge": (output_tokens / input_directed_edges) if input_directed_edges else 0.0,
            "mean_atk_distance": (sum_atk / atk_tokens) if atk_tokens else 0.0,
        },
    }


def main() -> int:
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    ap = argparse.ArgumentParser(description="Prototype frequency-guided DFS graph serialization.")
    ap.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--domains", default="small_molecule,peptide,protein,dna,rna")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-files-per-domain", type=int, default=1)
    ap.add_argument("--max-records-per-domain", type=int, default=20)
    ap.add_argument(
        "--max-nodes-per-record",
        type=int,
        default=5000,
        help="原型实验默认跳过过大的图；<=0 表示不过滤。",
    )
    ap.add_argument("--preview-tokens", type=int, default=160)
    ap.add_argument("--keep-default-edge-feature", action="store_true")
    args = ap.parse_args()

    start = time.time()
    domains = [x.strip() for x in str(args.domains).split(",") if x.strip()]
    rows = pick_records(
        args.graph_root,
        domains,
        args.split,
        args.max_files_per_domain,
        args.max_records_per_domain,
        int(args.max_nodes_per_record),
    )
    if not rows:
        raise RuntimeError("no records picked")
    guide = FrequencyGuide([graph for _domain, _path, _idx, graph in rows])
    omit_default_edge = not bool(args.keep_default_edge_feature)

    mode_counters: dict[str, Counter[str]] = {"ring": Counter(), "atk": Counter()}
    mode_lengths: dict[str, list[int]] = {"ring": [], "atk": []}
    mode_vocab: dict[str, Counter[str]] = {"ring": Counter(), "atk": Counter()}
    domain_summary: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"ring": Counter(), "atk": Counter()})
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    processed_pairs = 0
    for row_idx, (domain, graph_file, record_idx, graph) in enumerate(rows, 1):
        for mode, fn in (("ring", serialize_ring), ("atk", serialize_atk)):
            processed_pairs += 1
            try:
                seq, info = fn(graph, guide, omit_default_edge)
                decoded = deserialize(seq, mode, guide.default_edge_feature)
                original_sig = graph_signature(graph, info["old_to_new"])
                decoded_sig = graph_signature(decoded)
                ok = original_sig["nodes"] == decoded_sig["nodes"] and original_sig["edges"] == decoded_sig["edges"]
                counters = mode_counters[mode]
                counters["records"] += 1
                counters["fully_reversible"] += int(ok)
                counters["input_nodes"] += int(graph["num_nodes"])
                counters["input_directed_edges"] += int(len(graph["edges"][0]))
                counters["output_tokens"] += len(seq)
                for key, value in info["stats"].items():
                    if key.startswith("max_"):
                        counters[key] = max(int(counters.get(key, 0)), int(value))
                    elif key.startswith("mean_"):
                        continue
                    else:
                        counters[key] += int(value)
                mode_lengths[mode].append(len(seq))
                mode_vocab[mode].update(seq)
                ds = domain_summary[domain][mode]
                ds["records"] += 1
                ds["fully_reversible"] += int(ok)
                ds["output_tokens"] += len(seq)
                ds["input_nodes"] += int(graph["num_nodes"])
                ds["input_directed_edges"] += int(len(graph["edges"][0]))
                if not ok:
                    failures.append(
                        {
                            "mode": mode,
                            "domain": domain,
                            "graph_file": str(graph_file),
                            "record_index": record_idx,
                            "error": "signature_mismatch",
                        }
                    )
                if len(examples) < 10:
                    examples.append(
                        {
                            "mode": mode,
                            "domain": domain,
                            "graph_file": str(graph_file),
                            "record_index": int(record_idx),
                            "sample_id": graph.get("sample_id"),
                            "input_nodes": int(graph["num_nodes"]),
                            "input_directed_edges": int(len(graph["edges"][0])),
                            "output_tokens": len(seq),
                            "stats": info["stats"],
                            "reversible": bool(ok),
                            "readable_preview": readable_sequence(seq, int(args.preview_tokens)),
                            "preview_truncated": len(seq) > int(args.preview_tokens),
                        }
                    )
            except Exception as exc:
                mode_counters[mode]["records"] += 1
                failures.append(
                    {
                        "mode": mode,
                        "domain": domain,
                        "graph_file": str(graph_file),
                        "record_index": int(record_idx),
                        "error": str(exc),
                    }
                )
            if processed_pairs % 20 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "prototype_progress",
                            "records_done": row_idx,
                            "records_total": len(rows),
                            "mode_pairs_done": processed_pairs,
                            "failures": len(failures),
                            "elapsed_sec": time.time() - start,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    report = {
        "mode": "freq_guided_dfs_serialization_prototype",
        "important_scope": "small sampled prototype; no full corpus materialization",
        "graph_root": str(args.graph_root),
        "domains": domains,
        "split": args.split,
        "max_files_per_domain": int(args.max_files_per_domain),
        "max_records_per_domain": int(args.max_records_per_domain),
        "max_nodes_per_record": int(args.max_nodes_per_record),
        "records_picked": len(rows),
        "frequency_policy": {
            "root_order": "node feature frequency, then degree, then compact node label, then original id",
            "neighbor_order": "atom-edge-atom triplet frequency, then edge frequency, then destination node frequency",
            "main_path": "highest-scored child is emitted without parentheses; other children are branches",
            "tie_break_warning": "final original-id tie break makes this deterministic for current stored graph order, not a full graph-isomorphism canonical form",
        },
        "token_format": {
            "node": "N:<atomic_num,charge,aromatic,chiral_tag,degree,valence>; readable preview uses @/@@ for chiral tag",
            "edge": "E:<bond_type,aromatic,stereo>; omitted when it is the sampled default edge feature unless --keep-default-edge-feature is used",
            "ring": "R<num> open/close marker with reusable local ring ids",
            "atk": "@k backward reference from current atom occurrence to a previous atom occurrence",
            "branch": "( ... )",
            "component_sep": "<SEP>",
        },
        "default_edge_feature": list(guide.default_edge_feature),
        "omit_default_edge_feature": bool(omit_default_edge),
        "summary": {
            mode: summarize_mode(mode_counters[mode], mode_lengths[mode], mode_vocab[mode])
            for mode in ("ring", "atk")
        },
        "domain_summary": {
            domain: {
                mode: {
                    **dict(counter),
                    "mean_tokens_per_record": (counter["output_tokens"] / counter["records"]) if counter["records"] else 0.0,
                    "mean_tokens_per_input_node": (counter["output_tokens"] / counter["input_nodes"]) if counter["input_nodes"] else 0.0,
                }
                for mode, counter in per_mode.items()
            }
            for domain, per_mode in domain_summary.items()
        },
        "examples": examples,
        "failures": failures[:50],
        "failure_count": len(failures),
        "elapsed_sec": time.time() - start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {
        "records_picked": report["records_picked"],
        "failure_count": report["failure_count"],
        "ring": {
            "fully_reversible": report["summary"]["ring"]["counts"].get("fully_reversible", 0),
            "records": report["summary"]["ring"]["counts"].get("records", 0),
            "mean_tokens": report["summary"]["ring"]["lengths"].get("mean", 0),
            "vocab_size": report["summary"]["ring"]["vocab_size"],
        },
        "atk": {
            "fully_reversible": report["summary"]["atk"]["counts"].get("fully_reversible", 0),
            "records": report["summary"]["atk"]["counts"].get("records", 0),
            "mean_tokens": report["summary"]["atk"]["lengths"].get("mean", 0),
            "vocab_size": report["summary"]["atk"]["vocab_size"],
        },
        "out": str(args.out),
        "elapsed_sec": report["elapsed_sec"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
