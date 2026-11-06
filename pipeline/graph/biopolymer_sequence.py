"""Recover standard biopolymer sequences from the project's heavy graphs.

RDKit's :func:`MolToSequence` reads ``AtomPDBResidueInfo`` annotations.  Those
annotations are deliberately not part of the six node features / three edge
features used by the model, so a molecule reconstructed from FDDFS cannot be
passed directly to that writer.  The chemical graph still contains enough
information for the standard linear polymers produced by
``chem_graph_builder``.  This module inverts that exact grammar:

* peptides/proteins: follow the N--CA--C(=O) backbone and match each residue
  component against the same 20 amino-acid templates used by the encoder;
* DNA/RNA: find each sugar ring, follow O3'--P phosphodiester links, and match
  each residue component against the corresponding nucleotide template.

The decoder is intentionally strict.  A graph that is branched, cyclic,
multi-component, or not an exact standard residue graph raises ``ValueError``
instead of being repaired or mapped to ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping


def _arrays() -> Any:
    import numpy as np

    return np


def _nx() -> Any:
    import networkx as nx

    return nx


def _node_identity(feature: Iterable[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in feature)
    return values[0], values[1], values[2]


def _edge_identity(feature: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(value) for value in feature)
    return values[0], values[1]


def _colored_graph(
    node_features: Any,
    src: Iterable[int],
    dst: Iterable[int],
    edge_features: Any,
) -> Any:
    """Normalize the bidirectional project graph into one labelled nx.Graph."""

    np = _arrays()
    nx = _nx()
    nodes = np.asarray(node_features, dtype=np.int64).reshape((-1, 6))
    edges = np.asarray(edge_features, dtype=np.int64).reshape((-1, 3))
    src_values = [int(value) for value in src]
    dst_values = [int(value) for value in dst]
    if not (len(src_values) == len(dst_values) == int(edges.shape[0])):
        raise ValueError("edge arrays have inconsistent lengths")

    graph = nx.Graph()
    for node_id, feature in enumerate(nodes):
        full = tuple(int(value) for value in feature.tolist())
        graph.add_node(
            int(node_id),
            feature=full,
            identity=_node_identity(full),
            role="",
        )

    for u, v, feature in zip(src_values, dst_values, edges, strict=True):
        if u == v:
            raise ValueError(f"self edge at node {u}")
        if not 0 <= u < len(nodes) or not 0 <= v < len(nodes):
            raise ValueError(f"edge endpoint outside graph: {(u, v)}")
        full = tuple(int(value) for value in feature.tolist())
        if graph.has_edge(u, v):
            if graph.edges[u, v]["feature"] != full:
                raise ValueError(f"reverse edge features disagree for {(u, v)}")
            continue
        graph.add_edge(
            int(u),
            int(v),
            feature=full,
            identity=_edge_identity(full),
        )
    if graph.number_of_nodes() == 0:
        raise ValueError("empty graph")
    if not nx.is_connected(graph):
        raise ValueError("biopolymer graph must contain exactly one connected component")
    return graph


def _project_graph(graph_data: Mapping[str, Any]) -> Any:
    src, dst = graph_data["edges"]
    return _colored_graph(
        graph_data["node_features"],
        src,
        dst,
        graph_data["edge_features"],
    )


def _template_graph(template: Mapping[str, Any], *, full_labels: bool) -> Any:
    graph = _colored_graph(
        template["node_features"],
        template["edge_src"],
        template["edge_dst"],
        template["edge_features"],
    )
    if not full_labels:
        for node in graph.nodes:
            graph.nodes[node]["match_label"] = graph.nodes[node]["identity"]
        for u, v in graph.edges:
            graph.edges[u, v]["match_label"] = graph.edges[u, v]["identity"]
    else:
        for node in graph.nodes:
            graph.nodes[node]["match_label"] = graph.nodes[node]["feature"]
        for u, v in graph.edges:
            graph.edges[u, v]["match_label"] = graph.edges[u, v]["feature"]
    return graph


def _atomic_number(graph: Any, node: int) -> int:
    return int(graph.nodes[node]["feature"][0])


def _is_aromatic_node(graph: Any, node: int) -> bool:
    return bool(int(graph.nodes[node]["feature"][2]))


def _bond_type(graph: Any, u: int, v: int) -> int:
    return int(graph.edges[u, v]["feature"][0])


def _is_single(graph: Any, u: int, v: int) -> bool:
    return _bond_type(graph, u, v) == 1 and not bool(
        int(graph.edges[u, v]["feature"][1])
    )


def _is_double(graph: Any, u: int, v: int) -> bool:
    return _bond_type(graph, u, v) == 2 and not bool(
        int(graph.edges[u, v]["feature"][1])
    )


def _unique(values: Iterable[int], label: str) -> int:
    items = list(dict.fromkeys(int(value) for value in values))
    if len(items) != 1:
        raise ValueError(f"expected one {label}, found {len(items)}")
    return items[0]


@dataclass(frozen=True)
class _PeptideResidue:
    n: int
    ca: int
    carbonyl_c: int
    carbonyl_o: int


def _peptide_backbone_candidates(graph: Any) -> list[_PeptideResidue]:
    candidates: list[_PeptideResidue] = []
    for n in graph.nodes:
        if _atomic_number(graph, n) != 7 or _is_aromatic_node(graph, n):
            continue
        for ca in graph.neighbors(n):
            if not _is_single(graph, n, ca):
                continue
            if _atomic_number(graph, ca) != 6 or _is_aromatic_node(graph, ca):
                continue
            # A carbonyl carbon cannot itself be a C-alpha.
            if any(
                _atomic_number(graph, neighbor) == 8
                and _is_double(graph, ca, neighbor)
                for neighbor in graph.neighbors(ca)
            ):
                continue
            for carbonyl_c in graph.neighbors(ca):
                if carbonyl_c == n or not _is_single(graph, ca, carbonyl_c):
                    continue
                if _atomic_number(graph, carbonyl_c) != 6 or _is_aromatic_node(
                    graph, carbonyl_c
                ):
                    continue
                oxygens = [
                    neighbor
                    for neighbor in graph.neighbors(carbonyl_c)
                    if _atomic_number(graph, neighbor) == 8
                    and _is_double(graph, carbonyl_c, neighbor)
                ]
                if len(oxygens) != 1:
                    continue
                candidates.append(
                    _PeptideResidue(
                        n=int(n),
                        ca=int(ca),
                        carbonyl_c=int(carbonyl_c),
                        carbonyl_o=int(oxygens[0]),
                    )
                )

    by_ca: dict[int, _PeptideResidue] = {}
    for residue in candidates:
        previous = by_ca.setdefault(residue.ca, residue)
        if previous != residue:
            raise ValueError(f"ambiguous peptide backbone around C-alpha {residue.ca}")
    residues = list(by_ca.values())
    if not residues:
        raise ValueError("no N-CA-C(=O) peptide backbone found")
    if len({residue.n for residue in residues}) != len(residues):
        raise ValueError("one peptide nitrogen belongs to multiple backbone candidates")
    if len({residue.carbonyl_c for residue in residues}) != len(residues):
        raise ValueError("one carbonyl belongs to multiple backbone candidates")
    return residues


def _order_peptide_residues(
    graph: Any, residues: list[_PeptideResidue]
) -> tuple[list[_PeptideResidue], list[tuple[int, int]], int]:
    by_n = {residue.n: index for index, residue in enumerate(residues)}
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    peptide_edges: list[tuple[int, int]] = []

    for index, residue in enumerate(residues):
        targets = [
            by_n[neighbor]
            for neighbor in graph.neighbors(residue.carbonyl_c)
            if neighbor in by_n
            and by_n[neighbor] != index
            and _is_single(graph, residue.carbonyl_c, neighbor)
        ]
        targets = list(dict.fromkeys(targets))
        if len(targets) > 1:
            raise ValueError(f"branched peptide carbonyl at node {residue.carbonyl_c}")
        if not targets:
            continue
        target = int(targets[0])
        if target in predecessor:
            raise ValueError(f"branched peptide nitrogen at node {residues[target].n}")
        successor[index] = target
        predecessor[target] = index
        peptide_edges.append((residue.carbonyl_c, residues[target].n))

    starts = [index for index in range(len(residues)) if index not in predecessor]
    if len(starts) != 1:
        raise ValueError(f"expected one peptide N terminus, found {len(starts)}")
    order_indices: list[int] = []
    seen: set[int] = set()
    current = starts[0]
    while current not in seen:
        seen.add(current)
        order_indices.append(current)
        if current not in successor:
            break
        current = successor[current]
    if len(order_indices) != len(residues) or len(peptide_edges) != len(residues) - 1:
        raise ValueError("peptide backbone is cyclic, disconnected, or does not cover all residues")

    ordered = [residues[index] for index in order_indices]
    terminal = ordered[-1]
    terminal_oxygens = [
        neighbor
        for neighbor in graph.neighbors(terminal.carbonyl_c)
        if _atomic_number(graph, neighbor) == 8
        and _is_single(graph, terminal.carbonyl_c, neighbor)
    ]
    terminal_o = _unique(terminal_oxygens, "C-terminal OXT")
    if graph.degree[terminal_o] != 1:
        raise ValueError("C-terminal OXT must be terminal")
    for residue in ordered[:-1]:
        if any(
            _atomic_number(graph, neighbor) == 8
            and _is_single(graph, residue.carbonyl_c, neighbor)
            for neighbor in graph.neighbors(residue.carbonyl_c)
        ):
            raise ValueError("non-terminal peptide carbonyl carries a terminal oxygen")
    return ordered, peptide_edges, int(terminal_o)


@lru_cache(maxsize=1)
def _peptide_templates() -> dict[str, Any]:
    from pipeline.graph.chem_graph_builder import (
        STANDARD_AA_ORDER,
        _standard_protein_templates,
    )

    templates: dict[str, Any] = {}
    raw_templates = _standard_protein_templates()["internal"]
    for amino_acid in STANDARD_AA_ORDER:
        template = _template_graph(raw_templates[amino_acid], full_labels=False)
        for node, role in ((0, "N"), (1, "CA"), (2, "C")):
            template.nodes[node]["role"] = role
        templates[amino_acid] = template
    return templates


def _unique_isomorphic_label(
    component: Any,
    templates: Mapping[str, Any],
    *,
    full_labels: bool,
) -> str:
    nx = _nx()
    label_key = "feature" if full_labels else "identity"

    def node_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return left[label_key] == right[label_key] and left.get("role", "") == right.get(
            "role", ""
        )

    def edge_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return left[label_key] == right[label_key]

    matches = [
        label
        for label, template in templates.items()
        if component.number_of_nodes() == template.number_of_nodes()
        and component.number_of_edges() == template.number_of_edges()
        and nx.is_isomorphic(
            component,
            template,
            node_match=node_match,
            edge_match=edge_match,
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"residue matched {len(matches)} templates: {matches}")
    return str(matches[0])


def graph_to_protein_sequence(graph_data: Mapping[str, Any]) -> str:
    """Decode a standard, linear peptide/protein heavy graph from N to C."""

    nx = _nx()
    graph = _project_graph(graph_data)
    residues = _peptide_backbone_candidates(graph)
    ordered, peptide_edges, terminal_o = _order_peptide_residues(graph, residues)

    separated = graph.copy()
    separated.remove_edges_from(peptide_edges)
    separated.remove_node(terminal_o)
    components = list(nx.connected_components(separated))
    if len(components) != len(ordered):
        raise ValueError(
            f"peptide-bond cuts produced {len(components)} components for {len(ordered)} residues"
        )
    component_by_node: dict[int, set[int]] = {}
    for nodes in components:
        for node in nodes:
            component_by_node[int(node)] = set(int(value) for value in nodes)

    templates = _peptide_templates()
    sequence: list[str] = []
    consumed: set[int] = set()
    for residue in ordered:
        nodes = component_by_node.get(residue.ca)
        if nodes is None or nodes.intersection(consumed):
            raise ValueError("peptide residue components overlap or are missing")
        consumed.update(nodes)
        component = separated.subgraph(nodes).copy()
        component.nodes[residue.n]["role"] = "N"
        component.nodes[residue.ca]["role"] = "CA"
        component.nodes[residue.carbonyl_c]["role"] = "C"
        sequence.append(
            _unique_isomorphic_label(component, templates, full_labels=False)
        )
    return "".join(sequence)


@dataclass(frozen=True)
class _NucleotideResidue:
    ring: frozenset[int]
    c1: int
    c2: int
    c3: int
    c4: int
    o4: int
    c5: int
    o3: int
    o5: int
    phosphate: int | None


def _outside_neighbors(graph: Any, node: int, excluded: set[int]) -> list[int]:
    return [int(neighbor) for neighbor in graph.neighbors(node) if neighbor not in excluded]


def _nucleotide_residues(graph: Any) -> list[_NucleotideResidue]:
    nx = _nx()
    residues: list[_NucleotideResidue] = []
    seen_rings: set[frozenset[int]] = set()
    for cycle in nx.cycle_basis(graph):
        ring = frozenset(int(node) for node in cycle)
        if len(ring) != 5 or ring in seen_rings:
            continue
        atomic_numbers = sorted(_atomic_number(graph, node) for node in ring)
        if atomic_numbers != [6, 6, 6, 6, 8]:
            continue
        if any(_is_aromatic_node(graph, node) for node in ring):
            continue
        ring_subgraph = graph.subgraph(ring)
        if ring_subgraph.number_of_edges() != 5 or any(
            not _is_single(graph, u, v) for u, v in ring_subgraph.edges
        ):
            continue
        o4 = _unique(
            (node for node in ring if _atomic_number(graph, node) == 8),
            "sugar-ring oxygen",
        )
        o4_carbon_neighbors = [
            int(node)
            for node in graph.neighbors(o4)
            if node in ring and _atomic_number(graph, node) == 6
        ]
        if len(o4_carbon_neighbors) != 2:
            continue
        c1_candidates = [
            node
            for node in o4_carbon_neighbors
            if any(
                _atomic_number(graph, outside) == 7
                for outside in _outside_neighbors(graph, node, set(ring))
            )
        ]
        if len(c1_candidates) != 1:
            continue
        c1 = int(c1_candidates[0])
        c4 = int(next(node for node in o4_carbon_neighbors if node != c1))

        carbon_path_graph = graph.subgraph(set(ring).difference({o4}))
        try:
            carbon_path = nx.shortest_path(carbon_path_graph, c1, c4)
        except nx.NetworkXNoPath:
            continue
        if len(carbon_path) != 4:
            continue
        _, c2, c3, _ = (int(node) for node in carbon_path)

        c5_candidates = [
            node
            for node in _outside_neighbors(graph, c4, set(ring))
            if _atomic_number(graph, node) == 6 and _is_single(graph, c4, node)
        ]
        if len(c5_candidates) != 1:
            continue
        c5 = int(c5_candidates[0])
        o5_candidates = [
            node
            for node in graph.neighbors(c5)
            if node != c4
            and _atomic_number(graph, node) == 8
            and _is_single(graph, c5, node)
        ]
        if len(o5_candidates) != 1:
            continue
        o5 = int(o5_candidates[0])

        o3_candidates = [
            node
            for node in _outside_neighbors(graph, c3, set(ring))
            if _atomic_number(graph, node) == 8 and _is_single(graph, c3, node)
        ]
        if len(o3_candidates) != 1:
            continue
        o3 = int(o3_candidates[0])
        phosphate_candidates = [
            node
            for node in graph.neighbors(o5)
            if node != c5 and _atomic_number(graph, node) == 15
        ]
        if len(phosphate_candidates) > 1:
            raise ValueError("one nucleotide O5' is attached to multiple phosphorus atoms")
        phosphate = int(phosphate_candidates[0]) if phosphate_candidates else None
        residues.append(
            _NucleotideResidue(
                ring=ring,
                c1=c1,
                c2=c2,
                c3=c3,
                c4=c4,
                o4=o4,
                c5=c5,
                o3=o3,
                o5=o5,
                phosphate=phosphate,
            )
        )
        seen_rings.add(ring)

    if not residues:
        raise ValueError("no nucleotide sugar rings found")
    if len({residue.c1 for residue in residues}) != len(residues):
        raise ValueError("overlapping nucleotide sugar-ring assignments")
    return residues


def _order_nucleotide_residues(
    graph: Any,
    residues: list[_NucleotideResidue],
    cross_feature: tuple[int, int, int],
) -> tuple[list[_NucleotideResidue], list[tuple[int, int]]]:
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    links: list[tuple[int, int]] = []
    for current_index, current in enumerate(residues):
        if current.phosphate is None:
            continue
        previous_candidates = [
            previous_index
            for previous_index, previous in enumerate(residues)
            if previous_index != current_index
            and graph.has_edge(previous.o3, current.phosphate)
        ]
        if len(previous_candidates) != 1:
            raise ValueError(
                f"nucleotide phosphate {current.phosphate} has {len(previous_candidates)} O3' predecessors"
            )
        previous_index = int(previous_candidates[0])
        edge_feature = tuple(
            int(value) for value in graph.edges[residues[previous_index].o3, current.phosphate]["feature"]
        )
        if edge_feature != cross_feature:
            raise ValueError(
                f"unexpected phosphodiester edge feature {edge_feature}, expected {cross_feature}"
            )
        if previous_index in successor or current_index in predecessor:
            raise ValueError("branched nucleotide phosphodiester backbone")
        successor[previous_index] = current_index
        predecessor[current_index] = previous_index
        links.append((residues[previous_index].o3, current.phosphate))

    starts = [index for index in range(len(residues)) if index not in predecessor]
    if len(starts) != 1:
        raise ValueError(f"expected one nucleotide 5' terminus, found {len(starts)}")
    order_indices: list[int] = []
    seen: set[int] = set()
    current = starts[0]
    while current not in seen:
        seen.add(current)
        order_indices.append(current)
        if current not in successor:
            break
        current = successor[current]
    if len(order_indices) != len(residues) or len(links) != len(residues) - 1:
        raise ValueError("nucleotide backbone is cyclic, disconnected, or incomplete")
    if residues[order_indices[0]].phosphate is not None:
        raise ValueError("uncapped nucleotide 5' terminus unexpectedly has a phosphate")
    return [residues[index] for index in order_indices], links


@lru_cache(maxsize=8)
def _nucleotide_templates(target_type: str, position: str) -> dict[str, Any]:
    from pipeline.graph.chem_graph_builder import (
        STANDARD_DNA_ORDER,
        STANDARD_RNA_ORDER,
        _standard_nucleotide_templates,
    )

    raw = _standard_nucleotide_templates(target_type)
    alphabet = STANDARD_DNA_ORDER if target_type == "dna" else STANDARD_RNA_ORDER
    templates: dict[str, Any] = {}
    for base in alphabet:
        residue_template = raw[position][base]
        template = _template_graph(residue_template, full_labels=True)
        template.nodes[int(residue_template["o3_idx"])]["role"] = "O3"
        p_idx = residue_template["p_idx"]
        if p_idx is not None:
            template.nodes[int(p_idx)]["role"] = "P"
        templates[base] = template
    return templates


def graph_to_nucleotide_sequence(
    graph_data: Mapping[str, Any], *, target_type: str
) -> str:
    """Decode an uncapped standard DNA or RNA graph in the 5' to 3' direction."""

    if target_type not in {"dna", "rna"}:
        raise ValueError(f"target_type must be 'dna' or 'rna', got {target_type!r}")
    nx = _nx()
    from pipeline.graph.chem_graph_builder import _standard_nucleotide_templates

    graph = _project_graph(graph_data)
    residues = _nucleotide_residues(graph)
    raw_templates = _standard_nucleotide_templates(target_type)
    cross_feature = tuple(int(value) for value in raw_templates["cross_edge_feature"])
    ordered, links = _order_nucleotide_residues(graph, residues, cross_feature)

    separated = graph.copy()
    separated.remove_edges_from(links)
    components = list(nx.connected_components(separated))
    if len(components) != len(ordered):
        raise ValueError(
            f"phosphodiester cuts produced {len(components)} components for {len(ordered)} residues"
        )
    component_by_node: dict[int, set[int]] = {}
    for nodes in components:
        for node in nodes:
            component_by_node[int(node)] = set(int(value) for value in nodes)

    sequence: list[str] = []
    consumed: set[int] = set()
    length = len(ordered)
    for index, residue in enumerate(ordered):
        nodes = component_by_node.get(residue.c1)
        if nodes is None or nodes.intersection(consumed):
            raise ValueError("nucleotide residue components overlap or are missing")
        consumed.update(nodes)
        component = separated.subgraph(nodes).copy()
        component.nodes[residue.o3]["role"] = "O3"
        if residue.phosphate is not None:
            component.nodes[residue.phosphate]["role"] = "P"
        if length == 1:
            position = "single"
        elif index == 0:
            position = "first"
        elif index == length - 1:
            position = "last"
        else:
            position = "internal"
        sequence.append(
            _unique_isomorphic_label(
                component,
                _nucleotide_templates(target_type, position),
                full_labels=True,
            )
        )
    return "".join(sequence)


def graph_to_biopolymer_sequence(
    graph_data: Mapping[str, Any], *, target_type: str
) -> str:
    """Dispatch graph decoding for peptide, protein, DNA, or RNA targets."""

    if target_type in {"peptide", "protein"}:
        return graph_to_protein_sequence(graph_data)
    if target_type in {"dna", "rna"}:
        return graph_to_nucleotide_sequence(graph_data, target_type=target_type)
    raise ValueError(f"unsupported biopolymer target_type: {target_type!r}")


__all__ = [
    "graph_to_biopolymer_sequence",
    "graph_to_nucleotide_sequence",
    "graph_to_protein_sequence",
]
