"""Conservative completion of aromatic N-H information lost by heavy graphs.

The project heavy-atom representation does not store atom hydrogen counts.
Consequently pyridine-like ``n`` and pyrrole-like ``[nH]`` can have the same
six node features.  This module does not alter atoms, bonds, charges, or the
stored heavy graph.  It only enumerates explicit-H assignments on the exact
colliding aromatic-nitrogen feature and accepts assignments that:

1. pass a full RDKit sanitization; and
2. round-trip to the exact original heavy graph.

Multiple accepted assignments are retained because the representation cannot
in general identify the original proton/tautomer location.  Callers must not
use a reference target to choose among them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from pipeline.graph.chem_graph_builder import mol_to_graph_data


# The empirically confirmed collision between aromatic ``n`` and ``[nH]``.
AROMATIC_N_COLLISION_FEATURE = (7, 0, 1, 0, 2, 3)


@dataclass(frozen=True)
class AromaticNHCompletion:
    """One graph-preserving, chemically valid explicit-H completion."""

    atom_indices: tuple[int, ...]
    canonical_smiles: str


@dataclass(frozen=True)
class AromaticNHRepairResult:
    """Outcome of a conservative aromatic-N-H completion search."""

    raw_valid: bool
    raw_error: str | None
    candidate_atom_indices: tuple[int, ...]
    assignments_total: int
    assignments_tested: int
    search_exhausted: bool
    search_skipped_reason: str | None
    completions: tuple[AromaticNHCompletion, ...]
    graph_preservation_failures: int

    @property
    def repaired(self) -> bool:
        return (not self.raw_valid) and bool(self.completions)

    @property
    def ambiguous(self) -> bool:
        return len(self.completions) > 1

    @property
    def selected_smiles(self) -> str | None:
        """Reference-free deterministic representative for deployment tests."""

        if self.raw_valid or not self.completions:
            return None
        return min(item.canonical_smiles for item in self.completions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_valid": self.raw_valid,
            "raw_error": self.raw_error,
            "candidate_atom_indices": list(self.candidate_atom_indices),
            "assignments_total": self.assignments_total,
            "assignments_tested": self.assignments_tested,
            "search_exhausted": self.search_exhausted,
            "search_skipped_reason": self.search_skipped_reason,
            "completion_count": len(self.completions),
            "repaired": self.repaired,
            "ambiguous": self.ambiguous,
            "selected_smiles": self.selected_smiles,
            "graph_preservation_failures": self.graph_preservation_failures,
            "completions": [
                {
                    "atom_indices": list(item.atom_indices),
                    "canonical_smiles": item.canonical_smiles,
                }
                for item in self.completions
            ],
        }


@dataclass(frozen=True)
class MoleculeSemanticComparison:
    """Fail-closed result for one canonical-SMILES molecule round trip.

    RDKit's raw aromatic flags and CW/CCW chiral enums are representation
    details: the former can be normalized to a Kekule form, while the latter
    depends on atom/neighbor ordering.  This result only accepts those two
    narrow normalizations.  Element, formal charge, topology, degree, total
    valence, CIP configuration, E/Z stereo, and every ordinary bond order must
    still be preserved.
    """

    equivalent: bool
    reason: str
    canonical_isomeric_smiles_equal: bool
    mesomer_equal: bool
    semantic_graph_isomorphic: bool
    aromatic_kekule_edge_normalizations: int
    aromatic_atom_normalizations: int
    chiral_parity_normalizations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "reason": self.reason,
            "canonical_isomeric_smiles_equal": (
                self.canonical_isomeric_smiles_equal
            ),
            "mesomer_equal": self.mesomer_equal,
            "semantic_graph_isomorphic": self.semantic_graph_isomorphic,
            "aromatic_kekule_edge_normalizations": (
                self.aromatic_kekule_edge_normalizations
            ),
            "aromatic_atom_normalizations": self.aromatic_atom_normalizations,
            "chiral_parity_normalizations": self.chiral_parity_normalizations,
        }


def _array(value: Any, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(shape)


def _edge_rows(graph: dict[str, Any]) -> tuple[tuple[int, int, int, int, int], ...]:
    edge_features = _array(graph["edge_features"], (-1, 3))
    src, dst = graph["edges"]
    rows = [
        (int(u), int(v), int(feature[0]), int(feature[1]), int(feature[2]))
        for u, v, feature in zip(src, dst, edge_features, strict=True)
    ]
    return tuple(sorted(rows))


def heavy_graph_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare represented heavy graphs independently of atom numbering.

    Canonical SMILES reparsing is allowed to renumber atoms.  Row-wise array
    equality therefore rejects many chemically identical round trips.  Exact
    node and edge features still have to match, but under a graph isomorphism.
    """

    import networkx as nx

    def colored_graph(graph: dict[str, Any]) -> Any:
        nodes = _array(graph["node_features"], (-1, 6))
        edge_features = _array(graph["edge_features"], (-1, 3))
        src_raw, dst_raw = graph["edges"]
        src = [int(value) for value in np.asarray(src_raw, dtype=np.int64).reshape(-1)]
        dst = [int(value) for value in np.asarray(dst_raw, dtype=np.int64).reshape(-1)]
        if not (len(src) == len(dst) == len(edge_features)):
            raise ValueError("edge arrays have inconsistent lengths")
        num_nodes = int(graph.get("num_nodes", len(nodes)))
        if num_nodes != len(nodes):
            raise ValueError("num_nodes disagrees with node_features")

        result = nx.Graph()
        for index, feature in enumerate(nodes):
            result.add_node(index, feature=tuple(int(value) for value in feature))

        directed: dict[tuple[int, int], tuple[int, int, int]] = {}
        for u, v, feature in zip(src, dst, edge_features, strict=True):
            if not 0 <= u < num_nodes or not 0 <= v < num_nodes or u == v:
                raise ValueError(f"invalid directed edge: {(u, v)}")
            key = (u, v)
            if key in directed:
                raise ValueError(f"duplicate directed edge: {key}")
            directed[key] = tuple(int(value) for value in feature)
        for (u, v), feature in directed.items():
            if directed.get((v, u)) != feature:
                raise ValueError(f"reverse edge feature mismatch: {(u, v)}")
            if u < v:
                result.add_edge(u, v, feature=feature)
        return result

    left_graph = colored_graph(left)
    right_graph = colored_graph(right)
    if (
        left_graph.number_of_nodes() != right_graph.number_of_nodes()
        or left_graph.number_of_edges() != right_graph.number_of_edges()
    ):
        return False
    node_match = nx.algorithms.isomorphism.categorical_node_match("feature", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("feature", None)
    return bool(
        nx.is_isomorphic(
            left_graph,
            right_graph,
            node_match=node_match,
            edge_match=edge_match,
        )
    )


_AROMATIC_KEKULE_BOND_TYPES = frozenset({"AROMATIC", "SINGLE", "DOUBLE"})


def _semantic_molecule_graph(molecule: Any) -> Any:
    """Build an atom-order-independent graph of represented chemistry."""

    import networkx as nx

    result = nx.Graph()
    for atom in molecule.GetAtoms():
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        result.add_node(
            int(atom.GetIdx()),
            invariant=(
                int(atom.GetAtomicNum()),
                int(atom.GetFormalCharge()),
                int(atom.GetDegree()),
                int(atom.GetTotalValence()),
                cip,
            ),
            cip=cip,
            aromatic=bool(atom.GetIsAromatic()),
            chiral_tag=int(atom.GetChiralTag()),
        )
    for bond in molecule.GetBonds():
        result.add_edge(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            bond_type=str(bond.GetBondType()),
            aromatic=bool(bond.GetIsAromatic()),
            in_ring=bool(bond.IsInRing()),
            stereo=int(bond.GetStereo()),
        )
    return result


def _semantic_heavy_graph(graph: dict[str, Any], source_molecule: Any) -> Any:
    """Build semantic labels from the unnormalized stored six/three columns."""

    import networkx as nx

    nodes = _array(graph["node_features"], (-1, 6))
    edge_features = _array(graph["edge_features"], (-1, 3))
    src_raw, dst_raw = graph["edges"]
    src = [int(value) for value in np.asarray(src_raw, dtype=np.int64).reshape(-1)]
    dst = [int(value) for value in np.asarray(dst_raw, dtype=np.int64).reshape(-1)]
    if not (len(src) == len(dst) == len(edge_features)):
        raise ValueError("edge arrays have inconsistent lengths")
    num_nodes = int(graph.get("num_nodes", len(nodes)))
    if num_nodes != len(nodes):
        raise ValueError("num_nodes disagrees with node_features")
    if int(source_molecule.GetNumAtoms()) != num_nodes:
        raise ValueError("source molecule atom count disagrees with heavy graph")

    result = nx.Graph()
    for index, feature in enumerate(nodes):
        atom = source_molecule.GetAtomWithIdx(index)
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        result.add_node(
            index,
            invariant=(
                int(feature[0]),
                int(feature[1]),
                int(feature[4]),
                int(feature[5]),
                cip,
            ),
            cip=cip,
            aromatic=bool(int(feature[2])),
            chiral_tag=int(feature[3]),
        )

    bond_type_names = {
        0: "UNSPECIFIED",
        1: "SINGLE",
        2: "DOUBLE",
        3: "TRIPLE",
        4: "AROMATIC",
    }
    directed: dict[tuple[int, int], tuple[int, int, int]] = {}
    for u, v, feature in zip(src, dst, edge_features, strict=True):
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes or u == v:
            raise ValueError(f"invalid directed edge: {(u, v)}")
        key = (u, v)
        if key in directed:
            raise ValueError(f"duplicate directed edge: {key}")
        directed[key] = tuple(int(value) for value in feature)
    for (u, v), feature in directed.items():
        if directed.get((v, u)) != feature:
            raise ValueError(f"reverse edge feature mismatch: {(u, v)}")
        if u < v:
            bond_type, aromatic, stereo = feature
            if bond_type not in bond_type_names:
                raise ValueError(f"unknown bond type id: {bond_type}")
            result.add_edge(
                u,
                v,
                bond_type=bond_type_names[bond_type],
                aromatic=bool(aromatic),
                in_ring=False,
                stereo=int(stereo),
            )
    bridges = {tuple(sorted(edge)) for edge in nx.bridges(result)}
    for u, v in result.edges():
        result.edges[u, v]["in_ring"] = tuple(sorted((u, v))) not in bridges
    return result


def _is_aromatic_kekule_edge_pair(left: Any, right: Any) -> bool:
    """Return whether an edge difference is a narrow ring resonance view."""

    return bool(
        left["in_ring"]
        and right["in_ring"]
        and (left["aromatic"] or right["aromatic"])
        and {left["bond_type"], right["bond_type"]}
        <= _AROMATIC_KEKULE_BOND_TYPES
    )


def _semantic_edge_match(left: Any, right: Any) -> bool:
    if left["stereo"] != right["stereo"]:
        return False
    if (
        left["bond_type"] == right["bond_type"]
        and left["aromatic"] == right["aromatic"]
    ):
        return True
    return _is_aromatic_kekule_edge_pair(left, right)


def compare_molecule_semantics(
    left: Any,
    right: Any,
    *,
    left_heavy_graph: dict[str, Any] | None = None,
    right_heavy_graph: dict[str, Any] | None = None,
) -> MoleculeSemanticComparison:
    """Compare two sanitized molecules while allowing two safe normalizations.

    The canonical isomeric identity is the primary chemistry gate.  A Mesomer
    hash additionally certifies resonance equivalence, but is never sufficient
    by itself because it does not distinguish every E/Z pair.  The explicit
    graph isomorphism provides defense in depth and records exactly which raw
    feature differences were allowed.
    """

    import networkx as nx
    from rdkit import Chem
    from rdkit.Chem import rdMolHash

    left_molecule = Chem.Mol(left)
    right_molecule = Chem.Mol(right)
    Chem.SanitizeMol(left_molecule)
    Chem.SanitizeMol(right_molecule)
    Chem.AssignStereochemistry(left_molecule, cleanIt=True, force=True)
    Chem.AssignStereochemistry(right_molecule, cleanIt=True, force=True)

    left_smiles = str(
        Chem.MolToSmiles(
            left_molecule, canonical=True, isomericSmiles=True
        )
    )
    right_smiles = str(
        Chem.MolToSmiles(
            right_molecule, canonical=True, isomericSmiles=True
        )
    )
    canonical_equal = bool(left_smiles and left_smiles == right_smiles)
    left_mesomer = rdMolHash.MolHash(
        left_molecule, rdMolHash.HashFunction.Mesomer
    )
    right_mesomer = rdMolHash.MolHash(
        right_molecule, rdMolHash.HashFunction.Mesomer
    )
    mesomer_equal = left_mesomer == right_mesomer
    if not canonical_equal:
        return MoleculeSemanticComparison(
            equivalent=False,
            reason="canonical_isomeric_identity_mismatch",
            canonical_isomeric_smiles_equal=False,
            mesomer_equal=mesomer_equal,
            semantic_graph_isomorphic=False,
            aromatic_kekule_edge_normalizations=0,
            aromatic_atom_normalizations=0,
            chiral_parity_normalizations=0,
        )

    if not mesomer_equal:
        return MoleculeSemanticComparison(
            equivalent=False,
            reason="mesomer_identity_mismatch",
            canonical_isomeric_smiles_equal=True,
            mesomer_equal=False,
            semantic_graph_isomorphic=False,
            aromatic_kekule_edge_normalizations=0,
            aromatic_atom_normalizations=0,
            chiral_parity_normalizations=0,
        )

    # Prefer the already-proven exact colored-graph mapping.  A molecule can
    # have several symmetry-related isomorphisms; a relaxed matcher may visit
    # a non-zero aromatic/chiral mapping before the exact one.  Short-circuiting
    # here makes the diagnostic canonical and avoids a false normalization.
    if (
        left_heavy_graph is not None
        and right_heavy_graph is not None
        and heavy_graph_equal(left_heavy_graph, right_heavy_graph)
    ):
        return MoleculeSemanticComparison(
            equivalent=True,
            reason="exact_semantic_graph",
            canonical_isomeric_smiles_equal=True,
            mesomer_equal=True,
            semantic_graph_isomorphic=True,
            aromatic_kekule_edge_normalizations=0,
            aromatic_atom_normalizations=0,
            chiral_parity_normalizations=0,
        )

    left_graph = (
        _semantic_heavy_graph(left_heavy_graph, left_molecule)
        if left_heavy_graph is not None
        else _semantic_molecule_graph(left_molecule)
    )
    right_graph = _semantic_molecule_graph(right_molecule)
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "invariant", None
    )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        left_graph,
        right_graph,
        node_match=node_match,
        edge_match=_semantic_edge_match,
    )
    for mapping in matcher.isomorphisms_iter():
        aromatic_edges = 0
        aromatic_atoms = 0
        chiral_parities = 0
        mapping_valid = True
        for left_index, right_index in mapping.items():
            left_node = left_graph.nodes[left_index]
            right_node = right_graph.nodes[right_index]
            if left_node["chiral_tag"] != right_node["chiral_tag"]:
                # CW/CCW can flip under an odd neighbor-order permutation.  A
                # matching assigned CIP configuration proves that no R/S
                # chemistry changed.  Unassigned tag drift remains rejected.
                if not (
                    left_node["cip"] in {"R", "S"}
                    and left_node["cip"] == right_node["cip"]
                ):
                    mapping_valid = False
                    break
                chiral_parities += 1
            if left_node["aromatic"] != right_node["aromatic"]:
                # An atom-level aromatic flag may only change when an incident
                # ring edge demonstrates the same aromatic/Kekule conversion.
                witnessed = False
                for left_neighbor in left_graph.neighbors(left_index):
                    right_neighbor = mapping[left_neighbor]
                    left_edge = left_graph.edges[left_index, left_neighbor]
                    right_edge = right_graph.edges[right_index, right_neighbor]
                    edge_drift = (
                        left_edge["bond_type"] != right_edge["bond_type"]
                        or left_edge["aromatic"] != right_edge["aromatic"]
                    )
                    if edge_drift and _is_aromatic_kekule_edge_pair(
                        left_edge, right_edge
                    ):
                        witnessed = True
                        break
                if not witnessed:
                    mapping_valid = False
                    break
                aromatic_atoms += 1
        if not mapping_valid:
            continue
        for left_u, left_v, left_edge in left_graph.edges(data=True):
            right_edge = right_graph.edges[mapping[left_u], mapping[left_v]]
            if (
                left_edge["bond_type"] != right_edge["bond_type"]
                or left_edge["aromatic"] != right_edge["aromatic"]
            ):
                aromatic_edges += 1
        return MoleculeSemanticComparison(
            equivalent=True,
            reason=(
                "exact_semantic_graph"
                if not (aromatic_edges or aromatic_atoms or chiral_parities)
                else "controlled_rdkit_representation_normalization"
            ),
            canonical_isomeric_smiles_equal=True,
            mesomer_equal=True,
            semantic_graph_isomorphic=True,
            aromatic_kekule_edge_normalizations=aromatic_edges,
            aromatic_atom_normalizations=aromatic_atoms,
            chiral_parity_normalizations=chiral_parities,
        )

    return MoleculeSemanticComparison(
        equivalent=False,
        reason="semantic_graph_or_stereo_mismatch",
        canonical_isomeric_smiles_equal=True,
        mesomer_equal=True,
        semantic_graph_isomorphic=False,
        aromatic_kekule_edge_normalizations=0,
        aromatic_atom_normalizations=0,
        chiral_parity_normalizations=0,
    )


def aromatic_n_collision_indices(graph: dict[str, Any]) -> tuple[int, ...]:
    """Return nodes with the exact lossy ``n``/``[nH]`` feature tuple."""

    nodes = _array(graph["node_features"], (-1, 6))
    collision = np.asarray(AROMATIC_N_COLLISION_FEATURE, dtype=np.int64)
    return tuple(
        int(index)
        for index, row in enumerate(nodes)
        if bool(np.array_equal(row, collision))
    )


def _nonempty_assignment_count(candidate_count: int) -> int:
    """Return the size of the complete non-empty explicit-H power set."""

    count = int(candidate_count)
    if count < 0:
        raise ValueError("candidate_count must be non-negative")
    return (1 << count) - 1 if count else 0


def _graph_to_mol_faithful(graph: dict[str, Any]) -> Any:
    """Rebuild every stored graph feature using the strict-evaluator rules.

    This intentionally lives in the repair sidecar rather than changing the
    project's established SMILES decoder.  In particular, bond type 0 and all
    recognized chiral enums are restored.  Stereo follows the strict evaluator:
    enums needing unavailable reference atoms are rejected rather than guessed.
    """

    from collections import Counter

    from rdkit import Chem

    node_features = _array(graph["node_features"], (-1, 6))
    edge_features = _array(graph["edge_features"], (-1, 3))
    src_raw, dst_raw = graph["edges"]
    src = [int(value) for value in np.asarray(src_raw, dtype=np.int64).reshape(-1)]
    dst = [int(value) for value in np.asarray(dst_raw, dtype=np.int64).reshape(-1)]
    if not (len(src) == len(dst) == len(edge_features)):
        raise ValueError(
            "edge arrays have inconsistent lengths: "
            f"src={len(src)} dst={len(dst)} features={len(edge_features)}"
        )
    num_nodes = int(graph.get("num_nodes", len(node_features)))
    if num_nodes != len(node_features):
        raise ValueError(
            f"num_nodes={num_nodes} disagrees with node_features={len(node_features)}"
        )

    bond_type_map = {
        0: Chem.BondType.UNSPECIFIED,
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for feature in node_features:
        atom = Chem.Atom(int(feature[0]))
        atom.SetFormalCharge(int(feature[1]))
        atom.SetIsAromatic(bool(int(feature[2])))
        try:
            atom.SetChiralTag(Chem.ChiralType.values[int(feature[3])])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown chiral tag id: {int(feature[3])}") from exc
        rw.AddAtom(atom)

    directed_seen: set[tuple[int, int]] = set()
    undirected_counts: Counter[tuple[int, int]] = Counter()
    undirected_features: dict[tuple[int, int], tuple[int, int, int]] = {}
    stereo_by_bond: dict[tuple[int, int], int] = {}
    for u, v, feature in zip(src, dst, edge_features, strict=True):
        if u == v:
            raise ValueError(f"self-edge is not valid in strict graph check: {u}")
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError(f"edge endpoint out of range: {(u, v)}")
        directed_key = (u, v)
        if directed_key in directed_seen:
            raise ValueError(f"duplicate directed edge: {(u, v)}")
        directed_seen.add(directed_key)

        key = (u, v) if u < v else (v, u)
        edge_feature = tuple(int(value) for value in feature.tolist())
        previous_feature = undirected_features.setdefault(key, edge_feature)
        if previous_feature != edge_feature:
            raise ValueError(
                f"reverse edge feature mismatch for {key}: "
                f"{previous_feature} != {edge_feature}"
            )
        undirected_counts[key] += 1

        bond_type_id, aromatic_id, stereo_id = edge_feature
        if bond_type_id not in bond_type_map:
            raise ValueError(f"unknown bond type id: {bond_type_id}")
        aromatic = bool(aromatic_id)
        bond_type = (
            Chem.BondType.AROMATIC if aromatic else bond_type_map[bond_type_id]
        )
        stereo_by_bond.setdefault(key, stereo_id)
        if rw.GetBondBetweenAtoms(u, v) is None:
            rw.AddBond(u, v, bond_type)
            bond = rw.GetBondBetweenAtoms(u, v)
            if bond is None:
                raise RuntimeError(f"RDKit failed to add bond {(u, v)}")
            bond.SetIsAromatic(aromatic or bond_type == Chem.BondType.AROMATIC)

    malformed = {key: count for key, count in undirected_counts.items() if count != 2}
    if malformed:
        raise ValueError(f"undirected edges are not represented twice: {malformed}")

    # Match the certified strict evaluator exactly: restore the stored enum but
    # do not invent stereo reference atoms that the graph does not contain.
    # RDKit accepts NONE/ANY/Z/E without those references.  CIS/TRANS/atrop
    # enums that require them fail here and become a structured build failure in
    # ``complete_aromatic_n_h`` rather than silently changing stereochemistry.
    for (left, right), stereo_id in stereo_by_bond.items():
        bond = rw.GetBondBetweenAtoms(left, right)
        if bond is None:
            raise RuntimeError(f"missing rebuilt bond {(left, right)}")
        try:
            stereo = Chem.BondStereo.values[stereo_id]
        except KeyError as exc:
            raise ValueError(f"unknown bond stereo id: {stereo_id}") from exc
        if stereo not in (
            Chem.BondStereo.STEREONONE,
            Chem.BondStereo.STEREOANY,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOE,
        ):
            raise ValueError(
                f"bond {(left, right)} stereo id {stereo_id} requires unavailable "
                "reference atoms"
            )
        try:
            bond.SetStereo(stereo)
        except ValueError as exc:
            raise ValueError(
                f"cannot restore bond {(left, right)} stereo id: {stereo_id}"
            ) from exc
    return rw.GetMol()


def _fresh_candidate(base_mol: Any, candidates: tuple[int, ...], selected: tuple[int, ...]) -> Any:
    from rdkit import Chem

    mol = Chem.Mol(base_mol)
    selected_set = set(selected)
    for index in candidates:
        atom = mol.GetAtomWithIdx(int(index))
        atom.SetNumExplicitHs(1 if index in selected_set else 0)
        # Aromatic nitrogen must not have RDKit silently infer an additional H.
        atom.SetNoImplicit(True)
    return mol


def materialize_aromatic_n_completion(
    graph: dict[str, Any], atom_indices: tuple[int, ...]
) -> Any:
    """Rebuild and sanitize one certified explicit-H completion."""

    from rdkit import Chem

    candidates = aromatic_n_collision_indices(graph)
    selected = tuple(int(value) for value in atom_indices)
    if not selected or selected != tuple(sorted(set(selected))):
        raise ValueError("completion atom indices must be nonempty sorted unique")
    if not set(selected) <= set(candidates):
        raise ValueError("completion atom indices include a non-candidate atom")
    molecule = _fresh_candidate(
        _graph_to_mol_faithful(graph), candidates, selected
    )
    Chem.SanitizeMol(molecule)
    return molecule


def complete_aromatic_n_h(
    graph: dict[str, Any],
    *,
    max_assignments: int = 4096,
    require_kekulize_failure: bool = True,
) -> AromaticNHRepairResult:
    """Enumerate graph-preserving explicit-H completions for one heavy graph.

    Raw-valid graphs are returned unchanged.  By default a failed graph is
    searched only when RDKit reports a kekulization failure; arbitrary valence
    or topology errors are deliberately outside this repair's scope.
    """

    from rdkit import Chem

    if int(max_assignments) <= 0:
        raise ValueError("max_assignments must be positive")

    candidates = aromatic_n_collision_indices(graph)
    assignments_total = _nonempty_assignment_count(len(candidates))
    try:
        base = _graph_to_mol_faithful(graph)
    except ValueError as exc:  # known malformed graph; runtime failures propagate
        return AromaticNHRepairResult(
            raw_valid=False,
            raw_error=f"{type(exc).__name__}: {exc}",
            candidate_atom_indices=candidates,
            assignments_total=assignments_total,
            assignments_tested=0,
            search_exhausted=False,
            search_skipped_reason="graph_build_failed",
            completions=(),
            graph_preservation_failures=0,
        )
    raw = Chem.Mol(base)
    try:
        Chem.SanitizeMol(raw)
    except Chem.MolSanitizeException as exc:
        raw_error = f"{type(exc).__name__}: {exc}"
    else:
        return AromaticNHRepairResult(
            raw_valid=True,
            raw_error=None,
            candidate_atom_indices=(),
            assignments_total=0,
            assignments_tested=0,
            search_exhausted=False,
            search_skipped_reason=None,
            completions=(),
            graph_preservation_failures=0,
        )

    search_skipped_reason: str | None = None
    if not candidates:
        search_skipped_reason = "no_collision_candidates"
    elif require_kekulize_failure and "kekul" not in raw_error.lower():
        search_skipped_reason = "not_kekulize_failure"
    elif assignments_total > int(max_assignments):
        # All-or-none is deliberate: a partial power-set search could not make
        # defensible uniqueness or all-completion claims.  No assignment is
        # tested when the complete search would exceed the configured cap.
        search_skipped_reason = "assignment_cap_exceeded"
    if search_skipped_reason is not None:
        return AromaticNHRepairResult(
            raw_valid=False,
            raw_error=raw_error,
            candidate_atom_indices=candidates,
            assignments_total=assignments_total,
            assignments_tested=0,
            search_exhausted=search_skipped_reason == "assignment_cap_exceeded",
            search_skipped_reason=search_skipped_reason,
            completions=(),
            graph_preservation_failures=0,
        )

    by_smiles: dict[str, AromaticNHCompletion] = {}
    assignments_tested = 0
    graph_preservation_failures = 0
    # Smaller numbers of added H are tried first.  We still enumerate the full
    # bounded search so ambiguity is measured rather than silently hidden.
    for size in range(1, len(candidates) + 1):
        for selected in combinations(candidates, size):
            assignments_tested += 1
            mol = _fresh_candidate(base, candidates, tuple(selected))
            try:
                Chem.SanitizeMol(mol)
                rebuilt = mol_to_graph_data(mol, sample_id="aromatic_n_h_completion")
                if not heavy_graph_equal(graph, rebuilt):
                    graph_preservation_failures += 1
                    continue
                smiles = str(
                    Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                )
            except Chem.MolSanitizeException:  # chemically invalid H assignments
                continue
            by_smiles.setdefault(
                smiles,
                AromaticNHCompletion(
                    atom_indices=tuple(int(value) for value in selected),
                    canonical_smiles=smiles,
                ),
            )

    completions = tuple(by_smiles[key] for key in sorted(by_smiles))
    return AromaticNHRepairResult(
        raw_valid=False,
        raw_error=raw_error,
        candidate_atom_indices=candidates,
        assignments_total=assignments_total,
        assignments_tested=assignments_tested,
        search_exhausted=False,
        search_skipped_reason=None,
        completions=completions,
        graph_preservation_failures=graph_preservation_failures,
    )


__all__ = [
    "AROMATIC_N_COLLISION_FEATURE",
    "AromaticNHCompletion",
    "AromaticNHRepairResult",
    "MoleculeSemanticComparison",
    "aromatic_n_collision_indices",
    "compare_molecule_semantics",
    "complete_aromatic_n_h",
    "heavy_graph_equal",
    "materialize_aromatic_n_completion",
]
