#!/usr/bin/env python3
"""Synthetic regression tests for conservative aromatic-N-H completion."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.graph.aromatic_n_repair import (  # noqa: E402
    AROMATIC_N_COLLISION_FEATURE,
    _fresh_candidate,
    _graph_to_mol_faithful,
    _nonempty_assignment_count,
    aromatic_n_collision_indices,
    complete_aromatic_n_h,
    heavy_graph_equal,
)
from pipeline.graph.chem_graph_builder import mol_to_graph_data  # noqa: E402


def graph(smiles: str) -> dict:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol_to_graph_data(mol, sample_id=smiles, smiles=smiles)


def collision_only_graph(count: int) -> dict:
    nodes = np.tile(
        np.asarray(AROMATIC_N_COLLISION_FEATURE, dtype=np.int64),
        (int(count), 1),
    )
    return {
        "num_nodes": int(count),
        "node_features": nodes,
        "edges": ([], []),
        "edge_features": np.empty((0, 3), dtype=np.int64),
    }


def main() -> int:
    from rdkit import Chem

    pyridine = complete_aromatic_n_h(graph("c1ccncc1"))
    assert pyridine.raw_valid
    assert not pyridine.repaired

    pyrrole_graph = graph("c1cc[nH]c1")
    node_rows = np.asarray(pyrrole_graph["node_features"], dtype=np.int64)
    assert any(tuple(int(x) for x in row) == AROMATIC_N_COLLISION_FEATURE for row in node_rows)
    pyrrole = complete_aromatic_n_h(pyrrole_graph)
    assert not pyrrole.raw_valid
    assert pyrrole.repaired
    assert len(pyrrole.completions) == 1
    assert "[nH]" in pyrrole.completions[0].canonical_smiles
    assert pyrrole.graph_preservation_failures == 0

    # Asymmetric substitution makes the two N-H tautomers distinct canonical
    # SMILES, exercising ambiguity retention and reference-free selection.
    imidazole_graph = graph("Cc1nc(F)c[nH]1")
    imidazole = complete_aromatic_n_h(imidazole_graph)
    assert not imidazole.raw_valid
    assert imidazole.repaired
    assert imidazole.assignments_tested == imidazole.assignments_total
    assert len(imidazole.completions) == 2
    assert imidazole.selected_smiles == min(
        item.canonical_smiles for item in imidazole.completions
    )
    assert imidazole.graph_preservation_failures == 0

    # Every collision N is put in an explicit state: exactly the selected
    # tautomer sites have one explicit H, and none may infer another H.
    chosen = imidazole.completions[0]
    candidate_indices = aromatic_n_collision_indices(imidazole_graph)
    candidate = _fresh_candidate(
        _graph_to_mol_faithful(imidazole_graph),
        candidate_indices,
        chosen.atom_indices,
    )
    chosen_indices = set(chosen.atom_indices)
    for index in candidate_indices:
        atom = candidate.GetAtomWithIdx(index)
        assert atom.GetNoImplicit()
        assert atom.GetNumExplicitHs() == int(index in chosen_indices)
    Chem.SanitizeMol(candidate)
    for index in candidate_indices:
        atom = candidate.GetAtomWithIdx(index)
        # RDKit may normalize the NoImplicit flag on an unprotonated aromatic N;
        # the chemically relevant postcondition is still exactly zero inferred H.
        assert atom.GetNumImplicitHs() == 0
        assert atom.GetNumExplicitHs() == int(index in chosen_indices)

    # The accepted molecule must encode back to the exact six/three-column
    # graph, including edge direction and order-independent feature identity.
    repaired_mol = Chem.MolFromSmiles(pyrrole.selected_smiles)
    assert repaired_mol is not None
    repaired_graph = mol_to_graph_data(repaired_mol, sample_id="roundtrip")
    assert heavy_graph_equal(pyrrole_graph, repaired_graph)

    exhausted = complete_aromatic_n_h(imidazole_graph, max_assignments=1)
    assert exhausted.search_exhausted
    assert exhausted.search_skipped_reason == "assignment_cap_exceeded"
    assert exhausted.assignments_tested == 0
    assert not exhausted.completions

    # A different sanitization failure must not be disguised as an aromatic-N
    # repair.  Turning a tetra-coordinate carbon into fluorine creates an
    # explicit valence error while retaining a separate collision N.
    valence_graph = graph("c1cc[nH]c1.C(F)(F)(F)F")
    valence_nodes = np.asarray(valence_graph["node_features"], dtype=np.int64).copy()
    central = [
        index
        for index, row in enumerate(valence_nodes)
        if int(row[0]) == 6 and int(row[2]) == 0 and int(row[4]) == 4
    ]
    assert len(central) == 1
    valence_nodes[central[0], 0] = 9
    valence_graph["node_features"] = valence_nodes
    valence = complete_aromatic_n_h(valence_graph)
    assert not valence.raw_valid
    assert not valence.repaired
    assert valence.assignments_tested == 0
    assert valence.search_skipped_reason == "not_kekulize_failure"
    assert "valence" in str(valence.raw_error).lower()

    # The repair-local strict builder accepts every feature enum handled by the
    # certified strict evaluator, including bond type 0 and enum values beyond
    # the old decoder's hard-coded chiral tags.
    special_graph = {
        "num_nodes": 2,
        "node_features": np.asarray(
            [
                [6, 0, 0, int(Chem.ChiralType.CHI_OTHER), 1, 0],
                [6, 0, 0, 0, 1, 0],
            ],
            dtype=np.int64,
        ),
        "edges": ([0, 1], [1, 0]),
        "edge_features": np.asarray(
            [
                [0, 0, int(Chem.BondStereo.STEREOANY)],
                [0, 0, int(Chem.BondStereo.STEREOANY)],
            ],
            dtype=np.int64,
        ),
    }
    special = _graph_to_mol_faithful(special_graph)
    assert int(special.GetAtomWithIdx(0).GetChiralTag()) == int(
        Chem.ChiralType.CHI_OTHER
    )
    special_bond = special.GetBondBetweenAtoms(0, 1)
    assert special_bond is not None
    assert special_bond.GetBondType() == Chem.BondType.UNSPECIFIED
    assert int(special_bond.GetStereo()) == int(Chem.BondStereo.STEREOANY)

    for raw_tag_id in Chem.ChiralType.values:
        tag_id = int(raw_tag_id)
        enum_graph = {
            "num_nodes": 1,
            "node_features": np.asarray(
                [[6, 0, 0, tag_id, 0, 0]], dtype=np.int64
            ),
            "edges": ([], []),
            "edge_features": np.empty((0, 3), dtype=np.int64),
        }
        enum_mol = _graph_to_mol_faithful(enum_graph)
        assert int(enum_mol.GetAtomWithIdx(0).GetChiralTag()) == tag_id
    strict_settable_stereo_ids = {
        int(Chem.BondStereo.STEREONONE),
        int(Chem.BondStereo.STEREOANY),
        int(Chem.BondStereo.STEREOZ),
        int(Chem.BondStereo.STEREOE),
    }
    for stereo_id in sorted(strict_settable_stereo_ids):
        enum_graph = {
            **special_graph,
            "edge_features": np.asarray(
                [[0, 0, stereo_id], [0, 0, stereo_id]], dtype=np.int64
            ),
        }
        enum_mol = _graph_to_mol_faithful(enum_graph)
        enum_bond = enum_mol.GetBondBetweenAtoms(0, 1)
        assert enum_bond is not None
        assert int(enum_bond.GetStereo()) == stereo_id

    # The graph omits stereo reference atoms.  Like the certified strict
    # evaluator, the repair must reject enums that require those references,
    # not invent an arbitrary pair.
    all_stereo_ids = {int(value) for value in Chem.BondStereo.values}
    for stereo_id in sorted(all_stereo_ids - strict_settable_stereo_ids):
        unsupported_graph = {
            **special_graph,
            "num_nodes": 3,
            "node_features": np.vstack(
                [
                    special_graph["node_features"],
                    np.asarray(
                        [AROMATIC_N_COLLISION_FEATURE], dtype=np.int64
                    ),
                ]
            ),
            "edge_features": np.asarray(
                [[0, 0, stereo_id], [0, 0, stereo_id]], dtype=np.int64
            ),
        }
        unsupported = complete_aromatic_n_h(unsupported_graph)
        assert unsupported.search_skipped_reason == "graph_build_failed"
        assert unsupported.assignments_tested == 0

    stereo_graph = graph("F/C=C/F")
    assert np.any(np.asarray(stereo_graph["edge_features"])[:, 2] != 0)
    stereo_mol = _graph_to_mol_faithful(stereo_graph)
    Chem.SanitizeMol(stereo_mol)
    stereo_roundtrip = mol_to_graph_data(stereo_mol, sample_id="stereo_roundtrip")
    assert heavy_graph_equal(stereo_graph, stereo_roundtrip)

    stereo_repair_graph = graph("F/C=C/F.c1cc[nH]c1")
    stereo_repair = complete_aromatic_n_h(stereo_repair_graph)
    assert stereo_repair.repaired
    assert stereo_repair.graph_preservation_failures == 0
    assert stereo_repair.selected_smiles is not None

    # Malformed direct callers return a structured no-search result rather than
    # aborting an entire audit process.
    malformed = collision_only_graph(1)
    malformed["node_features"] = np.vstack(
        [
            malformed["node_features"],
            np.asarray([[6, 0, 0, 10_000, 0, 0]], dtype=np.int64),
        ]
    )
    malformed["num_nodes"] = 2
    build_failure = complete_aromatic_n_h(malformed)
    assert not build_failure.raw_valid
    assert build_failure.search_skipped_reason == "graph_build_failed"
    assert build_failure.assignments_tested == 0

    # Default cap boundary is pure arithmetic; avoid 4,095 RDKit sanitizations
    # in this fast regression test.  A 13-candidate graph confirms the core uses
    # the all-or-none capped path and does not partially enumerate.
    assert _nonempty_assignment_count(12) == 4095
    assert _nonempty_assignment_count(12) <= 4096
    assert _nonempty_assignment_count(13) == 8191
    assert _nonempty_assignment_count(13) > 4096
    capped = complete_aromatic_n_h(collision_only_graph(13))
    assert capped.search_exhausted
    assert capped.search_skipped_reason == "assignment_cap_exceeded"
    assert capped.assignments_total == 8191
    assert capped.assignments_tested == 0
    assert not capped.completions

    print("aromatic N-H repair synthetic tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
