#!/usr/bin/env python3
"""Tests for recovering standard biopolymer sequences from heavy graphs."""

from __future__ import annotations

import random
import unittest

import numpy as np
from rdkit import Chem

from pipeline.finetune.fddfs_encode import build_graph
from pipeline.graph.biopolymer_sequence import (
    graph_to_nucleotide_sequence,
    graph_to_protein_sequence,
)
from pipeline.graph.chem_graph_builder import (
    NUCLEIC_ACID_FASTA_FLAVORS,
    STANDARD_AA_ORDER,
)
from pipeline.graph.to_smiles import graph_to_mol


def permute_nodes(graph: dict, seed: int) -> dict:
    """Return the same graph with a deterministic random node numbering."""

    count = int(graph["num_nodes"])
    order = list(range(count))
    random.Random(seed).shuffle(order)
    inverse = np.empty(count, dtype=np.int64)
    for new_id, old_id in enumerate(order):
        inverse[old_id] = new_id

    src, dst = graph["edges"]
    return {
        "num_nodes": count,
        "node_features": np.asarray(graph["node_features"], dtype=np.int64)[order],
        "edges": (
            inverse[np.asarray(src, dtype=np.int64)].tolist(),
            inverse[np.asarray(dst, dtype=np.int64)].tolist(),
        ),
        "edge_features": np.asarray(graph["edge_features"], dtype=np.int64).copy(),
    }


class ProteinSequenceRecoveryTest(unittest.TestCase):
    def assert_roundtrip(self, sequence: str, seed: int = 20260824) -> None:
        graph = build_graph(
            target_type="protein",
            sample_id=f"protein-{sequence}",
            sequence=sequence,
        )
        self.assertEqual(graph_to_protein_sequence(graph), sequence)
        self.assertEqual(
            graph_to_protein_sequence(permute_nodes(graph, seed)),
            sequence,
        )

    def test_all_single_amino_acids(self) -> None:
        for amino_acid in STANDARD_AA_ORDER:
            with self.subTest(amino_acid=amino_acid):
                self.assert_roundtrip(amino_acid)

    def test_all_dipeptides(self) -> None:
        for first in STANDARD_AA_ORDER:
            for second in STANDARD_AA_ORDER:
                sequence = first + second
                with self.subTest(sequence=sequence):
                    self.assert_roundtrip(sequence)

    def test_backbone_and_side_chain_cases(self) -> None:
        cases = (
            "ACDEFGHIKLMNPQRSTVWY",
            "YWVTSRQPNMLKIHGFEDCA",
            "ILILLLII",
            "HHWWWH",
            "PPGPGP",
        )
        for index, sequence in enumerate(cases):
            with self.subTest(sequence=sequence):
                self.assert_roundtrip(sequence, seed=20260824 + index)

    def test_rdkit_writer_needs_annotations(self) -> None:
        sequence = "ACDE"
        original = Chem.MolFromFASTA(sequence, flavor=0)
        self.assertIsNotNone(original)
        self.assertEqual(Chem.MolToSequence(original), sequence)

        graph = build_graph(
            target_type="protein",
            sample_id="protein-rdkit-metadata",
            sequence=sequence,
        )
        reconstructed = graph_to_mol(graph, sanitize=False)
        self.assertEqual(Chem.MolToSequence(reconstructed), "")


class NucleotideSequenceRecoveryTest(unittest.TestCase):
    def assert_roundtrip(
        self,
        target_type: str,
        sequence: str,
        seed: int = 20260824,
    ) -> None:
        graph = build_graph(
            target_type=target_type,
            sample_id=f"{target_type}-{sequence}",
            sequence=sequence,
        )
        self.assertEqual(
            graph_to_nucleotide_sequence(graph, target_type=target_type),
            sequence,
        )
        self.assertEqual(
            graph_to_nucleotide_sequence(
                permute_nodes(graph, seed),
                target_type=target_type,
            ),
            sequence,
        )

    def test_dna(self) -> None:
        cases = ("A", "AC", "ACGT", "ACGTT", "AAAA")
        cases += tuple(a + b for a in "ACGT" for b in "ACGT")
        for index, sequence in enumerate(cases):
            with self.subTest(sequence=sequence):
                self.assert_roundtrip("dna", sequence, seed=20260824 + index)

    def test_rna(self) -> None:
        cases = ("U", "AU", "ACGU", "UGCAA", "UUUU")
        cases += tuple(a + b for a in "ACGU" for b in "ACGU")
        for index, sequence in enumerate(cases):
            with self.subTest(sequence=sequence):
                self.assert_roundtrip("rna", sequence, seed=20260924 + index)

    def test_dna_and_rna_are_not_interchangeable(self) -> None:
        dna = build_graph(
            target_type="dna",
            sample_id="dna-cross-type",
            sequence="ACGT",
        )
        rna = build_graph(
            target_type="rna",
            sample_id="rna-cross-type",
            sequence="ACGU",
        )
        with self.assertRaises(ValueError):
            graph_to_nucleotide_sequence(dna, target_type="rna")
        with self.assertRaises(ValueError):
            graph_to_nucleotide_sequence(rna, target_type="dna")

    def test_rdkit_writer_needs_annotations(self) -> None:
        for target_type, sequence in (("dna", "ACGT"), ("rna", "ACGU")):
            with self.subTest(target_type=target_type):
                flavor = NUCLEIC_ACID_FASTA_FLAVORS[target_type]
                original = Chem.MolFromFASTA(sequence, flavor=flavor)
                self.assertIsNotNone(original)
                self.assertEqual(Chem.MolToSequence(original), sequence)

                graph = build_graph(
                    target_type=target_type,
                    sample_id=f"{target_type}-rdkit-metadata",
                    sequence=sequence,
                )
                reconstructed = graph_to_mol(graph, sanitize=False)
                self.assertEqual(Chem.MolToSequence(reconstructed), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
