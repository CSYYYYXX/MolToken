#!/usr/bin/env python3
"""Worker-side semantic checks for the paper's selected BPE merge paths."""
import argparse
import json
import socket
import unittest
import xml.etree.ElementTree as ET

import plot_bpe_merge_hierarchy as p


class MergeHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocabulary, cls.labels, cls.expansions, cls.rules = p.load_book()
        cls.cases = p.example_contexts(vocabulary, cls.expansions)

    def test_grey_context_cannot_connect_selected_atoms(self):
        edges = [(0, 2, 1), (2, 1, 1)]
        self.assertEqual(p.component_count({0, 1}, edges), 2)
        self.assertEqual(p.component_count({0, 1, 2}, edges), 1)
        self.assertEqual(p.component_count({0}, edges), 1)
        self.assertEqual(p.component_count(set(), edges), 0)

    def test_every_arrow_is_a_direct_binary_rule(self):
        self.assertEqual(len(p.PATHS), 3)
        for lane, path in p.PATHS.items():
            with self.subTest(lane=lane):
                self.assertEqual(path[0], 42)
                self.assertEqual(len(path), 5)
                for previous, token in zip(path, path[1:]):
                    rule = self.rules[token]
                    self.assertIn(previous, (rule["left"], rule["right"]))
                    self.assertEqual(self.expansions[token], self.expansions[rule["left"]] + self.expansions[rule["right"]])

    def test_every_interval_exactly_matches_the_frozen_expansion(self):
        for lane, path in p.PATHS.items():
            case = p.selected_context(self.cases, lane)
            intervals = p.path_intervals(case, lane, self.expansions, self.rules)
            for token in path:
                with self.subTest(lane=lane, token=token):
                    start, end = intervals[token]
                    self.assertEqual(tuple(case["ids"][start:end]), self.expansions[token])

    def test_all_selected_graphs_are_connected_without_context(self):
        for lane, path in p.PATHS.items():
            case = p.selected_context(self.cases, lane)
            intervals = p.path_intervals(case, lane, self.expansions, self.rules)
            _, edges = p.atom_layout(case["graph"])
            for token in path:
                with self.subTest(lane=lane, token=token):
                    self.assertEqual(p.component_count(p.covered_atoms(case, intervals[token]), edges), 1)

    def test_each_displayed_merge_adds_atoms_and_preserves_previous_atoms(self):
        for lane, path in p.PATHS.items():
            case = p.selected_context(self.cases, lane)
            intervals = p.path_intervals(case, lane, self.expansions, self.rules)
            for previous, token in zip(path, path[1:]):
                with self.subTest(lane=lane, token=token):
                    self.assertLess(p.covered_atoms(case, intervals[previous]), p.covered_atoms(case, intervals[token]))

    def test_saved_artifacts_match_current_paths_and_simplified_labels(self):
        saved = json.loads((p.OUTPUT / "selected_merges.json").read_text())
        self.assertEqual(saved["paths"], p.PATHS)
        for lane in p.LANES:
            self.assertIn(lane, saved["contexts"])
            for coverage in saved["contexts"][lane]["atom_coverage"].values():
                self.assertEqual(coverage["components"], 1)
        svg = (p.OUTPUT / "bpe_merge_hierarchy.svg").read_text()
        for removed in ("literal FDDFS symbols", "base symbols", "Sequence-", "Guanidino precursor"):
            self.assertNotIn(removed, svg)
        for suffix in ("png", "pdf", "svg"):
            self.assertGreater((p.OUTPUT / f"bpe_merge_hierarchy.{suffix}").stat().st_size, 1000)

    def test_each_card_restores_its_complete_abbreviated_sequence(self):
        svg = ET.parse(p.OUTPUT / "bpe_merge_hierarchy.svg").getroot()
        for lane, path in p.PATHS.items():
            for token in path[1:]:
                with self.subTest(lane=lane, token=token):
                    node = svg.find(f".//*[@id='sequence-{lane}-{token}']")
                    self.assertIsNotNone(node)
                    self.assertEqual("".join(node.itertext()).strip(), p.abbreviated_sequence(self.expansions[token], self.labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    args = parser.parse_args()
    p.require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the confirmed worker")
    unittest.main(argv=[__file__], verbosity=2)
