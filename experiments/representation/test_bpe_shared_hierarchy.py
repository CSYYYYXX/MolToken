#!/usr/bin/env python3
"""Worker-side semantic checks for the shared-trunk illustration."""
import argparse
import json
import socket
import unittest
import xml.etree.ElementTree as ET

import plot_bpe_shared_hierarchy as s


class SharedHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.labels, cls.expansions, cls.rules, cls.paths, cls.cases = s.prepare()
        cls.nodes, cls.edges, _ = s.design(cls.root)

    def test_genuine_paths_and_complete_symbol_expansions(self):
        for path in self.paths.values():
            for a, b in zip(path, path[1:]):
                rule = self.rules[b]
                self.assertIn(a, (rule["left"], rule["right"]))
                self.assertEqual(self.expansions[b], self.expansions[rule["left"]] + self.expansions[rule["right"]])

    def test_shared_trunk_is_drawn_only_once_and_branches_twice(self):
        self.assertEqual(sum(node["token"] == self.root for node in self.nodes.values()), 1)
        self.assertEqual(sum(node["token"] == 629 for node in self.nodes.values()), 1)
        self.assertEqual(sum(node["token"] == 634 for node in self.nodes.values()), 1)
        self.assertEqual([b for a, b, _ in self.edges if a == "shared"], ["ncn", "amino"])
        self.assertEqual([b for a, b, _ in self.edges if a == "adenine"], ["dna", "rna"])

    def test_cross_domain_token_identity_includes_atom_features_and_bonds(self):
        for tid in (self.root, 629):
            values = [s.signature(case, case["intervals"][tid]) for case in self.cases.values()]
            self.assertTrue(all(value == values[0] for value in values))

    def test_displayed_fragments_are_connected_and_strictly_grow(self):
        for node in self.nodes.values():
            case = self.cases[node["case"]]
            current = s.p.covered_atoms(case, case["intervals"][node["token"]])
            self.assertEqual(s.p.component_count(current, case["edges"]), 1)
            if node["prior"] is not None:
                previous = s.p.covered_atoms(case, case["intervals"][node["prior"]])
                self.assertLess(previous, current)

    def test_saved_artifacts_and_collapsed_paths(self):
        saved = json.loads((s.OUTPUT / "selected_merges.json").read_text())
        self.assertEqual(saved["paths"], self.paths)
        self.assertEqual(len(saved["display_nodes"]), 9)
        self.assertEqual([edge["actual_path"] for edge in saved["display_edges"]], [e[2] for e in self.edges])
        svg = ET.parse(s.OUTPUT / "bpe_merge_hierarchy.svg").getroot()
        ids = {n.get("id") for n in svg.iter() if n.get("id")}
        for key in self.nodes:
            self.assertTrue(any(gid.startswith(f"card-{key}__") for gid in ids))
        for suffix in ("png", "pdf", "svg"):
            self.assertGreater((s.OUTPUT / f"bpe_merge_hierarchy.{suffix}").stat().st_size, 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    args = parser.parse_args()
    s.p.require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the confirmed worker")
    unittest.main(argv=[__file__], verbosity=2)
