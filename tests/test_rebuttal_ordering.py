"""The ablation must preserve graph semantics and implement every ordering key."""
from pathlib import Path
import sys
import unittest

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(PIPE / 'tests'))
from test_fddfs_cpp_equivalence import synthetic_graphs
from experiments.rebuttal.common import DEFAULT_EDGE, FixedOrderGuide, cpp_encoder, serialize_ids
from pipeline.serialization.fddfs import build_frequency_guide, build_vocab, encode_fddfs_token, serialize_fddfs, verify_roundtrip


class OrderingTests(unittest.TestCase):
    def test_cpp_matches_python_in_both_modes_and_roundtrips(self):
        graphs = list(synthetic_graphs().values())
        guide = build_frequency_guide(graphs)
        guide.default_edge_feature = DEFAULT_EDGE
        vocab = build_vocab(guide.node_counts, guide.edge_counts, DEFAULT_EDGE, True)
        for fixed in (False, True):
            encoder = cpp_encoder(guide, vocab, fixed=fixed)
            for graph in graphs:
                with self.subTest(fixed=fixed, nodes=graph['num_nodes']):
                    expected = serialize_fddfs(graph, FixedOrderGuide() if fixed else guide)
                    actual = serialize_ids(encoder, graph, vocab['ring_base'])
                    self.assertEqual(list(actual['ids']), [encode_fddfs_token(x, vocab) for x in expected.sequence])
                    self.assertTrue(verify_roundtrip(graph, expected))

    def test_fixed_is_invariant_to_frequency_and_preserves_default_bond(self):
        graphs = list(synthetic_graphs().values())
        guide = build_frequency_guide(graphs)
        guide.default_edge_feature = DEFAULT_EDGE
        vocab = build_vocab(guide.node_counts, guide.edge_counts, DEFAULT_EDGE, True)
        before = [list(serialize_ids(cpp_encoder(guide, vocab, fixed=True), g, vocab['ring_base'])['ids']) for g in graphs]
        for counts in (guide.node_counts, guide.edge_counts, guide.triplet_counts):
            for i, key in enumerate(counts):
                counts[key] = (i + 1) * 100003
        after = [list(serialize_ids(cpp_encoder(guide, vocab, fixed=True), g, vocab['ring_base'])['ids']) for g in graphs]
        self.assertEqual(before, after)
        self.assertEqual(guide.default_edge_feature, DEFAULT_EDGE)


if __name__ == '__main__':
    unittest.main()
