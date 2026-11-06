from __future__ import annotations

import re
import sys

import numpy as np
import pytest

from pipeline.serialization.fddfs import deserialize_fddfs
from pipeline.serialization.freq_dfs_core import encode_feature


CARBON = encode_feature("N", (6, 0, 0, 0, 0, 0))
NITROGEN = encode_feature("N", (7, 0, 0, 0, 0, 0))
OXYGEN = encode_feature("N", (8, 0, 0, 0, 0, 0))
DOUBLE = encode_feature("E", (2, 0, 0))
SINGLE = (1, 0, 0)


def test_deep_linear_chain_does_not_use_python_recursion() -> None:
    node_count = 40_000
    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(200)
        graph = deserialize_fddfs([CARBON] * node_count, SINGLE)
    finally:
        sys.setrecursionlimit(previous_limit)

    assert graph["num_nodes"] == node_count
    src, dst = graph["edges"]
    assert len(src) == len(dst) == 2 * (node_count - 1)
    assert graph["node_features"].shape == (node_count, 6)
    assert graph["edge_features"].shape == (2 * (node_count - 1), 3)


def test_deeply_nested_branches_use_explicit_stack() -> None:
    node_count = 5_000
    tokens = [CARBON]
    for _ in range(node_count - 1):
        tokens.extend(("(", CARBON))
    tokens.extend(")" for _ in range(node_count - 1))

    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(200)
        graph = deserialize_fddfs(tokens, SINGLE)
    finally:
        sys.setrecursionlimit(previous_limit)

    assert graph["num_nodes"] == node_count
    src, dst = graph["edges"]
    assert len(src) == len(dst) == 2 * (node_count - 1)


def test_branch_and_ring_semantics_are_preserved() -> None:
    graph = deserialize_fddfs(
        [CARBON, "R1", "(", DOUBLE, OXYGEN, ")", NITROGEN, CARBON, "R1"],
        SINGLE,
    )

    assert graph["num_nodes"] == 4
    assert graph["edges"] == (
        [0, 1, 0, 2, 2, 3, 0, 3],
        [1, 0, 2, 0, 3, 2, 3, 0],
    )
    assert graph["node_features"].tolist() == [
        [6, 0, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0],
    ]
    assert np.asarray(graph["edge_features"]).tolist() == [
        [2, 0, 0],
        [2, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ]


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        ([CARBON, "(", OXYGEN], "missing branch close at pos=3"),
        ([CARBON, "(", OXYGEN, "<SEP>"], "missing branch close at pos=3"),
        ([CARBON, "(", ")"], "expected node token at pos=2, got=)"),
        ([CARBON, "R1"], "unclosed ring markers: ['R1']"),
    ],
)
def test_malformed_sequences_keep_existing_errors(
    tokens: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        deserialize_fddfs(tokens, SINGLE)
