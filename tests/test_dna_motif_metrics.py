import math

import pytest

from pipeline.benchmarks.dna_motif_metrics import motif_diversity


def test_population_gini_matches_pairwise_definition():
    counts = [0, 1, 2, 2, 7]
    gini = sum(abs(a - b) for a in counts for b in counts) / (2 * len(counts) * sum(counts))
    assert motif_diversity(counts) == pytest.approx(1 - gini)


def test_equal_counts_and_zero_hit_matrices():
    assert motif_diversity([7, 7, 7]) == pytest.approx(1)
    assert motif_diversity([9, 0, 0]) == pytest.approx(1 / 3)
    assert motif_diversity([0, 0]) is None


@pytest.mark.parametrize("counts", [[], [-1, 1], [math.nan, 1], [math.inf, 1]])
def test_invalid_counts_fail(counts):
    with pytest.raises(ValueError):
        motif_diversity(counts)
