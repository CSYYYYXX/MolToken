"""Synthetic regressions for count overflow and retained-output-only correction."""
from copy import deepcopy
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import cosine

from pipeline.benchmarks.moses_numeric import (
    COUNT_COSINE_KEYS, count_cosine_float64, float64_count_cosine,
)


def test_high_counts_match_arbitrary_precision_integer_dot_product():
    ref = {"a": 200000, "b": 30000, "c": 10000}
    gen = {"a": 10000, "b": 20000, "c": 15000}
    dot = sum(ref[k] * gen[k] for k in ref)
    rr, gg = sum(v*v for v in ref.values()), sum(v*v for v in gen.values())
    assert rr * gg > np.iinfo(np.int64).max
    expected = dot / math.sqrt(rr) / math.sqrt(gg)
    assert count_cosine_float64(ref, gen) == pytest.approx(expected, abs=1e-15)


def test_small_counts_and_missing_keys_match_upstream_formula():
    ref, gen = {"a": 2, "b": 3}, {"b": 2, "c": 4}
    expected = 1 - cosine(np.array([2, 3, 0]), np.array([0, 2, 4]))
    assert count_cosine_float64(ref, gen) == pytest.approx(expected, abs=1e-15)
    assert count_cosine_float64(ref, ref) == pytest.approx(1)
    assert count_cosine_float64({"a": 1}, {"b": 1}) == 0
    assert math.isnan(count_cosine_float64({}, gen))
    assert math.isnan(count_cosine_float64(ref, {}))


def test_context_restores_upstream_even_on_failure():
    original = lambda _ref, _gen: -1
    module = SimpleNamespace(cos_similarity=original)
    with pytest.raises(RuntimeError, match="native failure"):
        with float64_count_cosine(module):
            assert module.cos_similarity is count_cosine_float64
            raise RuntimeError("native failure")
    assert module.cos_similarity is original
    with pytest.raises(AttributeError):
        with float64_count_cosine(SimpleNamespace()):
            pass

