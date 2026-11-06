"""Numerical compatibility for upstream MOSES count cosine, without metric changes."""
from contextlib import contextmanager


COUNT_COSINE_KEYS = ("Frag/Test", "Scaf/Test", "Frag/TestSF", "Scaf/TestSF")
COUNT_COSINE_COMPATIBILITY = {
    "id": "moses-count-cosine-float64-v1",
    "affected_metrics": list(COUNT_COSINE_KEYS),
    "change": "cast count vectors to float64 before scipy.spatial.distance.cosine",
    "reason": "int64 uu*vv can overflow inside SciPy cosine and produce clipped scores",
    "unchanged": "upstream fragment/scaffold definitions, counts, cosine formula and samples",
}


def count_cosine_float64(ref_counts, gen_counts):
    import numpy as np
    from scipy.spatial.distance import cosine

    # Same empty-count behavior and key ordering as upstream MOSES.
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys], dtype=np.float64)
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys], dtype=np.float64)
    return float(1 - cosine(ref_vec, gen_vec))


@contextmanager
def float64_count_cosine(metrics_module):
    """Patch this process only; restore even if upstream evaluation raises."""
    original = metrics_module.cos_similarity
    metrics_module.cos_similarity = count_cosine_float64
    try:
        yield
    finally:
        metrics_module.cos_similarity = original
