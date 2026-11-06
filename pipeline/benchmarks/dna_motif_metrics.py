"""Population-Gini motif diversity; the caller specifies the motif universe.

This arithmetic alone does not establish a DNA-Diffusion paper reproduction.
Dropping zero-hit motifs changes the statistic, so the caller must retain its
declared matrix collection, including matrices with no matches.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def motif_diversity(counts: Sequence[int | float]) -> float | None:
    values = sorted(float(x) for x in counts)
    if not values:
        raise ValueError("motif universe is empty")
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("motif counts must be finite and non-negative")
    total = math.fsum(values)
    if total == 0:
        return None  # No usage distribution exists; not perfect diversity.
    n = len(values)
    gini = 2 * math.fsum(i * x for i, x in enumerate(values, 1)) / (n * total) - (n + 1) / n
    return 1.0 - gini
