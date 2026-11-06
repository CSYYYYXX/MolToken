"""Sequence/distribution metrics for the reconstructed PepBench-Gen track.

No sequence is repaired or resampled. Conditional metrics retain duplicate
valid slots; validity and coverage are always reported against all raw slots.
Distance and embedding PR definitions follow OmegAMP's published helpers.
"""
from collections import Counter
import math
import random

import numpy as np
from scipy.spatial.distance import cdist, jensenshannon

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA)


def sequence_is_valid(sequence):
    return isinstance(sequence, str) and 5 <= len(sequence) <= 80 and not set(sequence) - AA_SET


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "median": None, "p05": None, "p95": None}
    if not np.isfinite(values).all():
        raise ValueError("non-finite metric values")
    return {"count": len(values), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95))}


def slot_metrics(rows):
    if not rows:
        raise ValueError("no generation slots")
    valid = []
    statuses = Counter()
    exact = 0
    seen = set()
    for row in rows:
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("empty or duplicate slot ID")
        seen.add(row["id"])
        statuses[row["status"]] += 1
        if row["status"] == "valid":
            sequence = row["decoded_sequence"]
            if not sequence_is_valid(sequence):
                raise ValueError(f"slot {row['id']} declared valid but violates 20AA/5–80")
            valid.append(sequence)
            exact += len(sequence) == int(row["condition"])
    n, v, u = len(rows), len(valid), len(set(valid))
    return {"requested": n, "valid": v, "validity": v / n, "unique_valid": u,
            "uniqueness_within_valid": u / v if v else None,
            "unique_valid_over_requested": u / n,
            "exact_length_matches": exact, "exact_length_rate": exact / n,
            "status_counts": dict(statuses)}, valid


def novelty_metrics(sequences, references, *, workers=1, query_batch=64, reference_batch=4096):
    """Exact nearest normalized Levenshtein, not an approximate similarity search."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    reference = sorted(set(references))
    if not reference or any(not isinstance(s, str) or not s for s in reference):
        raise ValueError("nonempty reference sequences required")
    if not sequences:
        return {"valid_slots": 0, "exact_novelty_within_valid": None,
                "nearest_normalized_edit_distance": distribution_summary([])}
    # Compute each unique sequence once, then restore raw-slot multiplicities.
    unique = sorted(set(sequences))
    nearest = {}
    for i in range(0, len(unique), query_batch):
        queries = unique[i:i + query_batch]
        best = np.ones(len(queries), dtype=np.float64)
        for j in range(0, len(reference), reference_batch):
            distances = process.cdist(
                queries, reference[j:j + reference_batch],
                scorer=Levenshtein.normalized_distance, dtype=np.float64, workers=workers,
            )
            best = np.minimum(best, distances.min(axis=1))
        nearest.update(zip(queries, best, strict=True))
    distances = [float(nearest[s]) for s in sequences]
    reference_set = set(reference)
    novel = sum(s not in reference_set for s in sequences)
    return {"valid_slots": len(sequences), "reference_unique_sequences": len(reference),
            "exact_novel_slots": novel, "exact_novelty_within_valid": novel / len(sequences),
            "nearest_normalized_edit_distance": distribution_summary(distances),
            "distance_definition": "min_reference Levenshtein(g,r)/max(len(g),len(r))",
            "per_unique_sequence_nearest_distance": {s: float(v) for s, v in nearest.items()}}


def diversity_metric(sequences, *, seed=42, maximum_slots=2000):
    from rapidfuzz.distance import Levenshtein

    if maximum_slots < 2:
        raise ValueError("diversity subset must allow at least two slots")
    indices = sorted(random.Random(seed).sample(range(len(sequences)), min(len(sequences), maximum_slots)))
    total, pairs = 0.0, 0
    for i, index in enumerate(indices):
        for other in indices[i + 1:]:
            total += Levenshtein.normalized_distance(sequences[index], sequences[other])
            pairs += 1
    return {"mean_normalized_edit_distance": total / pairs if pairs else None,
            "pairs": pairs, "valid_slot_indices": indices, "seed": seed,
            "sampling": "valid raw slots without replacement; duplicate sequences retained"}


def marginal_jsd(sequences, references):
    if not references:
        raise ValueError("empty distribution reference")
    if not sequences:
        return {"aa_composition_jsd_bits": None, "length_jsd_bits": None}
    def counts(items, *, length):
        counter = Counter(map(len, items)) if length else Counter("".join(items))
        support = range(5, 81) if length else AA
        return np.array([counter[k] for k in support], dtype=np.float64)
    return {"aa_composition_jsd_bits": float(jensenshannon(counts(sequences, length=False), counts(references, length=False), base=2) ** 2),
            "length_jsd_bits": float(jensenshannon(counts(sequences, length=True), counts(references, length=True), base=2) ** 2)}


def frechet_distance(real, generated):
    """Gaussian Fréchet distance with sample covariance (ddof=1), in float64."""
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("embeddings must be matrices with the same feature dimension")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("non-finite embeddings")
    if min(len(real), len(generated)) < 2:
        return None
    a, b = np.atleast_2d(np.cov(real, rowvar=False)), np.atleast_2d(np.cov(generated, rowvar=False))
    values, vectors = np.linalg.eigh(a)
    # A covariance is PSD; clipping roundoff eigenvalues is not data filtering.
    sqrt_a = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    product = sqrt_a @ b @ sqrt_a
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T) / 2), 0)).sum()
    result = float(np.square(real.mean(axis=0) - generated.mean(axis=0)).sum() + np.trace(a) + np.trace(b) - 2 * cross)
    tolerance = 1e-7 * max(1.0, float(np.trace(a) + np.trace(b)))
    if result < -tolerance or not math.isfinite(result):
        raise ValueError(f"invalid Fréchet distance: {result}")
    return max(0.0, result)


def embedding_precision_recall(real, generated, *, neighborhood=3, batch_size=256):
    """kNN manifold precision/recall, retaining duplicate embeddings/slots."""
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("incompatible embedding matrices")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("non-finite embeddings")
    if neighborhood < 1 or batch_size < 1:
        raise ValueError("positive k and batch size required")
    if min(len(real), len(generated)) <= neighborhood:
        return {"precision": None, "recall": None, "reason": "fewer than k+1 valid/reference slots", "k": neighborhood}
    def radii(features):
        return np.concatenate([
            np.partition(cdist(features[i:i + batch_size], features), neighborhood, axis=1)[:, neighborhood]
            for i in range(0, len(features), batch_size)])
    def coverage(queries, support, radius):
        hits = sum(int((cdist(queries[i:i + batch_size], support) <= radius).any(axis=1).sum())
                   for i in range(0, len(queries), batch_size))
        return hits / len(queries)
    return {"precision": coverage(generated, real, radii(real)),
            "recall": coverage(real, generated, radii(generated)), "k": neighborhood}
