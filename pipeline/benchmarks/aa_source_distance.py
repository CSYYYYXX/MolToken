"""Exact normalized edit distances with length-bound pruning, never ANN."""
from itertools import islice
import json
import math
from pathlib import Path
import time

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from pipeline.benchmarks.pepbench_metrics import novelty_metrics


def nearest_source_distances(sequences, index, anchors, *, workers=8,
                             reference_batch=65536, query_batch=128):
    """Anchors must be real members of the indexed sources (CLI reads one).

    References of length R cannot improve distance d for query length Q when
    abs(Q-R)/max(Q,R) >= d. This is an exact Levenshtein lower bound, not a
    heuristic similarity threshold. Duplicate references can be collapsed
    locally because only a minimum is computed; query multiplicity is handled
    by the caller's raw-slot aggregation.
    """
    index = Path(index)
    summary = json.loads((index / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("source index incomplete")
    if min(workers, reference_batch, query_batch) < 1:
        raise ValueError("positive worker and block sizes required")
    unique = sorted(set(sequences))
    if any(not seq or len(seq) > summary["maximum_stored_length"] for seq in unique):
        raise ValueError("query outside index length coverage")
    if not unique:
        return {}, {"complete": True, "query_unique": 0, "compared_pairs": 0}
    initial = novelty_metrics(unique, anchors, workers=workers)
    best = np.array([initial["per_unique_sequence_nearest_distance"][seq] for seq in unique])
    lengths = np.array(list(map(len, unique)))
    totals = {"complete": False, "query_unique": len(unique), "compared_pairs": 0,
              "reference_records_read": 0, "whole_length_bound_skipped_records": 0}
    started = time.time()
    # Small buckets refine upper bounds before the dominant length-64 bucket.
    buckets = sorted(((int(k), v) for k, v in summary["length_counts"].items()
                      if int(k) <= summary["maximum_stored_length"]), key=lambda item: item[1])
    for reference_length, expected in buckets:
        denominator = np.maximum(lengths, reference_length)
        lower = np.abs(lengths - reference_length) / denominator
        if not np.any(lower < best):
            totals["whole_length_bound_skipped_records"] += expected
            continue
        seen = 0
        with (index / f"length_{reference_length:04d}.txt").open() as handle:
            while lines := list(islice(handle, reference_batch)):
                refs = list(dict.fromkeys(line.rstrip("\n") for line in lines))
                seen += len(lines)
                if any(len(seq) != reference_length for seq in refs):
                    raise ValueError(f"length-{reference_length} bucket contains a different length")
                active = np.flatnonzero(lower < best)
                for offset in range(0, len(active), query_batch):
                    selected = active[offset:offset + query_batch]
                    # Distances above this cutoff cannot improve any selected
                    # query. RapidFuzz returns cutoff+1, which also cannot win.
                    cutoff = math.ceil(float(np.max(best[selected] * denominator[selected])))
                    raw = process.cdist([unique[i] for i in selected], refs,
                        scorer=Levenshtein.distance, score_cutoff=cutoff,
                        dtype=np.uint32, workers=workers)
                    values = raw.min(axis=1) / denominator[selected]
                    best[selected] = np.minimum(best[selected], values)
                    totals["compared_pairs"] += len(selected) * len(refs)
                if seen % (reference_batch * 16) == 0:
                    print(f"DISTANCE_PROGRESS length={reference_length} records={seen}/{expected} active_queries={len(active)} elapsed={time.time()-started:.1f}", flush=True)
        if seen != expected:
            raise ValueError(f"length {reference_length}: read {seen}, expected {expected}")
        totals["reference_records_read"] += seen
        print(f"DISTANCE_BUCKET_COMPLETE length={reference_length} records={seen} elapsed={time.time()-started:.1f}", flush=True)
    omitted_minimum = summary["minimum_omitted_length"]
    if omitted_minimum is not None:
        lower = (omitted_minimum - lengths) / omitted_minimum
        if np.any(lower < best):
            raise ValueError("unindexed longer references could improve nearest distance; expand source index")
    totals.update(complete=True, elapsed_seconds=time.time() - started,
                  unindexed_lengths_proven_unable_to_improve=True,
                  reference_scope=summary["reference_scope"], source_records_total=summary["source_records_total"])
    return {seq: float(value) for seq, value in zip(unique, best, strict=True)}, totals
