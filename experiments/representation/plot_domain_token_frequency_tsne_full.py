#!/usr/bin/env python3
"""Use all saved frequency features for affinities; PCA is initialization only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import resource
import socket
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

import plot_domain_token_frequency_tsne as base


def fit_full_frequencies(frequencies, pca_scores, *, perplexity=30, seed=42, iterations=1500, threads=16):
    base.require(pca_scores.shape[0] == frequencies.shape[0] and pca_scores.shape[1] >= 2,
                 "PCA initialization/sample rows differ")
    initialization = np.array(pca_scores[:, :2], dtype=np.float32, copy=True)
    standard_deviation = float(initialization[:, 0].std())
    base.require(standard_deviation > 0, "PCA initialization has zero variance")
    initialization *= 1e-4 / standard_deviation
    # No projection is passed as X: all raw-frequency dimensions define affinities.
    model = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12, learning_rate="auto",
                 max_iter=iterations, metric="euclidean", init=initialization, random_state=seed,
                 method="barnes_hut", angle=.5, n_jobs=threads, verbose=2)
    with threadpool_limits(limits=threads):
        embedding = model.fit_transform(frequencies)
    base.require(np.isfinite(embedding).all() and embedding.shape == (frequencies.shape[0], 2), "Invalid embedding")
    base.require(np.isfinite(model.kl_divergence_) and model.kl_divergence_ < 1e10, "Incomplete t-SNE optimization")
    return embedding, {"input_frequency_dimensions": frequencies.shape[1], "pca_preprojection": False,
                       "pca_used_only_for_initial_coordinates": True, "perplexity": perplexity,
                       "iterations": int(model.n_iter_) + 1, "learning_rate": float(model.learning_rate_),
                       "kl_divergence": float(model.kl_divergence_)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--source", type=Path, default=base.OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    base.require("worker" in args.expected_host and socket.gethostname() == args.expected_host,
                 "Use the confirmed allocated CPU worker")
    output = args.output or args.source / "full_frequency"
    output.mkdir(parents=True, exist_ok=True)
    base.require(not (output / "summary.json").exists(), "Completed output exists; choose a new directory")
    start = time.monotonic()
    source_summary = json.loads((args.source / "summary.json").read_text())
    frequencies = load_npz(args.source / "token_frequencies.npz")
    pca_scores = np.load(args.source / "pca_scores.npy", allow_pickle=False)
    with (args.source / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    base.require(frequencies.shape[0] == len(rows) == source_summary["records"], "Sample count differs")
    base.require([int(row["sample_index"]) for row in rows] == list(range(len(rows))), "Sample row order differs")
    base.require(np.isfinite(frequencies.data).all() and (frequencies.data >= 0).all()
                 and np.allclose(np.asarray(frequencies.sum(axis=1)).ravel(), 1), "Invalid normalized frequencies")
    protocol = {"source": str(args.source), "worker": socket.gethostname(), "threads": args.threads,
                "frequencies": "Reuse the identical 10000 sampled raw frequency vectors; no corpus reread",
                "metric": "euclidean", "perplexity": 30, "seed": 42, "max_iter": 1500,
                "affinity_features": "All 17204 compact vocabulary columns; no feature dropping or preprojection",
                "initialization": "First two saved PCA scores scaled to first-coordinate standard deviation 1e-4; initialization only",
                "rationale": "50-component PCA retained only 13.0879% variance; inspect full-frequency affinities without projection loss",
                "parameter_selection": "Fixed before viewing either embedding; not chosen for visual separation",
                "domain_labels_used_in_fit": False, "test_accessed": False}
    base.write_json(output / "protocol.json", protocol)
    embedding, stats = fit_full_frequencies(frequencies, pca_scores, threads=args.threads)
    np.save(output / "embedding.npy", embedding)
    for row, point in zip(rows, embedding.tolist()):
        row.update(tsne_1=point[0], tsne_2=point[1])
    with (output / "coordinates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = base.make_figure(embedding, [int(row["domain_id"]) for row in rows])
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"domain_token_frequency_tsne.{suffix}", dpi=450)
    base.plt.close(fig)
    summary = {"complete": True, "protocol": protocol, "projection": stats,
               **{key: source_summary[key] for key in ("records", "domain_counts", "feature_audit", "body_length_stats")},
               "elapsed_seconds": time.monotonic() - start,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "interpretation": source_summary["interpretation"]}
    base.write_json(output / "summary.json", summary)
    base.event("FULL_FREQUENCY_TSNE_COMPLETE", output=str(output), **stats)


if __name__ == "__main__":
    main()
