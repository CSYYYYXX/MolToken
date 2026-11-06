#!/usr/bin/env python3
"""Worker-only, label-free t-SNE of balanced compact-BPE token frequencies."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import socket
import struct
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.sparse import csr_matrix, save_npz
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scaling"))
from audit_domainwise_compact import DOMAIN_NAMES, splitmix64_array

RECORDED_STORAGE = Path("./storage/molscaling")
WORKSPACE = Path(__file__).resolve().parents[5]
OUTPUT = WORKSPACE / "HoloMol/doc/results/analysis/token_frequency_tsne_20260916"
HEADER = struct.Struct("<8sIQQi")
NAMES = ("Small molecule", "Peptide", "Protein", "DNA", "RNA")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#6B55A3")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event(name, **values):
    print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "event": name, **values}), flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def relocate(path, recorded_root, actual_root):
    return actual_root / Path(path).relative_to(recorded_root)


def draw_samples(manifest, per_domain, seed, recorded_root, actual_root):
    require(manifest["status"] == "complete" and manifest["schema_version"] == "domain_split_raw_indices_v1",
            "Expected the completed frozen domain/split manifest")
    quotas = dict(per_domain) if isinstance(per_domain, Mapping) else {d: per_domain for d in DOMAIN_NAMES}
    require(set(quotas) == set(DOMAIN_NAMES), "Sample quotas must name exactly the five domains")
    require(all(isinstance(n, (int, np.integer)) and n >= 0 for n in quotas.values())
            and sum(quotas.values()) > 0, "Sample quotas must be nonnegative integers with a positive total")
    rows = []
    for domain_id, domain in enumerate(DOMAIN_NAMES):
        selected_count = int(quotas[domain])
        view = manifest["views"][domain]["train"]
        shards = sorted(view["shards"], key=lambda entry: entry["shard_id"])
        counts = np.array([entry["records"] for entry in shards], dtype=np.int64)
        total = int(counts.sum())
        require(total == view["records"] and total >= selected_count, f"Invalid training population: {domain}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, domain_id]))
        selected = np.sort(rng.choice(total, size=selected_count, replace=False))
        ends = counts.cumsum()
        starts = ends - counts
        for shard_pos, entry in enumerate(shards):
            positions = selected[(selected >= starts[shard_pos]) & (selected < ends[shard_pos])]
            if not len(positions):
                continue
            path = relocate(entry["raw_indices_file"], recorded_root, actual_root)
            require(path.stat().st_size == entry["records"] * 4, f"Raw-index size mismatch: {path}")
            index = np.memmap(path, mode="r", dtype="<u4")
            raw = np.asarray(index[positions - starts[shard_pos]], dtype=np.int64)
            require(len(np.unique(raw)) == len(raw), f"Duplicate selected record identities: {domain}")
            for position, raw_id in zip(positions.tolist(), raw.tolist()):
                rows.append({"domain": domain, "domain_id": domain_id, "shard_id": entry["shard_id"],
                             "shard": Path(entry["token_file"]).name, "raw_index": raw_id,
                             "train_view_index": position})
        event("sampled_training_ids", domain=domain, population=total, selected=selected_count)
    order = np.random.default_rng(np.random.SeedSequence([seed, 100])).permutation(len(rows))
    rows = [{"sample_index": i, **rows[j]} for i, j in enumerate(order.tolist())]
    require(Counter(row["domain"] for row in rows) == Counter(quotas), "Selected sample differs from its quotas")
    require(len({(row["shard_id"], row["raw_index"]) for row in rows}) == len(rows), "Repeated record identities")
    return rows


def relative_frequency_matrix(sequences, vocab_size):
    data, indices, indptr = [], [], [0]
    for sequence in sequences:
        sequence = np.asarray(sequence)
        require(sequence.ndim == 1 and sequence.size > 0, "Empty or non-vector compact body")
        require(np.issubdtype(sequence.dtype, np.integer), "Non-integer token ID")
        require(int(sequence.min()) >= 0 and int(sequence.max()) < vocab_size, "Token outside compact vocabulary")
        ids, counts = np.unique(sequence, return_counts=True)
        indices.extend(ids.tolist())
        data.extend((counts / sequence.size).tolist())
        indptr.append(len(data))
    result = csr_matrix((np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32),
                         np.array(indptr, dtype=np.int32)), shape=(len(sequences), vocab_size))
    require(np.allclose(np.asarray(result.sum(axis=1)).ravel(), 1), "Frequency row does not sum to one")
    return result


def read_sample_features(manifest, rows, recorded_root, actual_root):
    paths = {key: relocate(manifest["dataset"][key], recorded_root, actual_root)
             for key in ("compact_dir", "index_dir", "domain_id_dir")}
    sequences = [None] * len(rows)
    assignment = manifest["split_assignment"]
    max_token = -1
    for shard in sorted({row["shard"] for row in rows}):
        selected = sorted((row for row in rows if row["shard"] == shard), key=lambda row: row["raw_index"])
        payload_path = paths["compact_dir"] / shard
        with payload_path.open("rb") as stream:
            magic, version, n_records, _, shard_max_token = HEADER.unpack(stream.read(HEADER.size))
            require(magic == b"YTTMIBPE" and version == 1, f"Unsupported compact format: {payload_path}")
            max_token = max(max_token, shard_max_token)
            offsets = np.memmap(paths["index_dir"] / f"{shard}.payload_offsets.u64", mode="r", dtype="<u8")
            lengths = np.memmap(paths["index_dir"] / f"{shard}.lengths.u32", mode="r", dtype="<u4")
            domains = np.memmap(paths["domain_id_dir"] / f"{shard}.domain_ids.u16", mode="r", dtype="<u2")
            require(len(offsets) == len(lengths) == len(domains) == n_records, f"Sidecar count mismatch: {shard}")
            raw = np.array([row["raw_index"] for row in selected], dtype=np.uint64)
            require(int(raw.max()) < n_records, f"Selected record out of range: {shard}")
            keys = (np.uint64(selected[0]["shard_id"]) << np.uint64(32)) | raw
            buckets = splitmix64_array(keys ^ np.uint64(assignment["seed"])) % np.uint64(assignment["modulus"])
            low, high = assignment["train_buckets"]
            require(bool(((buckets >= low) & (buckets <= high)).all()), "Selected record is not in the frozen training split")
            for row in selected:
                raw_id = row["raw_index"]
                require(int(domains[raw_id]) == row["domain_id"], f"Domain mismatch: {shard}/{raw_id}")
                offset, length = int(offsets[raw_id]), int(lengths[raw_id])
                require(offset >= HEADER.size + 4 and length > 0, f"Invalid indexed body: {shard}/{raw_id}")
                stream.seek(offset - 4)
                require(struct.unpack("<I", stream.read(4))[0] == length, f"Indexed length mismatch: {shard}/{raw_id}")
                blob = stream.read(length * 4)
                require(len(blob) == length * 4, f"Truncated compact body: {shard}/{raw_id}")
                tokens = np.frombuffer(blob, dtype="<i4").copy()
                require(int(tokens.min()) >= 0 and int(tokens.max()) <= shard_max_token,
                        f"Body IDs exceed header vocabulary: {shard}/{raw_id}")
                sequences[row["sample_index"]] = tokens
                row["body_tokens"] = length
                row["unique_tokens"] = int(np.unique(tokens).size)
        event("read_selected_payloads", shard=shard, selected=len(selected))
    require(all(sequence is not None for sequence in sequences), "A selected record was not read")
    frequencies = relative_frequency_matrix(sequences, max_token + 1)
    duplicate_sequences = len(sequences) - len({tuple(sequence.tolist()) for sequence in sequences})
    return frequencies, {"vocabulary_columns": max_token + 1, "observed_token_types": int(np.unique(frequencies.indices).size),
                         "nonzero_frequencies": int(frequencies.nnz), "exact_sequence_duplicate_records": duplicate_sequences,
                         "payload_tokens_read": sum(row["body_tokens"] for row in rows),
                         "records_over_511_body_tokens": sum(row["body_tokens"] > 511 for row in rows)}


def project_frequencies(frequencies, components, perplexity, seed, iterations, threads):
    require(0 < components < min(frequencies.shape), "Invalid PCA dimension")
    require(0 < perplexity < frequencies.shape[0], "Invalid t-SNE perplexity")
    start = time.monotonic()
    with threadpool_limits(limits=threads):
        pca = PCA(n_components=components, svd_solver="arpack", whiten=False, random_state=seed)
        reduced = pca.fit_transform(frequencies)
        event("pca_complete", dimensions=components, explained_variance=float(pca.explained_variance_ratio_.sum()),
              elapsed_seconds=time.monotonic() - start)
        model = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12, learning_rate="auto",
                     max_iter=iterations, init="pca", metric="euclidean", random_state=seed,
                     method="barnes_hut", angle=.5, n_jobs=threads, verbose=2)
        embedding = model.fit_transform(reduced)
    require(embedding.shape == (frequencies.shape[0], 2) and np.isfinite(embedding).all(), "Invalid t-SNE coordinates")
    require(np.isfinite(model.kl_divergence_) and model.kl_divergence_ < 1e10, "t-SNE did not finish normal optimization")
    return embedding, reduced, pca, {"pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                                    "pca_components": components, "tsne_perplexity": perplexity,
                                    "tsne_kl_divergence": float(model.kl_divergence_), "tsne_iterations": int(model.n_iter_) + 1,
                                    "tsne_learning_rate": float(model.learning_rate_),
                                    "elapsed_seconds": time.monotonic() - start}


def make_figure(embedding, domain_ids, *, marker_size=5, alpha=.64):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#B0B6BC", "axes.linewidth": .6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    fig.subplots_adjust(left=.055, right=.99, bottom=.13, top=.985)
    # Sample order was randomly interleaved before any fitting: no domain is always on top.
    ax.scatter(embedding[:, 0], embedding[:, 1], c=[COLORS[int(i)] for i in domain_ids],
               s=marker_size, alpha=alpha, edgecolors="none", rasterized=False)
    ax.set_xlabel("t-SNE 1", labelpad=4)
    ax.set_ylabel("t-SNE 2", labelpad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.035)
    handles = [Line2D([], [], marker="o", ls="none", color=color, markersize=5, label=name)
               for name, color in zip(NAMES, COLORS)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.52, .037), ncol=5, frameon=False,
               fontsize=8.5, handletextpad=.45, columnspacing=1.7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    require(fig.legends[0].get_window_extent(renderer).y1 < ax.xaxis.label.get_window_extent(renderer).y0,
            "Legend overlaps the plot label")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-domain", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the confirmed CPU worker")
    require(args.threads > 0 and args.max_iter >= 300, "Invalid optimization settings")
    args.output.mkdir(parents=True, exist_ok=True)
    require(not (args.output / "summary.json").exists(), "Completed output already exists; use a new output directory")
    started = time.monotonic()
    manifest_path = args.manifest or args.storage_root / "data/processed/domain_split_manifests_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "worker": socket.gethostname(),
                "manifest": str(manifest_path), "storage_root": str(args.storage_root),
                "per_domain": args.per_domain, "seed": args.seed, "pca_components": args.pca_components,
                "perplexity": args.perplexity, "max_iter": args.max_iter, "threads": args.threads,
                "sampling": "Uniform without replacement within each full frozen training-domain view, then interleaved",
                "features": "All raw compact body token counts / full record body length; no TF-IDF or feature standardization",
                "special_tokens": "No PAD/BOS/EOS are added; raw compact body IDs 0,1,2 are valid content and are retained",
                "preprojection": "Centered sparse PCA (ARPACK), without whitening, to 50 dimensions by default",
                "length_cap": None, "domain_labels_used_in_fit": False, "test_accessed": False,
                "duplicates": "Distinct record identities sampled without replacement; duplicate sequences are not removed",
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                             "matplotlib": matplotlib.__version__},
                "reference": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"}
    write_json(args.output / "protocol.json", protocol)
    rows = draw_samples(manifest, args.per_domain, args.seed, RECORDED_STORAGE, args.storage_root)
    frequencies, audit = read_sample_features(manifest, rows, RECORDED_STORAGE, args.storage_root)
    save_npz(args.output / "token_frequencies.npz", frequencies)
    with (args.output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    event("features_complete", shape=frequencies.shape, **audit)
    embedding, reduced, pca, stats = project_frequencies(frequencies, args.pca_components, args.perplexity,
                                                        args.seed, args.max_iter, args.threads)
    np.save(args.output / "embedding.npy", embedding)
    np.save(args.output / "pca_scores.npy", reduced)
    np.savez_compressed(args.output / "pca_model.npz", components=pca.components_, mean=pca.mean_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    for row, point in zip(rows, embedding.tolist()):
        row.update(tsne_1=point[0], tsne_2=point[1])
    with (args.output / "coordinates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = make_figure(embedding, [row["domain_id"] for row in rows])
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"domain_token_frequency_tsne.{suffix}", dpi=450)
    plt.close(fig)
    lengths = {domain: [row["body_tokens"] for row in rows if row["domain"] == domain] for domain in DOMAIN_NAMES}
    summary = {"complete": True, "protocol": protocol, "feature_audit": audit, "projection": stats,
               "records": len(rows), "domain_counts": dict(Counter(row["domain"] for row in rows)),
               "body_length_stats": {domain: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                                             "p99": float(np.quantile(values, .99)), "max": max(values)}
                                     for domain, values in lengths.items()},
               "elapsed_seconds": time.monotonic() - started,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "interpretation": "Exploratory local neighborhoods of token composition; not learned representations, transfer evidence, or global distances"}
    write_json(args.output / "summary.json", summary)
    event("TOKEN_FREQUENCY_TSNE_COMPLETE", output=str(args.output), records=len(rows), **stats)


if __name__ == "__main__":
    main()
