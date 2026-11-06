#!/usr/bin/env python3
"""New, non-overwriting proportional-record t-SNE; execute on the CPU worker."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import socket
import time

import numpy as np
from scipy.sparse import save_npz
from sklearn import config_context
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import plot_domain_token_frequency_tsne as base
from plot_domain_token_frequency_tsne_full import fit_full_frequencies

OUTPUT = base.WORKSPACE / "HoloMol/doc/results/analysis/token_frequency_tsne_100k_proportional_20260916"


def proportional_quotas(populations, total):
    """Largest-remainder allocation using integers, with stable domain-order ties."""
    base.require(set(populations) == set(base.DOMAIN_NAMES), "Expected the five domain populations")
    base.require(all(isinstance(n, int) and n > 0 for n in populations.values()), "Invalid populations")
    population = sum(populations.values())
    base.require(isinstance(total, int) and 0 < total <= population, "Invalid total sample count")
    quotas = {d: total * populations[d] // population for d in base.DOMAIN_NAMES}
    order = sorted(base.DOMAIN_NAMES, key=lambda d: -(total * populations[d] % population))
    for domain in order[:total - sum(quotas.values())]:
        quotas[domain] += 1
    base.require(sum(quotas.values()) == total and all(quotas[d] <= populations[d] for d in quotas),
                 "Invalid allocated quotas")
    return quotas


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_density_figure(embedding):
    """Same coordinates and all points; bin colors are raw counts, not domain labels."""
    fig, ax = base.plt.subplots(figsize=(7.8, 6.0))
    fig.subplots_adjust(left=.055, right=.89, bottom=.13, top=.985)
    artist = ax.hexbin(embedding[:, 0], embedding[:, 1], gridsize=150, mincnt=1,
                       bins="log", cmap="viridis", linewidths=0)
    base.require(int(artist.get_array().sum()) == len(embedding), "Density bins lost observations")
    colorbar = fig.colorbar(artist, ax=ax, fraction=.046, pad=.04)
    colorbar.set_label("Samples per bin")
    ax.set_xlabel("t-SNE 1", labelpad=4)
    ax.set_ylabel("t-SNE 2", labelpad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.035)
    return fig


def write_notes(output, summary):
    protocol = summary["protocol"]
    table = "\n".join(
        f"| {name} | {protocol['training_populations'][domain]:,} | "
        f"{100 * protocol['training_populations'][domain] / protocol['training_population_total']:.4f}% | "
        f"{summary['domain_counts'][domain]:,} |"
        for domain, name in zip(base.DOMAIN_NAMES, base.NAMES))
    (output / "README.md").write_text(
        f"# 五域 token 频率 t-SNE：按记录比例抽样 {summary['records']:,} 条\n\n"
        "独立新增结果；此前每域 2,000 条的 10,000 条图及其数据未覆盖。\n\n"
        "## 抽样\n\n"
        "从冻结 compact 训练集按各域记录数分层无放回抽样，最大余数法取整，合计精确匹配请求数量。"
        "不是按 token 总量加权；不读取验证/测试集，不补齐稀有域，不剔除重复序列。"
        "种子 42；各域内随机抽样后全局随机交错，避免某域总被画在最上层。\n\n"
        "| Domain | 训练记录数 | 训练记录占比 | 实际抽样数 |\n"
        "|---|---:|---:|---:|\n" + table + "\n\n"
        "多肽占比很小，因此点少是原始组成的体现，不额外过采样或放大其标记。\n\n"
        "## 特征与降维\n\n"
        "每条样本使用全部 BPE body token 的计数除以该条 body 长度，无截断、TF-IDF 或标准化。"
        "原始 ID 0/1/2 是内容 token，保留；不添加 PAD/BOS/EOS。"
        f"邻域使用全部 {summary['feature_audit']['vocabulary_columns']:,} 个词表维度的欧氏距离。"
        "PCA 的两个坐标仅用于初始化，不用于计算高维相似度。\n\n"
        f"t-SNE：perplexity={protocol['perplexity']}, seed={protocol['seed']}, "
        f"最多 {protocol['max_iter']} 次迭代，Barnes–Hut angle=0.5，early exaggeration=12，"
        f"learning_rate=auto（实际 {summary['projection']['learning_rate']:.6g}）。"
        "参数沿用前次，不按分群好看程度筛选；域标签不输入降维。\n\n"
        "## 图片及复现文件\n\n"
        "- `domain_token_frequency_tsne.{png,pdf,svg}`：全部抽样点的透明散点图；底部为域图例。\n"
        "- `domain_token_frequency_tsne_density.{png,pdf,svg}`：同一组坐标的六边形计数图，"
        "全部点进入统计，颜色表示每格原始样本数，对数色标；不是域颜色，也不是原始空间概率密度。\n"
        "- `samples.csv`：逐条 domain/shard/raw_index/完整长度。\n"
        "- `coordinates.csv`, `embedding.npy`：同一行序的全部二维坐标。\n"
        "- `token_frequencies.npz`：全部稀疏频率特征；`pca_scores.npy`, `pca_model.npz`：初始化来源。\n"
        "- `protocol.json`, `summary.json`：抽样配额、参数、完整性检查、环境及执行统计。\n\n"
        "运行于 tmux 43 的用户分配 CPU worker，使用现有环境，无联网安装、GPU 或新资源申请。"
        "日志位于 `molscaling/logs/token_frequency_tsne_20260916/proportional_100k.log`。\n\n"
        "该图仅描述 token 组成的探索性局部邻域，不是模型 embedding、跨域迁移证据或"
        "可靠的全局距离/密度估计。点密集时有遮挡；密度视图用于定位重叠，不能恢复高维密度。"
        "本次重新拟合 t-SNE，不能直接把坐标与此前 10,000 条图逐点对齐。\n",
        encoding="utf-8")
    (output / "captions.tex").write_text(
        "% Scatter caption\n"
        f"Token-composition visualization of {summary['records']:,} training records, sampled "
        "without replacement in proportion to domain-level record counts. Each record is "
        "represented by its full length-normalized BPE token-frequency vector. t-SNE uses "
        "Euclidean affinities in the full vocabulary space; PCA is used for initialization "
        "only. Colors indicate domains; all sampled records are shown. "
        "The visualization is exploratory and does not establish transfer or global distances.\n\n"
        "% Density companion caption\n"
        "Hexagonal bin counts of the same t-SNE coordinates, with a logarithmic color scale. "
        "All sampled records contribute; colors encode raw sample counts, not domain labels "
        "or density in the original feature space.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--total", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--working-memory-mib", type=int, default=512)
    args = parser.parse_args()
    base.require("worker" in args.expected_host and socket.gethostname() == args.expected_host,
                 "Use the confirmed allocated CPU worker")
    base.require(args.threads > 0 and args.max_iter >= 300 and args.working_memory_mib > 0
                 and 0 < args.perplexity < args.total, "Invalid optimization settings")
    # Refuse ANY existing output directory, including previous unfinished results.
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest_path = args.manifest or args.storage_root / "data/processed/domain_split_manifests_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    populations = {d: int(manifest["views"][d]["train"]["records"]) for d in base.DOMAIN_NAMES}
    quotas = proportional_quotas(populations, args.total)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(), "worker": socket.gethostname(),
        "manifest": str(manifest_path), "storage_root": str(args.storage_root),
        "training_populations": populations, "training_population_total": sum(populations.values()),
        "domain_quotas": quotas, "total_records": args.total,
        "sampling": "Record-proportional, largest remainder, uniform without replacement within frozen training domains",
        "seed": args.seed, "perplexity": args.perplexity, "max_iter": args.max_iter, "threads": args.threads,
        "working_memory_mib": args.working_memory_mib,
        "features": "All compact body token counts / complete body length; no TF-IDF, standardization or truncation",
        "raw_body_ids_0_1_2_retained": True, "added_special_tokens": False,
        "affinities": "Full sparse frequency vectors, Euclidean; exact nearest neighbors, Barnes-Hut embedding",
        "pca_preprojection": False, "initialization": "Centered sparse ARPACK PCA2, first-axis std scaled to 1e-4",
        "domain_labels_used_in_fit": False, "test_accessed": False, "validation_accessed": False,
        "duplicate_sequences": "Retained; record identities must be unique",
        "scatter": {"marker_size": 1.6, "alpha": .32, "domain_overdraw": "Random interleaving"},
        "density": {"gridsize": 150, "count_scale": "log", "domain_reweighting": False},
        "versions": {"numpy": base.np.__version__, "scipy": base.scipy.__version__,
                     "sklearn": base.sklearn.__version__, "matplotlib": base.matplotlib.__version__},
    }
    base.write_json(args.output / "protocol.json", protocol)
    base.event("proportional_sampling_plan", populations=populations, quotas=quotas, total=args.total)
    rows = base.draw_samples(manifest, quotas, args.seed, base.RECORDED_STORAGE, args.storage_root)
    frequencies, audit = base.read_sample_features(manifest, rows, base.RECORDED_STORAGE, args.storage_root)
    base.require(len(rows) == args.total and frequencies.shape[0] == args.total, "Wrong total record count")
    save_npz(args.output / "token_frequencies.npz", frequencies)
    write_csv(args.output / "samples.csv", rows)
    base.event("features_complete", shape=frequencies.shape, **audit)
    with threadpool_limits(limits=args.threads), config_context(working_memory=args.working_memory_mib):
        pca = PCA(n_components=2, svd_solver="arpack", random_state=args.seed)
        scores = pca.fit_transform(frequencies)
        np.save(args.output / "pca_scores.npy", scores)
        np.savez_compressed(args.output / "pca_model.npz", components=pca.components_, mean=pca.mean_,
                            explained_variance_ratio=pca.explained_variance_ratio_)
        base.event("pca_initialization_complete", explained_variance=float(pca.explained_variance_ratio_.sum()))
        embedding, stats = fit_full_frequencies(frequencies, scores, perplexity=args.perplexity,
                                                seed=args.seed, iterations=args.max_iter, threads=args.threads)
    np.save(args.output / "embedding.npy", embedding)
    for row, point in zip(rows, embedding.tolist()):
        row.update(tsne_1=point[0], tsne_2=point[1])
    write_csv(args.output / "coordinates.csv", rows)
    fig = base.make_figure(embedding, [row["domain_id"] for row in rows], marker_size=1.6, alpha=.32)
    base.require(len(fig.axes[0].collections[0].get_offsets()) == args.total, "Scatter lost observations")
    for stem, figure in (("domain_token_frequency_tsne", fig),
                         ("domain_token_frequency_tsne_density", make_density_figure(embedding))):
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(args.output / f"{stem}.{suffix}", dpi=450)
        base.plt.close(figure)
    lengths = {d: [row["body_tokens"] for row in rows if row["domain"] == d] for d in base.DOMAIN_NAMES}
    summary = {
        "complete": True, "protocol": protocol, "feature_audit": audit, "projection": stats,
        "records": len(rows), "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "body_length_stats": {d: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                  "p99": float(np.quantile(v, .99)), "max": max(v)} for d, v in lengths.items() if v},
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "interpretation": "Exploratory token-composition neighborhoods, not model representations or transfer evidence",
    }
    write_notes(args.output, summary)
    base.write_json(args.output / "summary.json", summary)
    base.event("PROPORTIONAL_TOKEN_FREQUENCY_TSNE_COMPLETE", output=str(args.output), records=len(rows), **stats)


if __name__ == "__main__":
    main()
