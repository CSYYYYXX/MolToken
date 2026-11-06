#!/usr/bin/env python3
"""Plot exported validation tables without opening checkpoints or test data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analyze_domainwise_adaptive_curves import analyze_adaptive_curve as analyze_curve

DOMAINS = ("small_molecule", "protein", "dna", "rna")
LABELS = {"small_molecule": "Small molecule", "protein": "Protein", "dna": "DNA", "rna": "RNA"}
COLORS = {"accepted": "#15803d", "higher_boundary": "#b91c1c", "lower_boundary": "#b91c1c",
          "sparse_support": "#a16207", "poor_fit": "#7e22ce", "incomplete": "#64748b"}
CAPTIONS = {"accepted": "Bracket supported", "higher_boundary": "Minimum at right edge",
            "lower_boundary": "Minimum at left edge", "sparse_support": "Valley seen; sparse support",
            "poor_fit": "Valley seen; poor quadratic fit", "incomplete": "Incomplete: no fit"}
MARKERS = {256: "o", 384: "s", 512: "^"}
DOMAIN_ZH = {"small_molecule": "小分子", "protein": "蛋白质", "dna": "DNA", "rna": "RNA"}
BUDGET_COLORS = dict(zip((f"C{i}" for i in range(6)),
                         ("#3288ff", "#0891b2", "#f58232", "#9467df", "#dc4265", "#43b96d")))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_snapshot(table_dir):
    completed = read_csv(table_dir / "completed_runs.csv")
    pending = read_csv(table_dir / "pending_runs.csv")
    seen = set()
    for row in completed + pending:
        if row["run_id"] in seen:
            raise ValueError(f"duplicate run: {row['run_id']}")
        seen.add(row["run_id"])
        row["parameter_count"] = int(row["parameter_count"])
        row["target_flops_6nd"] = float(row["target_flops_6nd"])
    for row in completed:
        row["validation_loss"] = float(row["validation_loss"])
        row["hidden_size"] = int(row["hidden_size"])
        if row["metric_split"] != "validation" or row["validation_complete_view"] != "true":
            raise ValueError(f"not a complete validation observation: {row['run_id']}")
        if not math.isfinite(row["validation_loss"]) or row["validation_loss"] <= 0:
            raise ValueError(f"invalid loss: {row['run_id']}")
    return completed, pending


def classify(analysis):
    if analysis["bracketing_status"] == "bracketed":
        return "accepted"
    index = analysis["discrete_argmin"]["index"]
    if index == analysis["valid_run_count"] - 1:
        return "higher_boundary"
    if index == 0:
        return "lower_boundary"
    if not analysis["boundary_evidence"]["fit_meets_r_squared_threshold"]:
        return "poor_fit"
    return "sparse_support"


def analyze_snapshot(completed, pending):
    curves = []
    for domain in DOMAINS:
        for budget in (f"C{i}" for i in range(6)):
            select = lambda row: row["domain"] == domain and row["budget"] == budget
            points = sorted(filter(select, completed), key=lambda row: row["parameter_count"])
            missing = sorted(filter(select, pending), key=lambda row: row["parameter_count"])
            if len(points) + len(missing) < 5:
                raise ValueError(f"expected at least five planned models for {domain}/{budget}")
            budgets = {row["target_flops_6nd"] for row in points + missing}
            if len(budgets) != 1:
                raise ValueError(f"different compute budgets in {domain}/{budget}")
            analysis = analyze_curve(points) if len(points) >= 5 else None
            curves.append({"domain": domain, "budget": budget, "target_flops": budgets.pop(),
                           "points": points, "pending": missing, "analysis": analysis,
                           "visual_category": classify(analysis) if analysis else "incomplete"})
    return curves


def verify_exported_diagnostics(curves, table_dir):
    exported = {(r["domain"], r["budget"]): r for r in read_csv(table_dir / "validation_curve_diagnostics.csv")}
    computed = [c for c in curves if c["analysis"]]
    if set(exported) != {(c["domain"], c["budget"]) for c in computed}:
        raise ValueError("exported and recomputed curve sets differ")
    for curve in computed:
        old = exported[(curve["domain"], curve["budget"])]
        new = curve["analysis"]
        if old["validation_bracketing_status"] != new["bracketing_status"]:
            raise ValueError(f"bracketing decision changed: {curve['domain']}/{curve['budget']}")
        for key in ("r_squared", "rmse", "quadratic_coefficient_a"):
            if not math.isclose(float(old[key]), new["quadratic_fit"][key], rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError(f"recomputed {key} differs for {curve['domain']}/{curve['budget']}")


def curve_panel(ax, curve, compact=False):
    points, missing, analysis = curve["points"], curve["pending"], curve["analysis"]
    category = curve["visual_category"]
    color = COLORS[category]
    all_n = sorted(row["parameter_count"] / 1e6 for row in points + missing)
    ax.set_xscale("log")
    ax.set_xlim(all_n[0] / 1.06, all_n[-1] * 1.06)
    ax.set_xticks(all_n, [f"{n:.2f}" for n in all_n], rotation=25, fontsize=8)
    ax.minorticks_off()
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.18)
    ax.set_title(f"{curve['budget']} | C={curve['target_flops']:.3g} | {len(points)}/{len(points)+len(missing)}", fontsize=10, loc="left")
    if points:
        ns = [row["parameter_count"] / 1e6 for row in points]
        ys = [row["validation_loss"] for row in points]
        ax.plot(ns, ys, color="#94a3b8", lw=0.9, zorder=1)
        for row in points:
            ax.scatter(row["parameter_count"] / 1e6, row["validation_loss"], s=36,
                       color="#0f172a", marker=MARKERS[row["hidden_size"]], zorder=4)
        best = min(points, key=lambda row: row["validation_loss"])
        ax.scatter(best["parameter_count"] / 1e6, best["validation_loss"], marker="*", s=125,
                   facecolor="#f59e0b", edgecolor="#78350f", linewidth=0.5, zorder=5)
        ax.margins(y=0.30)
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No completed observations", ha="center", transform=ax.transAxes, fontsize=10, color="#64748b")
    if missing:
        ax.scatter([row["parameter_count"] / 1e6 for row in missing], [0.045] * len(missing),
                   transform=ax.get_xaxis_transform(), color="#94a3b8", marker="|", s=110, zorder=7)
    detail = CAPTIONS[category]
    if analysis:
        fit, evidence = analysis["quadratic_fit"], analysis["boundary_evidence"]
        xx = np.linspace(math.log10(points[0]["parameter_count"]), math.log10(points[-1]["parameter_count"]), 200)
        yy = fit["quadratic_coefficient_a"] * xx**2 + fit["linear_coefficient_b"] * xx + fit["intercept_c"]
        ax.plot(10**xx / 1e6, yy, "--", color="#2563eb", lw=1.7, zorder=2)
        if evidence["vertex_is_fitted_minimum"] and evidence["vertex_relation"] == "inside_observed_n":
            ax.axvline(fit["vertex_parameter_count"] / 1e6, color=color, lw=1.1, linestyle=":")
            if not compact:
                detail += f"\nFitted candidate {fit['vertex_parameter_count']/1e6:.2f}M; sides {evidence['measured_points_below_vertex']}/{evidence['measured_points_above_vertex']}"
        elif not evidence["vertex_is_fitted_minimum"]:
            detail += "\nConcave: vertex is NOT a minimum"
        else:
            detail += "\nFitted minimum outside measured range"
        detail += f"\n$R^2$={fit['r_squared']:.3f}"
    ax.text(0.03, 0.97, detail, transform=ax.transAxes, va="top", fontsize=7.4 if compact else 8.3,
            color=color, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2})
    for spine in ax.spines.values():
        spine.set_edgecolor(color)


def save_figure(fig, output_dir, name):
    for extension in ("png", "svg", "pdf"):
        target = output_dir / f"{name}.{extension}"
        temporary = target.with_suffix(target.suffix + ".tmp")
        fig.savefig(temporary, format=extension, dpi=180, bbox_inches="tight", facecolor="white")
        os.replace(temporary, target)
    plt.close(fig)


def plot_curves(curves, output_dir, *, snapshot_label="physical-machine snapshot"):
    legend = [Line2D([], [], color="#0f172a", marker="o", lw=0, label="Observed full-validation loss"),
              Line2D([], [], color="#2563eb", ls="--", label="Quadratic in log10(N), at least 5 observed"),
              Line2D([], [], color="#f59e0b", marker="*", lw=0, markersize=11, label="Best observed model"),
              Line2D([], [], color="#94a3b8", marker="|", lw=0, markersize=11, label="Pending N (no assigned loss)")]
    for domain in DOMAINS:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.1))
        for ax, curve in zip(axes.flat, [c for c in curves if c["domain"] == domain]):
            curve_panel(ax, curve)
            ax.set_xlabel("Total parameters / M (log scale)")
            ax.set_ylabel("Validation causal NLL (lower is better)")
        fig.suptitle(f"{LABELS[domain]} | fixed-compute curves | {snapshot_label}", fontsize=17, y=0.995)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.956), ncol=2, fontsize=9, frameon=False)
        fig.text(0.5, 0.007, "Dotted line = fitted candidate, NOT an accepted optimum unless green. Shapes: hidden width 256=circle, 384=square, 512=triangle. Seed=42; validation only.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.035, 1, 0.91), h_pad=3, w_pad=2)
        save_figure(fig, output_dir, f"{domain}_isoflop")
    fig, axes = plt.subplots(4, 6, figsize=(24, 14))
    for ax, curve in zip(axes.flat, curves):
        curve_panel(ax, curve, compact=True)
    for i, domain in enumerate(DOMAINS):
        axes[i, 0].set_ylabel(f"{LABELS[domain]}\nValidation NLL", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Parameters / M (log scale)")
    observed = sum(len(c["points"]) for c in curves)
    planned = observed + sum(len(c["pending"]) for c in curves)
    fitted = sum(c["analysis"] is not None for c in curves)
    accepted = sum(c["visual_category"] == "accepted" for c in curves)
    fig.suptitle(f"24 IsoFLOP curves | {observed} completed / {planned} planned standard-domain runs | {fitted} fitted curves, {accepted} accepted brackets", fontsize=18)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    fig.text(0.5, 0.009, "Independent y scales reveal within-budget curvature; do not compare absolute losses across domains. No fitting across budgets or pending points.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.925), h_pad=2, w_pad=1.5)
    save_figure(fig, output_dir, "all_isoflop_curves")


def plot_peptide(completed, output_dir):
    rows = [row for row in completed if row["domain"] == "peptide"]
    ns = sorted({row["parameter_count"] for row in rows})
    epochs = sorted({float(row["target_epoch_fraction"]) for row in rows})
    lookup = {(row["parameter_count"], float(row["target_epoch_fraction"])): row["validation_loss"] for row in rows}
    values = np.array([[lookup[n, epoch] for epoch in epochs] for n in ns])
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(14, 5.5))
    for j, epoch in enumerate(epochs):
        ax.plot(np.array(ns)/1e6, values[:, j], "o-", label=f"{epoch:g} token-epoch")
    ax.set_xscale("log")
    ax.set_xticks(np.array(ns)/1e6, [f"{n/1e6:.2f}" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Total parameters / M (log scale)")
    ax.set_ylabel("Validation causal NLL")
    ax.set_title("Equal-data slices (NOT equal compute)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    im = heat.imshow(values, cmap="YlGnBu_r", aspect="auto")
    heat.set_xticks(range(len(epochs)), [f"{e:g}" for e in epochs])
    heat.set_yticks(range(len(ns)), [f"{n/1e6:.2f}M" for n in ns])
    heat.set_xlabel("Training data / token-epochs")
    heat.set_ylabel("Total parameters")
    for i in range(len(ns)):
        for j in range(len(epochs)):
            heat.text(j, i, f"{values[i,j]:.4f}", ha="center", va="center", fontsize=10,
                      color="white" if values[i,j] < 7 else "#0f172a")
    heat.scatter(range(len(epochs)), values.argmin(axis=0), s=1600, marker="s", facecolors="none", edgecolors="#f59e0b", linewidths=2)
    fig.colorbar(im, ax=heat, label="Validation NLL", shrink=0.9)
    fig.suptitle("Peptide: 25 independent models | descriptive N x D surface, not IsoFLOP bracketing", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir, "peptide_equal_data")


def format_overlay_axis(ax, ns):
    ax.set_xscale("log")
    low, high = min(ns), max(ns)
    ax.set_xlim(low / 1.07, high * 1.07)
    ticks = np.geomspace(low, high, 5)
    ax.set_xticks(ticks, [f"{n:.3g}" for n in ticks])
    ax.minorticks_off()
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel("总参数量 N（百万，对数刻度）", fontsize=10, labelpad=10)
    ax.set_ylabel("验证交叉熵（nat/token，越低越好）", fontsize=10, labelpad=8)
    ax.tick_params(labelsize=9, color="#d1d5db")
    ax.grid(axis="y", alpha=0.16)
    for spine in ax.spines.values():
        spine.set_color("#e5e7eb")
    ax.margins(y=0.12)


def budget_overlay_panel(ax, curves):
    """One domain, one shared y axis; color identifies a fixed compute budget."""
    domain = curves[0]["domain"]
    ns = [p["parameter_count"] / 1e6 for c in curves for p in c["points"] + c["pending"]]
    count = sum(len(c["points"]) for c in curves)
    handles = []
    for c in sorted(curves, key=lambda item: item["target_flops"]):
        points, analysis = c["points"], c["analysis"]
        color = BUDGET_COLORS[c["budget"]]
        label = f"{c['budget']} · {c['target_flops']:.3g} FLOPs · {len(points)}/{len(points)+len(c['pending'])}"
        style = ("-" if analysis["boundary_evidence"]["fit_meets_r_squared_threshold"] else "--") if analysis else "None"
        handles.append(Line2D([], [], color=color, marker="o", lw=1.8 if analysis else 0,
                              ls=style, markersize=4, label=label, alpha=1 if points else 0.45))
        if not points:
            continue
        observed = ax.scatter([p["parameter_count"] / 1e6 for p in points],
                              [p["validation_loss"] for p in points], color=color, s=30, zorder=4)
        observed.set_gid(f"observed:{c['budget']}")
        # Missing points have no y value. Do not connect incomplete observations.
        if analysis is None:
            continue
        best = min(points, key=lambda p: p["validation_loss"])
        best_marker = ax.scatter(best["parameter_count"] / 1e6, best["validation_loss"], s=95,
                                 facecolors="none", edgecolors=color, linewidths=1.2, zorder=5)
        best_marker.set_gid(f"observed_minimum:{c['budget']}")
        f = analysis["quadratic_fit"]
        xx = np.linspace(math.log10(points[0]["parameter_count"]), math.log10(points[-1]["parameter_count"]), 200)
        yy = f["quadratic_coefficient_a"] * xx**2 + f["linear_coefficient_b"] * xx + f["intercept_c"]
        line, = ax.plot(10**xx / 1e6, yy, color=color, ls=style, lw=1.8, zorder=2)
        line.set_gid(f"quadratic_fit:{c['budget']}")
        if analysis["bracketing_status"] == "bracketed":
            vertex_x = f["vertex_log10_n"]
            vertex_y = f["quadratic_coefficient_a"] * vertex_x**2 + f["linear_coefficient_b"] * vertex_x + f["intercept_c"]
            marker = ax.scatter(10**vertex_x / 1e6, vertex_y, marker="D", s=38,
                                facecolors="white", edgecolors=color, linewidths=1.7, zorder=6)
            marker.set_gid(f"accepted_validation_vertex:{c['budget']}")
    format_overlay_axis(ax, ns)
    planned = count + sum(len(c["pending"]) for c in curves)
    ax.set_title(f"{DOMAIN_ZH[domain]} · {count}/{planned} 点 · 固定 FLOPs", loc="left", fontsize=15, pad=66)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2,
              frameon=False, fontsize=8.6, handlelength=1.8, columnspacing=1.5, borderaxespad=0,
              labelspacing=0.75)


def peptide_overlay_panel(ax, completed):
    rows = [p for p in completed if p["domain"] == "peptide"]
    epochs = sorted({float(p["target_epoch_fraction"]) for p in rows})
    for epoch, color in zip(epochs, ("#3288ff", "#f58232", "#43b96d", "#eb6aab", "#9470ed")):
        points = sorted((p for p in rows if float(p["target_epoch_fraction"]) == epoch), key=lambda p:p["parameter_count"])
        ax.plot([p["parameter_count"]/1e6 for p in points], [p["validation_loss"] for p in points],
                "o-", color=color, markersize=4.7, linewidth=1.25, label=f"{epoch:g} token-epoch")
        best = min(points, key=lambda p:p["validation_loss"])
        ax.scatter(best["parameter_count"]/1e6, best["validation_loss"], s=95,
                   facecolors="none", edgecolors=color, linewidths=1.2, zorder=5)
    format_overlay_axis(ax, [p["parameter_count"]/1e6 for p in rows])
    ax.set_title(f"多肽 · {len(rows)}/25 点 · 固定数据量（非等 FLOPs）", loc="left", fontsize=15, pad=66)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=3, fontsize=9,
              frameon=False, borderaxespad=0, labelspacing=0.75)


def plot_domain_overlays(curves, completed, output_dir, *, snapshot_label="物理机已完成结果快照"):
    with plt.rc_context({"font.family": ["Droid Sans Fallback", "DejaVu Sans"], "axes.unicode_minus": False}):
        fig, axes = plt.subplots(3, 2, figsize=(14.8, 16.4))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.055, top=0.815, hspace=0.69, wspace=0.24)
        for ax, domain in zip(axes.flat, DOMAINS):
            budget_overlay_panel(ax, [c for c in curves if c["domain"] == domain])
        peptide_overlay_panel(axes[2, 0], completed)
        notes = axes[2, 1]
        notes.axis("off")
        notes.text(0.02, 1.14, "如何读这张图", fontsize=15, transform=notes.transAxes)
        notes.text(0.02, 0.99,
                   "同一领域内，各预算共用坐标轴；颜色区分 C0 → C5。\n"
                   "图例列出 FLOPs 与已完成点数，便于看预算升高后的变化。\n\n"
                   "●  实测点：固定全量 validation loss\n"
                   "○  空心圈：已拟合曲线的实测最低值\n"
                   "—  实线：至少五点、全部观测的二次拟合，R² ≥ 0.8\n"
                   "--  虚线：至少五点但二次拟合较差，R² < 0.8\n"
                   "◇  菱形：通过原包夹规则的 validation 候选顶点\n\n"
                   "不足五点仅画散点，不拟合、不标最低、不补造 loss。\n"
                   "实线不代表已包夹：还要检查内部最低与两侧支持。\n"
                   "多肽线只连接等数据量实测点，不是 IsoFLOP 拟合。\n\n"
                   "各领域纵轴独立，不能跨领域比较绝对 NLL。\n"
                   "单 seed=42；不含 test 或正在训练的中间 loss。",
                   fontsize=10.5, linespacing=1.65, va="top", transform=notes.transAxes, color="#374151")
        fig.text(0.075, 0.978, "HoloMol · 分领域 Scaling 曲线", fontsize=24, va="top")
        fig.text(0.075, 0.947, "同领域叠加不同计算预算，观察曲线下移与最低点位置的变化", fontsize=13, color="#374151")
        standard_done = sum(len(c["points"]) for c in curves)
        standard_planned = standard_done + sum(len(c["pending"]) for c in curves)
        peptide_done = sum(p["domain"] == "peptide" for p in completed)
        fig.text(0.075, 0.924, f"{snapshot_label}：{len(completed)}/{standard_planned+peptide_done} 点 · 标准领域 {standard_done}/{standard_planned} · 多肽 {peptide_done} · validation only", fontsize=10.7, color="#6b7280")
        fig.text(0.075, 0.903, "颜色仅标记预算；实线/虚线表示拟合质量，不等于包夹通过。不足五点只画散点；补点后使用全部观测。", fontsize=10.7, color="#6b7280")
        save_figure(fig, output_dir, "domainwise_overlay")
        for domain in DOMAINS:
            fig, ax = plt.subplots(figsize=(10.4, 7.2))
            budget_overlay_panel(ax, [c for c in curves if c["domain"] == domain])
            fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.77)
            fig.text(0.105, 0.055, "点 = 全量 validation；线 = 全部观测的二次拟合（虚线 R² < 0.8）；空心圈 = 已测最低。\n不足五点不拟合、不标最低；菱形仅标通过原规则的 validation 顶点。", fontsize=10, linespacing=1.7, color="#64748b")
            save_figure(fig, output_dir, f"{domain}_overlay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    completed, pending = load_snapshot(args.tables)
    curves = analyze_snapshot(completed, pending)
    verify_exported_diagnostics(curves, args.tables)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.tables.absolute()), "metric_split": "validation", "test_accessed": False,
              "completed_runs": len(completed), "pending_runs": len(pending),
              "diagnostic_counts": dict(Counter(c["visual_category"] for c in curves)), "curves": curves}
    (args.output_dir / "recomputed_diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["domain", "budget", "completed", "visual_category", "source_rule_status", "observed_best_n_m", "fitted_candidate_n_m", "r_squared", "left_support", "right_support", "accepted_n_m"]
    with (args.output_dir / "bracketing_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for c in curves:
            a = c["analysis"]
            row = {"domain": c["domain"], "budget": c["budget"], "completed": len(c["points"]), "visual_category": c["visual_category"]}
            if c["points"]:
                row["observed_best_n_m"] = min(c["points"], key=lambda p:p["validation_loss"])["parameter_count"]/1e6
            if a:
                f,e = a["quadratic_fit"],a["boundary_evidence"]
                row.update(source_rule_status=a["bracketing_status"], r_squared=f["r_squared"],
                           left_support=e["measured_points_below_vertex"], right_support=e["measured_points_above_vertex"])
                if e["vertex_is_fitted_minimum"]:
                    row["fitted_candidate_n_m"] = f["vertex_parameter_count"]/1e6
                if a["bracketing_status"] == "bracketed":
                    row["accepted_n_m"] = f["vertex_parameter_count"]/1e6
            writer.writerow(row)
    plot_curves(curves, args.output_dir)
    plot_peptide(completed, args.output_dir)
    plot_domain_overlays(curves, completed, args.output_dir)
    names = ["domainwise_overlay", *(f"{domain}_overlay" for domain in DOMAINS),
             "all_isoflop_curves", *(f"{domain}_isoflop" for domain in DOMAINS), "peptide_equal_data"]
    html = ['<!doctype html><meta charset="utf-8"><title>HoloMol validation curves</title>',
            '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}p{line-height:1.6}</style>',
            '<h1>HoloMol：已完成 106 点的 validation 曲线</h1>',
            '<p>主视图按领域合并，不同颜色叠加 C0–C5：同一领域共用坐标轴，观察 FLOPs 增加时曲线的变化。空心圈标记已拟合曲线的实测最低；菱形仅表示通过原规则的 validation 候选顶点。实线/虚线是至少五点、全部观测的二次拟合，虚线表示 R²&lt;0.8；不足五点仅画散点、不标最低。多肽单独按数据量分组。</p>',
            '<p>后面的逐预算细节图沿用原样：蓝虚线为二次拟合，金色星标为已观测最低；绿边框=包夹通过，红色=边界最低，黄色=一侧支撑不足，紫色=拟合较差，灰色=缺点。原机静态快照，不混入训练中间 loss。</p>',
            '<p>13 条完整曲线中：2 条包夹通过，6 条边界最低，4 条有谷底但一侧支撑不足，1 条已有谷底但二次拟合较差。多肽是等数据量切片，不用于 IsoFLOP 最优点。</p>']
    for name in names:
        html.append(f'<h2>{name}</h2><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p><a href="{name}.png"><img src="{name}.svg"></a>')
    (args.output_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps({"completed": len(completed), "categories": report["diagnostic_counts"], "output": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
