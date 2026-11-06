#!/usr/bin/env python3
"""Presentation-only 5-column IsoFLOP + N*/D*/L* multi-to-single figure.

Reuse the saved 65-run domain observations and fits. No checkpoints or logs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

from plot_single_domain_publication import (
    ROOT, PAPER, DOMAINS, FRONTIER_COLOR, require, close, style, log_ticks, export, write_csv,
)

SOURCE = ROOT / "scaling-csy/doc/results/multi_to_single_scaling_20260906"
OUTPUT = PAPER / "figures/multi_to_single_publication_20260915"
EXCLUDED = "excluded_user_reported_mixture_issue"
FIELDS = ("N_opt", "D_opt", "L_opt")


def load_inputs(source=SOURCE):
    with (source / "domain_losses.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("compute", "n", "loss"):
            row[field] = float(row[field])
    curves = json.loads((source / "fit_summary.json").read_text())
    frontiers = json.loads((source / "three_fits_summary.json").read_text())
    return rows, curves, frontiers


def validate(rows, saved, frontiers):
    require(saved["fit_method"] == "all_points_quadratic_in_log10_n", "Unexpected source fit method")
    require(set(row["domain"] for row in rows) == set(DOMAINS), "Domain set changed")
    keys = [(row["domain"], row["target_label"], row["model_id"]) for row in rows]
    require(len(set(keys)) == len(keys), "Duplicate observations")
    run_keys = {d: {(r["target_label"], r["model_id"]) for r in rows if r["domain"] == d} for d in DOMAINS}
    require(all(value == run_keys[DOMAINS[0]] for value in run_keys.values()), "Domain run sets differ")
    require(len(run_keys[DOMAINS[0]]) == saved["runs"], "Run count disagrees with saved fit")
    curves = {(c["domain"], c["compute"]): c for c in saved["curves"]}
    require(len(curves) == len(saved["curves"]), "Duplicate curves")
    require(set(curves) == {(r["domain"], r["compute"]) for r in rows}, "Missing/extra curve")
    fit_by_domain = {f["domain"]: f for f in frontiers}
    require(len(fit_by_domain) == len(frontiers) == len(DOMAINS), "Frontier domain count differs")
    require(set(fit_by_domain) == set(DOMAINS), "Frontier domains differ")
    omitted, displayed, noneligible = [], [], []
    for key, curve in curves.items():
        selected = sorted((r for r in rows if (r["domain"], r["compute"]) == key), key=lambda r: r["n"])
        require(len(selected) == curve["points"], f"Curve count differs: {key}")
        n = np.array([r["n"] for r in selected])
        loss = np.array([r["loss"] for r in selected])
        require(np.isfinite(n).all() and np.isfinite(loss).all() and np.all(n > 0) and np.all(loss > 0),
                f"Invalid observation: {key}")
        if curve["status"] == EXCLUDED:
            require(key[0] == "peptide" and curve["fit_points"] == 0 and curve["n_opt"] is None,
                    f"Unrecognized historical exclusion: {key}")
            omitted.extend(selected)
            continue
        require(curve["fit_points"] == len(selected), f"Saved fit is not all-point: {key}")
        x = np.log10(n)
        coefficients = np.polyfit(x, loss, 2)
        require(np.allclose(coefficients, curve["coefficients"], rtol=1e-9, atol=1e-9),
                f"Saved quadratic coefficients disagree with observations: {key}")
        a, b, _ = curve["coefficients"]
        vertex_x = -b / (2 * a)
        require(close(vertex_x, curve["fitted_vertex_log10_n"]), f"Vertex position differs: {key}")
        interior = a > 0 and x.min() < vertex_x < x.max() and 0 < np.argmin(loss) < len(loss) - 1
        require((curve["status"] == "interior") == interior, f"Original eligibility changed: {key}")
        if interior:
            require(close(curve["n_opt"], 10**vertex_x), f"N optimum changed: {key}")
        else:
            require(curve["status"] in ("unbracketed", "nonconvex") and curve["n_opt"] is None,
                    f"Unexpected noneligible curve: {key}")
            noneligible.append({"domain": key[0], "C": key[1], "status": curve["status"]})
        displayed.extend(selected)
    for domain, frontier in fit_by_domain.items():
        eligible = {c["compute"]: c for c in saved["curves"] if c["domain"] == domain and c["status"] == "interior"}
        vertices = frontier["vertices"]
        require(len(vertices) == len(eligible) == frontier["vertex_count"], f"Vertex count differs: {domain}")
        require({v["C"] for v in vertices} == set(eligible), f"Frontier uses other budgets: {domain}")
        for vertex in vertices:
            curve = eligible[vertex["C"]]
            require(close(vertex["N_opt"], curve["n_opt"]), "Frontier N differs")
            require(close(vertex["D_opt"], vertex["C"] / (6 * vertex["N_opt"])), "Mixed-token D identity fails")
            require(close(vertex["L_opt"], np.polyval(curve["coefficients"], np.log10(vertex["N_opt"]))),
                    "Frontier loss differs")
        x = np.log10([v["C"] for v in vertices])
        for field in FIELDS:
            y = np.array([v[field] for v in vertices])
            if field != "L_opt":
                y = np.log10(y)
            fit = frontier[field]
            slope, intercept = np.polyfit(x, y, 1)
            require(close(slope, fit["slope"]) and close(intercept, fit["intercept"]),
                    f"Saved linear fit disagrees with vertices: {domain}/{field}")
            pred = fit["slope"] * x + fit["intercept"]
            r2 = 1 - float(np.sum((y - pred)**2) / np.sum((y - y.mean())**2))
            require(close(r2, fit["r2"]), f"R2 differs: {domain}/{field}")
        require(close(frontier["N_opt"]["slope"] + frontier["D_opt"]["slope"], 1), "Alpha+beta identity fails")
    return {"runs": saved["runs"], "source_observations": len(rows), "displayed_observations": len(displayed),
            "displayed_by_domain": dict(Counter(r["domain"] for r in displayed)),
            "historically_omitted_observations": len(omitted),
            "historically_omitted_budgets": sorted({r["compute"] for r in omitted}),
            "displayed_curves": sum(c["status"] != EXCLUDED for c in saved["curves"]),
            "eligible_vertices": sum(f["vertex_count"] for f in frontiers),
            "noneligible_curves": noneligible, "new_exclusions": 0,
            "saved_coefficients_used_unchanged": True, "checkpoints_or_training_logs_loaded": False,
            "test_accessed": False, "hash_gates": False}


def frontier_values(frontier, field, compute):
    fit = frontier[field]
    value = fit["slope"] * np.log10(compute) + fit["intercept"]
    return value if field == "L_opt" else 10**value


def isoflop_panel(ax, domain, rows, saved, colors):
    curves = sorted((c for c in saved["curves"] if c["domain"] == domain and c["status"] != EXCLUDED),
                    key=lambda c: c["compute"])
    all_loss = []
    for curve in curves:
        selected = sorted((r for r in rows if r["domain"] == domain and r["compute"] == curve["compute"]),
                          key=lambda r: r["n"])
        n = np.array([r["n"] for r in selected])
        loss = np.array([r["loss"] for r in selected])
        color = colors[curve["compute"]]
        grid = np.geomspace(n.min(), n.max(), 256)
        y = np.polyval(curve["coefficients"], np.log10(grid))
        all_loss.extend(loss)
        all_loss.extend(y)
        ax.plot(grid / 1e6, y, color=color, lw=1.25,
                ls="-" if curve["status"] == "interior" else ":", zorder=2,
                gid=f"isoflop-fit-{domain}-{curve['compute']}")
        ax.scatter(n / 1e6, loss, s=13, color=color, edgecolors="white", linewidths=.3, zorder=4,
                   gid=f"observations-{domain}-{curve['compute']}")
        if curve["status"] == "interior":
            nn = curve["n_opt"]
            ax.scatter(nn / 1e6, np.polyval(curve["coefficients"], np.log10(nn)), s=28,
                       marker="D", facecolors=color, edgecolors=color, linewidths=.65, zorder=5)
        else:
            nn = 10**curve["fitted_vertex_log10_n"]
            if n.min() < nn < n.max():
                ax.scatter(nn / 1e6, np.polyval(curve["coefficients"], np.log10(nn)), s=24,
                           marker="x", color=color, linewidths=1.0, zorder=5)
    ax.set_xscale("log")
    ax.margins(x=.045, y=.08)
    log_ticks(ax, "x")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if domain == "peptide":
        require(min(all_loss) > 4.65 and max(all_loss) < 5.0, "Original peptide display would clip retained data")
        ax.set_ylim(4.65, 5.0)
        ax.yaxis.set_major_locator(MultipleLocator(.10))
    ax.grid(True, which="major")
    ax.set_xlabel(r"Parameters, $N$ (M)", fontsize=9.1, labelpad=4)
    return curves


def frontier_panel(ax, frontier, field, *, final_row):
    vertices = frontier["vertices"]
    c = np.array([v["C"] for v in vertices])
    factor = {"N_opt": 1e6, "D_opt": 1e9, "L_opt": 1}[field]
    low, high = c.min(), c.max()
    limits = (low / 1.15, high * 1.15)
    grid = np.geomspace(low, high, 256)
    ax.plot(grid / 1e18, frontier_values(frontier, field, grid) / factor, color=FRONTIER_COLOR, lw=1.5,
            gid=f"frontier-fit-{frontier['domain']}-{field}")
    for a, b in ((limits[0], low), (high, limits[1])):
        gx = np.geomspace(a, b, 30)
        ax.plot(gx / 1e18, frontier_values(frontier, field, gx) / factor, color=FRONTIER_COLOR,
                lw=1.0, linestyle=(0, (3, 2)), alpha=.75)
    ax.scatter(c / 1e18, np.array([v[field] for v in vertices]) / factor, s=23, color=FRONTIER_COLOR,
               edgecolors="white", linewidths=.35, zorder=4, gid=f"frontier-vertices-{frontier['domain']}-{field}")
    ax.set_xscale("log")
    ax.set_xlim(*(np.array(limits) / 1e18))
    if field != "L_opt":
        ax.set_yscale("log")
    ax.margins(y=.12)
    log_ticks(ax, "x")
    if field == "L_opt":
        ax.yaxis.set_major_locator(MaxNLocator(4))
    else:
        log_ticks(ax, "y")
    ax.grid(True, which="major")
    if final_row:
        ax.set_xlabel(r"Compute, $C$ ($10^{18}$ FLOPs)", fontsize=9.0, labelpad=4)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    symbol = {"N_opt": r"\alpha", "D_opt": r"\beta", "L_opt": "s"}[field]
    fit = frontier[field]
    note = ax.text(.5, -.45 if final_row else -.16,
                   rf"${symbol}={fit['slope']:.3f}$   $R^2={fit['r2']:.3f}$",
                   transform=ax.transAxes, ha="center", va="center", fontsize=9.2,
                   color="#40515D", clip_on=False)
    return note


def combined_figure(rows, saved, frontiers):
    style()
    plt.rcParams.update({"xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
    fig = plt.figure(figsize=(12.8, 7.8))
    grid = fig.add_gridspec(7, 5, left=.065, right=.99, bottom=.095, top=.975,
                           height_ratios=[1.45, 1.15, 1, .34, 1, .34, 1], hspace=0, wspace=.33)
    axes = np.empty((4, 5), dtype=object)
    budgets = sorted({c["compute"] for c in saved["curves"]})
    colors = dict(zip(budgets, plt.cm.viridis(np.linspace(.06, .90, len(budgets)))))
    frontiers = {f["domain"]: f for f in frontiers}
    legends, annotations = [], []
    for col, domain in enumerate(DOMAINS):
        ax = fig.add_subplot(grid[0, col])
        axes[0, col] = ax
        curves = isoflop_panel(ax, domain, rows, saved, colors)
        if col == 0:
            ax.set_ylabel("Loss", fontsize=10.5)
        legend_ax = fig.add_subplot(grid[1, col])
        legend_ax.axis("off")
        handles = [Line2D([], [], color=colors[c["compute"]], lw=1.2, marker="o", markersize=3,
                          label=f"{c['compute'] / 1e18:g}",
                          linestyle="-" if c["status"] == "interior" else ":") for c in curves]
        # Matplotlib fills columns; transpose to read budgets left-to-right in rows of four.
        handles = [h for offset in range(4) for h in handles[offset::4]]
        legend = legend_ax.legend(handles=handles, title=r"$C$ ($10^{18}$ FLOPs)", ncol=4,
                                 loc="upper center", bbox_to_anchor=(.5, .58), frameon=False,
                                 fontsize=8, title_fontsize=8.5, handlelength=1.0,
                                 handletextpad=.3, columnspacing=.75, labelspacing=.25, borderaxespad=0)
        legends.append(legend)
        for row, (field, grid_row) in enumerate(zip(FIELDS, (2, 4, 6)), 1):
            ax = fig.add_subplot(grid[grid_row, col], sharex=axes[1, col] if row > 1 else None)
            axes[row, col] = ax
            annotations.append(frontier_panel(ax, frontiers[domain], field, final_row=row == 3))
            if col == 0:
                ax.set_ylabel({"N_opt": r"$N^*$ (M)", "D_opt": r"$D^*$ (B tokens)", "L_opt": r"Loss, $L^*$"}[field], fontsize=10.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for col, legend in enumerate(legends):
        box = legend.get_window_extent(renderer).padded(2)
        label_box = axes[0, col].xaxis.label.get_window_extent(renderer)
        require(not box.overlaps(label_box),
                f"Budget legend overlaps N label: {DOMAINS[col]}, legend={box.bounds}, label={label_box.bounds}")
        require(box.y0 > axes[1, col].get_window_extent(renderer).y1,
                f"Budget legend overlaps frontier: {DOMAINS[col]}, legend={box.bounds}, frontier_top={axes[1, col].get_window_extent(renderer).y1}")
        if col:
            require(not box.overlaps(legends[col - 1].get_window_extent(renderer).padded(2)),
                    f"Adjacent budget legends overlap: {DOMAINS[col - 1]}/{DOMAINS[col]}")
    data_boxes = [ax.get_window_extent(renderer) for ax in axes.flat]
    for annotation in annotations:
        box = annotation.get_window_extent(renderer).padded(2)
        require(not any(box.overlaps(data) for data in data_boxes), "Coefficient annotation covers plotted data")
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the confirmed CPU worker")
    rows, saved, frontiers = load_inputs(args.source)
    checks = validate(rows, saved, frontiers)
    fig, _ = combined_figure(rows, saved, frontiers)
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = export(fig, args.output / "multi_to_single_all_domains", args.dpi)
    exclusions = {(c["domain"], c["compute"]) for c in saved["curves"] if c["status"] == EXCLUDED}
    write_csv(args.output / "observations.csv", [{**r, "displayed": (r["domain"], r["compute"]) not in exclusions} for r in rows])
    write_csv(args.output / "vertices.csv", [{"domain": f["domain"], **v} for f in frontiers for v in f["vertices"]])
    write_csv(args.output / "coefficients.csv", [{"domain": f["domain"], "vertices": f["vertex_count"],
              **{f"{field}_{term}": f[field][term] for field in FIELDS for term in ("slope", "intercept", "r2")}} for f in frontiers])
    (args.output / "source_fits.json").write_text(json.dumps({"isoflop": saved, "frontiers": frontiers}, indent=2) + "\n")
    metadata = {"rendered_at": datetime.now(timezone.utc).isoformat(), "worker": socket.gethostname(),
                "source": str(args.source), "validation": checks, "artifacts": artifacts,
                "metric": "Per-domain training-loss proxy: sum of domain_loss_sum / sum of domain_loss_tokens across last 10 train log windows; no independent test evaluation",
                "D_definition": "D*=C/(6N*): total mixed-training token budget, not target-domain token count or unique dataset size",
                "L_definition": "L*=s*log10(C)+b; linear loss, logarithmic compute; not a log-log loss power law",
                "fit_policy": "Render saved all-point quadratics and saved eligible-vertex frontiers; semantic OLS checks do not replace coefficients",
                "layout": "Compact 12.8 x 7.8 inch canvas. Five domains horizontally; IsoFLOP, N*, D*, L* vertically. Each budget legend reads across four entries per row. Coefficients stay outside plot interiors. No subtitles, global legend or footnote; explanations are in the companion caption.",
                "column_order": list(DOMAINS),
                "exclusion_note": "Historical post-hoc peptide omission attributed by user to mixture issues; not independently verified. No new data exclusions."}
    (args.output / "figure_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("MULTI_TO_SINGLE_PUBLICATION_COMPLETE", args.output, json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
