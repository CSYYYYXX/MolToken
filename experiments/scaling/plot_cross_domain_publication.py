#!/usr/bin/env python3
"""Presentation-only exports of the two frozen cross-domain scaling figures.

Run on an allocated CPU worker. Read scalar JSON only; do not fit, evaluate,
scan checkpoints, or update the original result publishers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator, NullLocator
import numpy as np

from compare_domain_scaling_frontiers import extrapolation_segments
from plot_single_domain_publication import (
    COLORS as BUDGET_PALETTE, DOMAINS, NAMES, ROOT, PAPER,
    budget_label, close, export, frontier_values, log_ticks, require, style,
)


FRONTIER_SOURCE = ROOT / "HoloMol/doc/results/analysis/scaling_frontier_comparison_20260910"
CROSS_SOURCE = ROOT / "HoloMol/doc/results/single_to_multi_scaling_20260909"
PLAN = Path("./storage/molscaling/runs/domainwise_cross_domain_eval_20260909/plan.json")
OUTPUT = PAPER / "figures/cross_domain_publication_20260914"
GROUPS = ("single", "multi", "target_normalized")
GROUP_COLORS = dict(zip(GROUPS, ("#2166AC", "#D95F02", "#16805D")))
GROUP_LABELS = {
    "single": r"Single $\rightarrow$ single",
    "multi": r"Multi $\rightarrow$ single: total $C$",
    "target_normalized": r"Multi $\rightarrow$ single: target $rC$",
}
EXPECTED_VERTICES = dict(zip(DOMAINS, ((6, 10, 10), (3, 5, 5), (6, 7, 7), (6, 10, 10), (6, 10, 10))))
EXPECTED_EVALUATIONS = dict(zip(DOMAINS, (32, 35, 24, 14, 31)))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_normalization(record):
    """Target accounting changes C and D, never model size N or loss L."""
    ratio = record["ratio"]
    require(0 < ratio < 1, "Invalid target token ratio")
    multi, target = record["multi"], record["target_normalized"]
    source = {v["budget"]: v for v in multi["vertices"]}
    require(len(source) == len(multi["vertices"]), "Duplicate multi vertex budgets")
    require(set(source) == {v["budget"] for v in target["vertices"]}
            and len(source) == len(target["vertices"]), "Target vertex selection differs")
    for v in target["vertices"]:
        original = source[v["budget"]]
        for field in ("C", "D", "N", "L"):
            expected = original[field] * (ratio if field in ("C", "D") else 1)
            require(close(v[field], expected), f"Target normalization changed {field}")
        require(v["strong_support"] == original["strong_support"], "Target support label differs")
    for field in ("N", "D", "L"):
        require(close(multi["fits"][field]["slope"], target["fits"][field]["slope"]),
                f"Target frontier slope differs: {field}")
        c = np.geomspace(*multi["C_range"], 5)
        expected = frontier_values(multi, field, c) * (ratio if field == "D" else 1)
        require(np.allclose(frontier_values(target, field, ratio * c), expected, rtol=1e-9, atol=1e-9),
                f"Target fitted line changes {field} incorrectly")


def load_frontiers():
    source = read(FRONTIER_SOURCE / "summary.json")["analyses"]["recorded_objectives"]
    rendering = read(FRONTIER_SOURCE / "plot_rendering.json")
    sensitivity = read(FRONTIER_SOURCE / "protein_point_exclusion_refit.json")
    budget = sensitivity["excluded_budget"]
    require(rendering["refitted"] is True and budget == rendering["fit_excluded_budget"] == "fx3p5e18",
            "The supplied figure is the saved protein point-exclusion sensitivity")
    require(not rendering["display_exclusions"], "Unexpected additional display-only exclusions")
    protein = sensitivity["analyses"]["recorded_objectives"]
    require(protein["single"] == source["protein"]["single"], "Protein single frontier changed")
    require(close(protein["ratio"], source["protein"]["ratio"]), "Protein ratio changed")
    for group in ("multi", "target_normalized"):
        original = source["protein"][group]["vertices"]
        retained = [v for v in original if v["budget"] != budget]
        require(len(original) == len(retained) + 1 and protein[group]["vertices"] == retained,
                f"Unexpected protein sensitivity point selection: {group}")
    # The exact saved refit is loaded, not recomputed for a presentation change.
    records = {**source, "protein": protein}
    counts = {}
    for domain in DOMAINS:
        record = records[domain]
        counts[domain] = {}
        for group, expected in zip(GROUPS, EXPECTED_VERTICES[domain]):
            frontier = record[group]
            vertices = frontier["vertices"]
            require(len(vertices) == expected, f"Unexpected vertex count: {domain}/{group}")
            require(frontier["C_range"] == [min(v["C"] for v in vertices), max(v["C"] for v in vertices)],
                    f"Supported C range differs: {domain}/{group}")
            require(all(math.isfinite(v[f]) and v[f] > 0 for v in vertices for f in ("C", "N", "D", "L")),
                    f"Nonpositive/nonfinite vertex: {domain}/{group}")
            counts[domain][group] = {
                "vertices": len(vertices), "weak": sum(not v["strong_support"] for v in vertices),
            }
        validate_normalization(record)
    return records, {
        "counts": counts, "protein_excluded_budget": budget,
        "protein_exclusion_is_preexisting_sensitivity": True,
        "new_fits": False, "new_exclusions": False,
        "target_accounting": "(C, N, D, L) -> (r*C, N, r*D, L)",
    }


def validate_cross(summary, diagnostics, plan, expected_counts=EXPECTED_EVALUATIONS):
    rows = summary["rows"]
    by_id = {r["run_id"]: r for r in rows}
    require(len(by_id) == len(rows), "Duplicate evaluated run IDs")
    require(dict(Counter(r["train_domain"] for r in rows)) == expected_counts,
            "The frozen domain evaluation counts changed")
    require(summary["plot_style"] == "observed_polyline_no_fit", "Expected unfitted polylines")
    require(summary["completed"] == summary["eligible"] == len(rows), "Incomplete local snapshot")
    require(plan["metric_split"] == "validation" and plan["test_accessed"] is False
            and plan["context_length"] == 512, "Cross-domain protocol changed")
    planned = {r["run_id"]: r for r in plan["rows"]}
    require(len(planned) == len(plan["rows"]) and set(planned) == set(by_id), "Frozen plan/evaluation IDs differ")
    unavailable = {r["run_id"] for r in plan["unavailable"]}
    require(not unavailable.intersection(by_id) and len(plan["unavailable"]) == summary["off_host_unavailable"],
            "Unavailable checkpoints were added or availability changed")
    series, seen = {}, []
    for domain in DOMAINS:
        series[domain] = []
        for line in diagnostics[domain]:
            if line["metric"] != "all_domain_loss":
                continue  # Other target metrics belong to the separate original six-panel pages.
            require(line["style"] == "observed_polyline_no_fit", "A source curve is not a polyline")
            points = sorted((r for r in rows if r["train_domain"] == domain and r["budget"] == line["budget"]),
                            key=lambda r: (r["parameter_count"], r["run_id"]))
            require(line["point_ids"] == [r["run_id"] for r in points] and bool(points),
                    f"Saved line point membership/order changed: {domain}/{line['budget']}")
            for r in points:
                p = planned[r["run_id"]]
                require(r["train_domain"] == p["domain"] and r["budget"] == p["budget"]
                        and close(r["target_flops"], p["target_flops"])
                        and r["parameter_count"] == p["parameter_count"], "Evaluated point differs from plan")
                require(math.isfinite(r["all_domain_loss"]) and r["all_domain_loss"] > 0, "Invalid measured loss")
            series[domain].append({"budget": line["budget"], "points": points})
            seen.extend(line["point_ids"])
    require(len(seen) == len(rows) and set(seen) == set(by_id), "Some measurements are omitted or repeated")
    return series


def compute_ticks(ax):
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(NullLocator())


def vertices(ax, frontier, group, xfield, yfield, xscale=1, yscale=1):
    for strong in (True, False):
        selected = [v for v in frontier["vertices"] if v["strong_support"] == strong]
        if selected:
            ax.scatter([v[xfield] / xscale for v in selected], [v[yfield] / yscale for v in selected],
                       s=15, edgecolors=GROUP_COLORS[group], linewidths=.75,
                       facecolors=GROUP_COLORS[group] if strong else "white", zorder=4)


def frontier_panel(ax, record, field):
    scale = {"N": 1e6, "D": 1e9, "L": 1}[field]
    for group in GROUPS:
        frontier = record[group]
        vertices(ax, frontier, group, "C", field, yscale=scale)
        grid = np.geomspace(*frontier["C_range"], 150)
        ax.plot(grid, frontier_values(frontier, field, grid) / scale,
                color=GROUP_COLORS[group], lw=1.35, zorder=3)
    ax.set_xscale("log")
    if field != "L":
        ax.set_yscale("log")
    # The data-supported viewport is fixed before adding clipped dashed extensions.
    limits = (ax.get_xlim(), ax.get_ylim())
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    for group in GROUPS:
        for x, y in extrapolation_segments(record[group], field, *limits, scale):
            ax.plot(x, y, color=GROUP_COLORS[group], lw=1.05, linestyle=(0, (3.2, 2.2)),
                    zorder=2, scalex=False, scaley=False)
    compute_ticks(ax)
    if field != "L":
        log_ticks(ax, "y")
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True)
    ax.set_ylabel({"N": r"Parameters, $N^*$ (M)", "D": r"Tokens, $D^*$ (B)", "L": r"Loss, $L^*$"}[field])


def nd_panel(ax, record, peptide=False):
    for group in GROUPS:
        frontier = record[group]
        ordered = sorted(frontier["vertices"], key=lambda v: v["C"])
        ax.plot([v["N"] / 1e6 for v in ordered], [v["D"] / 1e9 for v in ordered],
                color=GROUP_COLORS[group], lw=1.05, zorder=3)
        vertices(ax, frontier, group, "N", "D", 1e6, 1e9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel(r"Tokens, $D^*$ (B)")
    ax.grid(True)
    if peptide:
        inset = ax.inset_axes((.37, .14, .60, .28), facecolor="#FCFCFD", zorder=6)
        for group in ("single", "target_normalized"):
            frontier = record[group]
            ordered = sorted(frontier["vertices"], key=lambda v: v["C"])
            inset.plot([v["N"] / 1e6 for v in ordered], [v["D"] / 1e6 for v in ordered],
                       color=GROUP_COLORS[group], lw=.85)
            vertices(inset, frontier, group, "N", "D", 1e6, 1e6)
        inset.set_xlim(left=0)
        inset.set_ylim(bottom=0)
        # N has the same units as the parent panel; do not crowd its baseline.
        inset.set_ylabel(r"$D^*$ (M)", fontsize=6, labelpad=.5)
        inset.tick_params(labelsize=5.5, length=2, pad=1)
        inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.grid(True, linewidth=.35)
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#AAB0B7")
            spine.set_linewidth(.5)


def check_bottom_legends(fig, axes, panel_labels):
    """Legends belong below every data panel, axis label, and panel caption."""
    axes = list(axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text = [ax.xaxis.label for ax in axes if ax.get_xlabel()] + list(panel_labels)
    boundary = min([ax.get_window_extent(renderer).y0 for ax in axes]
                   + [label.get_window_extent(renderer).y0 for label in text])
    for legend in fig.legends:
        require(legend.get_window_extent(renderer).y1 + 2 < boundary,
                "A bottom legend overlaps the plots or their lower labels")


def comparison_figure(records):
    fig, axes = plt.subplots(5, 4, figsize=(12.6, 12.0))
    fig.subplots_adjust(left=.066, right=.988, bottom=.162, top=.982, wspace=.31, hspace=.40)
    handles = [Line2D([], [], color=GROUP_COLORS[g], lw=1.7, label=GROUP_LABELS[g]) for g in GROUPS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .075),
               ncol=3, frameon=False, fontsize=8, handlelength=2.3, columnspacing=2.2)
    meaning = [Line2D([], [], color="#666666", marker="o", linestyle="none", markersize=3.8,
                      label="Supported vertex"),
               Line2D([], [], color="#666666", marker="o", mfc="white", linestyle="none", markersize=3.8,
                      label="Weak-fit vertex"),
               Line2D([], [], color="#666666", lw=1.2, label="Fit within vertex range"),
               Line2D([], [], color="#666666", lw=1.2, ls="--", label="Extrapolation")]
    fig.legend(handles=meaning, loc="upper center", bbox_to_anchor=(.52, .044),
               ncol=4, frameon=False, fontsize=7, handlelength=2.1, columnspacing=1.8)
    panel_labels = []
    for row, domain in enumerate(DOMAINS):
        record = records[domain]
        box = axes[row, 0].get_position()
        # Row names are lower panel captions, not headings above the data.
        label_y = box.y0 - (.053 if row == len(DOMAINS) - 1 else .029)
        panel_labels.append(fig.text(box.x0, label_y, f"({chr(97 + row)}) {NAMES[domain]}",
                                     fontsize=9, fontweight="semibold"))
        panel_labels.append(fig.text(axes[row, -1].get_position().x1, label_y,
                                     rf"$r = {100 * record['ratio']:.4f}\%$", ha="right", fontsize=7.5, color="#555555"))
        for col, field in enumerate(("N", "D", "L")):
            frontier_panel(axes[row, col], record, field)
        nd_panel(axes[row, 3], record, peptide=domain == "peptide")
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"Compute, $C$ or $rC$ (FLOPs)")
    axes[-1, 3].set_xlabel(r"Parameters, $N^*$ (M)")
    check_bottom_legends(fig, axes.flat, panel_labels)
    return fig


def cross_panel(ax, lines, planned):
    for line in lines:
        points = line["points"]
        ax.plot([p["parameter_count"] / 1e6 for p in points], [p["all_domain_loss"] for p in points],
                color=BUDGET_PALETTE[int(line["budget"][1:])], lw=1.25,
                marker="o", markersize=3.4, markeredgewidth=.3, zorder=3)
    ax.set_xscale("log")
    # Retain the original planned N range, including unavailable checkpoints.
    ax.set_xlim(min(p["parameter_count"] for p in planned) / 1e6 / 1.15,
                max(p["parameter_count"] for p in planned) / 1e6 * 1.15)
    log_ticks(ax, "x")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel(r"Parameters, $N$ (M)", labelpad=3)
    ax.set_ylabel("Loss", labelpad=4)
    ax.grid(True)


def cross_figure(series, plan):
    fig = plt.figure(figsize=(13.4, 4.0))
    outer = fig.add_gridspec(1, 5, left=.045, right=.991, bottom=.430, top=.973,
                            wspace=.36)
    axes, panel_labels = [], []
    for i, domain in enumerate(DOMAINS):
        ax = fig.add_subplot(outer[0, i])
        axes.append(ax)
        planned = [p for p in plan["rows"] + plan["unavailable"] if p["domain"] == domain]
        cross_panel(ax, series[domain], planned)
        box = ax.get_position()
        center = (box.x0 + box.x1) / 2
        panel_labels.append(fig.text(center, .290,
                                     f"({chr(97 + i)}) {NAMES[domain]}", ha="center",
                                     fontsize=9, fontweight="semibold"))
        handles = []
        for budget in sorted({p["budget"] for p in planned}):
            c = next(p["target_flops"] for p in planned if p["budget"] == budget)
            has_points = any(line["budget"] == budget for line in series[domain])
            handles.append(Line2D([], [], color=BUDGET_PALETTE[int(budget[1:])],
                                  lw=1.2, marker="o" if has_points else None, markersize=3,
                                  label=budget_label(c) + ("" if has_points else r"$^{\dagger}$")))
        fig.legend(handles=handles, title=r"$C$ (FLOPs)",
                   title_fontsize=6.8, fontsize=6.5, ncol=2, frameon=False,
                   loc="upper center", bbox_to_anchor=(center, .235), borderaxespad=0,
                   handlelength=1.25, handletextpad=.4, columnspacing=.8, labelspacing=.35)
    check_bottom_legends(fig, axes, panel_labels)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--single-to-all-only", action="store_true",
                        help="Update only the single-to-all artwork; leave comparison artwork untouched")
    args = parser.parse_args()
    require("worker" in args.expected_host and socket.gethostname() == args.expected_host,
            f"Run on the confirmed allocated worker, not {socket.gethostname()}")
    records, audit = load_frontiers()
    summary = read(CROSS_SOURCE / "summary.json")
    diagnostics = read(CROSS_SOURCE / "polyline_points.json")
    plan = read(PLAN)
    series = validate_cross(summary, diagnostics, plan)
    require(summary["off_host_unavailable"] == 81, "Expected the supplied 136/81 frozen snapshot")
    style()
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    if not args.single_to_all_only:
        files += export(comparison_figure(records), args.output / "three_frontiers", args.dpi)
    files += export(cross_figure(series, plan), args.output / "single_to_all_overview", args.dpi)
    # Store the plotted numbers and original fit coefficients for a paper handoff.
    if not args.single_to_all_only:
        write(args.output / "source_frontiers.json", {
            "source": str(FRONTIER_SOURCE), "source_rendering": read(FRONTIER_SOURCE / "plot_rendering.json"),
            "records": {d: {"ratio": records[d]["ratio"], **{g: records[d][g] for g in GROUPS}} for d in DOMAINS},
            "audit": audit,
        })
    write(args.output / "source_single_to_all.json", {
        "source": str(CROSS_SOURCE), "snapshot_at": summary["at"],
        "status": {k: v for k, v in summary.items() if k != "rows"},
        "rows": [{k: r[k] for k in ("run_id", "train_domain", "budget", "parameter_count", "target_flops", "all_domain_loss", "total_loss_tokens")}
                 for r in summary["rows"]],
        "lines": {d: [{"budget": s["budget"], "point_ids": [p["run_id"] for p in s["points"]]} for s in series[d]] for d in DOMAINS},
        "plan": {category: [{k: p[k] for k in ("run_id", "domain", "budget", "parameter_count", "target_flops")}
                             for p in plan[category]] for category in ("rows", "unavailable")},
        "protocol": {k: plan[k] for k in ("metric_split", "test_accessed", "context_length", "union_weighting")},
    })
    write(args.output / "rendering.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "worker": socket.gethostname(),
        "files": files, "dpi": args.dpi, "data_unchanged": True, "new_fits": False,
        "new_exclusions": False, "new_checkpoint_evaluations": False,
        "frontiers": audit, "single_to_all": {
            "observations": len(summary["rows"]), "off_host_unavailable": summary["off_host_unavailable"],
            "domain_counts": EXPECTED_EVALUATIONS, "polylines": sum(len(lines) for lines in series.values()),
            "line_order": "increasing parameter_count, then run_id, within domain/budget",
            "style": "observed_polyline_no_fit", "x_axis": "log", "loss_axis": "linear",
        },
        "loss_labels": "Loss; split, token weighting and proxy definitions retained in captions and source JSON",
        "layout": {"three_frontiers": "Five domain rows; panel names below rows; shared legends at bottom",
                   "single_to_all": "One row of five domains; each panel name and own budget legend directly below; no overall title"},
        "render_scope": "single_to_all_only" if args.single_to_all_only else "both_figures",
    })
    print(json.dumps({"complete": True, "files": files, "frontier_vertices": sum(sum(x["vertices"] for x in v.values()) for v in audit["counts"].values()),
                      "evaluations": len(summary["rows"]), "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
