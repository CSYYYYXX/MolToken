#!/usr/bin/env python3
"""Publication rendering of frozen single-domain experiments (worker only).

This is a presentation-only export. All 227 observations and the saved all-point
quadratic/frontier fits are retained. No checkpoint or training log is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
PAPER = ROOT / "scaling-csy/doc/paper_materials_20260728"
SNAPSHOT = ROOT / "HoloMol/doc/results/domainwise145_visualization_20260906/snapshot.json"
FRONTIERS = ROOT / "HoloMol/doc/results/analysis/scaling_frontier_comparison_20260910/summary.json"
OUTPUT = PAPER / "figures/single_to_single_publication_20260913"
DOMAINS = ("small_molecule", "peptide", "protein", "dna", "rna")
NAMES = dict(zip(DOMAINS, ("Small molecule", "Peptide", "Protein", "DNA", "RNA")))
EXPECTED_COUNTS = dict(zip(DOMAINS, (49, 41, 45, 41, 51)))
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#6656A6")
FRONTIER_COLOR = "#0072B2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def quadratic(curve, n):
    return np.polyval(curve["coefficients_centered"],
                     np.log10(n) - curve["log10_N_center"])


def frontier_values(frontier, field, compute):
    fit = frontier["fits"][field]
    y = fit["slope"] * (np.log10(compute) - fit["log10_C_pivot"]) + fit["value_at_pivot"]
    return y if field == "L" else 10.0 ** y


def validation_evidence(point):
    """The standard collector and native peptide supplements use two schemas."""
    if "metric_split" in point:
        require(point["metric_split"] == "validation"
                and str(point["validation_complete_view"]).lower() == "true",
                f"Incomplete/nonvalidation metric: {point['run_id']}")
        source = "flattened_standard_collector"
    else:
        require(point["domain"] == "peptide"
                and point["scientific_protocol"] == "peptide_isoflop_b1024_v1",
                f"Unrecognized metric schema: {point['run_id']}")
        evidence = point["final_validation"]
        require(evidence["complete_view"] is True and close(evidence["input_token_coverage"], 1)
                and int(evidence["input_tokens"]) == int(evidence["expected_input_tokens"])
                and close(evidence["loss"], point["validation_loss"]),
                f"Incomplete/inconsistent native validation: {point['run_id']}")
        source = "native_peptide_final_validation"
    return {"metric_split": "validation", "validation_complete_view": True,
            "validation_evidence_source": source}


def validate_inputs(snapshot, report, expected_counts=None):
    """Check scientific content rather than whole-file/source hash identity."""
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    require(snapshot["metric_split"] == "validation", "Expected validation loss")
    require(snapshot["test_accessed"] is False, "Test results must not enter these fits")
    require(snapshot["historical_fixed_d_peptide_included"] is False,
            "Historical fixed-D peptide runs must remain separate")
    require(not snapshot["pending_rows"], "The publication snapshot must be complete")
    rows = snapshot["completed_rows"]
    by_id = {r["run_id"]: r for r in rows}
    require(len(by_id) == len(rows), "Duplicate completed run IDs")
    require(dict(Counter(r["domain"] for r in rows)) == expected_counts,
            "Unexpected completed domain counts")
    analysis = report["analyses"]["recorded_objectives"]
    source_curves = {(c["domain"], c["budget"]): c for c in snapshot["curves"]}
    require(len(source_curves) == len(snapshot["curves"]), "Duplicate snapshot curves")
    seen = []
    total_vertices = strong_vertices = 0
    for domain in expected_counts:
        record = analysis[domain]
        frontier = record["single"]
        curves = record["single_curves"]
        vertices = {v["budget"]: v for v in frontier["vertices"]}
        require(len(vertices) == len(frontier["vertices"]), "Duplicate frontier budgets")
        require(len(vertices) == len(curves), "An IsoFLOP vertex is missing")
        for curve in curves:
            key = (domain, curve["budget"])
            original = source_curves[key]
            ids = curve["run_ids"]
            require(len(ids) == len(set(ids)), f"Duplicate curve IDs: {key}")
            require(set(ids) == {p["run_id"] for p in original["points"]},
                    f"Point selection differs between frozen inputs: {key}")
            require(len(ids) == curve["count"], f"Point count differs: {key}")
            points = [by_id[rid] for rid in ids]
            require(ids == [p["run_id"] for p in sorted(points, key=lambda p: float(p["parameter_count"]))],
                    f"Saved fit point order differs: {key}")
            for point in points:
                require(point["domain"] == domain and point["budget"] == curve["budget"],
                        f"Point assigned to wrong curve: {point['run_id']}")
                validation_evidence(point)
                require(close(point["target_flops_6nd"], curve["C"]), f"Budget differs: {key}")
                for field in ("parameter_count", "validation_loss", "actual_epoch_fraction"):
                    require(math.isfinite(float(point[field])) and float(point[field]) > 0,
                            f"Invalid {field}: {point['run_id']}")
            n = np.array([float(p["parameter_count"]) for p in points])
            loss = np.array([float(p["validation_loss"]) for p in points])
            require(np.allclose(loss - quadratic(curve, n), curve["residuals"],
                                atol=1e-9, rtol=1e-9), f"Saved fit uses different losses/N: {key}")
            # OLS identity is checked, but the saved coefficients are used to render.
            coefficients = np.polyfit(np.log10(n) - curve["log10_N_center"], loss, 2)
            require(np.allclose(coefficients, curve["coefficients_centered"], atol=1e-9, rtol=1e-9),
                    f"Saved quadratic is not the all-point fit: {key}")
            vertex = vertices[curve["budget"]]
            require(curve["status"] == "interior", f"Noninterior fitted vertex: {key}")
            for field in ("C", "N", "D", "L"):
                require(close(vertex[field], curve["vertex"][field]), f"Vertex differs: {key}/{field}")
            require(close(vertex["D"], vertex["C"] / (6 * vertex["N"])), f"D* identity fails: {key}")
            require(close(vertex["L"], quadratic(curve, vertex["N"])), f"Vertex loss differs: {key}")
            require(vertex["strong_support"] == curve["strong_support"], f"Support label differs: {key}")
            predicted = quadratic(curve, n)
            r2 = 1 - float(np.sum((loss - predicted)**2) / np.sum((loss - loss.mean())**2))
            best = int(np.argmin(loss))
            strong = (len(n) >= 5 and r2 >= .8 and min(best, len(n) - best - 1) >= 2
                      and min(int(np.sum(n < vertex["N"])), int(np.sum(n > vertex["N"]))) >= 2
                      and abs(int(np.argmin(predicted)) - best) <= 1)
            require(close(r2, curve["r2"]), f"Quadratic R2 differs: {key}")
            require(strong == curve["strong_support"], f"Original support criteria differ: {key}")
            seen.extend(ids)
        for field in ("N", "D", "L"):
            fit = frontier["fits"][field]
            vv = frontier["vertices"]
            xx = np.log10([v["C"] for v in vv]) - fit["log10_C_pivot"]
            yy = np.array([v[field] for v in vv])
            if field != "L":
                yy = np.log10(yy)
            slope, level = np.polyfit(xx, yy, 1)
            require(close(slope, fit["slope"]) and close(level, fit["value_at_pivot"]),
                    f"Frontier is not the saved all-vertex fit: {domain}/{field}")
            require(fit["count"] == len(vv), f"Frontier count differs: {domain}/{field}")
        total_vertices += len(vertices)
        strong_vertices += sum(v["strong_support"] for v in vertices.values())
        require(frontier["strong_vertex_count"] == sum(v["strong_support"] for v in vertices.values()),
                f"Supported-vertex count differs: {domain}")
    require(Counter(seen) == Counter(by_id.keys()), "Not every completed run is used exactly once")
    require(total_vertices == len(source_curves), "Snapshot has unused curves")
    return {"observations": len(rows), "domain_counts": expected_counts,
            "isoflop_curves": total_vertices, "supported_vertices": strong_vertices,
            "weak_vertices": total_vertices - strong_vertices,
            "all_observations_retained": True, "new_fits": False,
            "checkpoints_loaded": False, "test_accessed": False, "hash_gates": False}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "semibold",
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.axisbelow": True, "grid.color": "#E4E7EB", "grid.linewidth": .55,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "mathtext.fontset": "dejavusans",
    })


def log_ticks(ax, which):
    """Readable raw-unit labels on log axes, including sub-decade spans."""
    lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
    ticks = [m * 10.0**p for p in range(math.floor(math.log10(lo)) - 1,
                                       math.ceil(math.log10(hi)) + 1)
             for m in (1, 2, 5) if lo <= m * 10.0**p <= hi]
    if len(ticks) < 3:
        ticks = [m * 10.0**p for p in range(math.floor(math.log10(lo)) - 1,
                                           math.ceil(math.log10(hi)) + 1)
                 for m in (1, 1.5, 2, 3, 4, 5, 6, 8) if lo <= m * 10.0**p <= hi]
    if len(ticks) > 5:
        ticks = [ticks[i] for i in np.linspace(0, len(ticks) - 1, 5).astype(int)]
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    axis.set_minor_locator(NullLocator())


def budget_label(compute):
    exponent = math.floor(math.log10(compute))
    return rf"${compute / 10**exponent:.3g}\times10^{{{exponent}}}$"


def iso_handles(curves):
    return [Line2D([], [], color=COLORS[int(c["budget"][1:])], lw=1.2,
                   marker="o", markersize=3, label=budget_label(c["C"])) for c in curves]


def isoflop_panel(ax, domain, snapshot, record, title):
    rows = {r["run_id"]: r for r in snapshot["completed_rows"]}
    curves = sorted(record["single_curves"], key=lambda c: c["C"])
    for curve in curves:
        points = [rows[rid] for rid in curve["run_ids"]]
        n = np.array([float(p["parameter_count"]) for p in points])
        loss = np.array([float(p["validation_loss"]) for p in points])
        color = COLORS[int(curve["budget"][1:])]
        grid = np.geomspace(n.min(), n.max(), 256)
        ax.plot(grid / 1e6, quadratic(curve, grid), color=color, lw=1.25,
                ls="-" if curve["strong_support"] else ":", zorder=2)
        ax.scatter(n / 1e6, loss, s=13, c=color, edgecolors="white", linewidths=.35, zorder=4)
        vertex = curve["vertex"]
        ax.scatter(vertex["N"] / 1e6, vertex["L"], marker="D", s=27,
                   facecolors=color if curve["strong_support"] else "white",
                   edgecolors=color, linewidths=.85, zorder=5)
    ax.set_xscale("log")
    ax.margins(x=.045, y=.09)
    log_ticks(ax, "x")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(True, which="major")
    ax.set_xlabel(r"Parameters, $N$ (M)")
    ax.set_ylabel("Loss")
    ax.set_title(title, loc="left", pad=7)
    ax.legend(handles=iso_handles(curves), title=r"Compute, $C$ (FLOPs)",
              title_fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(.5, -.23), frameon=False, handlelength=1.35,
              columnspacing=.9, handletextpad=.45, borderaxespad=0)
    return curves


def frontier_panel(ax, frontier, field, *, token_unit=None):
    vertices = frontier["vertices"]
    c_min, c_max = frontier["C_range"]
    exponent = math.floor(math.log10(c_max))
    c_unit = 10.0**exponent
    unit = 1e6 if field == "N" else (1e6 if field == "D" and max(v["D"] for v in vertices) < 1e9
                                    else 1e9 if field == "D" else 1)
    if field == "D" and token_unit is not None:
        require(token_unit in (1e6, 1e9), "Token display unit must be M or B")
        unit = token_unit
    unit_name = "M" if unit == 1e6 else "B"
    delta = (math.log10(c_max) - math.log10(c_min)) * .09
    xmin, xmax = 10**(math.log10(c_min) - delta), 10**(math.log10(c_max) + delta)
    grid = np.geomspace(c_min, c_max, 200)
    ax.plot(grid / c_unit, frontier_values(frontier, field, grid) / unit,
            color=FRONTIER_COLOR, lw=1.45, zorder=2)
    # Short dashed extensions fill the margins without implying measured data.
    for low, high in ((xmin, c_min), (c_max, xmax)):
        grid = np.geomspace(low, high, 30)
        ax.plot(grid / c_unit, frontier_values(frontier, field, grid) / unit,
                color=FRONTIER_COLOR, lw=1.05, ls=(0, (3, 2.4)), alpha=.75, zorder=2)
    for vertex in vertices:
        ax.scatter(vertex["C"] / c_unit, vertex[field] / unit, s=21,
                   facecolors=FRONTIER_COLOR if vertex["strong_support"] else "white",
                   edgecolors=FRONTIER_COLOR, linewidths=.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(xmin / c_unit, xmax / c_unit)
    if field != "L":
        ax.set_yscale("log")
    ax.margins(y=.19)
    log_ticks(ax, "x")
    if field != "L":
        log_ticks(ax, "y")
    else:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, which="major")
    ax.set_xlabel(rf"Compute, $C$ ($10^{{{exponent}}}$ FLOPs)")
    labels = {"N": r"Optimal $N^*$ (M)", "D": rf"Optimal $D^*$ ({unit_name} tokens)",
              "L": r"Loss, $L^*$"}
    ax.set_ylabel(labels[field])
    slope = frontier["fits"][field]["slope"]
    slope_name = {"N": r"\alpha", "D": r"\beta", "L": "s"}[field]
    annotation_y = .95 if field == "L" else .055
    ax.text(.97, annotation_y, rf"${slope_name}={slope:.3f}$", transform=ax.transAxes,
            ha="right", va="top" if field == "L" else "bottom", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.2})


def support_handles():
    return [Line2D([], [], marker="o", linestyle="none", color=FRONTIER_COLOR,
                   markersize=4, label="Supported vertex"),
            Line2D([], [], marker="o", linestyle="none", color=FRONTIER_COLOR,
                   markerfacecolor="white", markersize=4, label="Weak-fit vertex"),
            Line2D([], [], color=FRONTIER_COLOR, lw=1.4, label="All-vertex fit"),
            Line2D([], [], color=FRONTIER_COLOR, ls="--", lw=1, label="Extrapolation")]


def export(fig, stem, dpi):
    files = []
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    require(bbox.x0 >= -.01 and bbox.y0 >= -.01
            and bbox.x1 <= width + .01 and bbox.y1 <= height + .01,
            f"Clipped publication artwork: {stem.name}, content={bbox.bounds}, page={(width, height)}")
    # Preserve explicitly chosen publication page sizes (no tight-bbox resizing).
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, dpi=dpi)
        files.append(path.name)
    plt.close(fig)
    return files


def combined_domain_figure(snapshot, report):
    """Five domain columns, each with IsoFLOP and all three frontiers.

    A dedicated budget-legend strip keeps labels out of the measured curves.
    The lower three panels share compute within each column. Token display
    units are consistently billions across columns, including peptide.
    """
    records = report["analyses"]["recorded_objectives"]
    fig = plt.figure(figsize=(12.6, 8.8))
    grid = fig.add_gridspec(7, 5, left=.060, right=.990, bottom=.105, top=.940,
                            width_ratios=[1] * 5,
                            height_ratios=[1.35, .80, 1, .20, 1, .20, 1],
                            hspace=0, wspace=.32)
    axes = np.empty((4, 5), dtype=object)
    budget_legends = []
    for column, domain in enumerate(DOMAINS):
        record = records[domain]
        ax = fig.add_subplot(grid[0, column])
        axes[0, column] = ax
        curves = isoflop_panel(ax, domain, snapshot, record,
                               f"({chr(97 + column)}) {NAMES[domain]}")
        ax.get_legend().remove()
        ax.set_title("", loc="left")
        ax.set_title(f"({chr(97 + column)}) {NAMES[domain]}",
                     loc="center", fontsize=10, pad=10)
        ax.set_xlabel(r"Parameters, $N$ (M)", fontsize=8.5)
        ax.set_ylabel("Loss" if column == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)

        # Same compute unit as the frontier x-axis, so physical budgets remain
        # explicit without repeating a long scientific-notation label six times.
        exponent = math.floor(math.log10(record["single"]["C_range"][1]))
        unit = 10.0**exponent
        handles = [Line2D([], [], color=COLORS[int(c["budget"][1:])], lw=1.2,
                          marker="o", markersize=3, label=f"{c['C'] / unit:.3g}")
                   for c in curves]
        # Matplotlib fills legend columns first; reorder for row-wise reading.
        handles = [handle for offset in range(3) for handle in handles[offset::3]]
        legend_ax = fig.add_subplot(grid[1, column])
        legend_ax.axis("off")
        legend = legend_ax.legend(handles=handles, title=rf"$C$ ($10^{{{exponent}}}$ FLOPs)",
                         title_fontsize=7.6, fontsize=7.5, ncol=3,
                         loc="center", bbox_to_anchor=(.5, .32), frameon=False,
                         columnspacing=.85, handlelength=1.2, handletextpad=.45,
                         borderaxespad=0, labelspacing=.27)
        budget_legends.append(legend)

        for row, (field, grid_row) in enumerate(zip(("N", "D", "L"), (2, 4, 6)), start=1):
            ax = fig.add_subplot(grid[grid_row, column],
                                 sharex=axes[1, column] if row > 1 else None)
            axes[row, column] = ax
            frontier_panel(ax, record["single"], field, token_unit=1e9)
            ax.tick_params(labelsize=8)
            for annotation in ax.texts:
                annotation.set_fontsize(8.5)
            if row < 3:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(rf"Compute, $C$ ($10^{{{exponent}}}$ FLOPs)", fontsize=8.3)
            labels = {"N": r"$N^*$ (M)", "D": r"$D^*$ (B tokens)", "L": r"Loss, $L^*$"}
            ax.set_ylabel(labels[field] if column == 0 else "", fontsize=9)

    grey = "#555555"
    handles = [
        Line2D([], [], color=grey, marker="o", linestyle="none", markersize=4,
               label="Measured loss (IsoFLOP)"),
        Line2D([], [], color=grey, marker="D", linestyle="none", markersize=4,
               label="Fitted minimum (IsoFLOP)"),
        Line2D([], [], color=FRONTIER_COLOR, marker="o", markerfacecolor="white",
               linestyle="none", markersize=4, label="Weak-fit optimum"),
        Line2D([], [], color=grey, lw=1.3, label="Fit"),
        Line2D([], [], color=grey, lw=1.3, ls=":", label="Weak quadratic fit"),
        Line2D([], [], color=grey, lw=1.1, ls="--", label="Extrapolation"),
    ]
    fig.legend(handles=handles, ncol=6, frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(.53, .021),
               columnspacing=1.5, handlelength=1.5, handletextpad=.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for column, legend in enumerate(budget_legends):
        box = legend.get_window_extent(renderer).padded(2)
        require(not box.overlaps(axes[0, column].xaxis.label.get_window_extent(renderer)),
                f"Budget legend overlaps parameter label: {DOMAINS[column]}")
        require(box.y0 > axes[1, column].get_window_extent(renderer).y1,
                f"Budget legend overlaps N frontier: {DOMAINS[column]}")
    return fig, axes


def render(snapshot, report, output, dpi=450):
    style()
    records = report["analyses"]["recorded_objectives"]
    combined, _ = combined_domain_figure(snapshot, report)
    artifacts = export(combined, output / "single_to_single_all_domains", dpi)
    for domain in DOMAINS:
        fig, ax = plt.subplots(figsize=(3.5, 3.15))
        fig.subplots_adjust(left=.17, right=.98, bottom=.32, top=.90)
        isoflop_panel(ax, domain, snapshot, records[domain], NAMES[domain])
        artifacts.extend(export(fig, output / f"isoflop_{domain}", dpi))
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
        fig.subplots_adjust(left=.075, right=.985, bottom=.28, top=.78, wspace=.48)
        for index, (ax, field) in enumerate(zip(axes, ("N", "D", "L"))):
            frontier_panel(ax, records[domain]["single"], field)
            ax.set_title(f"({chr(97 + index)})", loc="left", pad=7)
        fig.suptitle(NAMES[domain], x=.075, y=.98, ha="left", fontsize=10, fontweight="semibold")
        fig.legend(handles=support_handles(), loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(.52, .005), handlelength=1.6, columnspacing=1.4)
        artifacts.extend(export(fig, output / f"frontiers_{domain}", dpi))

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.5))
    fig.subplots_adjust(left=.08, right=.975, bottom=.12, top=.965, wspace=.30, hspace=.75)
    for index, domain in enumerate(DOMAINS):
        isoflop_panel(axes.flat[index], domain, snapshot, records[domain],
                      f"({chr(97 + index)}) {NAMES[domain]}")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.text(.03, .91, "Single-domain pretraining", fontsize=10, weight="semibold")
    handles = [Line2D([], [], marker="o", linestyle="none", color="#555555", markersize=4,
                      label="Measured loss"),
               Line2D([], [], marker="D", linestyle="none", color="#555555", markersize=4,
                      label="Supported fitted minimum"),
               Line2D([], [], marker="D", linestyle="none", color="#555555", markerfacecolor="white",
                      markersize=4, label="Weak-fit minimum"),
               Line2D([], [], color="#555555", lw=1.2, label="Supported quadratic fit"),
               Line2D([], [], color="#555555", lw=1.2, ls=":", label="Weak quadratic fit")]
    legend_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, .82),
                     frameon=False, labelspacing=1.3, handlelength=2.0, fontsize=8)
    artifacts.extend(export(fig, output / "single_to_single_isoflop", dpi))

    fig, axes = plt.subplots(5, 3, figsize=(7.2, 9.8))
    fig.subplots_adjust(left=.078, right=.985, bottom=.085, top=.95, wspace=.50, hspace=.80)
    for row, domain in enumerate(DOMAINS):
        for col, field in enumerate(("N", "D", "L")):
            ax = axes[row, col]
            frontier_panel(ax, records[domain]["single"], field)
            if col == 0:
                ax.set_title(f"({chr(97 + row)}) {NAMES[domain]}", loc="left", pad=9)
    for column, heading in enumerate((r"$N^* \propto C^\alpha$", r"$D^* \propto C^\beta$",
                                      r"$L^* = s\,\log_{10} C + b$")):
        position = axes[0, column].get_position()
        fig.text(position.x0 + position.width / 2, .989, heading,
                 ha="center", va="top", fontsize=10)
    fig.legend(handles=support_handles(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(.53, .021), handlelength=1.6, columnspacing=1.5)
    artifacts.extend(export(fig, output / "single_to_single_frontiers", dpi))
    return artifacts


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evidence(snapshot, report, validation, output, inputs, artifacts):
    records = report["analyses"]["recorded_objectives"]
    points = [{**{key: row[key] for key in ("run_id", "domain", "budget", "parameter_count",
              "target_flops_6nd", "validation_loss", "actual_epoch_fraction", "actual_input_tokens",
              "actual_flops_6nd")}, **validation_evidence(row)}
              for row in snapshot["completed_rows"]]
    write_csv(output / "observations.csv", points)
    vertices, coefficients, weak = [], [], []
    for domain in DOMAINS:
        for curve in records[domain]["single_curves"]:
            vertices.append({"domain": domain, "budget": curve["budget"], **curve["vertex"],
                             "quadratic_r2": curve["r2"], "observations": curve["count"]})
            if not curve["strong_support"]:
                weak.append({"domain": domain, "budget": curve["budget"], "r2": curve["r2"],
                             "grid_min_agreement": curve["fitted_observed_grid_min_agreement"]})
        frontier = records[domain]["single"]
        coefficients.append({"domain": domain, "vertices": frontier["vertex_count"],
                             "supported_vertices": frontier["strong_vertex_count"],
                             **{f"{field}_{key}": frontier["fits"][field][key]
                                for field in ("N", "D", "L") for key in ("slope", "intercept", "r2")}})
    write_csv(output / "vertices.csv", vertices)
    write_csv(output / "coefficients.csv", coefficients)
    metrics = {"schema": "single_domain_publication_v1",
               "rendered_at": datetime.now(timezone.utc).isoformat(), "worker": socket.gethostname(),
               "source_updated_at": snapshot["updated_at"], "inputs": inputs,
               "validation": validation, "weak_curves": weak,
               "epoch_ranges": {domain: [min(float(p["actual_epoch_fraction"]) for p in points if p["domain"] == domain),
                                          max(float(p["actual_epoch_fraction"]) for p in points if p["domain"] == domain)]
                                for domain in DOMAINS},
               "fit_policy": "Saved all-observation quadratics; saved all-interior-vertex frontiers; no trim or exclusion",
               "metric": "Full same-domain validation NLL per predicted FDDFS/BPE token",
               "D_definition": "D*=C/(6N*), nominal training input tokens incl. EOS; not unique corpus size",
               "L_definition": "L*=s*log10(C)+b (linear-loss, logarithmic-compute fit; not log-log loss)",
               "styles": {"frontier_hollow": "weak quadratic vertex, retained in frontier fit",
                          "frontier_dashed": "short extrapolation to axis boundary",
                          "isoflop_dotted": "weak all-point quadratic fit"},
               "coefficients": coefficients, "artifacts": artifacts}
    (output / "figure_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Small self-contained fit records, excluding unrelated multi-domain analyses.
    (output / "source_fits.json").write_text(json.dumps(
        {domain: {"single": records[domain]["single"], "single_curves": records[domain]["single_curves"]}
         for domain in DOMAINS}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    items = [("single_to_single_all_domains", "All domains — IsoFLOP and frontiers"),
             ("single_to_single_isoflop", "IsoFLOP overview"),
             ("single_to_single_frontiers", "Compute-optimal frontiers")]
    items.extend((f"{kind}_{domain}", f"{NAMES[domain]} — {kind}")
                 for domain in DOMAINS for kind in ("isoflop", "frontiers"))
    cards = "\n".join(f'<section><h2>{title}</h2><p><a href="{stem}.pdf">PDF</a> · '
                       f'<a href="{stem}.svg">SVG</a> · <a href="{stem}.png">PNG</a></p>'
                       f'<img src="{stem}.svg" alt="{title}"></section>' for stem, title in items)
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Single-domain scaling figures</title>'
        '<style>body{font:16px system-ui;margin:35px auto;max-width:1000px;padding:0 20px;color:#20242a}'
        'section{border-top:1px solid #ddd;margin:30px 0;padding:15px 0}img{max-width:100%;display:block}'
        'a{color:#0072b2}h2{font-size:20px}</style><h1>Single-domain scaling figures</h1>'
        '<p>227 observations · same-domain validation · no point exclusions. '
        'Open vertices mark weak fits; see README.md and captions.tex before citing.</p>' + cards + '</html>\n',
        encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--frontiers", type=Path, default=FRONTIERS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    require(socket.gethostname() == args.expected_host and "worker" in args.expected_host,
            "Run only on the allocated worker, never the jump host")
    snapshot = json.loads(args.snapshot.read_text())
    report = json.loads(args.frontiers.read_text())
    validation = validate_inputs(snapshot, report)
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = render(snapshot, report, args.output, args.dpi)
    metrics = write_evidence(snapshot, report, validation, args.output,
                             {"snapshot": str(args.snapshot), "frontiers": str(args.frontiers)}, artifacts)
    print(json.dumps({"output": str(args.output), "validation": validation,
                      "figures": len(artifacts) // 3, "formats": ["pdf", "svg", "png"],
                      "epoch_ranges": metrics["epoch_ranges"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
