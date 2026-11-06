#!/usr/bin/env python3
"""Restyle frozen paper figures at ICLR's physical 5.5-inch text width.

Worker only. This does not select observations, train, evaluate, or change fits.
The older exporters remain untouched. Legends use compact units so their type
is genuinely 9 pt in a PDF included at width=\\linewidth, not only at zoom.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
import numpy as np

import plot_single_domain_publication as single
import plot_multi_to_single_publication as multi
import plot_cross_domain_publication as cross
import plot_frontier_comparison_2x5 as comparison

PAPER = single.PAPER
SOURCE = PAPER / "figures"
OUTPUT = SOURCE / "iclr_readable_20260917"
WIDTH = 5.5
DOMAINS = single.DOMAINS
NAMES = ("Small\nmolecule", "Peptide", "Protein", "DNA", "RNA")
MANIFEST = {}


def read(path):
    return json.loads(path.read_text())


def table(path, numbers):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in numbers:
            row[key] = float(row[key])
    return rows


def style():
    single.style()
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.labelpad": 2, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "grid.linewidth": .4, "lines.linewidth": 1.0,
    })


def numeric_artists(ax):
    """Compare actual plotted arrays before/after typography, not file hashes."""
    return ([np.asarray(line.get_xydata()).copy() for line in ax.lines],
            [np.asarray(c.get_offsets()).copy() for c in ax.collections],
            ax.get_xlim(), ax.get_ylim(), ax.get_xscale(), ax.get_yscale())


def assert_unchanged(ax, old):
    new = numeric_artists(ax)
    for a, b in zip(old[:2], new[:2]):
        single.require(len(a) == len(b), "Plot artist count changed")
        for x, y in zip(a, b):
            single.require(np.array_equal(x, y, equal_nan=True), "Plotted numbers changed")
    single.require(old[2:] == new[2:], "Axis scale or viewport changed")


def sparse_ticks(ax, which, maximum=2):
    axis = ax.xaxis if which == "x" else ax.yaxis
    lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
    scale = ax.get_xscale() if which == "x" else ax.get_yscale()
    if scale == "log":
        ticks = [x for x in axis.get_majorticklocs() if lo <= x <= hi]
        if which == "x":
            # Rebuild candidates instead of thinning an already sparse locator
            # (which can otherwise leave adjacent labels such as 50 and 100).
            decades = [10.0**e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))+1)
                       if lo <= 10.0**e <= hi]
            ticks = decades if len(decades) >= 2 else [
                m*10.0**e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))+1)
                for m in (1, 2, 5) if lo <= m*10.0**e <= hi]
        if len(ticks) < 2:
            ticks = [m * 10.0**e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))+1)
                     for m in (1, 2, 5) if lo <= m * 10.0**e <= hi]
        if which == "x":
            inner = [t for t in ticks if .08 <= (math.log(t)-math.log(lo))/(math.log(hi)-math.log(lo)) <= .92]
            if len(inner) >= 2:
                ticks = inner
        if len(ticks) > maximum:
            ticks = [ticks[i] for i in np.linspace(0, len(ticks)-1, maximum).astype(int)]
        axis.set_major_locator(FixedLocator(ticks))
    else:
        axis.set_major_locator(MaxNLocator(nbins=maximum, min_n_ticks=2))
    axis.set_minor_locator(NullLocator())
    axis.offsetText.set_fontsize(9)


def compact_axis(ax, *, ylabel=None, xlabel=None):
    before = numeric_artists(ax)
    ax.tick_params(axis="both", labelsize=9, length=2, pad=2)
    sparse_ticks(ax, "x")
    sparse_ticks(ax, "y", 3)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=3)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=2)
    ax.set_title("")
    ax.set_title("", loc="left")
    for line in ax.lines:
        line.set_linewidth(min(line.get_linewidth(), 1.1))
        if line.get_markersize() > 0:
            line.set_markersize(min(line.get_markersize(), 2.7))
    for collection in ax.collections:
        if hasattr(collection, "get_sizes"):
            collection.set_sizes(np.minimum(collection.get_sizes(), 14))
    assert_unchanged(ax, before)


def erase_notes(ax):
    for text in list(ax.texts):
        text.remove()


def save(fig, output, stem, details):
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    sizes = [float(t.get_fontsize()) for t in texts]
    single.require(min(sizes) >= 9, f"Sub-9pt text in {stem}: {min(sizes)}")
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    single.require(box.x0 >= -.025 and box.y0 >= -.025 and box.x1 <= width+.025 and box.y1 <= height+.025,
                   f"Clipped figure {stem}: {box.bounds}, canvas {width}x{height}")
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{extension}", dpi=350)
    MANIFEST[stem] = {"width_inches": float(width), "height_inches": float(height),
                      "nominal_text_pt_range": [min(sizes), max(sizes)],
                      "figure_bbox_inches": list(box.bounds), **details}
    plt.close(fig)
    print("EXPORTED", stem, MANIFEST[stem], flush=True)


def budget_legend(fig, axis, values, colors, exponent, *, ncol=2):
    handles = [Line2D([], [], color=color, lw=1, marker="o", markersize=2,
                       label=f"{value / 10**exponent:.3g}".removeprefix("0"))
               for value, color in zip(values, colors)]
    axis.axis("off")
    return axis.legend(handles=handles, loc="upper center", ncol=ncol, fontsize=9,
                       title=rf"$C/10^{{{exponent}}}$", title_fontsize=10,
                       frameon=False, handlelength=.42, handletextpad=.12,
                       columnspacing=.18, borderpad=0, borderaxespad=0, labelspacing=.10)


def single_inputs():
    base = SOURCE / "single_to_single_publication_20260913"
    rows = table(base / "observations.csv", ("parameter_count", "validation_loss"))
    records = read(base / "source_fits.json")
    single.require(len(rows) == 227 and sum(len(r["single_curves"]) for r in records.values()) == 27,
                   "Single-domain frozen counts changed")
    return {"completed_rows": rows}, records


def scaling_grid(output, mode):
    style()
    height = 7.5
    fig = plt.figure(figsize=(WIDTH, height))
    bands = [(5.85, 1.10), (3.55, .80), (2.25, .85), (1.0, .85)]
    axes = np.array([[fig.add_axes([.115+c*.183, y/height, .118, h/height])
                      for c in range(5)] for y, h in bands])
    if mode == "single":
        snapshot, records = single_inputs()
    else:
        base = SOURCE / "multi_to_single_publication_20260915"
        rows = table(base / "observations.csv", ("compute", "n", "loss"))
        record = read(base / "source_fits.json")
        saved, frontiers = record["isoflop"], record["frontiers"]
        audit = multi.validate(rows, saved, frontiers)
        fby = {f["domain"]: f for f in frontiers}
        budgets = sorted({c["compute"] for c in saved["curves"]})
        colors = {c: plt.get_cmap("viridis")(i / (len(budgets)-1)) for i, c in enumerate(budgets)}
    total_observations, total_vertices = 0, 0
    legends = []
    for col, domain in enumerate(DOMAINS):
        iso = axes[0, col]
        if mode == "single":
            curves = single.isoflop_panel(iso, domain, snapshot, records[domain], "")
            iso.get_legend().remove()
            values = [c["C"] for c in curves]
            palette = [single.COLORS[int(c["budget"][1:])] for c in curves]
            total_observations += sum(len(c["run_ids"]) for c in curves)
            total_vertices += len(curves)
        else:
            curves = multi.isoflop_panel(iso, domain, rows, saved, colors)
            values = [c["compute"] for c in curves]
            palette = [colors[v] for v in values]
            total_observations += sum(len(x.get_offsets()) for x in iso.collections if (x.get_gid() or "").startswith("observations-"))
        compact_axis(iso, ylabel="Loss" if col == 0 else "", xlabel=r"$N$ (M)")
        # Domain labels are identification, not interpretative headings.
        box = iso.get_position()
        fig.text((box.x0+box.x1)/2, .969, NAMES[col], ha="center", va="center", fontsize=10)
        legend_axis = fig.add_axes([box.x0-.018, 4.58/height, box.width+.036, .91/height])
        exponent = math.floor(math.log10(max(values)))
        legends.append(budget_legend(fig, legend_axis, values, palette, exponent))
        for row, field in enumerate(("N", "D", "L"), 1):
            ax = axes[row, col]
            if mode == "single":
                frontier = records[domain]["single"]
                single.frontier_panel(ax, frontier, field, token_unit=1e9 if field == "D" else None)
                exp = math.floor(math.log10(max(v["C"] for v in frontier["vertices"])))
                fit = frontier["fits"][field]
                slope = fit["slope"]
                r2 = fit["r2"]
            else:
                frontier = fby[domain]
                key = field + "_opt"
                multi.frontier_panel(ax, frontier, key, final_row=True)
                exp = 18
                slope, r2 = frontier[key]["slope"], frontier[key]["r2"]
                if field == "L":
                    total_vertices += len(frontier["vertices"])
            erase_notes(ax)
            compact_axis(ax, ylabel=({"N": r"$N^*$ (M)", "D": r"$D^*$ (B)", "L": r"Loss, $L^*$"}[field] if col == 0 else ""),
                         xlabel=(rf"$C/10^{{{exp}}}$" if row == 3 else ""))
            if mode == "single" and domain == "protein":
                before = numeric_artists(ax)
                ax.set_xticks([2e17/10**exp, 1e18/10**exp])
                assert_unchanged(ax, before)
            if row != 3:
                ax.tick_params(axis="x", labelbottom=False)
            symbol = {"N": r"\alpha", "D": r"\beta", "L": "s"}[field]
            y = -.10 if row != 3 else -.51
            ax.text(.5, y, rf"${symbol}={slope:.3f}$"+"\n"+rf"$R^2={r2:.3f}$",
                    transform=ax.transAxes, va="top", ha="center", fontsize=9, linespacing=1.15)
    if mode == "single":
        single.require((total_observations, total_vertices) == (227, 27), "Single displayed points differ")
        handles = [Line2D([], [], marker="D", color="#666666", linestyle="none", markersize=3, label="Supported"),
                   Line2D([], [], marker="D", color="#666666", mfc="white", linestyle="none", markersize=3, label="Weak fit"),
                   Line2D([], [], color="#666666", ls="--", lw=1, label="Extrapolation")]
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(.54, .016), ncol=3,
                   frameon=False, fontsize=9, handlelength=1.1, columnspacing=1)
    else:
        single.require((total_observations, total_vertices) == (297, 42), "Multi displayed points differ")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for col, legend in enumerate(legends):
        rect = legend.get_window_extent(renderer)
        lower = axes[1, col].get_window_extent(renderer).y1+3
        upper = axes[0, col].xaxis.label.get_window_extent(renderer).y0-3
        single.require(rect.height < upper-lower,
                       f"Budget legend needs more vertical room: {mode}/{DOMAINS[col]}: {rect.height} > {upper-lower}")
        desired_top = (lower+upper+rect.height)/2
        legend.set_bbox_to_anchor(((rect.x0+rect.x1)/2/fig.bbox.width, desired_top/fig.bbox.height),
                                  transform=fig.transFigure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for col, legend in enumerate(legends):
        rect = legend.get_window_extent(renderer)
        single.require(rect.y0 > axes[1, col].get_window_extent(renderer).y1+1,
                       f"Budget legend overlaps a frontier: {mode}/{DOMAINS[col]}")
        single.require(rect.y1 < axes[0, col].xaxis.label.get_window_extent(renderer).y0-1,
                       f"Budget legend overlaps parameter label: {mode}/{DOMAINS[col]}")
        if col:
            single.require(legends[col-1].get_window_extent(renderer).x1 < rect.x0,
                           "Neighboring budget legends overlap")
    save(fig, output, f"{mode}_to_single_all_domains", {"observations": total_observations,
         "vertices": total_vertices, "frozen_data_and_fits": True, "domain_order": list(DOMAINS)})


def cross_figure(output):
    style()
    source = read(SOURCE / "cross_domain_publication_20260914/source_single_to_all.json")
    by_id = {r["run_id"]: r for r in source["rows"]}
    single.require(len(by_id) == 136 and len(source["plan"]["unavailable"]) == 81, "Cross snapshot changed")
    fig, axes = plt.subplots(1, 5, figsize=(WIDTH, 3.8))
    fig.subplots_adjust(left=.115, right=.965, top=.975, bottom=.60, wspace=.65)
    names = []
    for col, (domain, ax) in enumerate(zip(DOMAINS, axes)):
        lines = [{"budget": line["budget"], "points": [by_id[r] for r in line["point_ids"]]}
                 for line in source["lines"][domain]]
        planned = [r for r in source["plan"]["rows"]+source["plan"]["unavailable"] if r["domain"] == domain]
        cross.cross_panel(ax, lines, planned)
        compact_axis(ax, ylabel="Loss" if col == 0 else "", xlabel=r"$N$ (M)")
        box = ax.get_position()
        center = (box.x0+box.x1)/2
        names.append(fig.text(center, .455, NAMES[col], ha="center", va="center", fontsize=10))
        budgets = sorted({p["budget"] for p in planned})
        values = [next(p["target_flops"] for p in planned if p["budget"] == b) for b in budgets]
        exponent = math.floor(math.log10(max(values)))
        handles = [Line2D([], [], color=single.COLORS[int(b[1:])], lw=1,
                          marker="o" if any(line["budget"] == b for line in lines) else None,
                          markersize=2, label=f"{v/10**exponent:.3g}"+
                          ("" if any(line["budget"] == b for line in lines) else r"$^{\dagger}$"))
                   for b, v in zip(budgets, values)]
        fig.legend(handles=handles, title=rf"$C/10^{{{exponent}}}$", title_fontsize=10,
                   bbox_to_anchor=(center, .36), loc="upper center", ncol=1,
                   frameon=False, fontsize=9, labelspacing=.18, handlelength=.8,
                   handletextpad=.35, borderaxespad=0)
    count = sum(len(line.get_xdata()) for ax in axes for line in ax.lines)
    single.require(count == 136, "Cross plotted point count changed")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, name, legend in zip(axes, names, fig.legends):
        upper = ax.xaxis.label.get_window_extent(renderer).y0-3
        lower = legend.get_window_extent(renderer).y1+3
        single.require(name.get_window_extent(renderer).height < upper-lower,
                       "Insufficient space for domain label at the target font size")
        name.set_y((upper+lower)/2/fig.bbox.height)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, name, legend in zip(axes, names, fig.legends):
        single.require(name.get_window_extent(renderer).y1 < ax.xaxis.label.get_window_extent(renderer).y0-1,
                       "Domain name overlaps parameter label")
        single.require(legend.get_window_extent(renderer).y1 < name.get_window_extent(renderer).y0-1,
                       "Domain name overlaps its legend")
    save(fig, output, "single_to_all_overview", {"observations": count, "unavailable": 81,
         "no_fit": True, "one_row": True, "per_panel_legends": True})


def comparison_figure(output):
    style()
    records = comparison.load_snapshot()["records"]
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH, 5.0))
    fig.subplots_adjust(left=.115, right=.965, top=.975, bottom=.28, wspace=.65, hspace=.75)
    for col, domain in enumerate(DOMAINS):
        cross.frontier_panel(axes[0, col], records[domain], "L")
        cross.nd_panel(axes[1, col], records[domain], peptide=domain == "peptide")
        compact_axis(axes[0, col], ylabel=r"Loss, $L^*$" if col == 0 else "", xlabel="")
        axes[0, col].xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/1e18:g}"))
        if domain == "protein":
            before = numeric_artists(axes[0, col])
            axes[0, col].set_xticks([5e17, 2e18])
            assert_unchanged(axes[0, col], before)
        compact_axis(axes[1, col], ylabel=r"$D^*$ (B)" if col == 0 else "", xlabel=r"$N^*$ (M)")
        if domain == "peptide":
            # Retain its exact inset data, enlarge it and use only endpoint ticks.
            inset = axes[1, col].child_axes[0]
            inset.set_axes_locator(None)
            inset.set_position([.32, .070, .145, .09])
            compact_axis(inset, ylabel="", xlabel="")
            inset.set_yticks([0, 60])
            inset.set_xticks([0, 90])
            inset.tick_params(labelsize=9, pad=1, length=1)
            inset.text(.5, 1.10, r"Peptide $D^*$ (M)", transform=inset.transAxes, ha="center", fontsize=9)
    fig.text(.54, .633, r"$C$ or $rC$ ($10^{18}$ FLOPs)", ha="center", fontsize=10)
    handles = [Line2D([], [], color=cross.GROUP_COLORS[g], lw=1.2, label=label)
               for g, label in zip(cross.GROUPS, (r"Single $\to$ single", r"Multi $\to$ single ($C$)", r"Multi $\to$ single ($rC$)"))]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.74, .09), ncol=1,
               frameon=False, fontsize=9, labelspacing=.28, handlelength=1.3)
    counts = [sum(len(c.get_offsets()) for ax in row for c in ax.collections) for row in axes]
    single.require(counts == [111, 111], "Comparison vertices changed")
    save(fig, output, "three_frontiers_2x5", {"vertices_per_row": counts,
         "protein_preexisting_sensitivity_retained": True, "peptide_inset_retained": True,
         "normalization": "(C,N,D,L) -> (rC,N,rD,L)", "no_refit": True})


def legacy_main_module():
    path = PAPER / "scripts/plot_scaling_publication.py"
    spec = importlib.util.spec_from_file_location("frozen_scaling_publication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main_inputs():
    old = legacy_main_module()
    rows, excluded = old.load_isoflop_rows()
    curves = old.fit_curves(rows, "loss_final")
    # Reconstruct the old rounded-table curves using its unchanged method and
    # verify all vertices/selections. Headline lines use saved coefficients.
    saved = table(SOURCE / "scaling_publication_20260826/isoflop_vertices.csv",
                  ("compute_flops", "n_star", "d_star_tokens", "l_star", "n_points_total", "n_points_vertex_fit"))
    single.require(len(curves) == len(saved) == 10 and len(rows) == 65, "Main frozen counts changed")
    for c, v in zip(curves, saved):
        for key, field in (("compute", "compute_flops"), ("n_star", "n_star"),
                           ("d_star", "d_star_tokens"), ("l_star", "l_star")):
            single.require(np.isclose(c[key], v[field], rtol=1e-8, atol=1e-9), "Main frozen vertices changed")
        single.require(len(c["rows"]) == v["n_points_total"] and sum(c["mask"]) == v["n_points_vertex_fit"],
                       "Main curve selection changed")
    metrics = read(SOURCE / "scaling_publication_20260826/figure_metrics.json")
    fits = [(metrics[key][slope], metrics[key]["intercept"], metrics[key]["r_squared"])
            for key, slope in (("n_scaling", "a"), ("d_scaling", "b"), ("loss_scaling", "slope"))]
    return old, rows, excluded, curves, metrics, fits


def main_figure(output):
    style()
    old, rows, excluded, curves, metrics, fits = main_inputs()
    flagship = metrics["small_flagship_validation"]
    small = {**flagship, "L": flagship["observed_loss"]}
    a1 = metrics["a1_low_compute_validation"]
    ci = {key: dict(zip(("lower", "upper"), metrics[name]["bootstrap_95_ci"]))
          for key, name in (("a", "n_scaling"), ("b", "d_scaling"))}
    fig = plt.figure(figsize=(WIDTH, 6.95))
    positions = [[.13, .565, .335, .285], [.635, .565, .35, .285],
                 [.13, .215, .335, .235], [.635, .215, .35, .235]]
    axes = [fig.add_axes(pos) for pos in positions]
    old.plot_isoflop_panel(axes[0], curves)
    # Remove the automatic colorbar layout and reserve an explicit strip.
    colorbar = fig.axes[-1]
    colorbar.set_axes_locator(None)
    colorbar.set_position([.13, .935, .335, .016])
    colorbar.xaxis.label.set_text(r"$C$ ($10^{18}$ FLOPs)")
    colorbar.xaxis.label.set_fontsize(10)
    colorbar.tick_params(labelsize=9)
    axes[0].set_position(positions[0])
    old.plot_n_scaling_panel(axes[1], curves, fits[0], ci["a"], a1, small)
    old.plot_d_scaling_panel(axes[2], curves, fits[1], ci["b"], a1, small)
    old.plot_loss_scaling_panel(axes[3], curves, fits[2], small)
    inset = axes[3].child_axes[0]
    inset.remove()
    handles = axes[0].get_legend().legend_handles
    axes[0].get_legend().remove()
    for i, ax in enumerate(axes):
        erase_notes(ax)
        ax.set_title("", loc="left")
        ax.set_xlabel(r"$N$" if i == 0 else r"$C$ (FLOPs)", fontsize=10)
        ax.set_ylabel(("Loss", r"$N^*$", r"$D^*$", r"Loss, $L^*$")[i], fontsize=10)
        ax.tick_params(labelsize=9, length=2, pad=2)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        if i:
            ax.set_xticks([1e18, 1e20])
        if i in (0, 3):
            ax.yaxis.set_major_locator(MaxNLocator(4))
        for c in ax.collections:
            if hasattr(c, "get_sizes"):
                c.set_sizes(np.minimum(c.get_sizes(), 22))
    fig.legend(handles, ["Used", r"Trimmed ($\Delta L>0.05$)", "Vertex"],
               loc="center", bbox_to_anchor=(.29, .905), ncol=1, frameon=False,
               fontsize=9, handlelength=1, labelspacing=.1)
    # Formula/CI blocks are outside data frames, never on the fitted lines.
    for ax, symbol, fit, interval in ((axes[1], "N", fits[0], ci["a"]),
                                      (axes[2], "D", fits[1], ci["b"])):
        ax.text(.02, 1.03, rf"${symbol}^*\propto C^{{{fit[0]:.3f}}}$, $R^2={fit[2]:.3f}$"+"\n"+
                f"95% CI [{interval['lower']:.3f}, {interval['upper']:.3f}]",
                transform=ax.transAxes, va="bottom", fontsize=9, linespacing=1.2)
    axes[3].text(.02, 1.03, rf"$L^*={fits[2][0]:.3f}\log_{{10}}C$"+"\n"+
                 rf"$+{fits[2][1]:.3f}$, $R^2={fits[2][2]:.3f}$",
                 transform=axes[3].transAxes, va="bottom", fontsize=9, linespacing=1.2)
    axes[2].text(.04, .88, "40.9B cap", transform=axes[2].transAxes,
                ha="left", va="bottom", fontsize=9, color=old.RED)
    # The former tiny inset becomes an external detail panel at full font size.
    detail = fig.add_axes([.685, .045, .295, .080])
    pred, observed = flagship["predicted_loss"], flagship["observed_loss"]
    detail.errorbar(0, pred, yerr=old.RUN_TO_RUN_SIGMA, fmt="D", ms=4,
                    mfc="white", mec=old.GREEN, color=old.GREEN, capsize=2)
    detail.scatter(1, observed, marker="P", s=24, color=old.GREEN)
    detail.plot([0, 1], [pred, observed], color=old.GREEN, lw=.8)
    detail.set_xlim(-.5, 1.5)
    detail.set_ylim(4.0305, 4.043)
    detail.set_yticks([4.035, 4.040])
    detail.set_xticks([0, 1], ["Pred.", "Obs."])
    detail.set_ylabel("Loss", fontsize=10)
    detail.tick_params(labelsize=9)
    detail.text(.5, 1.03, "0.55B detail", transform=detail.transAxes, ha="center", fontsize=9)
    fig.text(.13, .11, "Low-$C$: ratio 1.05\n0.55B: ratio 1.03\n"+
             rf"$\Delta L={flagship['loss_error']:+.4f}$ ({flagship['abs_error_in_run_to_run_sigma']:.2f}$\sigma$)",
             va="top", fontsize=9, linespacing=1.35)
    save(fig, output, "scaling_law_main", {"runs": len(rows), "curves": 10,
         "frozen_headline_coefficients": fits, "frozen_vertices_and_selections_verified": True,
         "tiny_inset_moved_outside": True})


def bpe_figure(output):
    """Use an 8.5x5.25-inch landscape appendix sheet for the dense DAG."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "representation"))
    import plot_bpe_shared_hierarchy as bpe
    import pipeline.finetune.fddfs_encode as encode
    p = bpe.p
    worker_root = Path("./storage/molscaling/data")
    p.BOOK = worker_root / p.BOOK.relative_to(p.DATA)
    p.VOCAB = worker_root / p.VOCAB.relative_to(p.DATA)
    encode.DEFAULT_GUIDE_PATH = worker_root / "fddfs_guide_recount_v1/guide_counts.json"
    root, labels, expansions, rules, paths, cases = bpe.prepare()
    nodes, connections, _ = bpe.design(root)
    original = read(SOURCE / "bpe_shared_hierarchy_20260915/selected_merges.json")
    single.require(paths == original["paths"] and root == original["root"]["id"], "BPE selected paths changed")
    style()
    width, height = 8.5, 5.25
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    placement = {
        "root": (.43, 3.2, .62, 1.1), "shared": (1.62, 3.2, 1.14, 1.72),
        "ncn": (3.06, 4.24, 1.24, 1.78), "amino": (3.06, 1.95, 1.24, 2.12),
        "imidazole": (4.61, 4.24, 1.28, 1.78), "adenine": (4.61, 1.95, 1.28, 2.12),
        "histidine": (7.0, 4.35, 2.7, 1.48), "dna": (7.0, 2.76, 2.7, 1.48),
        "rna": (7.0, 1.17, 2.7, 1.48),
    }
    for key, values in placement.items():
        nodes[key].update(dict(zip(("x", "y", "width", "height"), values)))
    for source, target, path in connections:
        a, b = nodes[source], nodes[target]
        before = {id(x) for x in ax.get_children()}
        ax.add_patch(FancyArrowPatch((a["x"]+a["width"]/2+.025, a["y"]),
                                    (b["x"]-b["width"]/2-.025, b["y"]),
                                    arrowstyle="-|>", mutation_scale=7, lw=.8, color="#91A0AE",
                                    linestyle="--" if len(path)>2 else "-", zorder=1))
        bpe.mark_new(ax, before, f"arrow-{source}-{target}")
    title_map = {"root": "Shared\natom", "shared": "Aromatic C–N",
                 "ncn": "N–C–N motif", "amino": "Amino-\nheteroaromatic\nmotif",
                 "imidazole": "Imidazole\nmotif", "adenine": "Adenine-related\nmotif"}
    for key, node in nodes.items():
        x, y, w, h = (node[t] for t in ("x", "y", "width", "height"))
        left, bottom = x-w/2, y-h/2
        before = {id(t) for t in ax.get_children()}
        ax.add_patch(FancyBboxPatch((left, bottom), w, h, boxstyle="round,pad=0.01,rounding_size=.04",
                                   facecolor="#F5F9FB" if key in ("root", "shared") else "white",
                                   edgecolor="#B4CCD8", lw=.7, zorder=2))
        ax.text(x, bottom+h-.08, title_map.get(key, node["title"]), ha="center", va="top",
                fontsize=10, linespacing=1.05, color="#344250", zorder=6)
        case = cases[node["case"]]
        if key == "root":
            ax.add_patch(Circle((x, y-.02), .13, facecolor="white", edgecolor=p.OLD, lw=1, zorder=5))
            ax.text(x, y-.02, "N", ha="center", va="center", fontsize=14, color=p.OLD, zorder=6)
        else:
            interval, prior = case["intervals"][node["token"]], case["intervals"][node["prior"]]
            seq = p.abbreviated_sequence(expansions[node["token"]], labels)
            last = key in ("histidine", "dna", "rna")
            sequence_width = 1.16 if last else w-.16
            lines = textwrap.wrap(seq, width=int(sequence_width*72/5.42), break_long_words=False, break_on_hyphens=False)
            single.require(" ".join(lines) == seq, "BPE sequence symbols lost")
            sequence_height = len(lines)*.15
            graph_bottom = bottom+.23 if last else bottom+.30+sequence_height
            graph_top = bottom+h-(.56 if key == "amino" else .40 if key in ("adenine", "imidazole") else .30)
            single.require(graph_top > graph_bottom+.24, f"BPE graph squeezed: {key}")
            readable_fragment(ax, p, case, interval, prior,
                              (left+.03, graph_bottom, 1.40 if last else w-.06, graph_top-graph_bottom))
            # Chemical atom glyphs are 9 pt, with the original selected atoms,
            # exact bonds and one-hop reference context retained.
            for t in ax.texts:
                if id(t) not in before and t.get_text() in ("C", "N", "O", "P", "S"):
                    t.set_fontsize(9)
            sx = left+2.07 if last else x
            sy = bottom+.23+(graph_top-graph_bottom)/2 if last else bottom+.22+sequence_height/2
            ax.text(sx, sy, "\n".join(lines), ha="center", va="center",
                    fontsize=9, family="DejaVu Sans Mono", linespacing=1.1, color="#687888", zorder=6)
        ax.text(x, bottom+.055, rf"$T_{{{node['token']}}}$", ha="center", va="bottom", fontsize=10, zorder=6)
        bpe.mark_new(ax, before, "card-"+key)
    handles = [Line2D([], [], marker="o", mfc="white", mec=color, color="none", markersize=4)
               for color in (p.OLD, p.ADDED, p.CONTEXT)]
    handles.append(Line2D([], [], color="#91A0AE", ls="--", lw=.8))
    fig.legend(handles, ["Inherited", "Added", "Reference context", "Two merges"],
               loc="center", bbox_to_anchor=(.51, .032), ncol=4, frameon=False,
               fontsize=10, handlelength=1.2, columnspacing=1.8)
    # Preserve grouping/provenance for the existing native-PPTX exporter.
    evidence = {**original, "display_nodes": list(nodes.values()), "presentation": "ICLR landscape, 9–10 pt"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_merges.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n")
    save(fig, output, "bpe_merge_hierarchy", {"landscape_appendix": True,
         "same_root_paths_atoms_and_bonds": True, "abbreviated_sequence_symbols_retained": True,
         "nodes": len(nodes), "edges": len(connections)})


def readable_fragment(ax, p, case, interval, prior_interval, box):
    """Same RDKit coordinates/selected graph; fit to an inch-sized card."""
    from matplotlib.patches import Circle
    selected, prior = p.covered_atoms(case, interval), p.covered_atoms(case, prior_interval)
    single.require(prior < selected and p.component_count(selected, case["edges"]) == 1,
                   "BPE topology changed")
    visible = set(selected)
    for u, v, _ in case["edges"]:
        if u in selected or v in selected:
            visible.update((u, v))
    xx, yy, width, height = box
    coords = case["coords"]
    occupied = coords[sorted(visible)]
    center = (occupied.max(axis=0)+occupied.min(axis=0))/2
    span = np.maximum(np.ptp(occupied, axis=0), .8)
    scale = min((width-.13)/span[0], (height-.13)/span[1], .46)
    pts = (coords-center)*scale+np.array([xx+width/2, yy+height/2])
    distance = min(np.linalg.norm(pts[a]-pts[b]) for a in selected for b in selected if a != b)
    single.require(distance >= .12, f"BPE glyph crowding: {distance:.3f} in, card {box}")
    for u, v, order in case["edges"]:
        if u not in visible or v not in visible or not ({u, v}&selected):
            continue
        a, b = pts[u], pts[v]
        color = "#5F6B77" if u in selected and v in selected else p.CONTEXT
        delta = b-a
        normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)*.014
        shifts = (0,) if order in (1, 4) else (-1, 1) if order == 2 else (-1.5, 0, 1.5)
        for shift in shifts:
            xy = np.array([a+shift*normal, b+shift*normal])
            ax.plot(xy[:,0], xy[:,1], color=color, lw=.65, ls="--" if order == 4 else "-", zorder=3)
    for atom in sorted(visible):
        if atom not in selected:
            ax.add_patch(Circle(pts[atom], .027, fc="white", ec=p.CONTEXT, lw=.5, zorder=4))
            continue
        color = p.OLD if atom in prior else p.ADDED
        ax.add_patch(Circle(pts[atom], min(.073, distance*.43), fc="white", ec=color, lw=.8, zorder=5))
        symbol = {6:"C",7:"N",8:"O",15:"P",16:"S"}[int(case["graph"]["node_features"][atom][0])]
        ax.text(*pts[atom], symbol, fontsize=9, ha="center", va="center", color=color, zorder=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figures", nargs="+", choices=("single", "multi", "cross", "comparison", "main", "bpe"),
                        default=["single", "multi", "cross", "comparison"])
    args = parser.parse_args()
    single.require(socket.gethostname() == args.expected_host and "worker" in args.expected_host,
                   "Use the confirmed allocated worker")
    if (args.output / "rendering.json").exists():
        MANIFEST.update(read(args.output / "rendering.json")["figures"])
    for name in args.figures:
        if name in ("single", "multi"):
            scaling_grid(args.output, name)
        else:
            {"cross": cross_figure, "comparison": comparison_figure,
             "main": main_figure, "bpe": bpe_figure}[name](args.output)
    (args.output / "rendering.json").write_text(json.dumps({
        "worker": socket.gethostname(), "template_text_width_inches": WIDTH,
        "body_font_pt": 10, "min_nominal_figure_font_pt": 9,
        "data_or_fit_changes": False, "original_files_overwritten": False,
        "figures": MANIFEST}, indent=2)+"\n")
    print("ICLR_FIGURES_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
