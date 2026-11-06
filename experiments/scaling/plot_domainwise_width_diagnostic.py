#!/usr/bin/env python3
"""Worker-only plots of recorded architecture diagnostics.

Reuses recorded coefficients without refitting, filtering observations, opening
checkpoints, or changing the primary gallery's bracketing decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np

from plot_domainwise_completed_snapshot import plt, save_figure


def read_diagnostic(path):
    points = None
    fits = {}
    for line in path.read_text().splitlines():
        if line.startswith("ALL_TEN_POINTS "):
            if points is not None:
                raise ValueError("duplicate observation record")
            points = json.loads(line.removeprefix("ALL_TEN_POINTS "))
        elif line.startswith("FIT_DIAGNOSTIC "):
            fit = json.loads(line.removeprefix("FIT_DIAGNOSTIC "))
            if fit["name"] in fits:
                raise ValueError(f"duplicate fit: {fit['name']}")
            fits[fit["name"]] = fit
    if not points or not {"quadratic", "width_offset_diagnostic"}.issubset(fits):
        raise ValueError("expected observations and both recorded diagnostic fits")
    if (len(fits["quadratic"]["coefs_centered_logN"]) != 3
            or len(fits["width_offset_diagnostic"]["coefs_centered_logN"]) != 4):
        raise ValueError("expected quadratic and quadratic-plus-width coefficients")
    if {p["hidden_size"] for p in points} != {256, 384}:
        raise ValueError("this diagnostic uses the recorded H=384 indicator")
    return points, fits


def make_figure(points, fits, title, annotate):
    log_n = np.log10([p["N"] for p in points])
    center = log_n.mean()
    grid = np.linspace(log_n.min(), log_n.max(), 300)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.78, bottom=0.19)
    fig.suptitle(title, fontsize=16, y=0.98)
    primary = fits["quadratic"]["coefs_centered_logN"]
    offset = fits["width_offset_diagnostic"]["coefs_centered_logN"]
    ax.plot(10**grid / 1e6, np.polyval(primary, grid - center),
            color="#64748b", ls="--", lw=2, label="Primary all-point quadratic")
    colors = {256: "#0284c7", 384: "#ea580c"}
    for width, marker in ((256, "o"), (384, "s")):
        group = [p for p in points if p["hidden_size"] == width]
        ax.scatter([p["N"] / 1e6 for p in group], [p["loss"] for p in group],
                   c=colors[width], marker=marker, s=58, zorder=3,
                   label=f"H={width} observations ({len(group)})")
        local = np.linspace(np.log10(min(p["N"] for p in group)),
                            np.log10(max(p["N"] for p in group)), 150)
        ax.plot(10**local / 1e6,
                np.polyval(offset[:3], local - center) + offset[3] * (width == 384),
                color=colors[width], ls=":", lw=2,
                label=f"H={width} offset diagnostic")
    by_id = {p["run_id"]: p for p in points}
    for i, run_id in enumerate(annotate):
        p = by_id[run_id]
        ax.annotate(f"{p['N'] / 1e6:.2f}M, {p['layers']} layers\n{p['loss']:.5f}",
                    (p["N"] / 1e6, p["loss"]),
                    xytext=(12, -32 if i % 2 == 0 else 20), textcoords="offset points",
                    color=colors[p["hidden_size"]], fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": colors[p["hidden_size"]]})
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 1.5, 2, 3, 4, 6)))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel("Total parameters / M (log scale)")
    ax.set_ylabel("Full-validation loss (nat/token)")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=9)
    fig.text(0.125, 0.055,
             f"All {len(points)} observations retained; recorded fits, no new fitting.\n"
             "Single seed; width and depth co-vary. Diagnostic only; primary bracketing unchanged.",
             fontsize=10, color="#475569")
    return fig


def read_local_review(path, points_record, local_record):
    required = {points_record, local_record, "PRIMARY_ALL_POINTS"}
    records = {}
    for line in path.read_text().splitlines():
        name, _, payload = line.partition(" ")
        if name in required:
            if name in records:
                raise ValueError(f"duplicate recorded diagnostic: {name}")
            records[name] = json.loads(payload)
    return (records[points_record], records["PRIMARY_ALL_POINTS"]["analysis"],
            records[local_record])


def make_local_figure(points, primary, local, title):
    """Show all observations and the recorded same-width fit, with a valley zoom."""
    by_n = {p["parameter_count"]: p for p in points}
    local_points = [by_n[n] for n in local["N"]]
    widths = {p["hidden_size"] for p in local_points}
    if len(widths) != 1:
        raise ValueError("the recorded same-width diagnostic mixes model widths")
    width = widths.pop()
    fit = primary["quadratic_fit"]
    primary_coef = [fit[k] for k in (
        "quadratic_coefficient_a", "linear_coefficient_b", "intercept_c")]
    global_vertex = fit["vertex_parameter_count"] / 1e6
    local_vertex = local["vertex_N"] / 1e6
    grid = np.geomspace(min(by_n), max(by_n), 400)
    local_grid = np.geomspace(min(local["N"]), max(local["N"]), 200)
    best = min(points, key=lambda p: p["validation_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(top=0.73, bottom=0.22, wspace=0.23)
    fig.suptitle(title, fontsize=16, y=0.99)
    palette = ("#0284c7", "#ea580c", "#16a34a", "#9333ea")
    for ax in axes:
        for i, h in enumerate(sorted({p["hidden_size"] for p in points})):
            group = [p for p in points if p["hidden_size"] == h]
            ax.scatter([p["parameter_count"] / 1e6 for p in group],
                       [p["validation_loss"] for p in group], s=46,
                       color=palette[i % len(palette)], zorder=3,
                       label=f"H={h} observations ({len(group)})")
        ax.plot(grid / 1e6, np.polyval(primary_coef, np.log10(grid)),
                color="#64748b", ls="--", lw=2,
                label=f"All {len(points)} points: N*={global_vertex:.2f}M, R²={fit['r_squared']:.3f}")
        ax.plot(local_grid / 1e6,
                np.polyval(local["quadratic_coefficient"], np.log10(local_grid)),
                color="#0284c7", lw=2,
                label=f"H={width} diagnostic: N*={local_vertex:.2f}M, R²={local['R2']:.3f}")
        ax.scatter([best["parameter_count"] / 1e6], [best["validation_loss"]],
                   s=135, facecolors="none", edgecolors="#111827", lw=1.5, zorder=4,
                   label=f"Observed best: {best['parameter_count'] / 1e6:.2f}M")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 3, 4, 6)))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlabel("Total parameters / M (log scale)")
        ax.set_ylabel("Full-validation loss (nat/token)")
        ax.grid(alpha=0.2)
    axes[0].set_title("All model sizes")
    axes[1].set_title("Valley detail (same observations)")
    axes[1].set_xlim(min(local["N"]) / 1e6 * 0.97,
                     max(max(local["N"]) / 1e6, global_vertex) * 1.16)
    axes[1].set_ylim(min(p["validation_loss"] for p in local_points) - 0.08,
                     max(p["validation_loss"] for p in local_points) + 0.04)
    axes[1].xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 2.5, 3, 4, 5, 6, 8)))
    axes[1].axvline(global_vertex, color="#64748b", lw=1, ls=":")
    axes[1].axvline(local_vertex, color="#0284c7", lw=1, ls=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False, fontsize=9)
    fig.text(0.08, 0.07,
             f"All {len(points)} observations retained. Recorded fits only; no new fitting. "
             f"The H={width} curve uses all {len(local_points)} same-width observations.\n"
             "Single seed; width and depth co-vary. Diagnostic only; primary bracketing status unchanged.",
             fontsize=10, color="#475569")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--diagnostic-log", type=Path)
    source.add_argument("--local-review", type=Path)
    parser.add_argument("--points-record", default="ALL_C2_POINTS")
    parser.add_argument("--local-record", default="H128_DIAGNOSTIC_ONLY")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="dna_c4_width_diagnostic")
    parser.add_argument("--title", default="DNA C4: architecture diagnostic")
    parser.add_argument("--annotate", nargs="*", default=[])
    args = parser.parse_args()
    if args.local_review:
        points, primary, local = read_local_review(
            args.local_review, args.points_record, args.local_record)
        fig = make_local_figure(points, primary, local, args.title)
    else:
        points, fits = read_diagnostic(args.diagnostic_log)
        fig = make_figure(points, fits, args.title, args.annotate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output_dir, args.name)
    print(json.dumps({"source": str(args.local_review or args.diagnostic_log), "point_count": len(points),
                      "output_prefix": str(args.output_dir / args.name),
                      "primary_bracketing_changed": False}))


if __name__ == "__main__":
    main()
