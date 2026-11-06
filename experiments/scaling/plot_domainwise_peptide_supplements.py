"""Descriptive fixed-data plots, including sparsely sampled peptide supplements.

Missing N/data combinations stay missing. No fitting, interpolation, or
IsoFLOP bracketing is performed here. Inputs are validated result-table rows.
This module does not replace the running watcher until explicitly integrated.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPOCH_COLORS = {
    0.25: "#3288ff", 0.5: "#f58232", 1.0: "#43b96d",
    2.0: "#eb6aab", 4.0: "#9470ed",
}


def peptide_grid(completed, pending=()):
    """Return measured slices and a NaN-masked grid, never synthesized losses."""
    rows = [p for p in completed if p["domain"] == "peptide"]
    missing = [p for p in pending if p["domain"] == "peptide"]
    planned = rows + missing
    ns = sorted({int(p["parameter_count"]) for p in planned})
    epochs = sorted({float(p["target_epoch_fraction"]) for p in planned})
    if not ns:
        raise ValueError("No peptide observations or planned points")
    coordinates = [(int(p["parameter_count"]), float(p["target_epoch_fraction"]))
                   for p in planned]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Duplicate peptide N/data coordinate in result tables")
    unknown = set(epochs) - set(EPOCH_COLORS)
    if unknown:
        raise ValueError(f"Peptide data budgets outside the declared design: {unknown}")
    values = np.full((len(ns), len(epochs)), np.nan)
    slices = {}
    for epoch in epochs:
        points = sorted((p for p in rows if float(p["target_epoch_fraction"]) == epoch),
                        key=lambda p: int(p["parameter_count"]))
        slices[epoch] = points
        for point in points:
            loss = float(point["validation_loss"])
            if not np.isfinite(loss) or loss <= 0:
                raise ValueError(f"Invalid peptide validation loss: {point['run_id']}")
            values[ns.index(int(point["parameter_count"])), epochs.index(epoch)] = loss
    return ns, epochs, values, slices, len(rows), len(planned)


def equal_data_figure(completed, pending=()):
    ns, epochs, values, slices, done, planned = peptide_grid(completed, pending)
    pending_coordinates = {
        (int(p["parameter_count"]), float(p["target_epoch_fraction"]))
        for p in pending if p["domain"] == "peptide"
    }
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(14, 5.5))
    for epoch, points in slices.items():
        ax.plot([int(p["parameter_count"]) / 1e6 for p in points],
                [float(p["validation_loss"]) for p in points], "o-",
                color=EPOCH_COLORS[epoch], label=f"{epoch:g} token-epoch")
    ax.set_xscale("log")
    ax.set_xticks(np.array(ns) / 1e6, [f"{n/1e6:.2f}" for n in ns],
                  rotation=30, ha="right", fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("Total parameters / M (log scale)")
    ax.set_ylabel("Validation causal NLL")
    ax.set_title("Measured equal-data slices (NOT equal compute)", pad=48)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
              fontsize=8.5, frameon=False)
    ax.grid(alpha=0.2)
    cmap = plt.colormaps["YlGnBu_r"].copy()
    cmap.set_bad("#e5e7eb")
    im = heat.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto")
    heat.set_xticks(range(len(epochs)), [f"{e:g}" for e in epochs])
    heat.set_yticks(range(len(ns)), [f"{n/1e6:.2f}M" for n in ns])
    heat.set_xlabel("Training data / token-epochs")
    heat.set_ylabel("Total parameters")
    for i in range(len(ns)):
        for j in range(len(epochs)):
            value = values[i, j]
            if np.isfinite(value):
                label = f"{value:.4f}"
            else:
                label = "P" if (ns[i], epochs[j]) in pending_coordinates else "—"
            heat.text(j, i, label, ha="center", va="center", fontsize=9,
                      color="white" if np.isfinite(value) and value < 7 else "#0f172a")
    for j in range(len(epochs)):
        if np.isfinite(values[:, j]).any():
            i = np.nanargmin(values[:, j])
            heat.scatter(j, i, s=1200, marker="s", facecolors="none",
                         edgecolors="#f59e0b", linewidths=2)
    fig.colorbar(im, ax=heat, label="Validation NLL", shrink=0.9)
    fig.suptitle(f"Peptide: {done}/{planned} complete runs | fixed data, not IsoFLOP bracketing",
                 fontsize=14)
    fig.text(0.5, 0.012, "P: planned, not measured. Dash: not planned. No interpolated losses; lines connect measured points only.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def overlay_panel(ax, completed, pending=(), *, format_axis):
    ns, epochs, _, slices, done, planned = peptide_grid(completed, pending)
    for epoch in epochs:
        points = slices[epoch]
        color = EPOCH_COLORS[epoch]
        ax.plot([int(p["parameter_count"]) / 1e6 for p in points],
                [float(p["validation_loss"]) for p in points], "o-", color=color,
                markersize=4.7, linewidth=1.25, label=f"{epoch:g} token-epoch")
        if points:
            best = min(points, key=lambda p: float(p["validation_loss"]))
            marker = ax.scatter(int(best["parameter_count"]) / 1e6,
                                float(best["validation_loss"]), s=95,
                                facecolors="none", edgecolors=color, linewidths=1.2, zorder=5)
            marker.set_gid(f"observed_peptide_minimum:{epoch:g}")
    format_axis(ax, [n / 1e6 for n in ns])
    ax.set_title(f"多肽 · {done}/{planned} 点 · 固定数据量（非等 FLOPs）",
                 loc="left", fontsize=15, pad=66)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=3, fontsize=9,
              frameon=False, borderaxespad=0, labelspacing=0.75)
