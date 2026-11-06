#!/usr/bin/env python3
"""Transpose columns 3–4 of the frozen comparison into a publication 2x5 figure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import socket

import numpy as np

import plot_cross_domain_publication as pub


SOURCE = pub.OUTPUT / "source_frontiers.json"
OUTPUT = pub.PAPER / "figures/frontier_comparison_2x5_20260915"


def load_snapshot(source=SOURCE):
    snapshot = pub.read(source)
    records = snapshot["records"]
    pub.require(set(records) == set(pub.DOMAINS), "The five source domains must be retained")
    for domain in pub.DOMAINS:
        record = records[domain]
        pub.validate_normalization(record)
        for group, expected in zip(pub.GROUPS, pub.EXPECTED_VERTICES[domain]):
            frontier = record[group]
            vertices = frontier["vertices"]
            pub.require(len(vertices) == expected, f"Saved vertex count changed: {domain}/{group}")
            pub.require(frontier["C_range"] == [min(v["C"] for v in vertices), max(v["C"] for v in vertices)],
                        f"Saved compute range changed: {domain}/{group}")
            pub.require(all(np.isfinite(v[f]) and v[f] > 0 for v in vertices for f in ("C", "N", "D", "L")),
                        f"Invalid saved vertex: {domain}/{group}")
    return snapshot


def comparison_figure(records):
    pub.style()
    fig, axes = pub.plt.subplots(2, 5, figsize=(13.4, 5.5))
    fig.subplots_adjust(left=.057, right=.99, bottom=.16, top=.97, wspace=.34, hspace=.43)
    for col, domain in enumerate(pub.DOMAINS):
        record = records[domain]
        # Reuse the exact old panel renderers: coefficients, vertices, viewports,
        # hollow markers, dashed extensions and the peptide inset are unchanged.
        pub.frontier_panel(axes[0, col], record, "L")
        pub.nd_panel(axes[1, col], record, peptide=domain == "peptide")
        axes[0, col].set_xlabel(r"Compute, $C$ or $rC$ (FLOPs)")
        axes[1, col].set_xlabel(r"Parameters, $N^*$ (M)")
        if col:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")
    # Keep only the essential three-series key. Domain order, marker semantics
    # and ratios belong in the companion caption, not another block of subtitles.
    handles = [pub.Line2D([], [], color=pub.GROUP_COLORS[group], lw=1.7, label=pub.GROUP_LABELS[group])
               for group in pub.GROUPS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.524, .045),
               ncol=3, frameon=False, fontsize=8, handlelength=2.0, columnspacing=2.4)
    pub.check_bottom_legends(fig, axes.flat, [])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for col, domain in enumerate(pub.DOMAINS):
        pub.require(axes[0, col].xaxis.label.get_window_extent(renderer).y0
                    > axes[1, col].get_window_extent(renderer).y1 + 2,
                    f"Top-row axis label overlaps the lower panel: {domain}")
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    pub.require("worker" in args.expected_host and socket.gethostname() == args.expected_host,
                "Render on the confirmed allocated CPU worker")
    snapshot = load_snapshot(args.source)
    fig, axes = comparison_figure(snapshot["records"])
    displayed = [sum(len(collection.get_offsets()) for ax in axes[row] for collection in ax.collections)
                 for row in range(2)]
    pub.require(displayed == [111, 111], f"Unexpected displayed vertices: {displayed}")
    args.output.mkdir(parents=True, exist_ok=True)
    files = pub.export(fig, args.output / "three_frontiers_2x5", args.dpi)
    pub.write(args.output / "source_frontiers.json", snapshot)
    pub.write(args.output / "rendering.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "worker": socket.gethostname(),
        "source": str(args.source), "files": files, "dpi": args.dpi,
        "layout": {"rows": ["L* versus C or rC", "D* versus N*"], "columns": list(pub.DOMAINS),
                   "source_columns_one_based": [3, 4], "shape": [2, 5], "size_inches": [13.4, 5.5]},
        "displayed_vertices_per_main_row": displayed, "peptide_inset_retained": True,
        "data_unchanged": True, "fits_unchanged": True, "new_exclusions": False,
        "new_checkpoint_evaluations": False, "original_artwork_overwritten": False,
        "target_accounting": "(C,N,D,L) -> (rC,N,rD,L)",
        "scientific_notes": {
            "loss": "Single validation versus multi training-loss proxy; exploratory, not a matched held-out comparison",
            "nd": "Saved per-budget quadratic vertices in ascending compute order; linear axes, no new fit",
            "protein": "Retains the saved sensitivity excluding multi C=3.5e18 and its target-accounted counterpart",
        },
    })
    print("FRONTIER_COMPARISON_2X5_COMPLETE", args.output, files, displayed, flush=True)


if __name__ == "__main__":
    main()
