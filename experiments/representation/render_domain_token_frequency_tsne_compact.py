#!/usr/bin/env python3
"""Redraw saved t-SNE coordinates without axes; never sample or fit a projection."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import socket

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DOMAINS = ("small_molecule", "peptide", "protein", "dna", "rna")
NAMES = ("Small molecule", "Peptide", "Protein", "DNA", "RNA")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#6B55A3")
STEM = "domain_token_frequency_tsne_centered"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--output-stem", default=STEM)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--vertical-scale", type=float, default=1.0,
                        help="Display y/x scale; 0.8 flattens the point cloud by 20%% without changing coordinates or fonts.")
    args = parser.parse_args()
    if "worker" not in args.expected_host or socket.gethostname() != args.expected_host:
        raise RuntimeError("Run on the confirmed allocated CPU worker")
    if not np.isfinite(args.vertical_scale) or args.vertical_scale <= 0:
        raise ValueError("--vertical-scale must be finite and positive")
    if not args.output_stem or Path(args.output_stem).name != args.output_stem:
        raise ValueError("--output-stem must be a filename stem, not a path")

    source = args.input_dir.resolve()
    if args.no_overwrite:
        for extension in ("png", "pdf", "svg", "json"):
            target = source / f"{args.output_stem}.{extension}"
            if target.exists():
                raise FileExistsError(target)
    summary = json.loads((source / "summary.json").read_text())
    embedding = np.load(source / "embedding.npy", allow_pickle=False)
    with (source / "coordinates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    domain_ids = np.asarray([int(row["domain_id"]) for row in rows])
    counts = Counter(row["domain"] for row in rows)
    if embedding.shape != (summary["records"], 2) or len(rows) != len(embedding):
        raise ValueError("Saved coordinates and original record count disagree")
    if dict(counts) != summary["domain_counts"]:
        raise ValueError("Saved domain counts disagree with the original summary")
    if not all(int(row["sample_index"]) == i and row["domain"] == DOMAINS[domain_ids[i]]
               for i, row in enumerate(rows)):
        raise ValueError("Original sample order or domain mapping is inconsistent")
    csv_coordinates = np.asarray([[float(row["tsne_1"]), float(row["tsne_2"])] for row in rows])
    np.testing.assert_array_equal(embedding, csv_coordinates)
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite saved coordinates")

    font_path = font_manager.findfont("Nimbus Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Nimbus Roman", "font.size": 12.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8.6, 5.3 * args.vertical_scale), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    scatter_settings = summary["protocol"]["scatter"]
    points = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=[COLORS[i] for i in domain_ids],
                        s=scatter_settings["marker_size"], alpha=scatter_settings["alpha"],
                        edgecolors="none", rasterized=False)
    # Center the main circular cloud, not the cloud-plus-legend bounding box.
    # Quantiles define only the visual center; every saved point stays visible.
    core_left, core_right = np.quantile(embedding[:, 0], [.0005, .9995])
    cloud_center_x = float((core_left + core_right) / 2)
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    cloud_center_y = float((y_min + y_max) / 2)
    y_span = float(y_max - y_min) * 1.03
    # Keep the original horizontal scale and limits while reducing display height.
    # Only the plotting transform changes; the saved t-SNE coordinates do not.
    x_span = y_span * fig.get_figwidth() / fig.get_figheight() * args.vertical_scale
    ax.set_xlim(cloud_center_x - x_span / 2, cloud_center_x + x_span / 2)
    ax.set_ylim(cloud_center_y - y_span / 2, cloud_center_y + y_span / 2)
    ax.set_aspect(args.vertical_scale, adjustable="box")
    ax.set_axis_off()
    handles = [Line2D([], [], marker="o", ls="none", color=color, markersize=5.5,
                      markeredgewidth=0, label=name) for name, color in zip(NAMES, COLORS)]
    # The gap beside the circular cloud is used for the legend; isolated points
    # farther right are retained and are checked for overlap below.
    legend = fig.legend(handles=handles, loc="center left",
                        bbox_to_anchor=(62.5, cloud_center_y), bbox_transform=ax.transData,
                        ncol=1, frameon=False, fontsize=11, handletextpad=.35,
                        labelspacing=.9, borderaxespad=0, borderpad=0,
                        handlelength=.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    rendered = ax.transData.transform(embedding)
    padding = 3 * fig.dpi / 72
    in_legend = ((rendered[:, 0] >= legend_box.x0 - padding)
                 & (rendered[:, 0] <= legend_box.x1 + padding)
                 & (rendered[:, 1] >= legend_box.y0 - padding)
                 & (rendered[:, 1] <= legend_box.y1 + padding))
    if in_legend.any():
        raise ValueError(f"Legend would cover points: {embedding[in_legend].tolist()}")
    center_on_canvas = fig.transFigure.inverted().transform(
        ax.transData.transform((cloud_center_x, cloud_center_y)))
    np.testing.assert_allclose(center_on_canvas, [.5, .5], atol=1e-12)
    unit_steps = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
    display_scale = float((unit_steps[2, 1] - unit_steps[0, 1]) /
                          (unit_steps[1, 0] - unit_steps[0, 0]))
    np.testing.assert_allclose(display_scale, args.vertical_scale, atol=1e-12)
    if not (0 <= legend_box.x0 < legend_box.x1 <= fig.bbox.width and
            0 <= legend_box.y0 < legend_box.y1 <= fig.bbox.height):
        raise ValueError("Legend extends beyond the fixed canvas")
    np.testing.assert_array_equal(points.get_offsets(), embedding)
    if not all(ax.get_xlim()[0] < x < ax.get_xlim()[1] and
               ax.get_ylim()[0] < y < ax.get_ylim()[1] for x, y in embedding):
        raise ValueError("Plot limits clip saved observations")
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = source / f"{args.output_stem}.{extension}"
        # Do not tight-crop: the intentional left blank margin is essential for
        # keeping the circular cloud centered when inserted in the manuscript.
        fig.savefig(path, dpi=450, bbox_inches=None, pad_inches=0, facecolor="white")
        outputs.append(str(path))
        print(f"Saved {path}", flush=True)
    plt.close(fig)
    audit = {"source": str(source), "records": len(embedding), "domain_counts": dict(counts),
             "projection_refit": False, "resampled": False, "coordinates_changed": False,
             "sample_order_changed": False, "palette": dict(zip(DOMAINS, COLORS)),
             "marker_size": scatter_settings["marker_size"], "alpha": scatter_settings["alpha"],
             "font": "Nimbus Roman (Times-compatible)", "font_path": font_path,
             "axes_visible": False, "legend": "right, vertically centered, no point overlap",
             "figure_size_inches": fig.get_size_inches().tolist(),
             "display_y_to_x_scale": display_scale,
             "text_and_marker_shapes_stretched": False,
             "cloud_center_data": [cloud_center_x, cloud_center_y],
             "cloud_center_canvas": center_on_canvas.tolist(),
             "center_definition": "Midpoint of 0.05/99.95-percentile x extent; all points retained",
             "left_and_right_margin_equal": True, "tight_crop": False,
             "original_figures_overwritten": False, "outputs": outputs}
    (source / f"{args.output_stem}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Validated: {len(embedding):,} unchanged points; no clipping or legend overlap", flush=True)


if __name__ == "__main__":
    main()
