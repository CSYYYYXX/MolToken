#!/usr/bin/env python3
"""Combine completed fixed-FLOPs runs into overlay, facet, and CSV outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_single_matrix_plotter() -> Any:
    path = Path(__file__).with_name("plot_fixed_flops_sweep.py")
    spec = importlib.util.spec_from_file_location("fixed_flops_plotter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_completed(matrix_paths: list[Path]) -> list[dict[str, Any]]:
    rows_by_run: dict[str, dict[str, Any]] = {}
    for matrix_path in matrix_paths:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        for row in matrix["rows"]:
            out_dir = Path(row["out_dir"])
            summary_path = out_dir / "summary.json"
            run_config_path = out_dir / "run_config.json"
            final_dir = out_dir / "final"
            if not summary_path.exists() or not final_dir.is_dir():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
            if int(summary.get("global_step", -1)) < int(row["max_steps"]):
                continue
            loss = summary.get("final_logged_train_loss")
            if loss is None or not math.isfinite(float(loss)):
                continue
            planned_actual_tokens = int(row["actual_tokens"])
            parameter_count = int(row["parameter_count"])
            observed_train_loss_tokens_local = int(summary.get("total_train_loss_tokens", 0))
            world_size = int(run_config.get("world_size", 1))
            observed_train_loss_tokens_global = observed_train_loss_tokens_local * world_size
            observed_flops_6nd = 6.0 * parameter_count * observed_train_loss_tokens_global
            target_flops_6nd = float(row["target_flops_6nd"])
            rows_by_run[str(out_dir)] = {
                "target_label": row["flops_target_label"],
                "target_flops_6nd": target_flops_6nd,
                "model_id": row["model_id"],
                "model_label": row["model_label"],
                "parameter_count": parameter_count,
                "epoch_fraction": float(row["epoch_fraction"]),
                "over_one_epoch": bool(row.get("over_one_epoch", float(row["epoch_fraction"]) > 1.0)),
                "max_steps": int(row["max_steps"]),
                "global_step": int(summary["global_step"]),
                "training_loss": float(loss),
                "trailing_train_loss": summary.get("trailing_train_loss"),
                "actual_tokens": planned_actual_tokens,
                "actual_token_budget_b": float(row["actual_token_budget_b"]),
                "world_size": world_size,
                "observed_train_loss_tokens_local": observed_train_loss_tokens_local,
                "observed_train_loss_tokens_global": observed_train_loss_tokens_global,
                "observed_token_relative_error": (
                    (observed_train_loss_tokens_global - planned_actual_tokens) / planned_actual_tokens
                ),
                "actual_flops_6nd": float(row["flops_6nd"]),
                "observed_flops_6nd": observed_flops_6nd,
                "observed_flops_relative_error": (observed_flops_6nd - target_flops_6nd) / target_flops_6nd,
                "train_tokens_per_epoch": float(row["train_tokens_per_epoch"]),
                "final_checkpoint": str(final_dir),
                "out_dir": str(out_dir),
                "source_matrix": str(matrix_path),
            }
    return sorted(
        rows_by_run.values(),
        key=lambda row: (float(row["target_flops_6nd"]), int(row["parameter_count"])),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no completed fixed-FLOPs rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_series(
    ax: Any,
    rows: list[dict[str, Any]],
    color: Any,
    plotter: Any,
    *,
    series_label: str | None,
    annotate_all: bool,
) -> None:
    import numpy as np

    rows = sorted(rows, key=lambda row: int(row["parameter_count"]))
    xs = [float(row["parameter_count"]) for row in rows]
    ys = [float(row["training_loss"]) for row in rows]
    curve_x, curve_y = plotter.smooth_curve(xs, ys)
    curve_x = np.asarray(curve_x, dtype=float)
    curve_y = np.asarray(curve_y, dtype=float)
    boundary = float(rows[0]["target_flops_6nd"]) / (6.0 * float(rows[0]["train_tokens_per_epoch"]))

    if curve_x.min() < boundary < curve_x.max():
        boundary_y = float(np.interp(np.log(boundary), np.log(curve_x), curve_y))
        dashed = curve_x < boundary
        solid = curve_x > boundary
        ax.plot(
            [*curve_x[dashed], boundary],
            [*curve_y[dashed], boundary_y],
            color=color,
            linewidth=2.0,
            linestyle="--",
        )
        ax.plot(
            [boundary, *curve_x[solid]],
            [boundary_y, *curve_y[solid]],
            color=color,
            linewidth=2.0,
            label=series_label,
        )
    elif curve_x.max() <= boundary:
        ax.plot(curve_x, curve_y, color=color, linewidth=2.0, linestyle="--", label=series_label)
    else:
        ax.plot(curve_x, curve_y, color=color, linewidth=2.0, label=series_label)

    regular = [row for row in rows if not row["over_one_epoch"]]
    over_epoch = [row for row in rows if row["over_one_epoch"]]
    if regular:
        ax.scatter(
            [row["parameter_count"] for row in regular],
            [row["training_loss"] for row in regular],
            color=color,
            edgecolors="white",
            linewidths=0.7,
            s=42,
            zorder=3,
        )
    if over_epoch:
        ax.scatter(
            [row["parameter_count"] for row in over_epoch],
            [row["training_loss"] for row in over_epoch],
            facecolors="none",
            edgecolors=color,
            marker="D",
            linewidths=1.7,
            s=50,
            zorder=3,
        )

    best = min(rows, key=lambda row: float(row["training_loss"]))
    ax.scatter(
        [best["parameter_count"]],
        [best["training_loss"]],
        facecolors="none",
        edgecolors=color,
        linewidths=2.2,
        s=125,
        zorder=4,
    )
    annotate_rows = rows if annotate_all else [best]
    offsets = [(4, 6), (4, -14), (4, 14), (4, -22)]
    for index, row in enumerate(annotate_rows):
        epoch = f", {row['epoch_fraction']:.2f} ep" if row["over_one_epoch"] else ""
        loss = f"\n{row['training_loss']:.3f}" if not annotate_all else ""
        ax.annotate(
            f"{row['model_label']}{epoch}{loss}",
            (row["parameter_count"], row["training_loss"]),
            textcoords="offset points",
            xytext=offsets[index % len(offsets)] if annotate_all else (5, 6),
            fontsize=7,
            color=color,
        )


def save_overlay(
    path: Path,
    grouped: dict[float, list[dict[str, Any]]],
    budgets: list[float],
    colors: Any,
    plotter: Any,
    title: str,
    *,
    figsize: tuple[float, float] = (9.2, 6.2),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    compact_parameter_ticks: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

    run_count = sum(len(grouped[budget]) for budget in budgets)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for budget, color in zip(budgets, colors):
        draw_series(
            ax,
            grouped[budget],
            color,
            plotter,
            series_label=f"{budget:.1e}",
            annotate_all=False,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("Final training loss")
    ax.set_title(f"{title} ({run_count} runs)")
    ax.margins(x=0.04, y=0.08)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if compact_parameter_ticks:
        ticks = [5e6, 1e7, 2e7, 4e7, 6e7, 8e7, 1.2e8]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}M"))
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.08)
    else:
        ax.grid(True, which="both", alpha=0.24)
    legend_location = "upper center" if compact_parameter_ticks else "upper right"
    legend_columns = 2 if len(budgets) >= 8 else 1
    budget_legend = ax.legend(
        title="Fixed FLOPs (6ND)",
        frameon=False,
        loc=legend_location,
        ncol=legend_columns,
    )
    ax.add_artist(budget_legend)
    style_handles = [
        Line2D([0], [0], color="#555555", linewidth=2, label="<=1 epoch"),
        Line2D([0], [0], color="#555555", linewidth=2, linestyle="--", label=">1 epoch"),
        Line2D(
            [0],
            [0],
            marker="D",
            markerfacecolor="none",
            markeredgecolor="#555555",
            linestyle="none",
            label=">1 epoch run",
        ),
    ]
    ax.legend(handles=style_handles, frameon=False, loc="lower left", fontsize=8)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", action="append", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = collect_completed(args.matrix_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "all_fixed_flops_results.csv"
    write_csv(csv_path, rows)

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["target_flops_6nd"])].append(row)
    budgets = sorted(grouped)

    import matplotlib.pyplot as plt
    plotter = load_single_matrix_plotter()
    colors = plt.get_cmap("plasma")([0.12 + 0.76 * i / max(len(budgets) - 1, 1) for i in range(len(budgets))])

    overlay_full_path = args.output_dir / "all_fixed_flops_overlay_full.png"
    save_overlay(
        overlay_full_path,
        grouped,
        budgets,
        colors,
        plotter,
        "All fixed-compute scaling runs",
    )

    focus_grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["parameter_count"]) <= 1_000_000_000:
            focus_grouped[float(row["target_flops_6nd"])].append(row)
    focus_losses = [float(row["training_loss"]) for group in focus_grouped.values() for row in group]
    focus_ylim = (
        math.floor((min(focus_losses) - 0.01) * 100.0) / 100.0,
        math.ceil((max(focus_losses) + 0.01) * 100.0) / 100.0,
    )
    overlay_path = args.output_dir / "all_fixed_flops_overlay.png"
    save_overlay(
        overlay_path,
        focus_grouped,
        budgets,
        colors,
        plotter,
        "Fixed-compute scaling (zoomed)",
        figsize=(6.8, 8.2),
        xlim=(5e6, 1.2e8),
        ylim=focus_ylim,
        compact_parameter_ticks=True,
    )

    cols = 2
    facet_rows = math.ceil(len(budgets) / cols)
    fig, axes = plt.subplots(facet_rows, cols, figsize=(12.5, 4.7 * facet_rows), constrained_layout=True)
    axes_list = list(getattr(axes, "flat", [axes]))
    for ax, budget, color in zip(axes_list, budgets, colors):
        draw_series(ax, grouped[budget], color, plotter, series_label=None, annotate_all=True)
        ax.set_xscale("log")
        ax.set_xlabel("Model parameters")
        ax.set_ylabel("Final training loss")
        ax.set_title(f"Fixed FLOPs: {budget:.1e} (6ND)")
        ax.grid(True, which="both", alpha=0.24)
        ax.margins(x=0.05, y=0.14)
    for ax in axes_list[len(budgets):]:
        ax.axis("off")
    facet_path = args.output_dir / "all_fixed_flops_facets.png"
    fig.savefig(facet_path, dpi=220)
    fig.savefig(facet_path.with_suffix(".pdf"))
    plt.close(fig)

    result = {
        "completed_runs": len(rows),
        "budgets": [{"flops": budget, "runs": len(grouped[budget])} for budget in budgets],
        "csv": str(csv_path),
        "overlay": str(overlay_path),
        "overlay_full": str(overlay_full_path),
        "facets": str(facet_path),
    }
    (args.output_dir / "all_fixed_flops_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
