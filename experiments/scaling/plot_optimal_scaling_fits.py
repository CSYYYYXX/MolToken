#!/usr/bin/env python3
"""Fit compute-optimal model and token scaling against fixed FLOPs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_result_collector() -> Any:
    path = Path(__file__).with_name("plot_all_fixed_flops_results.py")
    spec = importlib.util.spec_from_file_location("all_fixed_flops_plotter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no optimal points")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, Any]:
    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2:
        raise ValueError("at least two optimal points are required for a linear fit")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum_squares = float(np.sum((y - predicted) ** 2))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0.0 else 1.0
    degrees_of_freedom = len(x) - 2
    slope_standard_error = None
    intercept_standard_error = None
    slope_95_ci = None
    intercept_95_ci = None
    if degrees_of_freedom > 0:
        centered_sum_squares = float(np.sum((x - np.mean(x)) ** 2))
        residual_variance = residual_sum_squares / degrees_of_freedom
        if centered_sum_squares > 0.0:
            slope_standard_error = math.sqrt(residual_variance / centered_sum_squares)
            intercept_standard_error = math.sqrt(
                residual_variance * (1.0 / len(x) + float(np.mean(x)) ** 2 / centered_sum_squares)
            )
            try:
                from scipy.stats import t

                critical_value = float(t.ppf(0.975, degrees_of_freedom))
            except Exception:
                critical_value = 1.96
            slope_95_ci = [
                float(slope - critical_value * slope_standard_error),
                float(slope + critical_value * slope_standard_error),
            ]
            intercept_95_ci = [
                float(intercept - critical_value * intercept_standard_error),
                float(intercept + critical_value * intercept_standard_error),
            ]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "residual_sum_squares": residual_sum_squares,
        "root_mean_squared_error": math.sqrt(residual_sum_squares / len(x)),
        "degrees_of_freedom": degrees_of_freedom,
        "slope_standard_error": slope_standard_error,
        "slope_95_ci": slope_95_ci,
        "intercept_standard_error": intercept_standard_error,
        "intercept_95_ci": intercept_95_ci,
        "point_count": int(len(x)),
        "predicted": predicted.tolist(),
    }


def draw_fit(
    path: Path,
    rows: list[dict[str, Any]],
    fit: dict[str, Any],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    x_term: str,
    y_term: str,
    title: str,
    y_tick_step: float,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    xs = np.asarray([float(row[x_key]) for row in rows], dtype=float)
    ys = np.asarray([float(row[y_key]) for row in rows], dtype=float)
    log_loss = np.asarray([float(row["log10_training_loss"]) for row in rows], dtype=float)
    observed_x_min = float(xs.min())
    observed_x_max = float(xs.max())
    x_min = math.floor(observed_x_min)
    x_max = math.ceil(observed_x_max)
    y_min = math.floor(float(ys.min()))
    y_max = math.ceil(float(ys.max()))
    minimum_y_intervals = 5
    missing_intervals = max(0, minimum_y_intervals - (y_max - y_min))
    y_min -= missing_intervals // 2
    y_max += missing_intervals - missing_intervals // 2

    grid = np.linspace(observed_x_min, observed_x_max, 200)
    line = float(fit["slope"]) * grid + float(fit["intercept"])
    left_grid = np.linspace(x_min, observed_x_min, 80)
    right_grid = np.linspace(observed_x_max, x_max, 80)
    left_line = float(fit["slope"]) * left_grid + float(fit["intercept"])
    right_line = float(fit["slope"]) * right_grid + float(fit["intercept"])

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    points = ax.scatter(
        xs,
        ys,
        c=log_loss,
        cmap="plasma",
        s=68,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.plot(grid, line, color="#303030", linewidth=2.0, label="OLS linear fit")
    ax.plot(left_grid, left_line, color="#303030", linewidth=1.8, linestyle="--", label="Linear extrapolation")
    ax.plot(right_grid, right_line, color="#303030", linewidth=1.8, linestyle="--")
    sorted_indices = np.argsort(xs)
    ranks = {int(index): rank for rank, index in enumerate(sorted_indices)}
    label_offsets = [14, -18, 22, -26]
    for index, (x, y, row) in enumerate(zip(xs, ys, rows)):
        rank = ranks[index]
        offset_y = label_offsets[rank % len(label_offsets)]
        if rank == len(rows) - 1:
            offset_y = -20
        ax.annotate(
            f"{float(row['target_flops_6nd']):.1e}",
            (x, y),
            textcoords="offset points",
            xytext=(0, offset_y),
            ha="center",
            fontsize=7,
            color="#303030",
        )

    equation = (
        f"{y_term} = {float(fit['slope']):.6f} {x_term} {float(fit['intercept']):+.6f}\n"
        f"slope = {float(fit['slope']):.6f}; intercept = {float(fit['intercept']):.6f}\n"
        f"R^2 = {float(fit['r_squared']):.4f}"
    )
    ax.text(
        0.965,
        0.965,
        equation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.92, "boxstyle": "square,pad=0.4"},
    )
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("log10(Final training loss)")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"1e{int(round(value))}" if math.isclose(value, round(value), abs_tol=1e-9) else "")
    )

    def format_original_quantity(value: float, _: int) -> str:
        if not math.isclose(value, round(value), abs_tol=1e-9):
            return ""
        original = 10 ** int(round(value))
        if original >= 1_000_000_000_000:
            return f"{original / 1_000_000_000_000:g}T"
        if original >= 1_000_000_000:
            return f"{original / 1_000_000_000:g}B"
        if original >= 1_000_000:
            return f"{original / 1_000_000:g}M"
        if original >= 1_000:
            return f"{original / 1_000:g}K"
        return f"{original:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(format_original_quantity))
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", action="append", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--require-budget-count", type=int, default=0)
    args = ap.parse_args()

    collector = load_result_collector()
    completed = collector.collect_completed(args.matrix_json)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[float(row["target_flops_6nd"])].append(row)
    if args.require_budget_count and len(grouped) != args.require_budget_count:
        raise ValueError(f"found {len(grouped)} completed budgets, expected {args.require_budget_count}")

    optimal_rows: list[dict[str, Any]] = []
    for budget in sorted(grouped):
        curve_rows = sorted(grouped[budget], key=lambda row: int(row["parameter_count"]))
        best = min(
            curve_rows,
            key=lambda row: (float(row["training_loss"]), int(row["parameter_count"])),
        )
        best_index = curve_rows.index(best)
        left_neighbor = curve_rows[best_index - 1] if best_index > 0 else None
        right_neighbor = curve_rows[best_index + 1] if best_index + 1 < len(curve_rows) else None
        parameter_count = int(best["parameter_count"])
        planned_actual_tokens = int(best["actual_tokens"])
        training_tokens_d = int(best["observed_train_loss_tokens_global"])
        training_loss = float(best["training_loss"])
        left_neighbor_higher = (
            left_neighbor is not None and float(left_neighbor["training_loss"]) > training_loss
        )
        right_neighbor_higher = (
            right_neighbor is not None and float(right_neighbor["training_loss"]) > training_loss
        )
        minimum_bracketed = left_neighbor_higher and right_neighbor_higher
        if parameter_count <= 0 or training_tokens_d <= 0 or training_loss <= 0.0:
            raise ValueError(f"non-positive value in optimal row for budget {budget:.3e}")
        optimal_rows.append(
            {
                "target_label": best["target_label"],
                "target_flops_6nd": budget,
                "model_id": best["model_id"],
                "model_label": best["model_label"],
                "parameter_count": parameter_count,
                "planned_actual_tokens": planned_actual_tokens,
                "training_tokens_D": training_tokens_d,
                "training_token_budget_b": training_tokens_d / 1e9,
                "world_size": int(best["world_size"]),
                "observed_token_relative_error": float(best["observed_token_relative_error"]),
                "observed_flops_6nd": float(best["observed_flops_6nd"]),
                "observed_flops_relative_error": float(best["observed_flops_relative_error"]),
                "epoch_fraction": float(best["epoch_fraction"]),
                "over_one_epoch": bool(best["over_one_epoch"]),
                "final_training_loss": training_loss,
                "trailing_train_loss": best["trailing_train_loss"],
                "curve_point_count": len(curve_rows),
                "minimum_bracketed": minimum_bracketed,
                "left_neighbor_higher": left_neighbor_higher,
                "right_neighbor_higher": right_neighbor_higher,
                "left_neighbor_parameter_count": int(left_neighbor["parameter_count"]) if left_neighbor else None,
                "left_neighbor_training_loss": float(left_neighbor["training_loss"]) if left_neighbor else None,
                "right_neighbor_parameter_count": int(right_neighbor["parameter_count"]) if right_neighbor else None,
                "right_neighbor_training_loss": float(right_neighbor["training_loss"]) if right_neighbor else None,
                "log10_parameter_count": math.log10(parameter_count),
                "log10_training_tokens_D": math.log10(training_tokens_d),
                "log10_training_loss": math.log10(training_loss),
                "log10_target_flops_6nd": math.log10(budget),
                "out_dir": best["out_dir"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fixed_flops_optimal_points.csv"
    write_csv(csv_path, optimal_rows)

    log_flops = [float(row["log10_target_flops_6nd"]) for row in optimal_rows]
    fit_n = linear_fit(log_flops, [float(row["log10_parameter_count"]) for row in optimal_rows])
    fit_d = linear_fit(log_flops, [float(row["log10_training_tokens_D"]) for row in optimal_rows])

    log_n_path = args.output_dir / "optimal_logN_vs_logFLOPs_linear_fit.png"
    draw_fit(
        log_n_path,
        optimal_rows,
        fit_n,
        x_key="log10_target_flops_6nd",
        y_key="log10_parameter_count",
        x_label="Fixed FLOPs (6ND, log10 scale)",
        y_label="Model parameters N (log10 scale)",
        x_term="log10(C)",
        y_term="log10(N)",
        title="Compute-optimal model size vs training FLOPs",
        y_tick_step=1.0,
    )
    log_d_path = args.output_dir / "optimal_logD_vs_logFLOPs_linear_fit.png"
    draw_fit(
        log_d_path,
        optimal_rows,
        fit_d,
        x_key="log10_target_flops_6nd",
        y_key="log10_training_tokens_D",
        x_label="Fixed FLOPs (6ND, log10 scale)",
        y_label="Training tokens D (log10 scale)",
        x_term="log10(C)",
        y_term="log10(D)",
        title="Compute-optimal training tokens vs training FLOPs",
        y_tick_step=1.0,
    )

    for fit in (fit_n, fit_d):
        fit.pop("predicted", None)
    result = {
        "point_count": len(optimal_rows),
        "bracketed_point_count": sum(bool(row["minimum_bracketed"]) for row in optimal_rows),
        "unbracketed_budgets": [
            float(row["target_flops_6nd"]) for row in optimal_rows if not row["minimum_bracketed"]
        ],
        "loss_definition": "summary.final_logged_train_loss",
        "D_definition": "summary.total_train_loss_tokens multiplied by run_config.world_size",
        "log_base": 10,
        "selection": "lowest observed final training loss within each fixed-FLOPs budget",
        "fit_log10_N_vs_log10_flops": fit_n,
        "fit_log10_D_vs_log10_flops": fit_d,
        "optimal_points_csv": str(csv_path),
        "logN_plot": str(log_n_path),
        "logD_plot": str(log_d_path),
    }
    summary_path = args.output_dir / "optimal_scaling_fit_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
