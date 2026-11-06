#!/usr/bin/env python3
"""Fit the Chinchilla loss law to completed fixed-compute training runs.

The fitted model is

    L(N, D) = E + A / N**alpha + B / D**beta

All five fitted parameters are represented in log space, so they remain
strictly positive. The objective is the mean Huber loss of natural-log loss
residuals. This script only reads small run metadata and summary files; it
does not launch training or load checkpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

# Each optimization handles only 44 rows. Threaded BLAS is slower here and
# oversubscribes CPUs when bootstrap workers run in parallel.
for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_variable, "1")

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp


PARAMETER_NAMES = ("E", "A", "B", "alpha", "beta", "a", "b", "G")


def collect_completed_runs(matrix_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Collect one validated row per completed output directory."""

    rows_by_output: dict[str, dict[str, Any]] = {}
    for matrix_path in matrix_paths:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if not isinstance(matrix.get("rows"), list):
            raise ValueError(f"matrix has no rows list: {matrix_path}")
        for matrix_row in matrix["rows"]:
            output_dir = Path(matrix_row["out_dir"])
            summary_path = output_dir / "summary.json"
            run_config_path = output_dir / "run_config.json"
            final_dir = output_dir / "final"
            if not summary_path.is_file() or not run_config_path.is_file() or not final_dir.is_dir():
                continue

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            required_steps = int(matrix_row["max_steps"])
            completed_steps = int(summary.get("global_step", -1))
            if completed_steps < required_steps:
                continue

            parameter_count = int(matrix_row["parameter_count"])
            local_loss_tokens = int(summary.get("total_train_loss_tokens", 0))
            world_size = int(run_config.get("world_size", 0))
            training_tokens = local_loss_tokens * world_size
            final_loss = float(summary.get("final_logged_train_loss", math.nan))
            if parameter_count <= 0:
                raise ValueError(f"non-positive parameter count in {output_dir}")
            if local_loss_tokens <= 0 or world_size <= 0 or training_tokens <= 0:
                raise ValueError(f"non-positive observed training tokens in {output_dir}")
            if not math.isfinite(final_loss) or final_loss <= 0.0:
                raise ValueError(f"invalid final training loss in {output_dir}: {final_loss}")

            target_flops = float(matrix_row["target_flops_6nd"])
            observed_flops = 6.0 * parameter_count * training_tokens
            key = str(output_dir.resolve())
            row = {
                "model_id": matrix_row["model_id"],
                "model_label": matrix_row["model_label"],
                "parameter_count_N": parameter_count,
                "training_tokens_D": training_tokens,
                "final_training_loss": final_loss,
                "target_flops_6nd": target_flops,
                "observed_flops_6nd": observed_flops,
                "observed_flops_relative_error": (observed_flops - target_flops) / target_flops,
                "global_step": completed_steps,
                "max_steps": required_steps,
                "world_size": world_size,
                "local_loss_tokens": local_loss_tokens,
                "output_dir": key,
                "source_matrix": str(matrix_path.resolve()),
            }
            previous = rows_by_output.get(key)
            if previous is not None:
                comparable = ("parameter_count_N", "training_tokens_D", "final_training_loss")
                if any(previous[field] != row[field] for field in comparable):
                    raise ValueError(f"conflicting duplicate run metadata for {key}")
            rows_by_output[key] = row

    return sorted(
        rows_by_output.values(),
        key=lambda row: (
            float(row["target_flops_6nd"]),
            int(row["parameter_count_N"]),
            str(row["output_dir"]),
        ),
    )


def decode_parameters(theta: np.ndarray) -> dict[str, float]:
    log_e, log_a_coefficient, log_b_coefficient, log_alpha, log_beta = theta
    alpha = float(np.exp(log_alpha))
    beta = float(np.exp(log_beta))
    exponent_sum = alpha + beta
    scaling_a = beta / exponent_sum
    scaling_b = alpha / exponent_sum
    log_g = (log_alpha + log_a_coefficient - log_beta - log_b_coefficient) / exponent_sum
    return {
        "E": float(np.exp(log_e)),
        "A": float(np.exp(log_a_coefficient)),
        "B": float(np.exp(log_b_coefficient)),
        "alpha": alpha,
        "beta": beta,
        "a": scaling_a,
        "b": scaling_b,
        "G": float(np.exp(log_g)),
    }


def predict_log_loss(theta: np.ndarray, log_n: np.ndarray, log_d: np.ndarray) -> np.ndarray:
    alpha = np.exp(theta[3])
    beta = np.exp(theta[4])
    components = np.vstack(
        (
            np.full_like(log_n, theta[0]),
            theta[1] - alpha * log_n,
            theta[2] - beta * log_d,
        )
    )
    return logsumexp(components, axis=0)


def huber_objective_and_gradient(
    theta: np.ndarray,
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    """Return mean Huber(log prediction - log observation) and its gradient."""

    alpha = np.exp(theta[3])
    beta = np.exp(theta[4])
    components = np.vstack(
        (
            np.full_like(log_n, theta[0]),
            theta[1] - alpha * log_n,
            theta[2] - beta * log_d,
        )
    )
    log_prediction = logsumexp(components, axis=0)
    component_weights = np.exp(components - log_prediction)
    residual = log_prediction - log_loss
    absolute_residual = np.abs(residual)
    quadratic = absolute_residual <= delta
    losses = np.where(
        quadratic,
        0.5 * residual**2,
        delta * (absolute_residual - 0.5 * delta),
    )
    influence = np.where(quadratic, residual, delta * np.sign(residual))

    prediction_jacobian = np.column_stack(
        (
            component_weights[0],
            component_weights[1],
            component_weights[2],
            -component_weights[1] * alpha * log_n,
            -component_weights[2] * beta * log_d,
        )
    )
    gradient = np.mean(influence[:, None] * prediction_jacobian, axis=0)
    return float(np.mean(losses)), gradient


def parameter_bounds(log_loss: np.ndarray) -> list[tuple[float, float]]:
    # E is an irreducible loss floor, so it must lie below every observed loss.
    return [
        (math.log(1e-8), float(np.min(log_loss) - 1e-10)),
        (-40.0, 200.0),
        (-40.0, 200.0),
        (math.log(1e-4), math.log(5.0)),
        (math.log(1e-4), math.log(5.0)),
    ]


def random_initial_theta(
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    minimum_loss = float(np.exp(np.min(log_loss)))
    median_loss = float(np.exp(np.median(log_loss)))
    e_value = minimum_loss * rng.uniform(0.15, 0.98)
    alpha = float(np.exp(rng.uniform(math.log(0.03), math.log(1.25))))
    beta = float(np.exp(rng.uniform(math.log(0.03), math.log(1.25))))
    reducible_loss = max(median_loss - e_value, 0.02 * median_loss)
    n_fraction = rng.uniform(0.15, 0.85)
    n_contribution = reducible_loss * n_fraction
    d_contribution = reducible_loss * (1.0 - n_fraction)
    return np.asarray(
        [
            math.log(e_value),
            math.log(n_contribution) + alpha * float(np.median(log_n)),
            math.log(d_contribution) + beta * float(np.median(log_d)),
            math.log(alpha),
            math.log(beta),
        ],
        dtype=float,
    )


def fit_multistart(
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    *,
    rng: np.random.Generator,
    starts: int,
    delta: float,
    max_iterations: int,
    warm_theta: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if starts < 2:
        raise ValueError("at least two L-BFGS starts are required")
    bounds = parameter_bounds(log_loss)
    initial_thetas: list[np.ndarray] = []
    if warm_theta is not None:
        clipped_warm = np.asarray(
            [np.clip(value, lower, upper) for value, (lower, upper) in zip(warm_theta, bounds)],
            dtype=float,
        )
        initial_thetas.append(clipped_warm)
        perturbation_count = min(3, starts - 1)
        for _ in range(perturbation_count):
            perturbed = clipped_warm + rng.normal(0.0, [0.08, 0.35, 0.35, 0.18, 0.18])
            initial_thetas.append(
                np.asarray(
                    [np.clip(value, lower, upper) for value, (lower, upper) in zip(perturbed, bounds)],
                    dtype=float,
                )
            )
    while len(initial_thetas) < starts:
        initial_thetas.append(random_initial_theta(log_n, log_d, log_loss, rng))

    results: list[OptimizeResult] = []
    for initial_theta in initial_thetas:
        result = minimize(
            huber_objective_and_gradient,
            initial_theta,
            args=(log_n, log_d, log_loss, delta),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": max_iterations,
                "maxfun": 3 * max_iterations,
                "ftol": 1e-11,
                "gtol": 1e-8,
                "maxls": 30,
            },
        )
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            results.append(result)
    if not results:
        raise RuntimeError("all L-BFGS starts failed")
    best = min(results, key=lambda result: float(result.fun))
    diagnostics = {
        "requested_starts": starts,
        "finite_starts": len(results),
        "converged_starts": sum(bool(result.success) for result in results),
        "best_converged": bool(best.success),
        "best_status": int(best.status),
        "best_message": str(best.message),
        "best_iterations": int(best.nit),
        "best_function_evaluations": int(best.nfev),
        "best_objective": float(best.fun),
        "iterations_median": float(np.median([result.nit for result in results])),
        "iterations_max": int(max(result.nit for result in results)),
        "function_evaluations_median": float(np.median([result.nfev for result in results])),
        "function_evaluations_max": int(max(result.nfev for result in results)),
        "objective_min": float(min(result.fun for result in results)),
        "objective_median": float(np.median([result.fun for result in results])),
        "objective_max": float(max(result.fun for result in results)),
    }
    return np.asarray(best.x, dtype=float), diagnostics


def fit_metrics(
    theta: np.ndarray,
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    delta: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    log_prediction = predict_log_loss(theta, log_n, log_d)
    prediction = np.exp(log_prediction)
    observed = np.exp(log_loss)
    log_residual = log_prediction - log_loss
    raw_residual = prediction - observed
    huber, _ = huber_objective_and_gradient(theta, log_n, log_d, log_loss, delta)
    log_total_sum_squares = float(np.sum((log_loss - np.mean(log_loss)) ** 2))
    raw_total_sum_squares = float(np.sum((observed - np.mean(observed)) ** 2))
    metrics = {
        "mean_huber_log_residual": huber,
        "rmse_log_loss": float(np.sqrt(np.mean(log_residual**2))),
        "mae_log_loss": float(np.mean(np.abs(log_residual))),
        "median_absolute_log_loss": float(np.median(np.abs(log_residual))),
        "max_absolute_log_loss": float(np.max(np.abs(log_residual))),
        "rmse_training_loss": float(np.sqrt(np.mean(raw_residual**2))),
        "mae_training_loss": float(np.mean(np.abs(raw_residual))),
        "r_squared_log_loss": (
            1.0 - float(np.sum(log_residual**2)) / log_total_sum_squares
            if log_total_sum_squares > 0.0
            else 1.0
        ),
        "r_squared_training_loss": (
            1.0 - float(np.sum(raw_residual**2)) / raw_total_sum_squares
            if raw_total_sum_squares > 0.0
            else 1.0
        ),
    }
    return metrics, prediction, log_residual


def run_bootstrap(
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    fitted_theta: np.ndarray,
    *,
    samples: int,
    starts: int,
    delta: float,
    max_iterations: int,
    seed: int,
    jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    report_every = max(1, samples // 20)
    child_seeds = np.random.SeedSequence(seed).spawn(samples)
    tasks = [
        (
            bootstrap_index,
            int(child_seed.generate_state(1)[0]),
            log_n,
            log_d,
            log_loss,
            fitted_theta,
            starts,
            delta,
            max_iterations,
        )
        for bootstrap_index, child_seed in enumerate(child_seeds)
    ]

    if jobs == 1:
        results = map(fit_bootstrap_sample, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)
        results = executor.map(fit_bootstrap_sample, tasks, chunksize=1)
    try:
        for completed_count, (record, failure) in enumerate(results, start=1):
            if record is not None:
                records.append(record)
            if failure is not None:
                failures.append(failure)
            if completed_count % report_every == 0 or completed_count == samples:
                print(
                    f"bootstrap {completed_count}/{samples}: "
                    f"success={len(records)} failure={len(failures)}",
                    flush=True,
                )
    finally:
        if jobs != 1:
            executor.shutdown(wait=True)
    records.sort(key=lambda record: int(record["bootstrap_index"]))
    failures.sort(key=lambda failure: int(failure["bootstrap_index"]))
    return records, failures


def fit_bootstrap_sample(
    task: tuple[
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        float,
        int,
    ],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    (
        bootstrap_index,
        seed,
        log_n,
        log_d,
        log_loss,
        fitted_theta,
        starts,
        delta,
        max_iterations,
    ) = task
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(log_n), size=len(log_n))
    try:
        theta, diagnostics = fit_multistart(
            log_n[sampled_indices],
            log_d[sampled_indices],
            log_loss[sampled_indices],
            rng=rng,
            starts=starts,
            delta=delta,
            max_iterations=max_iterations,
            warm_theta=fitted_theta,
        )
        decoded = decode_parameters(theta)
        if not all(math.isfinite(decoded[name]) and decoded[name] > 0.0 for name in PARAMETER_NAMES):
            raise ValueError("non-finite or non-positive decoded bootstrap parameter")
        return (
            {
                "bootstrap_index": bootstrap_index,
                **decoded,
                "objective": diagnostics["best_objective"],
                "best_converged": diagnostics["best_converged"],
            },
            None,
        )
    except Exception as error:
        return None, {"bootstrap_index": bootstrap_index, "error": str(error)}


def percentile_intervals(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("no successful bootstrap fits")
    intervals: dict[str, dict[str, float]] = {}
    for name in PARAMETER_NAMES:
        values = np.asarray([float(record[name]) for record in records], dtype=float)
        lower, median, upper = np.percentile(values, [2.5, 50.0, 97.5])
        intervals[name] = {
            "lower_2.5_percent": float(lower),
            "median": float(median),
            "upper_97.5_percent": float(upper),
        }
    return intervals


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_error_plot(
    output_path: Path,
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
    log_residual: np.ndarray,
    delta: float,
    metrics: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    observed = np.asarray([float(row["final_training_loss"]) for row in rows], dtype=float)
    parameter_count = np.asarray([float(row["parameter_count_N"]) for row in rows], dtype=float)
    training_tokens = np.asarray([float(row["training_tokens_D"]) for row in rows], dtype=float)
    color_values = np.log10([float(row["target_flops_6nd"]) for row in rows])
    residual_milli = 1000.0 * log_residual

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    common_scatter = {
        "c": color_values,
        "cmap": "viridis",
        "s": 48,
        "edgecolors": "white",
        "linewidths": 0.7,
        "zorder": 3,
    }
    scatter = axes[0].scatter(observed, prediction, **common_scatter)
    lower = min(float(observed.min()), float(prediction.min()))
    upper = max(float(observed.max()), float(prediction.max()))
    padding = 0.05 * (upper - lower)
    axes[0].plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#303030")
    axes[0].set_xlim(lower - padding, upper + padding)
    axes[0].set_ylim(lower - padding, upper + padding)
    axes[0].set_xlabel("Observed final training loss")
    axes[0].set_ylabel("Predicted final training loss")
    axes[0].set_title("Predicted vs observed")
    axes[0].text(
        0.04,
        0.96,
        f"RMSE(log L) = {metrics['rmse_log_loss']:.5f}\nR^2(log L) = {metrics['r_squared_log_loss']:.4f}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.9, "boxstyle": "square,pad=0.35"},
    )

    axes[1].scatter(parameter_count, residual_milli, **common_scatter)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Model parameters N")
    axes[1].set_ylabel("1000 x [ln(L_pred) - ln(L_true)]")
    axes[1].set_title("Residual vs model size")

    axes[2].scatter(training_tokens, residual_milli, **common_scatter)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Training tokens D")
    axes[2].set_ylabel("1000 x [ln(L_pred) - ln(L_true)]")
    axes[2].set_title("Residual vs training tokens")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.22)
    for axis in axes[1:]:
        axis.axhline(0.0, color="#303030", linewidth=1.2)
        axis.axhline(1000.0 * delta, color="#777777", linewidth=1.0, linestyle="--")
        axis.axhline(-1000.0 * delta, color="#777777", linewidth=1.0, linestyle="--")

    colorbar = fig.colorbar(scatter, ax=axes, pad=0.015, shrink=0.93)
    colorbar.set_label("log10(Fixed FLOPs, 6ND)")
    fig.suptitle(f"Chinchilla loss fit errors ({len(rows)} existing fixed-compute runs)", fontsize=14)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-runs", type=int, default=44)
    parser.add_argument("--random-starts", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-starts", type=int, default=4)
    parser.add_argument("--bootstrap-jobs", type=int, default=1)
    parser.add_argument("--huber-delta", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--isoflop-a", type=float, default=0.582937)
    parser.add_argument("--isoflop-b", type=float, default=0.417100)
    args = parser.parse_args()
    if args.random_starts < 2 or args.bootstrap_starts < 2:
        parser.error("both start counts must be at least two")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.bootstrap_jobs <= 0:
        parser.error("--bootstrap-jobs must be positive")
    if args.huber_delta <= 0.0:
        parser.error("--huber-delta must be positive")
    return args


def main() -> None:
    args = parse_args()
    started_at = time.time()
    rows = collect_completed_runs(args.matrix_json)
    if len(rows) != args.require_runs:
        raise ValueError(f"found {len(rows)} unique completed runs, expected {args.require_runs}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_csv = args.output_dir / "chinchilla_input_44runs.csv"
    write_csv(input_csv, rows)
    log_n = np.log([float(row["parameter_count_N"]) for row in rows])
    log_d = np.log([float(row["training_tokens_D"]) for row in rows])
    log_loss = np.log([float(row["final_training_loss"]) for row in rows])

    main_rng = np.random.default_rng(args.seed)
    fitted_theta, optimization = fit_multistart(
        log_n,
        log_d,
        log_loss,
        rng=main_rng,
        starts=args.random_starts,
        delta=args.huber_delta,
        max_iterations=args.max_iterations,
    )
    parameters = decode_parameters(fitted_theta)
    metrics, prediction, log_residual = fit_metrics(
        fitted_theta,
        log_n,
        log_d,
        log_loss,
        args.huber_delta,
    )

    bootstrap_records, bootstrap_failures = run_bootstrap(
        log_n,
        log_d,
        log_loss,
        fitted_theta,
        samples=args.bootstrap_samples,
        starts=args.bootstrap_starts,
        delta=args.huber_delta,
        max_iterations=args.max_iterations,
        seed=args.seed + 1,
        jobs=args.bootstrap_jobs,
    )
    minimum_bootstrap_success = math.ceil(0.9 * args.bootstrap_samples)
    if len(bootstrap_records) < minimum_bootstrap_success:
        raise RuntimeError(
            f"only {len(bootstrap_records)}/{args.bootstrap_samples} bootstrap fits succeeded; "
            f"at least {minimum_bootstrap_success} are required"
        )
    confidence_intervals = percentile_intervals(bootstrap_records)

    bootstrap_csv = args.output_dir / "chinchilla_bootstrap_samples.csv"
    write_csv(bootstrap_csv, bootstrap_records)
    interval_rows = [
        {"parameter": name, "estimate": parameters[name], **confidence_intervals[name]}
        for name in PARAMETER_NAMES
    ]
    interval_csv = args.output_dir / "chinchilla_parameter_intervals.csv"
    write_csv(interval_csv, interval_rows)

    error_plot = args.output_dir / "chinchilla_fit_error.png"
    draw_error_plot(error_plot, rows, prediction, log_residual, args.huber_delta, metrics)

    fitted_rows = []
    for row, predicted_loss, residual in zip(rows, prediction, log_residual):
        fitted_rows.append(
            {
                **row,
                "predicted_training_loss": float(predicted_loss),
                "log_residual_pred_minus_true": float(residual),
                "absolute_log_residual": float(abs(residual)),
            }
        )
    fitted_csv = args.output_dir / "chinchilla_fitted_points.csv"
    write_csv(fitted_csv, fitted_rows)

    result = {
        "model": "L(N,D) = E + A / N^alpha + B / D^beta",
        "positive_parameterization": {
            "E": "exp(theta_E)",
            "A": "exp(theta_A)",
            "B": "exp(theta_B)",
            "alpha": "exp(theta_alpha)",
            "beta": "exp(theta_beta)",
        },
        "objective": f"mean Huber(ln(L_pred) - ln(L_true), delta={args.huber_delta:g})",
        "run_count": len(rows),
        "N_definition": "exact matrix parameter_count",
        "D_definition": "summary.total_train_loss_tokens multiplied by run_config.world_size",
        "loss_definition": "summary.final_logged_train_loss",
        "source_matrices": [str(path.resolve()) for path in args.matrix_json],
        "configuration": {
            "random_starts": args.random_starts,
            "bootstrap_samples_requested": args.bootstrap_samples,
            "bootstrap_starts": args.bootstrap_starts,
            "bootstrap_jobs": args.bootstrap_jobs,
            "huber_delta": args.huber_delta,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "bootstrap_method": "pairs bootstrap: resample 44 runs with replacement",
            "confidence_interval_method": "2.5th and 97.5th bootstrap percentiles",
        },
        "parameters": parameters,
        "confidence_intervals_95_percent": confidence_intervals,
        "isoflop_comparison": {
            "isoflop_a": args.isoflop_a,
            "chinchilla_a": parameters["a"],
            "a_difference_chinchilla_minus_isoflop": parameters["a"] - args.isoflop_a,
            "isoflop_b": args.isoflop_b,
            "chinchilla_b": parameters["b"],
            "b_difference_chinchilla_minus_isoflop": parameters["b"] - args.isoflop_b,
            "isoflop_a_plus_b": args.isoflop_a + args.isoflop_b,
            "chinchilla_a_plus_b": parameters["a"] + parameters["b"],
        },
        "optimization": optimization,
        "fit_metrics": metrics,
        "bootstrap": {
            "successful_samples": len(bootstrap_records),
            "failed_samples": len(bootstrap_failures),
            "failures": bootstrap_failures,
        },
        "artifacts": {
            "input_csv": str(input_csv),
            "fitted_points_csv": str(fitted_csv),
            "bootstrap_samples_csv": str(bootstrap_csv),
            "parameter_intervals_csv": str(interval_csv),
            "error_plot_png": str(error_plot),
            "error_plot_pdf": str(error_plot.with_suffix(".pdf")),
        },
        "elapsed_seconds": time.time() - started_at,
    }
    result_json = args.output_dir / "chinchilla_fit_results.json"
    result["artifacts"]["result_json"] = str(result_json)
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
