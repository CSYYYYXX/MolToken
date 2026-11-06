#!/usr/bin/env python3
"""Fit an IsoFLOP-constrained Chinchilla loss law to existing runs only.

With fixed IsoFLOP constants a, b = 1-a, and G, the model is

    alpha = b*s
    beta  = a*s
    A     = (a/b)*B*G**s
    L(N,D) = E + A/N**alpha + B/D**beta

Only E, B, and s are fitted. They are represented in log space to keep them
strictly positive. No training or checkpoint loading is performed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Every optimization contains at most 44 rows. One BLAS thread per bootstrap
# process avoids severe oversubscription on these very small arrays.
for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_variable, "1")

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp


FITTED_PARAMETER_NAMES = ("E", "B", "s")
RESTORED_PARAMETER_NAMES = ("E", "A", "B", "alpha", "beta", "s")


def collect_completed_runs(matrix_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Collect and validate one row per completed run output directory."""

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
            smoothed_loss = float(summary.get("trailing_train_loss", math.nan))
            smoothing_tokens = int(summary.get("trailing_train_loss_tokens", 0)) * world_size
            final_logged_loss = float(summary.get("final_logged_train_loss", math.nan))
            epoch_fraction = float(matrix_row["epoch_fraction"])
            over_one_epoch = bool(matrix_row.get("over_one_epoch", epoch_fraction > 1.0))
            if parameter_count <= 0:
                raise ValueError(f"non-positive parameter count in {output_dir}")
            if local_loss_tokens <= 0 or world_size <= 0 or training_tokens <= 0:
                raise ValueError(f"non-positive observed training tokens in {output_dir}")
            if not math.isfinite(smoothed_loss) or smoothed_loss <= 0.0 or smoothing_tokens <= 0:
                raise ValueError(f"invalid smoothed final training loss in {output_dir}")
            if not math.isfinite(final_logged_loss) or final_logged_loss <= 0.0:
                raise ValueError(f"invalid final logged training loss in {output_dir}")
            if over_one_epoch != (epoch_fraction > 1.0):
                raise ValueError(f"inconsistent epoch marker in {output_dir}")

            target_flops = float(matrix_row["target_flops_6nd"])
            observed_flops = 6.0 * parameter_count * training_tokens
            key = str(output_dir.resolve())
            row = {
                "model_id": matrix_row["model_id"],
                "model_label": matrix_row["model_label"],
                "parameter_count_N": parameter_count,
                "training_tokens_D": training_tokens,
                "smoothed_final_training_loss": smoothed_loss,
                "trailing_loss_smoothing_tokens": smoothing_tokens,
                "final_logged_training_loss": final_logged_loss,
                "target_flops_6nd": target_flops,
                "observed_flops_6nd": observed_flops,
                "observed_flops_relative_error": (observed_flops - target_flops) / target_flops,
                "epoch_fraction": epoch_fraction,
                "over_one_epoch": over_one_epoch,
                "train_tokens_per_epoch": float(matrix_row["train_tokens_per_epoch"]),
                "global_step": completed_steps,
                "max_steps": required_steps,
                "world_size": world_size,
                "local_loss_tokens": local_loss_tokens,
                "output_dir": key,
                "source_matrix": str(matrix_path.resolve()),
            }
            previous = rows_by_output.get(key)
            if previous is not None:
                comparable = (
                    "parameter_count_N",
                    "training_tokens_D",
                    "smoothed_final_training_loss",
                    "epoch_fraction",
                )
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


def decode_parameters(theta: np.ndarray, *, a: float, b: float, g: float) -> dict[str, float]:
    log_e, log_b_coefficient, log_s = theta
    s = float(np.exp(log_s))
    alpha = b * s
    beta = a * s
    log_a_coefficient = math.log(a / b) + log_b_coefficient + s * math.log(g)
    return {
        "E": float(np.exp(log_e)),
        "A": float(np.exp(log_a_coefficient)),
        "B": float(np.exp(log_b_coefficient)),
        "alpha": alpha,
        "beta": beta,
        "s": s,
    }


def model_components(
    theta: np.ndarray,
    log_n: np.ndarray,
    log_d: np.ndarray,
    *,
    a: float,
    b: float,
    g: float,
) -> tuple[np.ndarray, float]:
    log_e, log_b_coefficient, log_s = theta
    s = float(np.exp(log_s))
    log_g = math.log(g)
    components = np.vstack(
        (
            np.full_like(log_n, log_e),
            math.log(a / b) + log_b_coefficient + s * (log_g - b * log_n),
            log_b_coefficient - a * s * log_d,
        )
    )
    return components, s


def predict_log_loss(
    theta: np.ndarray,
    log_n: np.ndarray,
    log_d: np.ndarray,
    *,
    a: float,
    b: float,
    g: float,
) -> np.ndarray:
    components, _ = model_components(theta, log_n, log_d, a=a, b=b, g=g)
    return logsumexp(components, axis=0)


def huber_objective_and_gradient(
    theta: np.ndarray,
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    delta: float,
    a: float,
    b: float,
    g: float,
) -> tuple[float, np.ndarray]:
    components, s = model_components(theta, log_n, log_d, a=a, b=b, g=g)
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

    log_g = math.log(g)
    prediction_jacobian = np.column_stack(
        (
            component_weights[0],
            component_weights[1] + component_weights[2],
            component_weights[1] * s * (log_g - b * log_n)
            - component_weights[2] * a * s * log_d,
        )
    )
    gradient = np.mean(influence[:, None] * prediction_jacobian, axis=0)
    return float(np.mean(losses)), gradient


def parameter_bounds(log_loss: np.ndarray) -> list[tuple[float, float]]:
    return [
        (math.log(1e-8), float(np.min(log_loss) - 1e-10)),
        (-40.0, 200.0),
        (math.log(1e-4), math.log(5.0)),
    ]


def random_initial_theta(
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    rng: np.random.Generator,
    *,
    a: float,
    b: float,
    g: float,
) -> np.ndarray:
    minimum_loss = float(np.exp(np.min(log_loss)))
    median_loss = float(np.exp(np.median(log_loss)))
    e_value = minimum_loss * rng.uniform(0.15, 0.98)
    s = float(np.exp(rng.uniform(math.log(0.03), math.log(1.5))))
    log_n_median = float(np.median(log_n))
    log_d_median = float(np.median(log_d))
    log_q = float(
        logsumexp(
            [
                math.log(a / b) + s * (math.log(g) - b * log_n_median),
                -a * s * log_d_median,
            ]
        )
    )
    reducible_loss = max(median_loss - e_value, 0.02 * median_loss)
    log_b_coefficient = math.log(reducible_loss) - log_q + rng.normal(0.0, 0.4)
    return np.asarray([math.log(e_value), log_b_coefficient, math.log(s)], dtype=float)


def fit_multistart(
    log_n: np.ndarray,
    log_d: np.ndarray,
    log_loss: np.ndarray,
    *,
    rng: np.random.Generator,
    starts: int,
    delta: float,
    max_iterations: int,
    a: float,
    b: float,
    g: float,
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
        perturbation_count = min(2, starts - 1)
        for _ in range(perturbation_count):
            perturbed = clipped_warm + rng.normal(0.0, [0.08, 0.35, 0.18])
            initial_thetas.append(
                np.asarray(
                    [np.clip(value, lower, upper) for value, (lower, upper) in zip(perturbed, bounds)],
                    dtype=float,
                )
            )
    while len(initial_thetas) < starts:
        initial_thetas.append(random_initial_theta(log_n, log_d, log_loss, rng, a=a, b=b, g=g))

    results: list[OptimizeResult] = []
    for initial_theta in initial_thetas:
        result = minimize(
            huber_objective_and_gradient,
            initial_theta,
            args=(log_n, log_d, log_loss, delta, a, b, g),
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
    *,
    delta: float,
    a: float,
    b: float,
    g: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    log_prediction = predict_log_loss(theta, log_n, log_d, a=a, b=b, g=g)
    prediction = np.exp(log_prediction)
    observed = np.exp(log_loss)
    log_residual = log_prediction - log_loss
    raw_residual = prediction - observed
    huber, _ = huber_objective_and_gradient(theta, log_n, log_d, log_loss, delta, a, b, g)
    log_tss = float(np.sum((log_loss - np.mean(log_loss)) ** 2))
    raw_tss = float(np.sum((observed - np.mean(observed)) ** 2))
    metrics = {
        "mean_huber_log_residual": huber,
        "rmse_log_loss": float(np.sqrt(np.mean(log_residual**2))),
        "mae_log_loss": float(np.mean(np.abs(log_residual))),
        "median_absolute_log_loss": float(np.median(np.abs(log_residual))),
        "max_absolute_log_loss": float(np.max(np.abs(log_residual))),
        "rmse_training_loss": float(np.sqrt(np.mean(raw_residual**2))),
        "mae_training_loss": float(np.mean(np.abs(raw_residual))),
        "r_squared_log_loss": 1.0 - float(np.sum(log_residual**2)) / log_tss if log_tss else 1.0,
        "r_squared_training_loss": 1.0 - float(np.sum(raw_residual**2)) / raw_tss if raw_tss else 1.0,
    }
    return metrics, prediction, log_residual


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
        float,
        float,
        float,
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
        a,
        b,
        g,
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
            a=a,
            b=b,
            g=g,
            warm_theta=fitted_theta,
        )
        decoded = decode_parameters(theta, a=a, b=b, g=g)
        if not all(
            math.isfinite(decoded[name]) and decoded[name] > 0.0
            for name in RESTORED_PARAMETER_NAMES
        ):
            raise ValueError("non-finite or non-positive bootstrap parameter")
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
    a: float,
    b: float,
    g: float,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    child_seeds = np.random.SeedSequence(seed).spawn(samples)
    tasks = [
        (
            index,
            int(child_seed.generate_state(1)[0]),
            log_n,
            log_d,
            log_loss,
            fitted_theta,
            starts,
            delta,
            max_iterations,
            a,
            b,
            g,
        )
        for index, child_seed in enumerate(child_seeds)
    ]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    report_every = max(1, samples // 10)
    if jobs == 1:
        results = map(fit_bootstrap_sample, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)
        results = executor.map(fit_bootstrap_sample, tasks, chunksize=1)
    try:
        for completed, (record, failure) in enumerate(results, start=1):
            if record is not None:
                records.append(record)
            if failure is not None:
                failures.append(failure)
            if completed % report_every == 0 or completed == samples:
                print(
                    f"{label} bootstrap {completed}/{samples}: "
                    f"success={len(records)} failure={len(failures)}",
                    flush=True,
                )
    finally:
        if jobs != 1:
            executor.shutdown(wait=True)
    records.sort(key=lambda row: int(row["bootstrap_index"]))
    failures.sort(key=lambda row: int(row["bootstrap_index"]))
    return records, failures


def percentile_intervals(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("no successful bootstrap fits")
    intervals: dict[str, dict[str, float]] = {}
    for name in RESTORED_PARAMETER_NAMES:
        values = np.asarray([float(record[name]) for record in records], dtype=float)
        lower, median, upper = np.percentile(values, [2.5, 50.0, 97.5])
        intervals[name] = {
            "lower_2.5_percent": float(lower),
            "median": float(median),
            "upper_97.5_percent": float(upper),
        }
    return intervals


def predict_from_decoded(parameters: dict[str, float], n: np.ndarray, d: np.ndarray) -> np.ndarray:
    log_components = np.vstack(
        (
            np.full_like(n, math.log(parameters["E"]), dtype=float),
            math.log(parameters["A"]) - parameters["alpha"] * np.log(n),
            math.log(parameters["B"]) - parameters["beta"] * np.log(d),
        )
    )
    return np.exp(logsumexp(log_components, axis=0))


def compute_valley_alignment(
    all_rows: list[dict[str, Any]],
    main_parameters: dict[str, float],
    *,
    a: float,
    g: float,
) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[float(row["target_flops_6nd"])].append(row)
    alignment_rows: list[dict[str, Any]] = []
    for compute in sorted(grouped):
        main_rows = sorted(
            (row for row in grouped[compute] if not bool(row["over_one_epoch"])),
            key=lambda row: int(row["parameter_count_N"]),
        )
        if not main_rows:
            raise ValueError(f"no <=1 epoch points for compute {compute:.3e}")
        observed_best = min(main_rows, key=lambda row: float(row["smoothed_final_training_loss"]))
        candidate_n = np.asarray([float(row["parameter_count_N"]) for row in main_rows])
        candidate_d = compute / (6.0 * candidate_n)
        candidate_prediction = predict_from_decoded(main_parameters, candidate_n, candidate_d)
        predicted_discrete_index = int(np.argmin(candidate_prediction))
        predicted_discrete_best = main_rows[predicted_discrete_index]
        predicted_continuous_n = g * (compute / 6.0) ** a
        observed_index = main_rows.index(observed_best)
        left_higher = (
            observed_index > 0
            and float(main_rows[observed_index - 1]["smoothed_final_training_loss"])
            > float(observed_best["smoothed_final_training_loss"])
        )
        right_higher = (
            observed_index + 1 < len(main_rows)
            and float(main_rows[observed_index + 1]["smoothed_final_training_loss"])
            > float(observed_best["smoothed_final_training_loss"])
        )
        alignment_rows.append(
            {
                "target_flops_6nd": compute,
                "main_point_count": len(main_rows),
                "over_one_epoch_point_count": len(grouped[compute]) - len(main_rows),
                "predicted_continuous_optimal_N": predicted_continuous_n,
                "observed_optimal_N": int(observed_best["parameter_count_N"]),
                "observed_optimal_model": observed_best["model_label"],
                "observed_optimal_smoothed_loss": float(observed_best["smoothed_final_training_loss"]),
                "predicted_discrete_optimal_N": int(predicted_discrete_best["parameter_count_N"]),
                "predicted_discrete_optimal_model": predicted_discrete_best["model_label"],
                "predicted_discrete_optimal_loss": float(candidate_prediction[predicted_discrete_index]),
                "observed_matches_predicted_discrete_optimum": (
                    int(observed_best["parameter_count_N"])
                    == int(predicted_discrete_best["parameter_count_N"])
                ),
                "observed_to_continuous_N_ratio": (
                    float(observed_best["parameter_count_N"]) / predicted_continuous_n
                ),
                "continuous_optimum_bracketed_by_main_points": (
                    float(candidate_n.min()) < predicted_continuous_n < float(candidate_n.max())
                ),
                "observed_minimum_has_higher_immediate_neighbors": left_higher and right_higher,
            }
        )
    return alignment_rows


def draw_fixed_compute_curves(
    path: Path,
    all_rows: list[dict[str, Any]],
    main_parameters: dict[str, float],
    sensitivity_parameters: dict[str, float],
    *,
    a: float,
    g: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[float(row["target_flops_6nd"])].append(row)
    budgets = sorted(grouped)
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 16.5), constrained_layout=True)
    for axis, compute in zip(axes.flat, budgets):
        rows = sorted(grouped[compute], key=lambda row: int(row["parameter_count_N"]))
        main_rows = [row for row in rows if not bool(row["over_one_epoch"])]
        over_rows = [row for row in rows if bool(row["over_one_epoch"])]
        predicted_optimal_n = g * (compute / 6.0) ** a
        minimum_n = min(float(row["parameter_count_N"]) for row in rows)
        maximum_n = max(float(row["parameter_count_N"]) for row in rows)
        grid_n = np.geomspace(
            min(0.78 * minimum_n, 0.70 * predicted_optimal_n),
            max(1.22 * maximum_n, 1.30 * predicted_optimal_n),
            400,
        )
        grid_d = compute / (6.0 * grid_n)
        main_curve = predict_from_decoded(main_parameters, grid_n, grid_d)
        sensitivity_curve = predict_from_decoded(sensitivity_parameters, grid_n, grid_d)
        optimal_d = compute / (6.0 * predicted_optimal_n)
        optimal_loss = float(
            predict_from_decoded(
                main_parameters,
                np.asarray([predicted_optimal_n]),
                np.asarray([optimal_d]),
            )[0]
        )

        axis.plot(grid_n, main_curve, color="#1f1f1f", linewidth=2.0)
        axis.plot(grid_n, sensitivity_curve, color="#0072B2", linewidth=1.5, linestyle="--")
        if main_rows:
            axis.scatter(
                [row["parameter_count_N"] for row in main_rows],
                [row["smoothed_final_training_loss"] for row in main_rows],
                color="#D55E00",
                edgecolors="white",
                linewidths=0.7,
                s=44,
                zorder=3,
            )
            observed_best = min(
                main_rows,
                key=lambda row: float(row["smoothed_final_training_loss"]),
            )
            axis.scatter(
                [observed_best["parameter_count_N"]],
                [observed_best["smoothed_final_training_loss"]],
                facecolors="none",
                edgecolors="#D55E00",
                linewidths=2.0,
                s=125,
                zorder=4,
            )
        if over_rows:
            axis.scatter(
                [row["parameter_count_N"] for row in over_rows],
                [row["smoothed_final_training_loss"] for row in over_rows],
                facecolors="none",
                edgecolors="#777777",
                marker="D",
                linewidths=1.5,
                s=50,
                zorder=3,
            )
        axis.axvline(predicted_optimal_n, color="#555555", linestyle=":", linewidth=1.2)
        axis.scatter(
            [predicted_optimal_n],
            [optimal_loss],
            color="#009E73",
            marker="*",
            s=105,
            zorder=5,
        )
        axis.set_xscale("log")
        axis.set_title(f"Fixed FLOPs = {compute:.1e}", fontsize=11)
        axis.set_xlabel("Model parameters N")
        axis.set_ylabel("Smoothed final training loss")
        axis.grid(True, which="both", alpha=0.22)

    legend = [
        Line2D([0], [0], color="#1f1f1f", linewidth=2.0, label="Main fit curve (<=1 epoch)"),
        Line2D([0], [0], color="#0072B2", linewidth=1.5, linestyle="--", label="All-44 sensitivity curve"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D55E00", markeredgecolor="white", markersize=7, label="Observed <=1 epoch"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="none", markeredgecolor="#777777", markersize=7, label="Observed >1 epoch"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#009E73", markeredgecolor="#009E73", markersize=11, label="Fixed-IsoFLOP predicted valley"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#D55E00", markeredgewidth=1.8, markersize=10, label="Observed <=1 epoch minimum"),
    ]
    figure.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("IsoFLOP-constrained Chinchilla predictions by fixed compute", fontsize=15)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def draw_fit_errors(
    path: Path,
    main_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    main_parameters: dict[str, float],
    sensitivity_parameters: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for axis, rows, parameters, title in (
        (axes[0], main_rows, main_parameters, f"Main fit ({len(main_rows)} runs, <=1 epoch)"),
        (axes[1], all_rows, sensitivity_parameters, f"Sensitivity fit ({len(all_rows)} runs, all points)"),
    ):
        observed = np.asarray([float(row["smoothed_final_training_loss"]) for row in rows])
        n_values = np.asarray([float(row["parameter_count_N"]) for row in rows])
        d_values = np.asarray([float(row["training_tokens_D"]) for row in rows])
        predicted = predict_from_decoded(parameters, n_values, d_values)
        colors = np.log10([float(row["target_flops_6nd"]) for row in rows])
        scatter = axis.scatter(
            observed,
            predicted,
            c=colors,
            cmap="viridis",
            s=46,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        lower = min(float(observed.min()), float(predicted.min()))
        upper = max(float(observed.max()), float(predicted.max()))
        padding = 0.05 * (upper - lower)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#303030")
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel("Observed smoothed final loss")
        axis.set_ylabel("Predicted loss")
        axis.set_title(title)
        axis.grid(True, alpha=0.22)
    colorbar = figure.colorbar(scatter, ax=axes, pad=0.02)
    colorbar.set_label("log10(Fixed FLOPs, 6ND)")
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_dataset(
    rows: list[dict[str, Any]],
    *,
    random_starts: int,
    delta: float,
    max_iterations: int,
    seed: int,
    a: float,
    b: float,
    g: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any], dict[str, float], np.ndarray, np.ndarray]:
    log_n = np.log([float(row["parameter_count_N"]) for row in rows])
    log_d = np.log([float(row["training_tokens_D"]) for row in rows])
    log_loss = np.log([float(row["smoothed_final_training_loss"]) for row in rows])
    theta, optimization = fit_multistart(
        log_n,
        log_d,
        log_loss,
        rng=np.random.default_rng(seed),
        starts=random_starts,
        delta=delta,
        max_iterations=max_iterations,
        a=a,
        b=b,
        g=g,
    )
    parameters = decode_parameters(theta, a=a, b=b, g=g)
    metrics, prediction, residual = fit_metrics(
        theta,
        log_n,
        log_d,
        log_loss,
        delta=delta,
        a=a,
        b=b,
        g=g,
    )
    return theta, parameters, optimization, metrics, prediction, residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-runs", type=int, default=44)
    parser.add_argument("--require-main-runs", type=int, default=38)
    parser.add_argument("--isoflop-a", type=float, default=0.582937)
    parser.add_argument("--isoflop-g", type=float, default=3.48e-3)
    parser.add_argument("--random-starts", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-starts", type=int, default=4)
    parser.add_argument("--bootstrap-jobs", type=int, default=4)
    parser.add_argument("--huber-delta", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if not 0.0 < args.isoflop_a < 1.0:
        parser.error("--isoflop-a must be between zero and one")
    if args.isoflop_g <= 0.0:
        parser.error("--isoflop-g must be positive")
    if min(args.random_starts, args.bootstrap_starts) < 2:
        parser.error("start counts must be at least two")
    if args.bootstrap_samples <= 0 or args.bootstrap_jobs <= 0:
        parser.error("bootstrap sample and job counts must be positive")
    if args.huber_delta <= 0.0:
        parser.error("--huber-delta must be positive")
    return args


def main() -> None:
    args = parse_args()
    started_at = time.time()
    a = float(args.isoflop_a)
    b = 1.0 - a
    g = float(args.isoflop_g)
    all_rows = collect_completed_runs(args.matrix_json)
    if len(all_rows) != args.require_runs:
        raise ValueError(f"found {len(all_rows)} completed runs, expected {args.require_runs}")
    main_rows = [row for row in all_rows if not bool(row["over_one_epoch"])]
    if len(main_rows) != args.require_main_runs:
        raise ValueError(f"found {len(main_rows)} <=1 epoch runs, expected {args.require_main_runs}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_csv = args.output_dir / "isoflop_constrained_input_44runs.csv"
    write_csv(input_csv, all_rows)

    main_theta, main_parameters, main_optimization, main_metrics, main_prediction, main_residual = fit_dataset(
        main_rows,
        random_starts=args.random_starts,
        delta=args.huber_delta,
        max_iterations=args.max_iterations,
        seed=args.seed,
        a=a,
        b=b,
        g=g,
    )
    sensitivity_theta, sensitivity_parameters, sensitivity_optimization, sensitivity_metrics, sensitivity_prediction, sensitivity_residual = fit_dataset(
        all_rows,
        random_starts=args.random_starts,
        delta=args.huber_delta,
        max_iterations=args.max_iterations,
        seed=args.seed + 100,
        a=a,
        b=b,
        g=g,
    )

    main_log_n = np.log([float(row["parameter_count_N"]) for row in main_rows])
    main_log_d = np.log([float(row["training_tokens_D"]) for row in main_rows])
    main_log_loss = np.log([float(row["smoothed_final_training_loss"]) for row in main_rows])
    all_log_n = np.log([float(row["parameter_count_N"]) for row in all_rows])
    all_log_d = np.log([float(row["training_tokens_D"]) for row in all_rows])
    all_log_loss = np.log([float(row["smoothed_final_training_loss"]) for row in all_rows])

    main_bootstrap, main_bootstrap_failures = run_bootstrap(
        main_log_n,
        main_log_d,
        main_log_loss,
        main_theta,
        samples=args.bootstrap_samples,
        starts=args.bootstrap_starts,
        delta=args.huber_delta,
        max_iterations=args.max_iterations,
        seed=args.seed + 1,
        jobs=args.bootstrap_jobs,
        a=a,
        b=b,
        g=g,
        label="main",
    )
    sensitivity_bootstrap, sensitivity_bootstrap_failures = run_bootstrap(
        all_log_n,
        all_log_d,
        all_log_loss,
        sensitivity_theta,
        samples=args.bootstrap_samples,
        starts=args.bootstrap_starts,
        delta=args.huber_delta,
        max_iterations=args.max_iterations,
        seed=args.seed + 2,
        jobs=args.bootstrap_jobs,
        a=a,
        b=b,
        g=g,
        label="all44",
    )
    minimum_success = math.ceil(0.9 * args.bootstrap_samples)
    if len(main_bootstrap) < minimum_success or len(sensitivity_bootstrap) < minimum_success:
        raise RuntimeError("fewer than 90% of bootstrap fits succeeded")
    main_intervals = percentile_intervals(main_bootstrap)
    sensitivity_intervals = percentile_intervals(sensitivity_bootstrap)

    main_bootstrap_csv = args.output_dir / "isoflop_constrained_bootstrap_main.csv"
    sensitivity_bootstrap_csv = args.output_dir / "isoflop_constrained_bootstrap_all44.csv"
    write_csv(main_bootstrap_csv, main_bootstrap)
    write_csv(sensitivity_bootstrap_csv, sensitivity_bootstrap)
    main_intervals_csv = args.output_dir / "isoflop_constrained_intervals_main.csv"
    sensitivity_intervals_csv = args.output_dir / "isoflop_constrained_intervals_all44.csv"
    write_csv(
        main_intervals_csv,
        [
            {"parameter": name, "estimate": main_parameters[name], **main_intervals[name]}
            for name in RESTORED_PARAMETER_NAMES
        ],
    )
    write_csv(
        sensitivity_intervals_csv,
        [
            {"parameter": name, "estimate": sensitivity_parameters[name], **sensitivity_intervals[name]}
            for name in RESTORED_PARAMETER_NAMES
        ],
    )

    fitted_rows: list[dict[str, Any]] = []
    main_predictions_by_output = {
        row["output_dir"]: (float(prediction), float(residual))
        for row, prediction, residual in zip(main_rows, main_prediction, main_residual)
    }
    for row, prediction, residual in zip(all_rows, sensitivity_prediction, sensitivity_residual):
        main_values = main_predictions_by_output.get(row["output_dir"])
        fitted_rows.append(
            {
                **row,
                "main_fit_prediction": main_values[0] if main_values else None,
                "main_fit_log_residual": main_values[1] if main_values else None,
                "all44_sensitivity_prediction": float(prediction),
                "all44_sensitivity_log_residual": float(residual),
            }
        )
    fitted_csv = args.output_dir / "isoflop_constrained_fitted_points.csv"
    write_csv(fitted_csv, fitted_rows)

    valley_rows = compute_valley_alignment(all_rows, main_parameters, a=a, g=g)
    valley_csv = args.output_dir / "isoflop_constrained_valley_alignment.csv"
    write_csv(valley_csv, valley_rows)
    curve_plot = args.output_dir / "isoflop_constrained_fixed_compute_curves.png"
    draw_fixed_compute_curves(
        curve_plot,
        all_rows,
        main_parameters,
        sensitivity_parameters,
        a=a,
        g=g,
    )
    error_plot = args.output_dir / "isoflop_constrained_fit_errors.png"
    draw_fit_errors(error_plot, main_rows, all_rows, main_parameters, sensitivity_parameters)

    result = {
        "model": "L(N,D) = E + [(a/b) B G^s] / N^(b s) + B / D^(a s)",
        "fixed_isoflop_parameters": {"a": a, "b": b, "G": g},
        "positive_parameterization": {
            "E": "exp(theta_E)",
            "B": "exp(theta_B)",
            "s": "exp(theta_s)",
        },
        "restoration": {
            "alpha": "b*s",
            "beta": "a*s",
            "A": "(a/b)*B*G^s",
        },
        "objective": f"mean Huber(ln(L_pred) - ln(L_true), delta={args.huber_delta:g})",
        "N_definition": "exact matrix parameter_count",
        "D_definition": "summary.total_train_loss_tokens multiplied by run_config.world_size",
        "loss_definition": "summary.trailing_train_loss",
        "source_matrices": [str(path.resolve()) for path in args.matrix_json],
        "configuration": {
            "all_run_count": len(all_rows),
            "main_run_count": len(main_rows),
            "excluded_over_one_epoch_count": len(all_rows) - len(main_rows),
            "random_starts_per_fit": args.random_starts,
            "bootstrap_samples_per_fit": args.bootstrap_samples,
            "bootstrap_starts": args.bootstrap_starts,
            "bootstrap_jobs": args.bootstrap_jobs,
            "huber_delta": args.huber_delta,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "bootstrap_method": "pairs bootstrap within each fitted dataset",
            "confidence_interval_method": "2.5th and 97.5th bootstrap percentiles",
        },
        "main_fit_leq_one_epoch": {
            "run_count": len(main_rows),
            "parameters": main_parameters,
            "confidence_intervals_95_percent": main_intervals,
            "optimization": main_optimization,
            "fit_metrics": main_metrics,
            "bootstrap": {
                "successful_samples": len(main_bootstrap),
                "failed_samples": len(main_bootstrap_failures),
                "failures": main_bootstrap_failures,
            },
        },
        "sensitivity_fit_all_44": {
            "run_count": len(all_rows),
            "parameters": sensitivity_parameters,
            "confidence_intervals_95_percent": sensitivity_intervals,
            "optimization": sensitivity_optimization,
            "fit_metrics": sensitivity_metrics,
            "bootstrap": {
                "successful_samples": len(sensitivity_bootstrap),
                "failed_samples": len(sensitivity_bootstrap_failures),
                "failures": sensitivity_bootstrap_failures,
            },
        },
        "valley_alignment": {
            "budget_count": len(valley_rows),
            "observed_matches_predicted_discrete_count": sum(
                bool(row["observed_matches_predicted_discrete_optimum"])
                for row in valley_rows
            ),
            "continuous_optimum_bracketed_count": sum(
                bool(row["continuous_optimum_bracketed_by_main_points"])
                for row in valley_rows
            ),
            "rows": valley_rows,
        },
        "artifacts": {
            "input_csv": str(input_csv),
            "fitted_points_csv": str(fitted_csv),
            "main_bootstrap_csv": str(main_bootstrap_csv),
            "all44_bootstrap_csv": str(sensitivity_bootstrap_csv),
            "main_intervals_csv": str(main_intervals_csv),
            "all44_intervals_csv": str(sensitivity_intervals_csv),
            "valley_alignment_csv": str(valley_csv),
            "fixed_compute_curves_png": str(curve_plot),
            "fixed_compute_curves_pdf": str(curve_plot.with_suffix(".pdf")),
            "fit_errors_png": str(error_plot),
            "fit_errors_pdf": str(error_plot.with_suffix(".pdf")),
        },
        "elapsed_seconds": time.time() - started_at,
    }
    result_json = args.output_dir / "isoflop_constrained_chinchilla_results.json"
    result["artifacts"]["result_json"] = str(result_json)
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
