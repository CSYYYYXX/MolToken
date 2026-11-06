#!/usr/bin/env python3
"""Diagnostic fit of the existing peptide N x D grid, using validation NLL.

This is not a new IsoFLOP experiment or evidence of empirical bracketing.
Run on a worker; no checkpoint, training data, or test data is opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from fit_chinchilla_loss import (
    decode_parameters,
    fit_metrics,
    fit_multistart,
    parameter_bounds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=32)
    args = parser.parse_args()
    with args.completed_csv.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["domain"] == "peptide"]
    if len(rows) != 25 or len({row["run_id"] for row in rows}) != 25:
        raise ValueError("Expected the original 25 distinct peptide runs")
    if any(row["metric_split"] != "validation" or row["validation_complete_view"] != "true" for row in rows):
        raise ValueError("Every peptide row must report complete validation")
    n = np.asarray([int(row["parameter_count"]) for row in rows], dtype=float)
    d = np.asarray([int(row["actual_input_tokens"]) for row in rows], dtype=float)
    loss = np.asarray([float(row["validation_loss"]) for row in rows])
    log_n, log_d, log_loss = np.log(n), np.log(d), np.log(loss)
    # Every observed N can reach each diagnostic C with D within that N's
    # observed range. Account for small run-specific token-budget overshoots.
    c_low = max(6 * size * min(d[n == size]) for size in set(n))
    c_high = min(6 * size * max(d[n == size]) for size in set(n))
    if c_low >= c_high:
        raise ValueError("No common compute interval across observed model sizes")
    compute_probes = [float(c_low), math.sqrt(c_low * c_high), float(c_high)]
    slices = []
    for epoch in sorted({float(row["target_epoch_fraction"]) for row in rows}):
        group = sorted((row for row in rows if float(row["target_epoch_fraction"]) == epoch), key=lambda row: int(row["parameter_count"]))
        best = min(group, key=lambda row: float(row["validation_loss"]))
        slices.append({
            "token_epoch": epoch,
            "observed_best_N": int(best["parameter_count"]),
            "observed_best_validation_loss": float(best["validation_loss"]),
            "right_boundary_minus_best_loss": float(group[-1]["validation_loss"]) - float(best["validation_loss"]),
        })
    fits = []
    for delta in (0.001, 0.01, 1.0):
        for seed in (20260906, 20260907):
            theta, diagnostics = fit_multistart(log_n, log_d, log_loss, rng=np.random.default_rng(seed), starts=args.starts, delta=delta, max_iterations=2500)
            if not diagnostics["best_converged"]:
                raise RuntimeError(f"Peptide optimization did not converge: {diagnostics}")
            parameters = decode_parameters(theta)
            metrics, prediction, _ = fit_metrics(theta, log_n, log_d, log_loss, delta)
            metrics = {key.replace("training_loss", "validation_loss"): value for key, value in metrics.items()}
            model_errors = [{"N": int(size), "mean_prediction_minus_observation": float(np.mean((prediction - loss)[n == size]))} for size in sorted(set(n))]
            implied_optima = []
            for c in compute_probes:
                optimal_n = parameters["G"] * (c / 6) ** parameters["a"]
                optimal_d = c / (6 * optimal_n)
                implied_optima.append({
                    "C": c, "N": optimal_n, "D": optimal_d,
                    "N_inside_observed_range": bool(min(n) <= optimal_n <= max(n)),
                    "D_inside_observed_range": bool(min(d) <= optimal_d <= max(d)),
                })
            entry = {
                "huber_delta_log_loss": delta,
                "optimizer_initialization_seed": seed,
                "parameters": parameters,
                "metrics": metrics,
                "optimizer": diagnostics,
                "near_parameter_bounds": [name for name, value, (lower, upper) in zip(("log_E", "log_A", "log_B", "log_alpha", "log_beta"), theta, parameter_bounds(log_loss)) if min(abs(value - lower), abs(value - upper)) < 0.01],
                "mean_residual_by_model_size": model_errors,
                "model_implied_optima_not_empirical": implied_optima,
            }
            fits.append(entry)
            print(json.dumps({"delta": delta, "seed": seed, "r2": metrics["r_squared_validation_loss"], "optima_N": [item["N"] for item in entry["model_implied_optima_not_empirical"]]}, allow_nan=False), flush=True)
    report = {
        "purpose": "diagnostic_existing_peptide_surface_not_bracketing_acceptance",
        "source_csv": str(args.completed_csv.resolve()),
        "source_evidence": "physical_machine_reported_complete_validation",
        "run_ids": [row["run_id"] for row in rows],
        "N_field": "parameter_count",
        "D_field": "actual_input_tokens",
        "loss_field": "validation_loss",
        "test_data_accessed": False,
        "new_training_started": False,
        "empirical_isoflop_bracketing_established": False,
        "N_range": [int(min(n)), int(max(n))],
        "D_range": [int(min(d)), int(max(d))],
        "common_compute_probe_interval": [float(c_low), float(c_high)],
        "equal_data_slices": slices,
        "fits": fits,
        "limitations": ["One training seed; optimizer starts are not independent training replications.", "No directly measured fixed-compute peptide curve; implied optima are model-dependent.", "Data above one token-epoch is repeated exposure, not additional unique data.", "Model-size changes also change width/depth; report systematic residuals.", "Positive additive law is monotone in N at fixed D; observed local reversals cannot be modeled by this form."],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
