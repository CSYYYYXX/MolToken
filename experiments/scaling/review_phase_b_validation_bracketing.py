#!/usr/bin/env python3
"""Build the validation-only Phase-B bracketing review for the Phase-C gate.

This command is deliberately non-intervening: it never launches or signals a
training process.  It first requires the strict Phase-B verifier to accept all
60 primary runs, then reads only ``summary.final_validation.loss`` from those
runs.  The test view and test metrics are never opened or used.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from finalize_domainwise_phase_matrix import (
    C_PHASE_B_NEIGHBORS,
    EXPECTED_B_BUDGETS,
    EXPECTED_C_BUDGETS,
    REVIEW_SCHEMA,
    STANDARD_DOMAINS,
    review_template,
    validate_phase_c_review,
)
from verify_domainwise_campaign import atomic_write_json, load_json, verify_campaign


EXPECTED_PRIMARY_RUNS = 60
EXPECTED_CURVES = 12
PRODUCER_SCHEMA = "domainwise_phase_b_validation_review_producer_v1"
BRACKETING_METHOD_VERSION = "log10n_quadratic_bracketing_v1"
METHOD_VERSION = "log10n_quadratic_bracketing_placement_v2"
PLACEMENT_SCHEMA = "domainwise_validation_placement_resolution_v1"


class ReviewBlocked(RuntimeError):
    """Evidence cannot safely authorize the next phase."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def solve_linear_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [list(row) + [float(rhs)] for row, rhs in zip(matrix, vector)]
    scale = max(abs(value) for row in matrix for value in row)
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= max(1e-15 * scale, 1e-30):
            raise ReviewBlocked("quadratic design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def quadratic_fit(log10_n: list[float], losses: list[float]) -> dict[str, Any]:
    if len(log10_n) != 5 or len(losses) != 5:
        raise ReviewBlocked("each Phase-B curve must have exactly five observations")
    if len(set(log10_n)) != 5:
        raise ReviewBlocked("Phase-B curve has duplicate parameter counts")
    center = (min(log10_n) + max(log10_n)) / 2.0
    half_range = (max(log10_n) - min(log10_n)) / 2.0
    if half_range <= 0:
        raise ReviewBlocked("Phase-B log10(N) span is zero")
    u = [(value - center) / half_range for value in log10_n]
    columns = [[value * value, value, 1.0] for value in u]
    normal = [
        [sum(row[i] * row[j] for row in columns) for j in range(3)]
        for i in range(3)
    ]
    rhs = [sum(row[i] * loss for row, loss in zip(columns, losses)) for i in range(3)]
    a_scaled, b_scaled, c_scaled = solve_linear_3x3(normal, rhs)
    predictions = [a_scaled * value * value + b_scaled * value + c_scaled for value in u]
    residuals = [actual - predicted for actual, predicted in zip(losses, predictions)]
    mean_loss = sum(losses) / len(losses)
    ss_res = sum(value * value for value in residuals)
    ss_total = sum((value - mean_loss) ** 2 for value in losses)
    r_squared = 1.0 - ss_res / ss_total if ss_total > 0 else None

    a = a_scaled / (half_range * half_range)
    b = b_scaled / half_range - 2.0 * a_scaled * center / (half_range * half_range)
    c = (
        c_scaled
        - b_scaled * center / half_range
        + a_scaled * center * center / (half_range * half_range)
    )
    vertex_u = -b_scaled / (2.0 * a_scaled) if abs(a_scaled) > 1e-15 else None
    vertex_log10_n = center + half_range * vertex_u if vertex_u is not None else None
    vertex_n = (
        10.0**vertex_log10_n
        if vertex_log10_n is not None and -300.0 <= vertex_log10_n <= 300.0
        else None
    )
    slopes = [(2.0 * a_scaled * value + b_scaled) / half_range for value in u]
    return {
        "basis": "loss = a*(log10_N)^2 + b*log10_N + c",
        "quadratic_coefficient_a": a,
        "linear_coefficient_b": b,
        "intercept_c": c,
        "curvature_second_derivative": 2.0 * a,
        "vertex_log10_n": vertex_log10_n,
        "vertex_parameter_count": vertex_n,
        "predicted_losses": predictions,
        "residuals": residuals,
        "rmse": math.sqrt(ss_res / len(losses)),
        "r_squared": r_squared,
        "fitted_slope_lower_boundary": slopes[0],
        "fitted_slope_upper_boundary": slopes[-1],
    }


def analyze_curve(
    observations: list[dict[str, Any]],
    *,
    loss_relative_tolerance: float = 1e-5,
    minimum_r_squared: float = 0.8,
    vertex_margin_fraction: float = 0.02,
) -> dict[str, Any]:
    """Classify one five-point IsoFLOP curve, conservatively and deterministically."""
    if len(observations) != 5:
        raise ReviewBlocked(f"curve has {len(observations)} observations instead of five")
    ordered = sorted(observations, key=lambda item: int(item["parameter_count"]))
    counts = [int(item["parameter_count"]) for item in ordered]
    model_ids = [str(item["model_id"]) for item in ordered]
    run_ids = [str(item["run_id"]) for item in ordered]
    losses = [float(item["validation_loss"]) for item in ordered]
    if len(set(counts)) != 5 or counts != sorted(counts):
        raise ReviewBlocked("curve parameter counts are not five unique increasing values")
    if any(count <= 0 for count in counts) or any(not finite_number(loss) or loss <= 0 for loss in losses):
        raise ReviewBlocked("curve contains an invalid parameter count or validation loss")

    log10_n = [math.log10(count) for count in counts]
    fit = quadratic_fit(log10_n, losses)
    loss_tolerance = max(1e-10, max(abs(loss) for loss in losses) * loss_relative_tolerance)
    slope_tolerance = loss_tolerance / (max(log10_n) - min(log10_n))
    curvature_tolerance = 2.0 * loss_tolerance / ((max(log10_n) - min(log10_n)) ** 2)
    minimum = min(losses)
    minimizing = [index for index, loss in enumerate(losses) if loss <= minimum + loss_tolerance]
    discrete_index = min(range(5), key=lambda index: losses[index])
    other_losses = [loss for index, loss in enumerate(losses) if index != discrete_index]
    margin_to_second = min(other_losses) - losses[discrete_index]
    unique_minimum = len(minimizing) == 1 and margin_to_second > loss_tolerance
    differences = [losses[index + 1] - losses[index] for index in range(4)]
    increasing = all(delta > loss_tolerance for delta in differences)
    decreasing = all(delta < -loss_tolerance for delta in differences)
    predicted = [float(value) for value in fit["predicted_losses"]]
    fitted_min_index = min(range(5), key=lambda index: predicted[index])
    r_squared = fit["r_squared"]
    good_fit = finite_number(r_squared) and float(r_squared) >= minimum_r_squared
    curvature = float(fit["curvature_second_derivative"])
    convex = curvature > curvature_tolerance
    vertex = fit["vertex_log10_n"]
    span = max(log10_n) - min(log10_n)
    lower_limit = min(log10_n) + vertex_margin_fraction * span
    upper_limit = max(log10_n) - vertex_margin_fraction * span
    vertex_inside = finite_number(vertex) and lower_limit < float(vertex) < upper_limit
    lower_slope = float(fit["fitted_slope_lower_boundary"])
    upper_slope = float(fit["fitted_slope_upper_boundary"])

    status = "ambiguous"
    direction = "ambiguous"
    decision = "requires_manual_resolution"
    reasons: list[str] = []
    points_below_vertex = (
        sum(value < float(vertex) for value in log10_n) if finite_number(vertex) else 0
    )
    points_above_vertex = (
        sum(value > float(vertex) for value in log10_n) if finite_number(vertex) else 0
    )
    if not unique_minimum:
        reasons.append("discrete validation minimum is tied or below the material-loss margin")
    elif discrete_index in {1, 2, 3}:
        local_bracket = (
            losses[discrete_index - 1] > losses[discrete_index] + loss_tolerance
            and losses[discrete_index + 1] > losses[discrete_index] + loss_tolerance
        )
        fitted_agreement = abs(fitted_min_index - discrete_index) <= 1
        if (
            local_bracket
            and convex
            and vertex_inside
            and points_below_vertex >= 2
            and points_above_vertex >= 2
            and lower_slope < -slope_tolerance
            and upper_slope > slope_tolerance
            and fitted_agreement
            and good_fit
        ):
            status = "bracketed"
            direction = "keep"
            decision = "accepted_validation_bracket"
        else:
            reasons.append(
                "interior minimum lacks a central discrete argmin and at least two measured "
                "points on each side of the fitted vertex"
            )
    elif discrete_index == 0:
        fitted_direction = (
            fitted_min_index == 0
            and lower_slope > slope_tolerance
            and upper_slope > slope_tolerance
        )
        if increasing and fitted_direction and good_fit:
            status = "unbracketed_lower_n"
            direction = "lower_n"
            decision = "move_neighboring_phase_c_curve_toward_lower_n"
        else:
            reasons.append("lower-boundary minimum lacks monotone and fitted-slope agreement")
    else:
        fitted_direction = (
            fitted_min_index == 4
            and lower_slope < -slope_tolerance
            and upper_slope < -slope_tolerance
        )
        if decreasing and fitted_direction and good_fit:
            status = "unbracketed_higher_n"
            direction = "higher_n"
            decision = "move_neighboring_phase_c_curve_toward_higher_n"
        else:
            reasons.append("upper-boundary minimum lacks monotone and fitted-slope agreement")

    vertex_relation = "undefined"
    if finite_number(vertex):
        if float(vertex) < min(log10_n):
            vertex_relation = "below_observed_n"
        elif float(vertex) > max(log10_n):
            vertex_relation = "above_observed_n"
        else:
            vertex_relation = "inside_observed_n"
    return {
        "valid_run_count": 5,
        "bracketing_status": status,
        "placement_direction": direction,
        "decision": decision,
        "model_ids": model_ids,
        "run_ids": run_ids,
        "parameter_counts": counts,
        "validation_losses": losses,
        "discrete_argmin": {
            "index": discrete_index,
            "model_id": model_ids[discrete_index],
            "parameter_count": counts[discrete_index],
            "validation_loss": losses[discrete_index],
            "unique_within_tolerance": unique_minimum,
            "margin_to_second_best": margin_to_second,
        },
        "quadratic_fit": fit,
        "boundary_evidence": {
            "observed_loss_differences_in_increasing_n_order": differences,
            "strictly_increasing_beyond_tolerance": increasing,
            "strictly_decreasing_beyond_tolerance": decreasing,
            "fitted_discrete_argmin_index": fitted_min_index,
            "fitted_lower_slope": lower_slope,
            "fitted_upper_slope": upper_slope,
            "vertex_relation": vertex_relation,
            "vertex_is_fitted_minimum": convex,
            "measured_points_below_vertex": points_below_vertex,
            "measured_points_above_vertex": points_above_vertex,
            "fit_meets_r_squared_threshold": good_fit,
        },
        "thresholds": {
            "loss_absolute_tolerance": loss_tolerance,
            "loss_relative_tolerance": loss_relative_tolerance,
            "slope_tolerance": slope_tolerance,
            "curvature_tolerance": curvature_tolerance,
            "minimum_r_squared": minimum_r_squared,
            "vertex_margin_fraction": vertex_margin_fraction,
        },
        "ambiguity_reasons": reasons,
    }


def phase_rows(proposal: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    rows = proposal.get("rows")
    if not isinstance(rows, list):
        raise ReviewBlocked("proposal has no rows list")
    selected = [row for row in rows if isinstance(row, dict) and row.get("execution_phase") == phase]
    return selected


def resolve_curve_placement(analysis: dict[str, Any]) -> dict[str, Any]:
    """Resolve the next search direction without upgrading headline evidence.

    A curve can lack two measured points on each side of its fitted minimum
    while clearly identifying which side needs more support.  A boundary
    minimum can likewise identify a search direction despite a small reversal
    elsewhere in the curve.  Neither case establishes a headline optimum.
    The original classifier output remains intact alongside this decision.
    """
    raw_status = str(analysis["bracketing_status"])
    fit = analysis["quadratic_fit"]
    boundary = analysis["boundary_evidence"]
    thresholds = analysis["thresholds"]
    last_index = int(analysis["valid_run_count"]) - 1
    argmin = int(analysis["discrete_argmin"]["index"])
    fitted_argmin = int(boundary["fitted_discrete_argmin_index"])
    unique = analysis["discrete_argmin"]["unique_within_tolerance"] is True
    good_fit = boundary["fit_meets_r_squared_threshold"] is True
    below = int(boundary["measured_points_below_vertex"])
    above = int(boundary["measured_points_above_vertex"])
    slope_tolerance = float(thresholds["slope_tolerance"])
    lower_slope = float(fit["fitted_slope_lower_boundary"])
    upper_slope = float(fit["fitted_slope_upper_boundary"])
    differences = boundary["observed_loss_differences_in_increasing_n_order"]
    reversal_limit = max(float(thresholds["loss_absolute_tolerance"]), float(fit["rmse"]))
    lower_direction_reversal = max([0.0, *(-float(value) for value in differences)])
    higher_direction_reversal = max([0.0, *(float(value) for value in differences)])
    direction = "ambiguous"
    rule = "unresolved_validation_evidence"
    if raw_status != "ambiguous":
        direction = str(analysis["placement_direction"])
        rule = "original_bracketing_classifier_resolved"
    elif unique and good_fit:
        interior_convex_vertex = (
            boundary["vertex_is_fitted_minimum"] is True
            and boundary["vertex_relation"] == "inside_observed_n"
            and lower_slope < -slope_tolerance
            and upper_slope > slope_tolerance
            and abs(fitted_argmin - argmin) <= 1
        )
        if interior_convex_vertex and below == 1 and above >= 2 and argmin <= 1:
            direction = "lower_n"
            rule = "fitted_minimum_has_insufficient_lower_n_support"
        elif interior_convex_vertex and above == 1 and below >= 2 and argmin >= last_index - 1:
            direction = "higher_n"
            rule = "fitted_minimum_has_insufficient_higher_n_support"
        elif (
            argmin == fitted_argmin == 0
            and lower_slope > slope_tolerance
            and upper_slope > slope_tolerance
            and lower_direction_reversal <= reversal_limit
        ):
            direction = "lower_n"
            rule = "boundary_minimum_and_fitted_trend_agree_with_residual_scale_reversal"
        elif (
            argmin == fitted_argmin == last_index
            and lower_slope < -slope_tolerance
            and upper_slope < -slope_tolerance
            and higher_direction_reversal <= reversal_limit
        ):
            direction = "higher_n"
            rule = "boundary_minimum_and_fitted_trend_agree_with_residual_scale_reversal"
    return {
        "schema_version": PLACEMENT_SCHEMA,
        "status": "resolved" if direction != "ambiguous" else "unresolved",
        "direction": direction,
        "rule": rule,
        "purpose": "place_neighboring_phase_c_models_only",
        "headline_bracketing_accepted": raw_status == "bracketed",
        "raw_bracketing_status": raw_status,
        "evidence": {
            "unique_discrete_minimum": unique,
            "discrete_argmin_index": argmin,
            "fitted_discrete_argmin_index": fitted_argmin,
            "fit_meets_r_squared_threshold": good_fit,
            "measured_points_below_vertex": below,
            "measured_points_above_vertex": above,
            "maximum_reversal_for_lower_n_direction": lower_direction_reversal,
            "maximum_reversal_for_higher_n_direction": higher_direction_reversal,
            "boundary_reversal_limit_max_loss_tolerance_and_fit_rmse": reversal_limit,
        },
    }


def curve_placement_direction(curve: dict[str, Any]) -> str:
    resolution = curve.get("placement_resolution")
    if resolution is not None:
        if not isinstance(resolution, dict) or resolution.get("status") != "resolved":
            raise ReviewBlocked("Phase-B curve has unresolved validation placement evidence")
        direction = str(resolution.get("direction"))
    else:
        direction = str(curve.get("placement_direction"))
    if direction not in {"keep", "lower_n", "higher_n"}:
        raise ReviewBlocked("Phase-B curve has unresolved validation placement direction")
    return direction


def validate_placement_replay(
    curve: dict[str, Any], analysis_config: dict[str, Any], model_counts: dict[str, int]
) -> str:
    """Recompute a v2 review's raw classification and placement from its losses."""
    arrays = [curve.get(key) for key in ("model_ids", "run_ids", "parameter_counts", "validation_losses")]
    if any(not isinstance(values, list) or len(values) != 5 for values in arrays):
        raise ValueError("v2 placement replay requires all five validation observations")
    model_ids, run_ids, counts, losses = arrays
    if len(set(map(str, model_ids))) != 5 or len(set(map(str, run_ids))) != 5:
        raise ValueError("v2 placement replay has duplicate model or run IDs")
    observations = []
    for model_id, run_id, count, loss in zip(model_ids, run_ids, counts, losses):
        if model_id not in model_counts or count != model_counts[model_id]:
            raise ValueError("v2 placement replay parameter counts differ from proposal models")
        if not finite_number(loss) or float(loss) <= 0:
            raise ValueError("v2 placement replay has invalid validation loss")
        observations.append(dict(model_id=model_id, run_id=run_id, parameter_count=count, validation_loss=loss))
    config = {}
    for key in ("loss_relative_tolerance", "minimum_r_squared", "vertex_margin_fraction"):
        value = analysis_config.get(key)
        if not finite_number(value):
            raise ValueError(f"v2 placement replay lacks numerical analysis_config.{key}")
        config[key] = float(value)
    if not (0 < config["loss_relative_tolerance"] < 1 and 0 <= config["minimum_r_squared"] <= 1 and 0 <= config["vertex_margin_fraction"] < 0.5):
        raise ValueError("v2 placement replay has out-of-range analysis thresholds")
    replay = analyze_curve(observations, **config)
    for key, value in replay.items():
        if curve.get(key) != value:
            raise ValueError(f"v2 placement replay disagrees with original classification field {key}")
    if curve.get("placement_resolution") != resolve_curve_placement(replay):
        raise ValueError("v2 placement replay disagrees with deterministic placement resolution")
    return curve_placement_direction(curve)


def proposal_phase_c_groups(proposal: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in phase_rows(proposal, "C_completion"):
        key = f"{row.get('domain')}/{row.get('budget_label')}"
        groups.setdefault(key, []).append(row)
    expected = {f"{domain}/{budget}" for domain in STANDARD_DOMAINS for budget in EXPECTED_C_BUDGETS}
    if set(groups) != expected or any(len(rows) != 5 for rows in groups.values()):
        raise ReviewBlocked("proposal Phase-C rows are not the exact 4x3x5 design")
    return groups


def adjusted_model_ids(
    proposal_ids: list[str],
    eligible_ids: list[str],
    required_directions: set[str],
) -> tuple[str, list[str]]:
    if len(proposal_ids) != 5 or len(set(proposal_ids)) != 5:
        raise ReviewBlocked("proposal Phase-C curve does not have five unique models")
    positions = {model_id: index for index, model_id in enumerate(eligible_ids)}
    if any(model_id not in positions for model_id in proposal_ids):
        raise ReviewBlocked("proposal contains a model that violates the four-epoch floor")
    lower = [model_id for model_id in eligible_ids if positions[model_id] < positions[proposal_ids[0]]]
    higher = [model_id for model_id in eligible_ids if positions[model_id] > positions[proposal_ids[-1]]]
    need_lower = "lower_n" in required_directions
    need_higher = "higher_n" in required_directions
    if not need_lower and not need_higher:
        return "keep_proposal", proposal_ids
    if need_lower and not lower:
        raise ReviewBlocked("no four-epoch-compliant model exists below the proposal boundary")
    if need_higher and not higher:
        raise ReviewBlocked("no candidate model exists above the proposal boundary")
    if need_lower and need_higher:
        return "adjust_both", [lower[-1], *proposal_ids[1:4], higher[0]]
    if need_lower:
        return "adjust_lower_n", [lower[-1], *proposal_ids[:4]]
    return "adjust_higher_n", [*proposal_ids[1:], higher[0]]


def build_phase_c_review_curves(
    proposal: dict[str, Any],
    phase_b_curves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    b_by_key = {
        f"{curve['domain']}/{curve['budget_label']}": curve for curve in phase_b_curves
    }
    if len(b_by_key) != EXPECTED_CURVES:
        raise ReviewBlocked("Phase-B curve decision set is incomplete or duplicated")
    placement_by_key = {
        key: curve_placement_direction(curve) for key, curve in b_by_key.items()
    }
    groups = proposal_phase_c_groups(proposal)
    specs = sorted(
        (spec for spec in proposal.get("model_specs", []) if isinstance(spec, dict)),
        key=lambda item: int(item["parameter_count"]),
    )
    if (
        len(specs) < 5
        or len({str(spec["model_id"]) for spec in specs}) != len(specs)
        or len({int(spec["parameter_count"]) for spec in specs}) != len(specs)
    ):
        raise ReviewBlocked("proposal model_specs are missing, duplicated, or too short")
    result: list[dict[str, Any]] = []
    for key in sorted(groups):
        domain, budget = key.split("/", 1)
        rows = sorted(groups[key], key=lambda row: int(row["parameter_count"]))
        proposal_ids = [str(row["model_id"]) for row in rows]
        minimum_count = max(float(row["minimum_parameter_count_four_epoch"]) for row in rows)
        eligible_ids = [
            str(spec["model_id"])
            for spec in specs
            if int(spec["parameter_count"]) >= minimum_count
        ]
        neighbor_keys = [f"{domain}/{neighbor}" for neighbor in sorted(C_PHASE_B_NEIGHBORS[budget])]
        directional_references = [
            neighbor
            for neighbor in neighbor_keys
            if placement_by_key[neighbor] != "keep"
        ]
        required = {
            placement_by_key[neighbor]
            for neighbor in directional_references
        }
        if not required.issubset({"lower_n", "higher_n"}):
            raise ReviewBlocked(f"invalid Phase-B placement direction for {key}: {sorted(required)}")
        decision, selected_ids = adjusted_model_ids(proposal_ids, eligible_ids, required)
        result.append(
            {
                "domain": domain,
                "budget_label": budget,
                "decision": decision,
                "responds_to_phase_b_curves": directional_references,
                "model_ids": selected_ids,
                "proposal_model_ids": proposal_ids,
                "required_directions": sorted(required),
                "minimum_parameter_count_four_epoch": minimum_count,
            }
        )
    return result


def load_b_rows(matrix_paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    proposal_paths: set[str] = set()
    for path in matrix_paths:
        matrix = load_json(path)
        if not isinstance(matrix, dict) or not isinstance(matrix.get("metadata"), dict):
            raise ReviewBlocked(f"invalid Phase-B matrix: {path}")
        metadata = matrix["metadata"]
        proposal_paths.add(str(Path(str(metadata.get("proposal_json", ""))).expanduser().resolve()))
        raw_rows = matrix.get("rows")
        if not isinstance(raw_rows, list):
            raise ReviewBlocked(f"Phase-B matrix has no rows list: {path}")
        rows.extend(raw_rows)
    if len(proposal_paths) != 1:
        raise ReviewBlocked(f"Phase-B matrices disagree on proposal path: {sorted(proposal_paths)}")
    return rows, sorted(proposal_paths)


def pending_review(
    proposal: dict[str, Any],
    c_rows: list[dict[str, Any]],
    verification: dict[str, Any],
    *,
    status: str,
    blockers: list[str],
) -> dict[str, Any]:
    result = review_template(proposal, c_rows)
    result.update(
        {
            "status": status,
            "generated_at": now_iso(),
            "test_data_accessed": False,
            "phase_b_primary_runs_validated": int(verification.get("counts", {}).get("accepted", 0)),
            "phase_b_curves_reviewed": 0,
            "all_phase_b_curves_bracketed": None,
            "review_basis": "summary.final_validation.loss only",
            "phase_b_verification": {
                "status": verification.get("status"),
                "counts": verification.get("counts"),
                "matrix_issues": verification.get("matrix_issues"),
                "checkpoint_load_validation": verification.get("checkpoint_load_validation"),
            },
            "approval_blockers": blockers,
            "producer": {
                "schema_version": PRODUCER_SCHEMA,
                "script": Path(__file__).name,
                "method_version": METHOD_VERSION,
            },
            "schema_evidence": {
                "target_review_schema": REVIEW_SCHEMA,
                "expected_primary_runs": EXPECTED_PRIMARY_RUNS,
                "expected_phase_b_curves": EXPECTED_CURVES,
                "validation_metric": "summary.final_validation.loss",
                "finalizer_compatibility_check": "not_run",
            },
        }
    )
    return result


def build_review(
    matrix_paths: list[Path],
    proposal_path: Path,
    *,
    loss_relative_tolerance: float = 1e-5,
    minimum_r_squared: float = 0.8,
    vertex_margin_fraction: float = 0.02,
) -> dict[str, Any]:
    proposal = load_json(proposal_path)
    if not isinstance(proposal, dict) or not isinstance(proposal.get("metadata"), dict):
        raise ReviewBlocked("proposal JSON is invalid")
    c_rows = phase_rows(proposal, "C_completion")
    verification = verify_campaign(
        matrix_paths,
        expected_total_rows=EXPECTED_PRIMARY_RUNS,
        load_checkpoints=True,
    )
    counts = verification.get("counts", {})
    if verification.get("status") == "incomplete":
        return pending_review(
            proposal,
            c_rows,
            verification,
            status="pending_validation_review",
            blockers=[
                f"strict Phase-B verification has {counts.get('accepted', 0)}/60 accepted "
                f"and {counts.get('pending', 0)} pending runs"
            ],
        )
    if (
        verification.get("status") != "complete"
        or int(counts.get("accepted", -1)) != EXPECTED_PRIMARY_RUNS
        or int(counts.get("pending", -1)) != 0
        or int(counts.get("rejected", -1)) != 0
    ):
        return pending_review(
            proposal,
            c_rows,
            verification,
            status="blocked_invalid_phase_b",
            blockers=["strict Phase-B verification did not accept exactly 60 primary runs"],
        )

    b_rows, matrix_proposals = load_b_rows(matrix_paths)
    if len(b_rows) != EXPECTED_PRIMARY_RUNS or len({str(row.get("run_id")) for row in b_rows}) != EXPECTED_PRIMARY_RUNS:
        raise ReviewBlocked("Phase-B matrices do not contain 60 unique run IDs")
    if matrix_proposals != [str(proposal_path.resolve())]:
        raise ReviewBlocked("review proposal differs from the proposal frozen into Phase-B")
    accepted = {str(run["run_id"]): run for run in verification["runs"] if run["status"] == "accepted"}
    observations: dict[str, list[dict[str, Any]]] = {}
    for row in b_rows:
        run_id = str(row["run_id"])
        if run_id not in accepted:
            raise ReviewBlocked(f"strict verifier did not accept {run_id}")
        summary_path = Path(str(accepted[run_id]["attempt_dir"])) / "summary.json"
        summary = load_json(summary_path)
        final_validation = summary.get("final_validation") if isinstance(summary, dict) else None
        loss = final_validation.get("loss") if isinstance(final_validation, dict) else None
        if not finite_number(loss) or float(loss) <= 0:
            raise ReviewBlocked(f"invalid final validation loss for {run_id}")
        domain = str(row.get("domain"))
        budget = str(row.get("budget_label"))
        key = f"{domain}/{budget}"
        observations.setdefault(key, []).append(
            {
                "run_id": run_id,
                "model_id": str(row["model_id"]),
                "parameter_count": int(row["parameter_count"]),
                "validation_loss": float(loss),
            }
        )

    expected_keys = {f"{domain}/{budget}" for domain in STANDARD_DOMAINS for budget in EXPECTED_B_BUDGETS}
    if set(observations) != expected_keys:
        raise ReviewBlocked(
            f"Phase-B curve set mismatch: missing={sorted(expected_keys-set(observations))} "
            f"extra={sorted(set(observations)-expected_keys)}"
        )
    phase_b_curves: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        domain, budget = key.split("/", 1)
        analysis = analyze_curve(
            observations[key],
            loss_relative_tolerance=loss_relative_tolerance,
            minimum_r_squared=minimum_r_squared,
            vertex_margin_fraction=vertex_margin_fraction,
        )
        phase_b_curves.append({
            "domain": domain,
            "budget_label": budget,
            **analysis,
            "placement_resolution": resolve_curve_placement(analysis),
        })

    blockers = [
        f"unresolved Phase-B placement: {curve['domain']}/{curve['budget_label']}"
        for curve in phase_b_curves
        if curve["placement_resolution"]["status"] != "resolved"
    ]
    try:
        phase_c_curves = build_phase_c_review_curves(proposal, phase_b_curves)
    except ReviewBlocked as exc:
        phase_c_curves = review_template(proposal, c_rows)["phase_c_curves"]
        blockers.append(str(exc))

    review: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "status": "approved_for_phase_c_freeze" if not blockers else "not_approved_for_phase_c_freeze",
        "proposal_matrix_version": proposal["metadata"].get("matrix_version"),
        "test_data_accessed": False,
        "phase_b_primary_runs_validated": EXPECTED_PRIMARY_RUNS,
        "phase_b_curves_reviewed": EXPECTED_CURVES,
        "all_phase_b_curves_bracketed": all(
            curve["bracketing_status"] == "bracketed" for curve in phase_b_curves
        ),
        "phase_b_curves": phase_b_curves,
        "phase_c_curves": phase_c_curves,
        "generated_at": now_iso(),
        "review_basis": "summary.final_validation.loss only",
        "phase_b_matrix_paths": [str(path) for path in matrix_paths],
        "phase_b_verification": {
            "status": verification["status"],
            "counts": verification["counts"],
            "matrix_issues": verification["matrix_issues"],
            "checkpoint_load_validation": verification["checkpoint_load_validation"],
        },
        "analysis_config": {
            "x": "log10(parameter_count)",
            "y": "final_validation.loss",
            "fit": "ordinary_least_squares_quadratic",
            "loss_relative_tolerance": loss_relative_tolerance,
            "minimum_r_squared": minimum_r_squared,
            "vertex_margin_fraction": vertex_margin_fraction,
            "raw_bracketing_method_version": BRACKETING_METHOD_VERSION,
            "placement_resolution_schema": PLACEMENT_SCHEMA,
        },
        "approval_blockers": blockers,
        "producer": {
            "schema_version": PRODUCER_SCHEMA,
            "script": Path(__file__).name,
            "method_version": METHOD_VERSION,
        },
        "schema_evidence": {
            "target_review_schema": REVIEW_SCHEMA,
            "expected_primary_runs": EXPECTED_PRIMARY_RUNS,
            "expected_phase_b_curves": EXPECTED_CURVES,
            "validation_metric": "summary.final_validation.loss",
            "finalizer_compatibility_check": "not_run" if blockers else "pending",
        },
    }
    if not blockers:
        # The finalizer validator requires the completed producer evidence.  Set
        # the candidate state before invoking it so the check is not circular;
        # any exception below replaces this optimistic candidate with ``failed``.
        review["schema_evidence"]["finalizer_compatibility_check"] = "passed"
        try:
            validate_phase_c_review(review, proposal, c_rows)
        except Exception as exc:
            review["status"] = "not_approved_for_phase_c_freeze"
            review["schema_evidence"]["finalizer_compatibility_check"] = "failed"
            review["approval_blockers"] = [
                f"finalizer compatibility validation failed: {type(exc).__name__}: {exc}"
            ]
        else:
            review["schema_evidence"]["finalizer_compatibility_check"] = "passed"
    return review


def main() -> int:
    script = Path(__file__).resolve()
    workspace = script.parents[5]
    campaign = workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1"
    followup = workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1_followup/phase_c"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", type=Path, action="append", default=None)
    parser.add_argument(
        "--proposal-json",
        type=Path,
        default=workspace / "scaling_data/reports/domainwise_scaling_matrix_proposal_v1.json",
    )
    parser.add_argument("--output-json", type=Path, default=followup / "phase_b_validation_review.json")
    parser.add_argument("--loss-relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--minimum-r-squared", type=float, default=0.8)
    parser.add_argument("--vertex-margin-fraction", type=float, default=0.02)
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    matrices = args.matrix_json or [
        campaign / "phase_b_matrix_local.json",
        campaign / "phase_b_matrix_remote.json",
    ]
    matrices = [path.expanduser().resolve() for path in matrices]
    proposal_path = args.proposal_json.expanduser().resolve()
    if not 0 < args.loss_relative_tolerance < 0.1:
        raise ValueError("--loss-relative-tolerance must be between zero and 0.1")
    if not 0 <= args.minimum_r_squared <= 1:
        raise ValueError("--minimum-r-squared must be between zero and one")
    if not 0 <= args.vertex_margin_fraction < 0.5:
        raise ValueError("--vertex-margin-fraction must be in [0, 0.5)")
    try:
        review = build_review(
            matrices,
            proposal_path,
            loss_relative_tolerance=args.loss_relative_tolerance,
            minimum_r_squared=args.minimum_r_squared,
            vertex_margin_fraction=args.vertex_margin_fraction,
        )
    except Exception as exc:
        # Keep unexpected evidence/schema conflicts fail-closed while still
        # producing an atomic, machine-readable review report.
        proposal = load_json(proposal_path)
        verification_stub = {
            "status": "review_failed",
            "counts": {"rows": 0, "accepted": 0, "pending": 0, "rejected": 0},
            "matrix_issues": [],
            "checkpoint_load_validation": True,
        }
        review = pending_review(
            proposal,
            phase_rows(proposal, "C_completion"),
            verification_stub,
            status="blocked_review_error",
            blockers=[f"{type(exc).__name__}: {exc}"],
        )
    output = args.output_json.expanduser().resolve()
    atomic_write_json(output, review)
    print(
        json.dumps(
            {
                "status": review["status"],
                "phase_b_primary_runs_validated": review["phase_b_primary_runs_validated"],
                "phase_b_curves_reviewed": review["phase_b_curves_reviewed"],
                "all_phase_b_curves_bracketed": review["all_phase_b_curves_bracketed"],
                "approval_blockers": review.get("approval_blockers", []),
                "output_json": str(output),
                "test_data_accessed": False,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 1 if args.require_approved and review["status"] != "approved_for_phase_c_freeze" else 0


if __name__ == "__main__":
    raise SystemExit(main())
