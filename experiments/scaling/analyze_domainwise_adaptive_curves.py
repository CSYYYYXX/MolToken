#!/usr/bin/env python3
"""Extend the unchanged five-point bracketing rules to ALL additional models.

Never select a favorable five-point subset, discard observations, or interpret
a concave quadratic vertex as a minimum. Test data is not an input.
"""
from __future__ import annotations

import math

from review_phase_b_validation_bracketing import analyze_curve, solve_linear_3x3


def analyze_adaptive_curve(observations, *, loss_relative_tolerance=1e-5,
                           minimum_r_squared=0.8, vertex_margin_fraction=0.02):
    count = len(observations)
    if count == 5:
        return analyze_curve(observations, loss_relative_tolerance=loss_relative_tolerance,
                             minimum_r_squared=minimum_r_squared,
                             vertex_margin_fraction=vertex_margin_fraction)
    if count < 5:
        raise ValueError("bracketing requires at least five distinct observed model sizes")
    points = sorted(observations, key=lambda p: int(p["parameter_count"]))
    ns = [int(p["parameter_count"]) for p in points]
    losses = [float(p["validation_loss"]) for p in points]
    if len(set(ns)) != count or min(ns) <= 0 or any(not math.isfinite(v) or v <= 0 for v in losses):
        raise ValueError("invalid or repeated model size/loss; repeats require separate seed analysis")
    xs = [math.log10(n) for n in ns]
    span = xs[-1] - xs[0]
    center, scale = (xs[0] + xs[-1])/2, span/2
    us = [(x-center)/scale for x in xs]
    design = [[u*u, u, 1.0] for u in us]
    normal = [[sum(row[i]*row[j] for row in design) for j in range(3)] for i in range(3)]
    rhs = [sum(row[i]*y for row,y in zip(design,losses)) for i in range(3)]
    aa,bb,cc = solve_linear_3x3(normal,rhs)
    predicted = [aa*u*u+bb*u+cc for u in us]
    residuals = [y-p for y,p in zip(losses,predicted)]
    ss_res = sum(r*r for r in residuals)
    ss_total = sum((y-sum(losses)/count)**2 for y in losses)
    r2 = 1-ss_res/ss_total if ss_total > 0 else None
    a = aa/scale**2
    b = bb/scale-2*aa*center/scale**2
    c = cc-bb*center/scale+aa*center**2/scale**2
    vertex = center-scale*bb/(2*aa) if abs(aa)>1e-15 else None
    vertex_n = 10**vertex if vertex is not None and -300 <= vertex <= 300 else None
    slope_low = (2*aa*us[0]+bb)/scale
    slope_high = (2*aa*us[-1]+bb)/scale
    tol = max(1e-10,max(abs(y) for y in losses)*loss_relative_tolerance)
    slope_tol = tol/span
    curvature_tol = 2*tol/span**2
    discrete = min(range(count),key=losses.__getitem__)
    minimum = losses[discrete]
    minimizing = [i for i,y in enumerate(losses) if y <= minimum+tol]
    margin = min(y for i,y in enumerate(losses) if i!=discrete)-minimum
    unique = len(minimizing)==1 and margin>tol
    differences = [right-left for left,right in zip(losses,losses[1:])]
    increasing = all(d>tol for d in differences)
    decreasing = all(d<-tol for d in differences)
    fitted_index = min(range(count),key=predicted.__getitem__)
    convex = 2*a>curvature_tol
    good_fit = r2 is not None and r2>=minimum_r_squared
    below = sum(x<vertex for x in xs) if vertex is not None else 0
    above = sum(x>vertex for x in xs) if vertex is not None else 0
    inside = vertex is not None and xs[0]+vertex_margin_fraction*span<vertex<xs[-1]-vertex_margin_fraction*span
    relation = ("undefined" if vertex is None else "below_observed_n" if vertex<xs[0]
                else "above_observed_n" if vertex>xs[-1] else "inside_observed_n")
    interior = 0<discrete<count-1
    local_bracket = interior and losses[discrete-1]>minimum+tol and losses[discrete+1]>minimum+tol
    agreement = abs(fitted_index-discrete)<=1
    bracketed = unique and local_bracket and convex and inside and below>=2 and above>=2 and slope_low<-slope_tol and slope_high>slope_tol and agreement and good_fit
    status,direction,decision = "ambiguous","ambiguous","requires_manual_resolution"
    if bracketed:
        status,direction,decision = "bracketed","keep","accepted_validation_bracket"
    elif unique and discrete==0 and increasing and fitted_index==0 and slope_low>slope_tol and slope_high>slope_tol and good_fit:
        status,direction,decision = "unbracketed_lower_n","lower_n","add_smaller_model_at_same_budget"
    elif unique and discrete==count-1 and decreasing and fitted_index==count-1 and slope_low<-slope_tol and slope_high<-slope_tol and good_fit:
        status,direction,decision = "unbracketed_higher_n","higher_n","add_larger_model_at_same_budget"
    reasons=[]
    if not unique: reasons.append("discrete minimum not separated beyond loss tolerance")
    if not interior: reasons.append("best observed model is at a sampled boundary")
    if not convex: reasons.append("fitted vertex is not a positively curved minimum")
    if not inside: reasons.append("fitted vertex lacks the required interior margin")
    if below<2 or above<2: reasons.append(f"insufficient two-sided support: {below}/{above}")
    if not good_fit: reasons.append(f"quadratic R-squared {r2} below {minimum_r_squared}")
    if not agreement: reasons.append("fitted and observed discrete minima disagree")
    return {
        "valid_run_count":count,"bracketing_status":status,"placement_direction":direction,"decision":decision,
        "model_ids":[p["model_id"] for p in points],"run_ids":[p["run_id"] for p in points],
        "parameter_counts":ns,"validation_losses":losses,
        "discrete_argmin":{"index":discrete,"model_id":points[discrete]["model_id"],"parameter_count":ns[discrete],
                           "validation_loss":minimum,"unique_within_tolerance":unique,"margin_to_second_best":margin},
        "quadratic_fit":{"basis":"loss = a*(log10_N)^2 + b*log10_N + c","quadratic_coefficient_a":a,
                         "linear_coefficient_b":b,"intercept_c":c,"curvature_second_derivative":2*a,
                         "vertex_log10_n":vertex,"vertex_parameter_count":vertex_n,"predicted_losses":predicted,
                         "residuals":residuals,"rmse":math.sqrt(ss_res/count),"r_squared":r2,
                         "fitted_slope_lower_boundary":slope_low,"fitted_slope_upper_boundary":slope_high},
        "boundary_evidence":{"observed_loss_differences_in_increasing_n_order":differences,
                            "strictly_increasing_beyond_tolerance":increasing,"strictly_decreasing_beyond_tolerance":decreasing,
                            "fitted_discrete_argmin_index":fitted_index,"fitted_lower_slope":slope_low,"fitted_upper_slope":slope_high,
                            "vertex_relation":relation,"vertex_is_fitted_minimum":convex,
                            "measured_points_below_vertex":below,"measured_points_above_vertex":above,
                            "fit_meets_r_squared_threshold":good_fit},
        "thresholds":{"loss_absolute_tolerance":tol,"loss_relative_tolerance":loss_relative_tolerance,
                      "slope_tolerance":slope_tol,"curvature_tolerance":curvature_tol,
                      "minimum_r_squared":minimum_r_squared,"vertex_margin_fraction":vertex_margin_fraction},
        "ambiguity_reasons":reasons,"analysis_scope":"all_completed_distinct_model_sizes_no_subset_selection",
    }
