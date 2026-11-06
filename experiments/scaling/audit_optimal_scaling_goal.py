#!/usr/bin/env python3
"""Audit all artifacts required by the compute-optimal scaling goal."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_BUDGET_COUNTS = {
    3.5e17: 13,
    5.0e17: 5,
    7.0e17: 5,
    1.0e18: 5,
    1.3e18: 5,
    1.6e18: 5,
    2.0e18: 8,
    2.4e18: 5,
    2.9e18: 5,
    3.5e18: 11,
}


def load_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing JSON: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"invalid JSON {path}: {exc}")
        return {}


def check_image(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing image: {path}")
        return {}
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as image:
            width, height = image.size
            deviation = float(ImageStat.Stat(image.convert("L")).stddev[0])
        if width < 800 or height < 600:
            issues.append(f"image too small: {path} ({width}x{height})")
        if deviation < 1.0:
            issues.append(f"image appears blank: {path} (std={deviation})")
        return {"path": str(path), "width": width, "height": height, "grayscale_stddev": deviation}
    except Exception as exc:
        issues.append(f"cannot inspect image {path}: {exc}")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--plot-dir", type=Path, required=True)
    args = ap.parse_args()

    issues: list[str] = []
    completion = load_json(args.experiment_root / "completion_verification.json", issues)
    if completion.get("ok") is not True or int(completion.get("verified_rows", -1)) != 39:
        issues.append("extension completion verification is not 39/39 OK")

    combined = load_json(args.plot_dir / "all_fixed_flops_summary.json", issues)
    if int(combined.get("completed_runs", -1)) != 67:
        issues.append(f"combined completed run count is {combined.get('completed_runs')}, expected 67")
    observed_budget_counts = {
        float(item["flops"]): int(item["runs"]) for item in combined.get("budgets", [])
    }
    if observed_budget_counts != EXPECTED_BUDGET_COUNTS:
        issues.append(f"budget counts mismatch: {observed_budget_counts}")

    fit_summary = load_json(args.plot_dir / "optimal_scaling_fit_summary.json", issues)
    if int(fit_summary.get("point_count", -1)) != 10:
        issues.append(f"optimal point count is {fit_summary.get('point_count')}, expected 10")
    if int(fit_summary.get("bracketed_point_count", -1)) != 10:
        issues.append(
            f"bracketed optimum count is {fit_summary.get('bracketed_point_count')}, expected 10; "
            f"unbracketed={fit_summary.get('unbracketed_budgets')}"
        )
    for key in ("fit_log10_N_vs_log10_flops", "fit_log10_D_vs_log10_flops"):
        fit = fit_summary.get(key) or {}
        for field in ("slope", "intercept", "r_squared"):
            value = fit.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                issues.append(f"{key}.{field} is not finite: {value}")
        if int(fit.get("point_count", -1)) != 10:
            issues.append(f"{key}.point_count is {fit.get('point_count')}, expected 10")

    optimal_csv = args.plot_dir / "fixed_flops_optimal_points.csv"
    optimal_rows: list[dict[str, str]] = []
    if not optimal_csv.exists():
        issues.append(f"missing optimal-point CSV: {optimal_csv}")
    else:
        with optimal_csv.open(encoding="utf-8") as f:
            optimal_rows = list(csv.DictReader(f))
        if len(optimal_rows) != 10:
            issues.append(f"optimal-point CSV has {len(optimal_rows)} rows, expected 10")
        budgets = {float(row["target_flops_6nd"]) for row in optimal_rows}
        if budgets != set(EXPECTED_BUDGET_COUNTS):
            issues.append(f"optimal-point CSV budgets mismatch: {sorted(budgets)}")

    image_results = [
        check_image(args.plot_dir / "all_fixed_flops_overlay.png", issues),
        check_image(args.plot_dir / "all_fixed_flops_overlay_full.png", issues),
        check_image(args.plot_dir / "all_fixed_flops_facets.png", issues),
        check_image(args.plot_dir / "optimal_logN_vs_logFLOPs_linear_fit.png", issues),
        check_image(args.plot_dir / "optimal_logD_vs_logFLOPs_linear_fit.png", issues),
    ]
    for name in (
        "all_fixed_flops_overlay.pdf",
        "all_fixed_flops_overlay_full.pdf",
        "all_fixed_flops_facets.pdf",
        "optimal_logN_vs_logFLOPs_linear_fit.pdf",
        "optimal_logD_vs_logFLOPs_linear_fit.pdf",
    ):
        path = args.plot_dir / name
        if not path.exists() or path.stat().st_size < 10_000:
            issues.append(f"missing or undersized PDF: {path}")

    result = {
        "ok": not issues,
        "issues": issues,
        "expected_new_runs": 39,
        "expected_combined_runs": 67,
        "expected_budgets": 10,
        "expected_optimal_points": 10,
        "budget_counts": observed_budget_counts,
        "fit_log10_N_vs_log10_flops": fit_summary.get("fit_log10_N_vs_log10_flops"),
        "fit_log10_D_vs_log10_flops": fit_summary.get("fit_log10_D_vs_log10_flops"),
        "images": image_results,
        "optimal_points_csv": str(optimal_csv),
    }
    output_path = args.plot_dir / "goal_completion_audit.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "issues": issues, "output": str(output_path)}, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
