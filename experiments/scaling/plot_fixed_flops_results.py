#!/usr/bin/env python3
"""Plot fixed-FLOPs final training loss against model parameter count."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_train_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "train":
                events.append(row)
    return events


def final_training_loss(summary: dict[str, Any] | None, events: list[dict[str, Any]]) -> tuple[float | None, str]:
    if summary is not None:
        value = summary.get("final_logged_train_loss")
        if value is not None:
            return float(value), "summary.final_logged_train_loss"
        value = summary.get("trailing_train_loss")
        if value is not None:
            return float(value), "summary.trailing_train_loss"
    if events:
        value = events[-1].get("loss")
        if value is not None:
            return float(value), "train_log.last_loss"
    return None, ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def smooth_curve(xs: list[float], ys: list[float], points: int = 200) -> tuple[list[float], list[float]]:
    if len(xs) < 3:
        return xs, ys
    try:
        import numpy as np
        from scipy.interpolate import PchipInterpolator

        order = np.argsort(np.asarray(xs, dtype=float))
        x = np.asarray(xs, dtype=float)[order]
        y = np.asarray(ys, dtype=float)[order]
        grid = np.geomspace(float(x.min()), float(x.max()), points)
        interp = PchipInterpolator(x, y)
        return [float(v) for v in grid], [float(v) for v in interp(grid)]
    except Exception:
        import numpy as np

        order = np.argsort(np.asarray(xs, dtype=float))
        x = np.asarray(xs, dtype=float)[order]
        y = np.asarray(ys, dtype=float)[order]
        grid = np.geomspace(float(x.min()), float(x.max()), points)
        return [float(v) for v in grid], [float(v) for v in np.interp(grid, x, y)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--include-running", action="store_true")
    args = ap.parse_args()

    matrix = json.loads(args.matrix_json.read_text(encoding="utf-8"))
    rows = list(matrix["rows"])
    output_dir = args.output_dir or (args.matrix_json.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        out_dir = Path(row["out_dir"])
        summary = read_json(out_dir / "summary.json")
        events = read_train_events(out_dir / "train_log.jsonl")
        global_step = int((summary or {}).get("global_step") or (events[-1].get("step") if events else 0) or 0)
        max_steps = int(row["max_steps"])
        complete = bool(summary and global_step >= max_steps)
        loss, loss_source = final_training_loss(summary, events)
        final_ckpt = (out_dir / "final").exists()
        summary_rows.append(
            {
                "model_id": row["model_id"],
                "model_label": row["model_label"],
                "parameter_count": int(row["parameter_count"]),
                "target_label": row["flops_target_label"],
                "target_flops_6nd": float(row["target_flops_6nd"]),
                "flops_6nd": float(row["flops_6nd"]),
                "flops_relative_error": float(row.get("flops_relative_error", 0.0)),
                "actual_tokens": int(row["actual_tokens"]),
                "actual_token_budget_b": float(row["actual_token_budget_b"]),
                "max_steps": max_steps,
                "global_step": global_step,
                "complete": complete,
                "final_checkpoint_exists": final_ckpt,
                "training_loss": loss,
                "loss_source": loss_source,
                "out_dir": str(out_dir),
            }
        )

    write_csv(output_dir / "fixed_flops_results.csv", summary_rows)

    plot_rows = [
        r
        for r in summary_rows
        if r["training_loss"] is not None and (args.include_running or r["complete"])
    ]
    if not plot_rows:
        (output_dir / "fixed_flops_results_summary.json").write_text(
            json.dumps({"rows": summary_rows, "figures": {}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"rows": len(summary_rows), "plotted": 0, "output_dir": str(output_dir)}, ensure_ascii=False))
        return

    import matplotlib.pyplot as plt

    plot_rows.sort(key=lambda r: int(r["parameter_count"]))
    xs = [float(r["parameter_count"]) for r in plot_rows]
    ys = [float(r["training_loss"]) for r in plot_rows]
    sx, sy = smooth_curve(xs, ys)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(sx, sy, color="#1f78b4", linewidth=2.0, alpha=0.9, label="smoothed")
    ax.scatter(xs, ys, color="#e31a1c", s=42, zorder=3, label="runs")
    for r in plot_rows:
        ax.annotate(
            str(r["model_label"]),
            (float(r["parameter_count"]), float(r["training_loss"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Final training loss")
    target = float(plot_rows[0]["target_flops_6nd"])
    ax.set_title(f"Fixed-FLOPs cross-section ({target:.2e} 6ND FLOPs)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    png = output_dir / "fixed_flops_loss_vs_params.png"
    pdf = output_dir / "fixed_flops_loss_vs_params.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)

    info = {
        "rows": summary_rows,
        "plotted": len(plot_rows),
        "figures": {
            "fixed_flops_loss_vs_params_png": str(png),
            "fixed_flops_loss_vs_params_pdf": str(pdf),
            "fixed_flops_results_csv": str(output_dir / "fixed_flops_results.csv"),
        },
    }
    (output_dir / "fixed_flops_results_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(summary_rows), "plotted": len(plot_rows), "png": str(png)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
