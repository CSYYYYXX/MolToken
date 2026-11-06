#!/usr/bin/env python3
"""Plot final training loss versus parameters for each fixed-FLOPs budget."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def last_train_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines: deque[str] = deque(maxlen=1000)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
    last = None
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "train":
            last = row
    return last


def smooth_curve(xs: list[float], ys: list[float], points: int = 200) -> tuple[list[float], list[float]]:
    if len(xs) < 3:
        return xs, ys
    import numpy as np

    order = np.argsort(np.asarray(xs, dtype=float))
    x = np.asarray(xs, dtype=float)[order]
    y = np.asarray(ys, dtype=float)[order]
    grid = np.geomspace(float(x.min()), float(x.max()), points)
    try:
        from scipy.interpolate import PchipInterpolator

        values = PchipInterpolator(x, y)(grid)
    except Exception:
        values = np.interp(grid, x, y)
    return grid.tolist(), values.tolist()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_group(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
    rows.sort(key=lambda row: int(row["parameter_count"]))
    xs = [float(row["parameter_count"]) for row in rows]
    ys = [float(row["training_loss"]) for row in rows]
    sx, sy = smooth_curve(xs, ys)
    target_flops = float(rows[0]["target_flops_6nd"])
    train_tokens_per_epoch = float(rows[0]["train_tokens_per_epoch"])
    one_epoch_parameter_boundary = target_flops / (6.0 * train_tokens_per_epoch)

    import numpy as np

    curve_x = np.asarray(sx, dtype=float)
    curve_y = np.asarray(sy, dtype=float)
    if curve_x.min() < one_epoch_parameter_boundary < curve_x.max():
        boundary_y = float(
            np.interp(
                np.log(one_epoch_parameter_boundary),
                np.log(curve_x),
                curve_y,
            )
        )
        dashed = curve_x < one_epoch_parameter_boundary
        solid = curve_x > one_epoch_parameter_boundary
        ax.plot(
            [*curve_x[dashed], one_epoch_parameter_boundary],
            [*curve_y[dashed], boundary_y],
            color="#2b6f9f",
            linewidth=2.0,
            linestyle="--",
            label="fit (>1 epoch)",
        )
        ax.plot(
            [one_epoch_parameter_boundary, *curve_x[solid]],
            [boundary_y, *curve_y[solid]],
            color="#2b6f9f",
            linewidth=2.0,
            label="fit (<=1 epoch)",
        )
    elif curve_x.max() <= one_epoch_parameter_boundary:
        ax.plot(sx, sy, color="#2b6f9f", linewidth=2.0, linestyle="--", label="fit (>1 epoch)")
    else:
        ax.plot(sx, sy, color="#2b6f9f", linewidth=2.0, label="fit (<=1 epoch)")

    regular_rows = [row for row in rows if not bool(row["over_one_epoch"])]
    over_epoch_rows = [row for row in rows if bool(row["over_one_epoch"])]
    if regular_rows:
        ax.scatter(
            [float(row["parameter_count"]) for row in regular_rows],
            [float(row["training_loss"]) for row in regular_rows],
            color="#d73027",
            s=38,
            zorder=3,
            label="independent runs (<=1 epoch)",
        )
    if over_epoch_rows:
        ax.scatter(
            [float(row["parameter_count"]) for row in over_epoch_rows],
            [float(row["training_loss"]) for row in over_epoch_rows],
            facecolors="none",
            edgecolors="#d73027",
            marker="D",
            linewidths=1.6,
            s=48,
            zorder=3,
            label="independent runs (>1 epoch)",
        )
    best = min(rows, key=lambda row: float(row["training_loss"]))
    ax.scatter(
        [float(best["parameter_count"])],
        [float(best["training_loss"])],
        facecolors="none",
        edgecolors="#1a9850",
        linewidths=2.0,
        s=105,
        zorder=4,
        label="lowest observed",
    )
    for row in rows:
        epoch_suffix = f" ({float(row['epoch_fraction']):.2f} ep)" if row["over_one_epoch"] else ""
        ax.annotate(
            f"{row['model_label']}{epoch_suffix}",
            (float(row["parameter_count"]), float(row["training_loss"])),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Final training loss")
    ax.grid(True, which="both", alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--include-running", action="store_true")
    args = ap.parse_args()

    matrix = json.loads(args.matrix_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.matrix_json.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for row in matrix["rows"]:
        out_dir = Path(row["out_dir"])
        summary = read_json(out_dir / "summary.json")
        event = last_train_event(out_dir / "train_log.jsonl")
        global_step = int((summary or {}).get("global_step") or (event or {}).get("step") or 0)
        complete = bool(summary and global_step >= int(row["max_steps"]) and (out_dir / "final").exists())
        loss = None
        source = ""
        if summary and summary.get("final_logged_train_loss") is not None:
            loss = float(summary["final_logged_train_loss"])
            source = "summary.final_logged_train_loss"
        elif summary and summary.get("trailing_train_loss") is not None:
            loss = float(summary["trailing_train_loss"])
            source = "summary.trailing_train_loss"
        elif event and event.get("loss") is not None:
            loss = float(event["loss"])
            source = "train_log.last_loss"
        results.append(
            {
                "target_label": row["flops_target_label"],
                "target_flops_6nd": row["target_flops_6nd"],
                "model_id": row["model_id"],
                "model_label": row["model_label"],
                "parameter_count": row["parameter_count"],
                "epoch_fraction": row["epoch_fraction"],
                "train_tokens_per_epoch": row["train_tokens_per_epoch"],
                "max_steps": row["max_steps"],
                "global_step": global_step,
                "complete": complete,
                "over_one_epoch": bool(row.get("over_one_epoch", float(row["epoch_fraction"]) > 1.0)),
                "final_checkpoint_exists": (out_dir / "final").exists(),
                "training_loss": loss,
                "loss_source": source,
                "reused_completed_run": row.get("reused_completed_run", False),
                "out_dir": str(out_dir),
            }
        )

    write_csv(output_dir / "fixed_flops_sweep_results.csv", results)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row["training_loss"] is None:
            continue
        if not args.include_running and not row["complete"]:
            continue
        grouped[str(row["target_label"])].append(row)

    import matplotlib.pyplot as plt

    figures: dict[str, str] = {}
    target_order = [item["label"] for item in matrix["metadata"]["flops_targets"]]
    for label in target_order:
        rows = grouped.get(label, [])
        if not rows:
            continue
        target = float(rows[0]["target_flops_6nd"])
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        draw_group(ax, rows, f"Fixed FLOPs: {target:.2e} (6ND)")
        ax.legend(frameon=False, fontsize=8)
        path = output_dir / f"loss_vs_params_{label}.png"
        fig.savefig(path, dpi=200)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        figures[label] = str(path)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    for ax, label in zip(axes.flat, target_order):
        rows = grouped.get(label, [])
        if rows:
            target = float(rows[0]["target_flops_6nd"])
            draw_group(ax, rows, f"{target:.2e} FLOPs")
        else:
            ax.set_title(f"{label}: pending")
            ax.axis("off")
    for ax in list(axes.flat)[len(target_order):]:
        ax.axis("off")
    combined = output_dir / "fixed_flops_sweep_five_budgets.png"
    fig.savefig(combined, dpi=200)
    fig.savefig(combined.with_suffix(".pdf"))
    plt.close(fig)
    figures["combined"] = str(combined)

    summary = {"rows": results, "figures": figures}
    (output_dir / "fixed_flops_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": len(results), "complete": sum(bool(row["complete"]) for row in results), "figures": figures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
