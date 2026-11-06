#!/usr/bin/env python3
"""Plot FLOPs-aligned scaling runs and export train/domain log tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb

try:
    from .scaling_paths import DEFAULT_RUN_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_RUN_ROOT

DOMAIN_TABLE_FIELDS = (
    "domain_samples",
    "domain_sample_fraction",
    "domain_input_tokens",
    "domain_input_token_fraction",
    "domain_loss_tokens",
    "domain_loss_token_fraction",
    "domain_loss_sum",
    "domain_loss_contribution",
    "domain_loss",
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_for(row: dict[str, Any]) -> str:
    label = row.get("flops_target_label")
    if label:
        return str(label)
    ratio = float(row["ratio"])
    return f"f{int(round(ratio * 100)):03d}" if ratio < 10 else f"r{ratio:g}"


def train_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "train":
                rows.append(row)
    return rows


def final_training_loss(out_dir: Path, summary: dict[str, Any] | None, events: list[dict[str, Any]]) -> tuple[float | None, str]:
    if summary:
        trailing = summary.get("trailing_train_loss")
        if trailing is not None and int(summary.get("trailing_train_loss_tokens") or 0) > 0:
            return float(trailing), "summary.trailing_train_loss"
        logged = summary.get("final_logged_train_loss")
        if logged is not None:
            return float(logged), "summary.final_logged_train_loss"
    if events and events[-1].get("loss") is not None:
        return float(events[-1]["loss"]), "train_log.last_train"
    return None, "missing"


def ema(values: list[float], alpha: float) -> list[float]:
    smoothed: list[float] = []
    last: float | None = None
    for value in values:
        last = value if last is None else alpha * value + (1.0 - alpha) * last
        smoothed.append(last)
    return smoothed


def blend_with_white(color: str, amount: float) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def expand_domain_rows(base: dict[str, Any], event: dict[str, Any]) -> list[dict[str, Any]]:
    domains: set[str] = set()
    for field in DOMAIN_TABLE_FIELDS:
        value = event.get(field)
        if isinstance(value, dict):
            domains.update(str(k) for k in value)
    rows: list[dict[str, Any]] = []
    for domain in sorted(domains):
        row = dict(base)
        row["domain"] = domain
        for field in DOMAIN_TABLE_FIELDS:
            value = event.get(field)
            if isinstance(value, dict):
                row[field] = value.get(domain)
        rows.append(row)
    return rows


def check_required_domain_fields(events: list[dict[str, Any]]) -> bool:
    if not events:
        return False
    return all(field in events[-1] for field in DOMAIN_TABLE_FIELDS)


def make_plots(
    rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    output_dir: Path,
    model_filter: set[str] | None,
    alpha: float,
    line_cmap_name: str,
) -> dict[str, Any]:
    complete_rows = [r for r in final_rows if r["complete"] and r.get("final_training_loss") is not None]
    if model_filter:
        complete_rows = [r for r in complete_rows if r["model_id"] in model_filter]
    complete_rows.sort(key=lambda r: (r["model_id"], float(r["target_flops_6nd"])))

    model_ids = sorted({r["model_id"] for r in complete_rows})
    line_cmap = plt.get_cmap(line_cmap_name)
    color_values = [float(r["target_flops_6nd"]) for r in final_rows if not model_filter or r["model_id"] in model_filter]
    if not color_values:
        color_values = [float(r["target_flops_6nd"]) for r in event_rows if not model_filter or r["model_id"] in model_filter]
    color_norm = Normalize(vmin=min(color_values) if color_values else 0.0, vmax=max(color_values) if color_values else 1.0)

    def heat_color(row: dict[str, Any]) -> Any:
        return line_cmap(color_norm(float(row["target_flops_6nd"])))

    model_palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    all_plot_rows = [r for r in final_rows if not model_filter or r["model_id"] in model_filter]
    if not all_plot_rows:
        all_plot_rows = [r for r in event_rows if not model_filter or r["model_id"] in model_filter]
    model_order = sorted(
        {r["model_id"] for r in all_plot_rows},
        key=lambda model_id: min(
            int(r["parameter_count"])
            for r in all_plot_rows
            if r["model_id"] == model_id and r.get("parameter_count") is not None
        ),
    )
    model_base_color = {model_id: model_palette[i % len(model_palette)] for i, model_id in enumerate(model_order)}
    label_rank: dict[tuple[str, str], int] = {}
    label_count: dict[str, int] = {}
    for model_id in model_order:
        labels = sorted(
            {str(r["target_label"]) for r in all_plot_rows if r["model_id"] == model_id},
            key=lambda label: min(
                float(r["target_flops_6nd"])
                for r in all_plot_rows
                if r["model_id"] == model_id and str(r["target_label"]) == label
            ),
        )
        label_count[model_id] = len(labels)
        for i, label in enumerate(labels):
            label_rank[(model_id, label)] = i

    use_model_family_colors = len(model_order) > 1

    def model_family_color(row: dict[str, Any]) -> Any:
        model_id = str(row["model_id"])
        label = str(row["target_label"])
        n = max(label_count.get(model_id, 1), 1)
        rank = label_rank.get((model_id, label), 0)
        if n == 1:
            amount = 0.15
        else:
            # Small FLOPs are lighter, large FLOPs are darker, while the hue
            # stays fixed by model size.
            amount = 0.55 - 0.42 * (rank / (n - 1))
        return blend_with_white(model_base_color.get(model_id, "#333333"), amount)

    def row_color(row: dict[str, Any]) -> Any:
        if use_model_family_colors:
            return model_family_color(row)
        return heat_color(row)

    figures: dict[str, str] = {}
    if complete_rows:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for model_id in model_ids:
            sub = [r for r in complete_rows if r["model_id"] == model_id]
            ax.plot(
                [float(r["actual_token_budget_b"]) for r in sub],
                [float(r["final_training_loss"]) for r in sub],
                linewidth=2,
                label=sub[0]["model_label"],
                color=model_base_color.get(model_id, "#333333") if use_model_family_colors else "0.35",
                alpha=0.85 if use_model_family_colors else 0.6,
                zorder=1,
            )
            ax.scatter(
                [float(r["actual_token_budget_b"]) for r in sub],
                [float(r["final_training_loss"]) for r in sub],
                c=[row_color(r) for r in sub],
                s=42,
                edgecolors="black",
                linewidths=0.35,
                zorder=2,
            )
            for r in sub:
                ax.annotate(str(r["target_label"]), (float(r["actual_token_budget_b"]), float(r["final_training_loss"])), fontsize=7)
        ax.set_xlabel("Training data tokens (B)")
        ax.set_ylabel("Final training loss")
        ax.set_title("Final training loss vs data scale")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Model")
        if not use_model_family_colors:
            sm = plt.cm.ScalarMappable(norm=color_norm, cmap=line_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Target FLOPs (6ND)")
        fig.tight_layout()
        png = output_dir / "loss_vs_data_final.png"
        pdf = output_dir / "loss_vs_data_final.pdf"
        fig.savefig(png)
        fig.savefig(pdf)
        plt.close(fig)
        figures["loss_vs_data_final_png"] = str(png)
        figures["loss_vs_data_final_pdf"] = str(pdf)

    selected_events = [r for r in event_rows if (not model_filter or r["model_id"] in model_filter)]
    if selected_events:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
        for model_id in sorted({r["model_id"] for r in selected_events}):
            labels = sorted(
                {r["target_label"] for r in selected_events if r["model_id"] == model_id},
                key=lambda label: min(
                    float(r["target_flops_6nd"])
                    for r in selected_events
                    if r["model_id"] == model_id and r["target_label"] == label
                ),
            )
            for label in labels:
                sub = [r for r in selected_events if r["model_id"] == model_id and r["target_label"] == label]
                sub.sort(key=lambda r: int(r["step"]))
                if not sub:
                    continue
                losses = [float(r["loss"]) for r in sub]
                smooth = ema(losses, alpha)
                ax.plot(
                    [float(r["flops_6nd_at_step"]) for r in sub],
                    smooth,
                    linewidth=1.4,
                    alpha=0.9,
                    color=row_color(sub[0]),
                    label=f"{sub[0]['model_label']} {label}",
                )
        ax.set_xscale("log")
        ax.set_xlabel("Cumulative training FLOPs (6ND)")
        ax.set_ylabel("Training loss (EMA)")
        ax.set_title("Smoothed training loss vs FLOPs")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        if not use_model_family_colors:
            sm = plt.cm.ScalarMappable(norm=color_norm, cmap=line_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Target FLOPs (6ND)")
        fig.tight_layout()
        png = output_dir / "train_loss_vs_flops_smoothed.png"
        pdf = output_dir / "train_loss_vs_flops_smoothed.pdf"
        fig.savefig(png)
        fig.savefig(pdf)
        plt.close(fig)
        figures["train_loss_vs_flops_smoothed_png"] = str(png)
        figures["train_loss_vs_flops_smoothed_pdf"] = str(pdf)

    return {
        "figures": figures,
        "completed_rows_plotted": len(complete_rows),
        "training_loss_event_rows_plotted": len(selected_events),
        "line_colormap": line_cmap_name,
        "color_scheme": "model_family_shades" if use_model_family_colors else "target_flops_colormap",
        "model_base_colors": model_base_color if use_model_family_colors else {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--matrix-json", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--models", default="", help="Comma-separated model ids to plot, for example 0p06b,0p30b.")
    ap.add_argument("--target-labels", default="", help="Comma-separated target labels to plot, for example f025,f050,f090.")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--require-domain", action="store_true")
    ap.add_argument("--ema-alpha", type=float, default=0.08)
    ap.add_argument("--line-cmap", default="turbo", help="Matplotlib colormap for per-run lines ordered by target FLOPs.")
    args = ap.parse_args()

    exp_root = args.run_root / args.experiment_name
    matrix_path = args.matrix_json or (exp_root / "scaling_matrix.json")
    output_dir = args.output_dir or (exp_root / "results" / "figures_domainlog")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filter = {x.strip() for x in args.models.split(",") if x.strip()} or None
    target_filter = {x.strip() for x in args.target_labels.split(",") if x.strip()} or None

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_rows = matrix["rows"]

    final_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    missing_domain: list[str] = []

    for row in matrix_rows:
        if model_filter and row["model_id"] not in model_filter:
            continue
        label = label_for(row)
        if target_filter and label not in target_filter:
            continue
        out_dir = Path(row["out_dir"])
        summary = load_json(out_dir / "summary.json")
        run_config = load_json(out_dir / "run_config.json") or {}
        events = train_events(out_dir / "train_log.jsonl")
        max_steps = int(row["max_steps"])
        if summary:
            global_step = int(summary.get("global_step") or 0)
        elif events:
            global_step = int(events[-1].get("step") or 0)
        else:
            global_step = 0
        complete = bool(summary and global_step >= max_steps)
        if not complete:
            incomplete.append(f"{row['model_id']}/{label}:{global_step}/{max_steps}")
        log_domain_loss = bool((run_config.get("args") or {}).get("log_domain_loss"))
        domain_ok = check_required_domain_fields(events)
        if not (log_domain_loss and domain_ok):
            missing_domain.append(f"{row['model_id']}/{label}")
        loss, loss_source = final_training_loss(out_dir, summary, events)
        parameter_count = int((summary or {}).get("parameter_count", row["parameter_count"]))
        tokens_per_step = float(row["tokens_per_step"])
        base = {
            "model_id": row["model_id"],
            "model_label": row["model_label"],
            "target_label": label,
            "ratio": float(row["ratio"]),
            "parameter_count": parameter_count,
            "target_tokens": int(row["target_tokens"]),
            "actual_tokens": int(row["actual_tokens"]),
            "actual_token_budget_b": float(row["actual_token_budget_b"]),
            "target_flops_6nd": float(row.get("target_flops_6nd", row["flops_6nd"])),
            "flops_6nd": float(row["flops_6nd"]),
            "tokens_per_step": tokens_per_step,
            "max_steps": max_steps,
            "global_step": global_step,
            "complete": complete,
            "log_domain_loss": log_domain_loss,
            "domain_fields_ok": domain_ok,
            "out_dir": str(out_dir),
        }
        final_row = dict(base)
        final_row["final_training_loss"] = loss if complete else None
        final_row["current_training_loss"] = None if complete else loss
        final_row["loss_source"] = loss_source
        final_row["elapsed_sec"] = None if summary is None else summary.get("elapsed_sec")
        final_rows.append(final_row)

        for event in events:
            step = int(event.get("step") or 0)
            if step <= 0 or event.get("loss") is None:
                continue
            event_base = dict(base)
            event_base.update(
                {
                    "step": step,
                    "loss": float(event["loss"]),
                    "elapsed_sec": event.get("elapsed_sec"),
                    "lr": event.get("lr"),
                    "tokens_seen_est": float(step * tokens_per_step),
                    "flops_6nd_at_step": float(6.0 * parameter_count * step * tokens_per_step),
                }
            )
            event_rows.append(event_base)
            if domain_ok:
                domain_rows.extend(expand_domain_rows(event_base, event))

    if args.require_complete and incomplete:
        raise SystemExit("incomplete runs: " + ", ".join(incomplete))
    if args.require_domain and missing_domain:
        raise SystemExit("missing domain logging: " + ", ".join(missing_domain))

    final_rows.sort(key=lambda r: (r["model_id"], float(r["target_flops_6nd"])))
    event_rows.sort(key=lambda r: (r["model_id"], r["target_label"], int(r["step"])))
    domain_rows.sort(key=lambda r: (r["model_id"], r["target_label"], int(r["step"]), r["domain"]))
    write_csv(output_dir / "scaling_results_complete_and_running.csv", final_rows)
    write_csv(output_dir / "training_loss_events.csv", event_rows)
    write_csv(output_dir / "domain_training_loss_events.csv", domain_rows)

    plot_info = make_plots(matrix_rows, final_rows, event_rows, output_dir, model_filter, args.ema_alpha, args.line_cmap)
    manifest = {
        "matrix_json": str(matrix_path),
        "output_dir": str(output_dir),
        "models": sorted(model_filter) if model_filter else "all",
        "target_labels": sorted(target_filter) if target_filter else "all",
        "require_complete": bool(args.require_complete),
        "require_domain": bool(args.require_domain),
        "incomplete": incomplete,
        "missing_domain": missing_domain,
        "tables": {
            "results": str(output_dir / "scaling_results_complete_and_running.csv"),
            "training_loss_events": str(output_dir / "training_loss_events.csv"),
            "domain_training_loss_events": str(output_dir / "domain_training_loss_events.csv"),
        },
        **plot_info,
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
