#!/usr/bin/env python3
"""Plot per-domain training loss from domain-logging scaling runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

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
DOMAIN_ORDER = ("small_molecule", "protein", "peptide", "dna", "rna")
MODEL_PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
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


def has_domain_fields(event: dict[str, Any]) -> bool:
    return all(field in event for field in DOMAIN_TABLE_FIELDS)


def ema(values: list[float], alpha: float) -> list[float]:
    out: list[float] = []
    last: float | None = None
    for value in values:
        last = value if last is None else alpha * value + (1.0 - alpha) * last
        out.append(last)
    return out


def blend_with_white(color: str, amount: float) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def sorted_domains(domains: set[str]) -> list[str]:
    known = [d for d in DOMAIN_ORDER if d in domains]
    return known + sorted(domains - set(known))


def build_rows(
    matrix_rows: list[dict[str, Any]],
    model_filter: set[str] | None,
    target_filter: set[str] | None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
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
        global_step = int((summary or {}).get("global_step") or (events[-1].get("step") if events else 0) or 0)
        complete = bool(summary and global_step >= max_steps)
        if not complete:
            incomplete.append(f"{row['model_id']}/{label}:{global_step}/{max_steps}")
        log_domain_loss = bool((run_config.get("args") or {}).get("log_domain_loss"))
        domain_ok = bool(events and has_domain_fields(events[-1]))
        if not (log_domain_loss and domain_ok):
            missing_domain.append(f"{row['model_id']}/{label}")
        parameter_count = int((summary or {}).get("parameter_count", row["parameter_count"]))
        tokens_per_step = float(row["tokens_per_step"])
        base = {
            "model_id": row["model_id"],
            "model_label": row["model_label"],
            "target_label": label,
            "ratio": float(row["ratio"]),
            "parameter_count": parameter_count,
            "target_flops_6nd": float(row.get("target_flops_6nd", row["flops_6nd"])),
            "actual_token_budget_b": float(row["actual_token_budget_b"]),
            "tokens_per_step": tokens_per_step,
            "max_steps": max_steps,
            "global_step": global_step,
            "complete": complete,
            "log_domain_loss": log_domain_loss,
            "domain_fields_ok": domain_ok,
            "out_dir": str(out_dir),
        }
        for event in events:
            if not has_domain_fields(event):
                continue
            step = int(event.get("step") or 0)
            if step <= 0:
                continue
            for domain, loss in event["domain_loss"].items():
                domain = str(domain)
                loss_tokens = float(event.get("domain_loss_tokens", {}).get(domain) or 0.0)
                row_out = dict(base)
                row_out.update(
                    {
                        "step": step,
                        "loss": float(event["loss"]) if event.get("loss") is not None else None,
                        "elapsed_sec": event.get("elapsed_sec"),
                        "lr": event.get("lr"),
                        "tokens_seen_est": float(step * tokens_per_step),
                        "flops_6nd_at_step": float(6.0 * parameter_count * step * tokens_per_step),
                        "domain": domain,
                        "domain_loss": float(loss),
                        "domain_loss_tokens": loss_tokens,
                        "domain_loss_token_fraction": float(event.get("domain_loss_token_fraction", {}).get(domain) or 0.0),
                        "domain_input_token_fraction": float(event.get("domain_input_token_fraction", {}).get(domain) or 0.0),
                        "domain_sample_fraction": float(event.get("domain_sample_fraction", {}).get(domain) or 0.0),
                        "domain_loss_contribution": float(event.get("domain_loss_contribution", {}).get(domain) or 0.0),
                    }
                )
                domain_rows.append(row_out)
    domain_rows.sort(key=lambda r: (r["model_id"], r["target_label"], r["domain"], int(r["step"])))
    return domain_rows, incomplete, missing_domain


def tail_summary(domain_rows: list[dict[str, Any]], tail_events: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in domain_rows:
        grouped[(row["model_id"], row["target_label"], row["domain"])].append(row)
    rows: list[dict[str, Any]] = []
    for (_model_id, _target_label, _domain), group in grouped.items():
        group.sort(key=lambda r: int(r["step"]))
        tail = group[-tail_events:] if tail_events > 0 else group
        tokens = sum(float(r["domain_loss_tokens"]) for r in tail)
        if tokens > 0:
            loss = sum(float(r["domain_loss"]) * float(r["domain_loss_tokens"]) for r in tail) / tokens
        else:
            loss = sum(float(r["domain_loss"]) for r in tail) / max(len(tail), 1)
        base = dict(group[-1])
        base["tail_events"] = len(tail)
        base["tail_domain_loss"] = loss
        base["tail_domain_loss_token_fraction"] = sum(float(r["domain_loss_token_fraction"]) for r in tail) / max(len(tail), 1)
        base["tail_domain_input_token_fraction"] = sum(float(r["domain_input_token_fraction"]) for r in tail) / max(len(tail), 1)
        base["tail_domain_sample_fraction"] = sum(float(r["domain_sample_fraction"]) for r in tail) / max(len(tail), 1)
        base["tail_domain_loss_contribution"] = sum(float(r["domain_loss_contribution"]) for r in tail) / max(len(tail), 1)
        rows.append(base)
    rows.sort(key=lambda r: (int(r["parameter_count"]), float(r["target_flops_6nd"]), r["domain"]))
    return rows


def model_colors(rows: list[dict[str, Any]]) -> dict[str, str]:
    model_ids = sorted({r["model_id"] for r in rows}, key=lambda m: min(int(r["parameter_count"]) for r in rows if r["model_id"] == m))
    return {model_id: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, model_id in enumerate(model_ids)}


def label_colors(rows: list[dict[str, Any]], base_colors: dict[str, str]) -> dict[tuple[str, str], tuple[float, float, float]]:
    colors: dict[tuple[str, str], tuple[float, float, float]] = {}
    for model_id in sorted(base_colors):
        labels = sorted(
            {r["target_label"] for r in rows if r["model_id"] == model_id},
            key=lambda label: min(float(r["target_flops_6nd"]) for r in rows if r["model_id"] == model_id and r["target_label"] == label),
        )
        n = max(len(labels), 1)
        for i, label in enumerate(labels):
            amount = 0.55 - 0.42 * (i / (n - 1)) if n > 1 else 0.15
            colors[(model_id, label)] = blend_with_white(base_colors[model_id], amount)
    return colors


def plot_domain_trajectories(domain_rows: list[dict[str, Any]], output_dir: Path, alpha: float) -> dict[str, str]:
    if not domain_rows:
        return {}
    domains = sorted_domains({r["domain"] for r in domain_rows})
    model_ids = sorted({r["model_id"] for r in domain_rows}, key=lambda m: min(int(r["parameter_count"]) for r in domain_rows if r["model_id"] == m))
    base_colors = model_colors(domain_rows)
    run_colors = label_colors(domain_rows, base_colors)
    fig, axes = plt.subplots(
        len(model_ids),
        len(domains),
        figsize=(3.35 * len(domains), 2.7 * len(model_ids)),
        dpi=160,
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in domain_rows:
        grouped[(row["model_id"], row["target_label"], row["domain"])].append(row)
    for i, model_id in enumerate(model_ids):
        model_label = next(r["model_label"] for r in domain_rows if r["model_id"] == model_id)
        labels = sorted(
            {r["target_label"] for r in domain_rows if r["model_id"] == model_id},
            key=lambda label: min(float(r["target_flops_6nd"]) for r in domain_rows if r["model_id"] == model_id and r["target_label"] == label),
        )
        for j, domain in enumerate(domains):
            ax = axes[i][j]
            for label in labels:
                sub = grouped.get((model_id, label, domain), [])
                if not sub:
                    continue
                sub.sort(key=lambda r: int(r["step"]))
                smooth = ema([float(r["domain_loss"]) for r in sub], alpha)
                ax.plot(
                    [float(r["flops_6nd_at_step"]) for r in sub],
                    smooth,
                    linewidth=1.3,
                    color=run_colors[(model_id, label)],
                    alpha=0.95,
                    label=str(label),
                )
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.22)
            if i == 0:
                ax.set_title(domain)
            if j == 0:
                ax.set_ylabel(f"{model_label}\nDomain loss")
            if i == len(model_ids) - 1:
                ax.set_xlabel("FLOPs (6ND)")
            if j == len(domains) - 1:
                ax.legend(title="Target", fontsize=7, title_fontsize=8)
    fig.suptitle("Per-domain training loss vs FLOPs", y=1.02)
    fig.tight_layout()
    png = output_dir / "domain_loss_vs_flops_by_domain.png"
    pdf = output_dir / "domain_loss_vs_flops_by_domain.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "domain_loss_vs_flops_by_domain_png": str(png),
        "domain_loss_vs_flops_by_domain_pdf": str(pdf),
    }


def plot_heatmap(
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    value_key: str,
    title: str,
    colorbar_label: str,
    filename: str,
    value_format: str,
    scale: float = 1.0,
) -> dict[str, str]:
    if not summary_rows:
        return {}
    domains = sorted_domains({r["domain"] for r in summary_rows})
    run_keys = sorted(
        {(r["model_id"], r["target_label"]) for r in summary_rows},
        key=lambda key: (
            min(int(r["parameter_count"]) for r in summary_rows if r["model_id"] == key[0]),
            min(float(r["target_flops_6nd"]) for r in summary_rows if r["model_id"] == key[0] and r["target_label"] == key[1]),
        ),
    )
    labels = []
    matrix: list[list[float]] = []
    for model_id, target_label in run_keys:
        sample = next(r for r in summary_rows if r["model_id"] == model_id and r["target_label"] == target_label)
        labels.append(f"{sample['model_label']} {target_label}")
        row_values = []
        for domain in domains:
            match = [r for r in summary_rows if r["model_id"] == model_id and r["target_label"] == target_label and r["domain"] == domain]
            row_values.append(float(match[0][value_key]) * scale if match else math.nan)
        matrix.append(row_values)
    fig_h = max(4.8, 0.34 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(1.55 * len(domains) + 3.8, fig_h), dpi=160)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for y, row_values in enumerate(matrix):
        for x, value in enumerate(row_values):
            if math.isnan(value):
                text = ""
            else:
                text = value_format.format(value)
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="white" if value > (max(max(v for v in row if not math.isnan(v)) for row in matrix) * 0.55) else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    png = output_dir / f"{filename}.png"
    pdf = output_dir / f"{filename}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return {f"{filename}_png": str(png), f"{filename}_pdf": str(pdf)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--matrix-json", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--models", default="", help="Comma-separated model ids to plot.")
    ap.add_argument("--target-labels", default="", help="Comma-separated target labels to plot.")
    ap.add_argument("--require-domain", action="store_true")
    ap.add_argument("--ema-alpha", type=float, default=0.08)
    ap.add_argument("--tail-events", type=int, default=5)
    args = ap.parse_args()

    exp_root = args.run_root / args.experiment_name
    matrix_path = args.matrix_json or (exp_root / "scaling_matrix.json")
    output_dir = args.output_dir or (exp_root / "results" / "figures_domain_loss")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filter = {x.strip() for x in args.models.split(",") if x.strip()} or None
    target_filter = {x.strip() for x in args.target_labels.split(",") if x.strip()} or None

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    domain_rows, incomplete, missing_domain = build_rows(matrix["rows"], model_filter, target_filter)
    if args.require_domain and missing_domain:
        raise SystemExit("missing domain logging: " + ", ".join(missing_domain))
    summary_rows = tail_summary(domain_rows, args.tail_events)
    write_csv(output_dir / "domain_loss_events.csv", domain_rows)
    write_csv(output_dir / "domain_loss_tail_summary.csv", summary_rows)

    figures: dict[str, str] = {}
    figures.update(plot_domain_trajectories(domain_rows, output_dir, args.ema_alpha))
    figures.update(
        plot_heatmap(
            summary_rows,
            output_dir,
            "tail_domain_loss",
            f"Tail per-domain loss (last {args.tail_events} logged events)",
            "Domain training loss",
            "domain_tail_loss_heatmap",
            "{:.2f}",
        )
    )
    figures.update(
        plot_heatmap(
            summary_rows,
            output_dir,
            "tail_domain_loss_token_fraction",
            f"Tail domain loss-token fraction (last {args.tail_events} logged events)",
            "Loss-token fraction (%)",
            "domain_loss_token_fraction_heatmap",
            "{:.1f}",
            scale=100.0,
        )
    )

    manifest = {
        "matrix_json": str(matrix_path),
        "output_dir": str(output_dir),
        "models": sorted(model_filter) if model_filter else "all",
        "target_labels": sorted(target_filter) if target_filter else "all",
        "require_domain": bool(args.require_domain),
        "ema_alpha": args.ema_alpha,
        "tail_events": args.tail_events,
        "incomplete": incomplete,
        "missing_domain": missing_domain,
        "domain_event_rows": len(domain_rows),
        "domain_tail_summary_rows": len(summary_rows),
        "tables": {
            "domain_loss_events": str(output_dir / "domain_loss_events.csv"),
            "domain_loss_tail_summary": str(output_dir / "domain_loss_tail_summary.csv"),
        },
        "figures": figures,
        "color_scheme": "model_rows_target_shades",
    }
    (output_dir / "domain_plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
