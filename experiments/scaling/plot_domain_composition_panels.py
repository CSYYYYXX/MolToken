#!/usr/bin/env python3
"""Plot per-step domain composition as 10-panel figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DOMAIN_ORDER = ("small_molecule", "protein", "peptide", "dna", "rna")
DOMAIN_COLORS = {
    "small_molecule": "#1f77b4",
    "protein": "#d62728",
    "peptide": "#9467bd",
    "dna": "#2ca02c",
    "rna": "#ff7f0e",
}
FRACTION_FIELDS = {
    "input_token": ("domain_input_token_fraction", "Input-token fraction (%)"),
    "loss_token": ("domain_loss_token_fraction", "Loss-token fraction (%)"),
    "sample": ("domain_sample_fraction", "Sample fraction (%)"),
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def ordered_domains(rows: list[dict[str, Any]]) -> list[str]:
    domains = {r["domain"] for r in rows}
    known = [d for d in DOMAIN_ORDER if d in domains]
    return known + sorted(domains - set(known))


def run_order(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return sorted(
        {(r["model_id"], r["target_label"]) for r in rows},
        key=lambda key: (
            min(int(float(r["parameter_count"])) for r in rows if r["model_id"] == key[0]),
            min(float(r["target_flops_6nd"]) for r in rows if r["model_id"] == key[0] and r["target_label"] == key[1]),
        ),
    )


def plot_fraction(
    rows: list[dict[str, Any]],
    output_dir: Path,
    fraction_kind: str,
    mode: str,
) -> dict[str, str]:
    field, ylabel = FRACTION_FIELDS[fraction_kind]
    domains = ordered_domains(rows)
    runs = run_order(rows)
    model_ids = []
    for model_id, _target_label in runs:
        if model_id not in model_ids:
            model_ids.append(model_id)
    target_labels = []
    for _model_id, target_label in runs:
        if target_label not in target_labels:
            target_labels.append(target_label)

    sharey = mode == "full"
    fig, axes = plt.subplots(
        len(model_ids),
        len(target_labels),
        figsize=(3.35 * len(target_labels), 2.75 * len(model_ids)),
        dpi=160,
        squeeze=False,
        sharey=sharey,
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model_id"], row["target_label"], row["domain"]), []).append(row)

    legend_handles = []
    legend_labels = []
    for i, model_id in enumerate(model_ids):
        for j, target_label in enumerate(target_labels):
            ax = axes[i][j]
            run_rows = [r for r in rows if r["model_id"] == model_id and r["target_label"] == target_label]
            if not run_rows:
                ax.axis("off")
                continue
            model_label = run_rows[0]["model_label"]
            for domain in domains:
                sub = grouped.get((model_id, target_label, domain), [])
                if not sub:
                    continue
                sub.sort(key=lambda r: int(r["step"]))
                values = [100.0 * float(r[field]) for r in sub]
                if mode == "deviation":
                    mean_value = sum(values) / max(len(values), 1)
                    values = [v - mean_value for v in values]
                line = ax.plot(
                    [int(r["step"]) for r in sub],
                    values,
                    color=DOMAIN_COLORS.get(domain),
                    linewidth=1.45,
                    alpha=0.92,
                    label=domain,
                )[0]
                if domain not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(domain)
            if mode == "full":
                ax.set_ylim(0, 100)
            elif mode == "deviation":
                ax.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.6)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{model_label} {target_label}", fontsize=10)
            if i == len(model_ids) - 1:
                ax.set_xlabel("Step")
            if j == 0:
                if mode == "deviation":
                    ax.set_ylabel("Deviation from run mean (pp)")
                else:
                    ax.set_ylabel(ylabel)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(domains), bbox_to_anchor=(0.5, 1.02))
    title_suffix = {
        "full": "0-100% scale",
        "zoom": "auto y-scale",
        "deviation": "deviation from run mean",
    }[mode]
    fig.suptitle(f"Per-logged-step domain composition: {fraction_kind.replace('_', ' ')} ({title_suffix})", y=1.08)
    fig.tight_layout()
    png = output_dir / f"domain_{fraction_kind}_fraction_by_step_10panels_{mode}.png"
    pdf = output_dir / f"domain_{fraction_kind}_fraction_by_step_10panels_{mode}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {f"{fraction_kind}_{mode}_png": str(png), f"{fraction_kind}_{mode}_pdf": str(pdf)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain-events-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--fraction-kinds",
        default="input_token,loss_token,sample",
        help="Comma-separated fraction kinds: input_token, loss_token, sample.",
    )
    ap.add_argument(
        "--modes",
        default="full,zoom,deviation",
        help="Comma-separated modes: full, zoom, deviation.",
    )
    args = ap.parse_args()

    rows = read_rows(args.domain_events_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinds = [k.strip() for k in args.fraction_kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in FRACTION_FIELDS]
    if unknown:
        raise SystemExit(f"unknown fraction kinds: {unknown}")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown_modes = [m for m in modes if m not in {"full", "zoom", "deviation"}]
    if unknown_modes:
        raise SystemExit(f"unknown modes: {unknown_modes}")

    figures: dict[str, str] = {}
    for kind in kinds:
        for mode in modes:
            figures.update(plot_fraction(rows, args.output_dir, kind, mode))
    manifest = {
        "domain_events_csv": str(args.domain_events_csv),
        "output_dir": str(args.output_dir),
        "domain_event_rows": len(rows),
        "runs": [f"{m}/{t}" for m, t in run_order(rows)],
        "domains": ordered_domains(rows),
        "fraction_kinds": kinds,
        "modes": modes,
        "note": "Each point is one logged training event. With LOG_EVERY=100, most points aggregate roughly the previous 100 optimizer steps, not a single micro-batch.",
        "figures": figures,
    }
    (args.output_dir / "domain_composition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
