#!/usr/bin/env python3
"""Build a fixed-total-FLOPs scaling matrix for FDDFS Qwen3 experiments.

Each row is one independent training run at the same target 6ND FLOPs budget.
This gives a cross-section for plotting final training loss against model size.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

try:
    from .scaling_paths import (
        DEFAULT_DATA_ROOT,
        DEFAULT_RUN_ROOT,
        DEFAULT_STATS,
        safe_path_component,
    )
except ImportError:  # direct script execution
    from scaling_paths import (
        DEFAULT_DATA_ROOT,
        DEFAULT_RUN_ROOT,
        DEFAULT_STATS,
        safe_path_component,
    )

DEFAULT_EXPERIMENT_NAME = "scaling_fddfs_mf80000_fixed_flops_domainlog_v1"
DEFAULT_TARGET_FLOPS_6ND = 3.6e18


def load_epoch_builder() -> Any:
    path = Path(__file__).with_name("build_epoch_fraction_scaling_matrix.py")
    spec = importlib.util.spec_from_file_location("epoch_fraction_scaling_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("empty scaling matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def default_label(target_flops: float) -> str:
    if target_flops <= 0:
        raise ValueError("target FLOPs must be positive")
    exponent = int(math.floor(math.log10(target_flops)))
    mantissa = target_flops / (10**exponent)
    mantissa_text = f"{mantissa:.3g}".replace(".", "p").replace("+", "")
    return f"fx{mantissa_text}e{exponent}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats-json", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--vocab-size", type=int, default=17207)
    ap.add_argument("--token-offset", type=int, default=3)
    ap.add_argument("--online-pad-length", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-flops-6nd", type=float, default=DEFAULT_TARGET_FLOPS_6ND)
    ap.add_argument("--target-label", default=None)
    ap.add_argument("--allow-over-one-epoch", action="store_true")
    args = ap.parse_args()

    args.stats_json = args.stats_json.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    args.experiment_name = safe_path_component(
        args.experiment_name, "--experiment-name"
    )
    if args.out_json is not None:
        args.out_json = args.out_json.expanduser().resolve()
    if args.out_csv is not None:
        args.out_csv = args.out_csv.expanduser().resolve()

    if args.target_flops_6nd <= 0:
        raise ValueError("--target-flops-6nd must be positive")

    epoch_builder = load_epoch_builder()
    stats = json.loads(args.stats_json.read_text(encoding="utf-8"))
    records = int(stats["records"])
    tokens_with_eos = int(stats["tokens_with_eos"])
    mean_tokens_with_eos = float(tokens_with_eos / records)
    train_records, val_records = epoch_builder.train_record_count(records, args.val_fraction)
    train_tokens_per_epoch = mean_tokens_with_eos * train_records
    label = safe_path_component(
        args.target_label or default_label(args.target_flops_6nd), "--target-label"
    )

    out_root = args.run_root / args.experiment_name
    matrix_json = args.out_json or (out_root / "scaling_matrix.json")
    matrix_csv = args.out_csv or (out_root / "scaling_matrix.csv")

    rows: list[dict[str, Any]] = []
    for spec in epoch_builder.MODEL_SPECS:
        params = epoch_builder.qwen3_packed_param_count(spec, args.vocab_size)
        samples_per_step = (
            int(spec["gpu_count"])
            * int(spec["per_device_train_batch_size"])
            * int(spec["gradient_accumulation_steps"])
        )
        if samples_per_step <= 0:
            raise ValueError(f"invalid samples_per_step for {spec['model_id']}")
        tokens_per_step = mean_tokens_with_eos * samples_per_step
        one_epoch_steps = int(math.floor(train_records / samples_per_step))
        requested_tokens = float(args.target_flops_6nd / (6.0 * max(params, 1)))
        requested_steps = max(1, int(round(requested_tokens / tokens_per_step)))
        max_steps = requested_steps
        clipped_to_one_epoch = False
        if not args.allow_over_one_epoch and max_steps > one_epoch_steps:
            max_steps = one_epoch_steps
            clipped_to_one_epoch = True

        actual_samples = max_steps * samples_per_step
        actual_tokens = float(max_steps * tokens_per_step)
        actual_flops = float(6.0 * params * actual_tokens)
        warmup_steps = max(1, int(round(max_steps * float(args.warmup_ratio))))
        run_name = f"{args.experiment_name}_{spec['model_id']}_{label}_g{spec['gpu_count']}"
        out_dir = out_root / spec["model_id"] / label
        rows.append(
            {
                "experiment_name": args.experiment_name,
                "model_id": spec["model_id"],
                "model_label": spec["model_label"],
                "flops_target_label": label,
                "epoch_fraction_target": float(requested_tokens / train_tokens_per_epoch),
                "epoch_fraction": float(actual_samples / train_records),
                "ratio": float(actual_tokens / max(params, 1)),
                "parameter_count": int(params),
                "target_tokens": int(round(requested_tokens)),
                "actual_tokens": int(round(actual_tokens)),
                "actual_samples": int(actual_samples),
                "actual_token_budget_b": float(actual_tokens / 1e9),
                "flops_6nd": actual_flops,
                "target_flops_6nd": float(args.target_flops_6nd),
                "flops_relative_error": float((actual_flops - args.target_flops_6nd) / args.target_flops_6nd),
                "tokens_per_step": float(tokens_per_step),
                "samples_per_step": int(samples_per_step),
                "requested_steps": int(requested_steps),
                "max_steps": int(max_steps),
                "one_epoch_steps": int(one_epoch_steps),
                "clipped_to_one_epoch": bool(clipped_to_one_epoch),
                "warmup_steps": int(warmup_steps),
                "warmup_ratio": float(args.warmup_ratio),
                "seed": int(args.seed),
                "vocab_size": int(args.vocab_size),
                "token_offset": int(args.token_offset),
                "online_pad_length": int(args.online_pad_length),
                "val_fraction": float(args.val_fraction),
                "train_records": int(train_records),
                "val_records": int(val_records),
                "train_tokens_per_epoch": int(round(train_tokens_per_epoch)),
                "mean_tokens_with_eos": float(mean_tokens_with_eos),
                "gpu_count": int(spec["gpu_count"]),
                "cpu_count": int(spec["cpu_count"]),
                "memory_mib": int(spec["memory_mib"]),
                "per_device_train_batch_size": int(spec["per_device_train_batch_size"]),
                "gradient_accumulation_steps": int(spec["gradient_accumulation_steps"]),
                "hidden_size": int(spec["hidden_size"]),
                "intermediate_size": int(spec["intermediate_size"]),
                "num_hidden_layers": int(spec["num_hidden_layers"]),
                "num_attention_heads": int(spec["num_attention_heads"]),
                "num_key_value_heads": int(spec["num_key_value_heads"]),
                "head_dim": int(spec["head_dim"]),
                "model_source": str(spec.get("source", "")),
                "run_name": run_name,
                "out_dir": str(out_dir),
            }
        )

    metadata = {
        "stats_json": str(args.stats_json),
        "records": records,
        "tokens_with_eos": tokens_with_eos,
        "train_records": int(train_records),
        "val_records": int(val_records),
        "train_tokens_per_epoch": int(round(train_tokens_per_epoch)),
        "mean_tokens_with_eos": float(mean_tokens_with_eos),
        "experiment_name": args.experiment_name,
        "run_root": str(out_root),
        "target_flops_6nd": float(args.target_flops_6nd),
        "target_label": label,
        "allow_over_one_epoch": bool(args.allow_over_one_epoch),
        "note": "Independent fixed-FLOPs cross-section; one row per model size. Default target keeps 0p06b below one train-sample epoch.",
        "domain_loss_logging_required": True,
    }
    matrix_json.parent.mkdir(parents=True, exist_ok=True)
    matrix_json.write_text(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(matrix_csv, rows)
    print(json.dumps({"matrix_json": str(matrix_json), "matrix_csv": str(matrix_csv), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
