#!/usr/bin/env python3
"""Build one-epoch-capped scaling matrix for FDDFS Qwen3 experiments.

Each point is an independent training run.  Target labels e020..e100 mean
0.2..1.0 train-sample epoch.  The largest point is capped at one train epoch.
"""

from __future__ import annotations

import argparse
import csv
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


MODEL_SPECS: list[dict[str, Any]] = [
    {
        "model_id": "0p06b",
        "model_label": "0.06B",
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "gpu_count": 1,
        "cpu_count": 16,
        "memory_mib": 128000,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
        "source": "reduced Qwen3-style model used in completed 0.06B runs",
    },
    {
        "model_id": "0p30b",
        "model_label": "0.30B",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "gpu_count": 4,
        "cpu_count": 56,
        "memory_mib": 448000,
        "per_device_train_batch_size": 384,
        "gradient_accumulation_steps": 4,
        "source": "reduced Qwen3-style 0.30B model from prior experiments",
    },
    {
        "model_id": "0p60b",
        "model_label": "0.60B",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "gpu_count": 4,
        "cpu_count": 56,
        "memory_mib": 448000,
        "per_device_train_batch_size": 256,
        "gradient_accumulation_steps": 6,
        "source": "Qwen/Qwen3-0.6B config shape; vocab replaced by project vocab",
    },
    {
        "model_id": "1p70b",
        "model_label": "1.70B",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "gpu_count": 8,
        "cpu_count": 96,
        "memory_mib": 768000,
        "per_device_train_batch_size": 128,
        "gradient_accumulation_steps": 12,
        "source": "Qwen/Qwen3-1.7B config shape; vocab replaced by project vocab",
    },
    {
        "model_id": "4p00b",
        "model_label": "4.00B",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "gpu_count": 8,
        "cpu_count": 112,
        "memory_mib": 896000,
        "per_device_train_batch_size": 64,
        "gradient_accumulation_steps": 24,
        "source": "Qwen/Qwen3-4B config shape; vocab replaced by project vocab",
    },
]


def qwen3_packed_param_count(spec: dict[str, Any], vocab_size: int) -> int:
    hidden = int(spec["hidden_size"])
    intermediate = int(spec["intermediate_size"])
    layers = int(spec["num_hidden_layers"])
    heads = int(spec["num_attention_heads"])
    kv_heads = int(spec["num_key_value_heads"])
    head_dim = int(spec["head_dim"])

    embed = int(vocab_size) * hidden
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    attention = hidden * q_dim + hidden * kv_dim + hidden * kv_dim + q_dim * hidden
    mlp = hidden * intermediate * 3
    layer_norms = hidden * 2
    final_norm = hidden
    return int(embed + layers * (attention + mlp + layer_norms) + final_norm)


def train_record_count(records: int, val_fraction: float) -> tuple[int, int]:
    if val_fraction <= 0:
        return int(records), 0
    stride = max(2, int(round(1.0 / val_fraction)))
    val = (int(records) + stride - 1) // stride
    return int(records) - val, val


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("empty scaling matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_fractions(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("empty epoch fractions")
    if any(x <= 0.0 or x > 1.0 for x in values):
        raise ValueError("epoch fractions must be in (0, 1]")
    return values


def label_for_fraction(fraction: float) -> str:
    return f"e{int(round(fraction * 100)):03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats-json", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", default="scaling_fddfs_mf80000_epoch_fraction_domainlog_v1")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--vocab-size", type=int, default=17207)
    ap.add_argument("--token-offset", type=int, default=3)
    ap.add_argument("--online-pad-length", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch-fractions", default="0.2,0.4,0.6,0.8,1.0")
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

    stats = json.loads(args.stats_json.read_text(encoding="utf-8"))
    records = int(stats["records"])
    tokens_with_eos = int(stats["tokens_with_eos"])
    mean_tokens_with_eos = float(tokens_with_eos / records)
    train_records, val_records = train_record_count(records, args.val_fraction)
    train_tokens_per_epoch = mean_tokens_with_eos * train_records
    fractions = parse_fractions(args.epoch_fractions)

    out_root = args.run_root / args.experiment_name
    matrix_json = args.out_json or (out_root / "scaling_matrix.json")
    matrix_csv = args.out_csv or (out_root / "scaling_matrix.csv")

    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        params = qwen3_packed_param_count(spec, args.vocab_size)
        samples_per_step = (
            int(spec["gpu_count"])
            * int(spec["per_device_train_batch_size"])
            * int(spec["gradient_accumulation_steps"])
        )
        if samples_per_step <= 0:
            raise ValueError(f"invalid samples_per_step for {spec['model_id']}")
        tokens_per_step = mean_tokens_with_eos * samples_per_step
        one_epoch_steps = int(math.floor(train_records / samples_per_step))
        if one_epoch_steps <= 0:
            raise ValueError(f"one_epoch_steps <= 0 for {spec['model_id']}")
        for fraction in fractions:
            label = label_for_fraction(fraction)
            max_steps = max(1, int(round(one_epoch_steps * fraction)))
            if fraction >= 1.0:
                max_steps = one_epoch_steps
            actual_samples = max_steps * samples_per_step
            actual_tokens = float(max_steps * tokens_per_step)
            warmup_steps = max(1, int(round(max_steps * float(args.warmup_ratio))))
            target_tokens = float(train_tokens_per_epoch * fraction)
            run_name = f"{args.experiment_name}_{spec['model_id']}_{label}_g{spec['gpu_count']}"
            out_dir = out_root / spec["model_id"] / label
            rows.append(
                {
                    "experiment_name": args.experiment_name,
                    "model_id": spec["model_id"],
                    "model_label": spec["model_label"],
                    "flops_target_label": label,
                    "epoch_fraction_target": float(fraction),
                    "epoch_fraction": float(actual_samples / train_records),
                    "ratio": float(actual_tokens / max(params, 1)),
                    "parameter_count": params,
                    "target_tokens": int(round(target_tokens)),
                    "actual_tokens": int(round(actual_tokens)),
                    "actual_samples": int(actual_samples),
                    "actual_token_budget_b": actual_tokens / 1e9,
                    "flops_6nd": float(6.0 * params * actual_tokens),
                    "target_flops_6nd": float(6.0 * params * target_tokens),
                    "tokens_per_step": float(tokens_per_step),
                    "samples_per_step": int(samples_per_step),
                    "max_steps": int(max_steps),
                    "one_epoch_steps": int(one_epoch_steps),
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
        "train_records": train_records,
        "val_records": val_records,
        "train_tokens_per_epoch": int(round(train_tokens_per_epoch)),
        "mean_tokens_with_eos": mean_tokens_with_eos,
        "experiment_name": args.experiment_name,
        "run_root": str(out_root),
        "epoch_fractions": fractions,
        "target_labels": [label_for_fraction(x) for x in fractions],
        "note": "Independent run per point; largest point is one train-sample epoch. max_steps is floor(train_records / samples_per_step) at e100.",
        "qwen3_config_source": "Official Qwen3 config shapes read from Hugging Face Qwen/Qwen3-{0.6B,1.7B,4B}; project vocab_size is used for training.",
        "domain_loss_logging_required": True,
    }
    matrix_json.parent.mkdir(parents=True, exist_ok=True)
    matrix_json.write_text(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(matrix_csv, rows)
    print(json.dumps({"matrix_json": str(matrix_json), "matrix_csv": str(matrix_csv), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
