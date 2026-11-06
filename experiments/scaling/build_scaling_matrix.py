#!/usr/bin/env python3
"""生成 FDDFS 预训练 scaling-law 实验矩阵。

这个脚本只读取小型 JSON 统计文件，不读取训练样本。矩阵中的
`target_tokens = parameter_count * ratio`，`max_steps` 按有效 model token
计算；当前 online-pad + packed-xformers 训练不会把 PAD 送入主计算。
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
    },
    {
        "model_id": "0p10b",
        "model_label": "0.10B",
        "hidden_size": 768,
        "intermediate_size": 2304,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 128,
        "gpu_count": 2,
        "cpu_count": 32,
        "memory_mib": 256000,
        "per_device_train_batch_size": 512,
        "gradient_accumulation_steps": 6,
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
        "gpu_count": 3,
        "cpu_count": 48,
        "memory_mib": 384000,
        "per_device_train_batch_size": 512,
        "gradient_accumulation_steps": 4,
    },
]


def qwen3_packed_param_count(spec: dict[str, Any], vocab_size: int) -> int:
    """按当前 PackedXformersQwen3ForCausalLM 的模块形状计算参数量。"""

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
    """复现训练脚本的 stride split 计数逻辑。"""

    if val_fraction <= 0:
        return int(records), 0
    stride = max(2, int(round(1.0 / val_fraction)))
    val = (int(records) + stride - 1) // stride
    return int(records) - val, val


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """把矩阵写成 CSV，方便后续检查和人工筛选。"""

    if not rows:
        raise ValueError("empty scaling matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats-json", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", default="scaling_fddfs_mf80000_v1")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--vocab-size", type=int, default=17207)
    ap.add_argument("--token-offset", type=int, default=3)
    ap.add_argument("--online-pad-length", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs="+", default=[4, 8, 12, 16, 20, 24, 30])
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
        tokens_per_step = mean_tokens_with_eos * samples_per_step
        for ratio in args.ratios:
            ratio_label = str(ratio).replace(".", "p")
            if float(ratio).is_integer():
                ratio_label = str(int(ratio))
            target_tokens = params * float(ratio)
            max_steps = int(math.ceil(target_tokens / tokens_per_step))
            actual_tokens = float(max_steps * tokens_per_step)
            warmup_steps = max(1, int(round(max_steps * float(args.warmup_ratio))))
            run_name = f"{args.experiment_name}_{spec['model_id']}_r{ratio_label}_g{spec['gpu_count']}"
            out_dir = out_root / spec["model_id"] / f"r{ratio_label}"
            rows.append(
                {
                    "experiment_name": args.experiment_name,
                    "model_id": spec["model_id"],
                    "model_label": spec["model_label"],
                    "ratio": float(ratio),
                    "parameter_count": params,
                    "target_tokens": int(round(target_tokens)),
                    "actual_tokens": int(round(actual_tokens)),
                    "actual_token_budget_b": actual_tokens / 1e9,
                    "epoch_fraction": actual_tokens / train_tokens_per_epoch,
                    "flops_6nd": float(6.0 * params * actual_tokens),
                    "tokens_per_step": float(tokens_per_step),
                    "samples_per_step": int(samples_per_step),
                    "max_steps": int(max_steps),
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
                    "run_name": run_name,
                    "out_dir": str(out_dir),
                }
            )

    matrix_json.parent.mkdir(parents=True, exist_ok=True)
    matrix_json.write_text(json.dumps({"metadata": {
        "stats_json": str(args.stats_json),
        "records": records,
        "tokens_with_eos": tokens_with_eos,
        "train_records": train_records,
        "val_records": val_records,
        "train_tokens_per_epoch": int(round(train_tokens_per_epoch)),
        "mean_tokens_with_eos": mean_tokens_with_eos,
        "note": "max_steps uses effective model tokens; PAD is excluded by packed-xformers.",
    }, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(matrix_csv, rows)
    print(json.dumps({"matrix_json": str(matrix_json), "matrix_csv": str(matrix_csv), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
