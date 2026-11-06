#!/usr/bin/env python3
"""Validate the non-embedding parameter formula against reported parameter_count.

The document quotes embedding fractions (73.7% at 0p006b, 25.9% at 0p017b,
13.5% at 0p098b, 1.2% at 4p00b). Reproduce those before trusting the C1 refit.
"""

import csv
import json
import math
from pathlib import Path

CSV_PATH = Path(
    "./storage/molscaling"
    "/runs/scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1"
    "/plots/all_fixed_flops_10curves/all_fixed_flops_results.csv"
)


def qwen3_non_embedding(cfg):
    """Non-embedding params for the Qwen3 variant used here (tied lm_head)."""
    # Config is nested under 'args'
    args = cfg.get("args", cfg)
    hidden = args["hidden_size"]
    inter = args["intermediate_size"]
    layers = args["num_hidden_layers"]
    heads = args["num_attention_heads"]
    kv_heads = args["num_key_value_heads"]
    head_dim = args["head_dim"]

    q = hidden * heads * head_dim
    k = hidden * kv_heads * head_dim
    v = hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    attn = q + k + v + o

    # Qwen3 q_norm / k_norm are per-head-dim RMSNorm weights.
    qk_norm = 2 * head_dim

    mlp = 3 * hidden * inter

    # RMSNorm: weight only, no bias.
    layer_norms = 2 * hidden

    per_layer = attn + qk_norm + mlp + layer_norms
    return layers * per_layer + hidden  # + final norm


def main():
    seen = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            model_id = row["model_id"]
            if model_id in seen:
                continue
            cfg_path = Path(row["out_dir"]) / "run_config.json"
            if not cfg_path.is_file():
                print(f"MISSING run_config: {cfg_path}")
                continue
            cfg = json.loads(cfg_path.read_text())
            seen[model_id] = (int(row["parameter_count"]), cfg)

    print(f"{'model':>9} {'N_csv':>12} {'N_cfg':>12} {'nonemb':>12} "
          f"{'emb':>12} {'emb%':>7} {'resid':>10}")
    print("-" * 82)
    worst = 0.0
    for model_id in sorted(seen, key=lambda m: seen[m][0]):
        n_csv, cfg = seen[model_id]
        n_cfg = int(cfg["parameter_count"])
        nonemb = qwen3_non_embedding(cfg)
        args = cfg.get("args", cfg)
        emb = args["vocab_size"] * args["hidden_size"]
        resid = n_cfg - (nonemb + emb)
        worst = max(worst, abs(resid) / n_cfg)
        print(f"{model_id:>9} {n_csv:>12,} {n_cfg:>12,} {nonemb:>12,} "
              f"{emb:>12,} {100*emb/n_cfg:>6.1f}% {resid:>10,}")

    print("-" * 82)
    print(f"worst relative residual: {worst:.3e}")
    first_cfg = next(iter(seen.values()))[1]
    first_args = first_cfg.get("args", first_cfg)
    print(f"vocab_size = {first_args['vocab_size']}")


if __name__ == "__main__":
    main()
