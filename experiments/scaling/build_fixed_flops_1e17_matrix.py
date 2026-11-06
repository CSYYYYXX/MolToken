#!/usr/bin/env python3
"""Build the A1.5b pre-registered out-of-sample sweep at C = 1e17 FLOPs.

Three bracket points around the predicted compute-optimal minimum
(N* = 10.86M, from the parabola-vertex extrapolation frozen under
doc/paper_materials_20260728/preregistration_A1p5b/):

    0p008b  L=4   N=7.55M   below the vertex
    0p011b  L=8   N=10.70M  ~ at the vertex
    0p017b  L=16  N=17.00M  above the vertex

All three reuse existing extra-small Qwen3 shapes (hidden=256, inter=768,
heads=2, kv=1, head_dim=128); only depth varies. Every run stays well below
one epoch (0.10-0.22), so the D1 corpus ceiling is untouched.
"""

from __future__ import annotations

import build_fixed_flops_sweep_matrix as builder


def extra_small_spec(layers: int, model_id: str, model_label: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_label": model_label,
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    }


builder.DEFAULT_EXPERIMENT_NAME = "scaling_fddfs_mf80000_fixed_flops1e17_a1p5b_domainlog_v1"
builder.FLOPS_TARGETS = [("fx1e17", 1.0e17)]
builder.MODEL_SPECS = [
    extra_small_spec(4, "0p008b", "0.008B"),
    extra_small_spec(8, "0p011b", "0.011B"),
    extra_small_spec(16, "0p017b", "0.017B"),
]


if __name__ == "__main__":
    builder.main()
