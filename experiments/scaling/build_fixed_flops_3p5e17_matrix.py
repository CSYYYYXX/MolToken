#!/usr/bin/env python3
"""Build the prioritized 3.5e17-FLOPs sweep over extra-small Qwen3 shapes."""

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


builder.DEFAULT_EXPERIMENT_NAME = "scaling_fddfs_mf80000_fixed_flops3p5e17_smallmodels_domainlog_v1"
builder.FLOPS_TARGETS = [("fx3p5e17", 3.5e17)]
builder.MODEL_SPECS = [
    extra_small_spec(2, "0p006b", "0.006B"),
    extra_small_spec(4, "0p008b", "0.008B"),
    extra_small_spec(6, "0p009b", "0.009B"),
    extra_small_spec(8, "0p011b", "0.011B"),
    extra_small_spec(12, "0p014b", "0.014B"),
    extra_small_spec(14, "0p015b", "0.015B"),
    {
        "model_id": "0p022b",
        "model_label": "0.022B",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 9,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    },
    {
        "model_id": "0p031b",
        "model_label": "0.031B",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 14,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    },
    {
        "model_id": "0p026b",
        "model_label": "0.026B",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 11,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    },
    *builder.MODEL_SPECS,
]


if __name__ == "__main__":
    builder.main()
