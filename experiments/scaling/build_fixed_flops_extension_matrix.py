#!/usr/bin/env python3
"""Build the sparse fixed-FLOPs extension needed for ten complete curves."""

from __future__ import annotations

import build_fixed_flops_sweep_matrix as builder


def model_spec(
    model_id: str,
    model_label: str,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
    attention_heads: int,
    key_value_heads: int,
    batch_size: int = 1024,
    grad_accum: int = 6,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_label": model_label,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": key_value_heads,
        "head_dim": 128,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
    }


builder.DEFAULT_EXPERIMENT_NAME = "scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1"
builder.FLOPS_TARGETS = [
    ("fx5e17", 5.0e17),
    ("fx7e17", 7.0e17),
    ("fx1p3e18", 1.3e18),
    ("fx1p6e18", 1.6e18),
    ("fx2e18", 2.0e18),
    ("fx2p4e18", 2.4e18),
    ("fx2p9e18", 2.9e18),
    ("fx3p5e18", 3.5e18),
]

builder.MODEL_SPECS = [
    model_spec("0p009b", "0.009B", 256, 768, 6, 2, 1),
    model_spec("0p014b", "0.014B", 256, 768, 12, 2, 1),
    model_spec("0p022b", "0.022B", 384, 1152, 9, 3, 1),
    model_spec("0p027b", "0.027B", 384, 1152, 12, 3, 1),
    model_spec("0p031b", "0.031B", 384, 1152, 14, 3, 1),
    model_spec("0p041b", "0.041B", 384, 1152, 20, 3, 1),
    model_spec("0p047b", "0.047B", 512, 1536, 12, 4, 2),
    model_spec("0p053b", "0.053B", 512, 1536, 14, 4, 2),
    model_spec("0p059b", "0.059B", 512, 1536, 16, 4, 2),
    model_spec("0p065b", "0.065B", 512, 1536, 18, 4, 2),
    model_spec("0p071b", "0.071B", 512, 1536, 20, 4, 2),
    model_spec("0p078b", "0.078B", 512, 1536, 22, 4, 2),
    model_spec("0p098b", "0.098B", 768, 2304, 12, 6, 3, batch_size=512, grad_accum=12),
]

# Five points bracket the predicted minimum of every new curve.  The two
# existing sparse curves receive only the missing points around their minima.
builder.SWEEP_PLAN = {
    ("fx5e17", model_id) for model_id in ("0p009b", "0p014b", "0p022b", "0p031b", "0p041b")
} | {
    ("fx7e17", model_id) for model_id in ("0p014b", "0p022b", "0p027b", "0p041b", "0p059b")
} | {
    ("fx1p3e18", model_id) for model_id in ("0p022b", "0p031b", "0p041b", "0p059b", "0p098b")
} | {
    ("fx1p6e18", model_id) for model_id in ("0p027b", "0p041b", "0p053b", "0p071b", "0p098b")
} | {
    ("fx2e18", model_id) for model_id in ("0p047b", "0p053b", "0p065b", "0p071b")
} | {
    ("fx2p4e18", model_id) for model_id in ("0p031b", "0p047b", "0p059b", "0p071b", "0p098b")
} | {
    ("fx2p9e18", model_id) for model_id in ("0p041b", "0p053b", "0p065b", "0p078b", "0p098b")
} | {
    ("fx3p5e18", model_id) for model_id in ("0p027b", "0p041b", "0p047b", "0p053b", "0p071b")
}


if __name__ == "__main__":
    builder.main()
