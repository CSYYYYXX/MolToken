"""Semantic Qwen/token interface used by the current downstream model.

Acceptance is based on the model vocabulary and token roles that affect
execution. File and source hashes are not runtime gates.
"""

from __future__ import annotations

from typing import Any, Mapping


BASE_MODEL_VOCAB_SIZE = 17_207
MODEL_VOCAB_SIZE = 17_213
RESERVED_MASK_TOKEN_ID = 17_207
TARGET_TYPE_TOKEN_IDS = {
    "small_molecule": 17_208,
    "peptide": 17_209,
    "protein": 17_210,
    "dna": 17_211,
    "rna": 17_212,
}

QWEN_AXIS_B_MARKER = {
    "format": "axis_b_checkpoint_interface_v2",
    "base_model_vocab_size": BASE_MODEL_VOCAB_SIZE,
    "model_vocab_size": MODEL_VOCAB_SIZE,
    "reserved_mask_token_id": RESERVED_MASK_TOKEN_ID,
    "target_type_token_ids": dict(TARGET_TYPE_TOKEN_IDS),
}


def axis_b_marker_is_compatible(value: Any) -> bool:
    """Check execution-relevant checkpoint semantics, including legacy markers."""

    if not isinstance(value, Mapping):
        return False
    try:
        base_size = int(value.get("base_model_vocab_size", -1))
        model_size = int(value.get("model_vocab_size", -1))
    except (TypeError, ValueError):
        return False
    if (base_size, model_size) != (BASE_MODEL_VOCAB_SIZE, MODEL_VOCAB_SIZE):
        return False
    target_ids = value.get("target_type_token_ids")
    if target_ids is not None and dict(target_ids) != TARGET_TYPE_TOKEN_IDS:
        return False
    mask_id = value.get("reserved_mask_token_id")
    if mask_id is not None and int(mask_id) != RESERVED_MASK_TOKEN_ID:
        return False
    return True


__all__ = [
    "BASE_MODEL_VOCAB_SIZE",
    "MODEL_VOCAB_SIZE",
    "QWEN_AXIS_B_MARKER",
    "RESERVED_MASK_TOKEN_ID",
    "TARGET_TYPE_TOKEN_IDS",
    "axis_b_marker_is_compatible",
]
