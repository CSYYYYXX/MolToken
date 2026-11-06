"""SFT-only context packing that preserves the complete target token sequence.

Pack offline, then use the existing strict AxisBSFTCollator. No global collator
default changes. A test-time prompt must not be packed from the hidden target's
BPE length; inference needs an independently determined generation budget.
"""
from copy import deepcopy


def target_priority_budget(*, target_tokens, text_tokens, molecule_tokens, max_context=512):
    if target_tokens < 1 or text_tokens < 0 or molecule_tokens < 0 or max_context < 4:
        raise ValueError("invalid sequence lengths for target-priority packing")
    # Axis decoder: target type + BOS + raw BPE target + EOS.
    framing = 3
    available = max_context - target_tokens - text_tokens - framing
    if target_tokens + framing > max_context:
        status = "complete_target_exceeds_context"
    elif available < 0:
        status = "complete_target_and_text_exceed_context"
    else:
        status = "fits"
    kept = min(molecule_tokens, max(0, available))
    return {
        "status": status, "max_context": max_context, "framing_tokens": framing,
        "target_tokens": target_tokens, "text_tokens": text_tokens,
        "molecule_tokens_original": molecule_tokens,
        "molecule_token_cap": max(0, available), "molecule_tokens_kept": kept,
        "molecule_tokens_removed": molecule_tokens - kept,
        "condition_truncated": kept < molecule_tokens,
        "protein_condition_empty": molecule_tokens > 0 and kept == 0,
        "packed_context_tokens": target_tokens + text_tokens + framing + kept,
        "target_truncated": False,
    }


def pack_target_first(record, *, text_tokens, max_context=512):
    """Return (packed_record, report); impossible rows return (None, report).

    The caller must retain/report impossible rows, not silently drop them.
    A zero remaining protein budget is explicit; its record is text-only.
    This function expects the plain species text used by CodonTransformer,
    not an instruction assembled from condition modality/role metadata.
    """
    condition = record["condition"]
    if set(condition) != {"type", "text", "molecule_bpe_ids"}:
        raise ValueError("expected plain text plus direct molecular BPE condition")
    if condition["type"] != "both" or not condition["text"].strip() or not condition["molecule_bpe_ids"]:
        raise ValueError("target-priority input must contain species text and protein BPE")
    if not record["target_bpe_ids"] or text_tokens <= 0:
        raise ValueError("complete target and tokenized text must be nonempty")
    report = target_priority_budget(target_tokens=len(record["target_bpe_ids"]),
                                    text_tokens=text_tokens,
                                    molecule_tokens=len(condition["molecule_bpe_ids"]),
                                    max_context=max_context)
    report["id"] = record["id"]
    if report["status"] != "fits":
        return None, report
    packed = deepcopy(record)
    packed["condition"]["molecule_bpe_ids"] = condition["molecule_bpe_ids"][:report["molecule_tokens_kept"]]
    if report["protein_condition_empty"]:
        packed["condition"]["type"] = "text"
    packed.setdefault("meta", {}).update(
        context_policy="sft_complete_target_first_v1",
        target_truncated=False, condition_truncated=report["condition_truncated"],
        protein_condition_empty=report["protein_condition_empty"],
        molecule_prefix_tokens=report["molecule_tokens_kept"],
        molecule_prefix_tokens_original=report["molecule_tokens_original"],
        molecule_token_cap=report["molecule_token_cap"],
        target_bpe_tokens=report["target_tokens"], text_prefix_tokens=text_tokens,
        full_context_tokens=report["packed_context_tokens"],
    )
    return packed, report
