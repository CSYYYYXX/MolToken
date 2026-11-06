"""RAbD top-100 recovery with explicit representation-failure accounting.

The upstream AAR formula assumes fixed-length valid amino-acid outputs.
Expose that score only when the assumption holds. A separate conservative
fixed-slot score assigns zero to invalid/wrong-length candidates, never
shrinking the denominator via Python zip or replacement draws.
"""
import math

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def rank_candidates(rows, expected=10000, top=100):
    if len(rows) != expected or len({r["sample_index"] for r in rows}) != expected:
        raise ValueError("candidate count/unique draw IDs disagree with sampling budget")
    for row in rows:
        score = row["nll_per_requested_residue"]
        if score is not None and (not math.isfinite(score) or score < 0):
            raise ValueError("invalid candidate likelihood")
    # Stable ties follow draw order, matching upstream sorted(zip(...)).
    return sorted(rows, key=lambda r: (math.inf if r["nll_per_requested_residue"] is None
                                      else r["nll_per_requested_residue"], r["sample_index"]))[:top]


def score_selected(reference, rows, top=100):
    if not reference or set(reference) - AA or len(rows) != top:
        raise ValueError("reference alphabet or top-k count is invalid")
    matched = eligible = decoded = exact = 0
    for row in rows:
        seq = row.get("decoded_sequence")
        valid = bool(row["ended_with_eos"] and not row["invalid_model_id"]
                     and isinstance(seq, str) and seq and not set(seq) - AA)
        decoded += int(valid)
        if valid and len(seq) == len(reference):
            eligible += 1
            matched += sum(a == b for a, b in zip(reference, seq, strict=True))
            exact += int(seq == reference)
    denominator = top * len(reference)
    return {"selected": top, "decoded_valid": decoded, "valid_exact_length": eligible,
            "exact_sequences": exact, "matched_residues": matched,
            "requested_residues": denominator,
            "official_aar_assumptions_satisfied": eligible == top,
            "official_aar": matched / denominator if eligible == top else None,
            "fixed_slot_aar_invalid_or_wrong_length_zero": matched / denominator}


def aggregate_cases(cases):
    if not cases:
        raise ValueError("no RAbD cases")
    keys = ("selected", "decoded_valid", "valid_exact_length", "exact_sequences",
            "matched_residues", "requested_residues")
    result = {key: sum(row[key] for row in cases) for key in keys}
    valid = all(row["official_aar_assumptions_satisfied"] for row in cases)
    result["official_aar_assumptions_satisfied"] = valid
    aar = result["matched_residues"] / result["requested_residues"]
    result["official_aar"] = aar if valid else None
    result["fixed_slot_aar_invalid_or_wrong_length_zero"] = aar
    result["selected_validity"] = result["decoded_valid"] / result["selected"]
    result["selected_valid_exact_length_rate"] = result["valid_exact_length"] / result["selected"]
    return result


def score_selected_prefix(reference, rows, top=100):
    """User-approved prefix-L adapter; keep the original candidates/denominator.

    Extra residues are ignored. Missing positions and undecodable candidates
    contribute zero, without erasing correctly predicted existing positions.
    This does not turn free-EOS sampling into the author's fixed-length sampler.
    """
    if not reference or set(reference) - AA or len(rows) != top:
        raise ValueError("reference alphabet or top-k count is invalid")
    matched = valid_count = short = long = missing = removed = 0
    length = len(reference)
    for row in rows:
        seq = row.get("decoded_sequence")
        valid = bool(row["ended_with_eos"] and not row["invalid_model_id"]
                     and isinstance(seq, str) and seq and not set(seq) - AA)
        if not valid:
            continue
        valid_count += 1
        short += len(seq) < length
        long += len(seq) > length
        missing += max(0, length - len(seq))
        removed += max(0, len(seq) - length)
        matched += sum(a == b for a, b in zip(reference, seq[:length]))
    denominator = top * length
    return {"selected": top, "decoded_valid": valid_count, "too_short": short,
            "too_long": long, "missing_positions_valid_short": missing,
            "ignored_suffix_residues": removed, "matched_residues": matched,
            "requested_residues": denominator, "prefix_aar": matched / denominator}
