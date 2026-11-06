"""Bookkeeping around the author's CSI/GC/CFD functions, not new formulas.

Frequency references must be supplied explicitly. This module neither derives
them from evaluation outputs nor crops/resamples a generated sequence. Coding
fragments and full coding sequences have different requested stop semantics.
"""
from __future__ import annotations

import math
from collections import Counter

from Bio.Data import CodonTable
from Bio.Seq import Seq

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
DNA_ALPHABET = frozenset("ACGT")


def score_coding_output(dna, *, protein, terminal_stop, official,
                        csi_weights, codon_frequencies, genetic_code=11):
    """Score one raw request; invalid outputs remain explicit unscored slots.

CSI/GC/CFD describe all nonempty, in-frame ACGT outputs, including sequences
that fail translation matching. Consumers must keep the translation coverage
alongside these conditional property scores. GC and CSI are not expression
measurements; there is no blanket 'higher GC is better' interpretation.
"""
    if not protein or set(protein) - AMINO_ACIDS:
        raise ValueError("requested protein must contain canonical residues, without stop")
    if type(terminal_stop) is not bool or genetic_code not in (1, 11):
        raise ValueError("explicit full-CDS/fragment stop policy and code 1 or 11 required")
    result = {"dna": dna, "requested_residues": len(protein),
              "terminal_stop_requested": terminal_stop, "genetic_code": genetic_code,
              "frame_valid": False, "exact_length": False,
              "translation_matches": False, "translated_protein": None,
              "CSI": None, "GC_percent": None, "CFD_percent": None,
              "csi_undefined_reason": None}
    if dna is None:
        return {**result, "status": "decode_failure"}
    if not isinstance(dna, str):
        raise TypeError("decoded DNA must be str or explicit None for decode failure")
    if not dna:
        return {**result, "status": "empty_dna"}
    if set(dna) - DNA_ALPHABET:
        return {**result, "status": "noncanonical_dna"}
    if len(dna) % 3:
        return {**result, "status": "incomplete_codon"}

    translated = str(Seq(dna).translate(table=genetic_code, cds=False, to_stop=False))
    expected = protein + ("*" if terminal_stop else "")
    result.update(frame_valid=True, exact_length=len(dna) == 3 * len(expected),
                  translated_protein=translated, translation_matches=translated == expected,
                  status="scored")
    table = CodonTable.unambiguous_dna_by_id[genetic_code].forward_table
    amino_counts = {amino: sum(value == amino for value in table.values())
                   for amino in set(table.values())}
    informative = any(codon in table and amino_counts[table[codon]] > 1
                      for codon in (dna[i:i + 3] for i in range(0, len(dna), 3)))
    if informative:
        result["CSI"] = float(official.get_CSI_value(dna, csi_weights))
    else:
        # CAI-PyPI defines an empty geometric mean as NaN for M/W/stop-only DNA.
        result["csi_undefined_reason"] = "no_synonymous_sense_codons"
    result["GC_percent"] = float(official.get_GC_content(dna, lower=False))
    result["CFD_percent"] = float(official.get_cfd(dna, codon_frequencies, threshold=0.3))
    for name in ("CSI", "GC_percent", "CFD_percent"):
        if result[name] is not None and not math.isfinite(result[name]):
            raise ValueError(f"author evaluator returned non-finite {name}: {result[name]}")
    return result


def decode_coding_slot(sample, decode):
    """Keep every raw request; catch only expected invalid graph/sequence errors."""
    if sample["invalid_model_id"]:
        return None, "invalid_model_id", None
    if not sample["ended_with_eos"]:
        return None, "missing_eos", None
    if not sample["generated_bpe_ids"]:
        return None, "empty_generation", None
    try:
        dna = decode(sample["generated_bpe_ids"])
    except ValueError as exc:
        return None, "graph_decode_failure", str(exc)
    if not isinstance(dna, str):
        raise TypeError("graph decoder did not return a DNA string")
    return dna, "decoded", None


def summarize_coding_slots(rows):
    """Property means always carry their coverage, plus translation-matched means."""
    if not rows:
        raise ValueError("cannot summarize zero requested slots")
    metrics = [row["metrics"] for row in rows]
    result = {"requested": len(rows),
              "generation_statuses": dict(Counter(row["generation_status"] for row in rows)),
              "decoded": sum(m["dna"] is not None for m in metrics)}
    for key in ("frame_valid", "exact_length", "translation_matches"):
        count = sum(m[key] for m in metrics)
        result[key] = {"count": count, "rate_all_requests": count / len(rows)}
    for scope, selected in (("all_frame_valid", [m for m in metrics if m["frame_valid"]]),
                            ("translation_matched", [m for m in metrics if m["translation_matches"]])):
        result[scope] = {}
        for key in ("CSI", "GC_percent", "CFD_percent"):
            values = [m[key] for m in selected if m[key] is not None]
            result[scope][key] = {"scored": len(values), "coverage_all_requests": len(values) / len(rows),
                                  "mean": math.fsum(values) / len(values) if values else None}
    return result
