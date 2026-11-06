"""Input-only Codon context budgeting: never inspect a hidden reference DNA."""
from __future__ import annotations

import math

from pipeline.finetune.frozen_axis_errors import RingIdOverflow

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def fit_codon_input(protein, *, text_tokens, generation_tokens_per_codon,
                    encode_protein, max_context=512):
    """Fit a complete protein prefix plus a TRAIN-derived DNA budget estimate.

    The empirical density includes the model EOS token. It is an estimate, not
    a guarantee about unseen codon choices. Generation gets all remaining space;
    a missing EOS or wrong length stays a failed raw request, never a resample.
    The original protein and selected prefix are retained for matched-scope
    baseline scoring. No claim is made that this is the longest possible fit.
    """
    if not protein or set(protein) - AMINO_ACIDS:
        raise ValueError("expected a canonical protein without a stop symbol")
    if text_tokens < 1 or max_context < 4:
        raise ValueError("invalid text/context budget")
    if not math.isfinite(generation_tokens_per_codon) or generation_tokens_per_codon <= 0:
        raise ValueError("generation token density must be finite and positive")
    available = max_context - text_tokens - 2  # target type and model BOS
    if available < 2:
        raise ValueError("text and framing leave no condition/generation space")
    residues, attempts = len(protein), []
    while True:
        selected = protein[:residues]
        try:
            condition = encode_protein(selected)
        except RingIdOverflow as exc:
            attempts.append({"residues": residues, "reason": "ring_capacity", "error": str(exc)})
            if residues == 1:
                raise
            residues = max(1, min(residues - 1, int(residues * exc.ring_capacity / exc.max_ring_id * .9)))
            continue
        if not condition:
            raise ValueError("nonempty protein encoded to no tokens")
        terminal_stop = residues == len(protein)
        estimated_generation = math.ceil(generation_tokens_per_codon * (residues + int(terminal_stop)))
        estimated_cost = len(condition) + estimated_generation
        if estimated_cost <= available:
            return {"source_protein": protein, "protein": selected,
                    "source_residues": len(protein), "retained_residues": residues,
                    "terminal_stop": terminal_stop,
                    "scope": "full_CDS" if terminal_stop else "coding_prefix",
                    "protein_bpe_ids": condition, "text_tokens": text_tokens,
                    "prompt_tokens": text_tokens + len(condition) + 2,
                    "estimated_generation_tokens": estimated_generation,
                    "max_new_tokens": available - len(condition),
                    "max_context": max_context, "attempts": attempts,
                    "generation_tokens_per_codon": generation_tokens_per_codon,
                    "reference_DNA_used_for_budget": False}
        attempts.append({"residues": residues, "reason": "estimated_context",
                         "condition_tokens": len(condition), "estimated_generation_tokens": estimated_generation})
        if residues == 1:
            raise ValueError("one residue and estimated coding target do not fit context")
        residues = max(1, min(residues - 1, int(residues * available / estimated_cost * .97)))
