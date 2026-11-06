"""Aligned coding-prefix fallback for targets too long for the SFT window."""


def coding_prefix(protein, dna, residues):
    if not protein or len(dna) != 3 * (len(protein) + 1):
        raise ValueError("expected published protein and DNA including one terminal stop codon")
    if not 1 <= residues < len(protein):
        raise ValueError("fallback prefix must be nonempty and shorter than the source protein")
    # No partial codon and no invented terminal stop. This is a coding fragment,
    # not an assertion that the truncated DNA is a complete original gene.
    return protein[:residues], dna[:3 * residues]


def fit_coding_prefix(protein, dna, *, text_tokens, original_protein_bpe,
                      original_dna_bpe, encode_sequence, max_context=512):
    available = max_context - text_tokens - 3
    if available <= 0:
        raise ValueError("species text leaves no coding sequence budget")
    residues = min(len(protein) - 1, max(1, int(
        len(protein) * available / (original_protein_bpe + original_dna_bpe))))
    attempts = 0
    while True:
        aa, nt = coding_prefix(protein, dna, residues)
        condition = encode_sequence("protein", aa)
        target = encode_sequence("dna", nt)
        attempts += 1
        actual = len(condition) + len(target)
        if actual <= available:
            return {"protein": aa, "dna": nt, "condition_bpe_ids": condition,
                    "target_bpe_ids": target, "protein_residues_kept": residues,
                    "dna_bases_kept": len(nt), "encoding_attempts": attempts,
                    "full_context_tokens": actual + text_tokens + 3}
        if residues == 1:
            raise ValueError("even one complete AA/codon pair exceeds the available context")
        # Length is checked after actual frozen encoding; BPE token counts need
        # not be monotonic. Only reduce, and do not claim a maximal prefix.
        residues = max(1, min(residues - 1, int(residues * available / actual * .97)))
