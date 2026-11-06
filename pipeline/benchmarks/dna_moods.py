"""Native DNA-Diffusion released-hit reconstruction, separate from its headline suite.

The published hit fixtures support ps=.001, not the article's stated .0001.
Callers must disclose this asset-reconstruction convention. No claim is made
here about the unresolved Figure7 Gini aggregation or Enformer evaluation.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


class ReleasedDNAMotifScanner:
    def __init__(self, profile_json: Path):
        import MOODS.scan
        import MOODS.tools

        document = json.loads(profile_json.read_text())
        profiles = document["profiles"]
        self.matrix_ids = [m["matrix_id"] for m in profiles]
        if not self.matrix_ids or len(self.matrix_ids)!=len(set(self.matrix_ids)):
            raise ValueError("motif collection must be nonempty and contain distinct matrix IDs")
        self.profile_json = str(profile_json)
        self.background = [0.25]*4
        self.pseudocount = 0.001
        self.p_value = 0.0001
        matrices = [MOODS.tools.log_odds([m["counts"][b] for b in "ACGT"],self.background,self.pseudocount) for m in profiles]
        matrices += [MOODS.tools.reverse_complement(m,4) for m in matrices]
        self.lengths = [len(m[0]) for m in matrices]
        thresholds = [MOODS.tools.threshold_from_p_with_precision(m,self.background,self.p_value,MOODS.tools.DEFAULT_DP_PRECISION,4) for m in matrices]
        self.scanner = MOODS.scan.Scanner(7)
        self.scanner.set_motifs(matrices,self.background,thresholds)

    def hits(self, sequence: str) -> list[dict]:
        if len(sequence)!=200 or set(sequence)-set("ACGT"):
            raise ValueError("native DNA benchmark scanner requires canonical 200-bp DNA")
        n = len(self.matrix_ids)
        return [{"matrix_id":self.matrix_ids[i%n],"start":m.pos,"end":m.pos+self.lengths[i],
                 "strand":"+" if i<n else "-","score":m.score}
                for i,matches in enumerate(self.scanner.scan(sequence)) for m in matches]

    def counts(self, sequence: str) -> Counter:
        return Counter(hit["matrix_id"] for hit in self.hits(sequence))

    def provenance(self) -> dict:
        import importlib.metadata
        return {"MOODS_version":importlib.metadata.version("MOODS-python"),"profile_json":self.profile_json,
                "matrix_ids":self.matrix_ids,"pseudocount":self.pseudocount,"p_value":self.p_value,
                "log_base":"natural","threshold_background":self.background,
                "log_odds_background":self.background,"both_strands":True,
                "scope":"released-hit reconstruction; not complete DNA-Diffusion headline evaluator"}
