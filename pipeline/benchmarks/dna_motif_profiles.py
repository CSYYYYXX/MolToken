"""Load declared JASPAR profile data; full-precision JSON does not round counts."""
from __future__ import annotations

import json
from pathlib import Path


def load_profiles(assets: Path, profile_json: Path | None = None) -> list[dict]:
    if profile_json is not None:
        return json.loads(profile_json.read_text())["profiles"]
    # Historical text download used only as an explicitly selected control.
    from Bio import motifs
    profiles = []
    for filename in ("JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt", "MA0095.2_YY1.jaspar"):
        with (assets/filename).open() as handle:
            profiles.extend({"matrix_id":m.matrix_id,"counts":{b:list(m.counts[b]) for b in "ACGT"}} for m in motifs.parse(handle,"jaspar"))
    return profiles
