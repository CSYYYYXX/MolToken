"""DPLM released TM-align scoring and raw-slot-aware inner-TM aggregation.

Released code uses -fast, Chain_1 normalization and includes diagonal pairs.
The paper's 'against the rest' convention is also reported, separately.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import re
import subprocess


def parse_tmalign(output: str) -> dict:
    scores = {}
    for line in output.splitlines():
        if not line.startswith("TM-score"):
            continue
        match = re.search(r"TM-score\s*=\s*([0-9.eE+-]+).*Chain_([12])", line)
        if match is None:
            raise ValueError(f"unrecognized native TM-score line: {line}")
        key = f"Chain_{match[2]}"
        if key in scores:
            raise ValueError(f"duplicate native {key} score")
        scores[key] = float(match[1])
    if set(scores) != {"Chain_1", "Chain_2"}:
        raise ValueError("native TM-align did not return both chain-normalized scores")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError("native TM-score outside [0, 1]")
    return scores


def run_tmalign(binary: Path, query: str, reference: str) -> dict:
    # Use the original directed invocation, not the Chain_2 score from a
    # reversed pair: reversing inputs can change TM-align's optimization.
    result = subprocess.run([str(binary), query, reference, "-fast"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"TM-align failed ({result.returncode}) for {query} vs {reference}:\n"
                           f"{result.stdout}\n{result.stderr}")
    return parse_tmalign(result.stdout)


def score_inner_tm(rows, align_one, *, conditions, workers=4):
    if workers < 1:
        raise ValueError("workers must be positive")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate raw slot IDs")
    groups = {condition: [] for condition in conditions}
    for row in rows:
        if row["condition"] not in groups or type(row["folded"]) is not bool:
            raise ValueError("unexpected folded slot condition/status")
        if row["folded"] and (row["status"] != "valid" or not row["pdb"]):
            raise ValueError("folded slot must be valid with a PDB path")
        groups[row["condition"]].append(row)
    for condition, group in groups.items():
        if not group or sorted(r["sample_index"] for r in group) != list(range(len(group))):
            raise ValueError(f"{condition}: missing or duplicate raw slot indices")

    valid_groups = {c: [r for r in group if r["folded"]] for c, group in groups.items()}
    pairs = sorted({(a["pdb"], b["pdb"]) for group in valid_groups.values()
                    for a in group for b in group})
    cache = {}
    def align(pair):
        scores = align_one(*pair)
        if any(not math.isfinite(scores[k]) or not 0 <= scores[k] <= 1 for k in ("Chain_1", "Chain_2")):
            raise ValueError("non-finite/out-of-range TM-align result")
        return pair, scores
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (pair, scores) in enumerate(pool.map(align, pairs), 1):
            cache[pair] = scores
            if count % 100 == 0:
                print(f"TMALIGN_PROGRESS {count}/{len(pairs)} directed unique pairs", flush=True)

    def aggregate(group):
        m = len(group)
        if m == 0:
            return {"slots": 0, "ordered_pairs_including_self": 0, "ordered_pairs_excluding_self": 0,
                    "released_inner_tm_including_self": None, "paper_inner_tm_excluding_self": None}
        all_scores = [cache[a["pdb"], b["pdb"]]["Chain_1"] for a in group for b in group]
        other_scores = [cache[a["pdb"], b["pdb"]]["Chain_1"] for a in group for b in group if a["id"] != b["id"]]
        return {"slots": m, "ordered_pairs_including_self": len(all_scores),
                "ordered_pairs_excluding_self": len(other_scores),
                "released_inner_tm_including_self": math.fsum(all_scores) / len(all_scores),
                "paper_inner_tm_excluding_self": math.fsum(other_scores) / len(other_scores) if other_scores else None}

    per_condition, per_slot = {}, []
    for condition, group in groups.items():
        valid = valid_groups[condition]
        exact = [r for r in valid if r["exact_length_match"]]
        per_condition[condition] = {"requested": len(group), "folded": len(valid),
            "fold_coverage": len(valid) / len(group), "all_folded": aggregate(valid),
            "exact_length_only": aggregate(exact)}
        for row in group:
            item = {key: row[key] for key in ("id", "condition", "sample_index", "status", "folded", "pdb")}
            item.update(released_inner_tm_including_self=None, paper_inner_tm_excluding_self=None)
            if row["folded"]:
                scores = [cache[row["pdb"], other["pdb"]]["Chain_1"] for other in valid]
                rest = [cache[row["pdb"], other["pdb"]]["Chain_1"] for other in valid if other["id"] != row["id"]]
                item.update(released_inner_tm_including_self=math.fsum(scores) / len(scores),
                            paper_inner_tm_excluding_self=math.fsum(rest) / len(rest) if rest else None)
            per_slot.append(item)
    macro = {}
    for subset in ("all_folded", "exact_length_only"):
        macro[subset] = {}
        for metric in ("released_inner_tm_including_self", "paper_inner_tm_excluding_self"):
            values = [group[subset][metric] for group in per_condition.values()]
            macro[subset][metric] = math.fsum(values) / len(values) if all(v is not None for v in values) else None
    total_valid = sum(len(group) for group in valid_groups.values())
    summary = {"inner_tm_scoring_complete": True, "structure_suite_complete": False,
        "missing_suite_metrics": ["pdb-TM"], "requested": len(rows), "folded": total_valid,
        "fold_coverage": total_valid / len(rows), "unique_directed_alignments": len(cache),
        "per_condition": per_condition, "macro": macro,
        "normalization": "native TM-align Chain_1/query length; -fast; directed input order retained",
        "aggregation": "per query mean over same-condition raw folded slots, then query mean; duplicate slots retain multiplicity",
        "diagonal_policy": "released code includes self; paper 'against the rest' is separate and excludes only identical slot IDs, not duplicate sequences",
        "invalid_slots_replaced": False, "sequences_repaired_or_truncated": False}
    unique_pairs = [{"query": a, "reference": b, **scores} for (a, b), scores in cache.items()]
    return summary, per_slot, unique_pairs
