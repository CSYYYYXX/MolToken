#!/usr/bin/env python3
"""Decode raw BPE samples and compute basic validity/uniqueness/novelty.

Graph validity and RDKit chemical validity both use all generated attempts as
the denominator. Uniqueness is unique canonical SMILES / chemically valid
draws. Novelty is unseen unique SMILES / unique SMILES for the supplied
reference. This strict decoder performs no aromatic completion. These basic
diagnostics do not replace the official MOSES or GuacaMol evaluation adapters.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import csv as _csv
import json
import sys
from collections import Counter
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[2]
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

DATA = Path(str(_moltoken_paths.LOCAL_DATA_ROOT / ''))
BPE = DATA / "fddfs_bpe_full_prefix70_64k_mf80000_v1"
CODEBOOK = _moltoken_paths.TOKENIZER_ASSET_ROOT
VOCAB_JSON = CODEBOOK / "vocab.json"

MAX_RAW_TOKEN_ID = 17203

def load_codebook() -> dict:
    """加载 vocab + merges，构造 unfold 表与 id->token 映射。"""
    vobj = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
    tok2id = {str(k): int(v) for k, v in vobj["token_to_id"].items()}
    id2tok = {v: k for k, v in tok2id.items()}

    merges: dict[int, tuple[int, int]] = {}
    with (CODEBOOK / "merges.csv").open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            merges[int(row["new_id"])] = (int(row["left"]), int(row["right"]))

    return {
        "id2tok": id2tok,
        "ring_base": int(vobj["ring_base"]),
        "ring_min": int(vobj["ring_min"]),
        "default_edge": tuple(int(x) for x in vobj["default_edge_feature"]),
        "merges": merges,
        "min_merge_id": min(merges) if merges else 1 << 30,
    }


def unfold_bpe(ids: list[int], merges: dict[int, tuple[int, int]],
               min_merge_id: int) -> list[int]:
    """把 BPE 合并 id 递归展开成基础 id（显式栈，避免深递归）。"""
    out: list[int] = []
    stack = list(reversed(ids))
    budget = 64 * max(1, len(ids)) + 4096
    steps = 0
    while stack:
        steps += 1
        if steps > budget:
            raise RuntimeError("unfold exceeded step budget (cyclic merges?)")
        t = stack.pop()
        if t >= min_merge_id:
            pair = merges.get(t)
            if pair is None:
                raise KeyError(f"merge id {t} not in table")
            stack.append(pair[1])
            stack.append(pair[0])
        else:
            out.append(t)
    return out


def ids_to_tokens(base_ids: list[int], id2tok: dict[int, str],
                  ring_base: int, ring_min: int) -> list[str]:
    toks: list[str] = []
    for t in base_ids:
        s = id2tok.get(t)
        if s is not None:
            toks.append(s)
        elif t >= ring_base:
            toks.append(f"R{t - ring_base + ring_min}")
        else:
            raise KeyError(f"unmappable base id {t}")
    return toks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="sample_flagship_unconditional.py 的输出")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reference", type=Path, default=None,
                    help="novelty 参考集（每行一个 canonical SMILES）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-smiles", type=Path, default=None,
                    help="可选：把解出的 canonical SMILES 落盘")
    ap.add_argument("--max-records", type=int, default=0, help="0=全部")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from pipeline.serialization.fddfs import deserialize_fddfs
    from pipeline.graph.to_smiles import graph_to_smiles

    cb = load_codebook()
    id2tok = cb["id2tok"]
    ring_base = cb["ring_base"]
    ring_min = cb["ring_min"]
    default_edge = cb["default_edge"]
    merges = cb["merges"]
    min_merge_id = cb["min_merge_id"]

    ref: set[str] = set()
    if args.reference is not None and args.reference.is_file():
        with args.reference.open(encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    ref.add(s)

    stats: Counter = Counter()
    valid_smiles: list[str] = []
    examples: list[str] = []

    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.max_records and stats["total"] >= args.max_records:
                break
            stats["total"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            if bool(rec.get("invalid_model_id", False)):
                stats["invalid_model_id"] += 1
                continue
            if rec.get("ended_with_eos") is False:
                stats["incomplete_output"] += 1
                continue
            ids = [int(x) for x in rec.get("bpe_ids", [])]
            if not ids:
                stats["empty_generation"] += 1
                continue
            if max(ids) > MAX_RAW_TOKEN_ID:
                stats["id_out_of_range"] += 1
                continue
            stats["attempted"] += 1

            try:
                base_ids = unfold_bpe(ids, merges, min_merge_id)
                toks = ids_to_tokens(base_ids, id2tok, ring_base, ring_min)
            except Exception as exc:
                stats[f"fail_unfold:{type(exc).__name__}"] += 1
                continue
            try:
                graph = deserialize_fddfs(toks, default_edge)
            except Exception as exc:
                stats[f"fail_deserialize:{type(exc).__name__}"] += 1
                continue
            stats["graph_ok"] += 1
            try:
                smi = graph_to_smiles(graph)
            except Exception as exc:
                msg = str(exc)
                if "ekuliz" in msg:
                    stats["fail_kekulize"] += 1
                else:
                    stats[f"fail_smiles:{type(exc).__name__}"] += 1
                continue
            if not smi:
                stats["fail_empty_smiles"] += 1
                continue
            stats["smiles_ok"] += 1
            valid_smiles.append(smi)
            if len(examples) < args.examples:
                examples.append(smi)

    total = max(1, stats["total"])
    attempted = max(1, stats["attempted"])
    uniq = set(valid_smiles)
    n_novel = len([s for s in uniq if s not in ref]) if ref else None

    result = {
        "tag": args.tag or args.jsonl.stem,
        "jsonl": str(args.jsonl),
        "reference": str(args.reference) if args.reference else None,
        "reference_size": len(ref) if ref else 0,
        "total_records": stats["total"],
        "empty_generation": stats["empty_generation"],
        "incomplete_output": stats["incomplete_output"],
        "invalid_model_id": stats["invalid_model_id"],
        "attempted": stats["attempted"],
        # Use the fixed generation count as the validity denominator. Empty
        # and out-of-range outputs are invalid samples, not missing trials.
        "V_graph": stats["graph_ok"] / total,
        "V_chem": stats["smiles_ok"] / total,
        "V_graph_nonempty_attempted": stats["graph_ok"] / attempted,
        "V_chem_nonempty_attempted": stats["smiles_ok"] / attempted,
        "validity_denominator": "total_records",
        "n_graph_ok": stats["graph_ok"],
        "n_smiles_ok": stats["smiles_ok"],
        "n_unique": len(uniq),
        "U": len(uniq) / max(1, stats["smiles_ok"]),
        "n_novel": n_novel,
        "N": (n_novel / max(1, len(uniq))) if n_novel is not None else None,
        "counters": dict(stats),
        "examples": examples,
        "baseline_real_corpus": {
            "V_graph": 0.9987,
            "V_chem": 0.9175,
            "note": ("22000 real small_molecule records; the 8.25% V_chem gap is "
                     "representation loss (aromatic N-H kekulize), not a decoder bug"),
        },
        "caveat_N": ("reference set is a corpus subset -> N is an UPPER BOUND "
                     "(true novelty <= reported)"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    if args.out_smiles is not None:
        args.out_smiles.parent.mkdir(parents=True, exist_ok=True)
        with args.out_smiles.open("w", encoding="utf-8") as f:
            for s in sorted(uniq):
                f.write(s + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
