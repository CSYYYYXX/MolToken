#!/usr/bin/env python3
"""Build a canonical-SMILES reference from indexed compact-int corpus shards.

The inverse chain is int32 raw IDs -> BPE expansion -> FDDFS -> heavy-atom
graph -> canonical SMILES. Domain sidecars select small molecules independently
of their record order. The output is sorted unique SMILES plus metadata.
Novelty against a subset reference is an upper bound on whole-corpus novelty;
use the official benchmark references for official scores.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import csv as _csv
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PIPELINE = Path(str(Path(__file__).resolve().parents[2]))
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

DATA = Path(str(_moltoken_paths.LOCAL_DATA_ROOT / ''))
BPE_ROOT = DATA / "fddfs_bpe_full_prefix70_64k_mf80000_v1"
SHARD_DIR = BPE_ROOT / "encoded_compact_int_prefix70_mf80000_11w_v3"
INDEX_DIR = BPE_ROOT / "encoded_compact_int_prefix70_mf80000_11w_v3_index"
DOMAIN_DIR = BPE_ROOT / "encoded_compact_int_prefix70_mf80000_11w_v3_domain_ids"
CODEBOOK = _moltoken_paths.TOKENIZER_ASSET_ROOT
VOCAB_JSON = CODEBOOK / "vocab.json"

DOMAIN_ORDER = ("small_molecule", "peptide", "protein", "dna", "rna")
SMALL_MOLECULE_DOMAIN_ID = 0

_G: dict = {}


def load_codebook() -> dict:
    """加载 vocab + merges，构造 unfold 表和 id->token 映射。"""
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
        "unfold_tbl": merges,
        "min_merge_id": min(merges) if merges else 1 << 30,
    }


def unfold_bpe(ids: list[int], unfold_tbl: dict[int, tuple[int, int]],
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
            pair = unfold_tbl.get(t)
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


def _init_worker() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from pipeline.serialization.fddfs import deserialize_fddfs
    from pipeline.graph.to_smiles import graph_to_smiles

    _G.update(load_codebook())
    _G["deserialize_fddfs"] = deserialize_fddfs
    _G["graph_to_smiles"] = graph_to_smiles


def _one_shard(task: tuple[str, int]) -> tuple[str, list[str], dict]:
    """解码一个 shard 中的小分子记录。

    task = (shard 文件名, 该 shard 最多解码多少条小分子; <=0 表示全部)
    """
    shard_name, cap = task
    shard = SHARD_DIR / shard_name
    lengths_p = INDEX_DIR / f"{shard_name}.lengths.u32"
    offsets_p = INDEX_DIR / f"{shard_name}.payload_offsets.u64"
    domain_p = DOMAIN_DIR / f"{shard_name}.domain_ids.u16"

    stats: Counter = Counter()
    smiles_out: list[str] = []

    for p in (lengths_p, offsets_p, domain_p):
        if not p.exists():
            stats[f"missing:{p.name}"] += 1
            return shard_name, smiles_out, dict(stats)

    # 小文件整体读入（lengths ~117MB, offsets ~235MB, domains ~59MB）
    lengths = np.frombuffer(lengths_p.read_bytes(), dtype=np.uint32)
    offsets = np.frombuffer(offsets_p.read_bytes(), dtype=np.uint64)
    domains = np.frombuffer(domain_p.read_bytes(), dtype=np.uint16)

    n = min(lengths.shape[0], offsets.shape[0], domains.shape[0])
    stats["shard_records"] = int(n)

    # 关键：shard 内按 domain 分块，必须先定位小分子的精确下标
    sm_idx = np.where(domains[:n] == SMALL_MOLECULE_DOMAIN_ID)[0]
    stats["small_molecule_records"] = int(sm_idx.size)
    if sm_idx.size == 0:
        return shard_name, smiles_out, dict(stats)

    # 在小分子下标里均匀 stride 抽样（不是取前 N 个，避免子块内偏置）
    if cap > 0 and sm_idx.size > cap:
        step = sm_idx.size / cap
        pick = sm_idx[(np.arange(cap) * step).astype(np.int64)]
    else:
        pick = sm_idx
    stats["selected"] = int(pick.size)

    deserialize_fddfs = _G["deserialize_fddfs"]
    graph_to_smiles = _G["graph_to_smiles"]
    id2tok = _G["id2tok"]
    ring_base = _G["ring_base"]
    ring_min = _G["ring_min"]
    default_edge = _G["default_edge"]
    unfold_tbl = _G["unfold_tbl"]
    min_merge_id = _G["min_merge_id"]

    with shard.open("rb") as fh:
        for i in pick.tolist():
            ln = int(lengths[i])
            if ln <= 0:
                stats["empty"] += 1
                continue
            fh.seek(int(offsets[i]))
            seg = fh.read(ln * 4)
            if len(seg) != ln * 4:
                stats["short_read"] += 1
                continue
            stats["attempted"] += 1
            # payload dtype 是 uint32（实测确认）：
            #   * 按 uint16 读时每个奇数位恒为 0 = 小端 uint32 的高半字
            #   * stride == lengths[i]*4 + 4（逐条核对 len=37->152/41->168/40->164）
            #   * offsets[-1] + lengths[-1]*4 == 文件大小，精确闭合
            #   * uint16 读法 deserialize 0/200；uint32 读法 199/200
            ids = np.frombuffer(seg, dtype=np.uint32).tolist()
            if ids and max(ids) > 17203:
                stats["id_out_of_range"] += 1
                continue
            try:
                base_ids = unfold_bpe(ids, unfold_tbl, min_merge_id)
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
                key = "fail_kekulize" if "ekuliz" in msg else f"fail_smiles:{type(exc).__name__}"
                stats[key] += 1
                continue
            if not smi:
                stats["fail_empty_smiles"] += 1
                continue
            stats["smiles_ok"] += 1
            smiles_out.append(smi)

    return shard_name, smiles_out, dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-shard", type=int, default=200_000,
                    help="每个 shard 最多解码多少条小分子（0=全部，约 830 万/shard）")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path(str(_moltoken_paths.LOCAL_DATA_ROOT / 'novelty_ref_small_molecule.txt')))
    args = ap.parse_args()

    shards = sorted(p.name for p in SHARD_DIR.glob("shard_*.bin"))
    if not shards:
        raise FileNotFoundError(f"no shards under {SHARD_DIR}")
    print(f"shards={len(shards)}  per_shard={args.per_shard}  procs={args.procs}",
          flush=True)

    t0 = time.time()
    ref: set[str] = set()
    agg: Counter = Counter()
    tasks = [(s, args.per_shard) for s in shards]
    with ProcessPoolExecutor(max_workers=args.procs, initializer=_init_worker) as ex:
        for k, (name, smis, stats) in enumerate(ex.map(_one_shard, tasks), 1):
            before = len(ref)
            ref.update(smis)
            agg.update(stats)
            print(f"  [{k}/{len(shards)}] {name} "
                  f"sm_records={stats.get('small_molecule_records', 0):,} "
                  f"selected={stats.get('selected', 0):,} "
                  f"decoded={len(smis):,} new_unique={len(ref) - before:,} "
                  f"total_unique={len(ref):,} {time.time() - t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in sorted(ref):
            f.write(s + "\n")

    h = hashlib.sha256()
    with args.out.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    attempted = agg.get("attempted", 0)
    meta = {
        "shard_dir": str(SHARD_DIR),
        "n_shards": len(shards),
        "per_shard_cap": args.per_shard,
        "domain_order": list(DOMAIN_ORDER),
        "domain_id": SMALL_MOLECULE_DOMAIN_ID,
        "domain_name": "small_molecule",
        "sampling": ("np.where(domain==0) 定位精确下标后均匀 stride 抽样；"
                     "shard 内按 domain 分块，故不可用前 N 条"),
        "shard_records_total": agg.get("shard_records", 0),
        "small_molecule_records_total": agg.get("small_molecule_records", 0),
        "records_attempted": attempted,
        "graph_ok": agg.get("graph_ok", 0),
        "smiles_ok": agg.get("smiles_ok", 0),
        "graph_rate": agg.get("graph_ok", 0) / attempted if attempted else None,
        "smiles_rate": agg.get("smiles_ok", 0) / attempted if attempted else None,
        "unique_canonical_smiles": len(ref),
        "all_counters": dict(agg),
        "elapsed_sec": time.time() - t0,
        "out_file": str(args.out),
        "out_sha256": h.hexdigest(),
        "caveat": "subset of corpus -> novelty computed against this is an UPPER BOUND",
    }
    (args.out.parent / (args.out.name + ".meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
