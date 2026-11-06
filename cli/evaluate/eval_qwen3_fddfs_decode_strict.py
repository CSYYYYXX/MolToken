#!/usr/bin/env python3
"""严格评估 FDDFS 预训练模型生成序列能否反序列化成 graph。

这个脚本只关注语法/图反序列化，不跑 RDKit sanitize。每个 domain 固定生成
N 条样本，不因为成功样本数达到目标而提前停止。失败样本也会保存 BPE 展开后
的 FDDFS token 前后缀、括号平衡、ring id 成对闭合情况和 parser 错误原因。
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.serialization.fddfs import deserialize_fddfs  # noqa: E402
from cli.evaluate.generate_qwen3_fddfs_unconditional_eval import (  # noqa: E402
    BPEDecoder,
    DEFAULT_BPE_DIR,
    DEFAULT_CKPT,
    DEFAULT_DATA_ROOT,
    DEFAULT_MERGES,
    DEFAULT_RAW_FDDFS_ROOT,
    DEFAULT_VOCAB,
    DomainStartSampler,
    FDDFSVocab,
    graph_summary,
    json_dumps,
    load_model,
)
from pipeline.inference.generation import (  # noqa: E402
    generate_one,
    model_ids_to_raw_bpe_ids,
)


DEFAULT_RUN_ROOT = Path(str(_moltoken_paths.LOCAL_DATA_ROOT.parent / 'runs'))
DEFAULT_RUN = DEFAULT_RUN_ROOT / "qwen3_0p3b_fddfs_mf80000_onlinepad512_shuffle_pretrain_v1_5gpu_b1024_acc2"
DEFAULT_OUT = DEFAULT_RUN / "generation_fddfs_decode_strict_10perdomain_v1" / "samples.jsonl"
RING_RE = re.compile(r"^R(\d+)$")


def classify_decode_error(error_type: str, message: str) -> str:
    """把 parser 错误归成稳定类别，便于按 domain 汇总。"""

    text = f"{error_type}: {message}"
    lower = text.lower()
    if "unclosed ring" in lower:
        return "unclosed_ring"
    if "expected node token" in lower and "got=)" in lower:
        return "unexpected_branch_close"
    if "missing branch close" in lower:
        return "missing_branch_close"
    if "maximum recursion depth" in lower:
        return "recursion_depth"
    if "unexpected token" in lower:
        return "unexpected_token"
    if "special token" in lower:
        return "special_token_in_generation"
    if "raw bpe id" in lower or "merge map" in lower:
        return "bpe_decode_error"
    return f"{error_type}:{message[:80]}"


def ring_stats(tokens: list[str]) -> dict[str, Any]:
    """统计 FDDFS ring id 是否成对闭合。

    对当前 parser 来说，同一个 ``R<num>`` 第 1/3/5 次出现视为 open，
    第 2/4/6 次出现视为 close，因此每个 ring id 出现偶数次就是闭合。
    """

    counts: Counter[str] = Counter()
    active: set[str] = set()
    max_active = 0
    for token in tokens:
        if not RING_RE.match(token):
            continue
        counts[token] += 1
        if token in active:
            active.remove(token)
        else:
            active.add(token)
            max_active = max(max_active, len(active))
    odd = {token: count for token, count in sorted(counts.items()) if count % 2}
    return {
        "ring_tokens": int(sum(counts.values())),
        "distinct_ring_ids": int(len(counts)),
        "max_active_ring_ids_by_toggle": int(max_active),
        "ring_even_closed": not odd,
        "odd_ring_ids": odd,
    }


def paren_stats(tokens: list[str]) -> dict[str, Any]:
    """统计 DFS 分支括号是否平衡。"""

    balance = 0
    min_balance = 0
    opens = 0
    closes = 0
    first_negative_pos: int | None = None
    for pos, token in enumerate(tokens):
        if token == "(":
            opens += 1
            balance += 1
        elif token == ")":
            closes += 1
            balance -= 1
            if balance < 0 and first_negative_pos is None:
                first_negative_pos = pos
        min_balance = min(min_balance, balance)
    return {
        "open_parens": int(opens),
        "close_parens": int(closes),
        "final_balance": int(balance),
        "min_balance": int(min_balance),
        "balanced": bool(balance == 0 and min_balance >= 0),
        "first_negative_pos": first_negative_pos,
    }


def summarize_sample_tokens(tokens: list[str], preview: int) -> dict[str, Any]:
    """返回一条生成样本的结构统计。"""

    node_tokens = sum(1 for token in tokens if token.startswith("N:"))
    edge_tokens = sum(1 for token in tokens if token.startswith("E:"))
    return {
        "fddfs_token_len": int(len(tokens)),
        "node_tokens": int(node_tokens),
        "edge_tokens": int(edge_tokens),
        "prefix": tokens[:preview],
        "suffix": tokens[-preview:] if preview > 0 else [],
        "ring": ring_stats(tokens),
        "paren": paren_stats(tokens),
    }


def main() -> int:
    """CLI 入口。"""

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))
    ap = argparse.ArgumentParser(description="Strict FDDFS generation graph-decode evaluation.")
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--merges-csv", type=Path, default=DEFAULT_MERGES)
    ap.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--raw-fddfs-root", type=Path, default=DEFAULT_RAW_FDDFS_ROOT)
    ap.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--domains", nargs="+", default=["small_molecule", "peptide", "protein", "dna", "rna"])
    ap.add_argument("--samples-per-domain", type=int, default=10)
    ap.add_argument("--max-total-tokens", type=int, default=256)
    ap.add_argument("--min-total-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--token-offset", type=int, default=3)
    ap.add_argument("--split", default="train")
    ap.add_argument("--domain-start-pool-size", type=int, default=64)
    ap.add_argument("--domain-start-max-files", type=int, default=64)
    ap.add_argument("--domain-start-records-per-file", type=int, default=8)
    ap.add_argument("--domain-start-max-encode-len", type=int, default=4096)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--preview-tokens", type=int, default=80)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    vocab = FDDFSVocab.load(args.vocab_json)
    decoder = BPEDecoder(args.merges_csv, vocab)
    model, cfg, device = load_model(args.ckpt_dir, args.dtype)
    pad_token_id = int(cfg.pad_token_id)
    bos_token_id = int(cfg.bos_token_id)
    eos_token_id = int(cfg.eos_token_id)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_jsonl.parent / "summary.json"
    run_config = {
        "type": "run_config",
        "ckpt_dir": str(args.ckpt_dir),
        "merges_csv": str(args.merges_csv),
        "vocab_json": str(args.vocab_json),
        "raw_fddfs_root": str(args.raw_fddfs_root),
        "out_jsonl": str(args.out_jsonl),
        "domains": list(args.domains),
        "samples_per_domain": int(args.samples_per_domain),
        "max_total_tokens": int(args.max_total_tokens),
        "min_total_tokens": int(args.min_total_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "dtype": args.dtype,
        "device": str(device),
        "model_vocab_size": int(cfg.vocab_size),
        "bpe_base_vocab_size": int(decoder.base_vocab_size),
        "bpe_max_token_id": int(decoder.max_token_id),
        "fddfs_default_edge": list(vocab.default_edge),
        "data_root": str(DEFAULT_DATA_ROOT),
        "bpe_dir": str(DEFAULT_BPE_DIR),
    }
    print(json_dumps({"event": "start", **run_config}), flush=True)

    domain_stats: dict[str, Counter[str]] = {domain: Counter() for domain in args.domains}
    error_counts: dict[str, Counter[str]] = {domain: Counter() for domain in args.domains}
    started = time.time()
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        out.write(json_dumps(run_config) + "\n")
        for domain in args.domains:
            sampler = DomainStartSampler(
                args.raw_fddfs_root,
                domain,
                args.split,
                decoder,
                token_offset=args.token_offset,
                pool_size=args.domain_start_pool_size,
                max_files=args.domain_start_max_files,
                records_per_file=args.domain_start_records_per_file,
                max_encode_len=args.domain_start_max_encode_len,
                rng=rng,
            )
            for sample_id in range(int(args.samples_per_domain)):
                stats = domain_stats[domain]
                stats["attempts"] += 1
                record: dict[str, Any] = {
                    "type": "sample",
                    "domain": domain,
                    "sample_id": int(sample_id),
                }
                start_token = sampler.sample(rng)
                record["start_token_id"] = int(start_token)
                ids: list[int] | None = None
                raw_ids: list[int] | None = None
                fddfs_tokens: list[str] | None = None
                token_structure_counted = False
                try:
                    ids, hit_eos = generate_one(
                        model,
                        int(start_token),
                        device=device,
                        max_total_tokens=args.max_total_tokens,
                        min_total_tokens=args.min_total_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        pad_token_id=pad_token_id,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                    )
                    stats["generation_ok"] += 1
                    stats["hit_eos"] += int(bool(hit_eos))
                    raw_ids = model_ids_to_raw_bpe_ids(ids, eos_token_id=eos_token_id, token_offset=args.token_offset)
                    base_ids, fddfs_tokens = decoder.decode_raw_ids(raw_ids)
                    stats["bpe_decode_ok"] += 1
                    token_info = summarize_sample_tokens(fddfs_tokens, int(args.preview_tokens))
                    stats["ring_even_closed"] += int(bool(token_info["ring"]["ring_even_closed"]))
                    stats["paren_balanced"] += int(bool(token_info["paren"]["balanced"]))
                    token_structure_counted = True
                    graph = deserialize_fddfs(fddfs_tokens, vocab.default_edge)
                    stats["graph_decode_ok"] += 1
                    gsum = graph_summary(graph)
                    stats["nodes"] += int(gsum["num_nodes"])
                    stats["directed_edges"] += int(gsum["num_directed_edges"])
                    record.update(
                        {
                            "status": "graph_decode_ok",
                            "hit_eos": bool(hit_eos),
                            "model_token_len": int(len(ids)),
                            "raw_bpe_len": int(len(raw_ids)),
                            "base_id_len": int(len(base_ids)),
                            "model_token_ids": ids,
                            "raw_bpe_ids": raw_ids,
                            "token_info": token_info,
                            "graph": gsum,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    error_type = type(exc).__name__
                    error = str(exc)
                    reason = classify_decode_error(error_type, error)
                    error_counts[domain][reason] += 1
                    stats["failed"] += 1
                    record.update(
                        {
                            "status": "failed",
                            "error_type": error_type,
                            "error": error,
                            "failure_reason": reason,
                        }
                    )
                    if ids is not None:
                        record["model_token_len"] = int(len(ids))
                        record["model_token_ids"] = ids
                    if raw_ids is not None:
                        record["raw_bpe_len"] = int(len(raw_ids))
                        record["raw_bpe_ids"] = raw_ids
                    if fddfs_tokens is not None:
                        token_info = summarize_sample_tokens(fddfs_tokens, int(args.preview_tokens))
                        record["token_info"] = token_info
                        if not token_structure_counted:
                            stats["ring_even_closed"] += int(bool(token_info["ring"]["ring_even_closed"]))
                            stats["paren_balanced"] += int(bool(token_info["paren"]["balanced"]))

                out.write(json_dumps(record) + "\n")
                out.flush()
                print(
                    json_dumps(
                        {
                            "event": "sample_done",
                            "domain": domain,
                            "sample_id": sample_id,
                            "status": record["status"],
                            "reason": record.get("failure_reason"),
                            "hit_eos": record.get("hit_eos"),
                            "model_token_len": record.get("model_token_len"),
                            "fddfs_token_len": (record.get("token_info") or {}).get("fddfs_token_len"),
                            "ring_even_closed": ((record.get("token_info") or {}).get("ring") or {}).get("ring_even_closed"),
                            "paren_balanced": ((record.get("token_info") or {}).get("paren") or {}).get("balanced"),
                            "num_nodes": (record.get("graph") or {}).get("num_nodes"),
                        }
                    ),
                    flush=True,
                )

    summary: dict[str, Any] = {
        "event": "done",
        "elapsed_sec": round(time.time() - started, 3),
        "run_config": run_config,
        "domain_stats": {},
        "error_counts": {},
        "out_jsonl": str(args.out_jsonl),
    }
    total = Counter()
    for domain, counter in domain_stats.items():
        total.update(counter)
        attempts = int(counter["attempts"])
        item = {key: int(value) for key, value in counter.items()}
        if attempts:
            item["graph_decode_rate"] = counter["graph_decode_ok"] / attempts
            item["bpe_decode_rate"] = counter["bpe_decode_ok"] / attempts
            item["hit_eos_rate"] = counter["hit_eos"] / attempts
            item["ring_even_closed_rate"] = counter["ring_even_closed"] / attempts
            item["paren_balanced_rate"] = counter["paren_balanced"] / attempts
        if counter["graph_decode_ok"]:
            item["mean_nodes_success"] = counter["nodes"] / counter["graph_decode_ok"]
        summary["domain_stats"][domain] = item
        summary["error_counts"][domain] = error_counts[domain].most_common()

    attempts = int(total["attempts"])
    total_item = {key: int(value) for key, value in total.items()}
    if attempts:
        total_item["graph_decode_rate"] = total["graph_decode_ok"] / attempts
        total_item["bpe_decode_rate"] = total["bpe_decode_ok"] / attempts
        total_item["hit_eos_rate"] = total["hit_eos"] / attempts
        total_item["ring_even_closed_rate"] = total["ring_even_closed"] / attempts
        total_item["paren_balanced_rate"] = total["paren_balanced"] / attempts
    summary["total"] = total_item
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json_dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
