#!/usr/bin/env python3
"""Historical first-token-seeded sampling from a foundation checkpoint.

Start tokens are sampled from the small-molecule portion of indexed compact
training shards, using domain sidecars rather than assuming a record order.
This preserves the historical BOS + first-token prompting protocol. It is not
the official MOSES/GuacaMol SFT evaluator. Output contains raw BPE IDs; decoding
and chemistry scoring are separate operations in score_generation_vun.py.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PIPELINE = Path(__file__).resolve().parents[2]
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

DATA = Path(str(_moltoken_paths.LOCAL_DATA_ROOT / ''))
BPE = DATA / "fddfs_bpe_full_prefix70_64k_mf80000_v1"
SHARD_DIR = BPE / "encoded_compact_int_prefix70_mf80000_11w_v3"
IDX_DIR = BPE / "encoded_compact_int_prefix70_mf80000_11w_v3_index"
DOM_DIR = BPE / "encoded_compact_int_prefix70_mf80000_11w_v3_domain_ids"

SMALL_MOLECULE_DOMAIN_ID = 0
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MAX_RAW_TOKEN_ID = 17203


def collect_start_tokens(n_want: int, seed: int) -> tuple[list[int], dict]:
    """从真实小分子记录取首个 BPE token（原始 id，未加 offset）。"""
    rng = np.random.default_rng(seed)
    shards = sorted(p.name for p in SHARD_DIR.glob("shard_*.bin"))
    if not shards:
        raise FileNotFoundError(f"no shards under {SHARD_DIR}")

    out: list[int] = []
    stats = {"shards_used": 0, "records_read": 0, "short_read": 0, "out_of_range": 0}
    per_shard = max(1, n_want // max(1, len(shards)) + 1)

    for name in shards:
        lengths = np.frombuffer((IDX_DIR / f"{name}.lengths.u32").read_bytes(),
                                dtype=np.uint32)
        offsets = np.frombuffer((IDX_DIR / f"{name}.payload_offsets.u64").read_bytes(),
                                dtype=np.uint64)
        domains = np.frombuffer((DOM_DIR / f"{name}.domain_ids.u16").read_bytes(),
                                dtype=np.uint16)
        n = min(len(lengths), len(offsets), len(domains))
        sm_idx = np.where(domains[:n] == SMALL_MOLECULE_DOMAIN_ID)[0]
        if sm_idx.size == 0:
            continue
        take = min(per_shard, sm_idx.size)
        pick = rng.choice(sm_idx, size=take, replace=False)
        stats["shards_used"] += 1
        with (SHARD_DIR / name).open("rb") as fh:
            for i in pick.tolist():
                ln = int(lengths[i])
                if ln <= 0:
                    continue
                fh.seek(int(offsets[i]))
                seg = fh.read(4)          # 只要首个 uint32
                if len(seg) != 4:
                    stats["short_read"] += 1
                    continue
                tok = int(np.frombuffer(seg, dtype=np.uint32)[0])
                if tok > MAX_RAW_TOKEN_ID:
                    stats["out_of_range"] += 1
                    continue
                out.append(tok)
                stats["records_read"] += 1
        if len(out) >= n_want:
            break

    if not out:
        raise RuntimeError("collected zero start tokens")
    return out, stats


def load_model(ckpt_dir: Path, device: torch.device, dtype: str):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    cfg = Qwen3Config.from_pretrained(ckpt_dir)
    model = Qwen3ForCausalLM.from_pretrained(ckpt_dir, config=cfg,
                                             torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def sample_batch(model, start_tokens: list[int], token_offset: int,
                 max_new: int, temperature: float, top_k: int, top_p: float,
                 device: torch.device) -> list[list[int]]:
    """返回原始 BPE id 序列（已去掉 BOS / 起始 token 的 offset）。"""
    b = len(start_tokens)
    prompt = torch.tensor(
        [[BOS_TOKEN_ID, t + token_offset] for t in start_tokens],
        dtype=torch.long, device=device,
    )
    out = model.generate(
        input_ids=prompt,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        use_cache=True,
    )
    seqs: list[list[int]] = []
    for r in range(b):
        row = out[r].tolist()
        body = row[1:]                       # 去掉 BOS
        cut: list[int] = []
        for t in body:
            if t in (EOS_TOKEN_ID, PAD_TOKEN_ID):
                break
            cut.append(t)
        # 还原成原始 BPE id
        raw = [t - token_offset for t in cut
               if t >= token_offset and (t - token_offset) <= MAX_RAW_TOKEN_ID]
        seqs.append(raw)
    return seqs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--token-offset", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda_count={torch.cuda.device_count()}", flush=True)

    t0 = time.time()
    starts, st_stats = collect_start_tokens(args.n_samples, args.seed)
    print(f"start tokens: {len(starts)} collected, distinct={len(set(starts))}, "
          f"stats={st_stats}  {time.time() - t0:.1f}s", flush=True)

    model, cfg = load_model(args.ckpt_dir, device, args.dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.tag}  params={n_params / 1e6:.1f}M  "
          f"vocab={cfg.vocab_size} hidden={cfg.hidden_size} "
          f"layers={cfg.num_hidden_layers}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_empty = 0
    lens: list[int] = []
    t1 = time.time()
    with args.out.open("w", encoding="utf-8") as fout:
        while n_done < args.n_samples:
            b = min(args.batch_size, args.n_samples - n_done)
            batch_starts = [starts[(n_done + j) % len(starts)] for j in range(b)]
            seqs = sample_batch(model, batch_starts, args.token_offset,
                                args.max_new_tokens, args.temperature,
                                args.top_k, args.top_p, device)
            for j, seq in enumerate(seqs):
                if not seq:
                    n_empty += 1
                lens.append(len(seq))
                fout.write(json.dumps({
                    "idx": n_done + j,
                    "start_token": batch_starts[j],
                    "bpe_ids": seq,
                }) + "\n")
            n_done += b
            if n_done % (args.batch_size * 4) == 0 or n_done >= args.n_samples:
                el = time.time() - t1
                print(f"  {n_done}/{args.n_samples}  {el:.0f}s  "
                      f"{n_done / max(1e-6, el):.1f} samp/s  empty={n_empty}",
                      flush=True)

    arr = np.asarray(lens) if lens else np.zeros(1)
    meta = {
        "tag": args.tag,
        "ckpt_dir": str(args.ckpt_dir),
        "n_samples": n_done,
        "n_empty": n_empty,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "token_offset": args.token_offset,
        "seed": args.seed,
        "dtype": args.dtype,
        "model_params": n_params,
        "vocab_size": cfg.vocab_size,
        "start_tokens_collected": len(starts),
        "start_tokens_distinct": len(set(starts)),
        "start_token_stats": st_stats,
        "gen_len_mean": float(arr.mean()),
        "gen_len_median": float(np.median(arr)),
        "gen_len_p95": float(np.percentile(arr, 95)),
        "gen_len_max": int(arr.max()),
        "elapsed_sec": time.time() - t0,
        "note": ("start token taken from real small_molecule records to pin the "
                 "modality; NOT conditional generation"),
    }
    (args.out.parent / (args.out.name + ".meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
