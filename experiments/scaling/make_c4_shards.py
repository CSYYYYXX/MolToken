#!/usr/bin/env python3
"""为 C4 留出-loss 评测把 67 个 run 的 out_dir 按 checkpoint 大小做均衡分片。

大模型（4B/1.7B）单个 checkpoint 的 forward 成本远高于小模型，故用 LPT
（longest-processing-time，贪心把最大的先塞进当前最轻的桶）做负载均衡，
使各 worker 墙钟接近，而不是简单 round-robin。

只读 all_fixed_flops_results.csv 与各 run 的 final/pytorch_model.bin 大小，
输出 shard_00.txt ... shard_{N-1}.txt（每行一个 out_dir）。
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path

try:
    from .scaling_paths import DEFAULT_RUN_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_RUN_ROOT


DEFAULT_CSV_PATH = (
    DEFAULT_RUN_ROOT
    / "scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1"
    / "plots"
    / "all_fixed_flops_10curves"
    / "all_fixed_flops_results.csv"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True, help="放 shard_*.txt 的目录")
    ap.add_argument("--out-name", default="holdout_loss.json",
                    help="已存在该文件的 run 视为完成，跳过分片")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    with args.csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            out_dir = Path(row["out_dir"])
            ckpt = out_dir / "final" / "pytorch_model.bin"
            if not ckpt.is_file():
                print(f"WARN missing checkpoint, skipping: {out_dir}")
                continue
            if (out_dir / args.out_name).is_file():
                # 已评过，跳过（幂等）
                continue
            runs.append((ckpt.stat().st_size, str(out_dir)))

    if not runs:
        print("no runs to shard (all done or none found)")
        return 0

    # LPT: 大的先分配到当前累计字节最小的桶
    runs.sort(reverse=True)
    n = min(args.num_shards, len(runs))
    heap = [(0, i) for i in range(n)]  # (累计字节, shard_idx)
    heapq.heapify(heap)
    shards: list[list[str]] = [[] for _ in range(n)]
    shard_bytes = [0] * n
    for size, out_dir in runs:
        load, idx = heapq.heappop(heap)
        shards[idx].append(out_dir)
        shard_bytes[idx] += size
        heapq.heappush(heap, (load + size, idx))

    for i, shard in enumerate(shards):
        p = args.out_dir / f"shard_{i:02d}.txt"
        p.write_text("\n".join(shard) + "\n", encoding="utf-8")
        print(f"shard_{i:02d}: {len(shard)} runs, {shard_bytes[i]/1e9:.2f} GB -> {p}")

    print(f"total {len(runs)} runs into {n} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
