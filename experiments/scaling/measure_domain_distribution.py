#!/usr/bin/env python3
"""测量 domain_ids 的真实全局分布（跨整个文件均匀抽样，而非只看前 N 条）。

背景：实测发现按「每个 shard 前 300 万条」抽样得到
      {0:46.1%, 1:0.001%, 2:6.5%, 3:47.3%}，且 id=4(rna) 完全不出现，
      与已记录的语料占比（小分子 28.6% / 蛋白 33.5% / rna 30.4% /
      dna 7.3% / 肽 0.26%）严重不符 —— 说明 shard 内部按 domain 分块存放，
      前 N 抽样有偏。

domain 映射（来自 encode_fddfs_bpe_compact_int.py 源码，非猜测）：
      DOMAIN_ORDER = ("small_molecule", "peptide", "protein", "dna", "rna")
      即 small_molecule=0, peptide=1, protein=2, dna=3, rna=4

方法：GPFS 不支持 mmap，用 open+seek+read 按均匀 stride 抽样；
      另外把文件切成若干段分别统计，用来直接观察是否分块。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_RUN_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_DATA_ROOT, DEFAULT_RUN_ROOT


DEFAULT_DOMAIN_DIR = (
    DEFAULT_DATA_ROOT
    / "fddfs_bpe_full_prefix70_64k_mf80000_v1"
    / "encoded_compact_int_prefix70_mf80000_11w_v3_domain_ids"
)
DEFAULT_REPORT_DIR = DEFAULT_RUN_ROOT / "reports"

DOMAIN_ORDER = ("small_molecule", "peptide", "protein", "dna", "rna")
DOMAIN_NAME = {i: n for i, n in enumerate(DOMAIN_ORDER)}

DOCUMENTED = {
    "small_molecule": 28.6,
    "protein": 33.5,
    "rna": 30.4,
    "dna": 7.3,
    "peptide": 0.26,
}

ITEM = 2  # uint16


def read_at(fh, index: int, count: int = 1) -> np.ndarray:
    """从第 index 个 uint16 开始读 count 个。"""
    fh.seek(index * ITEM)
    buf = fh.read(count * ITEM)
    if len(buf) < count * ITEM:
        buf = buf[: (len(buf) // ITEM) * ITEM]
    return np.frombuffer(buf, dtype=np.uint16)


def stride_sample(path: Path, n_points: int, chunk: int = 256) -> tuple[Counter, int]:
    """跨整个文件均匀取 n_points 个位置，每个位置连读 chunk 条。"""
    n_total = path.stat().st_size // ITEM
    if n_total == 0:
        return Counter(), 0
    n_points = max(1, min(n_points, n_total))
    step = max(1, n_total // n_points)
    cnt: Counter = Counter()
    seen = 0
    with path.open("rb") as fh:
        for k in range(n_points):
            idx = k * step
            if idx >= n_total:
                break
            take = min(chunk, n_total - idx)
            arr = read_at(fh, idx, take)
            if arr.size == 0:
                continue
            vals, counts = np.unique(arr, return_counts=True)
            for v, c in zip(vals, counts):
                cnt[int(v)] += int(c)
            seen += int(arr.size)
    return cnt, n_total


def segment_profile(path: Path, segments: int, per_segment: int) -> list[Counter]:
    """把文件切成 segments 段，每段抽 per_segment 条，用来看是否按 domain 分块。"""
    n_total = path.stat().st_size // ITEM
    out: list[Counter] = []
    if n_total == 0:
        return out
    seg_len = n_total // segments
    with path.open("rb") as fh:
        for s in range(segments):
            start = s * seg_len
            take = min(per_segment, max(0, n_total - start))
            if take <= 0:
                out.append(Counter())
                continue
            arr = read_at(fh, start, take)
            cnt: Counter = Counter()
            if arr.size:
                vals, counts = np.unique(arr, return_counts=True)
                for v, c in zip(vals, counts):
                    cnt[int(v)] += int(c)
            out.append(cnt)
    return out


def pct_line(cnt: Counter) -> str:
    tot = sum(cnt.values())
    if not tot:
        return "(empty)"
    parts = []
    for k in sorted(cnt):
        parts.append(f"{DOMAIN_NAME.get(k, f'id{k}')}={100.0 * cnt[k] / tot:.2f}%")
    return "  ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points-per-file", type=int, default=40_000,
                    help="每个文件均匀取多少个抽样点")
    ap.add_argument("--chunk", type=int, default=256,
                    help="每个抽样点连读多少条")
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--per-segment", type=int, default=20_000)
    ap.add_argument("--domain-dir", type=Path, default=DEFAULT_DOMAIN_DIR)
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_REPORT_DIR / "domain_distribution.txt")
    ap.add_argument("--out-json", type=Path,
                    default=DEFAULT_REPORT_DIR / "domain_distribution.json")
    args = ap.parse_args()

    lines: list[str] = []

    def s(t: str = "") -> None:
        lines.append(str(t))

    files = sorted(p for p in args.domain_dir.iterdir()
                   if p.is_file() and p.name.endswith(".domain_ids.u16"))

    s("=" * 76)
    s("DOMAIN ID 映射（来自 encode_fddfs_bpe_compact_int.py 源码）")
    s("=" * 76)
    for i, n in enumerate(DOMAIN_ORDER):
        s(f"  {i} = {n}")
    s()
    s(f"  domain_ids 文件数 = {len(files)}")
    s(f"  抽样方式: 每文件均匀 {args.points_per_file:,} 个点 x 每点 {args.chunk} 条")

    grand: Counter = Counter()
    total_entries = 0
    per_file_rows = []

    for p in files:
        cnt, n_total = stride_sample(p, args.points_per_file, args.chunk)
        grand.update(cnt)
        total_entries += n_total
        per_file_rows.append((p.name, n_total, sum(cnt.values()), dict(cnt)))

    s()
    s("=" * 76)
    s("逐文件（stride 抽样）")
    s("=" * 76)
    for name, n_total, n_samp, cnt in per_file_rows:
        s(f"  {name}")
        s(f"    entries={n_total:,}  sampled={n_samp:,}")
        s(f"    {pct_line(Counter(cnt))}")

    s()
    s("=" * 76)
    s("汇总：真实全局分布 vs 已记录占比")
    s("=" * 76)
    tot = sum(grand.values())
    s(f"  抽样总条数 = {tot:,}   （全库 entries = {total_entries:,}）")
    s(f"  出现的 id  = {sorted(grand)}")
    s()
    s(f"  {'domain':<16}{'id':>4}{'sampled':>14}{'measured':>11}{'documented':>12}{'diff':>9}")
    for i, name in enumerate(DOMAIN_ORDER):
        c = grand.get(i, 0)
        meas = 100.0 * c / tot if tot else 0.0
        doc = DOCUMENTED.get(name)
        diff = f"{meas - doc:+.2f}" if doc is not None else "-"
        docs = f"{doc:.2f}%" if doc is not None else "-"
        s(f"  {name:<16}{i:>4}{c:>14,}{meas:>10.2f}%{docs:>12}{diff:>9}")
    unexpected = [k for k in grand if k not in DOMAIN_NAME]
    if unexpected:
        s(f"  !! 出现未定义 id: {unexpected}")

    s()
    s("=" * 76)
    s(f"分段剖面（每文件切 {args.segments} 段，判断是否按 domain 分块）")
    s("=" * 76)
    blocked_evidence = []
    for p in files[:3]:
        s(f"  --- {p.name} ---")
        segs = segment_profile(p, args.segments, args.per_segment)
        for si, cnt in enumerate(segs):
            s(f"    seg{si:02d}  {pct_line(cnt)}")
        nonempty = [c for c in segs if sum(c.values())]
        if nonempty:
            dom_sets = [max(c, key=c.get) for c in nonempty]
            blocked_evidence.append((p.name, dom_sets))

    s()
    s("=" * 76)
    s("结论")
    s("=" * 76)
    for name, dom_sets in blocked_evidence:
        names = [DOMAIN_NAME.get(d, f"id{d}") for d in dom_sets]
        distinct = len(set(dom_sets))
        s(f"  {name}: 各段主导 domain = {names}")
        s(f"     不同主导 domain 段数 = {distinct}/{len(dom_sets)}")
    s()
    s("  若各段主导 domain 不同 => shard 内按 domain 分块，")
    s("  则「取前 N 条」会严重偏向排在前面的 domain，")
    s("  参考集必须全量扫描或跨整个 index 均匀 stride 抽样。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps({
        "domain_order": list(DOMAIN_ORDER),
        "sampled_counts": {DOMAIN_NAME.get(k, f"id{k}"): v for k, v in sorted(grand.items())},
        "sampled_total": tot,
        "corpus_entries_total": total_entries,
        "measured_pct": {DOMAIN_NAME.get(k, f"id{k}"): (100.0 * v / tot if tot else None)
                         for k, v in sorted(grand.items())},
        "documented_pct": DOCUMENTED,
        "per_file": [{"file": n, "entries": e, "sampled": sm, "counts": c}
                     for n, e, sm, c in per_file_rows],
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
