#!/usr/bin/env python3
"""把固定前缀 merge 和继续训练得到的 suffix merge 合并成一个 codebook。

prefix merge 来自原始 codebook 的前 N 行；suffix merge 来自压缩后数据继续训练的
merges.csv。这个脚本只负责按顺序拼接并重写 step，不改变 token id。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="包含固定前缀 merge 的 merges.csv")
    parser.add_argument("--prefix-limit", required=True, type=int, help="取 prefix 的前多少行")
    parser.add_argument("--suffix", required=True, help="压缩后继续训练得到的 merges.csv")
    parser.add_argument("--out", required=True, help="输出合并后的 merges.csv")
    args = parser.parse_args()

    prefix_rows = read_rows(Path(args.prefix))[: args.prefix_limit]
    suffix_rows = read_rows(Path(args.suffix))
    if len(prefix_rows) != args.prefix_limit:
        raise RuntimeError(f"prefix rows={len(prefix_rows)} < prefix-limit={args.prefix_limit}")
    if not prefix_rows and not suffix_rows:
        raise RuntimeError("no rows to write")

    fieldnames = ["step", "left", "right", "new_id", "freq", "real_merges", "positions_seen"]
    out_rows: list[dict[str, str]] = []
    for row in prefix_rows + suffix_rows:
        out = {name: row.get(name, "0") for name in fieldnames}
        out["step"] = str(len(out_rows))
        out_rows.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    tmp_path.replace(out_path)
    print(f"wrote {len(out_rows)} merges to {out_path}")


if __name__ == "__main__":
    main()
