#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "int_bpe_bench"
MAGIC = b"YTTMIBPE"


def write_sample(path: Path, seqs: list[list[int]]) -> None:
    max_token = max(max(s) for s in seqs if s)
    total = sum(len(s) for s in seqs)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IQQi", 1, len(seqs), total, max_token))
        for seq in seqs:
            f.write(struct.pack("<I", len(seq)))
            f.write(struct.pack("<" + "i" * len(seq), *seq))


def naive(seqs: list[list[int]], num_merges: int, min_frequency: int):
    seqs = [list(s) for s in seqs]
    next_id = max(max(s) for s in seqs if s) + 1
    merges = 0
    rules = []
    for _ in range(num_merges):
        counts = Counter()
        for seq in seqs:
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += 1
        if not counts:
            break
        (left, right), freq = max(counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
        if freq < min_frequency:
            break
        real_merges = 0
        for i, seq in enumerate(seqs):
            out = []
            j = 0
            while j < len(seq):
                if j + 1 < len(seq) and seq[j] == left and seq[j + 1] == right:
                    out.append(next_id)
                    j += 2
                    real_merges += 1
                else:
                    out.append(seq[j])
                    j += 1
            seqs[i] = out
        rules.append((merges, left, right, next_id, freq, real_merges))
        next_id += 1
        merges += 1
    return merges, sum(len(s) for s in seqs), rules


def read_rules(path: Path):
    rules = []
    for line in path.read_text().splitlines()[1:]:
        if not line:
            continue
        step, left, right, new_id, freq, real_merges, _positions_seen = map(int, line.split(","))
        rules.append((step, left, right, new_id, freq, real_merges))
    return rules


def main() -> int:
    random.seed(7)
    cases = []
    random_case = []
    for _ in range(120):
        random_case.append([random.randrange(1, 9) for _ in range(random.randrange(4, 24))])
    random_case += [[1, 2, 1, 2, 1, 2, 3]] * 30
    cases.append(("random_motif", random_case, 80, 2, 4))
    cases.append(("rle_runs", [[1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2, 2], [1, 2, 2, 2, 1]], 12, 1, 3))
    cases.append(("coalesce", [[1, 2, 1], [1, 2, 1, 2, 1], [3, 1, 2, 1, 3]] * 4, 20, 1, 2))
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for name, seqs, num_merges, min_frequency, threads in cases:
            expected_merges, expected_tokens, expected_rules = naive(seqs, num_merges, min_frequency)
            sample = td_path / f"{name}.bin"
            out_dir = td_path / f"out_{name}"
            write_sample(sample, seqs)
            variants = [
                ("plain_sort", False, True, threads, "legacy-noreserve"),
                ("plain_nosort", False, False, threads, "legacy-noreserve"),
                ("plain_base_bucket", False, False, threads, "base-bucket"),
                ("plain_overshard", False, True, threads * 3, "legacy-noreserve"),
                ("rle_sort", True, True, threads, "legacy-noreserve"),
                ("rle_nosort", True, False, threads, "legacy-noreserve"),
            ]
            for variant, use_rle, sort_positions, shards, build_mode in variants:
                run_name = f"{name}_{variant}"
                run_out = td_path / f"out_{run_name}"
                cmd = [
                        str(BIN),
                        "--mode",
                        "single",
                        "--input",
                        str(sample),
                        "--out-dir",
                        str(run_out),
                        "--num-merges",
                        str(num_merges),
                        "--min-frequency",
                        str(min_frequency),
                        "--threads",
                        str(threads),
                        "--shards",
                        str(shards),
                        "--log-every",
                        "1000",
                        "--build-mode",
                        build_mode,
                    ]
                cmd.append("--rle" if use_rle else "--no-rle")
                if sort_positions:
                    cmd.append("--sort-positions")
                else:
                    cmd.append("--no-sort-positions")
                subprocess.run(cmd, check=True)
                summary = json.loads((run_out / "summary.json").read_text())
                assert summary["pq_min_frequency_filter"] is True, (run_name, summary)
                assert summary["count_wire_chunk_records"] == 65536, (run_name, summary)
                rules = read_rules(run_out / "merges.csv")
                assert summary["num_merges_performed"] == expected_merges, (run_name, summary, expected_merges)
                assert summary["final_tokens"] == expected_tokens, (run_name, summary, expected_tokens)
                assert rules == expected_rules, (run_name, rules[:10], expected_rules[:10])
            print("CASE_OK", name, expected_merges, expected_tokens)
    print("SMOKE_OK", len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
