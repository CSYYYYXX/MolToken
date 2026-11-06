#!/usr/bin/env python3
"""CPU-only native BPE timing pilot; never replaces the frozen FDDFS codebook.

Bernoulli 1% and 5% nested samples of every native training shard, with the
original source proportions. All preprocessing validation records are held out.
Namespaced characters are integer symbols, not literal textual tag prefixes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time

import numpy as np
import orjson
import zstandard

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host
from experiments.diagnostics.native_tokenizer_lengths import DOMAINS, summarize, write_json

NAMESPACE = {"small_molecule": "smiles", "peptide": "aa", "protein": "aa",
             "dna": "dna", "rna": "rna"}
# Declare biological alphabets independently of which rare letters land in a
# pilot sample (e.g. pyrrolysine O and selenocysteine U). These are single IDs.
BASE_ALPHABETS = {"aa": "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "dna": "ACGTRYSWKMBDHVN",
                  "rna": "ACGURYSWKMBDHVN"}
HEADER = struct.Struct("<8sIQQi")
MAGIC = b"YTTMIBPE"
LEVELS = ("p01", "p05")


def emit(event, **values):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event,
                          **values)), flush=True)


def header(records, tokens, base_size):
    return HEADER.pack(MAGIC, 1, records, tokens, base_size - 1)


def read_sequences(path):
    with Path(path).open("rb") as f:
        magic, version, records, tokens, max_id = HEADER.unpack(f.read(HEADER.size))
        if magic != MAGIC or version != 1:
            raise ValueError(f"Bad integer sequence header: {path}")
        actual_tokens = 0
        for _ in range(records):
            length, = struct.unpack("<I", f.read(4))
            seq = np.frombuffer(f.read(4 * length), dtype="<i4")
            if len(seq) != length or np.any(seq < 0) or np.any(seq > max_id):
                raise ValueError(f"Bad integer sequence: {path}")
            actual_tokens += length
            yield seq
        if actual_tokens != tokens or f.read(1):
            raise ValueError(f"Integer corpus count mismatch: {path}")


def scan_shard(task):
    entry, scope, index, data_root, output, seed = task
    start = time.monotonic()
    path = Path(data_root) / entry["path"]
    destination = Path(output) / f"{scope}_{index:04d}.jsonl"
    # A separate seeded stream per shard; p01 is a subset of p05 by construction.
    random_values = np.random.default_rng(seed + index).random(entry["records"])
    counts = {key: Counter() for key in (*LEVELS, "valid")}
    characters = set()
    seen = 0
    with path.open("rb") as raw, destination.open("wb") as out:
        with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            with io.BufferedReader(reader) as lines:
                for i, line in enumerate(lines):
                    seen += 1
                    if i >= entry["records"]:
                        raise ValueError(f"More rows than manifest: {path}")
                    value = random_values[i]
                    if scope == "train" and value >= .05:
                        continue
                    row = orjson.loads(line)
                    seq = row["normalized"]
                    if row["domain"] != entry["domain"] or not seq or not seq.isascii():
                        raise ValueError(f"Invalid domain/empty/non-ASCII native input: {path}:{i}")
                    level = "valid" if scope == "valid" else ("p01" if value < .01 else "p05")
                    out.write(orjson.dumps([level, seq]) + b"\n")
                    characters.update(seq)
                    counts[level][len(seq)] += 1
    if seen != entry["records"]:
        raise ValueError(f"Rows differ from manifest: {path}: {seen}/{entry['records']}")
    if scope == "train":
        counts["p05"].update(counts["p01"])
    result = dict(input=entry, scope=scope, index=index, selected_path=str(destination),
                  characters=sorted(characters), histograms=counts, scanned_records=seen,
                  elapsed_seconds=time.monotonic() - start)
    write_json(destination.with_suffix(".meta.json"), result)
    return result


def make_alphabet(rows):
    symbols = sorted({(NAMESPACE[r["input"]["domain"]], c)
                      for r in rows if r["scope"] == "train" for c in r["characters"]}
                     | {(ns, c) for ns, chars in BASE_ALPHABETS.items() for c in chars})
    alphabet = {namespace: {} for namespace in sorted(set(NAMESPACE.values()))}
    for token_id, (namespace, character) in enumerate(symbols):
        alphabet[namespace][character] = token_id
    for row in rows:
        unknown = set(row["characters"]) - alphabet[NAMESPACE[row["input"]["domain"]]].keys()
        if unknown:
            raise ValueError(f"Held-out symbols absent from training alphabet: {row['input']['path']}: {unknown}")
    return alphabet


def materialize_shard(task):
    row, alphabet, output = task
    mapping = alphabet[NAMESPACE[row["input"]["domain"]]]
    base_size = sum(map(len, alphabet.values()))
    # ASCII -> dense integer lookup; unknown symbols fail before this stage.
    table = np.full(128, -1, dtype="<i4")
    for character, token_id in mapping.items():
        table[ord(character)] = token_id
    levels = LEVELS if row["scope"] == "train" else ("valid",)
    paths = {level: Path(output) / f"{level}_{row['index']:04d}.bin" for level in levels}
    files = {level: path.open("wb") for level, path in paths.items()}
    counts = {level: [0, 0] for level in levels}
    try:
        for f in files.values():
            f.write(header(0, 0, base_size))
        with Path(row["selected_path"]).open("rb") as source:
            for line in source:
                level, seq = orjson.loads(line)
                ids = table[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
                if np.any(ids < 0):
                    raise ValueError(f"Unmapped character: {seq}")
                targets = LEVELS if level == "p01" else (level,)
                record = struct.pack("<I", len(ids)) + ids.tobytes()
                for target in targets:
                    files[target].write(record)
                    counts[target][0] += 1
                    counts[target][1] += len(ids)
        for level, f in files.items():
            f.seek(0)
            f.write(header(*counts[level], base_size))
    finally:
        for f in files.values():
            f.close()
    for level in levels:
        hist = Counter({int(k): v for k, v in row["histograms"][level].items()})
        if counts[level] != [sum(hist.values()), sum(k*v for k, v in hist.items())]:
            raise ValueError(f"Materialized counts changed: {row['input']['path']} {level}")
    return dict(index=row["index"], domain=row["input"]["domain"], scope=row["scope"],
                paths={k: str(v) for k, v in paths.items()}, counts=counts)


def concatenate(parts, destination, level, base_size):
    count = sum(p["counts"][level][0] for p in parts)
    tokens = sum(p["counts"][level][1] for p in parts)
    with destination.open("wb") as out:
        out.write(header(count, tokens, base_size))
        for part in sorted(parts, key=lambda p: p["index"]):
            with Path(part["paths"][level]).open("rb") as source:
                source.seek(HEADER.size)
                shutil.copyfileobj(source, out, length=8*1024*1024)
    return dict(records=count, tokens=tokens, path=str(destination))


def prepare(args):
    start = time.monotonic()
    root = args.output
    selected = root / "selected"
    binary = root / "binary_parts"
    selected.mkdir()
    binary.mkdir()
    manifest = json.loads(args.manifest.read_text())
    train = [r for r in manifest["scopes"]["full"] if "/train/" in r["path"]]
    valid = manifest["scopes"]["valid"]
    if not train or not valid or {r["path"] for r in train} & {r["path"] for r in valid}:
        raise ValueError("Training and validation shard sets must be nonempty and disjoint")
    tasks = [(entry, scope, i, str(args.data_root), str(selected), args.seed)
             for i, (scope, entry) in enumerate([*(('train', r) for r in train),
                                                *(('valid', r) for r in valid)])]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_shard, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == len(tasks):
                emit("scan_progress", completed=len(rows), total=len(tasks),
                     elapsed_seconds=time.monotonic()-start)
    scan_seconds = time.monotonic()-start
    rows.sort(key=lambda r: r["index"])
    alphabet = make_alphabet(rows)
    write_json(root / "alphabet.json", alphabet)
    base_size = sum(map(len, alphabet.values()))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(materialize_shard, [(r, alphabet, str(binary)) for r in rows]))
    training = {level: concatenate([p for p in parts if p["scope"] == "train"],
                                   root / f"train_{level}.bin", level, base_size)
                for level in LEVELS}
    validation = {domain: concatenate([p for p in parts if p["scope"] == "valid" and p["domain"] == domain],
                                      root / f"valid_{domain}.bin", "valid", base_size)
                  for domain in DOMAINS}
    histograms = {level: {d: Counter() for d in DOMAINS} for level in (*LEVELS, "valid")}
    for row in rows:
        for level, hist in row["histograms"].items():
            histograms[level][row["input"]["domain"]].update({int(k): v for k, v in hist.items()})
    statistics = {level: {d: summarize(h) for d, h in domains.items()}
                  for level, domains in histograms.items()}
    payload = dict(seed=args.seed, namespace=NAMESPACE, alphabet_size=base_size,
                   final_content_vocab_size=args.vocab_size, special_tokens=3,
                   fractions={"p01": .01, "p05": .05},
                   sampling="Nested Bernoulli record sample in every train shard; no domain rebalancing",
                   source_train_records=sum(r["records"] for r in train),
                   source_valid_records=sum(r["records"] for r in valid),
                   train_shards=len(train), valid_shards=len(valid),
                   training=training, validation=validation, statistics=statistics,
                   scan_seconds=scan_seconds, elapsed_seconds=time.monotonic()-start)
    write_json(root / "preparation.json", payload)
    emit("preparation_complete", seconds=payload["elapsed_seconds"], alphabet_size=base_size,
         training=training)
    return payload


def timed_command(command, directory):
    command = list(map(str, command))
    directory.mkdir(exist_ok=True)
    emit("command_start", directory=str(directory), command=command)
    start = time.monotonic()
    with (directory / "process.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    result = dict(wall_seconds=time.monotonic()-start,
                  peak_rss_gib=usage.ru_maxrss/(1024**2),
                  user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                  measurement="Linux wait4 rusage of the actual child process")
    write_json(directory / "resources.json", result)
    emit("command_complete", directory=str(directory), **result)
    return result


def smoke(binary, root):
    # Reuse the existing exact serial BPE oracle, not production-model weights.
    sys.path.insert(0, str(PIPE / "tools/yttm_int_bpe/tools"))
    from smoke_int_bpe import naive, read_rules, write_sample
    seqs = [[1, 1, 1, 2, 3, 1, 2], [1, 2, 3], [4, 5, 4, 5]] * 5 + [[6], [7]]
    path = root / "smoke.bin"
    write_sample(path, seqs)
    expected_merges, expected_tokens, expected_rules = naive(seqs, 20, 1)
    out = root / "smoke_train"
    timed_command([binary, "--mode", "single", "--input", path, "--out-dir", out,
                   "--num-merges", "20", "--min-frequency", "1", "--threads", "2",
                   "--shards", "2", "--no-rle", "--sort-positions", "--build-mode", "onepass"], out)
    got = json.loads((out / "summary.json").read_text())
    if got["num_merges_performed"] != expected_merges or got["final_tokens"] != expected_tokens or read_rules(out / "merges.csv") != expected_rules:
        raise ValueError("Compact integer BPE differs from serial oracle")
    apply = root / "smoke_apply"
    timed_command([binary, "--mode", "apply-encode", "--input", path, "--out-dir", apply,
                   "--apply-merges", out / "merges.csv", "--num-merges", "20", "--threads", "2",
                   "--shards", "2", "--no-rle", "--sort-positions", "--build-mode", "onepass",
                   "--encoded-output", root / "smoke_encoded.bin", "--encoded-compact-int", "--no-eos"], apply)
    if sum(len(s) for s in read_sequences(root / "smoke_encoded.bin")) != expected_tokens:
        raise ValueError("BPE train/apply mismatch")
    emit("smoke_pass", oracle="existing exact naive BPE", cases=len(seqs))


def check_namespaces(merges, alphabet):
    import csv
    token_namespace = {i: ns for ns, chars in alphabet.items() for i in chars.values()}
    counts = Counter(token_namespace.values())
    with merges.open() as f:
        for row in csv.DictReader(f):
            a, b, new = (int(row[k]) for k in ("left", "right", "new_id"))
            if token_namespace[a] != token_namespace[b] or new in token_namespace:
                raise ValueError(f"Invalid namespace or token ID in {merges}: {row}")
            token_namespace[new] = token_namespace[a]
            counts[token_namespace[new]] += 1
    return dict(counts)


def run(args):
    require_compute_host("native namespaced BPE timing pilot")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    emit("pilot_start", host=socket.gethostname(), cpu_affinity=len(os.sched_getaffinity(0)),
         output=str(args.output))
    binary = args.output / "int_bpe_compact"
    source = PIPE / "tools/yttm_int_bpe/tools/int_bpe_bench.cpp"
    timed_command(["g++", "-std=c++17", "-O3", "-pthread", "-DINT_BPE_COMPACT_NODE16",
                   "-DINT_BPE_COMPACT_POSITION", source, "-o", binary], args.output / "build")
    smoke(binary, args.output)
    prepared = prepare(args)
    alphabet = json.loads((args.output / "alphabet.json").read_text())
    merges = args.vocab_size - prepared["alphabet_size"]
    results = {}
    for level in LEVELS:
        directory = args.output / f"train_{level}"
        resources = timed_command([binary, "--mode", "single", "--input", prepared["training"][level]["path"],
                                  "--out-dir", directory, "--num-merges", merges, "--min-frequency", "1",
                                  "--threads", args.threads, "--shards", args.threads, "--no-rle",
                                  "--sort-positions", "--build-mode", "onepass", "--log-every", "1000"], directory)
        trained = json.loads((directory / "summary.json").read_text())
        if trained["num_merges_performed"] != merges:
            raise ValueError(f"Did not reach requested vocabulary size: {directory}")
        if trained["input_tokens"] != prepared["training"][level]["tokens"]:
            raise ValueError(f"Training corpus token count changed: {directory}")
        by_namespace = check_namespaces(directory / "merges.csv", alphabet)
        results[level] = dict(resources=resources, training=trained, vocabulary_by_namespace=by_namespace,
                              validation={})
        write_json(args.output / "progress.json", results)
        for domain in DOMAINS:
            evaluation = args.output / f"eval_{level}_{domain}"
            encoded = evaluation / "encoded.bin"
            measured = timed_command([binary, "--mode", "apply-encode", "--input", prepared["validation"][domain]["path"],
                                      "--out-dir", evaluation, "--apply-merges", directory / "merges.csv",
                                      "--num-merges", merges, "--threads", "16", "--shards", "16",
                                      "--no-rle", "--sort-positions", "--build-mode", "onepass",
                                      "--encoded-output", encoded, "--encoded-compact-int", "--no-eos"], evaluation)
            hist = Counter(len(seq) for seq in read_sequences(encoded))
            statistics = summarize(hist)
            if statistics["records"] != prepared["validation"][domain]["records"]:
                raise ValueError(f"Validation record count mismatch: {level} {domain}")
            statistics["native_mean"] = prepared["statistics"]["valid"][domain]["mean"]
            statistics["compression_ratio"] = prepared["validation"][domain]["tokens"] / statistics["total"]
            results[level]["validation"][domain] = dict(statistics=statistics, resources=measured,
                                                        histogram=hist)
            write_json(args.output / "progress.json", results)
    a, b = (results[level] for level in LEVELS)
    full_tokens = 25518197661  # Prior measured all-splits native total; conservative train-size proxy.
    projections = {}
    for key in ("wall_seconds", "peak_rss_gib"):
        x1, x2 = (r["training"]["input_tokens"] for r in (a, b))
        y1, y2 = (r["resources"][key] for r in (a, b))
        slope = (y2-y1)/(x2-x1)
        projections[key] = dict(p05_proportional=b["resources"][key]*full_tokens/x2,
                                two_point_linear=y2+slope*(full_tokens-x2))
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                   preparation=prepared, probes=results, projection=projections,
                   full_native_symbols_proxy=full_tokens, elapsed_seconds=time.monotonic()-start,
                   limitations=["Timing/memory extrapolations, not a full-corpus training measurement",
                                "All-splits native total is a slightly conservative proxy for train-only size",
                                "Peak RSS is measured by Linux wait4 on each actual C++ process",
                                "No GPU, no Qwen training, no change to frozen FDDFS tokenizer",
                                "1%/5% tokenizers are pilot artifacts, not formal full-corpus baselines"])
    write_json(args.output / "summary.json", summary)
    emit("pilot_complete", elapsed_seconds=summary["elapsed_seconds"], projection=projections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--vocab-size", type=int, default=17204)
    run(p.parse_args())


if __name__ == "__main__":
    main()
