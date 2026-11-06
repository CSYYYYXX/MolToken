#!/usr/bin/env python3
"""Native strings versus published Qwen tokens; do not reconstruct graphs/BPE.

All corpus reads execute on a compute worker. Existing FDDFS/BPE numbers are
copied as historical references, never treated as paired with a new cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from pipeline.runtime_host import require_compute_host

DOMAINS = ("small_molecule", "peptide", "protein", "dna", "rna")
REFERENCE_NAMES = ("compression_by_domain.csv", "length_distribution_existing.csv")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def relative_data_path(value: str) -> str:
    # Historical manifests use either of the two cluster mount aliases.
    return value.split("/molscaling/data/", 1)[1]


def quantile(hist: Counter, q: float) -> float:
    """Exact linear / type-7 quantile, without expanding the histogram."""
    n = sum(hist.values())
    if not n:
        raise ValueError("Cannot compute a quantile of zero records")
    rank = (n - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    before = 0
    low_value = None
    for value, count in sorted(hist.items()):
        after = before + count
        if low_value is None and lo < after:
            low_value = value
        if hi < after:
            return float(low_value + (value - low_value) * (rank - lo))
        before = after
    raise RuntimeError("Invalid histogram counts")


def summarize(hist: Counter) -> dict:
    n = sum(hist.values())
    total = sum(k * v for k, v in hist.items())
    return dict(records=n, total=total, mean=total / n,
                median=quantile(hist, .5), p90=quantile(hist, .9),
                p95=quantile(hist, .95), p99=quantile(hist, .99),
                minimum=min(hist), maximum=max(hist),
                over_511=sum(v for k, v in hist.items() if k > 511),
                over_512=sum(v for k, v in hist.items() if k > 512))


def prepare(args: argparse.Namespace) -> None:
    require_compute_host("native tokenizer input preparation")
    root = args.data_root
    clean = json.loads((root / "reports/clean_dedup_sharded_clean_normv2.latest.json").read_text())
    shards = json.loads((root / "manifests/dataset_manifest_normv2_all.json").read_text())
    full = [dict(domain=r["domain"], path=relative_data_path(r["output"]),
                 records=r["records_kept"]) for r in clean["inputs"]]
    valid = [dict(domain=r["domain"], path=relative_data_path(o["path"]),
                  records=o["records"])
             for r in shards["inputs"] for o in r["outputs"] if o["split"] == "valid"]
    for files in (full, valid):
        if len({r["path"] for r in files}) != len(files):
            raise ValueError("Duplicate input paths in the source manifest")
        for entry in files:
            entry["compressed_bytes"] = (root / entry["path"]).stat().st_size
    with (args.references / "compression_by_domain.csv").open() as handle:
        old = {r["domain"].lower().replace(" ", "_"): r for r in csv.DictReader(handle)
               if r["domain"] != "overall"}
    cohorts = {}
    for domain in DOMAINS:
        before = sum(r["records"] for r in full if r["domain"] == domain)
        after = int(old[domain]["num_graphs"])
        cohorts[domain] = dict(clean_records=before, historical_graph_records=after,
                              difference=before - after, difference_fraction=(before - after) / before,
                              valid_records=sum(r["records"] for r in valid if r["domain"] == domain))
    args.output.mkdir(parents=True, exist_ok=False)
    references = args.output / "references"
    references.mkdir()
    for name in REFERENCE_NAMES:
        (references / name).write_bytes((args.references / name).read_bytes())
    # This previously published small report contains overall BPE quantiles,
    # which were left blank in the later per-domain CSV. Preserve separately.
    name = "fddfs_mf80000_encoded_length_stats_v1.json"
    (references / name).write_bytes((root / "reports" / name).read_bytes())
    payload = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        native_field="normalized", data_root=str(root), tokenizer_dir=str(args.tokenizer_dir),
        native_unit={d: "SMILES characters" if d == "small_molecule" else
                     ("amino-acid residues" if d in ("peptide", "protein") else "nucleotides")
                     for d in DOMAINS},
        scopes={"full": full, "valid": valid}, cohorts=cohorts,
        references_source=str(args.references),
        comparison_note="Native/Qwen lengths are paired. Historical graph/BPE statistics use the full successful-graph cohort, not this exact new cohort. No paired Qwen/BPE ratio is reported.",
        protocol=dict(truncation=False, padding=False, add_special_tokens=False,
                      chat_template=False, recleaning=False, graph_encoding=False,
                      quantile_method="linear/type-7", valid_split="preprocessing valid, not a downstream test or model-selected sample"))
    write_json(args.output / "inputs.json", payload)
    # Reuse existing native shards for a 64-core full pass. The 19 clean files
    # alone leave a serial tail on UniRef50/RNAcentral, regardless of CPU count.
    full_shards = [dict(domain=r["domain"], path=relative_data_path(o["path"]),
                        records=o["records"], compressed_bytes=o["compressed_bytes"])
                   for r in shards["inputs"] for o in r["outputs"]]
    before, after = Counter(), Counter()
    for r in full:
        before[r["domain"]] += r["records"]
    for r in full_shards:
        after[r["domain"]] += r["records"]
        if not (root / r["path"]).is_file():
            raise FileNotFoundError(root / r["path"])
    if before != after or len({r["path"] for r in full_shards}) != len(full_shards):
        raise ValueError("Native full shard population differs from clean corpus")
    sharded = {**payload, "scopes": {**payload["scopes"], "full": full_shards},
               "full_input_note": "Existing native shards, all preprocessing splits; same clean cohort; graph data not read."}
    write_json(args.output / "inputs_full_sharded.json", sharded)
    print(json.dumps({"prepared": str(args.output), "cohorts": cohorts}, indent=2), flush=True)


def iter_rows(path: Path):
    import orjson
    import zstandard
    with path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text:
                for line in text:
                    yield orjson.loads(line)


@lru_cache(maxsize=1)
def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    # Apply tokenizer_config.json too: it adds four tokens absent from the
    # standalone tokenizer.json. Use the official local loader, not just BPE.
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_fast=True,
    ).backend_tokenizer
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def measure_file(task: tuple) -> dict:
    entry, data_root, tokenizer_dir, batch_size, limit, output = task
    require_compute_host("native sequence length measurement")
    start = time.monotonic()
    tokenizer = load_tokenizer(tokenizer_dir)
    native, qwen = Counter(), Counter()
    examples, pending = [], []
    seen = 0

    def consume() -> None:
        sequences = [r["normalized"] for r in pending]
        encoded = tokenizer.encode_batch(sequences, add_special_tokens=False)
        for row, seq, enc in zip(pending, sequences, encoded, strict=True):
            native[len(seq)] += 1
            qwen[len(enc.ids)] += 1
            if len(examples) < 2:
                if tokenizer.decode(enc.ids, skip_special_tokens=False) != seq:
                    raise ValueError(f"Qwen text round-trip failed: {entry['path']} {row['id']}")
                examples.append(dict(id=row["id"], source=row["source"],
                                     native_prefix=seq[:100], native_length=len(seq),
                                     qwen_length=len(enc.ids), roundtrip_ok=True))
        pending.clear()

    records = iter_rows(Path(data_root) / entry["path"])
    try:
        for row in records:
            if row["domain"] != entry["domain"]:
                raise ValueError(f"Domain mismatch in {entry['path']}: {row['domain']}")
            seq = row["normalized"]
            if not isinstance(seq, str) or not seq:
                raise ValueError(f"Missing native string: {entry['path']} {row['id']}")
            pending.append(row)
            seen += 1
            if len(pending) >= batch_size:
                consume()
            if limit and seen >= limit:
                break
        if pending:
            consume()
    finally:
        records.close()
    expected = min(limit, entry["records"]) if limit else entry["records"]
    if seen != expected:
        raise ValueError(f"Record count mismatch: {entry['path']}: read {seen}, expected {expected}")
    result = dict(input=entry, records=seen, elapsed_seconds=time.monotonic() - start,
                  native_histogram=sorted(native.items()), qwen_histogram=sorted(qwen.items()),
                  native=summarize(native), qwen=summarize(qwen), examples=examples)
    write_json(Path(output), result)
    return result


def measure(args: argparse.Namespace) -> None:
    host = require_compute_host("native/Qwen token statistics")
    import importlib.metadata
    import tokenizers
    manifest = json.loads(args.manifest.read_text())
    data_root = str(args.data_root or manifest["data_root"])
    tokenizer_dir = str(args.tokenizer_dir or manifest["tokenizer_dir"])
    entries = manifest["scopes"][args.scope]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "files").mkdir()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    tasks = [(r, data_root, tokenizer_dir, args.batch_size, args.limit_per_file,
              str(args.output / "files" / f"{i:03d}_{r['domain']}.json"))
             for i, r in enumerate(entries)]
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(measure_file, task) for task in tasks]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(json.dumps({"completed_file": r["input"]["path"], "records": r["records"],
                              "seconds": r["elapsed_seconds"]}), flush=True)
    groups = defaultdict(lambda: {"native": Counter(), "qwen": Counter()})
    for r in results:
        for domain in (r["input"]["domain"], "overall"):
            for kind in ("native", "qwen"):
                groups[domain][kind].update(dict(r[kind + "_histogram"]))
    rows = []
    for domain in (*DOMAINS, "overall"):
        values = groups[domain]
        chars, tokens = summarize(values["native"]), summarize(values["qwen"])
        for kind, stats in (("native", chars), ("qwen", tokens)):
            rows.append(dict(domain=domain, representation=kind, **stats,
                             chars_per_qwen_token=chars["total"] / tokens["total"]))
    with (args.output / "lengths.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = json.loads((Path(tokenizer_dir) / "repository_metadata.json").read_text())
    write_json(args.output / "summary.json", dict(
        complete=True, smoke_only=bool(args.limit_per_file), scope=args.scope,
        limit_per_file=args.limit_per_file, manifest=str(args.manifest), host=host,
        elapsed_seconds=time.monotonic() - start, workers=args.workers, batch_size=args.batch_size,
        tokenizer={"repository": metadata["id"], "revision": metadata["sha"],
                   "runtime_version": tokenizers.__version__,
                   "vocab_with_added_tokens": load_tokenizer(tokenizer_dir).get_vocab_size(),
                   "weights_loaded": False},
        zstandard_version=importlib.metadata.version("zstandard"),
        comparison_note=manifest["comparison_note"], protocol=manifest["protocol"],
        rows=rows))
    print(f"NATIVE_QWEN_LENGTHS_COMPLETE output={args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--tokenizer-dir", type=Path, required=True)
    prep.add_argument("--references", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=prepare)
    run = commands.add_parser("measure")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--data-root", type=Path)
    run.add_argument("--tokenizer-dir", type=Path)
    run.add_argument("--scope", choices=("valid", "full"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--limit-per-file", type=int, default=0,
                     help="Positive limit is a prefix smoke ONLY, never a random sample")
    run.set_defaults(func=measure)
    args = parser.parse_args()
    if args.command == "measure" and (args.workers < 1 or args.batch_size < 1 or args.limit_per_file < 0):
        parser.error("workers/batch-size must be positive and limit-per-file nonnegative")
    args.func(args)


if __name__ == "__main__":
    main()
