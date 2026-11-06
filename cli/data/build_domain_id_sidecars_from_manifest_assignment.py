#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import sys
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DOMAIN_NAMES = "small_molecule,peptide,protein,dna,rna"


def parse_domain_names(raw: str) -> list[str]:
    names = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not names:
        raise ValueError("--domain-names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"--domain-names contains duplicates: {raw}")
    return names


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def infer_summary_path(token_path: Path, out_dir: Path, serializer: str) -> Path:
    token_root = out_dir / "tokens"
    summary_root = out_dir / "summaries"
    rel = token_path.relative_to(token_root)
    suffix = f".{serializer}.bin"
    if not rel.name.endswith(suffix):
        raise ValueError(f"unexpected token filename for serializer={serializer}: {token_path}")
    return summary_root / rel.with_name(rel.name[: -len(suffix)] + ".json")


def load_file_items(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    out_dir = Path(str(manifest["out_dir"]))
    raw_serializer = manifest.get("serializer")
    if not isinstance(raw_serializer, str) or not raw_serializer.strip():
        raise ValueError(f"manifest has no explicit serializer: {manifest_path}")
    serializer = raw_serializer.strip().lower()
    if serializer != "fddfs":
        raise ValueError(
            f"current domain-sidecar builder accepts only FDDFS manifests, got "
            f"serializer={raw_serializer!r}: {manifest_path}"
        )
    raw_files = list(manifest.get("token_files") or [])
    if not raw_files:
        raise ValueError(f"manifest has no token_files: {manifest_path}")

    items: list[dict[str, Any]] = []
    start = time.time()
    for idx, raw in enumerate(raw_files, 1):
        token_path = Path(str(raw))
        summary_path = infer_summary_path(token_path, out_dir, serializer)
        summary = read_json(summary_path)
        items.append(
            {
                "path": str(token_path),
                "summary": str(summary_path),
                "domain": str(summary.get("domain") or "unknown"),
                "records": int(summary.get("records") or 0),
                "tokens": int(summary.get("tokens") or 0),
            }
        )
        if idx % 10000 == 0:
            print(
                f"[domain-sidecar] read summaries {idx}/{len(raw_files)} elapsed_sec={time.time() - start:.1f}",
                flush=True,
            )
    return items


def select_files(items: list[dict[str, Any]], token_budget: int, seed: int) -> list[dict[str, Any]]:
    files = list(items)
    rng = random.Random(seed)
    rng.shuffle(files)
    if token_budget <= 0:
        return files

    selected: list[dict[str, Any]] = []
    selected_tokens = 0
    for item in files:
        selected.append(item)
        selected_tokens += int(item["tokens"])
        if selected_tokens >= int(token_budget):
            break
    return selected


def assign_files(files: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    loads = [(0, i) for i in range(num_shards)]
    heapq.heapify(loads)
    assigned: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    assigned_tokens = [0 for _ in range(num_shards)]
    for item in sorted(files, key=lambda x: int(x["tokens"]), reverse=True):
        load, shard = heapq.heappop(loads)
        assigned[shard].append(item)
        tokens = int(item["tokens"])
        assigned_tokens[shard] += tokens
        heapq.heappush(loads, (load + tokens, shard))
    for shard_files in assigned:
        shard_files.sort(key=lambda x: str(x["path"]))
    return assigned


def write_repeated_u16(out, value: int, count: int, chunk_records: int) -> None:
    remaining = int(count)
    while remaining > 0:
        n = min(remaining, int(chunk_records))
        arr = array("H", [int(value)]) * n
        if sys.byteorder != "little":
            arr.byteswap()
        arr.tofile(out)
        remaining -= n


def output_path_for(out_dir: Path, shard_path: Path) -> Path:
    return out_dir / f"{shard_path.name}.domain_ids.u16"


def load_generation_config(raw_shard_dir: Path, seed: int | None, token_budget: int | None, num_shards: int | None) -> dict[str, Any]:
    rank0_path = raw_shard_dir / "rank_00000.done.json"
    if not rank0_path.exists():
        if seed is None or token_budget is None or num_shards is None:
            raise FileNotFoundError(
                f"missing {rank0_path}; pass --seed, --token-budget, and --num-shards explicitly"
            )
        return {"seed": int(seed), "token_budget": int(token_budget), "num_shards": int(num_shards)}
    rank0 = read_json(rank0_path)
    return {
        "seed": int(rank0["seed"] if seed is None else seed),
        "token_budget": int(rank0["token_budget"] if token_budget is None else token_budget),
        "num_shards": int(rank0["num_shards"] if num_shards is None else num_shards),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build per-record domain id sidecars for encoded compact-int shards by replaying "
            "the deterministic manifest-to-raw-shard assignment."
        )
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--raw-shard-dir", type=Path, required=True)
    ap.add_argument("--encoded-done-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--domain-names", default=DEFAULT_DOMAIN_NAMES)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--chunk-records", type=int, default=1_000_000)
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    domain_names = parse_domain_names(args.domain_names)
    domain_to_id = {name: i for i, name in enumerate(domain_names)}
    cfg = load_generation_config(args.raw_shard_dir, args.seed, args.token_budget, args.num_shards)
    final_done = read_json(args.encoded_done_json)
    final_shards = sorted(final_done.get("shards") or [], key=lambda x: int(x["rank"]))
    if len(final_shards) != int(cfg["num_shards"]):
        raise ValueError(f"final shard count mismatch: {len(final_shards)} vs {cfg['num_shards']}")

    print(
        "[domain-sidecar] config "
        f"num_shards={cfg['num_shards']} seed={cfg['seed']} token_budget={cfg['token_budget']} "
        f"domains={domain_names}",
        flush=True,
    )
    items = load_file_items(args.manifest)
    selected = select_files(items, int(cfg["token_budget"]), int(cfg["seed"]))
    assigned = assign_files(selected, int(cfg["num_shards"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    total_records = 0
    total_domain_records: Counter[str] = Counter()
    start = time.time()
    for shard in final_shards:
        rank = int(shard["rank"])
        shard_path = Path(str(shard["path"]))
        expected_records = int(shard["records"])
        shard_files = assigned[rank]
        assigned_records = sum(int(item["records"]) for item in shard_files)
        if assigned_records != expected_records:
            raise RuntimeError(
                f"record mismatch rank={rank}: assigned_records={assigned_records} final_records={expected_records}"
            )

        domain_records: Counter[str] = Counter()
        out_path = output_path_for(args.out_dir, shard_path)
        if out_path.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"sidecar exists; pass --overwrite to replace: {out_path}")
        if args.dry_run:
            for item in shard_files:
                domain_records[str(item["domain"])] += int(item["records"])
        else:
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp_path.open("wb") as out:
                for item in shard_files:
                    domain = str(item["domain"])
                    if domain not in domain_to_id:
                        raise ValueError(f"unknown domain {domain!r}; domain_names={domain_names}")
                    records = int(item["records"])
                    write_repeated_u16(out, domain_to_id[domain], records, int(args.chunk_records))
                    domain_records[domain] += records
            if tmp_path.stat().st_size != expected_records * 2:
                raise RuntimeError(
                    f"bad sidecar size rank={rank}: {tmp_path.stat().st_size} vs {expected_records * 2}"
                )
            os.replace(tmp_path, out_path)

        total_records += assigned_records
        total_domain_records.update(domain_records)
        summary = {
            "rank": rank,
            "encoded_shard": str(shard_path),
            "sidecar": str(out_path),
            "records": int(assigned_records),
            "domain_records": dict(sorted(domain_records.items())),
        }
        summaries.append(summary)
        print(
            f"[domain-sidecar] rank={rank} records={assigned_records} sidecar={out_path} "
            f"domains={dict(sorted(domain_records.items()))}",
            flush=True,
        )

    expected_total = int(final_done["records"])
    if total_records != expected_total:
        raise RuntimeError(f"total record mismatch: {total_records} vs {expected_total}")
    payload = {
        "mode": "domain_id_sidecars_from_manifest_assignment",
        "manifest": str(args.manifest),
        "raw_shard_dir": str(args.raw_shard_dir),
        "encoded_done_json": str(args.encoded_done_json),
        "out_dir": str(args.out_dir),
        "domain_names": domain_names,
        "domain_to_id": domain_to_id,
        "num_shards": int(cfg["num_shards"]),
        "seed": int(cfg["seed"]),
        "token_budget": int(cfg["token_budget"]),
        "records": int(total_records),
        "domain_records": dict(sorted(total_domain_records.items())),
        "dry_run": bool(args.dry_run),
        "elapsed_sec": time.time() - start,
        "shards": summaries,
    }
    summary_path = args.summary_json or (args.out_dir / "summary.json")
    if not args.dry_run:
        write_json(summary_path, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
