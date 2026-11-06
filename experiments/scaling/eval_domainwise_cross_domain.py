#!/usr/bin/env python3
"""Evaluate frozen single-domain checkpoints on all five frozen validation views.

No training, test access, checkpoint edits, or changes to the live campaign.
Use the native packed-xformers model, dataset, collator, and evaluate function.
The primary union loss weights domains by causal prediction tokens, not records.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

DOMAINS = ("small_molecule", "peptide", "protein", "dna", "rna")
PROTOCOL = "domainwise_cross_domain_full_validation_v1"
ROOT = Path(__file__).resolve().parents[4]
STORAGE = Path("./storage/molscaling")
DEFAULT_SNAPSHOT = ROOT / "doc/results/domainwise145_visualization_20260906/snapshot.json"
DEFAULT_MANIFEST = STORAGE / "data/processed/domain_split_manifests_v1/manifest.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def native_module():
    path = Path(__file__).resolve().parents[2] / "cli/pretrain/train_qwen3_pretrain.py"
    spec = importlib.util.spec_from_file_location("cross_domain_native_trainer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def totals(manifest):
    result = {}
    for domain in DOMAINS:
        view = manifest["views"][domain]["validation"]
        result[domain] = dict(samples=int(view["records"]),
                              input_tokens=int(view["effective_input_tokens_with_eos"]),
                              tokens=int(view["effective_input_tokens_with_eos"]) - int(view["records"]))
    return result


def check_completion(row, manifest):
    final = Path(row["checkpoint"])
    for name in ("config.json", "pytorch_model.bin"):
        if not (final / name).is_file():
            raise FileNotFoundError(final / name)
    cfg = read(final.parent / "run_config.json")["args"]
    summary = read(final.parent / "summary.json")
    marker = read(final.parent.parent / "complete.json")
    assert marker["status"] == "complete", final
    assert marker["run_id"] == row["run_id"], final
    assert Path(marker["attempt_dir"]).resolve() == final.parent.resolve(), final
    expected = dict(model_implementation="packed-xformers", dataset_mode="online-pad",
                    domain_name=row["domain"], online_pad_length=512,
                    token_offset=3, pad_token_id=0, eos_token_id=2, vocab_size=17207,
                    bpe_unfold_policy="none")
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f"{row['run_id']}: {key}={cfg[key]!r}, expected {value!r}")
    assert Path(cfg["domain_split_manifest"]).resolve() == Path(manifest).resolve(), final
    assert summary["parameter_count"] == row["parameter_count"], final
    v = summary["final_validation"]
    assert v["complete_view"] is True and v["input_token_coverage"] == 1.0, final
    for key, value in totals(read(manifest))[row["domain"]].items():
        assert v[key] == value, (final, key, v[key], value)
    assert math.isclose(v["loss"], row["validation_loss"], abs_tol=1e-8, rel_tol=0), final
    return cfg


def prepare(args):
    if (args.output / "plan.json").exists():
        raise FileExistsError(args.output / "plan.json")
    snap = read(args.snapshot)
    assert snap["metric_split"] == "validation" and snap["test_accessed"] is False
    available, unavailable, seen = [], [], set()
    for curve in snap["curves"]:
        for p in curve["points"]:
            rid = p["run_id"]
            if rid in seen:
                raise ValueError(f"duplicate scientific run {rid}")
            seen.add(rid)
            source = p.get("checkpoint_relative_to_workspace")
            if source is None:
                source = str(Path(p["attempt_dir"]) / "final")
            path = Path(source)
            if not path.is_absolute():
                path = ROOT.parent / path
            row = {k: p[k] for k in ("run_id", "domain", "budget", "model_id", "parameter_count", "validation_loss")}
            row.update(checkpoint=str(path), target_flops=float(p["target_flops_6nd"]),
                       actual_flops=float(p["actual_flops_6nd"]),
                       actual_epoch_fraction=float(p["actual_epoch_fraction"]))
            if not path.is_dir():
                if Path(source).is_absolute():
                    raise FileNotFoundError(f"Local completed checkpoint missing: {path}")
                unavailable.append(dict(row, reason="historical_physical_host_checkpoint_not_mounted"))
                continue
            check_completion(row, args.manifest)
            available.append(row)
    # First five are representatives at the highest locally available budget;
    # in-domain minimum only determines execution order, NOT inclusion in results.
    pilots = []
    for domain in DOMAINS:
        candidates = [p for p in available if p["domain"] == domain]
        budget = max(p["target_flops"] for p in candidates)
        pilots.append(min((p for p in candidates if p["target_flops"] == budget), key=lambda p: p["validation_loss"]))
    pilot_ids = {p["run_id"] for p in pilots}
    ordered = pilots + sorted((p for p in available if p["run_id"] not in pilot_ids),
                              key=lambda p: (p["budget"], p["domain"], p["parameter_count"]))
    for i, row in enumerate(ordered):
        row["lane"] = i % 2
        row["representative_first"] = row["run_id"] in pilot_ids
    plan = dict(protocol=PROTOCOL, created_at=stamp(), source_snapshot=str(args.snapshot),
                snapshot_updated_at=snap["updated_at"], manifest=str(args.manifest),
                metric_split="validation", test_accessed=False, context_length=512,
                truncate_body_to=511, append_eos=True, bpe_unfold=False,
                attention="native_xformers_block_diagonal_causal",
                union_weighting="causal_prediction_tokens", expected=totals(read(args.manifest)),
                rows=ordered, unavailable=unavailable, pending_training=snap["pending"],
                source_completed=len(seen), eligible=len(ordered),
                pilot_ids=[p["run_id"] for p in pilots])
    write(args.output / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ("source_completed", "eligible", "pending_training", "pilot_ids", "expected")}, indent=2))
    print("AVAILABLE_BY_DOMAIN", dict(Counter(p["domain"] for p in ordered)))
    print("UNAVAILABLE_BY_DOMAIN", dict(Counter(p["domain"] for p in unavailable)))


def aggregate(metrics, train_domain):
    def weighted(domains):
        n = sum(metrics[d]["tokens"] for d in domains)
        return sum(metrics[d]["loss"] * metrics[d]["tokens"] for d in domains) / n
    return dict(all_domain_loss=weighted(DOMAINS),
                other_domain_loss=weighted([d for d in DOMAINS if d != train_domain]),
                macro_domain_loss=sum(metrics[d]["loss"] for d in DOMAINS) / len(DOMAINS),
                total_loss_tokens=sum(metrics[d]["tokens"] for d in DOMAINS))


def check_result(result, row, plan):
    assert result["protocol"] == PROTOCOL and result["status"] == "complete"
    assert result["checkpoint"] == row["checkpoint"] and result["run_id"] == row["run_id"]
    assert result["manifest"] == plan["manifest"] and result["metric_split"] == "validation"
    assert set(result["domains"]) == set(DOMAINS)
    for d in DOMAINS:
        for k, v in plan["expected"][d].items():
            assert result["domains"][d][k] == v, (d, k)
        assert math.isfinite(result["domains"][d]["loss"])
    assert abs(result["domains"][row["domain"]]["loss"] - row["validation_loss"]) <= 1e-4
    for k, v in aggregate(result["domains"], row["domain"]).items():
        assert math.isclose(result[k], v, rel_tol=1e-12, abs_tol=1e-12)


def run(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import Qwen3Config
    if args.expected_host and socket.gethostname() != args.expected_host:
        raise RuntimeError(f"wrong host: {socket.gethostname()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Expected user-provided two-GPU worker")
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    tm = native_module()
    plan = read(args.output / "plan.json")
    manifest = read(plan["manifest"])
    datasets = {}
    for d in DOMAINS:
        paths = [Path(s["token_file"]) for s in manifest["views"][d]["validation"]["shards"]]
        datasets[d] = tm.CompactIntOnlinePadDataset(
            paths, Path(manifest["dataset"]["index_dir"]), "val", 0.0, "stride",
            3, 2, 0, 512, True, 42, 4.0, None, None, Path(plan["manifest"]), d)
    queue = [r for r in plan["rows"] if r["lane"] == args.device]
    if args.smoke:
        queue = [max(plan["rows"], key=lambda r: r["parameter_count"])]
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / f"{'smoke' if args.smoke else 'lane'}_{args.device}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    loaders = {d: DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=tm.collate_online_padded_samples,
                             persistent_workers=args.num_workers > 0) for d, ds in datasets.items()}
    for pos, row in enumerate(queue):
        target = args.output / ("smoke" if args.smoke else "results") / f"{row['run_id']}.json"
        if target.exists():
            if args.smoke:
                raise FileExistsError(target)
            check_result(read(target), row, plan)
            print("REUSE_COMPLETE", row["run_id"], flush=True)
            continue
        check_completion(row, plan["manifest"])
        start = time.monotonic()
        final = Path(row["checkpoint"])
        config = Qwen3Config.from_pretrained(final, local_files_only=True)
        config.use_cache = False
        model = tm.PackedXformersQwen3ForCausalLM(config)
        state = torch.load(final / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        assert sum(p.numel() for p in model.parameters()) == row["parameter_count"]
        model.to(device)
        print("START", pos + 1, len(queue), row["run_id"], flush=True)
        metrics = {}
        # Recheck native same-domain validation before attributing any cross-domain result.
        domain_order = (row["domain"],) + tuple(d for d in DOMAINS if d != row["domain"])
        for d in domain_order:
            begin = time.monotonic()
            last_heartbeat = [0.0]
            def progress(counts):
                if time.monotonic() - last_heartbeat[0] >= 30:
                    write(args.output / f"progress_lane{args.device}.json",
                          dict(at=stamp(), run_id=row["run_id"], eval_domain=d,
                               queue_position=pos + 1, queue_total=len(queue), **counts))
                    print("PROGRESS", row["run_id"], d, counts, flush=True)
                    last_heartbeat[0] = time.monotonic()
            loader = loaders[d]
            if args.smoke:
                # Use longest frozen validation records to test a conservative VRAM bound.
                import numpy as np
                lengths = np.concatenate([np.minimum(a[idx], 511) + 1
                                          for a, idx in zip(datasets[d].length_arrays, datasets[d].raw_index_arrays)])
                indices = np.argsort(lengths, kind="stable")[-min(len(lengths), args.batch_size):].tolist()
                loader = DataLoader(Subset(datasets[d], indices), batch_size=args.batch_size,
                                    collate_fn=tm.collate_online_padded_samples)
            metrics[d] = tm.evaluate(model, loader, device, 1 if args.smoke else 0, True, progress)
            metrics[d]["elapsed_sec"] = time.monotonic() - begin
            if not args.smoke:
                for key, expected in plan["expected"][d].items():
                    if metrics[d][key] != expected:
                        raise ValueError(f"Incomplete {row['run_id']}/{d}: {key} {metrics[d][key]} != {expected}")
                if d == row["domain"]:
                    diff = metrics[d]["loss"] - row["validation_loss"]
                    if abs(diff) > 1e-4:
                        raise ValueError(f"Native diagonal mismatch {row['run_id']}: delta={diff}")
            print("DOMAIN_DONE", row["run_id"], d, json.dumps(metrics[d]), flush=True)
        result = dict(row, protocol=PROTOCOL, status="smoke_only" if args.smoke else "complete",
                      metric_split="validation", test_accessed=False, manifest=plan["manifest"],
                      domains=metrics, completed_at=stamp(), elapsed_sec=time.monotonic() - start,
                      device=args.device, hostname=socket.gethostname(), batch_size=args.batch_size,
                      torch_version=torch.__version__, native_model_strict_loaded=True,
                      context_length=512, truncate_body_to=511, **aggregate(metrics, row["domain"]))
        if not args.smoke:
            check_result(result, row, plan)
        write(target, result)
        print("CHECKPOINT_DONE", row["run_id"], result["all_domain_loss"], flush=True)
        del model
        torch.cuda.empty_cache()
    print("LANE_COMPLETE", args.device, "smoke" if args.smoke else "formal", flush=True)


def summarize(args):
    plan = read(args.output / "plan.json")
    rows, pending = [], []
    for r in plan["rows"]:
        p = args.output / "results" / f"{r['run_id']}.json"
        if not p.exists():
            pending.append(r["run_id"])
            continue
        v = read(p)
        check_result(v, r, plan)
        rows.append(dict(run_id=r["run_id"], train_domain=r["domain"], budget=r["budget"],
                         parameter_count=r["parameter_count"], target_flops=r["target_flops"],
                         actual_epoch_fraction=r["actual_epoch_fraction"], checkpoint=r["checkpoint"],
                         **{d + "_loss": v["domains"][d]["loss"] for d in DOMAINS},
                         **aggregate(v["domains"], r["domain"]),
                         native_diagonal_delta=v["domains"][r["domain"]]["loss"] - r["validation_loss"]))
    write(args.output / "summary.json", dict(protocol=PROTOCOL, at=stamp(), complete=len(rows),
          eligible=plan["eligible"], unavailable=len(plan["unavailable"]),
          pending=pending, metric_split="validation", rows=rows))
    if rows:
        path = args.output / "summary.csv"
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    print("SUMMARY", len(rows), "/", plan["eligible"], "unavailable", len(plan["unavailable"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "run", "summarize"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--device", type=int, choices=(0, 1), default=0)
    p.add_argument("--expected-host", default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("invalid batch size / worker count")
    {"prepare": prepare, "run": run, "summarize": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
