#!/usr/bin/env python3
"""Build a multi-budget fixed-FLOPs sweep over small Qwen3-style models."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

try:
    from .scaling_paths import (
        DEFAULT_DATA_ROOT,
        DEFAULT_RUN_ROOT,
        DEFAULT_STATS,
        safe_path_component,
    )
except ImportError:  # direct script execution
    from scaling_paths import (
        DEFAULT_DATA_ROOT,
        DEFAULT_RUN_ROOT,
        DEFAULT_STATS,
        safe_path_component,
    )

DEFAULT_EXPERIMENT_NAME = "scaling_fddfs_mf80000_fixed_flops_sweep_overepoch_domainlog_v2"

FLOPS_TARGETS = [
    ("fx1e18", 1.0e18),
    ("fx2e18", 2.0e18),
    ("fx3p5e18", 3.5e18),
    ("fx6e18", 6.0e18),
    ("fx1e19", 1.0e19),
]

# Optional sparse selection used by follow-up sweeps.  The default remains the
# full Cartesian product for the original experiment.
SWEEP_PLAN: set[tuple[str, str]] | None = None

# All new rows use a 6144-sample optimizer batch on one GPU. Batch size is
# reduced as models grow and gradient accumulation is increased accordingly.
MODEL_SPECS: list[dict[str, Any]] = [
    {
        "model_id": "0p017b",
        "model_label": "0.017B",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    },
    {
        "model_id": "0p027b",
        "model_label": "0.027B",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 12,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    },
    {
        "model_id": "0p041b",
        "model_label": "0.041B",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 20,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    },
    {
        "model_id": "0p059b",
        "model_label": "0.059B",
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "per_device_train_batch_size": 1024,
        "gradient_accumulation_steps": 6,
    },
    {
        "model_id": "0p098b",
        "model_label": "0.098B",
        "hidden_size": 768,
        "intermediate_size": 2304,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 128,
        "per_device_train_batch_size": 512,
        "gradient_accumulation_steps": 12,
    },
    {
        "model_id": "0p183b",
        "model_label": "0.183B",
        "hidden_size": 768,
        "intermediate_size": 2304,
        "num_hidden_layers": 24,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 128,
        "per_device_train_batch_size": 384,
        "gradient_accumulation_steps": 16,
    },
    {
        "model_id": "0p301b",
        "model_label": "0.301B",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "per_device_train_batch_size": 384,
        "gradient_accumulation_steps": 16,
    },
    {
        "model_id": "0p458b",
        "model_label": "0.458B",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "per_device_train_batch_size": 256,
        "gradient_accumulation_steps": 24,
    },
    {
        "model_id": "0p687b",
        "model_label": "0.687B",
        "hidden_size": 1536,
        "intermediate_size": 4608,
        "num_hidden_layers": 24,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "per_device_train_batch_size": 128,
        "gradient_accumulation_steps": 48,
    },
    {
        "model_id": "0p907b",
        "model_label": "0.907B",
        "hidden_size": 1536,
        "intermediate_size": 4608,
        "num_hidden_layers": 32,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "per_device_train_batch_size": 96,
        "gradient_accumulation_steps": 64,
    },
]

REUSED_RUNS = {
    ("fx1e18", "0p017b"): Path("./storage/molscaling/runs/scaling_fddfs_mf80000_fixed_flops_sweep_smallmodels_domainlog_v1/0p017b/fx1e18"),
    ("fx1e18", "0p027b"): Path("./storage/molscaling/runs/scaling_fddfs_mf80000_fixed_flops_sweep_smallmodels_domainlog_v1/0p027b/fx1e18"),
    ("fx2e18", "0p041b"): Path("./storage/molscaling/runs/scaling_fddfs_mf80000_fixed_flops_sweep_smallmodels_domainlog_v1/0p041b/fx2e18"),
    ("fx3p5e18", "0p059b"): Path("./storage/molscaling/runs/scaling_fddfs_mf80000_flops_aligned_epoch1_domainlog_v1/0p06b/f350"),
    ("fx3p5e18", "0p098b"): Path("./storage/molscaling/runs/scaling_fddfs_mf80000_fixed_flops_sweep_smallmodels_domainlog_v1/0p098b/fx3p5e18"),
    ("fx3p5e18", "0p301b"): Path("./storage/molscaling/runs/scaling_fddfs_mf80000_fixed_flops3p5e18_domainlog_v1/0p30b/fx3p5e18"),
    ("fx3p5e18", "0p458b"): Path("./storage/molscaling/runs/scaling_fddfs_mf80000_fixed_flops3p5e18_domainlog_v1/0p60b/fx3p5e18"),
}

# Keep interrupted attempts immutable. Retries use fresh output directories so
# train logs from different random initializations can never be concatenated.
RETRY_RUN_DIR_NAMES = {
    ("fx2e18", "0p059b"): "fx2e18_retry1",
}


def load_epoch_builder() -> Any:
    path = Path(__file__).with_name("build_epoch_fraction_scaling_matrix.py")
    spec = importlib.util.spec_from_file_location("epoch_fraction_scaling_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats-json", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    ap.add_argument("--vocab-size", type=int, default=17207)
    ap.add_argument("--token-offset", type=int, default=3)
    ap.add_argument("--online-pad-length", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--use-historical-run-routing",
        action="store_true",
        help=(
            "reuse the original GPFS output directories and retry names; disabled "
            "by default so a new local experiment cannot silently consume old runs"
        ),
    )
    args = ap.parse_args()

    args.stats_json = args.stats_json.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    args.experiment_name = safe_path_component(
        args.experiment_name, "--experiment-name"
    )

    reused_runs = REUSED_RUNS if args.use_historical_run_routing else {}
    retry_run_dir_names = (
        RETRY_RUN_DIR_NAMES if args.use_historical_run_routing else {}
    )

    epoch_builder = load_epoch_builder()
    stats = json.loads(args.stats_json.read_text(encoding="utf-8"))
    records = int(stats["records"])
    tokens_with_eos = int(stats["tokens_with_eos"])
    mean_tokens = tokens_with_eos / records
    train_records, val_records = epoch_builder.train_record_count(records, args.val_fraction)
    train_tokens_per_epoch = mean_tokens * train_records
    out_root = args.run_root / args.experiment_name

    rows: list[dict[str, Any]] = []
    for label, target_flops in FLOPS_TARGETS:
        for raw_spec in MODEL_SPECS:
            spec = dict(raw_spec)
            if SWEEP_PLAN is not None and (label, str(spec["model_id"])) not in SWEEP_PLAN:
                continue
            params = epoch_builder.qwen3_packed_param_count(spec, args.vocab_size)
            gpu_count = 1
            batch_size = int(spec["per_device_train_batch_size"])
            grad_accum = int(spec["gradient_accumulation_steps"])
            samples_per_step = gpu_count * batch_size * grad_accum
            if samples_per_step != 6144:
                raise ValueError(f"{spec['model_id']} global batch is {samples_per_step}, expected 6144")
            tokens_per_step = mean_tokens * samples_per_step
            one_epoch_steps = math.floor(train_records / samples_per_step)
            requested_tokens = target_flops / (6.0 * params)
            requested_steps = max(1, round(requested_tokens / tokens_per_step))
            over_one_epoch = requested_steps > one_epoch_steps

            actual_samples = requested_steps * samples_per_step
            actual_tokens = requested_steps * tokens_per_step
            actual_flops = 6.0 * params * actual_tokens
            out_dir_name = retry_run_dir_names.get((label, spec["model_id"]), label)
            out_dir = out_root / spec["model_id"] / out_dir_name
            reused = (label, spec["model_id"]) in reused_runs
            if reused:
                out_dir = reused_runs[(label, spec["model_id"])]
            rows.append(
                {
                    "experiment_name": args.experiment_name,
                    "model_id": spec["model_id"],
                    "model_label": spec["model_label"],
                    "flops_target_label": label,
                    "epoch_fraction_target": requested_tokens / train_tokens_per_epoch,
                    "epoch_fraction": actual_samples / train_records,
                    "ratio": actual_tokens / params,
                    "parameter_count": params,
                    "target_tokens": round(requested_tokens),
                    "actual_tokens": round(actual_tokens),
                    "actual_samples": actual_samples,
                    "actual_token_budget_b": actual_tokens / 1e9,
                    "flops_6nd": actual_flops,
                    "target_flops_6nd": target_flops,
                    "flops_relative_error": (actual_flops - target_flops) / target_flops,
                    "tokens_per_step": tokens_per_step,
                    "samples_per_step": samples_per_step,
                    "requested_steps": requested_steps,
                    "max_steps": requested_steps,
                    "one_epoch_steps": one_epoch_steps,
                    "clipped_to_one_epoch": False,
                    "over_one_epoch": over_one_epoch,
                    "warmup_steps": max(1, round(requested_steps * args.warmup_ratio)),
                    "warmup_ratio": args.warmup_ratio,
                    "seed": args.seed,
                    "vocab_size": args.vocab_size,
                    "token_offset": args.token_offset,
                    "online_pad_length": args.online_pad_length,
                    "val_fraction": args.val_fraction,
                    "train_records": train_records,
                    "val_records": val_records,
                    "train_tokens_per_epoch": round(train_tokens_per_epoch),
                    "mean_tokens_with_eos": mean_tokens,
                    "gpu_count": gpu_count,
                    "cpu_count": 16,
                    "memory_mib": 192000,
                    "per_device_train_batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "hidden_size": spec["hidden_size"],
                    "intermediate_size": spec["intermediate_size"],
                    "num_hidden_layers": spec["num_hidden_layers"],
                    "num_attention_heads": spec["num_attention_heads"],
                    "num_key_value_heads": spec["num_key_value_heads"],
                    "head_dim": spec["head_dim"],
                    "model_source": "Qwen3-style shape with project vocabulary",
                    "run_name": f"{args.experiment_name}_{spec['model_id']}_{label}_g1",
                    "out_dir": str(out_dir),
                    "independent_run": True,
                    "domain_loss_logging_required": True,
                    "reused_completed_run": reused,
                }
            )

    if not rows:
        raise ValueError("sweep plan produced no matrix rows")

    matrix_json = out_root / "scaling_matrix.json"
    matrix_csv = out_root / "scaling_matrix.csv"
    metadata = {
        "stats_json": str(args.stats_json),
        "records": records,
        "tokens_with_eos": tokens_with_eos,
        "train_records": train_records,
        "val_records": val_records,
        "train_tokens_per_epoch": round(train_tokens_per_epoch),
        "mean_tokens_with_eos": mean_tokens,
        "experiment_name": args.experiment_name,
        "run_root": str(out_root),
        "flops_targets": [{"label": label, "target_flops_6nd": value} for label, value in FLOPS_TARGETS],
        "sparse_sweep_plan": SWEEP_PLAN is not None,
        "allow_over_one_epoch": True,
        "global_samples_per_step": 6144,
        "note": "Adaptive fixed-FLOPs sweep. Rows above one train-sample epoch are retained and explicitly marked.",
        "domain_loss_logging_required": True,
        "historical_run_routing_enabled": args.use_historical_run_routing,
        "reused_completed_runs": [
            {"target_label": label, "model_id": model_id, "out_dir": str(path)}
            for (label, model_id), path in sorted(reused_runs.items())
        ],
    }
    matrix_json.parent.mkdir(parents=True, exist_ok=True)
    matrix_json.write_text(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(matrix_csv, rows)
    print(
        json.dumps(
            {
                "matrix_json": str(matrix_json),
                "matrix_csv": str(matrix_csv),
                "rows": len(rows),
                "over_one_epoch": sum(bool(row["over_one_epoch"]) for row in rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
