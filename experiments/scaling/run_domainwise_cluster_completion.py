#!/usr/bin/env python3
"""Run a disjoint slice of the domain-wise completion plan on allocated GPUs.

One invocation owns one cluster allocation.  ``--device-indices`` names CUDA
indices inside that scheduler-provided allocation, never physical host GPU
IDs.  Each device can host multiple independent lanes after a real H200 smoke.
The runner never sets or rewrites ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .build_domainwise_cluster_completion_plan import SCHEMA_VERSION
except ImportError:  # direct script execution
    from build_domainwise_cluster_completion_plan import SCHEMA_VERSION


ATTEMPT_SCHEMA = "domainwise_cluster_attempt_v1"
COMPLETION_SCHEMA = "domainwise_cluster_completion_v1"
FAILURE_SCHEMA = "domainwise_cluster_failure_v1"
_DISTRIBUTED_ENV_KEYS = {
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps({"time": now(), **value}, ensure_ascii=False, allow_nan=False)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def read_run_ids(path: Path) -> list[str]:
    run_ids = [
        line.strip()
        for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not run_ids:
        raise ValueError(f"run-id whitelist is empty: {path}")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"run-id whitelist contains duplicates: {path}")
    return run_ids


def validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported plan schema: {plan.get('schema_version')!r}")
    rows = plan.get("rows")
    if not isinstance(rows, list) or int(plan.get("row_count", -1)) != len(rows):
        raise ValueError("plan rows are missing or row_count is inconsistent")
    if len(rows) != 39:
        raise ValueError(f"completion plan must contain 39 rows, found {len(rows)}")
    seen: set[str] = set()
    phase_root = Path(str(plan["phase_root"])).expanduser().resolve()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("plan row must be an object")
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in seen:
            raise ValueError(f"empty or duplicate plan run_id: {run_id!r}")
        seen.add(run_id)
        if row.get("source_status") == "not_started":
            expected_attempt = 1
        elif row.get("source_status") == "interrupted_restart_required":
            expected_attempt = 2
        else:
            raise ValueError(f"unsupported source status for {run_id}")
        if (
            int(row["attempt_number"]) != expected_attempt
            or int(row["restart_from_step"]) != 0
            or row.get("init_checkpoint") is not None
            or row.get("random_initialization") is not True
            or int(row["world_size"]) != 1
        ):
            raise ValueError(f"restart/initialization contract changed for {run_id}")
        micro = int(row["per_device_train_batch_size"])
        accum = int(row["gradient_accumulation_steps"])
        effective = int(row["effective_batch_records"])
        if micro <= 0 or accum <= 0 or micro * accum != effective:
            raise ValueError(f"effective record batch changed for {run_id}")
        steps = int(row["scheduler_total_steps"])
        if int(row["warmup_steps"]) != max(1, round(steps * float(row["warmup_ratio"]))):
            raise ValueError(f"warmup schedule changed for {run_id}")
        parameters = int(row["parameter_count"])
        target_tokens = int(row["target_input_tokens_with_eos"])
        target_flops = int(row["target_flops_6nd"])
        if parameters <= 0 or target_tokens <= 0 or target_flops <= 0:
            raise ValueError(f"non-positive scientific budget for {run_id}")
        if abs(6 * parameters * target_tokens - target_flops) > 3 * parameters:
            raise ValueError(f"token and 6ND targets disagree for {run_id}")
        run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        attempt_dir = Path(str(row["attempt_dir"])).expanduser().resolve()
        expected_run = phase_root / "runs" / run_id
        expected_attempt_dir = expected_run / f"attempt_{expected_attempt:03d}"
        if run_dir != expected_run or attempt_dir != expected_attempt_dir:
            raise ValueError(f"cluster output routing changed for {run_id}")
    return rows


def selected_rows(
    plan: dict[str, Any], run_ids: Iterable[str]
) -> list[dict[str, Any]]:
    rows = validate_plan(plan)
    by_id = {str(row["run_id"]): row for row in rows}
    requested = list(run_ids)
    unknown = [run_id for run_id in requested if run_id not in by_id]
    if unknown:
        raise ValueError(f"whitelist contains run IDs outside the completion plan: {unknown}")
    return [by_id[run_id] for run_id in requested]


def partition_lpt(
    rows: Iterable[dict[str, Any]], slot_count: int
) -> list[list[dict[str, Any]]]:
    """Deterministically balance planned 6ND across independent lanes."""

    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(slot_count)]
    loads = [0] * slot_count
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["target_flops_6nd"]), str(row["run_id"])),
    )
    for row in ordered:
        slot = min(range(slot_count), key=lambda index: (loads[index], index))
        partitions[slot].append(row)
        loads[slot] += int(row["target_flops_6nd"])
    return partitions


def lane_partition_index(
    *,
    global_gpu_index: int,
    global_gpu_count: int,
    lane_within_gpu: int,
    lanes_per_gpu: int,
) -> int:
    if not 0 <= global_gpu_index < global_gpu_count:
        raise ValueError("global_gpu_index is outside the allocation layout")
    if not 0 <= lane_within_gpu < lanes_per_gpu:
        raise ValueError("lane_within_gpu is outside the per-GPU layout")
    return lane_within_gpu * global_gpu_count + global_gpu_index


def parse_device_indices(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--device-indices must contain at least one allocation-local index")
    if any(not value.isdigit() for value in values):
        raise ValueError("--device-indices must be comma-separated non-negative integers")
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)):
        raise ValueError("--device-indices contains duplicates")
    return indices


def single_process_environment(
    source: dict[str, str], *, device_index: int, pipeline_root: Path
) -> dict[str, str]:
    environment = {str(key): str(value) for key, value in source.items()}
    for key in tuple(environment):
        if key in _DISTRIBUTED_ENV_KEYS or key.startswith("TORCHELASTIC_"):
            environment.pop(key)
    # LOCAL_RANK is an allocation-local torch device index.  The scheduler's
    # CUDA visibility is inherited unchanged; this runner never sets it.
    environment["WORLD_SIZE"] = "1"
    environment["RANK"] = "0"
    environment["LOCAL_RANK"] = str(device_index)
    environment["LOCAL_WORLD_SIZE"] = "1"
    environment["PYTHONPATH"] = str(pipeline_root) + (
        ":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def visible_cuda_device_count(python: Path) -> int:
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import torch; print(torch.cuda.device_count())",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    if not text.isdigit():
        raise RuntimeError(f"cannot parse torch CUDA device count: {text!r}")
    return int(text)


def validate_manifest(manifest_path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "domain_split_raw_indices_v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("domain split manifest is not complete domain_split_raw_indices_v1")
    dataset = manifest.get("dataset")
    views = manifest.get("views")
    if not isinstance(dataset, dict) or not isinstance(views, dict):
        raise ValueError("domain split manifest has no dataset/views objects")
    compact_dir = Path(str(dataset["compact_dir"])).expanduser().resolve()
    index_dir = Path(str(dataset["index_dir"])).expanduser().resolve()
    if not compact_dir.is_dir() or not index_dir.is_dir():
        raise FileNotFoundError(f"compact dataset paths are unavailable: {compact_dir}, {index_dir}")
    domains = {str(row["domain"]) for row in rows}
    for domain in domains:
        domain_views = views.get(domain)
        if not isinstance(domain_views, dict):
            raise ValueError(f"manifest has no domain view: {domain}")
        for split in ("train", "validation"):
            view = domain_views.get(split)
            if (
                not isinstance(view, dict)
                or int(view.get("records", 0)) <= 0
                or int(view.get("effective_input_tokens_with_eos", 0))
                < int(view.get("records", 0))
                or not isinstance(view.get("shards"), list)
                or not view["shards"]
            ):
                raise ValueError(f"invalid manifest view: {domain}/{split}")
            for shard in view["shards"]:
                token_file = Path(str(shard["token_file"])).expanduser().resolve()
                raw_indices = Path(str(shard["raw_indices_file"])).expanduser().resolve()
                if not token_file.is_file() or token_file.stat().st_size <= 0:
                    raise FileNotFoundError(token_file)
                expected_bytes = int(shard["records"]) * 4
                if not raw_indices.is_file() or raw_indices.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"raw-index size mismatch for {domain}/{split}: {raw_indices}"
                    )
    return manifest


def training_command(
    *,
    python: Path,
    pipeline_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    row: dict[str, Any],
    attempt_dir: Path,
    num_workers: int,
    profile_steps: int | None = None,
) -> list[str]:
    if profile_steps is not None and profile_steps <= 0:
        raise ValueError("profile_steps must be positive")
    dataset = manifest["dataset"]
    token_glob = str(Path(str(dataset["compact_dir"])) / "shard_*.bin")
    command = [
        str(python),
        str(pipeline_root / "cli" / "pretrain" / "train_qwen3_pretrain.py"),
        "--token-files",
        token_glob,
        "--val-token-files",
        token_glob,
        "--out-dir",
        str(attempt_dir),
        "--vocab-size",
        str(row["vocab_size"]),
        "--dataset-mode",
        "online-pad",
        "--model-implementation",
        "packed-xformers",
        "--compact-index-dir",
        str(dataset["index_dir"]),
        "--domain-split-manifest",
        str(manifest_path),
        "--domain-name",
        str(row["domain"]),
        "--online-pad-length",
        str(row["context_length"]),
        "--online-truncate-long-samples",
        "--online-shuffle",
        "--token-offset",
        "3",
        "--val-fraction",
        "0",
        "--bpe-unfold-policy",
        "none",
        "--bpe-unfold-prob",
        "0",
        "--hidden-size",
        str(row["hidden_size"]),
        "--intermediate-size",
        str(row["intermediate_size"]),
        "--num-hidden-layers",
        str(row["num_hidden_layers"]),
        "--num-attention-heads",
        str(row["num_attention_heads"]),
        "--num-key-value-heads",
        str(row["num_key_value_heads"]),
        "--head-dim",
        str(row["head_dim"]),
        "--max-position-embeddings",
        str(row["context_length"]),
        "--batch-size",
        str(row["per_device_train_batch_size"]),
        "--grad-accum",
        str(row["gradient_accumulation_steps"]),
        "--max-steps",
        str(profile_steps if profile_steps is not None else row["scheduler_total_steps"]),
        "--lr",
        "0.0002",
        "--min-lr-ratio",
        "0.1",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--adam-eps",
        "1e-08",
        "--max-grad-norm",
        "1.0",
        "--warmup-ratio",
        str(row["warmup_ratio"]),
        "--log-every",
        "20",
        "--eval-every",
        "0",
        "--eval-batches",
        "0",
        "--save-every",
        "0",
        "--num-workers",
        str(num_workers),
        "--seed",
        str(row["seed"]),
        "--optimizer",
        "adamw-fused",
        "--attn-implementation",
        "sdpa",
        "--sdpa-kernel",
        "auto",
        "--bf16",
        "--no-gradient-checkpointing",
    ]
    if profile_steps is None:
        command[command.index("--lr"):command.index("--lr")] = [
            "--target-input-tokens",
            str(row["target_input_tokens_with_eos"]),
        ]
        command.extend(["--final-eval", "--final-eval-batches", "0"])
    return command


def validate_completion(
    *,
    attempt_dir: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    summary = load_json(attempt_dir / "summary.json")
    run_config = load_json(attempt_dir / "run_config.json")
    if not isinstance(summary, dict) or not isinstance(run_config, dict):
        raise ValueError(f"training output is not a JSON object: {attempt_dir}")
    target = int(row["target_input_tokens_with_eos"])
    actual = int(summary["actual_input_tokens"])
    parameters = int(row["parameter_count"])
    if int(summary["target_input_tokens"]) != target or not 0 <= actual - target < int(
        row["context_length"]
    ):
        raise ValueError(f"input-token stop contract failed for {row['run_id']}")
    if (
        int(summary["parameter_count"]) != parameters
        or int(summary["target_flops_6nd"]) != 6 * parameters * target
        or int(summary["actual_flops_6nd"]) != 6 * parameters * actual
        or int(summary["scheduler_total_steps"]) != int(row["scheduler_total_steps"])
        or int(summary["warmup_steps"]) != int(row["warmup_steps"])
        or summary.get("termination_reason") != "target_input_tokens"
        or summary.get("model_implementation") != "packed-xformers"
        or summary.get("attention_backend")
        != "xformers_memory_efficient_block_diagonal_causal"
    ):
        raise ValueError(f"summary controls differ from the plan for {row['run_id']}")
    mean_loss = float(summary["mean_train_loss"])
    if not math.isfinite(mean_loss) or mean_loss <= 0 or int(summary["global_step"]) <= 0:
        raise ValueError(f"invalid training result for {row['run_id']}")

    domain = str(row["domain"])
    train_view = manifest["views"][domain]["train"]
    train_records = int(train_view["records"])
    train_input_tokens = int(train_view["effective_input_tokens_with_eos"])
    validation_view = manifest["views"][domain]["validation"]
    validation_records = int(validation_view["records"])
    validation_input_tokens = int(validation_view["effective_input_tokens_with_eos"])
    validation = summary.get("final_validation")
    if not isinstance(validation, dict) or validation.get("complete_view") is not True:
        raise ValueError(f"full validation is missing for {row['run_id']}")
    validation_loss = float(validation["loss"])
    validation_ppl = float(validation["ppl"])
    if (
        not math.isfinite(validation_loss)
        or validation_loss <= 0
        or not math.isfinite(validation_ppl)
        or validation_ppl <= 0
    ):
        raise ValueError(f"invalid validation loss for {row['run_id']}")
    if (
        int(validation["samples"]) != validation_records
        or int(validation["input_tokens"]) != validation_input_tokens
        or int(validation["tokens"]) != validation_input_tokens - validation_records
        or int(validation["expected_input_tokens"]) != validation_input_tokens
        or not math.isclose(
            float(validation["input_token_coverage"]), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(f"validation coverage differs from the manifest for {row['run_id']}")

    args = run_config.get("args")
    config = run_config.get("effective_config")
    if not isinstance(args, dict) or not isinstance(config, dict):
        raise ValueError(f"run_config is incomplete for {row['run_id']}")
    expected_args = {
        "vocab_size": int(row["vocab_size"]),
        "batch_size": int(row["per_device_train_batch_size"]),
        "grad_accum": int(row["gradient_accumulation_steps"]),
        "target_input_tokens": target,
        "max_steps": int(row["scheduler_total_steps"]),
        "seed": int(row["seed"]),
        "init_checkpoint": None,
        "model_implementation": "packed-xformers",
        "dataset_mode": "online-pad",
        "online_pad_length": int(row["context_length"]),
        "online_truncate_long_samples": True,
        "online_shuffle": True,
        "token_offset": 3,
        "val_fraction": 0.0,
        "bpe_unfold_policy": "none",
        "bpe_unfold_prob": 0.0,
        "final_eval": True,
        "final_eval_batches": 0,
        "save_every": 0,
    }
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise ValueError(f"run_config arguments differ from the plan for {row['run_id']}")
    if (
        int(run_config.get("train_items", -1)) != train_records
        or int(run_config.get("train_tokens_visible", -1)) != train_input_tokens
        or int(run_config.get("val_items", -1)) != validation_records
        or int(run_config.get("val_tokens_visible", -1)) != validation_input_tokens
        or int(run_config.get("world_size", -1)) != 1
        or run_config.get("training_engine") != "single_process"
        or run_config.get("train_sampler") != "DistributedAffineShuffleSampler"
    ):
        raise ValueError(f"dataset/sampler provenance differs for {row['run_id']}")
    if int(summary["train_input_tokens_per_epoch"]) != train_input_tokens:
        raise ValueError(f"train token epoch differs from the manifest for {row['run_id']}")
    if not math.isclose(
        float(summary["actual_epoch_fraction"]),
        actual / train_input_tokens,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"actual epoch fraction is inconsistent for {row['run_id']}")
    total_samples = int(summary["total_train_samples"])
    if (
        total_samples <= 0
        or int(summary["total_train_loss_tokens"]) != actual - total_samples
    ):
        raise ValueError(f"train token accounting is inconsistent for {row['run_id']}")
    expected_config = {
        "vocab_size": int(row["vocab_size"]),
        "hidden_size": int(row["hidden_size"]),
        "intermediate_size": int(row["intermediate_size"]),
        "num_hidden_layers": int(row["num_hidden_layers"]),
        "num_attention_heads": int(row["num_attention_heads"]),
        "num_key_value_heads": int(row["num_key_value_heads"]),
        "head_dim": int(row["head_dim"]),
        "max_position_embeddings": int(row["context_length"]),
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": True,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"model config differs from the plan for {row['run_id']}")
    final_dir = attempt_dir / "final"
    if Path(str(summary["final_checkpoint"])).expanduser().resolve() != final_dir:
        raise ValueError(f"final checkpoint path differs for {row['run_id']}")
    if not (final_dir / "config.json").is_file():
        raise FileNotFoundError(final_dir / "config.json")
    checkpoint = final_dir / "pytorch_model.bin"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
        raise FileNotFoundError(checkpoint)
    return summary


def run_one(
    *,
    row: dict[str, Any],
    device_index: int,
    python: Path,
    pipeline_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    num_workers: int,
    plan_path: Path,
    status_path: Path,
    status_lock: threading.Lock,
) -> None:
    run_id = str(row["run_id"])
    run_dir = Path(str(row["run_dir"])).expanduser().resolve()
    attempt_dir = Path(str(row["attempt_dir"])).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / ".cluster_run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"run is already owned by another cluster worker: {run_id}") from exc
    try:
        completion_path = run_dir / "complete.json"
        if completion_path.is_file():
            completion = load_json(completion_path)
            if (
                completion.get("schema_version") != COMPLETION_SCHEMA
                or completion.get("run_id") != run_id
                or Path(str(completion.get("attempt_dir", ""))).expanduser().resolve()
                != attempt_dir
            ):
                raise ValueError(f"completion marker identity mismatch for {run_id}")
            validate_completion(attempt_dir=attempt_dir, row=row, manifest=manifest)
            append_jsonl(
                status_path,
                {"event": "skip_complete", "run_id": run_id, "device_index": device_index},
                status_lock,
            )
            return
        if attempt_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite incomplete fixed attempt for {run_id}: {attempt_dir}"
            )
        attempt_dir.mkdir(parents=True)
        command = training_command(
            python=python,
            pipeline_root=pipeline_root,
            manifest=manifest,
            manifest_path=manifest_path,
            row=row,
            attempt_dir=attempt_dir,
            num_workers=num_workers,
        )
        attempt_record = {
            "schema_version": ATTEMPT_SCHEMA,
            "run_id": run_id,
            "source_status": row["source_status"],
            "attempt_number": int(row["attempt_number"]),
            "restart_from_step": 0,
            "init_checkpoint": None,
            "device_index_within_scheduler_allocation": device_index,
            "plan_json": str(plan_path),
            "domain_split_manifest": str(manifest_path),
            "row": row,
            "command": command,
            "started_at": now(),
        }
        atomic_json(attempt_dir / "cluster_attempt.json", attempt_record)
        append_jsonl(
            status_path,
            {
                "event": "run_start",
                "run_id": run_id,
                "attempt": row["attempt_name"],
                "device_index": device_index,
            },
            status_lock,
        )
        environment = single_process_environment(
            os.environ, device_index=device_index, pipeline_root=pipeline_root
        )
        console_path = attempt_dir / "console.log"
        with console_path.open("ab", buffering=0) as console:
            result = subprocess.run(
                command,
                cwd=pipeline_root,
                env=environment,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            atomic_json(
                attempt_dir / "failed.json",
                {
                    "schema_version": FAILURE_SCHEMA,
                    "run_id": run_id,
                    "returncode": result.returncode,
                    "console_log": str(console_path),
                    "failed_at": now(),
                },
            )
            raise RuntimeError(
                f"trainer failed for {run_id} with exit {result.returncode}; "
                f"see {console_path}"
            )
        summary = validate_completion(attempt_dir=attempt_dir, row=row, manifest=manifest)
        marker = {
            "schema_version": COMPLETION_SCHEMA,
            "status": "complete",
            "run_id": run_id,
            "attempt_dir": str(attempt_dir),
            "summary": str(attempt_dir / "summary.json"),
            "actual_input_tokens": int(summary["actual_input_tokens"]),
            "validation_loss": float(summary["final_validation"]["loss"]),
            "completed_at": now(),
        }
        atomic_json(completion_path, marker)
        append_jsonl(
            status_path,
            {
                "event": "run_complete",
                "run_id": run_id,
                "attempt": row["attempt_name"],
                "device_index": device_index,
                "validation_loss": marker["validation_loss"],
            },
            status_lock,
        )
    finally:
        lock_handle.close()


def run_lane(
    *,
    rows: list[dict[str, Any]],
    device_index: int,
    stop: threading.Event,
    common: dict[str, Any],
) -> None:
    for row in rows:
        if stop.is_set():
            return
        run_one(row=row, device_index=device_index, **common)


def main() -> int:
    script = Path(__file__).resolve()
    pipeline_default = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--run-id-file", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=pipeline_default)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            os.environ.get(
                "HOLOMOL_SCALING_PYTHON",
                "python",
            )
        ),
    )
    parser.add_argument("--device-indices", required=True)
    parser.add_argument("--global-gpu-offset", type=int, required=True)
    parser.add_argument("--global-gpu-count", type=int, required=True)
    parser.add_argument("--lanes-per-gpu", type=int, required=True)
    parser.add_argument("--num-workers-per-trainer", type=int, default=2)
    parser.add_argument(
        "--status-jsonl",
        type=Path,
        default=Path("./logs/domainwise_cluster_status.jsonl"),
    )
    args = parser.parse_args()
    if args.global_gpu_count <= 0 or args.global_gpu_offset < 0:
        raise ValueError("global GPU count/offset must be positive/non-negative")
    if args.lanes_per_gpu <= 0:
        raise ValueError("--lanes-per-gpu must be positive")
    if args.num_workers_per_trainer < 0:
        raise ValueError("--num-workers-per-trainer must be non-negative")

    plan_path = args.plan_json.expanduser().resolve()
    whitelist_path = args.run_id_file.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    # Keep a venv Python symlink intact instead of resolving to its base
    # interpreter and losing the environment prefix.
    python = args.python.expanduser().absolute()
    status_path = args.status_jsonl.expanduser().resolve()
    if not pipeline_root.is_dir():
        raise FileNotFoundError(pipeline_root)
    if not python.is_file():
        raise FileNotFoundError(python)
    device_indices = parse_device_indices(args.device_indices)
    if args.global_gpu_offset + len(device_indices) > args.global_gpu_count:
        raise ValueError("allocation GPU slice exceeds --global-gpu-count")
    visible_devices = visible_cuda_device_count(python)
    if visible_devices <= max(device_indices):
        raise RuntimeError(
            f"allocation exposes {visible_devices} CUDA devices, cannot use {device_indices}"
        )

    plan = load_json(plan_path)
    rows = selected_rows(plan, read_run_ids(whitelist_path))
    manifest_path = Path(str(plan["domain_split_manifest"])).expanduser().resolve()
    manifest = validate_manifest(manifest_path, rows)
    partitions = partition_lpt(rows, args.global_gpu_count * args.lanes_per_gpu)
    status_lock = threading.Lock()
    common = {
        "python": python,
        "pipeline_root": pipeline_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "num_workers": args.num_workers_per_trainer,
        "plan_path": plan_path,
        "status_path": status_path,
        "status_lock": status_lock,
    }
    lane_jobs: list[tuple[int, int, list[dict[str, Any]]]] = []
    for local_position, device_index in enumerate(device_indices):
        global_gpu_index = args.global_gpu_offset + local_position
        for lane_within_gpu in range(args.lanes_per_gpu):
            # Interleave lanes across GPUs.  LPT places the largest rows at
            # the lowest lane IDs; contiguous lane IDs would otherwise put
            # several of the largest runs on the same physical H200.
            global_lane = lane_partition_index(
                global_gpu_index=global_gpu_index,
                global_gpu_count=args.global_gpu_count,
                lane_within_gpu=lane_within_gpu,
                lanes_per_gpu=args.lanes_per_gpu,
            )
            lane_jobs.append((device_index, global_lane, partitions[global_lane]))
    append_jsonl(
        status_path,
        {
            "event": "allocation_start",
            "device_indices": device_indices,
            "global_gpu_offset": args.global_gpu_offset,
            "global_gpu_count": args.global_gpu_count,
            "lanes_per_gpu": args.lanes_per_gpu,
            "owned_rows": sum(len(rows) for _, _, rows in lane_jobs),
        },
        status_lock,
    )

    stop = threading.Event()
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=len(lane_jobs)) as executor:
        futures = {
            executor.submit(
                run_lane,
                rows=lane_rows,
                device_index=device_index,
                stop=stop,
                common=common,
            ): global_lane
            for device_index, global_lane, lane_rows in lane_jobs
        }
        for future in as_completed(futures):
            error = future.exception()
            if error is not None and first_error is None:
                first_error = error
                stop.set()
                append_jsonl(
                    status_path,
                    {
                        "event": "allocation_failure",
                        "global_lane": futures[future],
                        "error": f"{type(error).__name__}: {error}",
                    },
                    status_lock,
                )
    if first_error is not None:
        raise RuntimeError("one or more cluster lanes failed") from first_error
    append_jsonl(
        status_path,
        {
            "event": "allocation_complete",
            "global_gpu_offset": args.global_gpu_offset,
            "device_indices": device_indices,
        },
        status_lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
