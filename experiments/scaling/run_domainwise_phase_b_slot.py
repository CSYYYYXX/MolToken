#!/usr/bin/env python3
"""Run one frozen Phase-B worker slot sequentially on one physical GPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), **value}, ensure_ascii=False) + "\n")
        handle.flush()


def load_complete_summary(attempt_dir: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = attempt_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    target = int(row["target_input_tokens_with_eos"])
    if int(summary.get("target_input_tokens") or -1) != target:
        return None
    actual = int(summary.get("actual_input_tokens") or -1)
    if actual < target:
        return None
    if actual - target >= int(row["context_length"]):
        return None
    if int(summary.get("parameter_count") or -1) != int(row["parameter_count"]):
        return None
    if int(summary.get("actual_flops_6nd") or -1) != 6 * int(row["parameter_count"]) * actual:
        return None
    if int(summary.get("scheduler_total_steps") or -1) != int(row["scheduler_total_steps"]):
        return None
    if int(summary.get("warmup_steps") or -1) != int(row["warmup_steps"]):
        return None
    if int(summary.get("train_input_tokens_per_epoch") or -1) != int(row["train_input_tokens_per_epoch"]):
        return None
    if summary.get("termination_reason") != "target_input_tokens":
        return None
    for key in ("final_logged_train_loss", "trailing_train_loss", "mean_train_loss"):
        value = summary.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
            return None
    final_validation = summary.get("final_validation")
    if not isinstance(final_validation, dict):
        return None
    if final_validation.get("complete_view") is not True:
        return None
    if int(final_validation.get("tokens") or -1) != int(row["validation_loss_tokens"]):
        return None
    for key in ("loss", "ppl", "input_token_coverage"):
        value = final_validation.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
    if float(final_validation["input_token_coverage"]) != 1.0:
        return None
    if int(final_validation.get("input_tokens") or -1) != int(
        final_validation.get("expected_input_tokens") or -2
    ):
        return None
    scaling_config_path = attempt_dir / "scaling_config.json"
    try:
        scaling_config = json.loads(scaling_config_path.read_text(encoding="utf-8"))
        frozen_row = scaling_config["matrix_row"]
    except (KeyError, OSError, ValueError):
        return None
    for key in (
        "run_id",
        "seed",
        "parameter_count",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "target_input_tokens_with_eos",
        "scheduler_total_steps",
        "warmup_steps",
        "domain_split_manifest",
    ):
        if frozen_row.get(key) != row.get(key):
            return None
    final_dir = attempt_dir / "final"
    model_path = final_dir / "pytorch_model.bin"
    if not (final_dir / "config.json").is_file() or not model_path.is_file() or model_path.stat().st_size <= 0:
        return None
    return summary


def completed_attempt(run_dir: Path, row: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    run_dir = run_dir.resolve()
    marker = run_dir / "complete.json"
    if marker.is_file():
        try:
            marked = json.loads(marker.read_text(encoding="utf-8"))
            if marked.get("status") != "complete" or marked.get("run_id") != row.get("run_id"):
                raise ValueError("completion marker identity mismatch")
            attempt = Path(marked["attempt_dir"]).resolve()
            if attempt.parent != run_dir or not attempt.name.startswith("attempt_"):
                raise ValueError("completion marker attempt escapes its run directory")
            if Path(marked.get("summary", "")).resolve() != attempt / "summary.json":
                raise ValueError("completion marker summary path mismatch")
            summary = load_complete_summary(attempt, row)
            if summary is not None:
                return attempt, summary
        except (KeyError, OSError, ValueError):
            pass
    for attempt in sorted(run_dir.glob("attempt_*")):
        if not attempt.is_dir():
            continue
        summary = load_complete_summary(attempt, row)
        if summary is not None:
            return attempt, summary
    return None


def next_attempt_dir(run_dir: Path) -> Path:
    numbers: list[int] = []
    for path in run_dir.glob("attempt_*"):
        try:
            numbers.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return run_dir / f"attempt_{max(numbers, default=0) + 1:03d}"


def training_command(
    python: Path,
    pipeline_root: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    attempt_dir: Path,
    num_workers: int,
    lr: float,
    manifest_path: Path | None = None,
) -> list[str]:
    dataset = manifest["dataset"]
    token_glob = str(Path(dataset["compact_dir"]) / "shard_*.bin")
    selected_manifest_path = (
        manifest_path
        if manifest_path is not None
        else Path(row["domain_split_manifest"])
    )
    return [
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
        str(selected_manifest_path),
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
        str(row["scheduler_total_steps"]),
        "--target-input-tokens",
        str(row["target_input_tokens_with_eos"]),
        "--lr",
        str(lr),
        "--min-lr-ratio",
        "0.1",
        "--weight-decay",
        "0.1",
        "--warmup-ratio",
        str(row["warmup_ratio"]),
        "--log-every",
        "20",
        "--eval-every",
        "0",
        "--eval-batches",
        "0",
        "--final-eval",
        "--final-eval-batches",
        "0",
        "--save-every",
        "0",
        "--num-workers",
        str(num_workers),
        "--seed",
        str(row["seed"]),
        "--optimizer",
        "adamw-fused",
        "--bf16",
        "--no-gradient-checkpointing",
    ]


def main() -> int:
    script_path = Path(__file__).resolve()
    pipeline_root_default = script_path.parents[2]
    workspace_default = pipeline_root_default.parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, required=True)
    ap.add_argument("--worker-slot", required=True)
    ap.add_argument(
        "--domain-split-manifest",
        type=Path,
        default=None,
        help="Host-local manifest override (for example a per-host /dev/shm stage).",
    )
    ap.add_argument("--pipeline-root", type=Path, default=pipeline_root_default)
    ap.add_argument(
        "--python",
        type=Path,
        default=workspace_default / "local_runtime" / "envs" / "holomol-py311-cu121" / "bin" / "python",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-attempts-per-run", type=int, default=2)
    args = ap.parse_args()
    matrix_path = args.matrix_json.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    # Do not resolve the environment's ``bin/python`` symlink: invoking its
    # base-conda target directly bypasses this prefix's site-packages.
    python = args.python.expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix["metadata"].get("status") != "runnable_phase_b":
        raise ValueError("matrix is not frozen runnable_phase_b")
    host_id = str(matrix["metadata"].get("host_id", "local"))
    if not args.worker_slot.startswith(f"{host_id}_"):
        raise ValueError(
            f"worker slot is not qualified for matrix host {host_id!r}: {args.worker_slot}"
        )
    rows = [row for row in matrix["rows"] if row["worker_slot"] == args.worker_slot]
    if not rows:
        raise ValueError(f"worker slot has no rows: {args.worker_slot}")
    if any(str(row.get("assigned_host", host_id)) != host_id for row in rows):
        raise ValueError(f"worker slot includes a row for a different host: {args.worker_slot}")
    gpu_ids = {int(row["assigned_gpu"]) for row in rows}
    if len(gpu_ids) != 1:
        raise ValueError(f"worker slot spans physical GPUs: {gpu_ids}")
    gpu_id = next(iter(gpu_ids))
    manifest_path = (
        args.domain_split_manifest
        if args.domain_split_manifest is not None
        else Path(matrix["metadata"]["domain_split_manifest"])
    ).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("domain split manifest is incomplete")

    campaign_root = Path(matrix["metadata"]["run_root"]) / matrix["metadata"]["experiment_name"]
    status_path = campaign_root / "workers" / args.worker_slot / "status.jsonl"
    lock_root = Path(f"/dev/shm/holomol-{os.getuid()}/scaling")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"domainwise_phase_b_{args.worker_slot}.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"worker slot is already active: {args.worker_slot}") from exc
    lock_handle.write(f"pid={os.getpid()} time={now()} matrix={matrix_path}\n")
    lock_handle.flush()
    append_status(
        status_path,
        {
            "event": "worker_start",
            "host_id": host_id,
            "worker_slot": args.worker_slot,
            "gpu": gpu_id,
            "rows": len(rows),
            "matrix": str(matrix_path),
        },
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTHONPATH": str(pipeline_root)
            + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "2"),
            "MKL_NUM_THREADS": environment.get("MKL_NUM_THREADS", "2"),
            "PYTORCH_CUDA_ALLOC_CONF": environment.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
        }
    )
    failures = 0
    for row in rows:
        if row.get("matrix_status") != "runnable_phase_b":
            raise ValueError(f"non-runnable row reached worker: {row.get('run_id')}")
        if row.get("bpe_unfold_policy") != "none" or int(row.get("world_size", 0)) != 1:
            raise ValueError(f"row violates Phase-B representation/world-size freeze: {row['run_id']}")
        run_dir = Path(row["out_dir"]).resolve()
        expected_parent = (campaign_root / "runs").resolve()
        if not run_dir.is_relative_to(expected_parent):
            raise ValueError(f"run out_dir escapes campaign root: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        done = completed_attempt(run_dir, row)
        if done is not None:
            append_status(
                status_path,
                {"event": "skip_complete", "run_id": row["run_id"], "attempt_dir": str(done[0])},
            )
            continue

        success = False
        for _ in range(args.max_attempts_per_run):
            attempt_dir = next_attempt_dir(run_dir)
            attempt_dir.mkdir(parents=True, exist_ok=False)
            command = training_command(
                python,
                pipeline_root,
                manifest,
                row,
                attempt_dir,
                args.num_workers,
                args.lr,
                manifest_path,
            )
            atomic_json(
                attempt_dir / "scaling_config.json",
                {
                    "matrix_json": str(matrix_path),
                    "matrix_row": row,
                    "worker_slot": args.worker_slot,
                    "host_id": host_id,
                    "physical_gpu": gpu_id,
                    "visible_cuda_device": 0,
                    "domain_split_manifest": str(manifest_path),
                    "attempt_started_at": now(),
                    "command": command,
                },
            )
            append_status(
                status_path,
                {"event": "run_start", "run_id": row["run_id"], "attempt_dir": str(attempt_dir)},
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
            summary = load_complete_summary(attempt_dir, row) if result.returncode == 0 else None
            if summary is not None:
                atomic_json(
                    run_dir / "complete.json",
                    {
                        "status": "complete",
                        "run_id": row["run_id"],
                        "attempt_dir": str(attempt_dir),
                        "summary": str(attempt_dir / "summary.json"),
                        "completed_at": now(),
                    },
                )
                append_status(
                    status_path,
                    {
                        "event": "run_complete",
                        "run_id": row["run_id"],
                        "attempt_dir": str(attempt_dir),
                        "global_step": summary["global_step"],
                        "actual_input_tokens": summary["actual_input_tokens"],
                        "final_validation_loss": summary["final_validation"]["loss"],
                    },
                )
                success = True
                break
            append_status(
                status_path,
                {
                    "event": "run_failed",
                    "run_id": row["run_id"],
                    "attempt_dir": str(attempt_dir),
                    "returncode": result.returncode,
                    "console": str(console_path),
                },
            )
        if not success:
            failures += 1

    append_status(
        status_path,
        {
            "event": "worker_done",
            "worker_slot": args.worker_slot,
            "gpu": gpu_id,
            "failures": failures,
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
