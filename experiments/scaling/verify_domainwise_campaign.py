#!/usr/bin/env python3
"""Strict, side-effect-free validation of domain-wise scaling run artifacts.

The verifier never changes training artifacts or processes.  An optional report
is written atomically to a separate path.  A campaign which is merely unfinished
is a normal result (exit 0 unless ``--require-complete`` is requested).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_constant,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def resolved(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def add_if(reasons: list[str], condition: bool, message: str) -> None:
    if condition:
        reasons.append(message)


def command_value(command: list[Any], flag: str) -> str | None:
    tokens = [str(item) for item in command]
    positions = [i for i, token in enumerate(tokens) if token == flag]
    if len(positions) != 1:
        return None
    pos = positions[0]
    if pos + 1 >= len(tokens) or tokens[pos + 1].startswith("--"):
        return None
    return tokens[pos + 1]


def command_has_once(command: list[Any], flag: str) -> bool:
    return sum(str(item) == flag for item in command) == 1


def validate_matrix_contract(
    matrix_paths: list[Path],
    expected_total_rows: int,
) -> tuple[list[tuple[Path, dict[str, Any], dict[str, Any]]], list[str], dict[str, Any]]:
    rows: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    issues: list[str] = []
    host_ids: set[str] = set()
    run_ids: set[str] = set()
    out_dirs: set[str] = set()
    partition_ids: set[str] = set()
    source_matrices: set[str] = set()
    global_row_counts: set[int] = set()

    for matrix_path in matrix_paths:
        try:
            matrix = load_json(matrix_path)
        except Exception as exc:
            issues.append(f"{matrix_path}: invalid matrix JSON: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(matrix, dict) or not isinstance(matrix.get("metadata"), dict):
            issues.append(f"{matrix_path}: missing metadata object")
            continue
        metadata = matrix["metadata"]
        matrix_rows = matrix.get("rows")
        if not isinstance(matrix_rows, list):
            issues.append(f"{matrix_path}: missing rows list")
            continue
        host_id = str(metadata.get("host_id", ""))
        add_if(issues, metadata.get("status") != "runnable_phase_b", f"{matrix_path}: status is not runnable_phase_b")
        add_if(issues, not host_id, f"{matrix_path}: empty host_id")
        add_if(issues, host_id in host_ids, f"{matrix_path}: duplicate host_id {host_id!r}")
        host_ids.add(host_id)
        try:
            declared_host_rows = int(metadata.get("rows", -1))
        except (TypeError, ValueError):
            declared_host_rows = -1
        add_if(
            issues,
            declared_host_rows != len(matrix_rows),
            f"{matrix_path}: metadata rows does not match rows list",
        )
        if metadata.get("partition_id") is not None:
            partition_ids.add(str(metadata["partition_id"]))
        if metadata.get("partition_source_matrix") is not None:
            source_matrices.add(str(resolved(metadata["partition_source_matrix"])))
        if metadata.get("global_rows") is not None:
            try:
                global_row_counts.add(int(metadata["global_rows"]))
            except (TypeError, ValueError):
                issues.append(f"{matrix_path}: metadata global_rows is not an integer")

        campaign_root = (
            resolved(metadata.get("run_root", "."))
            / str(metadata.get("experiment_name", ""))
        ).resolve()
        expected_runs_root = (campaign_root / "runs").resolve()
        try:
            raw_layout = metadata.get("gpu_layout", {})
            if not isinstance(raw_layout, dict):
                raise TypeError("gpu_layout is not an object")
            layout = {int(key): int(value) for key, value in raw_layout.items()}
        except (TypeError, ValueError) as exc:
            issues.append(f"{matrix_path}: invalid gpu_layout: {exc}")
            layout = {}
        for index, row in enumerate(matrix_rows):
            prefix = f"{matrix_path}: row[{index}]"
            if not isinstance(row, dict):
                issues.append(f"{prefix}: row is not an object")
                continue
            run_id = str(row.get("run_id", ""))
            if not run_id:
                issues.append(f"{prefix}: missing run_id")
                continue
            add_if(issues, run_id in run_ids, f"{prefix}: duplicate run_id {run_id}")
            run_ids.add(run_id)
            try:
                out_dir = resolved(row["out_dir"])
                out_text = str(out_dir)
                add_if(issues, out_text in out_dirs, f"{prefix}: duplicate out_dir {out_dir}")
                out_dirs.add(out_text)
                add_if(
                    issues,
                    out_dir.parent != expected_runs_root,
                    f"{prefix}: out_dir is outside the exact campaign runs directory: {out_dir}",
                )
                gpu = int(row["assigned_gpu"])
                local_slot = int(row["gpu_local_slot"])
                expected_slot = f"{host_id}_gpu{gpu}_slot{local_slot}"
                add_if(issues, str(row.get("assigned_host")) != host_id, f"{prefix}: assigned_host mismatch")
                add_if(issues, str(row.get("worker_slot")) != expected_slot, f"{prefix}: worker_slot mismatch")
                add_if(issues, gpu not in layout, f"{prefix}: assigned GPU absent from layout")
                add_if(
                    issues,
                    gpu in layout and not 0 <= local_slot < layout[gpu],
                    f"{prefix}: gpu_local_slot outside layout",
                )
                n = int(row["parameter_count"])
                d = int(row["target_input_tokens_with_eos"])
                expected_flops = 6 * n * d
                add_if(
                    issues,
                    int(row["target_flops_after_integer_token_rounding"]) != expected_flops,
                    f"{prefix}: target FLOPs != 6*N*D",
                )
                micro = int(row["per_device_train_batch_size"])
                accum = int(row["gradient_accumulation_steps"])
                add_if(
                    issues,
                    micro * accum != int(row["effective_batch_records"]),
                    f"{prefix}: effective batch mismatch",
                )
                add_if(
                    issues,
                    int(row["scheduler_total_steps"]) != int(row["max_steps"]),
                    f"{prefix}: scheduler_total_steps != max_steps",
                )
                add_if(issues, int(row.get("world_size", 0)) != 1, f"{prefix}: world_size != 1")
                add_if(issues, row.get("bpe_unfold_policy") != "none", f"{prefix}: BPE unfold is enabled")
                add_if(
                    issues,
                    row.get("token_stop_authority") != "target_input_tokens_with_eos",
                    f"{prefix}: token stop authority mismatch",
                )
                add_if(issues, row.get("four_epoch_compliant") is not True, f"{prefix}: not four-epoch compliant")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                issues.append(f"{prefix}: invalid runnable fields: {type(exc).__name__}: {exc}")
                continue
            rows.append((matrix_path, metadata, row))

    add_if(issues, len(partition_ids) > 1, f"partition_id mismatch: {sorted(partition_ids)}")
    add_if(issues, len(source_matrices) > 1, f"partition source mismatch: {sorted(source_matrices)}")
    add_if(issues, len(global_row_counts) > 1, f"global_rows mismatch: {sorted(global_row_counts)}")
    declared_total = next(iter(global_row_counts), 0)
    target_total = int(expected_total_rows or declared_total)
    if target_total:
        add_if(issues, len(rows) != target_total, f"combined row count {len(rows)} != expected {target_total}")
    metadata_report = {
        "matrix_paths": [str(path) for path in matrix_paths],
        "hosts": sorted(host_ids),
        "partition_ids": sorted(partition_ids),
        "partition_source_matrices": sorted(source_matrices),
        "declared_global_rows": sorted(global_row_counts),
        "expected_total_rows": target_total or None,
    }
    return rows, issues, metadata_report


def validate_checkpoint(path: Path, *, load_checkpoint: bool) -> list[str]:
    reasons: list[str] = []
    if not path.is_file():
        return [f"missing checkpoint: {path}"]
    try:
        add_if(reasons, path.stat().st_size < 1024, f"checkpoint is implausibly small: {path}")
    except OSError as exc:
        return [f"cannot stat checkpoint: {type(exc).__name__}: {exc}"]
    if not load_checkpoint or reasons:
        return reasons
    try:
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict) or not state:
            reasons.append("checkpoint is not a non-empty state dictionary")
        else:
            tensors = [(name, value) for name, value in state.items() if torch.is_tensor(value)]
            if not tensors:
                reasons.append("checkpoint contains no tensors")
            for name, tensor in tensors:
                if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
                    reasons.append(f"checkpoint tensor is non-finite: {name}")
                    break
    except Exception as exc:
        reasons.append(f"checkpoint load failed: {type(exc).__name__}: {exc}")
    return reasons


def validate_completed_row(
    matrix_path: Path,
    metadata: dict[str, Any],
    row: dict[str, Any],
    *,
    load_checkpoint: bool,
) -> dict[str, Any]:
    run_id = str(row["run_id"])
    run_dir = resolved(row["out_dir"])
    marker_path = run_dir / "complete.json"
    attempts = sorted(path for path in run_dir.glob("attempt_*") if path.is_dir()) if run_dir.is_dir() else []
    base = {
        "run_id": run_id,
        "host_id": str(metadata.get("host_id")),
        "worker_slot": str(row.get("worker_slot")),
        "run_dir": str(run_dir),
        "attempt_count": len(attempts),
    }
    if not marker_path.is_file():
        orphan_summaries = [str(path / "summary.json") for path in attempts if (path / "summary.json").is_file()]
        return {
            **base,
            "status": "pending",
            "reasons": (["summary exists without strict complete marker"] if orphan_summaries else []),
            "orphan_summaries": orphan_summaries,
        }

    reasons: list[str] = []
    try:
        marker = load_json(marker_path)
    except Exception as exc:
        return {**base, "status": "rejected", "reasons": [f"invalid complete marker: {type(exc).__name__}: {exc}"]}
    if not isinstance(marker, dict):
        return {**base, "status": "rejected", "reasons": ["complete marker is not an object"]}
    add_if(reasons, marker.get("status") != "complete", "complete marker status mismatch")
    add_if(reasons, str(marker.get("run_id", "")) != run_id, "complete marker run_id mismatch")
    try:
        attempt_dir = resolved(marker["attempt_dir"])
        add_if(reasons, attempt_dir.parent != run_dir, "marked attempt is outside the exact run directory")
        add_if(reasons, not ATTEMPT_RE.fullmatch(attempt_dir.name), "marked attempt name is invalid")
    except Exception as exc:
        return {**base, "status": "rejected", "reasons": reasons + [f"invalid marked attempt: {exc}"]}
    summary_path = attempt_dir / "summary.json"
    try:
        marked_summary_path = resolved(marker["summary"])
        add_if(reasons, marked_summary_path != summary_path, "complete marker summary path mismatch")
    except Exception as exc:
        reasons.append(f"invalid complete marker summary path: {exc}")

    artifact_paths = {
        "summary": summary_path,
        "scaling_config": attempt_dir / "scaling_config.json",
        "run_config": attempt_dir / "run_config.json",
        "model_config": attempt_dir / "final" / "config.json",
        "checkpoint": attempt_dir / "final" / "pytorch_model.bin",
    }
    artifacts: dict[str, Any] = {}
    for name, path in artifact_paths.items():
        if name == "checkpoint":
            continue
        try:
            artifacts[name] = load_json(path)
        except Exception as exc:
            reasons.append(f"invalid {name}: {type(exc).__name__}: {exc}")

    scaling = artifacts.get("scaling_config")
    if isinstance(scaling, dict):
        add_if(reasons, scaling.get("matrix_row") != row, "scaling_config matrix_row differs from frozen row")
        try:
            add_if(reasons, resolved(scaling.get("matrix_json")) != matrix_path, "scaling_config matrix path mismatch")
        except Exception:
            reasons.append("scaling_config matrix path is invalid")
        add_if(reasons, str(scaling.get("worker_slot")) != str(row["worker_slot"]), "scaling_config worker slot mismatch")
        add_if(reasons, str(scaling.get("host_id")) != str(metadata["host_id"]), "scaling_config host mismatch")
        add_if(reasons, int(scaling.get("physical_gpu", -1)) != int(row["assigned_gpu"]), "scaling_config GPU mismatch")
        add_if(reasons, int(scaling.get("visible_cuda_device", -1)) != 0, "visible CUDA device is not zero")
        command = scaling.get("command")
        if not isinstance(command, list):
            reasons.append("scaling_config command is not a list")
        else:
            expected_values = {
                "--out-dir": str(attempt_dir),
                "--domain-name": str(row["domain"]),
                "--domain-split-manifest": str(row["domain_split_manifest"]),
                "--vocab-size": str(row["vocab_size"]),
                "--online-pad-length": str(row["context_length"]),
                "--hidden-size": str(row["hidden_size"]),
                "--intermediate-size": str(row["intermediate_size"]),
                "--num-hidden-layers": str(row["num_hidden_layers"]),
                "--num-attention-heads": str(row["num_attention_heads"]),
                "--num-key-value-heads": str(row["num_key_value_heads"]),
                "--head-dim": str(row["head_dim"]),
                "--batch-size": str(row["per_device_train_batch_size"]),
                "--grad-accum": str(row["gradient_accumulation_steps"]),
                "--max-steps": str(row["scheduler_total_steps"]),
                "--target-input-tokens": str(row["target_input_tokens_with_eos"]),
                "--seed": str(row["seed"]),
            }
            for flag, expected in expected_values.items():
                add_if(reasons, command_value(command, flag) != expected, f"command {flag} mismatch")
            for flag in ("--online-truncate-long-samples", "--online-shuffle", "--final-eval", "--bf16"):
                add_if(reasons, not command_has_once(command, flag), f"command is missing unique {flag}")

    run_config = artifacts.get("run_config")
    if isinstance(run_config, dict):
        run_args = run_config.get("args")
        if not isinstance(run_args, dict):
            reasons.append("run_config args is not an object")
        else:
            expected_args: dict[str, Any] = {
                "out_dir": str(attempt_dir),
                "vocab_size": int(row["vocab_size"]),
                "dataset_mode": "online-pad",
                "model_implementation": "packed-xformers",
                "domain_split_manifest": str(row["domain_split_manifest"]),
                "domain_name": str(row["domain"]),
                "online_pad_length": int(row["context_length"]),
                "online_truncate_long_samples": True,
                "online_shuffle": True,
                "token_offset": 3,
                "bpe_unfold_policy": "none",
                "bpe_unfold_prob": 0.0,
                "hidden_size": int(row["hidden_size"]),
                "intermediate_size": int(row["intermediate_size"]),
                "num_hidden_layers": int(row["num_hidden_layers"]),
                "num_attention_heads": int(row["num_attention_heads"]),
                "num_key_value_heads": int(row["num_key_value_heads"]),
                "head_dim": int(row["head_dim"]),
                "max_position_embeddings": int(row["context_length"]),
                "batch_size": int(row["per_device_train_batch_size"]),
                "grad_accum": int(row["gradient_accumulation_steps"]),
                "max_steps": int(row["scheduler_total_steps"]),
                "target_input_tokens": int(row["target_input_tokens_with_eos"]),
                "warmup_ratio": float(row["warmup_ratio"]),
                "seed": int(row["seed"]),
                "optimizer": "adamw-fused",
                "bf16": True,
                "gradient_checkpointing": False,
                "final_eval": True,
                "final_eval_batches": 0,
                "save_every": 0,
                "resume_global_step": 0,
            }
            for key, expected in expected_args.items():
                add_if(reasons, run_args.get(key) != expected, f"run_config args.{key} mismatch")
        add_if(reasons, int(run_config.get("world_size", 0)) != 1, "run_config world_size mismatch")
        add_if(reasons, int(run_config.get("parameter_count", -1)) != int(row["parameter_count"]), "run_config parameter count mismatch")
        add_if(reasons, int(run_config.get("train_items", -1)) != int(row["train_records"]), "run_config train item count mismatch")
        add_if(
            reasons,
            int(run_config.get("train_tokens_visible", -1)) != int(row["train_input_tokens_per_epoch"]),
            "run_config train token count mismatch",
        )
        add_if(reasons, not str(run_config.get("init_source", "")).startswith("random initialization"), "model was not randomly initialized")

    model_config = artifacts.get("model_config")
    if isinstance(model_config, dict):
        for key in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        ):
            expected_key = "context_length" if key == "max_position_embeddings" else key
            add_if(reasons, int(model_config.get(key, -1)) != int(row[expected_key]), f"model config {key} mismatch")

    summary = artifacts.get("summary")
    if isinstance(summary, dict):
        target = int(row["target_input_tokens_with_eos"])
        actual = int(summary.get("actual_input_tokens", -1))
        context = int(row["context_length"])
        n_params = int(row["parameter_count"])
        add_if(reasons, int(summary.get("target_input_tokens", -1)) != target, "summary target input tokens mismatch")
        add_if(reasons, actual < target or actual - target >= context, "summary actual input token budget/overshoot mismatch")
        add_if(reasons, int(summary.get("parameter_count", -1)) != n_params, "summary parameter count mismatch")
        add_if(reasons, int(summary.get("actual_flops_6nd", -1)) != 6 * n_params * actual, "summary actual FLOPs mismatch")
        add_if(reasons, int(summary.get("target_flops_6nd", -1)) != 6 * n_params * target, "summary target FLOPs mismatch")
        add_if(reasons, int(summary.get("scheduler_total_steps", -1)) != int(row["scheduler_total_steps"]), "summary scheduler horizon mismatch")
        add_if(reasons, int(summary.get("warmup_steps", -1)) != int(row["warmup_steps"]), "summary warmup mismatch")
        add_if(reasons, summary.get("termination_reason") != "target_input_tokens", "summary termination reason mismatch")
        add_if(
            reasons,
            int(summary.get("train_input_tokens_per_epoch", -1)) != int(row["train_input_tokens_per_epoch"]),
            "summary train epoch token count mismatch",
        )
        for key in ("final_logged_train_loss", "mean_train_loss"):
            add_if(reasons, not finite_number(summary.get(key)), f"summary {key} is missing/non-finite")
        final_validation = summary.get("final_validation")
        if not isinstance(final_validation, dict):
            reasons.append("summary final_validation is missing")
        else:
            add_if(reasons, final_validation.get("complete_view") is not True, "validation is not a complete view")
            add_if(
                reasons,
                int(final_validation.get("tokens", -1)) != int(row["validation_loss_tokens"]),
                "validation loss-token count mismatch",
            )
            for key in ("loss", "ppl"):
                add_if(reasons, not finite_number(final_validation.get(key)), f"validation {key} is missing/non-finite")
            if isinstance(run_config, dict):
                expected_val_input = int(run_config.get("val_tokens_visible", -1))
                expected_val_samples = int(run_config.get("val_items", -1))
                add_if(
                    reasons,
                    int(final_validation.get("input_tokens", -1)) != expected_val_input,
                    "validation input-token count mismatch",
                )
                add_if(
                    reasons,
                    int(final_validation.get("expected_input_tokens", -1)) != expected_val_input,
                    "validation expected input-token count mismatch",
                )
                add_if(
                    reasons,
                    int(final_validation.get("samples", -1)) != expected_val_samples,
                    "validation sample count mismatch",
                )
            coverage = final_validation.get("input_token_coverage")
            add_if(
                reasons,
                not finite_number(coverage) or not math.isclose(float(coverage), 1.0, rel_tol=0.0, abs_tol=1e-12),
                "validation input coverage is not exactly one",
            )
        try:
            add_if(
                reasons,
                resolved(summary.get("final_checkpoint")) != attempt_dir / "final",
                "summary final checkpoint path mismatch",
            )
        except Exception:
            reasons.append("summary final checkpoint path is invalid")

    reasons.extend(validate_checkpoint(artifact_paths["checkpoint"], load_checkpoint=load_checkpoint))
    return {
        **base,
        "status": "accepted" if not reasons else "rejected",
        "attempt_dir": str(attempt_dir),
        "reasons": reasons,
        "checkpoint_loaded": bool(load_checkpoint and not any("checkpoint" in item for item in reasons)),
    }


def verify_campaign(
    matrix_paths: Iterable[Path],
    *,
    expected_total_rows: int = 0,
    load_checkpoints: bool = False,
) -> dict[str, Any]:
    paths = [path.expanduser().resolve() for path in matrix_paths]
    if not paths:
        raise ValueError("at least one matrix path is required")
    rows, matrix_issues, metadata = validate_matrix_contract(paths, expected_total_rows)
    runs: list[dict[str, Any]] = []
    for path, meta, row in rows:
        try:
            runs.append(validate_completed_row(path, meta, row, load_checkpoint=load_checkpoints))
        except Exception as exc:
            # A corrupt or adversarial artifact must invalidate one row, never
            # crash the campaign-wide verifier and hide the remaining results.
            runs.append(
                {
                    "run_id": str(row.get("run_id", "<missing>")),
                    "host_id": str(meta.get("host_id", "<missing>")),
                    "worker_slot": str(row.get("worker_slot", "<missing>")),
                    "run_dir": str(row.get("out_dir", "<missing>")),
                    "status": "rejected",
                    "reasons": [f"artifact validation raised {type(exc).__name__}: {exc}"],
                }
            )
    counts = {status: sum(run["status"] == status for run in runs) for status in ("accepted", "pending", "rejected")}
    if matrix_issues or counts["rejected"]:
        status = "invalid"
    elif counts["pending"]:
        status = "incomplete"
    else:
        status = "complete"
    return {
        "schema_version": "domainwise_campaign_verification_v1",
        "status": status,
        "complete": status == "complete",
        "checkpoint_load_validation": bool(load_checkpoints),
        "matrix": metadata,
        "matrix_issues": matrix_issues,
        "counts": {"rows": len(runs), **counts},
        "runs": runs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, action="append", required=True)
    ap.add_argument("--expected-total-rows", type=int, default=0)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Do structural checks only; final acceptance should load checkpoint tensors.",
    )
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    result = verify_campaign(
        args.matrix_json,
        expected_total_rows=args.expected_total_rows,
        load_checkpoints=not args.skip_checkpoint_load,
    )
    if args.output_json is not None:
        atomic_write_json(args.output_json.expanduser().resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "matrix_issues": result["matrix_issues"],
                "checkpoint_load_validation": result["checkpoint_load_validation"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 1 if args.require_complete and not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
