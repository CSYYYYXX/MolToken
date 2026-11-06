#!/usr/bin/env python3
"""Run an isolated real-training batch/concurrency tuning probe.

Outputs never enter a formal phase run directory.  Each replica executes real
packed-xformers forward/backward passes and fused-AdamW optimizer steps using
the frozen train split; validation and test are not read.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_domainwise_batch_profile_smoke import (
    require_target_gpu_idle_by_uuid,
    smoke_one_requirement,
    validate_design_matrix,
)
from run_domainwise_phase_slot import (
    atomic_json,
    load_json,
    manifest_semantic_contract,
    safe_component,
    validate_manifest_for_matrix,
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(raw[raw.rfind(")") + 2 :].split()[19])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-matrix", type=Path, required=True)
    parser.add_argument("--domain-split-manifest", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-json", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--abort-temperature-c", type=float, default=90.0)
    parser.add_argument("--require-detached", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domain = safe_component(args.domain, "--domain")
    if min(args.micro_batch, args.grad_accum, args.concurrency) <= 0:
        raise ValueError("batch and concurrency values must be positive")
    if args.optimizer_steps < 2:
        raise ValueError("--optimizer-steps must be at least 2")

    def request_stop(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"tuning interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    identity = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "start_ticks": start_ticks(os.getpid()),
    }
    if args.require_detached and not (
        identity["ppid"] == 1
        and identity["pgid"] == identity["pid"]
        and identity["sid"] == identity["pid"]
    ):
        raise RuntimeError(f"tuning job is not a detached session leader: {identity}")

    matrix_path = args.design_matrix.expanduser().resolve()
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError("design matrix root must be an object")
    validate_design_matrix(matrix, "c")
    manifest_path = args.domain_split_manifest.expanduser().resolve()
    manifest = validate_manifest_for_matrix(matrix, manifest_path, check_files=True)
    if manifest_semantic_contract(manifest) != matrix["metadata"]["domain_split_semantic_contract"]:
        raise ValueError("staged manifest differs from design semantics")
    candidates = [
        row
        for row in matrix["rows"]
        if row.get("domain") == domain
        and (args.model_id is None or row.get("model_id") == args.model_id)
    ]
    if not candidates:
        raise ValueError(
            f"no Phase-C representative row for domain={domain} model_id={args.model_id}"
        )
    row = copy.deepcopy(max(candidates, key=lambda value: int(value["parameter_count"])))
    effective = int(row["effective_batch_records"])
    if args.micro_batch * args.grad_accum != effective:
        raise ValueError(
            f"probe changes effective batch: {args.micro_batch}*{args.grad_accum}!={effective}"
        )
    row.update(
        {
            "assigned_host": "local",
            "assigned_gpu": int(args.physical_gpu),
            "worker_slot": f"tuning_gpu{args.physical_gpu}_slot0",
            "gpu_local_slot": 0,
            "per_device_train_batch_size": int(args.micro_batch),
            "gradient_accumulation_steps": int(args.grad_accum),
            "effective_batch_records": effective,
            "expected_input_tokens_per_optimizer_step": (
                float(row["train_input_tokens_per_epoch"])
                / int(row["train_records"])
                * effective
            ),
        }
    )
    profile_id = safe_component(
        f"tune_{domain}_g{args.physical_gpu}_m{args.micro_batch}_a{args.grad_accum}_c{args.concurrency}",
        "profile_id",
    )
    requirement: dict[str, Any] = {
        "host_id": "local",
        "domain": domain,
        "hardware_class": f"tuning_{args.concurrency}way",
        "batch_profile_id": profile_id,
        "per_device_train_batch_size": int(args.micro_batch),
        "gradient_accumulation_steps": int(args.grad_accum),
        "effective_batch_records": effective,
        "minimum_parameter_count_tested": int(row["parameter_count"]),
        "minimum_device_memory_gib": 48 if args.physical_gpu >= 4 else 24,
        "minimum_concurrent_processes": int(args.concurrency),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tuning output: {output}")
    started_at = now()
    atomic_json(
        args.identity_json.expanduser().resolve(),
        {
            "schema_version": "domainwise_batch_tuning_identity_v1",
            "status": "running",
            **identity,
            "domain": domain,
            "physical_gpu": args.physical_gpu,
            "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "concurrency": args.concurrency,
            "optimizer_steps": args.optimizer_steps,
            "representative_run_id": row["run_id"],
            "parameter_count": row["parameter_count"],
            "started_at": started_at,
        },
    )
    try:
        require_target_gpu_idle_by_uuid(args.physical_gpu)
        entry = smoke_one_requirement(
            requirement=requirement,
            row=row,
            matrix_path=matrix_path,
            manifest=manifest,
            manifest_path=manifest_path,
            pipeline_root=args.pipeline_root.expanduser().resolve(),
            # Preserve the virtual-environment launcher path.  ``resolve()``
            # would follow its python symlink into the base Conda environment
            # and silently drop venv-local packages such as xformers.
            python=args.python.expanduser().absolute(),
            wrapper=Path(__file__).resolve().with_name("run_profile_smoke_train_wrapper.py"),
            output_root=output,
            matrix_generation_id=str(matrix["metadata"]["generation_id"]),
            poll_seconds=0.25,
            abort_temperature_c=args.abort_temperature_c,
            optimizer_steps=args.optimizer_steps,
        )
        status = "complete"
        error = None
    except Exception as exc:
        entry = None
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(
            args.identity_json.expanduser().resolve(),
            {
                "schema_version": "domainwise_batch_tuning_identity_v1",
                "status": status,
                **identity,
                "domain": domain,
                "physical_gpu": args.physical_gpu,
                "micro_batch": args.micro_batch,
                "grad_accum": args.grad_accum,
                "concurrency": args.concurrency,
                "optimizer_steps": args.optimizer_steps,
                "representative_run_id": row["run_id"],
                "parameter_count": row["parameter_count"],
                "started_at": started_at,
                "completed_at": now(),
                "error": error,
                "entry": entry,
            },
        )
    print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
