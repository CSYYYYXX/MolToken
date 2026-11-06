#!/usr/bin/env python3
"""Produce real optimizer-step evidence for a Phase-C/D design preview.

``run-host`` executes each profile assigned to one host.  A 48 GiB dual-slot
profile is exercised by two simultaneous real packed-xformers trainers on the
same physical GPU.  ``merge`` combines host fragments and only emits an
approved report after the finalizer's exact profile-coverage validator passes.
No command writes into a formal run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finalize_domainwise_phase_matrix import (
    PROFILE_EVIDENCE_SCHEMA,
    validate_profile_evidence,
)
from run_domainwise_phase_slot import (
    FROZEN_TRAINING_CONTROL,
    atomic_json,
    load_json,
    manifest_semantic_contract,
    matrix_semantic_fingerprint,
    safe_component,
    single_process_environment,
    training_command,
    validate_manifest_for_matrix,
)
from supervise_domainwise_phase_host import (
    active_scaling_processes,
    gpu_inventory,
    running_compute_processes,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def validate_design_matrix(matrix: dict[str, Any], phase: str) -> None:
    metadata = matrix.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("design matrix has no metadata")
    if (
        metadata.get("status") != "design_dry_run_not_runnable"
        or metadata.get("is_runnable") is not False
        or metadata.get("phase") != phase
        or metadata.get("training_control") != FROZEN_TRAINING_CONTROL
    ):
        raise ValueError("smoke accepts only a matching non-runnable C/D design preview")
    fingerprint = metadata.get("semantic_fingerprint")
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("value") != matrix_semantic_fingerprint(matrix)
    ):
        raise ValueError("design matrix semantic identity is inconsistent")
    gate = metadata.get("batch_profile_evidence_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("required") is not True
        or gate.get("satisfied") is not False
        or not isinstance(gate.get("requirements"), list)
        or not gate.get("profile_contract_id")
    ):
        raise ValueError("design matrix has no pending profile-evidence contract")


def replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"expected one command option {flag}")
    command[positions[0] + 1] = value


def remove_option(command: list[str], flag: str, *, takes_value: bool) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1:
        raise ValueError(f"expected one command option {flag}")
    index = positions[0]
    del command[index : index + (2 if takes_value else 1)]


def selected_row(
    matrix: dict[str, Any], requirement: dict[str, Any]
) -> dict[str, Any]:
    candidates = [
        row
        for row in matrix["rows"]
        if row["assigned_host"] == requirement["host_id"]
        and row["domain"] == requirement["domain"]
        and row["assigned_hardware_class"] == requirement["hardware_class"]
        and row["batch_profile_id"] == requirement["batch_profile_id"]
        and int(row["per_device_train_batch_size"])
        == int(requirement["per_device_train_batch_size"])
        and int(row["gradient_accumulation_steps"])
        == int(requirement["gradient_accumulation_steps"])
        and int(row["effective_batch_records"])
        == int(requirement["effective_batch_records"])
        and int(row["parameter_count"])
        >= int(requirement["minimum_parameter_count_tested"])
    ]
    if not candidates:
        raise ValueError(f"no design row implements smoke requirement: {requirement}")
    return max(candidates, key=lambda row: int(row["parameter_count"]))


def process_memory_by_pid() -> dict[int, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    values: dict[int, float] = {}
    for line in result.stdout.splitlines():
        if line.strip():
            pid, memory = [part.strip() for part in line.split(",", 1)]
            values[int(pid)] = float(memory)
    return values


def gpu_status(gpu: int) -> tuple[float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=temperature.gpu,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    temperature, total = [part.strip() for part in result.stdout.strip().split(",", 1)]
    return float(temperature), float(total)


def require_target_gpu_idle_by_uuid(gpu: int) -> str:
    """Fail closed unless ``gpu`` has no compute context in a fresh UUID probe."""

    inventory_rows = [row for row in gpu_inventory() if int(row["index"]) == int(gpu)]
    if len(inventory_rows) != 1 or not str(inventory_rows[0].get("uuid", "")).strip():
        raise RuntimeError(
            f"cannot resolve exactly one UUID for profile smoke target GPU{gpu}: "
            f"{inventory_rows}"
        )
    gpu_uuid = str(inventory_rows[0]["uuid"])
    conflicts = [
        row
        for row in running_compute_processes()
        if str(row.get("gpu_uuid", "")) == gpu_uuid
    ]
    if conflicts:
        raise RuntimeError(
            f"profile smoke target GPU{gpu} UUID {gpu_uuid} is not compute-idle: "
            + json.dumps(conflicts, ensure_ascii=False)
        )
    return gpu_uuid


def terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def kill(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def ensure_safe_temperature(temperature_c: float, abort_temperature_c: float, gpu: int) -> None:
    if float(temperature_c) >= float(abort_temperature_c):
        raise RuntimeError(
            f"profile smoke thermal abort GPU{gpu}: {temperature_c} C"
        )


class SmokeThermalControl:
    """Pause only live, directly spawned replica groups, with hysteresis."""

    def __init__(self, processes, event_path: Path, pause_c: float, resume_c: float):
        self.processes = processes
        self.event_path = event_path
        self.pause_c = pause_c
        self.resume_c = resume_c
        self.paused = False
        self.paused_at = None
        self.pause_count = 0
        self.paused_seconds = 0.0

    def signal_live(self, signum: int) -> None:
        for process in self.processes:
            if process.poll() is not None:
                continue
            try:
                if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != os.getsid(0):
                    raise RuntimeError(f"smoke replica group identity changed: {process.pid}")
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                continue

    def resume(self) -> None:
        if self.paused:
            self.signal_live(signal.SIGCONT)
            self.paused_seconds += time.monotonic() - self.paused_at
            self.paused = False
            self.paused_at = None

    def sample(self, temperature: float) -> None:
        action = None
        if not self.paused and temperature >= self.pause_c:
            # Mark ownership before signaling, so partial failures are resumed
            # by the caller's finally block as well.
            self.paused = True
            self.paused_at = time.monotonic()
            self.signal_live(signal.SIGSTOP)
            self.pause_count += 1
            action = "pause"
        elif self.paused and temperature <= self.resume_c:
            self.resume()
            action = "resume"
        elif self.paused and time.monotonic() - self.paused_at > 120:
            raise RuntimeError("profile smoke did not cool within 120 seconds")
        if action is not None:
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": now(), "action": action, "temperature_c": temperature}) + "\n")


def load_train_rows(run_dirs: list[Path]) -> list[list[dict[str, Any]]]:
    """Load exactly one ordered train-event stream per smoke replica."""

    return [
        [
            value
            for line in (run_dir / "train_log.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
            for value in [json.loads(line)]
            if value.get("event") == "train"
        ]
        for run_dir in run_dirs
    ]


def validate_smoke_measurements(
    *,
    summaries: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    train_rows: list[list[dict[str, Any]]],
    requirement: dict[str, Any],
    console_text: str,
    simultaneous_after_step: bool,
    peak_observed_aggregate: float,
) -> dict[str, Any]:
    concurrency = int(requirement["minimum_concurrent_processes"])
    if concurrency <= 0:
        raise ValueError("smoke requirement has invalid concurrency")
    if not (
        len(summaries) == len(memories) == len(train_rows) == concurrency
    ):
        raise RuntimeError(
            "profile smoke artifacts do not match the required process count"
        )
    if any(not rows for rows in train_rows):
        raise RuntimeError("profile smoke has no per-step train telemetry")
    required_grad_accum = int(requirement["gradient_accumulation_steps"])
    full_windows = [
        any(int(item.get("optimizer_micro_steps", 0)) == required_grad_accum for item in rows)
        for rows in train_rows
    ]
    if any(
        not isinstance(summary, dict)
        or int(summary.get("global_step", 0)) < 1
        or not math.isfinite(float(summary.get("mean_train_loss", math.nan)))
        or float(summary.get("mean_train_loss", 0)) <= 0
        or summary.get("model_implementation") != "packed-xformers"
        or summary.get("attention_backend")
        != "xformers_memory_efficient_block_diagonal_causal"
        for summary in summaries
    ):
        raise RuntimeError("profile smoke did not yield finite packed-xformers steps")
    if not all(full_windows):
        raise RuntimeError("profile smoke never completed a full gradient-accumulation window")
    lower_console = console_text.lower()
    if "outofmemoryerror" in lower_console or "cuda out of memory" in lower_console:
        raise RuntimeError("profile smoke observed OOM")
    if concurrency >= 2 and not simultaneous_after_step:
        raise RuntimeError("dual profile never overlapped after both optimizer steps")
    train_peak_allocated = max(
        (
            float(item.get("cuda_memory_gb_rank0", {}).get("max_allocated", 0.0))
            * 1024
            for rows in train_rows
            for item in rows
        ),
        default=0.0,
    )
    train_peak_reserved = max(
        (
            float(item.get("cuda_memory_gb_rank0", {}).get("max_reserved", 0.0))
            * 1024
            for rows in train_rows
            for item in rows
        ),
        default=0.0,
    )
    return {
        "complete_optimizer_steps": min(
            int(summary["global_step"]) for summary in summaries
        ),
        "backward_passes": min(
            sum(int(item["optimizer_micro_steps"]) for item in rows)
            for rows in train_rows
        ),
        "maximum_optimizer_micro_steps": min(
            max(int(item["optimizer_micro_steps"]) for item in rows)
            for rows in train_rows
        ),
        "full_gradient_window_verified": True,
        "peak_allocated_mib": max(
            train_peak_allocated,
            max(float(report["peak_allocated_mib"]) for report in memories),
        ),
        "peak_reserved_mib": max(
            train_peak_reserved,
            max(float(report["peak_reserved_mib"]) for report in memories),
        ),
        "device_total_mib": min(
            float(report["device_total_mib"]) for report in memories
        ),
        "aggregate_peak_reserved_mib": float(peak_observed_aggregate),
    }


def smoke_one_requirement(
    *,
    requirement: dict[str, Any],
    row: dict[str, Any],
    matrix_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    pipeline_root: Path,
    python: Path,
    wrapper: Path,
    output_root: Path,
    matrix_generation_id: str,
    poll_seconds: float,
    abort_temperature_c: float,
    optimizer_steps: int = 2,
) -> dict[str, Any]:
    profile_name = (
        f"{requirement['host_id']}__{requirement['domain']}__"
        f"{requirement['hardware_class']}__{requirement['batch_profile_id']}"
    )
    root = output_root / profile_name
    if root.exists():
        raise FileExistsError(f"refusing to overwrite smoke evidence: {root}")
    root.mkdir(parents=True)
    concurrency = int(requirement["minimum_concurrent_processes"])
    gpu = int(row["assigned_gpu"])
    expected_tokens = float(row["expected_input_tokens_per_optimizer_step"])
    if int(optimizer_steps) < 2:
        raise ValueError("profile smoke requires at least two optimizer steps")
    # Two updates are the formal minimum.  Tuning runs may request more to
    # measure steady-state throughput and thermals while exercising the same
    # full optimizer path.
    target_tokens = max(1, int(math.ceil(expected_tokens * int(optimizer_steps))))
    processes: list[subprocess.Popen[bytes]] = []
    consoles: list[Any] = []
    run_dirs: list[Path] = []
    memory_paths: list[Path] = []
    environment = single_process_environment(os.environ)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
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
    for replica in range(concurrency):
        run_dir = root / f"replica_{replica}"
        memory_path = run_dir / "cuda_memory.json"
        base = training_command(
            python,
            pipeline_root,
            manifest,
            row,
            run_dir,
            int(requirement["per_device_train_batch_size"]),
            int(requirement["gradient_accumulation_steps"]),
            0,
            manifest_path,
        )
        trainer_args = base[2:]
        replace_option(trainer_args, "--target-input-tokens", str(target_tokens))
        replace_option(trainer_args, "--max-steps", str(int(optimizer_steps)))
        replace_option(trainer_args, "--log-every", "1")
        replace_option(trainer_args, "--seed", str(42 + replica))
        remove_option(trainer_args, "--final-eval", takes_value=False)
        remove_option(trainer_args, "--final-eval-batches", takes_value=True)
        command = [
            str(python),
            str(wrapper),
            "--memory-report",
            str(memory_path),
            "--",
            *trainer_args,
        ]
        run_dir.mkdir(parents=True)
        atomic_json(
            run_dir / "smoke_contract.json",
            {
                "schema_version": "domainwise_profile_smoke_contract_v1",
                "matrix_json": str(matrix_path),
                "matrix_generation_id": matrix_generation_id,
                "requirement": requirement,
                "representative_run_id": row["run_id"],
                "parameter_count": row["parameter_count"],
                "physical_gpu": gpu,
                "replica": replica,
                "command": command,
                "started_at": now(),
            },
        )
        console = (run_dir / "console.log").open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=pipeline_root,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            # Keep every profile worker in the controller-owned detached job
            # session while retaining a separate process group for this
            # replica's existing terminate/kill handling.  If the detached
            # wrapper is lost, its durable SID still identifies all workers.
            process_group=0,
        )
        consoles.append(console)
        processes.append(process)
        run_dirs.append(run_dir)
        memory_paths.append(memory_path)

    simultaneous_after_step = concurrency == 1
    peak_observed_aggregate = 0.0
    peak_temperature = 0.0
    thermal = SmokeThermalControl(
        processes, root / "thermal_events.jsonl",
        min(85.0, abort_temperature_c - 3.0),
        min(79.0, abort_temperature_c - 9.0),
    )
    try:
        while any(process.poll() is None for process in processes):
            temperature, _ = gpu_status(gpu)
            peak_temperature = max(peak_temperature, temperature)
            ensure_safe_temperature(temperature, abort_temperature_c, gpu)
            thermal.sample(temperature)
            memory = process_memory_by_pid()
            alive_memory = [memory.get(process.pid, 0.0) for process in processes]
            peak_observed_aggregate = max(peak_observed_aggregate, sum(alive_memory))
            if concurrency >= 2 and all(value > 0 for value in alive_memory):
                train_steps_exist = all(
                    (run_dir / "train_log.jsonl").is_file()
                    and '"event": "train"'
                    in (run_dir / "train_log.jsonl").read_text(
                        encoding="utf-8", errors="replace"
                    )
                    for run_dir in run_dirs
                )
                simultaneous_after_step = simultaneous_after_step or train_steps_exist
            time.sleep(poll_seconds)
    except BaseException:
        thermal.resume()
        for process in processes:
            terminate(process)
        raise
    finally:
        thermal.resume()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate(process)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        kill(process)
                        process.wait(timeout=10)
        for console in consoles:
            console.close()

    returncodes = [int(process.returncode or 0) for process in processes]
    if any(code != 0 for code in returncodes):
        raise RuntimeError(f"profile smoke trainer failed: returncodes={returncodes} root={root}")
    summaries = [load_json(run_dir / "summary.json") for run_dir in run_dirs]
    memories = [load_json(path) for path in memory_paths]
    train_rows = load_train_rows(run_dirs)
    console_text = "\n".join(
        (run_dir / "console.log").read_text(encoding="utf-8", errors="replace").lower()
        for run_dir in run_dirs
    )
    measurements = validate_smoke_measurements(
        summaries=summaries,
        memories=memories,
        train_rows=train_rows,
        requirement=requirement,
        console_text=console_text,
        simultaneous_after_step=simultaneous_after_step,
        peak_observed_aggregate=peak_observed_aggregate,
    )
    entry = {
        "host_id": requirement["host_id"],
        "domain": requirement["domain"],
        "hardware_class": requirement["hardware_class"],
        "batch_profile_id": requirement["batch_profile_id"],
        "per_device_train_batch_size": requirement["per_device_train_batch_size"],
        "gradient_accumulation_steps": requirement["gradient_accumulation_steps"],
        "effective_batch_records": requirement["effective_batch_records"],
        "parameter_count_tested": int(row["parameter_count"]),
        "evidence_kind": "optimizer_step_smoke",
        "complete_optimizer_steps": measurements["complete_optimizer_steps"],
        "backward_passes": measurements["backward_passes"],
        "maximum_optimizer_micro_steps": measurements[
            "maximum_optimizer_micro_steps"
        ],
        "full_gradient_window_verified": measurements[
            "full_gradient_window_verified"
        ],
        "finite_loss": True,
        "oom_observed": False,
        "peak_allocated_mib": measurements["peak_allocated_mib"],
        "peak_reserved_mib": measurements["peak_reserved_mib"],
        "device_total_mib": measurements["device_total_mib"],
        "concurrent_processes": concurrency,
        "simultaneous_execution_verified": simultaneous_after_step,
        "aggregate_peak_reserved_mib": measurements[
            "aggregate_peak_reserved_mib"
        ],
        "physical_gpu": gpu,
        "peak_temperature_c": peak_temperature,
        "thermal_pause_count": thermal.pause_count,
        "thermal_paused_seconds": thermal.paused_seconds,
        "source_artifacts": [
            str(path)
            for run_dir, memory_path in zip(run_dirs, memory_paths)
            for path in (run_dir / "summary.json", memory_path, run_dir / "smoke_contract.json")
        ],
    }
    atomic_json(root / "evidence_entry.json", entry)
    return entry


def run_host(args: argparse.Namespace) -> int:
    matrix_path = args.design_matrix.expanduser().resolve()
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError("design matrix root must be an object")
    validate_design_matrix(matrix, args.phase)
    host = safe_component(args.host_id, "--host-id")
    requirements = [
        requirement
        for requirement in matrix["metadata"]["batch_profile_evidence_gate"]["requirements"]
        if requirement["host_id"] == host
    ]
    if not requirements:
        raise ValueError(f"design preview has no smoke requirements for host {host!r}")
    output = args.output.expanduser().resolve()
    phase_root = Path(matrix["metadata"]["output_base"]).expanduser().resolve()
    if output == phase_root or output.is_relative_to(phase_root / "runs"):
        raise ValueError("profile smoke output must not use the formal phase run directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite host smoke fragment root: {output}")
    manifest_path = args.domain_split_manifest.expanduser().resolve()
    manifest = validate_manifest_for_matrix(matrix, manifest_path, check_files=True)
    if manifest_semantic_contract(manifest) != matrix["metadata"]["domain_split_semantic_contract"]:
        raise ValueError("host smoke manifest differs from design matrix")
    allow_disjoint_overlap = bool(
        getattr(args, "allow_existing_scaling_processes_on_other_gpus", False)
    )
    scaling_processes = active_scaling_processes()
    compute_processes = running_compute_processes()
    if not allow_disjoint_overlap and (scaling_processes or compute_processes):
        raise RuntimeError("profile smoke requires an idle host with no scaling/compute processes")
    output.mkdir(parents=True)
    entries = []
    for requirement in requirements:
        row = selected_row(matrix, requirement)
        # In disjoint-overlap mode, other GPUs may remain busy with an existing
        # scaling campaign.  In every mode, re-resolve the physical UUID and
        # re-query compute contexts immediately before each requirement so a
        # later requirement cannot rely on a stale host-level observation.
        require_target_gpu_idle_by_uuid(int(row["assigned_gpu"]))
        entries.append(
            smoke_one_requirement(
                requirement=requirement,
                row=row,
                matrix_path=matrix_path,
                manifest=manifest,
                manifest_path=manifest_path,
                pipeline_root=args.pipeline_root.expanduser().resolve(),
                python=args.python.expanduser().absolute(),
                wrapper=Path(__file__).resolve().with_name(
                    "run_profile_smoke_train_wrapper.py"
                ),
                output_root=output / "profiles",
                matrix_generation_id=matrix["metadata"]["generation_id"],
                poll_seconds=args.poll_seconds,
                abort_temperature_c=args.abort_temperature_c,
            )
        )
    fragment = {
        "schema_version": "domainwise_batch_profile_evidence_host_fragment_v1",
        "status": "complete",
        "phase": args.phase,
        "host_id": host,
        "hardware_layout_id": matrix["metadata"]["hardware_layout"]["layout_id"],
        "profile_contract_id": matrix["metadata"]["batch_profile_evidence_gate"][
            "profile_contract_id"
        ],
        "design_matrix": str(matrix_path),
        "design_generation_id": matrix["metadata"]["generation_id"],
        "test_data_accessed": False,
        "entries": entries,
        "completed_at": now(),
    }
    atomic_json(output / "host_fragment.json", fragment)
    print(json.dumps({"host_fragment": str(output / "host_fragment.json"), "entries": len(entries)}))
    return 0


def merge(args: argparse.Namespace) -> int:
    matrix_path = args.design_matrix.expanduser().resolve()
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError("design matrix root must be an object")
    validate_design_matrix(matrix, args.phase)
    entries: list[dict[str, Any]] = []
    seen_hosts: set[str] = set()
    for fragment_path_raw in args.fragments:
        fragment_path = fragment_path_raw.expanduser().resolve()
        fragment = load_json(fragment_path)
        if (
            not isinstance(fragment, dict)
            or fragment.get("schema_version")
            != "domainwise_batch_profile_evidence_host_fragment_v1"
            or fragment.get("status") != "complete"
            or fragment.get("phase") != args.phase
            or fragment.get("profile_contract_id")
            != matrix["metadata"]["batch_profile_evidence_gate"]["profile_contract_id"]
            or fragment.get("design_generation_id") != matrix["metadata"]["generation_id"]
            or fragment.get("test_data_accessed") is not False
        ):
            raise ValueError(f"incompatible host evidence fragment: {fragment_path}")
        host = str(fragment["host_id"])
        if host in seen_hosts:
            raise ValueError(f"duplicate host evidence fragment: {host}")
        seen_hosts.add(host)
        entries.extend(fragment["entries"])
    gate = matrix["metadata"]["batch_profile_evidence_gate"]
    report = {
        "schema_version": PROFILE_EVIDENCE_SCHEMA,
        "status": "approved_for_formal_freeze",
        "phase": args.phase,
        "hardware_layout_id": matrix["metadata"]["hardware_layout"]["layout_id"],
        "profile_contract_id": gate["profile_contract_id"],
        "design_matrix": str(matrix_path),
        "design_generation_id": matrix["metadata"]["generation_id"],
        "test_data_accessed": False,
        "producer": {
            "script": Path(__file__).name,
            "method_version": "real_packed_xformers_full_optimizer_step_v1",
        },
        "entries": entries,
        "created_at": now(),
    }
    validate_profile_evidence(
        report,
        phase=args.phase,
        layout_id=str(matrix["metadata"]["hardware_layout"]["layout_id"]),
        requirements=gate["requirements"],
        contract_id=str(gate["profile_contract_id"]),
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        existing = load_json(output)
        comparable_existing = dict(existing)
        comparable_report = dict(report)
        comparable_existing.pop("created_at", None)
        comparable_report.pop("created_at", None)
        if comparable_existing != comparable_report:
            raise FileExistsError(f"refusing to overwrite different evidence report: {output}")
        print(json.dumps({"report": str(output), "write_disposition": "reused"}))
        return 0
    atomic_json(output, report)
    print(json.dumps({"report": str(output), "entries": len(entries), "write_disposition": "written"}))
    return 0


def main() -> int:
    script = Path(__file__).resolve()
    pipeline_default = script.parents[2]
    workspace = pipeline_default.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run-host", "merge"))
    parser.add_argument("--design-matrix", type=Path, required=True)
    parser.add_argument("--phase", choices=("c", "d"), required=True)
    parser.add_argument("--host-id")
    parser.add_argument("--domain-split-manifest", type=Path)
    parser.add_argument("--fragments", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=pipeline_default)
    parser.add_argument(
        "--python",
        type=Path,
        default=workspace
        / "local_runtime"
        / "envs"
        / "holomol-py311-cu121"
        / "bin"
        / "python",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--abort-temperature-c", type=float, default=90.0)
    parser.add_argument(
        "--allow-existing-scaling-processes-on-other-gpus",
        action="store_true",
        help=(
            "allow an existing scaling campaign only when every target GPU is "
            "freshly verified compute-idle by UUID before its smoke requirement"
        ),
    )
    args = parser.parse_args()
    if args.mode == "run-host":
        if not args.host_id or args.domain_split_manifest is None:
            parser.error("run-host requires --host-id and --domain-split-manifest")
        if args.poll_seconds <= 0 or not 70 <= args.abort_temperature_c <= 95:
            raise ValueError("invalid smoke polling/temperature controls")
        return run_host(args)
    if not args.fragments:
        parser.error("merge requires --fragments")
    return merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
