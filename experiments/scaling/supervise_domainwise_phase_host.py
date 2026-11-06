#!/usr/bin/env python3
"""Supervise one host's Phase-C/D slots from a single global frozen matrix.

The supervisor is phase- and host-qualified, keeps a finite per-slot worker
restart budget, and writes a host completion marker only after independently
revalidating every assigned run.  It never mutates the global matrix.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from domainwise_gpu_admission import probe_gpus, stuck_nvidia_smi_pids

from run_domainwise_phase_slot import (
    ATTEMPTS_EXHAUSTED_EXIT,
    SUPPORTED_EXECUTION_PHASES,
    SUPPORTED_STATUSES,
    atomic_json,
    completed_attempt,
    load_json,
    matrix_phase,
    phase_root,
    safe_component,
    validate_manifest_for_matrix,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def self_process_identity() -> dict[str, int]:
    raw = Path("/proc/self/stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError("cannot parse /proc/self/stat")
    tail = raw[close + 2 :].split()
    if len(tail) < 20:
        raise RuntimeError("short /proc/self/stat")
    return {
        "pid": os.getpid(),
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
    }


def append_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"time": now(), **value}, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        handle.flush()


def running_compute_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi compute-process query failed: {result.stderr.strip()}"
        )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"cannot parse nvidia-smi compute row: {line!r}")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": int(fields[3]),
            }
        )
    return rows


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory query failed: {result.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"cannot parse nvidia-smi inventory row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
            }
        )
    return rows


def invoked_script_argument(command: list[str]) -> str | None:
    """Identify the executed script, never a pathname inside diagnostic argv.

    A shell's entire ``-c`` command is one argv item; treating every item's
    basename as executable identity mistakes ordinary rg/sed checks for jobs.
    Formal workers here use a direct shebang or a Python script invocation.
    """
    if not command:
        return None
    executable = Path(command[0]).name
    if re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", executable) is None:
        return command[0]
    index = 1
    simple_flags = set("bBdEiIOPqRsSuvx")
    while index < len(command):
        token = command[index]
        if token == "-":
            return None
        if token == "--":
            return command[index + 1] if index + 1 < len(command) and command[index + 1] != "-" else None
        if token == "--check-hash-based-pycs":
            if index + 1 >= len(command):
                return None
            index += 2
            continue
        if token.startswith("--check-hash-based-pycs="):
            index += 1
            continue
        if token.startswith("--"):
            # Includes help/version and unsupported long interpreter options.
            return None
        if token.startswith("-"):
            # CPython accepts clusters such as -uB and -Iu. W/X consume
            # the entire remaining suffix, or exactly the next argv item;
            # those option values must never become executable identities.
            for offset, flag in enumerate(token[1:], 1):
                if flag in {"c", "m", "h", "V", "?"}:
                    return None
                if flag in {"W", "X"}:
                    if offset + 1 == len(token):
                        if index + 1 >= len(command):
                            return None
                        index += 1
                    break
                if flag not in simple_flags:
                    return None
            index += 1
            continue
        return token
    return None


def invoked_scaling_scripts(command: list[str], guarded_scripts: set[str]) -> set[str]:
    actual = invoked_script_argument(command)
    name = Path(actual).name if actual is not None else None
    return {name} if name in guarded_scripts else set()


def active_scaling_processes() -> list[dict[str, Any]]:
    """Find owned B/C/D workers or supervisors that make a handoff unsafe."""
    guarded_scripts = {
        "run_domainwise_phase_b_slot.py",
        "supervise_domainwise_phase_b_local.py",
        "run_domainwise_phase_slot.py",
        "supervise_domainwise_phase_host.py",
        "train_qwen3_pretrain.py",
    }
    found: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            raw_stat = (entry / "stat").read_text(encoding="utf-8")
            start_ticks = int(raw_stat[raw_stat.rfind(")") + 2 :].split()[19])
            command = [
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError, IndexError):
            continue
        matched = invoked_scaling_scripts(command, guarded_scripts)
        if matched:
            found.append(
                {
                    "pid": int(entry.name),
                    "start_ticks": start_ticks,
                    "scripts": sorted(matched),
                    "cmdline": command,
                }
            )
    return found


def scaling_snapshot_confirmed_exited(
    row: dict[str, Any], *, proc_root: Path = Path("/proc"),
) -> bool:
    """Ignore only a vanished PID or the same generation in a terminal state."""
    try:
        pid = int(row["pid"])
        if pid <= 0:
            return False
        directory = proc_root / str(pid)
        try:
            raw = (directory / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            # Missing stat alone is insufficient if the process directory exists.
            try:
                directory.stat()
            except (FileNotFoundError, ProcessLookupError):
                return True
            return False
        tail = raw[raw.rfind(")") + 2 :].split()
        if int(raw.split(" ", 1)[0]) != pid:
            return False
        if "start_ticks" not in row or int(tail[19]) != int(row["start_ticks"]):
            return False
        return tail[0] in {"Z", "X", "x"}
    except (OSError, ValueError, IndexError, KeyError, TypeError):
        return False


def unowned_scaling_processes(
    processes: list[dict[str, Any]], allowed_pids: set[int],
) -> list[dict[str, Any]]:
    return [
        row for row in processes
        if int(row["pid"]) not in allowed_pids
        and not scaling_snapshot_confirmed_exited(row)
    ]


def validate_startup_overlap_policy(
    *,
    selected_gpu_uuids: set[str],
    compute_processes: list[dict[str, Any]],
    scaling_processes: list[dict[str, Any]],
    allow_existing_compute_processes: bool,
    allow_existing_scaling_processes_on_other_gpus: bool,
) -> None:
    """Apply the fail-closed startup policy for the formal matrix GPUs."""

    if not selected_gpu_uuids:
        raise RuntimeError("formal matrix selected no GPU UUIDs on this host")
    selected_compute = [
        row
        for row in compute_processes
        if str(row.get("gpu_uuid", "")) in selected_gpu_uuids
    ]
    # The disjoint-overlap opt-in never relaxes the target-GPU compute gate.
    # Keep the older compute override independent for compatibility, and make
    # the two policies impossible to combine at the CLI below.
    if selected_compute and (
        allow_existing_scaling_processes_on_other_gpus
        or not allow_existing_compute_processes
    ):
        raise RuntimeError(
            "refusing to overlap existing GPU compute processes on formal matrix GPUs: "
            + json.dumps(selected_compute, ensure_ascii=False)
        )
    if scaling_processes and not allow_existing_scaling_processes_on_other_gpus:
        raise RuntimeError(
            "refusing Phase-B/C/D overlap before the prior supervisor and workers exit: "
            + json.dumps(scaling_processes, ensure_ascii=False)
        )


def _cmdline_arg(command: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"expected exactly one {flag}")
    return command[positions[0] + 1]


def process_group_members(pgid: int) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            tail = raw[raw.rfind(")") + 2 :].split()
            if int(tail[2]) != int(pgid):
                continue
            command = [
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
            members.append(
                {
                    "pid": int(entry.name),
                    "uid": entry.stat().st_uid,
                    "state": tail[0],
                    "start_ticks": int(tail[19]),
                    "cmdline": command,
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            continue
    return members


def validate_owned_worker_group(
    members: list[dict[str, Any]],
    *,
    slot: str,
    host_id: str,
    matrix_path: Path,
    worker_script: Path,
    trainer_script: Path,
    allowed_run_dirs: set[Path],
) -> None:
    for member in members:
        if int(member["uid"]) != os.getuid():
            raise PermissionError(f"worker group includes another uid: {member}")
        command = [str(token) for token in member["cmdline"]]
        scripts = {Path(token).name for token in command}
        if worker_script.name in scripts:
            if (
                Path(_cmdline_arg(command, "--matrix-json")).expanduser().resolve()
                != matrix_path
                or _cmdline_arg(command, "--host-id") != host_id
                or _cmdline_arg(command, "--worker-slot") != slot
            ):
                raise ValueError(f"worker group identity mismatch: {member}")
            continue
        if trainer_script.name in scripts:
            attempt_dir = Path(_cmdline_arg(command, "--out-dir")).expanduser().resolve()
            if attempt_dir.parent not in allowed_run_dirs or not attempt_dir.name.startswith(
                "attempt_"
            ):
                raise ValueError(f"trainer group out_dir is not owned by slot {slot}: {member}")
            continue
        # Ignore a zombie with no command only after it has been reaped; a live
        # unknown member makes signalling fail closed.
        if member["state"] == "Z" and not command:
            continue
        raise ValueError(f"worker group contains an unknown process: {member}")


def clean_owned_worker_group(
    pgid: int,
    *,
    slot: str,
    host_id: str,
    matrix_path: Path,
    worker_script: Path,
    trainer_script: Path,
    allowed_run_dirs: set[Path],
    timeout_seconds: float = 15.0,
    compute_gpu_ids: list[int] | None = None,
) -> int:
    members = process_group_members(pgid)
    if not members:
        return 0
    validate_owned_worker_group(
        members,
        slot=slot,
        host_id=host_id,
        matrix_path=matrix_path,
        worker_script=worker_script,
        trainer_script=trainer_script,
        allowed_run_dirs=allowed_run_dirs,
    )
    os.killpg(pgid, signal.SIGTERM)
    os.killpg(pgid, signal.SIGCONT)
    original_pids = {int(item["pid"]) for item in members}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = process_group_members(pgid)
        compute_pids = {int(item["pid"]) for item in cleanup_compute_processes(compute_gpu_ids)}
        if (
            (not remaining or all(item["state"] == "Z" for item in remaining))
            and not (original_pids & compute_pids)
        ):
            return len(members)
        time.sleep(0.1)
    remaining = process_group_members(pgid)
    validate_owned_worker_group(
        remaining,
        slot=slot,
        host_id=host_id,
        matrix_path=matrix_path,
        worker_script=worker_script,
        trainer_script=trainer_script,
        allowed_run_dirs=allowed_run_dirs,
    )
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        # The userspace process group can disappear before NVML drops its
        # final compute-context row.  Continue waiting for that exact old PID
        # set instead of treating ESRCH as either success or a fatal error.
        pass
    killed_pids = {int(item["pid"]) for item in members} | {
        int(item["pid"]) for item in remaining
    }
    kill_deadline = time.monotonic() + 10.0
    while time.monotonic() < kill_deadline:
        remaining = process_group_members(pgid)
        compute_pids = {int(item["pid"]) for item in cleanup_compute_processes(compute_gpu_ids)}
        if (
            (not remaining or all(item["state"] == "Z" for item in remaining))
            and not (killed_pids & compute_pids)
        ):
            return len(members)
        time.sleep(0.1)
    raise RuntimeError(
        f"owned worker group did not disappear after SIGKILL: pgid={pgid} "
        f"members={process_group_members(pgid)}"
    )


def cleanup_compute_processes(gpu_ids: list[int] | None) -> list[dict[str, Any]]:
    """Never let an unrelated broken GPU obscure one owned group's cleanup."""
    if gpu_ids is None:
        return running_compute_processes()
    probes = probe_gpus(gpu_ids)
    result: list[dict[str, Any]] = []
    for gpu in gpu_ids:
        probe = probes.get(gpu, {})
        if probe.get("status") != "ok" or not isinstance(probe.get("compute"), list):
            raise RuntimeError(f"cannot verify cleanup compute on GPU{gpu}: {probe}")
        result.extend(probe["compute"])
    return result


DEFERRED_ADMISSION_POLICY = "defer_unavailable_gpus_v1"
DEFERRED_IDLE_MEMORY_MIB = 256
_PENDING_CUDA_PROBES: dict[int, subprocess.Popen] = {}


def bounded_cuda_probe(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Bound even a CUDA child stuck in kernel D state; never wait in __exit__."""
    for pid, child in list(_PENDING_CUDA_PROBES.items()):
        if child.poll() is not None:
            _PENDING_CUDA_PROBES.pop(pid)
    if _PENDING_CUDA_PROBES:
        raise RuntimeError(f"unreaped owned CUDA probes block new preflight: {list(_PENDING_CUDA_PROBES)}")
    stuck = stuck_nvidia_smi_pids()
    if stuck:
        raise RuntimeError(f"owned blocked NVIDIA queries prevent CUDA preflight: {stuck}")
    child = subprocess.Popen(
        command, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _PENDING_CUDA_PROBES[child.pid] = child
    try:
        stdout, stderr = child.communicate(timeout=20)
    except (subprocess.TimeoutExpired, OSError, UnicodeError) as exc:
        # Signal only this newly created Popen generation. SIGKILL cannot
        # finish D-state kernel teardown, so a second wait must also be bounded.
        if child.poll() is None:
            try:
                child.kill()
            except OSError:
                pass
        try:
            child.communicate(timeout=0.2)
        except (subprocess.TimeoutExpired, OSError, UnicodeError):
            pass
        finally:
            for pipe in (child.stdout, child.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            if child.poll() is not None:
                _PENDING_CUDA_PROBES.pop(child.pid, None)
        raise RuntimeError(f"bounded CUDA preflight failed: {type(exc).__name__}: {exc}") from exc
    _PENDING_CUDA_PROBES.pop(child.pid, None)
    return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def validate_deferred_layout(
    phase: str,
    host_id: str,
    host_rows: list[dict[str, Any]],
    host_devices: dict[int, dict[str, Any]],
) -> dict[str, int]:
    """Partial admission changes scheduling only, never the frozen cohort."""
    if phase != "c" or host_id != "remote" or any(
        int(device["concurrent_slots"]) != 1 for device in host_devices.values()
    ):
        raise ValueError("deferred GPU admission requires Phase-C remote single-slot GPUs")
    slots: dict[str, int] = {}
    for row in host_rows:
        gpu = int(row["assigned_gpu"])
        slot = str(row["worker_slot"])
        if int(row["gpu_local_slot"]) != 0 or slot != f"remote_gpu{gpu}_slot0":
            raise ValueError("deferred GPU admission requires one canonical slot per GPU")
        if slot in slots and slots[slot] != gpu:
            raise ValueError("deferred GPU admission slot changes GPU")
        slots[slot] = gpu
    if not slots or len(slots) != len(set(slots.values())):
        raise ValueError("deferred GPU admission requires distinct assigned GPUs")
    return slots


def deferred_gpu_state(
    gpu: int, device: dict[str, Any], probe: dict[str, Any]
) -> dict[str, Any]:
    state: dict[str, Any] = {"gpu": gpu, "status": "pending_unavailable", "reason": ""}
    if probe.get("status") != "ok":
        state["reason"] = str(probe.get("error") or "GPU probe unavailable")
        return state
    inventory, compute = probe.get("inventory"), probe.get("compute")
    if not isinstance(inventory, dict) or not isinstance(compute, list):
        state["reason"] = "GPU probe lacks strict inventory/compute evidence"
        return state
    if int(inventory["index"]) != gpu or not inventory.get("uuid"):
        state["reason"] = "GPU inventory identity mismatch"
    elif compute:
        state.update(status="pending_foreign", reason="GPU has existing compute processes")
    elif int(inventory["memory_total_mib"]) < int(device["memory_gib"]) * 1024 - 1536:
        state["reason"] = "GPU memory is below the frozen hardware class"
    elif int(inventory["memory_used_mib"]) > DEFERRED_IDLE_MEMORY_MIB:
        state["reason"] = "GPU has unexplained memory use above 256 MiB despite no compute rows"
    else:
        state.update(status="ready", reason="GPU inventory valid and compute-idle")
    return state


def validate_isolated_gpu_environment(
    python: Path, pipeline_root: Path, gpu_uuid: str, required_memory_gib: int
) -> None:
    """Initialize CUDA in a short-lived UUID-isolated child, never this supervisor."""
    code = (
        "import json,torch,xformers.ops;"
        "assert torch.cuda.is_available() and torch.cuda.device_count()==1;"
        "torch.cuda.init();"
        "print(json.dumps({'device_count':torch.cuda.device_count(),"
        "'memory_total_mib':torch.cuda.get_device_properties(0).total_memory/1048576}))"
    )
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu_uuid)
    result = bounded_cuda_probe(
        [str(python), "-c", code], cwd=pipeline_root, env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"isolated GPU CUDA environment failed: {result.stderr.strip()}")
    value = json.loads(result.stdout.strip().splitlines()[-1])
    if int(value["device_count"]) != 1 or float(value["memory_total_mib"]) < (
        int(required_memory_gib) * 1024 - 1536
    ):
        raise RuntimeError(f"isolated GPU CUDA capacity mismatch: {value}")


def prepare_deferred_gpu(
    gpu: int, device: dict[str, Any], python: Path, pipeline_root: Path,
    *, validated_uuids: set[str], initial_probe: dict[str, Any] | None = None,
    expected_gpu_name: str | None = None,
    minimum_free_memory_mib: float | None = None,
) -> dict[str, Any]:
    """Admission failures remain pending; no worker/run attempt has started."""
    probe = probe_gpus([gpu]).get(gpu, {}) if initial_probe is None else initial_probe
    state = deferred_gpu_state(gpu, device, probe)
    if state["status"] != "ready":
        return state
    if expected_gpu_name is not None and probe["inventory"].get("name") != expected_gpu_name:
        return {"gpu": gpu, "status": "pending_unavailable", "reason": "GPU SKU differs from the local execution plan"}
    if minimum_free_memory_mib is not None and (
        float(probe["inventory"]["memory_total_mib"]) - float(probe["inventory"]["memory_used_mib"])
        < minimum_free_memory_mib
    ):
        return {"gpu": gpu, "status": "pending_unavailable", "reason": "GPU free memory cannot cover frozen profile aggregate peak"}
    uuid = str(probe["inventory"]["uuid"])
    if uuid not in validated_uuids:
        try:
            validate_isolated_gpu_environment(python, pipeline_root, uuid, int(device["memory_gib"]))
        except (RuntimeError, OSError, ValueError, IndexError, KeyError, subprocess.TimeoutExpired) as exc:
            return {"gpu": gpu, "status": "pending_unavailable", "reason": f"CUDA preflight: {exc}"}
        validated_uuids.add(uuid)
    # The CUDA child must have exited and released its context, and another
    # user may have acquired the GPU during preflight. Never reuse stale idle.
    fresh = probe_gpus([gpu]).get(gpu, {})
    state = deferred_gpu_state(gpu, device, fresh)
    if state["status"] == "ready" and expected_gpu_name is not None and fresh["inventory"].get("name") != expected_gpu_name:
        return {"gpu": gpu, "status": "pending_unavailable", "reason": "GPU SKU changed during local admission"}
    if state["status"] == "ready" and minimum_free_memory_mib is not None and (
        float(fresh["inventory"]["memory_total_mib"]) - float(fresh["inventory"]["memory_used_mib"])
        < minimum_free_memory_mib
    ):
        return {"gpu": gpu, "status": "pending_unavailable", "reason": "GPU free memory fell below frozen profile aggregate peak"}
    if state["status"] == "ready" and fresh["inventory"]["uuid"] != uuid:
        return {"gpu": gpu, "status": "pending_unavailable", "reason": "GPU UUID changed during admission"}
    return state


def admission_payload(identity: dict[str, Any], slots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "domainwise_gpu_admission_state_v1",
        "admission_policy": DEFERRED_ADMISSION_POLICY,
        **{key: identity[key] for key in (
            "pid", "start_ticks", "generation_nonce", "phase", "host_id",
            "matrix_json", "matrix_generation_id",
        )},
        **{key: identity[key] for key in ("execution_plan_json", "physical_host_id", "physical_hostname") if key in identity},
        "updated_at": now(),
        "slots": {slot: dict(value) for slot, value in slots.items()},
    }


def load_execution_plan(matrix_path: Path, plan_path: Path | None) -> dict[str, Any] | None:
    if plan_path is None:
        return None
    from domainwise_local_execution import load_local_execution_plan
    return load_local_execution_plan(matrix_path, plan_path)


def frozen_slot_memory_requirements(
    matrix: dict[str, Any], host_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Reuse the frozen actual evidence without relabelling its source host."""
    report = load_json(Path(matrix["metadata"]["batch_profile_evidence_gate"]["report_json"]))
    keys = ("domain", "batch_profile_id", "per_device_train_batch_size", "gradient_accumulation_steps", "effective_batch_records")
    requirements: dict[str, float] = {}
    for row in host_rows:
        candidates = [entry for entry in report["entries"] if (
            entry.get("host_id") == row["assigned_host"]
            and all(entry.get(key) == row[key] for key in keys)
            and int(entry.get("parameter_count_tested", 0)) >= int(row["parameter_count"])
            and entry.get("finite_loss") is True and entry.get("oom_observed") is False
            and entry.get("full_gradient_window_verified") is True
            and int(entry.get("complete_optimizer_steps", 0)) >= 1
        )]
        peaks = [float(entry.get("aggregate_peak_reserved_mib", math.nan)) for entry in candidates]
        if not peaks or any(not math.isfinite(peak) or peak <= 0 for peak in peaks):
            raise ValueError(f"no finite frozen aggregate memory proof for {row['run_id']}")
        slot = str(row["worker_slot"])
        requirements[slot] = max(requirements.get(slot, 0.0), *peaks)
    return requirements


def allowed_local_peer_pids(
    matrix_path: Path, plan: dict[str, Any] | None, processes: list[dict[str, Any]],
) -> set[int]:
    if plan is None:
        return set()
    from domainwise_local_execution import peer_owned_processes_allowed
    return peer_owned_processes_allowed(matrix_path, plan, processes)


def predecessor_admission_state(
    matrix: dict[str, Any], slot: str, gpu: int, plan: dict[str, Any] | None,
    processes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if plan is None:
        return None
    from domainwise_local_execution import local_predecessor_ready
    result = local_predecessor_ready(matrix, slot, process_rows=processes)
    if result.get("ready") is True:
        return None
    reasons = result.get("reasons") or ["local predecessor is not verified complete and released"]
    return {"gpu": gpu, "status": "pending_predecessor", "reason": "; ".join(str(reason) for reason in reasons)}


def validate_environment(python: Path, pipeline_root: Path, expected_gpus: list[int]) -> None:
    code = (
        "import json,torch,xformers.ops;"
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),"
        "'xformers':True}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=pipeline_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"training environment preflight failed: python={python} "
            f"cwd={pipeline_root} error={result.stderr.strip()}"
        )
    status = json.loads(result.stdout.strip().splitlines()[-1])
    if not status["cuda_available"] or int(status["device_count"]) <= max(expected_gpus):
        raise RuntimeError(f"training environment cannot see required GPUs: {status}")


def validate_global_matrix(
    matrix: dict[str, Any], requested_phase: str, host_id: str
) -> tuple[list[dict[str, Any]], list[str], dict[int, dict[str, Any]], Path]:
    """Validate the global contract and return this host's rows and slots."""
    safe_component(requested_phase, "--phase")
    safe_component(host_id, "--host-id")
    actual_phase, status = matrix_phase(matrix)
    if actual_phase != requested_phase:
        raise ValueError(
            f"requested phase {requested_phase!r} differs from matrix phase {actual_phase!r}"
        )
    root = phase_root(matrix)
    metadata = matrix["metadata"]
    layout = metadata.get("hardware_layout")
    devices = layout.get("devices") if isinstance(layout, dict) else None
    if not isinstance(devices, list) or not devices:
        raise ValueError("matrix has no frozen hardware devices")
    device_map: dict[tuple[str, int], dict[str, Any]] = {}
    for device in devices:
        if not isinstance(device, dict):
            raise ValueError("hardware device entry is not an object")
        device_host = safe_component(str(device["host_id"]), "hardware host_id")
        gpu = int(device["gpu_id"])
        slots = int(device["concurrent_slots"])
        memory = int(device["memory_gib"])
        if gpu < 0 or slots <= 0 or memory < 24:
            raise ValueError(f"invalid hardware device: {device}")
        key = (device_host, gpu)
        if key in device_map:
            raise ValueError(f"duplicate hardware device: {device_host}/gpu{gpu}")
        device_map[key] = device

    rows = matrix["rows"]
    seen_run_ids: set[str] = set()
    seen_out_dirs: set[Path] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("matrix row is not an object")
        run_id = safe_component(str(row["run_id"]), "run_id")
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate global run_id: {run_id}")
        seen_run_ids.add(run_id)
        out_dir = Path(str(row["out_dir"])).expanduser().resolve()
        if out_dir in seen_out_dirs or out_dir.parent != (root / "runs").resolve():
            raise ValueError(f"duplicate or escaping run out_dir: {out_dir}")
        seen_out_dirs.add(out_dir)
        row_host = safe_component(str(row["assigned_host"]), "assigned_host")
        gpu = int(row["assigned_gpu"])
        local_slot = int(row["gpu_local_slot"])
        device = device_map.get((row_host, gpu))
        if device is None:
            raise ValueError(f"row uses absent hardware device: {run_id}")
        expected_slot = f"{row_host}_gpu{gpu}_slot{local_slot}"
        if (
            not 0 <= local_slot < int(device["concurrent_slots"])
            or row.get("worker_slot") != expected_slot
            or row.get("matrix_status") != status
            or row.get("execution_phase") != SUPPORTED_EXECUTION_PHASES[actual_phase]
            or int(row.get("world_size", 0)) != 1
            or row.get("bpe_unfold_policy") != "none"
            or row.get("model_implementation") != "packed-xformers"
            or row.get("attention_backend")
            != "xformers_memory_efficient_block_diagonal_causal"
            or row.get("assigned_hardware_class") != device["hardware_class"]
            or int(row.get("assigned_device_memory_gib", -1)) != int(device["memory_gib"])
            or int(row.get("assigned_device_concurrent_slots", -1))
            != int(device["concurrent_slots"])
        ):
            raise ValueError(f"row hardware/phase contract mismatch: {run_id}")
        micro = int(row["per_device_train_batch_size"])
        accum = int(row["gradient_accumulation_steps"])
        if micro <= 0 or accum <= 0 or micro * accum != int(row["effective_batch_records"]):
            raise ValueError(f"row effective batch mismatch: {run_id}")

    host_rows = [row for row in rows if row["assigned_host"] == host_id]
    if not host_rows:
        raise ValueError(f"global matrix has no rows assigned to host {host_id!r}")
    host_devices = {
        gpu: device
        for (device_host, gpu), device in device_map.items()
        if device_host == host_id
    }
    if not host_devices:
        raise ValueError(f"hardware layout has no devices for host {host_id!r}")
    slots = sorted({str(row["worker_slot"]) for row in host_rows})
    return host_rows, slots, host_devices, root


def should_restart_worker(
    returncode: int, restart_count: int, max_worker_restarts: int
) -> bool:
    if returncode in {0, ATTEMPTS_EXHAUSTED_EXIT}:
        return False
    return int(restart_count) < int(max_worker_restarts)


def main() -> int:
    script_path = Path(__file__).resolve()
    pipeline_root_default = script_path.parents[2]
    workspace_default = pipeline_root_default.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(SUPPORTED_STATUSES), required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--domain-split-manifest", type=Path, default=None)
    parser.add_argument("--pipeline-root", type=Path, default=pipeline_root_default)
    parser.add_argument(
        "--python",
        type=Path,
        default=workspace_default
        / "local_runtime"
        / "envs"
        / "holomol-py311-cu121"
        / "bin"
        / "python",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--restart-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--max-worker-restarts", type=int, default=3)
    parser.add_argument("--max-total-attempts-per-run", type=int, default=4)
    parser.add_argument("--max-attempts-this-invocation", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--allow-existing-compute-processes", action="store_true")
    parser.add_argument("--defer-unavailable-gpus", action="store_true")
    parser.add_argument("--local-execution-plan", type=Path, default=None)
    parser.add_argument(
        "--allow-existing-scaling-processes-on-other-gpus",
        action="store_true",
        help=(
            "allow existing scaling processes while still requiring every GPU "
            "selected by this formal matrix to be compute-idle by UUID"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--generation-nonce", default="manual")
    parser.add_argument("--require-detached", action="store_true")
    args = parser.parse_args()
    if args.local_execution_plan is not None and (
        args.host_id != "remote" or args.phase != "c" or not args.defer_unavailable_gpus
    ):
        parser.error("--local-execution-plan requires Phase-C logical remote and deferred admission")
    if args.defer_unavailable_gpus and (
        args.allow_existing_compute_processes or args.allow_existing_scaling_processes_on_other_gpus
    ):
        parser.error("--defer-unavailable-gpus cannot be combined with overlap overrides")
    if (
        args.allow_existing_compute_processes
        and args.allow_existing_scaling_processes_on_other_gpus
    ):
        parser.error(
            "--allow-existing-compute-processes cannot be combined with "
            "--allow-existing-scaling-processes-on-other-gpus"
        )
    safe_component(args.host_id, "--host-id")
    safe_component(args.generation_nonce, "--generation-nonce")
    if args.poll_seconds <= 0 or args.restart_backoff_seconds < 0:
        raise ValueError("poll interval must be positive and restart backoff non-negative")
    if args.max_worker_restarts < 0:
        raise ValueError("--max-worker-restarts must be non-negative")
    if args.max_total_attempts_per_run <= 0 or args.max_attempts_this_invocation <= 0:
        raise ValueError("attempt limits must be positive")

    matrix_path = args.matrix_json.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    python = args.python.expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    matrix = load_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError("matrix root must be an object")
    host_rows, slots, host_devices, root = validate_global_matrix(
        matrix, args.phase, args.host_id
    )
    execution_plan_path = (
        args.local_execution_plan.expanduser().resolve()
        if args.local_execution_plan is not None else None
    )
    execution_plan = load_execution_plan(matrix_path, execution_plan_path)
    frozen_memory_by_slot = frozen_slot_memory_requirements(matrix, host_rows) if execution_plan else {}
    slot_gpu_ids = {str(row["worker_slot"]): int(row["assigned_gpu"]) for row in host_rows}
    if args.defer_unavailable_gpus:
        slot_gpu_ids = validate_deferred_layout(args.phase, args.host_id, host_rows, host_devices)
    manifest_path = (
        args.domain_split_manifest
        if args.domain_split_manifest is not None
        else Path(str(matrix["metadata"]["domain_split_manifest"]))
    ).expanduser().resolve()
    manifest = validate_manifest_for_matrix(matrix, manifest_path, check_files=True)

    lock_root = Path(f"/dev/shm/holomol-{os.getuid()}/scaling")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = (
        lock_root / f"domainwise_phase_{args.phase}_{args.host_id}_supervisor.lock"
    )
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"another Phase-{args.phase.upper()} supervisor owns {lock_path}"
        ) from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} time={now()} matrix={matrix_path}\n")
    lock_handle.flush()

    # A layout may describe more devices than this host's rows actually use.
    # Only row-assigned devices are formal matrix targets for overlap safety.
    gpu_ids = sorted({int(row["assigned_gpu"]) for row in host_rows})
    admission_states: dict[str, dict[str, Any]] = {}
    admission_probes: dict[int, dict[str, Any]] = {}
    validated_uuids: set[str] = set()
    if args.defer_unavailable_gpus:
        existing_scaling = active_scaling_processes()
        allowed_peer_pids = allowed_local_peer_pids(matrix_path, execution_plan, existing_scaling)
        unowned_scaling = unowned_scaling_processes(existing_scaling, allowed_peer_pids)
        if unowned_scaling:
            raise RuntimeError(f"refusing other owned scaling overlap: {unowned_scaling}")
        admission_probes = probe_gpus(gpu_ids)
        for slot, gpu in slot_gpu_ids.items():
            predecessor_state = predecessor_admission_state(matrix, slot, gpu, execution_plan, existing_scaling)
            if predecessor_state is not None:
                admission_states[slot] = predecessor_state
                continue
            admission_states[slot] = prepare_deferred_gpu(
                gpu, host_devices[gpu], python, pipeline_root, validated_uuids=validated_uuids,
                initial_probe=admission_probes.get(gpu, {}),
                expected_gpu_name=execution_plan["physical_gpu_names"][str(gpu)] if execution_plan else None,
                minimum_free_memory_mib=frozen_memory_by_slot.get(slot),
            )
        # Inventory is informational only; unavailable cards remain present in
        # the full slot ledger and cannot be interpreted as empty/complete.
        inventory = [p["inventory"] for p in admission_probes.values() if p.get("status") == "ok"]
    else:
        inventory_by_index = {row["index"]: row for row in gpu_inventory()}
        if any(gpu not in inventory_by_index for gpu in gpu_ids):
            raise RuntimeError(
                f"GPU inventory does not contain host layout {gpu_ids}: {list(inventory_by_index)}"
            )
        inventory = [inventory_by_index[gpu] for gpu in gpu_ids]
        for gpu in gpu_ids:
            device = host_devices[gpu]
            actual_mib = int(inventory_by_index[gpu]["memory_total_mib"])
            required_mib = int(device["memory_gib"]) * 1024
            # Vendor-reported capacity is normally slightly below the nominal GiB.
            if actual_mib < required_mib - 1536:
                raise RuntimeError(
                    f"GPU{gpu} memory {actual_mib} MiB is below frozen {required_mib} MiB class"
                )
        selected_uuids = {str(row["uuid"]) for row in inventory}
        validate_startup_overlap_policy(
            selected_gpu_uuids=selected_uuids,
            compute_processes=running_compute_processes(),
            scaling_processes=active_scaling_processes(),
            allow_existing_compute_processes=args.allow_existing_compute_processes,
            allow_existing_scaling_processes_on_other_gpus=(
                args.allow_existing_scaling_processes_on_other_gpus
            ),
        )
        validate_environment(python, pipeline_root, gpu_ids)
        if args.allow_existing_scaling_processes_on_other_gpus:
            validate_startup_overlap_policy(
                selected_gpu_uuids=selected_uuids,
                compute_processes=running_compute_processes(),
                scaling_processes=[],
                allow_existing_compute_processes=False,
                allow_existing_scaling_processes_on_other_gpus=True,
            )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "phase": args.phase,
                    "host_id": args.host_id,
                    "matrix": str(matrix_path),
                    "global_rows": len(matrix["rows"]),
                    "host_rows": len(host_rows),
                    "slots": slots,
                    "gpu_inventory": inventory,
                    "admission_policy": DEFERRED_ADMISSION_POLICY if args.defer_unavailable_gpus else "strict",
                    "admission_slots": admission_states,
                    "execution_plan_json": str(execution_plan_path) if execution_plan_path else None,
                    "physical_host_id": "local" if execution_plan_path else args.host_id,
                },
                ensure_ascii=False,
            )
        )
        return 0

    process_identity = self_process_identity()
    if args.require_detached and (
        process_identity["ppid"] != 1
        or process_identity["sid"] != process_identity["pid"]
        or process_identity["pgid"] != process_identity["pid"]
    ):
        raise RuntimeError(
            "detached supervisor must be reparented to PID1 and lead its session/process group: "
            f"{process_identity}"
        )

    supervisor_root = root / "supervisors" / args.host_id
    supervisor_root.mkdir(parents=True, exist_ok=True)
    status_path = supervisor_root / "status.jsonl"
    pid_path = supervisor_root / "supervisor.pid.json"
    identity_path = supervisor_root / "identity.json"
    atomic_json(
        identity_path,
        {
            "schema_version": "domainwise_phase_supervisor_identity_v1",
            "pid": os.getpid(),
            "ppid": process_identity["ppid"],
            "pgid": process_identity["pgid"],
            "sid": process_identity["sid"],
            "start_ticks": process_identity["start_ticks"],
            "generation_nonce": args.generation_nonce,
            "admission_policy": DEFERRED_ADMISSION_POLICY if args.defer_unavailable_gpus else "strict",
            "execution_plan_json": str(execution_plan_path) if execution_plan_path else None,
            "physical_host_id": "local" if execution_plan_path else args.host_id,
            **({"physical_hostname": execution_plan["physical_hostname"]} if execution_plan else {}),
            "phase": args.phase,
            "host_id": args.host_id,
            "matrix_json": str(matrix_path),
            "matrix_generation_id": matrix["metadata"]["generation_id"],
            "matrix_semantic_fingerprint": matrix["metadata"][
                "semantic_fingerprint"
            ]["value"],
            "phase_root": str(root),
            "worker_slots": slots,
            "owned_run_dirs": sorted(
                str(Path(str(row["out_dir"])).expanduser().resolve())
                for row in host_rows
            ),
            "script": str(script_path),
            "started_at": now(),
        },
    )
    atomic_json(pid_path, load_json(identity_path))
    append_status(
        status_path,
        {
            "event": "supervisor_start",
            "phase": args.phase,
            "host_id": args.host_id,
            "pid": os.getpid(),
            "generation_nonce": args.generation_nonce,
            "matrix": str(matrix_path),
            "global_rows": len(matrix["rows"]),
            "host_rows": len(host_rows),
            "slots": slots,
            "gpu_inventory": inventory,
        },
    )

    worker_script = script_path.with_name("run_domainwise_phase_slot.py")
    trainer_script = pipeline_root / "cli" / "pretrain" / "train_qwen3_pretrain.py"
    thermal_script = script_path.with_name("guard_domainwise_phase_thermal.py")
    if not worker_script.is_file() or not thermal_script.is_file():
        raise FileNotFoundError(
            f"missing generic worker or thermal guard: {worker_script} / {thermal_script}"
        )
    processes: dict[str, subprocess.Popen[bytes]] = {}
    log_handles: dict[str, Any] = {}
    restart_counts = {slot: 0 for slot in slots}
    allowed_run_dirs_by_slot = {
        slot: {
            Path(str(row["out_dir"])).expanduser().resolve()
            for row in host_rows
            if str(row["worker_slot"]) == slot
        }
        for slot in slots
    }
    finished: dict[str, int] = {}
    terminal_failures: dict[str, int] = {}
    shutdown_requested = False
    signal_received: int | None = None
    thermal_process: subprocess.Popen[bytes] | None = None
    thermal_log: Any | None = None
    restart_not_before: dict[str, float] = {}
    persisted_identity = load_json(identity_path)

    def publish_admission() -> None:
        if args.defer_unavailable_gpus:
            atomic_json(supervisor_root / "admission.json", admission_payload(persisted_identity, admission_states))

    def reject_unowned_scaling_overlap() -> None:
        allowed_pids: set[int] = set()
        for owned_slot, owned_process in processes.items():
            members = process_group_members(owned_process.pid)
            validate_owned_worker_group(
                members, slot=owned_slot, host_id=args.host_id, matrix_path=matrix_path,
                worker_script=worker_script, trainer_script=trainer_script,
                allowed_run_dirs=allowed_run_dirs_by_slot[owned_slot],
            )
            allowed_pids.update(int(member["pid"]) for member in members)
        observed_processes = active_scaling_processes()
        allowed_pids.update(allowed_local_peer_pids(matrix_path, execution_plan, observed_processes))
        unowned = unowned_scaling_processes(observed_processes, allowed_pids)
        if unowned:
            raise RuntimeError(f"refusing other owned scaling overlap during admission: {unowned}")

    def terminate_group(process: subprocess.Popen[bytes], signum: int) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal shutdown_requested, signal_received
        shutdown_requested = True
        signal_received = signum
        append_status(status_path, {"event": "supervisor_signal", "signal": signum})
        for process in processes.values():
            terminate_group(process, signal.SIGTERM)
        if thermal_process is not None:
            terminate_group(thermal_process, signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    def spawn_worker(slot: str) -> bool:
        if args.defer_unavailable_gpus:
            if execution_plan_path is not None and load_execution_plan(matrix_path, execution_plan_path) != execution_plan:
                raise RuntimeError("local execution plan changed after supervisor startup")
            reject_unowned_scaling_overlap()
            gpu = slot_gpu_ids[slot]
            predecessor_state = predecessor_admission_state(
                matrix, slot, gpu, execution_plan, active_scaling_processes(),
            )
            if predecessor_state is not None:
                admission_states[slot] = predecessor_state
                return False
            admission_states[slot] = prepare_deferred_gpu(
                gpu, host_devices[gpu], python, pipeline_root, validated_uuids=validated_uuids,
                initial_probe=admission_probes.get(gpu, {}),
                expected_gpu_name=execution_plan["physical_gpu_names"][str(gpu)] if execution_plan else None,
                minimum_free_memory_mib=frozen_memory_by_slot.get(slot),
            )
            if admission_states[slot]["status"] != "ready":
                return False
            if time.monotonic() < restart_not_before.get(slot, 0):
                admission_states[slot]["reason"] = "GPU ready; waiting for finite worker restart backoff"
                return False
            predecessor_state = predecessor_admission_state(
                matrix, slot, gpu, execution_plan, active_scaling_processes(),
            )
            if predecessor_state is not None:
                admission_states[slot] = predecessor_state
                return False
        worker_dir = root / "workers" / args.host_id / slot
        worker_dir.mkdir(parents=True, exist_ok=True)
        if slot not in log_handles:
            log_handles[slot] = (worker_dir / "worker_console.log").open(
                "ab", buffering=0
            )
        command = [
            str(python),
            str(worker_script),
            "--matrix-json",
            str(matrix_path),
            "--host-id",
            args.host_id,
            "--worker-slot",
            slot,
            "--pipeline-root",
            str(pipeline_root),
            "--python",
            str(python),
            "--domain-split-manifest",
            str(manifest_path),
            "--num-workers",
            str(args.num_workers),
            "--max-total-attempts-per-run",
            str(args.max_total_attempts_per_run),
            "--max-attempts-this-invocation",
            str(args.max_attempts_this_invocation),
        ]
        if execution_plan_path is not None:
            command.extend(["--local-execution-plan", str(execution_plan_path)])
        process = subprocess.Popen(
            command,
            cwd=pipeline_root,
            stdout=log_handles[slot],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[slot] = process
        if args.defer_unavailable_gpus:
            admission_states[slot] = {"gpu": slot_gpu_ids[slot], "status": "running", "reason": f"worker PID {process.pid}"}
        append_status(
            status_path,
            {
                "event": "worker_spawn",
                "phase": args.phase,
                "host_id": args.host_id,
                "worker_slot": slot,
                "restart_index": restart_counts[slot],
                "pid": process.pid,
                "pgid": process.pid,
            },
        )
        return True

    try:
        thermal_root = root / "thermal" / args.host_id
        thermal_root.mkdir(parents=True, exist_ok=True)
        thermal_log = (thermal_root / "guard_console.log").open("ab", buffering=0)
        thermal_command = [
            str(python),
            str(thermal_script),
            "--matrix-json",
            str(matrix_path),
            "--phase",
            args.phase,
            "--host-id",
            args.host_id,
            "--event-jsonl",
            str(thermal_root / "events.jsonl"),
            "--pid-file",
            str(thermal_root / "guard.pid"),
        ]
        if args.defer_unavailable_gpus:
            thermal_command.append("--defer-unavailable-gpus")
        if execution_plan_path is not None:
            thermal_command.extend(["--local-execution-plan", str(execution_plan_path)])
        thermal_process = subprocess.Popen(
            thermal_command,
            cwd=pipeline_root,
            stdout=thermal_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        append_status(
            status_path,
            {"event": "thermal_guard_spawn", "pid": thermal_process.pid},
        )
        if args.defer_unavailable_gpus:
            admission_probes = probe_gpus(gpu_ids)
        for slot in slots:
            spawn_worker(slot)
        publish_admission()

        while len(finished) + len(terminal_failures) < len(slots):
            if shutdown_requested:
                break
            if thermal_process.poll() is not None:
                append_status(
                    status_path,
                    {
                        "event": "thermal_guard_unexpected_exit",
                        "returncode": thermal_process.returncode,
                    },
                )
                shutdown_requested = True
                for process in processes.values():
                    terminate_group(process, signal.SIGTERM)
                break
            if args.defer_unavailable_gpus:
                # Query independently in parallel, retaining the helper's
                # cross-index UUID consistency checks. A bad card cannot
                # erase healthy observations or become an empty compute list.
                admission_probes = probe_gpus(gpu_ids)
            for slot in slots:
                if slot in finished or slot in terminal_failures:
                    continue
                process = processes.get(slot)
                if process is None:
                    # Deferred slots remain in the full host cohort. Neither
                    # an unavailable GPU nor waiting for one burns an attempt.
                    spawn_worker(slot)
                    continue
                code = process.poll()
                if code is None:
                    continue
                orphan_count = 0
                try:
                    orphan_count = clean_owned_worker_group(
                        process.pid,
                        slot=slot,
                        host_id=args.host_id,
                        matrix_path=matrix_path,
                        worker_script=worker_script,
                        trainer_script=trainer_script,
                        allowed_run_dirs=allowed_run_dirs_by_slot[slot],
                        compute_gpu_ids=[slot_gpu_ids[slot]] if args.defer_unavailable_gpus else None,
                    )
                except Exception as exc:
                    append_status(
                        status_path,
                        {
                            "event": "worker_group_cleanup_failed",
                            "worker_slot": slot,
                            "pgid": process.pid,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    terminal_failures[slot] = 1
                    if args.defer_unavailable_gpus:
                        admission_states[slot].update(status="failed", reason=f"worker group cleanup failed: {exc}")
                    shutdown_requested = True
                    for other in processes.values():
                        terminate_group(other, signal.SIGTERM)
                    break
                if orphan_count and code == 0:
                    # A nominally successful worker must not leave a trainer or
                    # loader behind; retry after cleaning and revalidation.
                    code = 1
                append_status(
                    status_path,
                    {
                        "event": "worker_exit",
                        "worker_slot": slot,
                        "restart_index": restart_counts[slot],
                        "pid": process.pid,
                        "returncode": code,
                        "orphan_group_members_cleaned": orphan_count,
                    },
                )
                if code == 0:
                    finished[slot] = code
                    if args.defer_unavailable_gpus:
                        processes.pop(slot)
                        admission_states[slot].update(status="complete", reason="worker completed all assigned rows")
                elif should_restart_worker(
                    code, restart_counts[slot], args.max_worker_restarts
                ):
                    restart_counts[slot] += 1
                    append_status(
                        status_path,
                        {
                            "event": "worker_restart_scheduled",
                            "worker_slot": slot,
                            "restart_index": restart_counts[slot],
                            "previous_returncode": code,
                        },
                    )
                    if args.defer_unavailable_gpus:
                        processes.pop(slot)
                        # Revalidate CUDA after a failed worker, not merely
                        # the previous successful admission's NVML snapshot.
                        validated_uuids.clear()
                        restart_not_before[slot] = time.monotonic() + args.restart_backoff_seconds
                    elif args.restart_backoff_seconds:
                        time.sleep(args.restart_backoff_seconds)
                    spawn_worker(slot)
                else:
                    terminal_failures[slot] = code
                    if args.defer_unavailable_gpus:
                        processes.pop(slot)
                        admission_states[slot].update(status="failed", reason=f"worker exhausted retries (exit {code})")
                    append_status(
                        status_path,
                        {
                            "event": "worker_restart_exhausted",
                            "worker_slot": slot,
                            "restarts": restart_counts[slot],
                            "returncode": code,
                        },
                    )
            publish_admission()
            if len(finished) + len(terminal_failures) < len(slots):
                time.sleep(args.poll_seconds)

        if shutdown_requested or terminal_failures:
            append_status(
                status_path,
                {
                    "event": "supervisor_failed",
                    "phase": args.phase,
                    "host_id": args.host_id,
                    "signal": signal_received,
                    "terminal_failures": terminal_failures,
                    "finished_slots": sorted(finished),
                },
            )
            return 128 + signal_received if signal_received is not None else 1

        invalid_runs: list[str] = []
        completed_rows: list[dict[str, Any]] = []
        for row in host_rows:
            run_dir = Path(str(row["out_dir"])).expanduser().resolve()
            complete = completed_attempt(
                run_dir, row, manifest, matrix_path, manifest_path
            )
            if complete is None:
                invalid_runs.append(str(row["run_id"]))
            else:
                completed_rows.append(
                    {
                        "run_id": row["run_id"],
                        "worker_slot": row["worker_slot"],
                        "attempt_dir": str(complete[0]),
                        "actual_input_tokens": complete[1]["actual_input_tokens"],
                        "final_validation_loss": complete[1]["final_validation"]["loss"],
                    }
                )
        if invalid_runs:
            append_status(
                status_path,
                {"event": "host_completion_revalidation_failed", "runs": invalid_runs},
            )
            return 1
        host_complete_path = root / "host_completion" / f"{args.host_id}.json"
        atomic_json(
            host_complete_path,
            {
                "schema_version": "domainwise_phase_host_completion_v1",
                "status": "complete",
                "phase": args.phase,
                "host_id": args.host_id,
                "matrix_json": str(matrix_path),
                "matrix_generation_id": matrix["metadata"]["generation_id"],
                "matrix_semantic_fingerprint": matrix["metadata"][
                    "semantic_fingerprint"
                ]["value"],
                "domain_split_manifest": str(manifest_path),
                "execution_plan_json": str(execution_plan_path) if execution_plan_path else None,
                "physical_host_id": "local" if execution_plan_path else args.host_id,
                "rows": len(completed_rows),
                "completed_runs": completed_rows,
                "completed_at": now(),
            },
        )
        append_status(
            status_path,
            {
                "event": "supervisor_complete",
                "phase": args.phase,
                "host_id": args.host_id,
                "rows": len(completed_rows),
                "host_completion": str(host_complete_path),
            },
        )
        return 0
    finally:
        # Resume every process group owned by the guard before terminating or
        # waiting on workers; a SIGSTOP'ed worker cannot otherwise act on TERM.
        if thermal_process is not None:
            terminate_group(thermal_process, signal.SIGTERM)
            try:
                thermal_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                for process in processes.values():
                    terminate_group(process, signal.SIGCONT)
                terminate_group(thermal_process, signal.SIGKILL)
        # These are exact Popen generations created by this supervisor. CONT
        # is harmless for running groups and prevents a stale STOP on handoff.
        for process in processes.values():
            terminate_group(process, signal.SIGCONT)
        for slot, process in processes.items():
            try:
                clean_owned_worker_group(
                    process.pid,
                    slot=slot,
                    host_id=args.host_id,
                    matrix_path=matrix_path,
                    worker_script=worker_script,
                    trainer_script=trainer_script,
                    allowed_run_dirs=allowed_run_dirs_by_slot[slot],
                    compute_gpu_ids=[slot_gpu_ids[slot]] if args.defer_unavailable_gpus else None,
                )
            except Exception as exc:
                append_status(
                    status_path,
                    {
                        "event": "final_worker_group_cleanup_failed",
                        "worker_slot": slot,
                        "pgid": process.pid,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        for handle in log_handles.values():
            handle.close()
        if thermal_log is not None:
            thermal_log.close()
        try:
            recorded_pid = load_json(pid_path)
            if (
                isinstance(recorded_pid, dict)
                and recorded_pid.get("pid") == os.getpid()
                and recorded_pid.get("generation_nonce") == args.generation_nonce
            ):
                pid_path.unlink()
        except (FileNotFoundError, OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
