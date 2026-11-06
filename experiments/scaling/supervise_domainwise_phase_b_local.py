#!/usr/bin/env python3
"""Supervise all physical-host Phase-B worker slots until they finish."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def append_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), **value}, ensure_ascii=False) + "\n")
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
        raise RuntimeError(f"nvidia-smi compute-process query failed: {result.stderr.strip()}")
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


def validate_domain_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    dataset = manifest["dataset"]
    compact_dir = Path(dataset["compact_dir"])
    index_dir = Path(dataset["index_dir"])
    if not compact_dir.is_dir() or not index_dir.is_dir():
        raise FileNotFoundError(f"domain manifest dataset paths are unavailable: {compact_dir} / {index_dir}")
    for domain_views in manifest["views"].values():
        for view in domain_views.values():
            for row in view["shards"]:
                token_path = Path(row["token_file"])
                indices_path = Path(row["raw_indices_file"])
                if not token_path.is_file() or not indices_path.is_file():
                    raise FileNotFoundError(f"domain manifest entry is unavailable: {token_path} / {indices_path}")
                if indices_path.stat().st_size != int(row["records"]) * 4:
                    raise RuntimeError(f"domain raw-index byte count mismatch: {indices_path}")
    if str(manifest_path).startswith("/dev/shm/"):
        ready_path = manifest_path.parent.parent / "READY.json"
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("status") != "ready" or Path(ready.get("staged_manifest", "")) != manifest_path:
            raise RuntimeError(f"tmpfs dataset stage is not ready for this manifest: {ready_path}")


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


def main() -> int:
    script_path = Path(__file__).resolve()
    pipeline_root_default = script_path.parents[2]
    workspace_default = pipeline_root_default.parents[2]
    default_matrix = (
        workspace_default
        / "local_runtime"
        / "runs"
        / "scaling"
        / "domainwise_scaling_fddfs_mf80000_v1"
        / "phase_b_matrix_local.json"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, default=default_matrix)
    ap.add_argument(
        "--domain-split-manifest",
        type=Path,
        default=None,
        help="Host-local manifest override (for example this host's /dev/shm stage).",
    )
    ap.add_argument("--pipeline-root", type=Path, default=pipeline_root_default)
    ap.add_argument(
        "--python",
        type=Path,
        default=workspace_default / "local_runtime" / "envs" / "holomol-py311-cu121" / "bin" / "python",
    )
    ap.add_argument("--poll-seconds", type=float, default=15.0)
    ap.add_argument("--allow-existing-compute-processes", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    matrix_path = args.matrix_json.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    # Preserve the venv/conda-prefix launcher path instead of resolving its
    # symlink to the base interpreter (which changes sys.prefix/site-packages).
    python = args.python.expanduser().absolute()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    metadata = matrix["metadata"]
    rows = matrix.get("rows", [])
    if metadata.get("status") != "runnable_phase_b" or not rows:
        raise ValueError("expected a non-empty frozen runnable_phase_b host matrix")
    if int(metadata.get("rows", -1)) != len(rows):
        raise ValueError("matrix metadata row count does not match its rows")
    host_id = str(metadata.get("host_id", "local"))
    if not host_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in host_id):
        raise ValueError(f"unsafe matrix host_id: {host_id!r}")
    if any(str(row.get("assigned_host", host_id)) != host_id for row in rows):
        raise ValueError(f"matrix contains rows assigned to a different host than {host_id!r}")
    slots = sorted({str(row["worker_slot"]) for row in matrix["rows"]})
    gpu_ids = sorted({int(row["assigned_gpu"]) for row in matrix["rows"]})
    layout = {int(key): int(value) for key, value in metadata["gpu_layout"].items()}
    if any(gpu < 0 or count <= 0 for gpu, count in layout.items()):
        raise ValueError(f"invalid matrix GPU layout: {layout}")
    if gpu_ids != sorted(layout) or len(slots) != sum(layout.values()):
        raise ValueError(
            "matrix rows do not match metadata GPU layout: "
            f"GPUs={gpu_ids}/{sorted(layout)} slots={len(slots)}/{sum(layout.values())}"
        )
    for row in rows:
        gpu = int(row["assigned_gpu"])
        local_slot = int(row["gpu_local_slot"])
        if not 0 <= local_slot < layout[gpu]:
            raise ValueError(f"row has an invalid GPU-local slot: {row['run_id']}")
        expected_slot = f"{host_id}_gpu{gpu}_slot{local_slot}"
        if str(row["worker_slot"]) != expected_slot:
            raise ValueError(
                f"row worker slot is not host-qualified: {row['worker_slot']} != {expected_slot}"
            )
    manifest_path = (
        args.domain_split_manifest
        if args.domain_split_manifest is not None
        else Path(metadata["domain_split_manifest"])
    ).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "domain_split_raw_indices_v1" or manifest.get("status") != "complete":
        raise ValueError("frozen domain split manifest is unavailable or incomplete")
    validate_domain_manifest(manifest_path, manifest)

    campaign_root = Path(matrix["metadata"]["run_root"]) / matrix["metadata"]["experiment_name"]
    campaign_root.mkdir(parents=True, exist_ok=True)
    lock_root = Path(f"/dev/shm/holomol-{os.getuid()}/scaling")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"domainwise_phase_b_supervisor_{host_id}.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another Phase-B supervisor owns {lock_path}") from exc
    lock_handle.write(f"pid={os.getpid()} time={now()} matrix={matrix_path}\n")
    lock_handle.flush()

    full_inventory = gpu_inventory()
    inventory_by_index = {int(row["index"]): row for row in full_inventory}
    if any(gpu not in inventory_by_index for gpu in gpu_ids):
        raise RuntimeError(f"GPU inventory does not contain frozen layout {gpu_ids}: {full_inventory}")
    inventory = [inventory_by_index[gpu] for gpu in gpu_ids]
    selected_uuids = {str(row["uuid"]) for row in inventory}
    existing = [
        row for row in running_compute_processes() if str(row["gpu_uuid"]) in selected_uuids
    ]
    if existing and not args.allow_existing_compute_processes:
        raise RuntimeError(
            "refusing to overlap existing GPU compute processes; inspect before retrying: "
            + json.dumps(existing, ensure_ascii=False)
        )
    validate_environment(python, pipeline_root, gpu_ids)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "host_id": host_id,
                    "matrix": str(matrix_path),
                    "rows": len(matrix["rows"]),
                    "slots": slots,
                    "gpu_inventory": inventory,
                },
                ensure_ascii=False,
            )
        )
        return 0

    status_path = campaign_root / f"supervisor_{host_id}_status.jsonl"
    pid_path = campaign_root / f"supervisor_{host_id}.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    append_status(
        status_path,
        {
            "event": "supervisor_start",
            "host_id": host_id,
            "pid": os.getpid(),
            "matrix": str(matrix_path),
            "slots": slots,
            "gpu_inventory": inventory,
        },
    )

    worker_script = script_path.with_name("run_domainwise_phase_b_slot.py")
    processes: dict[str, subprocess.Popen[bytes]] = {}
    log_handles: dict[str, Any] = {}

    def stop_children(signum: int, _frame: Any) -> None:
        append_status(status_path, {"event": "supervisor_signal", "signal": signum})
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    try:
        for slot in slots:
            worker_dir = campaign_root / "workers" / slot
            worker_dir.mkdir(parents=True, exist_ok=True)
            log_handle = (worker_dir / "worker_console.log").open("ab", buffering=0)
            log_handles[slot] = log_handle
            command = [
                str(python),
                str(worker_script),
                "--matrix-json",
                str(matrix_path),
                "--worker-slot",
                slot,
                "--pipeline-root",
                str(pipeline_root),
                "--python",
                str(python),
                "--domain-split-manifest",
                str(manifest_path),
            ]
            processes[slot] = subprocess.Popen(
                command,
                cwd=pipeline_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            append_status(
                status_path,
                {"event": "worker_spawn", "worker_slot": slot, "pid": processes[slot].pid},
            )

        pending = set(processes)
        returncodes: dict[str, int] = {}
        while pending:
            for slot in list(pending):
                code = processes[slot].poll()
                if code is None:
                    continue
                pending.remove(slot)
                returncodes[slot] = code
                append_status(
                    status_path,
                    {"event": "worker_exit", "worker_slot": slot, "returncode": code},
                )
            if pending:
                time.sleep(max(1.0, args.poll_seconds))
        failures = {slot: code for slot, code in returncodes.items() if code != 0}
        append_status(
            status_path,
            {"event": "supervisor_done", "returncodes": returncodes, "failures": failures},
        )
        return 1 if failures else 0
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        for handle in log_handles.values():
            handle.close()
        try:
            if pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
