"""Read-only approval and predecessor gates for executing logical remote C rows locally.

The frozen matrix is never rewritten. This module checks JSON/process evidence
only; physical GPU idleness remains a separate supervisor admission gate.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "domainwise_local_execution_plan_v1"
PLACEMENT_SCHEMA = "domainwise_execution_placement_v1"
_SCRIPT_ROOT = Path(__file__).resolve().parent
_SCRIPTS = {
    "supervisor": _SCRIPT_ROOT / "supervise_domainwise_phase_host.py",
    "worker": _SCRIPT_ROOT / "run_domainwise_phase_slot.py",
    "guard": _SCRIPT_ROOT / "guard_domainwise_phase_thermal.py",
    "trainer": _SCRIPT_ROOT.parents[1] / "cli/pretrain/train_qwen3_pretrain.py",
}
_REQUIRED_PLAN = {
    "schema_version", "status", "phase", "matrix_json", "matrix_generation_id",
    "matrix_semantic_fingerprint", "logical_host_id", "actual_execution_host",
    "gpu_map", "run_ids", "physical_hostname", "physical_gpu_names",
}
_DEAD = {"Z", "X", "x"}
_PROC_FIELDS = ("pid", "ppid", "pgid", "sid", "start_ticks", "state", "cmdline")


def _canonical(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"non-empty string required for {name}")
    return value


def _matrix_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    metadata, rows = matrix.get("metadata"), matrix.get("rows")
    if not isinstance(metadata, dict) or not isinstance(rows, list):
        raise ValueError("matrix requires metadata and rows")
    if (metadata.get("schema_version") != "domainwise_phase_matrix_v1"
            or metadata.get("phase") != "c" or metadata.get("status") != "runnable_phase_c"
            or metadata.get("is_runnable") is not True or metadata.get("rows") != 60
            or len(rows) != 60):
        raise ValueError("local execution requires the complete runnable C60 matrix")
    _text(metadata.get("generation_id"), "matrix generation")
    fingerprint = metadata.get("semantic_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("matrix semantic fingerprint must be an object")
    _text(fingerprint.get("value"), "matrix fingerprint")
    ids, directories = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("matrix row is not an object")
        run_id = _text(row.get("run_id"), "run_id")
        run_dir = _canonical(_text(row.get("out_dir"), "out_dir"))
        if run_id in ids or run_dir in directories:
            raise ValueError("duplicate matrix run ID/output directory")
        ids.add(run_id)
        directories.add(run_dir)
        if (row.get("matrix_status") != "runnable_phase_c"
                or row.get("execution_phase") != "C_completion"
                or row.get("assigned_host") not in {"local", "remote"}):
            raise ValueError("row is not part of the frozen C cohort")
    remote = [row for row in rows if row["assigned_host"] == "remote"]
    if len(remote) != 21:
        raise ValueError("frozen remote C cohort must contain exactly 21 rows")
    remote_gpus = set()
    for row in remote:
        gpu = _integer(row.get("assigned_gpu"), "remote assigned_gpu")
        if (gpu not in range(4) or row.get("gpu_local_slot") != 0
                or row.get("worker_slot") != f"remote_gpu{gpu}_slot0"
                or row.get("assigned_device_concurrent_slots") != 1
                or row.get("world_size") != 1 or row.get("gpu_count") != 1):
            raise ValueError("remote rows must use the four frozen single-slot GPUs")
        remote_gpus.add(gpu)
    if remote_gpus != set(range(4)):
        raise ValueError("remote cohort does not cover all four frozen GPU slots")
    return rows


def _validate_plan(matrix_path: Path, matrix: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    rows = _matrix_rows(matrix)
    if not isinstance(plan, dict) or not _REQUIRED_PLAN <= plan.keys():
        raise ValueError("local execution plan is missing required fields")
    if plan.keys() - _REQUIRED_PLAN - {"created_at", "reason"}:
        raise ValueError("unknown local execution plan fields/scientific overrides")
    metadata = matrix["metadata"]
    expected = {
        "schema_version": PLAN_SCHEMA, "status": "approved", "phase": "c",
        "matrix_json": str(matrix_path), "matrix_generation_id": metadata["generation_id"],
        "matrix_semantic_fingerprint": metadata["semantic_fingerprint"]["value"],
        "logical_host_id": "remote", "actual_execution_host": "local",
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("local execution plan matrix/status/host identity mismatch")
    mapping = plan.get("gpu_map")
    if not isinstance(mapping, dict) or set(mapping) != {str(gpu) for gpu in range(4)}:
        raise ValueError("GPU map must contain exactly logical GPUs 0-3")
    for gpu in range(4):
        if _integer(mapping[str(gpu)], "actual execution GPU") != gpu:
            raise ValueError("GPU map must preserve the same local physical index")
    names = plan.get("physical_gpu_names")
    if not isinstance(names, dict) or set(names) != set(mapping):
        raise ValueError("physical GPU names must cover exactly GPUs 0-3")
    for name in names.values():
        _text(name, "physical GPU name")
    if _text(plan.get("physical_hostname"), "physical hostname") != socket.gethostname():
        raise ValueError("local execution plan physical hostname mismatch")
    run_ids = plan.get("run_ids")
    if (not isinstance(run_ids, list) or any(not isinstance(item, str) for item in run_ids)
            or len(run_ids) != 21 or len(set(run_ids)) != 21
            or set(run_ids) != {row["run_id"] for row in rows if row["assigned_host"] == "remote"}):
        raise ValueError("plan run_ids must exactly equal the frozen remote 21-row cohort")
    for key in ("created_at", "reason"):
        if key in plan:
            _text(plan[key], key)
    return plan


def load_local_execution_plan(matrix_path: Path | str, plan_path: Path | str | None) -> dict[str, Any] | None:
    if plan_path is None:
        return None
    matrix_path = _canonical(matrix_path)
    return _validate_plan(matrix_path, _json(matrix_path), _json(_canonical(plan_path)))


def execution_placement_for_row(matrix_path: Path | str, plan_path: Path | str | None,
                                row: dict[str, Any]) -> dict[str, Any]:
    plan = load_local_execution_plan(matrix_path, plan_path)
    if plan is None:
        raise ValueError("execution placement requires an approved plan")
    matrix = _json(_canonical(matrix_path))
    matches = [item for item in matrix["rows"] if item.get("run_id") == row.get("run_id")]
    if len(matches) != 1 or matches[0] != row or row.get("assigned_host") != "remote":
        raise ValueError("execution placement row is not the exact frozen remote row")
    gpu = row["assigned_gpu"]
    return {
        "schema_version": PLACEMENT_SCHEMA, "plan_json": str(_canonical(plan_path)),
        "logical_host_id": "remote", "actual_execution_host": "local",
        "logical_worker_slot": row["worker_slot"], "logical_gpu": gpu,
        "actual_execution_gpu": plan["gpu_map"][str(gpu)],
        "physical_hostname": plan["physical_hostname"],
        "physical_gpu_name": plan["physical_gpu_names"][str(gpu)],
    }


def _read_proc(pid: int) -> dict[str, Any]:
    entry = Path("/proc") / str(pid)
    uid = entry.stat().st_uid
    if uid != os.getuid():
        raise ValueError("process belongs to another UID")
    raw = (entry / "stat").read_text(encoding="utf-8")
    opening, closing = raw.find("("), raw.rfind(")")
    tail = raw[closing + 2:].split()
    if opening < 0 or closing <= opening or len(tail) < 20:
        raise ValueError("malformed process stat")
    return {"pid": pid, "ppid": int(tail[1]), "pgid": int(tail[2]), "sid": int(tail[3]),
            "state": tail[0], "start_ticks": int(tail[19]), "comm": raw[opening + 1:closing],
            "uid": uid, "cmdline": [os.fsdecode(token) for token in
            (entry / "cmdline").read_bytes().split(b"\0") if token]}


def _proc_identity(row: dict[str, Any]) -> dict[str, Any]:
    pid = _integer(row.get("pid"), "process PID", minimum=1)
    live = _read_proc(pid)
    # Both full watcher snapshots and abbreviated process lists must resolve
    # to the same live generation. State may legitimately change S/R/T between
    # observations; identity, ancestry and the invocation may not.
    for key in ("pid", "ppid", "pgid", "sid", "start_ticks", "cmdline", "uid", "comm"):
        if key in row and row[key] != live.get(key):
            raise ValueError(f"process row {key} changed")
    row = live
    if "uid" in row and row["uid"] != os.getuid():
        raise ValueError("process UID mismatch")
    for key in ("ppid", "pgid", "sid", "start_ticks"):
        _integer(row[key], key, minimum=1)
    if not isinstance(row["state"], str) or len(row["state"]) != 1:
        raise ValueError("invalid process state")
    if not isinstance(row["cmdline"], list) or not all(isinstance(item, str) for item in row["cmdline"]):
        raise ValueError("invalid process cmdline")
    return row


def _owned_process_rows() -> list[dict[str, Any]]:
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == os.getuid():
                rows.append(_read_proc(int(entry.name)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows


def _arg(command: list[str], flag: str) -> str | None:
    indexes = [index for index, item in enumerate(command) if item == flag]
    if len(indexes) != 1 or indexes[0] + 1 == len(command):
        return None
    return command[indexes[0] + 1]


def _script(proc: dict[str, Any], role: str) -> bool:
    command = proc["cmdline"]
    if not command:
        return False
    index = 0
    if re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(command[0]).name):
        index = 1
        while index < len(command) and command[index] in {"-u", "-B", "-s", "-E"}:
            index += 1
    if index >= len(command):
        return False
    return _canonical(command[index]) == _SCRIPTS[role]


def _logical_local(proc: dict[str, Any], matrix_path: Path, role: str) -> bool:
    command = proc["cmdline"]
    supplied_matrix = _arg(command, "--matrix-json")
    return bool(_script(proc, role) and supplied_matrix is not None
                and _canonical(supplied_matrix) == matrix_path
                and _arg(command, "--host-id") == "local"
                and (role == "worker" or _arg(command, "--phase") == "c"))


def _leader(proc: dict[str, Any]) -> bool:
    return proc["pid"] == proc["pgid"] == proc["sid"]


def peer_owned_processes_allowed(matrix_path: Path | str, plan: dict[str, Any] | None,
                                 process_rows: list[dict[str, Any]]) -> set[int]:
    if plan is None:
        return set()
    matrix_path = _canonical(matrix_path)
    matrix = _json(matrix_path)
    _validate_plan(matrix_path, matrix, plan)
    local_rows = [row for row in matrix["rows"] if row["assigned_host"] == "local"]
    slots = {row["worker_slot"] for row in local_rows}
    by_dir = {_canonical(row["out_dir"]): row for row in local_rows}
    procs = []
    for value in process_rows:
        try:
            proc = _proc_identity(value)
            if proc["state"] not in _DEAD:
                procs.append(proc)
        except (OSError, ValueError, TypeError, KeyError):
            continue
    if len({proc["pid"] for proc in procs}) != len(procs):
        return set()
    supervisors = [proc for proc in procs if _logical_local(proc, matrix_path, "supervisor")
                   and _leader(proc) and proc["ppid"] == 1]
    if len(supervisors) != 1:
        return set()
    supervisor = supervisors[0]
    allowed = {supervisor["pid"]}
    children = [proc for proc in procs if proc["ppid"] == supervisor["pid"]
                and proc["start_ticks"] >= supervisor["start_ticks"] and _leader(proc)]
    workers = [proc for proc in children if _logical_local(proc, matrix_path, "worker")
               and _arg(proc["cmdline"], "--worker-slot") in slots]
    workers = [proc for proc in workers if sum(
        _arg(other["cmdline"], "--worker-slot") == _arg(proc["cmdline"], "--worker-slot")
        for other in workers) == 1]
    allowed.update(proc["pid"] for proc in workers)
    guards = [proc for proc in children if _logical_local(proc, matrix_path, "guard")]
    if len(guards) == 1:
        allowed.add(guards[0]["pid"])
    for worker in workers:
        trainers = []
        for proc in procs:
            out = _arg(proc["cmdline"], "--out-dir")
            if not _script(proc, "trainer") or out is None:
                continue
            attempt = _canonical(out)
            row = by_dir.get(attempt.parent)
            if (row is not None and re.fullmatch(r"attempt_[0-9]{3,}", attempt.name)
                    and row["worker_slot"] == _arg(worker["cmdline"], "--worker-slot")
                    and proc["ppid"] == proc["pgid"] == worker["pid"]
                    and proc["sid"] == worker["sid"]
                    and proc["start_ticks"] >= worker["start_ticks"]
                    and proc.get("comm") != "pt_data_worker"):
                trainers.append(proc)
        if len(trainers) == 1:
            trainer = trainers[0]
            allowed.add(trainer["pid"])
            by_pid = {proc["pid"]: proc for proc in procs}
            for proc in procs:
                current, visited = proc, set()
                while current["pid"] != trainer["pid"]:
                    if (current["pid"] in visited or current.get("comm") != "pt_data_worker"
                            or current["cmdline"] != trainer["cmdline"]
                            or current["pgid"] != worker["pid"] or current["sid"] != worker["sid"]):
                        break
                    visited.add(current["pid"])
                    parent = by_pid.get(current["ppid"])
                    if parent is None or parent["start_ticks"] > current["start_ticks"]:
                        break
                    current = parent
                else:
                    allowed.add(proc["pid"])
    return allowed


def _predecessor_complete(matrix: dict[str, Any], row: dict[str, Any]) -> None:
    metadata = matrix["metadata"]
    matrix_path = _canonical(metadata["output_base"]) / "matrix.json"
    run = _canonical(row["out_dir"])
    marker = _json(run / "complete.json")
    if (marker.get("status") != "complete" or marker.get("phase") != "c"
            or marker.get("run_id") != row["run_id"]):
        raise ValueError("completion marker identity mismatch")
    attempt = _canonical(_text(marker.get("attempt_dir"), "attempt_dir"))
    if attempt.parent != run or not re.fullmatch(r"attempt_[0-9]{3,}", attempt.name):
        raise ValueError("completion attempt is not an immediate run child")
    if marker.get("summary") != str(attempt / "summary.json"):
        raise ValueError("completion summary path mismatch")
    contract = _json(attempt / "attempt_contract.json")
    expected = {
        "schema_version": "domainwise_phase_attempt_contract_v1", "phase": "c",
        "matrix_json": str(matrix_path), "matrix_status": "runnable_phase_c",
        "matrix_generation_id": metadata["generation_id"],
        "matrix_semantic_fingerprint": metadata["semantic_fingerprint"]["value"],
        "run_id": row["run_id"], "host_id": "local", "worker_slot": row["worker_slot"],
        "physical_gpu": row["assigned_gpu"], "model_implementation": row["model_implementation"],
        "attention_backend": row["attention_backend"],
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("attempt contract identity mismatch")
    if contract.get("attempt_number") != int(attempt.name.removeprefix("attempt_")):
        raise ValueError("attempt contract number mismatch")
    profile = contract.get("runtime_profile")
    if not isinstance(profile, dict) or marker.get("runtime_profile") != profile:
        raise ValueError("completion runtime profile mismatch")
    micro = _integer(profile.get("per_device_train_batch_size"), "microbatch", minimum=1)
    accum = _integer(profile.get("gradient_accumulation_steps"), "gradient accumulation", minimum=1)
    if micro * accum != row["effective_batch_records"] or profile.get("effective_batch_records") != micro * accum:
        raise ValueError("completion effective batch mismatch")
    scaling = _json(attempt / "scaling_config.json")
    if (scaling.get("schema_version") != "domainwise_phase_scaling_config_v1"
            or scaling.get("matrix_row") != row or scaling.get("runtime_profile") != profile
            or any(scaling.get(key) != expected[key] for key in
                   ("matrix_json", "matrix_generation_id", "matrix_semantic_fingerprint"))):
        raise ValueError("scaling config frozen-row identity mismatch")
    summary = _json(attempt / "summary.json")
    target = _integer(row["target_input_tokens_with_eos"], "target tokens", minimum=1)
    actual = _integer(summary.get("actual_input_tokens"), "actual tokens", minimum=1)
    parameters = _integer(row["parameter_count"], "parameter count", minimum=1)
    if (summary.get("target_input_tokens") != target or not 0 <= actual - target < row["context_length"]
            or summary.get("parameter_count") != parameters
            or summary.get("model_implementation") != row["model_implementation"]
            or summary.get("attention_backend") != row["attention_backend"]
            or summary.get("termination_reason") != "target_input_tokens"
            or summary.get("target_flops_6nd") != 6 * parameters * target
            or summary.get("actual_flops_6nd") != 6 * parameters * actual
            or summary.get("scheduler_total_steps") != row["scheduler_total_steps"]):
        raise ValueError("summary frozen model/token-budget mismatch")
    _integer(summary.get("global_step"), "global step", minimum=1)
    validation = summary.get("final_validation")
    if (not isinstance(validation, dict) or validation.get("complete_view") is not True
            or validation.get("tokens") != row["validation_loss_tokens"]
            or validation.get("input_token_coverage") != 1.0
            or validation.get("input_tokens") != validation.get("expected_input_tokens")):
        raise ValueError("summary full-validation evidence mismatch")
    for key in ("samples", "tokens", "input_tokens"):
        _integer(validation.get(key), f"validation {key}", minimum=1)
    for value in (summary.get("mean_train_loss"), validation.get("loss"), validation.get("ppl")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("summary non-finite/non-positive loss")


def local_predecessor_ready(matrix: dict[str, Any], logical_remote_slot: str,
                            process_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    match = re.fullmatch(r"remote_gpu([0-3])_slot0", logical_remote_slot)
    predecessor = f"local_gpu{match[1]}_slot0" if match else None
    reasons: list[str] = []
    try:
        rows = _matrix_rows(matrix)
        if predecessor is None or logical_remote_slot not in {row["worker_slot"] for row in rows if row["assigned_host"] == "remote"}:
            raise ValueError("unknown logical remote slot")
        selected = [row for row in rows if row["assigned_host"] == "local" and row["worker_slot"] == predecessor]
        if not selected:
            raise ValueError("local predecessor queue is absent")
        run_dirs = {_canonical(row["out_dir"]) for row in selected}
        for row in selected:
            try:
                if row["assigned_gpu"] != int(match[1]) or row.get("gpu_local_slot") != 0:
                    raise ValueError("predecessor row physical placement mismatch")
                _predecessor_complete(matrix, row)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reasons.append(f"{row['run_id']}: {exc}")
        for value in _owned_process_rows() if process_rows is None else process_rows:
            try:
                proc = _proc_identity(value)
                if proc["state"] in _DEAD:
                    continue
                command = proc["cmdline"]
                # A still-live or orphaned declaration blocks admission even
                # when its executable/parent identity is not safe to whitelist.
                slot_declared = "--worker-slot" in command and predecessor in command
                out = _arg(command, "--out-dir")
                attempt_declared = out is not None and _canonical(out).parent in run_dirs
                if slot_declared or attempt_declared:
                    reasons.append(f"predecessor runtime remains live pid={proc['pid']}")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reasons.append(f"cannot validate process evidence: {exc}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reasons.append(str(exc))
    return {"ready": not reasons, "reasons": reasons, "predecessor_slot": predecessor}
