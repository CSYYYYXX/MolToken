#!/usr/bin/env python3
"""Identity-safe Phase-B handoff after strict 60/60 acceptance.

``probe`` is read-only.  ``handoff`` first asks every exact thermal-guard
generation to exit and verifies that each group it stopped was resumed (or no
longer exists).  It then terminates only the exact Phase-B supervisor/worker
groups for the supplied host matrix, always sending ``SIGCONT`` before TERM.
The command never deletes artifacts and never targets a process by a stale PID
file alone.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_domainwise_campaign import atomic_write_json, load_json


SCHEMA = "domainwise_phase_b_handoff_v1"
SUPERVISOR = "supervise_domainwise_phase_b_local.py"
WORKER = "run_domainwise_phase_b_slot.py"
TRAINER = "train_qwen3_pretrain.py"
WATCHDOG = "watch_domainwise_campaign.py"
GUARDS = {
    "guard_local_secondary_slots_thermal.py",
    "guard_local_primary_slots_thermal.py",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def one_arg(command: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(flag)
    return command[positions[0] + 1]


def has_script(command: list[str], path: Path) -> bool:
    path = path.resolve()
    return any(
        Path(token).name == path.name and Path(token).resolve() == path
        for token in command
    )


def process(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(int(pid))
    if root.stat().st_uid != os.getuid():
        raise PermissionError(pid)
    raw = (root / "stat").read_text(encoding="utf-8")
    tail = raw[raw.rfind(")") + 2 :].split()
    if len(tail) < 20:
        raise ValueError("short stat")
    return {
        "pid": int(pid),
        "state": tail[0],
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
        "cmdline": [
            token.decode("utf-8", errors="replace")
            for token in (root / "cmdline").read_bytes().split(b"\0")
            if token
        ],
    }


def owned_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            value = process(int(entry.name))
        except (FileNotFoundError, OSError, PermissionError, ProcessLookupError, ValueError):
            continue
        if value["state"] not in {"Z", "X", "x"}:
            rows.append(value)
    return rows


def classify(matrix_path: Path, campaign_root: Path) -> dict[str, list[dict[str, Any]]]:
    script_dir = Path(__file__).resolve().parent
    pipeline_root = script_dir.parents[1]
    paths = {
        "supervisors": script_dir / SUPERVISOR,
        "workers": script_dir / WORKER,
        "trainers": pipeline_root / "cli" / "pretrain" / TRAINER,
        "watchdogs": script_dir / WATCHDOG,
    }
    matrix = load_json(matrix_path)
    run_dirs = {canonical(row["out_dir"]) for row in matrix["rows"]}
    result: dict[str, list[dict[str, Any]]] = {
        "guards": [],
        "supervisors": [],
        "workers": [],
        "trainers": [],
        "watchdogs": [],
    }
    for item in owned_processes():
        command = item["cmdline"]
        names = {Path(token).name for token in command}
        try:
            if names & GUARDS:
                if canonical(one_arg(command, "--matrix-json")) == matrix_path:
                    script_name = next(name for name in GUARDS if name in names)
                    expected = script_dir / script_name
                    if not has_script(command, expected):
                        continue
                    result["guards"].append(
                        {
                            **item,
                            "event_jsonl": str(canonical(one_arg(command, "--event-jsonl"))),
                            "script": script_name,
                        }
                    )
            elif has_script(command, paths["supervisors"]):
                if canonical(one_arg(command, "--matrix-json")) == matrix_path:
                    result["supervisors"].append(item)
            elif has_script(command, paths["workers"]):
                if canonical(one_arg(command, "--matrix-json")) == matrix_path:
                    result["workers"].append(item)
            elif has_script(command, paths["trainers"]):
                attempt = canonical(one_arg(command, "--out-dir"))
                if attempt.parent in run_dirs and attempt.name.startswith("attempt_"):
                    result["trainers"].append({**item, "attempt_dir": str(attempt)})
            elif has_script(command, paths["watchdogs"]):
                # The long-running B watchdog was launched with its canonical
                # default campaign root, so ``--campaign-root`` can be absent.
                declared = (
                    canonical(one_arg(command, "--campaign-root"))
                    if "--campaign-root" in command
                    else canonical(Path(__file__).resolve().parents[5] / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1")
                )
                if declared == campaign_root:
                    result["watchdogs"].append(item)
        except (KeyError, OSError, StopIteration, TypeError, ValueError):
            continue
    return result


def compute_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}


def identity_key(identity: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(identity["pid"]),
        int(identity["start_ticks"]),
        int(identity["pgid"]),
    )


def guard_generation_events(guard: dict[str, Any]) -> list[dict[str, Any]]:
    path = canonical(guard["event_jsonl"])
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    start_names = {"guard_start", "primary_guard_start"}
    exit_names = {"guard_exit", "primary_guard_exit"}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            if value.get("event") in start_names and int(value.get("pid", -1)) == int(guard["pid"]):
                active = [value]
                continue
            if active:
                active.append(value)
                if value.get("event") in exit_names and int(value.get("pid", -1)) == int(guard["pid"]):
                    events = active
                    active = []
    return events or active


def verify_guard_releases(stopped_guards: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    stop_names = {"sigstop_sent", "sigstop_primary_sent"}
    cont_names = {"sigcont_sent", "sigcont_primary_sent"}
    exit_names = {"guard_exit", "primary_guard_exit"}
    for guard in stopped_guards:
        events = guard_generation_events(guard)
        if not any(
            event.get("event") in exit_names
            and int(event.get("pid", -1)) == int(guard["pid"])
            for event in events
        ):
            issues.append(f"guard pid={guard['pid']} has no matching graceful exit event")
        paused: dict[tuple[int, int, int], dict[str, Any]] = {}
        for event in events:
            identity = event.get("identity")
            if not isinstance(identity, dict):
                continue
            try:
                key = identity_key(identity)
            except (KeyError, TypeError, ValueError):
                continue
            if event.get("event") in stop_names:
                paused[key] = identity
            elif event.get("event") in cont_names:
                paused.pop(key, None)
        for key, identity in paused.items():
            try:
                current = process(key[0])
            except (FileNotFoundError, OSError, PermissionError, ProcessLookupError, ValueError):
                continue
            if (
                int(current["start_ticks"]) == key[1]
                and int(current["pgid"]) == key[2]
                and current["state"] in {"T", "t"}
            ):
                issues.append(
                    f"guard pid={guard['pid']} left its owned process group stopped: {identity}"
                )
    return issues


def validate_matrix(matrix_path: Path, host_id: str) -> dict[str, Any]:
    value = load_json(matrix_path)
    metadata = value.get("metadata", {}) if isinstance(value, dict) else {}
    if (
        metadata.get("status") != "runnable_phase_b"
        or metadata.get("host_id") != host_id
        or not isinstance(value.get("rows"), list)
        or not value["rows"]
    ):
        raise ValueError("handoff accepts only the matching host-qualified Phase-B matrix")
    return value


def probe(matrix_path: Path, campaign_root: Path, host_id: str) -> dict[str, Any]:
    validate_matrix(matrix_path, host_id)
    roles = classify(matrix_path, campaign_root)
    campaign_pids = {
        int(item["pid"])
        for role in ("supervisors", "workers", "trainers")
        for item in roles[role]
    }
    gpu_pids = compute_pids()
    return {
        "schema_version": SCHEMA,
        "observed_at": now(),
        "host_id": host_id,
        "matrix_json": str(matrix_path),
        "roles": roles,
        "role_counts": {key: len(value) for key, value in roles.items()},
        "campaign_compute_pids": sorted(campaign_pids & gpu_pids),
        "quiescent": not any(roles[key] for key in ("guards", "supervisors", "workers", "trainers"))
        and not (campaign_pids & gpu_pids),
    }


def handoff(
    matrix_path: Path,
    campaign_root: Path,
    host_id: str,
    *,
    stop_watchdog: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    before = probe(matrix_path, campaign_root, host_id)
    guards = list(before["roles"]["guards"])
    watchdogs = list(before["roles"]["watchdogs"]) if stop_watchdog else []
    for item in guards + watchdogs:
        # Guards/watchdog must be independently sessioned.  Signal the exact
        # process, not a PID-file value or an unvalidated process group.
        if int(item["pgid"]) != int(item["pid"]) or int(item["sid"]) != int(item["pid"]):
            raise RuntimeError(f"refusing process with unsafe session identity: {item}")
        os.kill(int(item["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = classify(matrix_path, campaign_root)
        if not current["guards"] and (not stop_watchdog or not current["watchdogs"]):
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("thermal guard/watchdog did not exit gracefully")
    release_issues = verify_guard_releases(guards)
    if release_issues:
        raise RuntimeError("; ".join(release_issues))

    current = classify(matrix_path, campaign_root)
    supervisors = current["supervisors"]
    workers = current["workers"]
    for item in supervisors:
        if (
            int(item["ppid"]) != 1
            or int(item["pgid"]) != int(item["pid"])
            or int(item["sid"]) != int(item["pid"])
        ):
            raise RuntimeError(f"refusing supervisor identity mismatch: {item}")
        os.kill(int(item["pid"]), signal.SIGTERM)
    for item in workers:
        if int(item["pgid"]) != int(item["pid"]) or int(item["sid"]) != int(item["pid"]):
            raise RuntimeError(f"refusing worker group identity mismatch: {item}")
        # Guards have exited and proved release.  CONT-before-TERM protects the
        # narrow race where a final stop was sent immediately before shutdown.
        os.killpg(int(item["pgid"]), signal.SIGCONT)
        os.killpg(int(item["pgid"]), signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        after = probe(matrix_path, campaign_root, host_id)
        if after["quiescent"] and (not stop_watchdog or not after["roles"]["watchdogs"]):
            return {
                "schema_version": SCHEMA,
                "status": "quiescent",
                "host_id": host_id,
                "matrix_json": str(matrix_path),
                "stopped_guard_generations": [
                    {key: item[key] for key in ("pid", "start_ticks", "script", "event_jsonl")}
                    for item in guards
                ],
                "stopped_watchdog_generations": [
                    {key: item[key] for key in ("pid", "start_ticks")}
                    for item in watchdogs
                ],
                "guard_release_verified": True,
                "completed_at": now(),
                "after": after,
            }
        time.sleep(0.25)
    raise RuntimeError("Phase-B processes did not reach a proven quiescent state")


def main() -> int:
    script = Path(__file__).resolve()
    workspace = script.parents[5]
    campaign = workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "handoff"))
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, default=campaign)
    parser.add_argument("--host-id", choices=("local", "remote"), required=True)
    parser.add_argument("--stop-watchdog", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    matrix_path = args.matrix_json.expanduser().resolve()
    campaign_root = args.campaign_root.expanduser().resolve()
    result = (
        probe(matrix_path, campaign_root, args.host_id)
        if args.action == "probe"
        else handoff(
            matrix_path,
            campaign_root,
            args.host_id,
            stop_watchdog=args.stop_watchdog,
            timeout_seconds=args.timeout_seconds,
        )
    )
    if args.output_json is not None:
        atomic_write_json(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if (args.action == "probe" or result.get("status") == "quiescent") else 1


if __name__ == "__main__":
    raise SystemExit(main())
