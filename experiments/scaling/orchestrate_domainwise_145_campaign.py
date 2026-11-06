#!/usr/bin/env python3
"""Durable fail-closed controller for the exact B60 + C60 + D25 campaign.

The controller is intentionally conservative.  Every externally visible
action is journaled before execution, all launches are reconciled against live
process generations, retries are finite, and malformed artifacts are retained
for inspection.  It uses validation only for the B->C review and performs a
checkpoint-loading strict verification at every phase boundary and again for
the final exact-145 union.

The normal entry point is the companion ``launch_domainwise_145_controller.sh``.
Direct ``--once`` execution is useful for audits/tests but does not satisfy the
detached production identity requirement.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from finalize_domainwise_phase_matrix import (
    PHASES,
    REVIEW_SCHEMA,
    validate_phase_c_review,
)
from domainwise_phase_lifecycle import prepare_cleanup_intent
from run_domainwise_detached_job import (
    RESULT_SCHEMA as DETACHED_RESULT_SCHEMA,
    SCHEMA as DETACHED_IDENTITY_SCHEMA,
    command_fingerprint,
)
from run_domainwise_phase_slot import atomic_json, load_json, safe_component
from supervise_domainwise_phase_host import active_scaling_processes, running_compute_processes
from watch_domainwise_followup_campaign import evaluate_campaign


STATE_SCHEMA = "domainwise_145_controller_state_v1"
PID_SCHEMA = "domainwise_145_controller_identity_v1"
COMPLETE_SCHEMA = "domainwise_145_controller_completion_v1"
TERMINAL_STEPS = {"complete", "manual_review_required", "halted"}
PHASE_EXPECTED = {"c": 60, "d": 25}


class TransientObservationError(RuntimeError):
    """A read-only observation failed without proving runtime termination."""


class CommandExecutionError(RuntimeError):
    """A subprocess returned a known non-zero status."""

    def __init__(self, message: str, *, returncode: int):
        super().__init__(message)
        self.returncode = int(returncode)


class RemoteActionUncertainError(RuntimeError):
    """SSH ended without proving whether a requested action executed."""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def python_launcher(value: Any) -> Path:
    # Resolving a venv Python symlink changes sys.prefix and package lookup.
    return Path(os.path.abspath(str(Path(str(value)).expanduser())))


def flock_probe(path: Path) -> dict[str, Any]:
    path = canonical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "blocked", "path": str(path)}
        fcntl.flock(handle, fcntl.LOCK_UN)
        return {"status": "acquired", "path": str(path)}
    finally:
        handle.close()


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def process_identity(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(int(pid))
    if root.stat().st_uid != os.getuid():
        raise PermissionError(pid)
    raw = (root / "stat").read_text(encoding="utf-8")
    tail = raw[raw.rfind(")") + 2 :].split()
    if len(tail) < 20:
        raise ValueError("short /proc stat")
    return {
        "pid": int(pid),
        "state": tail[0],
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
        "cmdline": [
            item.decode("utf-8", errors="replace")
            for item in (root / "cmdline").read_bytes().split(b"\0")
            if item
        ],
    }


def one_arg(command: list[str], flag: str) -> str:
    indexes = [index for index, token in enumerate(command) if token == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(flag)
    return command[indexes[0] + 1]


def exact_script(command: list[str], expected: Path) -> bool:
    expected = expected.resolve()
    for token in command:
        try:
            candidate = Path(token)
            if candidate.name == expected.name and candidate.resolve() == expected:
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


@dataclass(frozen=True)
class CampaignConfig:
    workspace: str
    pipeline_root: str
    python: str
    remote_python: str
    b_campaign_root: str
    b_local_matrix: str
    b_remote_matrix: str
    followup_root: str
    proposal_json: str
    shared_manifest: str
    local_staged_manifest: str
    remote_staged_manifest: str
    experiment_name: str
    remote_host: str
    remote_port: int
    remote_identity: str
    remote_workspace: str
    poll_seconds: float
    max_b_launches_per_host: int
    max_phase_launches_per_host: int
    max_smoke_attempts_per_host: int
    stale_seconds: float

    def contract(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.contract())


class StateStore:
    """Atomic state/history document; no external side effect precedes intent."""

    def __init__(self, path: Path, config: CampaignConfig, *, restore_venv_launchers: bool = False):
        self.path = path.expanduser().resolve()
        self.config = config
        if self.path.is_file():
            value = load_json(self.path)
            if not isinstance(value, dict) or value.get("schema_version") != STATE_SCHEMA:
                raise ValueError(f"invalid controller state: {self.path}")
            if value.get("configuration_fingerprint") != config.fingerprint:
                old_config = value.get("configuration", {})
                new_config = config.contract()
                changed = {key for key in set(old_config) | set(new_config)
                           if old_config.get(key) != new_config.get(key)}
                safe_restore = (
                    restore_venv_launchers
                    and value.get("step") == "wait_b"
                    and not value.get("terminal")
                    and bool(changed)
                    and changed <= {"python", "remote_python"}
                    and value.get("configuration_fingerprint") == canonical_fingerprint(old_config)
                    and all(
                        str(canonical(new_config[key])) == old_config[key]
                        and python_launcher(new_config[key]).is_file()
                        and os.access(python_launcher(new_config[key]), os.X_OK)
                        and (python_launcher(new_config[key]).parent.parent / "pyvenv.cfg").is_file()
                        for key in changed
                    )
                    and all(not any(value.get("launch_counts", {}).get(phase, {}).values())
                            for phase in ("c", "d"))
                    and all(not any(value.get("smoke_attempts", {}).get(phase, {}).values())
                            for phase in ("c", "d"))
                    and not any(action.get("status") in {"intent", "started"}
                                for action in value.get("actions", {}).values())
                )
                if not safe_restore:
                    raise RuntimeError(
                        "controller configuration differs from the persisted campaign; "
                        "refusing to fork the state machine"
                    )
                self.state = value
                old_fingerprint = value["configuration_fingerprint"]
                self.state["configuration"] = new_config
                self.state["configuration_fingerprint"] = config.fingerprint
                self.record(
                    "venv_launchers_restored",
                    old_configuration=old_config,
                    old_fingerprint=old_fingerprint,
                    new_fingerprint=config.fingerprint,
                    launcher_changes={key: {"old": old_config[key], "new": new_config[key]}
                                      for key in sorted(changed)},
                )
            self.state = value
        else:
            self.state = {
                "schema_version": STATE_SCHEMA,
                "campaign_generation_id": f"domainwise145_{config.fingerprint[:20]}",
                "configuration_fingerprint": config.fingerprint,
                "configuration": config.contract(),
                "revision": 0,
                "step": "wait_b",
                "terminal": False,
                "created_at": now(),
                "updated_at": now(),
                "actions": {},
                "launch_counts": {
                    "b": {"local": 0, "remote": 0},
                    "c": {"local": 0, "remote": 0},
                    "d": {"local": 0, "remote": 0},
                },
                "smoke_attempts": {"c": {"local": [], "remote": []}, "d": {"local": [], "remote": []}},
                "evidence": {},
                "history": [],
            }
            self.record("controller_state_created", step="wait_b")

    def save(self) -> None:
        self.state["updated_at"] = now()
        atomic_json(self.path, self.state)

    def record(self, event: str, **details: Any) -> None:
        self.state["revision"] = int(self.state.get("revision", 0)) + 1
        self.state.setdefault("history", []).append(
            {
                "revision": self.state["revision"],
                "time": now(),
                "event": event,
                **details,
            }
        )
        self.save()

    def transition(self, target: str, *, evidence: dict[str, Any] | None = None) -> None:
        previous = str(self.state["step"])
        self.state["step"] = target
        self.state["terminal"] = target == "complete"
        if evidence is not None:
            self.state.setdefault("evidence", {})[target] = evidence
        self.record("state_transition", previous=previous, target=target, evidence=evidence)

    def observe(self, key: str, value: dict[str, Any]) -> None:
        self.state.setdefault("evidence", {})[key] = value
        self.record("observation", key=key, summary=value)

    def begin_action(self, key: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        actions = self.state.setdefault("actions", {})
        existing = actions.get(key)
        if existing is not None:
            if existing.get("kind") != kind or existing.get("payload") != payload:
                raise RuntimeError(f"action key {key!r} was reused with different semantics")
            return existing
        entry = {
            "kind": kind,
            "payload": payload,
            "status": "intent",
            "intent_at": now(),
            "attempts": 0,
        }
        actions[key] = entry
        self.record("action_intent", action_key=key, kind=kind, payload=payload)
        return entry

    def action_started(self, key: str, **details: Any) -> None:
        action = self.state["actions"][key]
        action["status"] = "started"
        action["attempts"] = int(action.get("attempts", 0)) + 1
        action["started_at"] = now()
        action.update(details)
        self.record("action_started", action_key=key, details=details)

    def action_finished(self, key: str, result: dict[str, Any]) -> None:
        action = self.state["actions"][key]
        action["status"] = "complete"
        action["completed_at"] = now()
        action["result"] = result
        self.record("action_complete", action_key=key, result=result)

    def action_failed(self, key: str, error: str, *, terminal: bool = False) -> None:
        action = self.state["actions"][key]
        action["status"] = "failed"
        action["failed_at"] = now()
        action["error"] = error
        self.record("action_failed", action_key=key, error=error, terminal=terminal)

    def halt(self, reason: str, evidence: Any) -> None:
        previous = str(self.state["step"])
        self.state["step"] = "halted"
        self.state["terminal"] = False
        self.state["halt"] = {"reason": reason, "evidence": evidence, "time": now()}
        self.record("controller_halted", previous=previous, reason=reason, evidence=evidence)


class Operations(Protocol):
    def preflight_b_shared_flock(self) -> dict[str, Any]: ...
    def observe_b(self) -> dict[str, Any]: ...
    def launch_b_host(self, host_id: str, launch_nonce: str | None = None) -> dict[str, Any]: ...
    def revoke_b_launch_host(self, host_id: str) -> dict[str, Any]: ...
    def strict_b(self) -> dict[str, Any]: ...
    def handoff_b_host(self, host_id: str) -> dict[str, Any]: ...
    def review_b(self) -> dict[str, Any]: ...
    def review_is_approved(self) -> dict[str, Any]: ...
    def build_preview(self, phase: str) -> dict[str, Any]: ...
    def idle_probe(self, host_id: str) -> dict[str, Any]: ...
    def smoke_job_spec(self, phase: str, host_id: str, attempt: int) -> dict[str, Any]: ...
    def launch_job(self, spec: dict[str, Any]) -> dict[str, Any]: ...
    def probe_job(self, spec: dict[str, Any]) -> dict[str, Any]: ...
    def cleanup_job(self, spec: dict[str, Any]) -> dict[str, Any]: ...
    def validate_smoke_fragment(self, phase: str, host_id: str, path: Path) -> dict[str, Any]: ...
    def merge_smoke(self, phase: str, fragments: dict[str, Path]) -> dict[str, Any]: ...
    def formalize(self, phase: str) -> dict[str, Any]: ...
    def adopt_existing_formal_phase(self, phase: str) -> dict[str, Any]: ...
    def phase_hosts(self, phase: str) -> tuple[str, ...]: ...
    def observe_phase(self, phase: str) -> dict[str, Any]: ...
    def phase_launch_job_spec(self, phase: str, hosts: tuple[str, ...], attempt: int) -> dict[str, Any]: ...
    def stop_phase_host(self, phase: str, host_id: str, cleanup_action_id: str | None = None) -> dict[str, Any]: ...
    def strict_phase(self, phase: str) -> dict[str, Any]: ...
    def union_145(self) -> dict[str, Any]: ...


class ProductionOperations:
    def __init__(self, config: CampaignConfig, controller_root: Path, *, remote_runtime_disabled: bool = False, local_execution_plan: Path | None = None):
        self.config = config
        self.remote_runtime_disabled = bool(remote_runtime_disabled)
        self.root = controller_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = canonical(config.workspace)
        self.pipeline = canonical(config.pipeline_root)
        self.python = python_launcher(config.python)
        self.remote_python = python_launcher(config.remote_python)
        self.script_dir = Path(__file__).resolve().parent
        self.b_root = canonical(config.b_campaign_root)
        self.followup = canonical(config.followup_root)
        self.proposal = canonical(config.proposal_json)
        self.shared_manifest = canonical(config.shared_manifest)
        self.local_manifest = canonical(config.local_staged_manifest)
        self.remote_manifest = canonical(config.remote_staged_manifest)
        self.remote_identity = canonical(config.remote_identity)
        self.local_execution_plan = canonical(local_execution_plan) if local_execution_plan is not None else None
        if self.local_execution_plan is not None:
            from domainwise_local_execution import load_local_execution_plan

            load_local_execution_plan(self.matrix_path("c"), self.local_execution_plan)
            if not self.remote_runtime_disabled:
                raise ValueError("local execution plan requires remote runtime disabled")

    def phase_local_execution_plan(self, phase: str) -> Path | None:
        return self.local_execution_plan if phase == "c" else None

    def run(self, command: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            cwd=self.pipeline,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise CommandExecutionError(
                f"command failed rc={result.returncode}: {shlex.join(command)}\n"
                f"stdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}",
                returncode=result.returncode,
            )
        return result

    def ssh(self, command: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        if self.remote_runtime_disabled:
            raise TransientObservationError("remote runtime is disabled by operator")
        ssh_command = [
            "ssh",
            "-i",
            str(self.remote_identity),
            "-p",
            str(self.config.remote_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            self.config.remote_host,
            shlex.join(command),
        ]
        try:
            return self.run(ssh_command, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RemoteActionUncertainError(
                f"SSH timed out with unknown remote outcome: {shlex.join(command)}"
            ) from exc
        except CommandExecutionError as exc:
            if exc.returncode == 255:
                raise RemoteActionUncertainError(
                    f"SSH transport failed with unknown remote outcome: {exc}"
                ) from exc
            raise

    @staticmethod
    def compact_report(value: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: value.get(key)
            for key in (
                "schema_version",
                "status",
                "complete",
                "phase",
                "counts",
                "matrix_issues",
                "issues",
                "output_json",
                "generation_id",
                "test_data_accessed",
            )
            if key in value
        }
        return result

    def remote_flock_probe(self, path: Path) -> dict[str, Any]:
        remote_script = (
            canonical(self.config.remote_workspace)
            / "HoloMol/code/pipeline/experiments/scaling/orchestrate_domainwise_145_campaign.py"
        )
        result = self.ssh(
            [
                str(self.remote_python),
                str(remote_script),
                "--flock-probe-path",
                str(canonical(path)),
            ],
            timeout=30,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("remote flock probe produced no JSON")
        value = json.loads(lines[-1])
        if not isinstance(value, dict) or value.get("status") not in {
            "blocked",
            "acquired",
        }:
            raise RuntimeError(f"invalid remote flock probe: {value}")
        return value

    def preflight_b_shared_flock(self) -> dict[str, Any]:
        path = self.b_root / "controller_b_launch_cross_host_flock_preflight.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            contended = self.remote_flock_probe(path)
            if contended.get("status") != "blocked":
                raise RuntimeError(
                    "shared-storage flock is not mutually visible across hosts"
                )
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        released = self.remote_flock_probe(path)
        if released.get("status") != "acquired":
            raise RuntimeError("remote host cannot acquire released shared flock")
        return {
            "status": "verified",
            "path": str(path),
            "contended_probe": contended,
            "released_probe": released,
        }

    def observe_b(self) -> dict[str, Any]:
        status = self.root / "b_watchdog_current.json"
        command = [
            str(self.python),
            str(self.script_dir / "watch_domainwise_campaign.py"),
            "--campaign-root",
            str(self.b_root),
            "--local-matrix",
            self.config.b_local_matrix,
            "--remote-matrix",
            self.config.b_remote_matrix,
            "--status-json",
            str(status),
            "--alerts-jsonl",
            str(self.root / "b_watchdog_alerts.jsonl"),
            "--lock-file",
            f"/dev/shm/holomol-{os.getuid()}/scaling/domainwise145_controller_b_watch.lock",
            "--pid-file",
            str(self.root / "b_watchdog_once.pid.json"),
            "--remote-host",
            self.config.remote_host,
            "--remote-port",
            str(self.config.remote_port),
            "--remote-identity",
            str(self.remote_identity),
            "--once",
        ]
        try:
            self.run(command, timeout=90)
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            raise TransientObservationError(
                f"Phase-B live observation unavailable: {exc}"
            ) from exc
        report = load_json(status)
        if not isinstance(report, dict) or report.get("schema_version") != "domainwise_campaign_watchdog_v1":
            raise ValueError("B watchdog returned an invalid report")
        return report

    @staticmethod
    def b_host_activity(report: dict[str, Any], host_id: str) -> dict[str, Any]:
        host = report.get("hosts", {}).get(host_id)
        if not isinstance(host, dict):
            return {"observable": False, "active": None, "problems": ["host probe missing"]}
        supervisor = host.get("supervisor_process")
        workers = host.get("workers", {})
        active_workers = [
            slot
            for slot, item in workers.items()
            if isinstance(item, dict)
            and item.get("status") in {"running", "paused_thermal", "stopped_unowned", "identity_mismatch"}
        ]
        worker_identities = {
            slot: workers[slot]
            for slot in active_workers
            if slot in workers
        }
        problems = [
            f"{slot}:{item.get('status')}"
            for slot, item in workers.items()
            if isinstance(item, dict) and item.get("status") in {"stopped_unowned", "identity_mismatch"}
        ]
        return {
            "observable": True,
            "active": isinstance(supervisor, dict) or bool(active_workers),
            "supervisor": supervisor,
            "active_workers": active_workers,
            "worker_identities": worker_identities,
            "problems": problems,
        }

    def b_launch_authority_paths(self, host_id: str) -> tuple[Path, Path]:
        if host_id not in {"local", "remote"}:
            raise ValueError(f"invalid Phase-B host: {host_id}")
        return (
            self.b_root / f"controller_b_launch_authority_{host_id}.json",
            self.b_root / f"controller_b_launch_authority_{host_id}.lock",
        )

    def set_b_launch_authority(
        self, host_id: str, *, allowed: bool, launch_nonce: str | None
    ) -> dict[str, Any]:
        marker, lock_path = self.b_launch_authority_paths(host_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            value = {
                "schema_version": "domainwise_phase_b_launch_authority_v1",
                "host_id": host_id,
                "allowed": bool(allowed),
                "launch_nonce": launch_nonce,
                "updated_at": now(),
            }
            atomic_json(marker, value)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
        return {"status": "allowed" if allowed else "revoked", "path": str(marker), **value}

    def launch_b_host(
        self, host_id: str, launch_nonce: str | None = None
    ) -> dict[str, Any]:
        launch_nonce = launch_nonce or f"manual_{host_id}"
        safe_component(launch_nonce, "Phase-B launch nonce")
        authority = self.set_b_launch_authority(
            host_id, allowed=True, launch_nonce=launch_nonce
        )
        matrix = self.config.b_local_matrix if host_id == "local" else self.config.b_remote_matrix
        manifest = self.local_manifest if host_id == "local" else self.remote_manifest
        python = self.python if host_id == "local" else self.remote_python
        environment = [
            "/usr/bin/env",
            f"HOLOMOL_SCALING_CAMPAIGN_ROOT={self.b_root}",
            f"HOLOMOL_SCALING_RUN_ROOT={self.b_root.parent}",
            f"HOLOMOL_SCALING_ENV_ROOT={python.parent.parent}",
            f"HOLOMOL_SCALING_DOMAIN_MANIFEST={manifest}",
            f"HOLOMOL_SCALING_LAUNCH_AUTHORITY={authority['path']}",
            f"HOLOMOL_SCALING_LAUNCH_NONCE={launch_nonce}",
            f"MATRIX_JSON={matrix}",
        ]
        if host_id == "local":
            result = self.run(
                [
                    *environment,
                    "/usr/bin/bash",
                    str(self.script_dir / "launch_domainwise_phase_b_local.sh"),
                    "start",
                    "local",
                ],
                timeout=180,
            )
        else:
            remote_launcher = canonical(self.config.remote_workspace) / "HoloMol/code/pipeline/experiments/scaling/launch_domainwise_phase_b_local.sh"
            result = self.ssh(
                [
                    *environment,
                    "/usr/bin/bash",
                    str(remote_launcher),
                    "start",
                    "remote",
                ],
                timeout=180,
            )
        return {
            "status": "launch_command_complete",
            "host_id": host_id,
            "launch_nonce": launch_nonce,
            "authority": authority,
            "stdout": result.stdout[-2000:],
        }

    def revoke_b_launch_host(self, host_id: str) -> dict[str, Any]:
        return self.set_b_launch_authority(
            host_id, allowed=False, launch_nonce=None
        )

    def strict_b(self) -> dict[str, Any]:
        output = self.root / "phase_b_strict_checkpoint_verification.json"
        command = [
            str(self.python),
            str(self.script_dir / "verify_domainwise_campaign.py"),
            "--matrix-json",
            self.config.b_local_matrix,
            "--matrix-json",
            self.config.b_remote_matrix,
            "--expected-total-rows",
            "60",
            "--output-json",
            str(output),
            "--require-complete",
        ]
        self.run(command, timeout=3600)
        report = load_json(output)
        return {"path": str(output), **self.compact_report(report)}

    def handoff_b_host(self, host_id: str) -> dict[str, Any]:
        output = self.root / f"phase_b_handoff_{host_id}.json"
        matrix = self.config.b_local_matrix if host_id == "local" else self.config.b_remote_matrix
        command = [
            str(self.python if host_id == "local" else self.remote_python),
            str(self.script_dir / "quiesce_domainwise_phase_b.py"),
            "handoff",
            "--matrix-json",
            matrix,
            "--campaign-root",
            str(self.b_root),
            "--host-id",
            host_id,
            "--output-json",
            str(output),
            "--timeout-seconds",
            "120",
        ]
        if host_id == "local":
            command.append("--stop-watchdog")
            self.run(command, timeout=180)
        else:
            self.ssh(command, timeout=180)
        report = load_json(output)
        if report.get("status") != "quiescent" or report.get("guard_release_verified") is not True:
            raise RuntimeError(f"Phase-B handoff not proven on {host_id}: {report}")
        return self.compact_report(report) | {
            "status": "quiescent",
            "host_id": host_id,
            "guard_release_verified": True,
            "path": str(output),
        }

    def review_b(self) -> dict[str, Any]:
        output = self.followup / "phase_c" / "phase_b_validation_review.json"
        if output.is_file():
            existing = load_json(output)
            if not isinstance(existing, dict) or existing.get("schema_version") != REVIEW_SCHEMA:
                raise ValueError("existing Phase-B review is malformed; refusing overwrite")
            validated_runs = existing.get("phase_b_primary_runs_validated")
            final_review = (
                existing.get("status")
                in {
                    "approved_for_phase_c_freeze",
                    "not_approved_for_phase_c_freeze",
                    "blocked_review_error",
                    "blocked_invalid_phase_b",
                }
                or (
                    isinstance(validated_runs, int)
                    and not isinstance(validated_runs, bool)
                    and validated_runs == 60
                )
            )
            if final_review:
                if existing.get("test_data_accessed") is not False:
                    raise ValueError("existing final review violates the validation-only gate")
                # Validate an approved file against the current proposal, but
                # never regenerate it merely because controller state was lost.
                if existing.get("status") == "approved_for_phase_c_freeze":
                    approval = self.review_is_approved()
                    if approval.get("approved") is not True:
                        raise ValueError(f"existing approved review is invalid: {approval}")
                return {
                    "path": str(output),
                    "status": existing.get("status"),
                    "test_data_accessed": existing.get("test_data_accessed"),
                    "approval_blockers": existing.get("approval_blockers", []),
                    "phase_b_primary_runs_validated": existing.get("phase_b_primary_runs_validated"),
                    "phase_b_curves_reviewed": existing.get("phase_b_curves_reviewed"),
                    "write_disposition": "reused_existing_final_review",
                }
        command = [
            str(self.python),
            str(self.script_dir / "review_phase_b_validation_bracketing.py"),
            "--matrix-json",
            self.config.b_local_matrix,
            "--matrix-json",
            self.config.b_remote_matrix,
            "--proposal-json",
            str(self.proposal),
            "--output-json",
            str(output),
        ]
        self.run(command, timeout=3600)
        value = load_json(output)
        if not isinstance(value, dict):
            raise ValueError("review is not an object")
        return {
            "path": str(output),
            "status": value.get("status"),
            "test_data_accessed": value.get("test_data_accessed"),
            "approval_blockers": value.get("approval_blockers", []),
            "phase_b_primary_runs_validated": value.get("phase_b_primary_runs_validated"),
            "phase_b_curves_reviewed": value.get("phase_b_curves_reviewed"),
        }

    def review_is_approved(self) -> dict[str, Any]:
        path = self.followup / "phase_c" / "phase_b_validation_review.json"
        review = load_json(path)
        proposal = load_json(self.proposal)
        if not isinstance(review, dict) or review.get("schema_version") != REVIEW_SCHEMA:
            return {"approved": False, "reason": "review schema is invalid", "path": str(path)}
        if review.get("status") != "approved_for_phase_c_freeze" or review.get("test_data_accessed") is not False:
            return {"approved": False, "reason": str(review.get("approval_blockers", review.get("status"))), "path": str(path)}
        rows = [
            row
            for row in proposal.get("rows", [])
            if isinstance(row, dict) and row.get("execution_phase") == PHASES["c"]["execution_phase"]
        ]
        validate_phase_c_review(review, proposal, rows)
        return {"approved": True, "path": str(path), "semantic_fingerprint": canonical_fingerprint(review)}

    def phase_root(self, phase: str) -> Path:
        return self.followup / f"phase_{phase}"

    def preview_path(self, phase: str) -> Path:
        return self.phase_root(phase) / "design_dry_run_matrix.json"

    def matrix_path(self, phase: str) -> Path:
        return self.phase_root(phase) / "matrix.json"

    def smoke_report_path(self, phase: str) -> Path:
        return self.phase_root(phase) / "batch_profile_smoke" / "report.json"

    def build_preview(self, phase: str) -> dict[str, Any]:
        command = [
            str(self.python),
            str(self.script_dir / "finalize_domainwise_phase_matrix.py"),
            "--phase",
            phase,
            "--proposal-json",
            str(self.proposal),
            "--domain-split-manifest",
            str(self.shared_manifest),
            "--output-json",
            str(self.preview_path(phase)),
            "--run-root",
            str(self.followup.parent),
            "--experiment-name",
            self.config.experiment_name,
            "--design-dry-run",
        ]
        if phase == "c":
            command.extend(["--review-json", str(self.followup / "phase_c/phase_b_validation_review.json")])
        result = self.run(command, timeout=300)
        matrix = load_json(self.preview_path(phase))
        if (
            matrix.get("metadata", {}).get("status") != "design_dry_run_not_runnable"
            or matrix.get("metadata", {}).get("phase") != phase
            or len(matrix.get("rows", [])) != PHASE_EXPECTED[phase]
        ):
            raise RuntimeError("finalizer did not produce the expected design preview")
        return {
            "status": "preview_ready",
            "phase": phase,
            "path": str(self.preview_path(phase)),
            "generation_id": matrix["metadata"]["generation_id"],
            "profile_contract_id": matrix["metadata"]["batch_profile_evidence_gate"]["profile_contract_id"],
            "stdout": result.stdout[-1000:],
        }

    def idle_probe(self, host_id: str) -> dict[str, Any]:
        if host_id == "local":
            return {
                "host_id": host_id,
                "scaling_processes": active_scaling_processes(),
                "compute_processes": running_compute_processes(),
            }
        command = [
            str(self.remote_python),
            str(Path(__file__).resolve()),
            "--idle-probe",
            "--probe-host-id",
            "remote",
        ]
        try:
            result = self.ssh(command, timeout=30)
        except Exception as exc:
            raise TransientObservationError(
                f"remote idle observation unavailable: {exc}"
            ) from exc
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise TransientObservationError("remote idle probe produced no output")
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise TransientObservationError(
                f"remote idle probe returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict) or value.get("host_id") != "remote":
            raise TransientObservationError("remote idle probe is invalid")
        return value

    def smoke_job_spec(self, phase: str, host_id: str, attempt: int) -> dict[str, Any]:
        base = self.phase_root(phase) / "batch_profile_smoke" / f"attempt_{attempt:03d}" / host_id
        output_root = base / "host_run"
        fragment = output_root / "host_fragment.json"
        python = self.python if host_id == "local" else self.remote_python
        manifest = self.local_manifest if host_id == "local" else self.remote_manifest
        command = [
            str(python),
            str(self.script_dir / "run_domainwise_batch_profile_smoke.py"),
            "run-host",
            "--design-matrix",
            str(self.preview_path(phase)),
            "--phase",
            phase,
            "--host-id",
            host_id,
            "--domain-split-manifest",
            str(manifest),
            "--output",
            str(output_root),
            "--pipeline-root",
            str(self.pipeline),
            "--python",
            str(python),
            "--abort-temperature-c",
            "90",
        ]
        nonce = f"smoke_{phase}_{host_id}_{attempt:03d}_{uuid.uuid4().hex[:12]}"
        return self.make_job_spec(
            job_name=f"smoke_{phase}_{host_id}_{attempt:03d}",
            host_id=host_id,
            nonce=nonce,
            base=base,
            command=command,
            expected_artifact=fragment,
            extra={
                "phase": phase,
                "attempt": attempt,
                "fragment": str(fragment),
                "output_root": str(output_root),
                "max_runtime_seconds": 7200.0,
                "orphan_timeout_seconds": 300.0,
                "identity_publication_timeout_seconds": 60.0,
                "terminal_cleanup_timeout_seconds": 60.0,
            },
        )

    def make_job_spec(
        self,
        *,
        job_name: str,
        host_id: str,
        nonce: str,
        base: Path,
        command: list[str],
        expected_artifact: Path | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_component(job_name, "job_name")
        safe_component(nonce, "nonce")
        spec = {
            "job_name": job_name,
            "host_id": host_id,
            "generation_nonce": nonce,
            "base": str(base),
            "pid_json": str(base / "job.pid.json"),
            "result_json": str(base / "job.result.json"),
            "console_log": str(base / "job.console.log"),
            "lock_file": f"/dev/shm/holomol-{os.getuid()}/scaling/{job_name}.lock",
            "expected_artifact": str(expected_artifact) if expected_artifact else None,
            "command": command,
            "command_fingerprint": command_fingerprint(command),
            **(extra or {}),
        }
        if job_name.startswith("smoke_"):
            # One host-global lock spans C/D and every retry.  Combined with
            # the pre-spawn SID receipt, this closes cross-attempt TOCTOU.
            spec["lock_file"] = (
                f"/dev/shm/holomol-{os.getuid()}/scaling/"
                f"domainwise_profile_smoke_{host_id}.lock"
            )
        return spec

    def wrapper_command(self, spec: dict[str, Any]) -> list[str]:
        python = self.python if spec["host_id"] == "local" else self.remote_python
        command = [
            str(python),
            str(self.script_dir / "run_domainwise_detached_job.py"),
            "--job-name",
            str(spec["job_name"]),
            "--generation-nonce",
            str(spec["generation_nonce"]),
            "--pid-json",
            str(spec["pid_json"]),
            "--result-json",
            str(spec["result_json"]),
            "--console-log",
            str(spec["console_log"]),
            "--lock-file",
            str(spec["lock_file"]),
            "--cwd",
            str(self.pipeline),
            "--require-detached",
        ]
        if spec.get("expected_artifact"):
            command.extend(["--expected-artifact", str(spec["expected_artifact"])])
        return [*command, "--", *[str(item) for item in spec["command"]]]

    def launch_job(self, spec: dict[str, Any]) -> dict[str, Any]:
        base = canonical(spec["base"])
        base.mkdir(parents=True, exist_ok=True)
        wrapper = self.wrapper_command(spec)
        launch = ["setsid", "-f", *wrapper]
        if spec["host_id"] == "local":
            subprocess.run(
                launch,
                cwd=self.pipeline,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        else:
            # A non-interactive SSH stdout is a pipe, so nohup alone would
            # leave that pipe held by the detached process and make SSH wait.
            remote_shell = f"nohup {shlex.join(launch)} >/dev/null 2>&1 </dev/null"
            self.ssh(["bash", "-lc", remote_shell], timeout=30)
        deadline = time.monotonic() + 30
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.probe_job(spec)
            if last.get("state") in {
                "running",
                "starting",
                "orphan_session_running",
                "terminal_result_session_running",
                "orphan_completed_unwrapped",
                "complete",
                "failed",
                "invalid",
            }:
                return last
            time.sleep(0.25)
        # Missing after the full publication window proves no receipt; an
        # unavailable remote observation remains ambiguous and is reconciled
        # without spending another launch attempt.
        if last.get("state") in {"missing", "observation_unavailable"}:
            return {**last, "launch_probe_timeout": True}
        raise RuntimeError(f"detached job did not publish an identity/result: {spec['job_name']} last={last}")

    def probe_job(self, spec: dict[str, Any]) -> dict[str, Any]:
        if spec["host_id"] == "local":
            return probe_detached_job(spec, self.script_dir / "run_domainwise_detached_job.py")
        command = [
            str(self.remote_python),
            str(Path(__file__).resolve()),
            "--job-probe-json",
            json.dumps(spec, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        ]
        try:
            result = self.ssh(command, timeout=30)
        except Exception as exc:
            return {
                "state": "observation_unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return {"state": "observation_unavailable", "error": "remote probe produced no output"}
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            return {"state": "observation_unavailable", "error": f"invalid remote JSON: {exc}"}
        if not isinstance(value, dict):
            return {"state": "observation_unavailable", "error": "remote probe root is not an object"}
        return value

    def cleanup_job(self, spec: dict[str, Any]) -> dict[str, Any]:
        if spec["host_id"] == "local":
            return cleanup_detached_job(
                spec, self.script_dir / "run_domainwise_detached_job.py"
            )
        command = [
            str(self.remote_python),
            str(Path(__file__).resolve()),
            "--cleanup-job-json",
            json.dumps(spec, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        ]
        result = self.ssh(command, timeout=120)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("remote detached cleanup produced no output")
        value = json.loads(lines[-1])
        if not isinstance(value, dict) or value.get("status") != "clean":
            raise RuntimeError(f"remote detached cleanup failed: {value}")
        return value

    def validate_smoke_fragment(self, phase: str, host_id: str, path: Path) -> dict[str, Any]:
        fragment = load_json(path)
        preview = load_json(self.preview_path(phase))
        if (
            not isinstance(fragment, dict)
            or fragment.get("schema_version") != "domainwise_batch_profile_evidence_host_fragment_v1"
            or fragment.get("status") != "complete"
            or fragment.get("phase") != phase
            or fragment.get("host_id") != host_id
            or fragment.get("design_generation_id") != preview["metadata"]["generation_id"]
            or fragment.get("profile_contract_id")
            != preview["metadata"]["batch_profile_evidence_gate"]["profile_contract_id"]
            or fragment.get("test_data_accessed") is not False
            or not isinstance(fragment.get("entries"), list)
            or not fragment["entries"]
        ):
            raise ValueError(f"invalid smoke fragment: {path}")
        return {"status": "complete", "path": str(path), "entries": len(fragment["entries"])}

    def merge_smoke(self, phase: str, fragments: dict[str, Path]) -> dict[str, Any]:
        output = self.smoke_report_path(phase)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            str(self.script_dir / "run_domainwise_batch_profile_smoke.py"),
            "merge",
            "--design-matrix",
            str(self.preview_path(phase)),
            "--phase",
            phase,
            "--fragments",
            str(fragments["local"]),
            str(fragments["remote"]),
            "--output",
            str(output),
        ]
        self.run(command, timeout=300)
        report = load_json(output)
        if report.get("status") != "approved_for_formal_freeze" or report.get("test_data_accessed") is not False:
            raise RuntimeError("merged smoke evidence was not approved")
        return {"status": report["status"], "path": str(output), "entries": len(report.get("entries", []))}

    def formalize(self, phase: str) -> dict[str, Any]:
        command = [
            str(self.python),
            str(self.script_dir / "finalize_domainwise_phase_matrix.py"),
            "--phase",
            phase,
            "--proposal-json",
            str(self.proposal),
            "--domain-split-manifest",
            str(self.shared_manifest),
            "--smoke-report-json",
            str(self.smoke_report_path(phase)),
            "--output-json",
            str(self.matrix_path(phase)),
            "--run-root",
            str(self.followup.parent),
            "--experiment-name",
            self.config.experiment_name,
        ]
        if phase == "c":
            command.extend(["--review-json", str(self.followup / "phase_c/phase_b_validation_review.json")])
        result = self.run(command, timeout=300)
        matrix = load_json(self.matrix_path(phase))
        expected_status = PHASES[phase]["formal_status"]
        if (
            matrix.get("metadata", {}).get("status") != expected_status
            or matrix.get("metadata", {}).get("is_runnable") is not True
            or len(matrix.get("rows", [])) != PHASE_EXPECTED[phase]
        ):
            raise RuntimeError("formal matrix freeze failed validation")
        return {
            "status": "formal_matrix_ready",
            "phase": phase,
            "path": str(self.matrix_path(phase)),
            "generation_id": matrix["metadata"]["generation_id"],
            "stdout": result.stdout[-1000:],
        }

    def phase_hosts(self, phase: str) -> tuple[str, ...]:
        """Return the exact hosts which own rows in a frozen formal matrix."""
        matrix = load_json(self.matrix_path(phase))
        if not isinstance(matrix, dict) or not isinstance(matrix.get("rows"), list):
            raise ValueError(f"Phase-{phase.upper()} formal matrix is unavailable")
        hosts = tuple(
            sorted(
                {
                    safe_component(str(row["assigned_host"]), "assigned_host")
                    for row in matrix["rows"]
                    if isinstance(row, dict)
                }
            )
        )
        if not hosts or any(host not in {"local", "remote"} for host in hosts):
            raise ValueError(f"Phase-{phase.upper()} formal matrix has invalid hosts: {hosts}")
        return hosts

    def adopt_existing_formal_phase(self, phase: str) -> dict[str, Any]:
        """Validate and describe a formal phase produced by a safe prefetch.

        Phase D is scientifically independent of the Phase-B bracketing review,
        so it may be executed on otherwise idle, disjoint GPUs.  Adoption never
        trusts a marker alone: the normal structural verifier revalidates the
        immutable matrix, embedded smoke evidence, and every artifact observed
        so far before the main state machine is allowed to skip preview/smoke.
        """
        if phase != "d":
            return {"status": "not_applicable", "phase": phase}
        matrix_path = self.matrix_path(phase)
        if not matrix_path.is_file():
            return {"status": "absent", "phase": phase, "path": str(matrix_path)}
        matrix = load_json(matrix_path)
        metadata = matrix.get("metadata", {}) if isinstance(matrix, dict) else {}
        if (
            metadata.get("phase") != phase
            or metadata.get("status") != PHASES[phase]["formal_status"]
            or metadata.get("is_runnable") is not True
            or len(matrix.get("rows", [])) != PHASE_EXPECTED[phase]
        ):
            return {
                "status": "invalid",
                "phase": phase,
                "path": str(matrix_path),
                "reason": "existing formal matrix identity/shape is invalid",
            }
        observation = self.observe_phase(phase)
        structural = observation.get("structural_verification", {})
        counts = structural.get("counts", {})
        if (
            observation.get("status") == "invalid"
            or structural.get("matrix_issues")
            or int(counts.get("rejected", 0))
        ):
            return {
                "status": "invalid",
                "phase": phase,
                "path": str(matrix_path),
                "observation": observation,
            }
        if observation.get("host_probe_errors"):
            raise TransientObservationError(
                f"prefetched Phase-{phase.upper()} host observation is incomplete"
            )
        return {
            "status": "adoptable",
            "phase": phase,
            "path": str(matrix_path),
            "generation_id": metadata.get("generation_id"),
            "hosts": list(self.phase_hosts(phase)),
            "observation": observation,
        }

    def observe_phase(self, phase: str) -> dict[str, Any]:
        hosts = self.phase_hosts(phase)
        placement = self.phase_local_execution_plan(phase)
        observation = evaluate_campaign(
            matrix_path=self.matrix_path(phase),
            proposal_path=self.proposal,
            phase=phase,
            remote_host=self.config.remote_host,
            remote_port=self.config.remote_port,
            remote_identity=self.remote_identity,
            remote_python=self.remote_python,
            remote_script=self.script_dir / "watch_domainwise_followup_campaign.py",
            stale_seconds=self.config.stale_seconds,
            no_remote=self.remote_runtime_disabled or "remote" not in hosts,
            **({"local_execution_plan": placement} if placement is not None else {}),
        )
        if self.remote_runtime_disabled and "remote" in hosts and placement is None:
            # This is an explicit scheduling exclusion, NOT evidence that the
            # remote process tree is absent or that any remote row is complete.
            observation["hosts"]["remote"] = {
                "host_id": "remote", "observed": False,
                "runtime_admission_disabled": True,
                "status": "operator_disabled_unobserved",
                "role_counts": {}, "issues": [],
                "warnings": ["remote runtime disabled by operator; no process/idle claim"],
            }
            observation["disabled_runtime_hosts"] = ["remote"]
            if observation.get("artifacts_ready_for_strict_verification"):
                observation["status"] = "observation_incomplete"
                observation.setdefault("host_probe_errors", {})["remote"] = "operator disabled; no remote quiescence proof"
        return observation

    def phase_launcher_environment(self, phase: str) -> list[str]:
        return [
            "/usr/bin/env",
            f"HOLOMOL_SCALING_EXPERIMENT_NAME={self.config.experiment_name}",
            f"HOLOMOL_SCALING_RUN_ROOT={self.followup.parent}",
            f"HOLOMOL_SCALING_ENV_ROOT={self.python.parent.parent}",
            f"HOLOMOL_SCALING_REMOTE_ENV_ROOT={self.remote_python.parent.parent}",
            f"HOLOMOL_SCALING_REMOTE_HOST={self.config.remote_host}",
            f"HOLOMOL_SCALING_REMOTE_PORT={self.config.remote_port}",
            f"HOLOMOL_SCALING_REMOTE_IDENTITY={self.remote_identity}",
            f"HOLOMOL_SCALING_REMOTE_WORKSPACE={self.config.remote_workspace}",
            f"HOLOMOL_SCALING_LOCAL_DOMAIN_MANIFEST={self.local_manifest}",
            f"HOLOMOL_SCALING_REMOTE_DOMAIN_MANIFEST={self.remote_manifest}",
            f"HOLOMOL_SCALING_DOMAIN_MANIFEST={self.local_manifest}",
            f"HOLOMOL_SCALING_REMOTE_MATRIX_JSON={self.matrix_path(phase)}",
            f"MATRIX_JSON={self.matrix_path(phase)}",
        ]

    def phase_launch_job_spec(self, phase: str, hosts: tuple[str, ...], attempt: int) -> dict[str, Any]:
        placement = self.phase_local_execution_plan(phase)
        if self.remote_runtime_disabled and "remote" in hosts and placement is None:
            raise RuntimeError("refusing remote launch while disabled by operator")
        if placement is not None and len(hosts) != 1:
            raise RuntimeError("local execution plan requires separate logical host admission")
        host_key = "both" if set(hosts) == {"local", "remote"} else hosts[0]
        base = self.phase_root(phase) / "controller_launches" / f"{host_key}_attempt_{attempt:03d}"
        environment = self.phase_launcher_environment(phase)
        environment.append(
            f"HOLOMOL_SCALING_LAUNCH_ACTION_ID=controller_{phase}_{host_key}_{attempt:03d}"
        )
        if host_key == "both":
            command = [*environment, "/usr/bin/bash", str(self.script_dir / "launch_domainwise_phase_two_hosts.sh"), "start", phase]
            host_id = "local"
        elif host_key == "local":
            command = [*environment, "/usr/bin/bash", str(self.script_dir / "launch_domainwise_phase_host.sh"), "start", phase, "local"]
            host_id = "local"
        elif placement is not None:
            environment.extend([
                f"HOLOMOL_SCALING_LOCAL_EXECUTION_PLAN={placement}",
                "HOLOMOL_SCALING_DEFER_UNAVAILABLE_GPUS=1",
            ])
            command = [*environment, "/usr/bin/bash", str(self.script_dir / "launch_domainwise_phase_host.sh"), "start", phase, "remote"]
            host_id = "local"
        else:
            remote_environment = [
                token.replace(
                    f"HOLOMOL_SCALING_DOMAIN_MANIFEST={self.local_manifest}",
                    f"HOLOMOL_SCALING_DOMAIN_MANIFEST={self.remote_manifest}",
                ).replace(
                    f"HOLOMOL_SCALING_ENV_ROOT={self.python.parent.parent}",
                    f"HOLOMOL_SCALING_ENV_ROOT={self.remote_python.parent.parent}",
                )
                for token in environment
            ]
            if phase == "c":
                remote_environment.append("HOLOMOL_SCALING_DEFER_UNAVAILABLE_GPUS=1")
            command = [*remote_environment, "/usr/bin/bash", str(self.script_dir / "launch_domainwise_phase_host.sh"), "start", phase, "remote"]
            host_id = "remote"
        nonce = f"launch_{phase}_{host_key}_{attempt:03d}_{uuid.uuid4().hex[:12]}"
        return self.make_job_spec(
            job_name=f"launch_{phase}_{host_key}_{attempt:03d}",
            host_id=host_id,
            nonce=nonce,
            base=base,
            command=command,
            expected_artifact=None,
            extra={
                "phase": phase,
                "hosts": list(hosts),
                "physical_host_id": host_id,
                "local_execution_plan_json": str(placement) if placement is not None and host_key == "remote" else None,
                "admission_policy": "defer_unavailable_gpus_v1" if phase == "c" and host_key == "remote" else "strict_host_idle",
                "attempt": attempt,
                "max_runtime_seconds": 600.0,
                "orphan_timeout_seconds": 120.0,
                "identity_publication_timeout_seconds": 60.0,
                "terminal_cleanup_timeout_seconds": 60.0,
            },
        )

    def stop_phase_host(
        self,
        phase: str,
        host_id: str,
        cleanup_action_id: str | None = None,
    ) -> dict[str, Any]:
        cleanup_action_id = cleanup_action_id or (
            f"manual_{phase}_{host_id}_{uuid.uuid4().hex[:16]}"
        )
        safe_component(cleanup_action_id, "phase cleanup action id")
        intent = prepare_cleanup_intent(
            self.phase_root(phase), phase, host_id, cleanup_action_id
        )
        lifecycle_environment = [
            f"HOLOMOL_SCALING_LIFECYCLE_ACTION_ID={cleanup_action_id}",
            f"HOLOMOL_SCALING_LIFECYCLE_SEQUENCE={int(intent['sequence'])}",
        ]
        environment = self.phase_launcher_environment(phase)
        placement = self.phase_local_execution_plan(phase)
        if placement is not None and host_id == "remote":
            environment.extend([
                f"HOLOMOL_SCALING_LOCAL_EXECUTION_PLAN={placement}",
                "HOLOMOL_SCALING_DEFER_UNAVAILABLE_GPUS=1",
            ])
        if host_id == "local" or placement is not None:
            result = self.run(
                [
                    *environment,
                    *lifecycle_environment,
                    "/usr/bin/bash",
                    str(self.script_dir / "launch_domainwise_phase_host.sh"),
                    "stop",
                    phase,
                    host_id,
                ],
                timeout=240,
            )
        else:
            remote_launcher = canonical(self.config.remote_workspace) / "HoloMol/code/pipeline/experiments/scaling/launch_domainwise_phase_host.sh"
            remote_environment = [
                token.replace(
                    f"HOLOMOL_SCALING_DOMAIN_MANIFEST={self.local_manifest}",
                    f"HOLOMOL_SCALING_DOMAIN_MANIFEST={self.remote_manifest}",
                ).replace(
                    f"HOLOMOL_SCALING_ENV_ROOT={self.python.parent.parent}",
                    f"HOLOMOL_SCALING_ENV_ROOT={self.remote_python.parent.parent}",
                )
                for token in environment
            ]
            result = self.ssh(
                [
                    *remote_environment,
                    *lifecycle_environment,
                    "/usr/bin/bash",
                    str(remote_launcher),
                    "stop",
                    phase,
                    "remote",
                ],
                timeout=240,
            )
        return {
            "status": "stop_complete",
            "phase": phase,
            "host_id": host_id,
            "cleanup_action_id": cleanup_action_id,
            "cleanup_sequence": int(intent["sequence"]),
            "stdout": result.stdout[-2000:],
        }

    def strict_phase(self, phase: str) -> dict[str, Any]:
        output = self.root / f"phase_{phase}_strict_checkpoint_verification.json"
        command = [
            str(self.python),
            str(self.script_dir / "verify_domainwise_followup_campaign.py"),
            "--phase-matrix",
            str(self.matrix_path(phase)),
            "--proposal-json",
            str(self.proposal),
            "--output-json",
            str(output),
            "--require-complete",
        ]
        self.run(command, timeout=3600)
        bundle = load_json(output)
        phase_report = bundle.get("phases", [None])[0] if isinstance(bundle, dict) else None
        if not isinstance(phase_report, dict):
            raise ValueError("strict phase verifier did not return a phase report")
        return {"path": str(output), **self.compact_report(phase_report)}

    def union_145(self) -> dict[str, Any]:
        output = self.root / "union145_strict_checkpoint_verification.json"
        command = [
            str(self.python),
            str(self.script_dir / "verify_domainwise_followup_campaign.py"),
            "--phase-matrix",
            str(self.matrix_path("c")),
            "--phase-matrix",
            str(self.matrix_path("d")),
            "--b-matrix",
            self.config.b_local_matrix,
            "--b-matrix",
            self.config.b_remote_matrix,
            "--proposal-json",
            str(self.proposal),
            "--output-json",
            str(output),
            "--union145",
            "--require-complete",
        ]
        self.run(command, timeout=7200)
        report = load_json(output)
        return {"path": str(output), **self.compact_report(report), "design_contract": report.get("design_contract")}


def probe_detached_job(spec: dict[str, Any], wrapper_script: Path) -> dict[str, Any]:
    """Validate a detached job from its terminal result and/or live /proc."""
    expected = {
        "job_name": str(spec["job_name"]),
        "generation_nonce": str(spec["generation_nonce"]),
        "command_fingerprint": str(spec["command_fingerprint"]),
    }
    result_path = canonical(spec["result_json"])
    if result_path.is_file():
        result = load_json(result_path)
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != DETACHED_RESULT_SCHEMA
            or any(result.get(key) != value for key, value in expected.items())
            or int(result.get("pid", -1)) <= 1
            or int(result.get("start_ticks", -1)) <= 0
            or not isinstance(result.get("started_at"), str)
            or not isinstance(result.get("finished_at"), str)
        ):
            return {"state": "invalid", "reason": "detached result identity mismatch", "result": result}
        try:
            result_started = datetime.fromisoformat(result["started_at"])
            result_finished = datetime.fromisoformat(result["finished_at"])
            if result_started.tzinfo is None or result_finished.tzinfo is None:
                raise ValueError("timezone-naive timestamp")
        except ValueError as exc:
            return {
                "state": "invalid",
                "reason": f"detached result timestamp is invalid: {exc}",
                "result": result,
            }
        session_members = owned_session_members(int(result["pid"]))
        recycled_leader = next(
            (
                member
                for member in session_members
                if int(member["pid"]) == int(result["pid"])
                and int(member["start_ticks"]) != int(result.get("start_ticks", -1))
            ),
            None,
        )
        if recycled_leader is not None:
            # The old session is gone and Linux has since reused its numeric
            # leader PID.  Never adopt or signal that unrelated new session.
            session_members = []
        if session_members:
            return {
                "state": "terminal_result_session_running",
                "terminal_state": (
                    "complete"
                    if result.get("status") == "complete"
                    and int(result.get("returncode", -1)) == 0
                    else "failed"
                ),
                "result": result,
                "session_members": session_members,
                "ownership": "same_uid_members_of_terminal_result_sid",
            }
        return {
            "state": "complete" if result.get("status") == "complete" and int(result.get("returncode", -1)) == 0 else "failed",
            "result": result,
        }

    pid_path = canonical(spec["pid_json"])
    identity: dict[str, Any] | None = None
    if pid_path.is_file():
        try:
            loaded = load_json(pid_path)
            if (
                not isinstance(loaded, dict)
                or loaded.get("schema_version") != DETACHED_IDENTITY_SCHEMA
                or any(loaded.get(key) != value for key, value in expected.items())
                or int(loaded.get("pid", -1)) <= 1
                or int(loaded.get("sid", -1)) != int(loaded.get("pid", -2))
                or int(loaded.get("pgid", -1)) != int(loaded.get("pid", -2))
                or int(loaded.get("ppid", -1)) != 1
                or loaded.get("command") != [str(item) for item in spec["command"]]
                or canonical(loaded.get("result_json")) != result_path
                or not isinstance(loaded.get("started_at"), str)
            ):
                return {"state": "invalid", "reason": "detached PID record identity mismatch", "identity": loaded}
            identity_started = datetime.fromisoformat(str(loaded["started_at"]))
            if identity_started.tzinfo is None:
                raise ValueError("detached PID timestamp is timezone-naive")
            identity = loaded
        except Exception as exc:
            return {"state": "invalid", "reason": f"cannot read detached PID record: {type(exc).__name__}: {exc}"}

    candidates: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            proc = process_identity(int(entry.name))
        except (FileNotFoundError, OSError, PermissionError, ProcessLookupError, ValueError):
            continue
        command = proc["cmdline"]
        if not exact_script(command, wrapper_script):
            continue
        try:
            if (
                one_arg(command, "--job-name") == expected["job_name"]
                and one_arg(command, "--generation-nonce") == expected["generation_nonce"]
                and canonical(one_arg(command, "--result-json")) == result_path
            ):
                candidates.append(proc)
        except (OSError, ValueError):
            continue
    if len(candidates) > 1:
        return {"state": "invalid", "reason": "multiple detached job generations", "processes": candidates}
    if not candidates:
        # The wrapper records its session before Popen.  If it was SIGKILLed,
        # every surviving child is still discoverable by that immutable SID;
        # adopt this exact same-UID session instead of launching a duplicate.
        if identity is not None:
            session_members = owned_session_members(
                int(identity["sid"]), exclude_pid=int(identity["pid"])
            )
            leader = owned_session_members(int(identity["sid"]))
            if any(
                int(member["pid"]) == int(identity["pid"])
                and int(member["start_ticks"]) != int(identity["start_ticks"])
                for member in leader
            ):
                session_members = []
            if session_members:
                return {
                    "state": "orphan_session_running",
                    "identity": identity,
                    "session_members": session_members,
                    "ownership": "same_uid_members_of_pre_spawn_recorded_sid",
                }
        output_exists = bool(spec.get("expected_artifact") and canonical(spec["expected_artifact"]).exists())
        return {
            "state": "orphan_completed_unwrapped" if output_exists and identity is not None else "missing",
            "output_exists": output_exists,
            "identity": identity,
        }
    proc = candidates[0]
    if (
        int(proc["ppid"]) != 1
        or int(proc["pgid"]) != int(proc["pid"])
        or int(proc["sid"]) != int(proc["pid"])
    ):
        return {"state": "invalid", "reason": "detached process is not PPID1/session leader", "process": proc}
    if identity is None:
        return {
            "state": "starting",
            "process": proc,
            "identity_error": "wrapper is visible before its atomic PID record",
        }
    if (
        any(int(identity.get(key, -1)) != int(proc[key]) for key in ("pid", "ppid", "pgid", "sid", "start_ticks"))
    ):
        return {"state": "invalid", "reason": "detached PID identity mismatch", "identity": identity, "process": proc}
    return {"state": "running", "identity": identity, "process": proc}


def owned_session_members(
    sid: int, *, exclude_pid: int | None = None
) -> list[dict[str, Any]]:
    """Snapshot non-zombie, same-UID members of one exact Linux session."""

    members: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if exclude_pid is not None and pid == exclude_pid:
            continue
        try:
            member = process_identity(pid)
        except (FileNotFoundError, OSError, PermissionError, ProcessLookupError, ValueError):
            continue
        if int(member["sid"]) == int(sid) and member.get("state") not in {"Z", "X", "x"}:
            members.append(member)
    return sorted(members, key=lambda item: int(item["pid"]))


def signal_owned_snapshot(snapshot: dict[str, Any], signum: int) -> bool:
    """Signal a PID only after revalidating its immutable process identity."""

    pid = int(snapshot["pid"])
    try:
        current = process_identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        return False
    for key in ("pid", "start_ticks", "sid"):
        if int(current[key]) != int(snapshot[key]):
            raise RuntimeError(f"process identity changed before signal pid={pid} key={key}")
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        return False
    return True


def detached_orphan_age_seconds(probe: dict[str, Any]) -> float:
    if probe.get("state") == "terminal_result_session_running":
        result = probe.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("finished_at"), str):
            raise ValueError("terminal detached probe lacks its finish timestamp")
        timestamp = result["finished_at"]
    else:
        identity = probe.get("identity")
        if not isinstance(identity, dict) or not isinstance(identity.get("started_at"), str):
            raise ValueError("detached probe lacks its durable start timestamp")
        timestamp = identity["started_at"]
    started = datetime.fromisoformat(timestamp)
    if started.tzinfo is None:
        raise ValueError("detached start timestamp is timezone-naive")
    return max(0.0, (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds())


def recorded_age_seconds(value: Any) -> float:
    if not isinstance(value, str):
        raise ValueError("durable action timestamp is missing")
    started = datetime.fromisoformat(value)
    if started.tzinfo is None:
        raise ValueError("durable action timestamp is timezone-naive")
    return max(0.0, (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds())


def cleanup_detached_job(
    spec: dict[str, Any], wrapper_script: Path, *, timeout_seconds: float = 60.0
) -> dict[str, Any]:
    """Stop only the exact recorded detached session and prove it is empty.

    The PID receipt is never removed and command artifacts are never deleted.
    This makes a timed-out orphan auditable and prevents a later retry from
    overlapping unseen work from the previous generation.
    """

    if timeout_seconds <= 0:
        raise ValueError("cleanup timeout must be positive")
    probe = probe_detached_job(spec, wrapper_script)
    state = str(probe.get("state"))
    if state == "invalid":
        raise RuntimeError(f"refusing cleanup with invalid detached identity: {probe}")
    if state in {"complete", "failed", "orphan_completed_unwrapped", "missing"}:
        return {"status": "clean", "initial_state": state, "signals": [], "remaining": []}
    terminal_result = state == "terminal_result_session_running"
    if terminal_result:
        result = probe.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("terminal session probe lacks its result identity")
        sid = int(result["pid"])
        wrapper_pid = sid
    elif state == "starting":
        wrapper = probe.get("process")
        if not isinstance(wrapper, dict):
            raise RuntimeError("starting wrapper probe lacks process identity")
        sid = int(wrapper["sid"])
        wrapper_pid = int(wrapper["pid"])
    else:
        identity = probe.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError(f"cannot prove detached session ownership: {probe}")
        sid = int(identity["sid"])
        wrapper_pid = int(identity["pid"])
    signals: list[dict[str, Any]] = []

    if state in {"running", "starting"}:
        wrapper = probe.get("process")
        if not isinstance(wrapper, dict):
            raise RuntimeError("live wrapper probe lacks process identity")
        if signal_owned_snapshot(wrapper, signal.SIGTERM):
            signals.append({"pid": wrapper_pid, "signal": "TERM", "role": "wrapper"})
    elif state not in {
        "orphan_session_running",
        "terminal_result_session_running",
    }:
        raise RuntimeError(f"detached session is not safely cleanable: {probe}")

    deadline = time.monotonic() + timeout_seconds
    while True:
        members = owned_session_members(sid)
        if not members:
            return {
                "status": "clean",
                "initial_state": state,
                "sid": sid,
                "signals": signals,
                "remaining": [],
            }
        for member in members:
            # The wrapper owns signal propagation.  If it disappeared or has
            # not exited yet, exact member signalling still guarantees CONT
            # for every self-stopped training group before termination.
            if terminal_result or int(member["pid"]) != wrapper_pid:
                if signal_owned_snapshot(member, signal.SIGTERM):
                    signals.append({"pid": int(member["pid"]), "signal": "TERM"})
                try:
                    if signal_owned_snapshot(member, signal.SIGCONT):
                        signals.append({"pid": int(member["pid"]), "signal": "CONT"})
                except ProcessLookupError:
                    pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.25)

    # Escalation remains PID-by-PID and repeats the start-ticks/SID check; no
    # broad process-group or session kill is used.
    for member in owned_session_members(sid):
        try:
            if signal_owned_snapshot(member, signal.SIGKILL):
                signals.append({"pid": int(member["pid"]), "signal": "KILL"})
        except ProcessLookupError:
            pass
    kill_deadline = time.monotonic() + 10.0
    remaining = owned_session_members(sid)
    while remaining and time.monotonic() < kill_deadline:
        time.sleep(0.1)
        remaining = owned_session_members(sid)
    if remaining:
        raise RuntimeError(
            f"exact detached session still has live members after cleanup: sid={sid} members={remaining}"
        )
    return {
        "status": "clean",
        "initial_state": state,
        "sid": sid,
        "signals": signals,
        "remaining": [],
    }


def phase_host_counts(observation: dict[str, Any], matrix_path: Path) -> dict[str, dict[str, int]]:
    matrix = load_json(matrix_path)
    host_by_run = {str(row["run_id"]): str(row["assigned_host"]) for row in matrix["rows"]}
    counts = {host: {status: 0 for status in ("accepted", "pending", "rejected")} for host in ("local", "remote")}
    for run in observation.get("structural_verification", {}).get("runs", []):
        host = host_by_run.get(str(run.get("run_id")))
        status = str(run.get("status"))
        if host in counts and status in counts[host]:
            counts[host][status] += 1
    return counts


class CampaignController:
    def __init__(self, store: StateStore, operations: Operations):
        self.store = store
        self.ops = operations

    def summary(self, value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.get(key)
            for key in ("status", "complete", "phase", "counts", "health", "approved", "reason", "path")
            if key in value
        }

    def run_sync_action(
        self,
        key: str,
        kind: str,
        payload: dict[str, Any],
        function: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        entry = self.store.begin_action(key, kind, payload)
        if entry.get("status") == "complete":
            return dict(entry["result"])
        self.store.action_started(key)
        try:
            result = function()
        except (RemoteActionUncertainError, subprocess.TimeoutExpired) as exc:
            action = self.store.state["actions"][key]
            action["status"] = "started"
            action.setdefault("unresolved_outcomes", []).append(
                {"time": now(), "error": f"{type(exc).__name__}: {exc}"}
            )
            self.store.record(
                "action_outcome_unresolved",
                action_key=key,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise TransientObservationError(
                f"action {key} has an unresolved remote/subprocess outcome"
            ) from exc
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.store.action_failed(key, error, terminal=True)
            raise
        self.store.action_finished(key, result)
        return result

    def halt_exception(self, context: str, exc: BaseException) -> None:
        self.store.halt(context, {"error": f"{type(exc).__name__}: {exc}"})

    def tick(self) -> None:
        step = str(self.store.state["step"])
        if step in {"complete", "halted"}:
            return
        try:
            if step == "wait_b":
                self.tick_wait_b()
            elif step == "verify_b":
                self.tick_verify_b()
            elif step == "quiesce_b":
                self.tick_quiesce_b()
            elif step == "review_b":
                self.tick_review_b()
            elif step == "manual_review_required":
                self.tick_manual_review()
            elif step in {"c_preview", "d_preview"}:
                self.tick_preview(step[0])
            elif step in {"c_smoke", "d_smoke"}:
                self.tick_smoke(step[0])
            elif step in {"c_formalize", "d_formalize"}:
                self.tick_formalize(step[0])
            elif step in {"c_launch", "d_launch"}:
                self.tick_launch(step[0])
            elif step in {"c_run", "d_run"}:
                self.tick_run(step[0])
            elif step in {"c_verify", "d_verify"}:
                self.tick_verify_phase(step[0])
            elif step in {"c_quiesce", "d_quiesce"}:
                self.tick_quiesce_phase(step[0])
            elif step == "union_verify":
                self.tick_union()
            else:
                self.store.halt("unknown_state", {"step": step})
        except TransientObservationError as exc:
            # A failed read does not prove that a detached process stopped.
            # Keep the same state and budget so a later tick can reconcile it.
            self.store.record(
                "transient_observation_failure",
                step=step,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            if self.store.state["step"] != "halted":
                self.halt_exception(f"unhandled_failure_in_{step}", exc)

    @staticmethod
    def detached_job_expired(
        spec: dict[str, Any],
        probe: dict[str, Any],
        *,
        action_started_at: Any,
        first_orphan_observed_at: Any = None,
    ) -> tuple[bool, float]:
        state = str(probe.get("state"))
        if state == "terminal_result_session_running":
            timeout = float(spec.get("terminal_cleanup_timeout_seconds", 60.0))
            age = detached_orphan_age_seconds(probe)
        elif state == "starting":
            timeout = float(spec.get("identity_publication_timeout_seconds", 60.0))
            age = recorded_age_seconds(action_started_at)
        elif state == "orphan_session_running":
            timeout = float(spec.get("orphan_timeout_seconds", 0.0))
            age = recorded_age_seconds(first_orphan_observed_at)
        elif state == "running":
            timeout = float(spec.get("max_runtime_seconds", 0.0))
            age = detached_orphan_age_seconds(probe)
        else:
            raise ValueError(f"state is not a live detached job: {state}")
        if timeout <= 0:
            raise ValueError("detached job lacks a finite positive runtime timeout")
        return age >= timeout, age

    def phase_b_launch_nonce(self, host: str, attempt: int) -> str:
        nonce = (
            f"{self.store.state['campaign_generation_id']}_b_{host}_{attempt:03d}"
        )
        return safe_component(nonce, "Phase-B launch nonce")

    @staticmethod
    def immutable_b_runtime_identity(activity: dict[str, Any]) -> dict[str, Any]:
        def process_projection(value: Any) -> dict[str, Any] | None:
            if not isinstance(value, dict):
                return None
            process = value.get("process")
            source = process if isinstance(process, dict) else value
            return {
                key: source.get(key)
                for key in ("pid", "ppid", "pgid", "sid", "start_ticks")
                if key in source
            }

        workers = activity.get("worker_identities", {})
        return {
            "supervisor": process_projection(activity.get("supervisor")),
            "workers": {
                str(slot): process_projection(value)
                for slot, value in sorted(workers.items())
            }
            if isinstance(workers, dict)
            else {},
        }

    def settle_b_launches_for_boundary(self) -> bool:
        """Durably revoke late B starts before any B60 boundary can advance."""
        active = [
            (key, action)
            for key, action in list(self.store.state["actions"].items())
            if action.get("kind") == "phase_b_host_launch"
            and action.get("status") in {"intent", "started"}
        ]
        for key, action in active:
            if action.get("payload", {}).get("host_id") not in {
                "local",
                "remote",
            }:
                raise ValueError(f"invalid unresolved Phase-B launch action: {key}")

        revocations: dict[str, Any] = {}
        for host in ("local", "remote"):
            revocations[host] = self.run_sync_action(
                f"b_launch_authority_revoke_{host}",
                "phase_b_launch_authority_revocation",
                {
                    "host_id": host,
                    "campaign_generation_id": self.store.state[
                        "campaign_generation_id"
                    ],
                },
                lambda host=host: self.ops.revoke_b_launch_host(host),
            )

        if not self.store.state.get("b_launch_boundary_closed"):
            self.store.state["b_launch_boundary_closed"] = True
            self.store.record(
                "phase_b_launch_boundary_closed",
                revocations=revocations,
                unresolved_actions=[key for key, _ in active],
            )
        for key, action in active:
            action["revocation"] = revocations[
                str(action["payload"]["host_id"])
            ]
            self.store.action_failed(
                key,
                "Phase-B launch was durably revoked at the B60 boundary",
                terminal=True,
            )
        # An unresolved action forces a fresh B observation after revocation;
        # never run strict verification from the snapshot that triggered it.
        return bool(active)

    def tick_wait_b(self) -> None:
        flock_preflight = self.run_sync_action(
            "b_shared_flock_preflight",
            "cross_host_shared_flock_preflight",
            {
                "remote_host": self.store.config.remote_host,
                "remote_port": self.store.config.remote_port,
                "b_campaign_root": self.store.config.b_campaign_root,
            },
            self.ops.preflight_b_shared_flock,
        )
        if flock_preflight.get("status") != "verified":
            self.store.halt("phase_b_shared_flock_preflight_failed", flock_preflight)
            return
        report = self.ops.observe_b()
        completion = report.get("completion", {})
        self.store.observe("b_live", {"health": report.get("health"), "counts": completion.get("counts"), "time": report.get("time")})
        if completion.get("status") == "invalid" or int(completion.get("counts", {}).get("rejected", 0)):
            self.store.halt("phase_b_artifact_invalid", completion)
            return
        counts = completion.get("counts", {})
        if int(counts.get("accepted", 0)) == 60 and int(counts.get("pending", 0)) == 0:
            if self.settle_b_launches_for_boundary():
                return
            # Strict verification comes before signalling any live B process.
            self.store.transition("verify_b", evidence={"structural_counts": counts})
            return
        if self.store.state.get("b_launch_boundary_closed"):
            self.store.halt(
                "phase_b_completion_regressed_after_launch_revocation",
                {"counts": counts},
            )
            return
        for host in ("local", "remote"):
            activity = ProductionOperations.b_host_activity(report, host)
            if not activity["observable"]:
                continue
            if activity["problems"]:
                self.store.halt("phase_b_runtime_identity_invalid", {"host": host, **activity})
                return
            host_pending = sum(
                run.get("host_id") == host and run.get("status") == "pending"
                for run in completion.get("runs", [])
            )
            if host_pending and not activity["active"]:
                unresolved = [
                    (key, action)
                    for key, action in self.store.state["actions"].items()
                    if action.get("kind") == "phase_b_host_launch"
                    and action.get("payload", {}).get("host_id") == host
                    and action.get("status") in {"intent", "started"}
                ]
                if len(unresolved) > 1:
                    self.store.halt(
                        "phase_b_overlapping_launch_actions",
                        {"host": host, "actions": [key for key, _ in unresolved]},
                    )
                    return
                if unresolved:
                    key, action = unresolved[0]
                    recovered_attempt = int(
                        action.get("payload", {}).get("attempt", 0)
                    )
                    if not 0 < recovered_attempt <= self.store.config.max_b_launches_per_host:
                        self.store.halt(
                            "phase_b_launch_intent_budget_invalid",
                            {
                                "host": host,
                                "action_key": key,
                                "attempt": recovered_attempt,
                            },
                        )
                        return
                    counts_by_host = self.store.state["launch_counts"]["b"]
                    counts_by_host[host] = max(
                        int(counts_by_host[host]), recovered_attempt
                    )
                    self.store.record(
                        "phase_b_launch_budget_reconciled_from_intent",
                        host_id=host,
                        action_key=key,
                        attempt=recovered_attempt,
                        count=counts_by_host[host],
                    )
                    self.store.action_started(key, recovered_unresolved=True)
                    try:
                        nonce = str(
                            action.get("payload", {}).get("launch_nonce")
                            or self.phase_b_launch_nonce(host, recovered_attempt)
                        )
                        result = self.ops.launch_b_host(host, nonce)
                    except (RemoteActionUncertainError, subprocess.TimeoutExpired) as exc:
                        action = self.store.state["actions"][key]
                        action["status"] = "started"
                        action.setdefault("unresolved_outcomes", []).append(
                            {"time": now(), "error": f"{type(exc).__name__}: {exc}"}
                        )
                        self.store.record(
                            "phase_b_launch_outcome_unresolved",
                            action_key=key,
                            host_id=host,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    except Exception as exc:
                        self.store.action_failed(key, f"{type(exc).__name__}: {exc}")
                    else:
                        self.store.action_finished(key, result)
                    continue
                counts_by_host = self.store.state["launch_counts"]["b"]
                if int(counts_by_host[host]) >= self.store.config.max_b_launches_per_host:
                    self.store.halt("phase_b_supervisor_recovery_exhausted", {"host": host, "pending": host_pending})
                    return
                attempt = int(counts_by_host[host]) + 1
                key = f"b_{host}_launch_{attempt:03d}"
                existing = self.store.state["actions"].get(key)
                if existing is not None and existing.get("status") in {
                    "complete",
                    "failed",
                }:
                    existing_payload = existing.get("payload", {})
                    if (
                        existing.get("kind") != "phase_b_host_launch"
                        or existing_payload.get("host_id") != host
                        or int(existing_payload.get("attempt", -1)) != attempt
                    ):
                        self.store.halt(
                            "phase_b_terminal_launch_action_identity_invalid",
                            {"host": host, "action_key": key, "action": existing},
                        )
                        return
                    counts_by_host[host] = max(
                        int(counts_by_host[host]), attempt
                    )
                    self.store.record(
                        "phase_b_terminal_launch_budget_reconciled",
                        host_id=host,
                        action_key=key,
                        attempt=attempt,
                        action_status=existing.get("status"),
                        count=counts_by_host[host],
                    )
                    continue
                payload = {
                    "host_id": host,
                    "attempt": attempt,
                    "launch_nonce": self.phase_b_launch_nonce(host, attempt),
                }
                entry = self.store.begin_action(key, "phase_b_host_launch", payload)
                if entry.get("status") in {"complete", "failed"}:
                    # A prior crash may have committed the terminal action but
                    # not its budget counter.  Count it and advance on the next
                    # tick; never execute a terminal action key twice.
                    counts_by_host[host] = max(
                        int(counts_by_host[host]), attempt
                    )
                    self.store.record(
                        "phase_b_terminal_launch_budget_reconciled",
                        host_id=host,
                        action_key=key,
                        attempt=attempt,
                        action_status=entry.get("status"),
                        count=counts_by_host[host],
                    )
                    continue
                if entry.get("status") != "intent":
                    self.store.halt(
                        "phase_b_launch_action_state_invalid",
                        {"host": host, "action_key": key, "action": entry},
                    )
                    return
                counts_by_host[host] = attempt
                self.store.record("launch_budget_reserved", phase="b", host_id=host, attempt=attempt)
                try:
                    self.store.action_started(key)
                    result = self.ops.launch_b_host(
                        host, str(payload["launch_nonce"])
                    )
                except (RemoteActionUncertainError, subprocess.TimeoutExpired) as exc:
                    action = self.store.state["actions"][key]
                    action["status"] = "started"
                    action.setdefault("unresolved_outcomes", []).append(
                        {"time": now(), "error": f"{type(exc).__name__}: {exc}"}
                    )
                    self.store.record(
                        "phase_b_launch_outcome_unresolved",
                        action_key=key,
                        host_id=host,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception as exc:
                    self.store.action_failed(key, f"{type(exc).__name__}: {exc}")
                else:
                    self.store.action_finished(key, result)

    def tick_verify_b(self) -> None:
        if self.settle_b_launches_for_boundary():
            return
        result = self.run_sync_action("strict_b_60", "strict_checkpoint_verification", {"expected": 60}, self.ops.strict_b)
        if (
            result.get("status") == "complete"
            and result.get("complete") is True
            and result.get("counts", {}).get("accepted") == 60
            and result.get("counts", {}).get("pending") == 0
            and result.get("counts", {}).get("rejected") == 0
        ):
            self.store.transition("quiesce_b", evidence=result)
        else:
            self.store.halt("phase_b_strict_verification_failed", result)

    def tick_quiesce_b(self) -> None:
        if self.settle_b_launches_for_boundary():
            return
        results: dict[str, Any] = {}
        for host in ("local", "remote"):
            results[host] = self.run_sync_action(
                f"b_handoff_{host}",
                "phase_b_guard_and_runtime_handoff",
                {"host_id": host, "stop_watchdog": host == "local"},
                lambda host=host: self.ops.handoff_b_host(host),
            )
        if not all(value.get("status") == "quiescent" and value.get("guard_release_verified") is True for value in results.values()):
            self.store.halt("phase_b_handoff_not_proven", results)
            return
        # A cached handoff result is not evidence about a process generation
        # that appeared later.  Re-observe both hosts after every stop, and
        # bind any residual cleanup to that exact observed generation.
        report = self.ops.observe_b()
        self.store.observe(
            "b_post_handoff",
            {
                "health": report.get("health"),
                "counts": report.get("completion", {}).get("counts"),
                "time": report.get("time"),
            },
        )
        for host in ("local", "remote"):
            activity = ProductionOperations.b_host_activity(report, host)
            if not activity["observable"]:
                return
            if activity["problems"]:
                self.store.halt(
                    "phase_b_post_handoff_identity_invalid",
                    {"host_id": host, **activity},
                )
                return
            if activity["active"]:
                identity = self.immutable_b_runtime_identity(activity)
                fingerprint = canonical_fingerprint(identity)
                key = f"b_handoff_residual_{host}_{fingerprint[:16]}"
                existing = self.store.state["actions"].get(key)
                if existing is not None and existing.get("status") == "complete":
                    self.store.halt(
                        "phase_b_handoff_identity_still_live",
                        {
                            "host_id": host,
                            "runtime_identity_fingerprint": fingerprint,
                        },
                    )
                    return
                self.run_sync_action(
                    key,
                    "phase_b_guard_and_runtime_handoff",
                    {
                        "host_id": host,
                        "stop_watchdog": host == "local",
                        "runtime_identity_fingerprint": fingerprint,
                        "runtime_identity": identity,
                    },
                    lambda host=host: self.ops.handoff_b_host(host),
                )
                return
        self.store.transition(
            "review_b", evidence={"handoff": results, "post_handoff": "clean"}
        )

    def tick_review_b(self) -> None:
        result = self.run_sync_action(
            "phase_b_validation_only_review",
            "validation_only_bracketing_review",
            {"test_data_allowed": False},
            self.ops.review_b,
        )
        if result.get("test_data_accessed") is not False:
            self.store.halt("review_test_data_contract_violated", result)
        elif result.get("status") == "approved_for_phase_c_freeze":
            self.store.transition("c_preview", evidence=result)
        else:
            self.store.transition("manual_review_required", evidence=result)

    def tick_manual_review(self) -> None:
        result = self.ops.review_is_approved()
        self.store.observe("manual_review_gate", self.summary(result))
        if result.get("approved") is True:
            self.store.transition("c_preview", evidence=result)

    def tick_preview(self, phase: str) -> None:
        if phase == "d":
            adoption = self.run_sync_action(
                "d_prefetched_formal_adoption",
                "adopt_prefetched_formal_phase",
                {"phase": phase, "expected_rows": PHASE_EXPECTED[phase]},
                lambda: self.ops.adopt_existing_formal_phase(phase),
            )
            if adoption.get("status") == "invalid":
                self.store.halt("phase_d_prefetched_formal_invalid", adoption)
                return
            if adoption.get("status") == "adoptable":
                observation = adoption.get("observation", {})
                structural = observation.get("structural_verification", {})
                counts = structural.get("counts", {})
                evidence = {
                    "prefetched": True,
                    "path": adoption.get("path"),
                    "generation_id": adoption.get("generation_id"),
                    "hosts": adoption.get("hosts"),
                    "observation": self.summary(observation),
                }
                if (
                    int(counts.get("accepted", 0)) == PHASE_EXPECTED[phase]
                    and int(counts.get("pending", -1)) == 0
                    and int(counts.get("rejected", -1)) == 0
                ):
                    if self.phase_runtime_present(observation):
                        self.store.transition("d_run", evidence=evidence)
                    elif observation.get("status") == "ready_for_strict_verification":
                        self.store.transition("d_verify", evidence=evidence)
                    else:
                        self.store.halt(
                            "phase_d_prefetched_completion_not_quiescent",
                            evidence,
                        )
                elif self.phase_runtime_present(observation) or any(
                    int(counts.get(key, 0)) for key in ("accepted", "pending")
                ):
                    self.store.transition("d_run", evidence=evidence)
                else:
                    self.store.transition("d_launch", evidence=evidence)
                return
        result = self.run_sync_action(
            f"{phase}_design_preview",
            "build_design_preview",
            {"phase": phase, "runnable": False},
            lambda: self.ops.build_preview(phase),
        )
        if result.get("status") == "preview_ready":
            self.store.transition(f"{phase}_smoke", evidence=result)
        else:
            self.store.halt(f"phase_{phase}_preview_invalid", result)

    def successful_smoke_fragments(self, phase: str) -> dict[str, Path]:
        fragments: dict[str, Path] = {}
        for host in ("local", "remote"):
            for attempt in reversed(self.store.state["smoke_attempts"][phase][host]):
                if attempt.get("status") == "complete":
                    fragments[host] = canonical(attempt["fragment"])
                    break
        return fragments

    def reconcile_smoke_host(self, phase: str, host: str) -> str:
        attempts = self.store.state["smoke_attempts"][phase][host]
        if attempts and attempts[-1].get("status") not in {"complete", "failed"}:
            current = attempts[-1]
            probe = self.ops.probe_job(current["job"])
            current["last_probe"] = self.summary(probe) | {"state": probe.get("state")}
            if probe.get("state") == "observation_unavailable":
                self.store.record(
                    "smoke_job_observation_unavailable",
                    phase=phase,
                    host_id=host,
                    attempt=current["attempt"],
                    probe=probe,
                )
                return "observation_unavailable"
            if probe.get("state") in {
                "running",
                "starting",
                "orphan_session_running",
                "terminal_result_session_running",
            }:
                if (
                    probe.get("state") == "orphan_session_running"
                    and not isinstance(current.get("first_orphan_observed_at"), str)
                ):
                    current["first_orphan_observed_at"] = now()
                    self.store.record(
                        "smoke_orphan_first_observed",
                        phase=phase,
                        host_id=host,
                        attempt=current["attempt"],
                        first_orphan_observed_at=current["first_orphan_observed_at"],
                    )
                expired, age = self.detached_job_expired(
                    current["job"],
                    probe,
                    action_started_at=current.get("started_at"),
                    first_orphan_observed_at=current.get("first_orphan_observed_at"),
                )
                current["status"] = "running"
                self.store.record(
                    "smoke_detached_job_observed",
                    phase=phase,
                    host_id=host,
                    attempt=current["attempt"],
                    state=probe.get("state"),
                    age_seconds=age,
                    expired=expired,
                )
                if not expired:
                    return "running"
                cleanup = self.run_sync_action(
                    f"smoke_{phase}_{host}_{int(current['attempt']):03d}_orphan_cleanup",
                    "identity_safe_detached_job_cleanup",
                    {"phase": phase, "host_id": host, "job": current["job"]},
                    lambda: self.ops.cleanup_job(current["job"]),
                )
                post = self.ops.probe_job(current["job"])
                if post.get("state") in {"complete", "orphan_completed_unwrapped"}:
                    validation = self.ops.validate_smoke_fragment(
                        phase, host, canonical(current["fragment"])
                    )
                    current["status"] = "complete"
                    current["validation"] = validation
                    self.store.record(
                        "smoke_host_complete_after_orphan_cleanup",
                        phase=phase,
                        host_id=host,
                        attempt=current["attempt"],
                        cleanup=cleanup,
                    )
                    return "complete"
                current["status"] = "failed"
                current["failure"] = {
                    "reason": "orphan_timeout",
                    "age_seconds": age,
                    "cleanup": cleanup,
                    "post_cleanup_probe": post,
                }
                self.store.record(
                    "smoke_orphan_cleaned_and_failed",
                    phase=phase,
                    host_id=host,
                    attempt=current["attempt"],
                    failure=current["failure"],
                )
                return "failed"
            if probe.get("state") in {"complete", "orphan_completed_unwrapped"}:
                validation = self.ops.validate_smoke_fragment(phase, host, canonical(current["fragment"]))
                current["status"] = "complete"
                current["validation"] = validation
                self.store.record("smoke_host_complete", phase=phase, host_id=host, attempt=current["attempt"], validation=validation)
                return "complete"
            if probe.get("state") == "invalid":
                self.store.halt(f"phase_{phase}_smoke_job_identity_invalid", {"host": host, "probe": probe})
                return "halted"
            if probe.get("state") in {"failed", "missing"}:
                if probe.get("state") == "missing" and current.get("status") == "intent":
                    current["status"] = "started"
                    current["started_at"] = now()
                    self.store.record(
                        "smoke_host_launch_started",
                        phase=phase,
                        host_id=host,
                        attempt=current["attempt"],
                        recovered_intent=True,
                    )
                    try:
                        probe = self.ops.launch_job(current["job"])
                    except Exception as exc:
                        current["launch_exception_unresolved"] = {
                            "error": f"{type(exc).__name__}: {exc}",
                            "observed_at": now(),
                        }
                        self.store.record(
                            "smoke_host_launch_result_unresolved",
                            phase=phase,
                            host_id=host,
                            attempt=current["attempt"],
                            error=current["launch_exception_unresolved"]["error"],
                        )
                        return "observation_unavailable"
                    current["launch_probe"] = self.summary(probe) | {
                        "state": probe.get("state")
                    }
                    self.store.record(
                        "smoke_host_intent_relaunched",
                        phase=phase,
                        host_id=host,
                        attempt=current["attempt"],
                        state=probe.get("state"),
                    )
                    return "running"
                if probe.get("state") == "missing" and current.get("status") in {
                    "started",
                    "running",
                }:
                    age = recorded_age_seconds(current.get("started_at"))
                    publication_timeout = float(
                        current["job"].get("identity_publication_timeout_seconds", 60.0)
                    )
                    if publication_timeout <= 0:
                        raise ValueError("invalid detached identity publication timeout")
                    if age < publication_timeout:
                        self.store.record(
                            "smoke_job_identity_publication_pending",
                            phase=phase,
                            host_id=host,
                            attempt=current["attempt"],
                            age_seconds=age,
                        )
                        return "running"
                    cleanup = self.run_sync_action(
                        f"smoke_{phase}_{host}_{int(current['attempt']):03d}_publication_cleanup",
                        "identity_safe_detached_job_cleanup",
                        {
                            "phase": phase,
                            "host_id": host,
                            "reason": "identity_publication_timeout",
                            "job": current["job"],
                        },
                        lambda: self.ops.cleanup_job(current["job"]),
                    )
                    post = self.ops.probe_job(current["job"])
                    current["post_publication_cleanup_probe"] = post
                    self.store.record(
                        "smoke_identity_publication_cleanup_complete",
                        phase=phase,
                        host_id=host,
                        attempt=current["attempt"],
                        cleanup=cleanup,
                        post_probe=post,
                    )
                    if post.get("state") in {
                        "running",
                        "starting",
                        "orphan_session_running",
                        "terminal_result_session_running",
                        "observation_unavailable",
                    }:
                        return "running"
                    if post.get("state") in {
                        "complete",
                        "orphan_completed_unwrapped",
                    }:
                        validation = self.ops.validate_smoke_fragment(
                            phase, host, canonical(current["fragment"])
                        )
                        current["status"] = "complete"
                        current["validation"] = validation
                        return "complete"
                    if post.get("state") == "invalid":
                        self.store.halt(
                            f"phase_{phase}_smoke_job_identity_invalid",
                            {"host": host, "probe": post},
                        )
                        return "halted"
                    probe = post
                current["status"] = "failed"
                current["failure"] = probe
                self.store.record("smoke_host_failed", phase=phase, host_id=host, attempt=current["attempt"], probe=probe)
        if attempts and attempts[-1].get("status") == "complete":
            return "complete"
        if len(attempts) >= self.store.config.max_smoke_attempts_per_host:
            self.store.halt(f"phase_{phase}_smoke_attempts_exhausted", {"host": host, "attempts": attempts})
            return "halted"
        idle = self.ops.idle_probe(host)
        if idle.get("scaling_processes") or idle.get("compute_processes"):
            self.store.observe(f"{phase}_smoke_idle_{host}", {"status": "busy", "scaling": len(idle.get("scaling_processes", [])), "compute": len(idle.get("compute_processes", []))})
            return "waiting_for_idle"
        attempt_number = len(attempts) + 1
        spec = self.ops.smoke_job_spec(phase, host, attempt_number)
        entry = {
            "attempt": attempt_number,
            "status": "intent",
            "fragment": spec["fragment"],
            "output_root": spec["output_root"],
            "job": spec,
            "intent_at": now(),
        }
        attempts.append(entry)
        self.store.record("smoke_host_intent", phase=phase, host_id=host, attempt=attempt_number, job=spec)
        entry["status"] = "started"
        entry["started_at"] = now()
        self.store.record(
            "smoke_host_launch_started",
            phase=phase,
            host_id=host,
            attempt=attempt_number,
            recovered_intent=False,
        )
        try:
            probe = self.ops.launch_job(spec)
        except Exception as exc:
            # The launch command may have crossed the local/SSH boundary even
            # when its caller saw an exception.  Preserve the started attempt
            # and reconcile its exact receipt instead of risking a duplicate.
            entry["status"] = "started"
            entry["launch_exception_unresolved"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "observed_at": now(),
            }
            self.store.record(
                "smoke_host_launch_result_unresolved",
                phase=phase,
                host_id=host,
                attempt=attempt_number,
                error=entry["launch_exception_unresolved"]["error"],
            )
            return "observation_unavailable"
        if probe.get("state") in {"complete", "orphan_completed_unwrapped"}:
            validation = self.ops.validate_smoke_fragment(
                phase, host, canonical(entry["fragment"])
            )
            entry["status"] = "complete"
            entry["validation"] = validation
        elif probe.get("state") in {
            "running",
            "starting",
            "orphan_session_running",
            "terminal_result_session_running",
        }:
            entry["status"] = "running"
        elif probe.get("state") in {"observation_unavailable", "missing"}:
            entry["status"] = "started"
        elif probe.get("state") == "invalid":
            self.store.halt(
                f"phase_{phase}_smoke_job_identity_invalid",
                {"host": host, "probe": probe},
            )
            return "halted"
        else:
            entry["status"] = "failed"
            entry["failure"] = probe
        entry["launch_probe"] = self.summary(probe) | {"state": probe.get("state")}
        self.store.record("smoke_host_launched", phase=phase, host_id=host, attempt=attempt_number, state=probe.get("state"))
        return entry["status"]

    def tick_smoke(self, phase: str) -> None:
        statuses: dict[str, str] = {}
        for host in ("local", "remote"):
            statuses[host] = self.reconcile_smoke_host(phase, host)
            if self.store.state["step"] == "halted":
                return
        fragments = self.successful_smoke_fragments(phase)
        if set(fragments) != {"local", "remote"}:
            return
        result = self.run_sync_action(
            f"{phase}_smoke_merge",
            "merge_profile_smoke",
            {"phase": phase, "fragments": {key: str(value) for key, value in fragments.items()}},
            lambda: self.ops.merge_smoke(phase, fragments),
        )
        if result.get("status") == "approved_for_formal_freeze":
            self.store.transition(f"{phase}_formalize", evidence=result)
        else:
            self.store.halt(f"phase_{phase}_smoke_merge_not_approved", result)

    def tick_formalize(self, phase: str) -> None:
        result = self.run_sync_action(
            f"{phase}_formal_matrix",
            "immutable_formal_matrix_freeze",
            {"phase": phase, "expected_rows": PHASE_EXPECTED[phase]},
            lambda: self.ops.formalize(phase),
        )
        if result.get("status") == "formal_matrix_ready":
            self.store.transition(f"{phase}_launch", evidence=result)
        else:
            self.store.halt(f"phase_{phase}_formal_matrix_invalid", result)

    def active_launch_jobs(self, phase: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        values: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for key, action in self.store.state["actions"].items():
            if action.get("kind") not in {"two_host_phase_launch", "host_phase_recovery_launch"}:
                continue
            payload = action.get("payload", {})
            if payload.get("phase") != phase or action.get("status") not in {
                "intent",
                "started",
            }:
                continue
            spec = payload.get("job")
            if isinstance(spec, dict):
                values.append((key, action, spec))
        return values

    def reconcile_launch_jobs(self, phase: str) -> str:
        state = "none"
        for key, action, spec in self.active_launch_jobs(phase):
            probe = self.ops.probe_job(spec)
            if probe.get("state") == "observation_unavailable":
                action["last_probe"] = probe
                self.store.record(
                    "phase_launch_job_observation_unavailable",
                    action_key=key,
                    probe=probe,
                )
                state = "running"
            elif probe.get("state") in {
                "running",
                "starting",
                "orphan_session_running",
                "terminal_result_session_running",
            }:
                if (
                    probe.get("state") == "orphan_session_running"
                    and not isinstance(action.get("first_orphan_observed_at"), str)
                ):
                    action["first_orphan_observed_at"] = now()
                    self.store.record(
                        "phase_launch_orphan_first_observed",
                        action_key=key,
                        first_orphan_observed_at=action["first_orphan_observed_at"],
                    )
                expired, age = self.detached_job_expired(
                    spec,
                    probe,
                    action_started_at=action.get("started_at"),
                    first_orphan_observed_at=action.get("first_orphan_observed_at"),
                )
                action["status"] = "started"
                action["last_probe"] = {"state": probe.get("state"), "age_seconds": age}
                self.store.record(
                    "phase_launch_detached_job_observed",
                    action_key=key,
                    state=probe.get("state"),
                    age_seconds=age,
                    expired=expired,
                )
                if not expired:
                    state = "running"
                    continue
                cleanup = self.run_sync_action(
                    f"{key}_orphan_cleanup",
                    "identity_safe_detached_job_cleanup",
                    {"phase": phase, "job": spec},
                    lambda spec=spec: self.ops.cleanup_job(spec),
                )
                post = self.ops.probe_job(spec)
                if post.get("state") == "complete":
                    self.store.action_finished(
                        key,
                        {
                            "status": "launch_command_complete",
                            "probe": self.summary(post) | {"state": "complete"},
                            "cleanup": cleanup,
                        },
                    )
                    state = "complete"
                elif post.get("state") == "observation_unavailable":
                    action["last_probe"] = post
                    self.store.record(
                        "phase_launch_post_cleanup_observation_unavailable",
                        action_key=key,
                        probe=post,
                    )
                    state = "running"
                else:
                    self.store.action_failed(
                        key,
                        json.dumps(
                            {
                                "reason": "orphan_timeout",
                                "age_seconds": age,
                                "cleanup": cleanup,
                                "post_cleanup_probe": post,
                            },
                            ensure_ascii=False,
                        ),
                    )
                    state = "failed"
            elif probe.get("state") == "complete":
                self.store.action_finished(key, {"status": "launch_command_complete", "probe": self.summary(probe) | {"state": "complete"}})
                state = "complete"
            elif probe.get("state") == "invalid":
                self.store.halt(f"phase_{phase}_launch_job_identity_invalid", {"action": key, "probe": probe})
                return "halted"
            elif probe.get("state") == "missing" and action.get("status") == "intent":
                attempt = int(action.get("payload", {}).get("attempt", 0))
                hosts = tuple(str(item) for item in action.get("payload", {}).get("hosts", []))
                counts = self.store.state["launch_counts"][phase]
                for host in hosts:
                    counts[host] = max(int(counts[host]), attempt)
                self.store.record(
                    "launch_budget_reconciled_from_intent",
                    phase=phase,
                    action_key=key,
                    hosts=list(hosts),
                    counts=dict(counts),
                )
                self.store.action_started(key, recovered_intent=True)
                try:
                    launched = self.ops.launch_job(spec)
                except Exception as exc:
                    action["launch_exception_unresolved"] = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "observed_at": now(),
                    }
                    self.store.record(
                        "phase_launch_result_unresolved",
                        action_key=key,
                        error=action["launch_exception_unresolved"]["error"],
                    )
                    state = "running"
                    continue
                action["last_probe"] = self.summary(launched) | {
                    "state": launched.get("state")
                }
                self.store.record(
                    "phase_launch_intent_relaunched",
                    action_key=key,
                    state=launched.get("state"),
                )
                state = "running"
            elif probe.get("state") == "missing" and action.get("status") == "started":
                age = recorded_age_seconds(action.get("started_at"))
                publication_timeout = float(
                    spec.get("identity_publication_timeout_seconds", 60.0)
                )
                if publication_timeout <= 0:
                    raise ValueError("invalid detached identity publication timeout")
                if age < publication_timeout:
                    action["last_probe"] = {"state": "missing", "age_seconds": age}
                    self.store.record(
                        "phase_launch_identity_publication_pending",
                        action_key=key,
                        age_seconds=age,
                    )
                    state = "running"
                else:
                    cleanup = self.run_sync_action(
                        f"{key}_publication_cleanup",
                        "identity_safe_detached_job_cleanup",
                        {
                            "phase": phase,
                            "reason": "identity_publication_timeout",
                            "job": spec,
                        },
                        lambda spec=spec: self.ops.cleanup_job(spec),
                    )
                    post = self.ops.probe_job(spec)
                    action["post_publication_cleanup_probe"] = post
                    self.store.record(
                        "phase_launch_identity_publication_cleanup_complete",
                        action_key=key,
                        cleanup=cleanup,
                        post_probe=post,
                    )
                    if post.get("state") in {
                        "running",
                        "starting",
                        "orphan_session_running",
                        "terminal_result_session_running",
                        "observation_unavailable",
                    }:
                        state = "running"
                    elif post.get("state") == "complete":
                        self.store.action_finished(
                            key,
                            {
                                "status": "launch_command_complete",
                                "probe": self.summary(post) | {"state": "complete"},
                                "publication_cleanup": cleanup,
                            },
                        )
                        state = "complete"
                    elif post.get("state") == "invalid":
                        self.store.halt(
                            f"phase_{phase}_launch_job_identity_invalid",
                            {"action": key, "probe": post},
                        )
                        return "halted"
                    else:
                        self.store.action_failed(
                            key, json.dumps(post, ensure_ascii=False)
                        )
                        state = "failed"
            else:
                self.store.action_failed(key, json.dumps(probe, ensure_ascii=False))
                state = "failed"
        # Never let a later failed/completed action hide an earlier live or
        # observation-ambiguous launcher in the same reconciliation pass.
        if self.active_launch_jobs(phase):
            return "running"
        return state

    def cleanup_failed_launch_actions(self, phase: str) -> bool:
        pending = [
            (key, action)
            for key, action in list(self.store.state["actions"].items())
            if action.get("kind")
            in {"two_host_phase_launch", "host_phase_recovery_launch"}
            and action.get("payload", {}).get("phase") == phase
            and action.get("status") == "failed"
            and action.get("runtime_cleanup_status") != "complete"
        ]
        if not pending:
            return False
        for key, action in pending:
            hosts = tuple(
                str(item) for item in action.get("payload", {}).get("hosts", [])
            )
            if not hosts or not set(hosts) <= {"local", "remote"}:
                raise ValueError(f"failed launch action has invalid host scope: {key}")
            cleanup: dict[str, Any] = {}
            for host in hosts:
                cleanup[host] = self.run_sync_action(
                    f"{key}_runtime_cleanup_{host}",
                    "identity_safe_phase_cleanup",
                    {"phase": phase, "host_id": host, "failed_launch_action": key},
                    lambda phase=phase, host=host, action_key=f"{key}_runtime_cleanup_{host}": self.ops.stop_phase_host(
                        phase, host, action_key
                    ),
                )
            action["runtime_cleanup_status"] = "complete"
            action["runtime_cleanup"] = cleanup
            self.store.record(
                "failed_phase_launch_runtime_cleanup_complete",
                phase=phase,
                action_key=key,
                cleanup=cleanup,
            )
        return True

    def reconcile_started_phase_cleanups(self, phase: str) -> bool:
        """Finish every previously journaled host stop before using a new snapshot.

        A remote stop can have an unknown SSH outcome.  In that case its action
        remains ``started``.  Even if the following watchdog snapshot happens
        to look idle, launching from it would race a delayed copy of the old
        stop command.  Re-run the identity-safe/idempotent stop under the same
        action key and force a fresh controller tick after it is proven.
        """
        pending = [
            (key, action)
            for key, action in list(self.store.state["actions"].items())
            if action.get("kind") == "identity_safe_phase_cleanup"
            and action.get("payload", {}).get("phase") == phase
            and action.get("status") in {"intent", "started"}
        ]
        if not pending:
            return False
        for key, action in pending:
            payload = action.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"phase cleanup action has invalid payload: {key}")
            host = str(payload.get("host_id"))
            if host not in {"local", "remote"}:
                raise ValueError(f"phase cleanup action has invalid host: {key}")
            self.run_sync_action(
                key,
                "identity_safe_phase_cleanup",
                dict(payload),
                lambda phase=phase, host=host, action_key=key: self.ops.stop_phase_host(
                    phase, host, action_key
                ),
            )
        # Never act on the observation that preceded a stop, even when all
        # stops completed synchronously.  The next tick must observe afresh.
        return True

    @staticmethod
    def phase_runtime_present(observation: dict[str, Any]) -> bool:
        return any(
            sum(int(value) for value in host.get("role_counts", {}).values()) > 0
            for host in observation.get("hosts", {}).values()
        )

    @staticmethod
    def phase_hosts_idle_clean(
        observation: dict[str, Any], expected_hosts: tuple[str, ...]
    ) -> bool:
        if (
            set(observation.get("hosts", {})) != set(expected_hosts)
            or not expected_hosts
            or observation.get("host_probe_errors")
        ):
            return False
        for host in observation["hosts"].values():
            if (
                host.get("issues")
                or host.get("runtime_admission_disabled")
                or any(int(value) for value in host.get("role_counts", {}).values())
                or host.get("selected_gpu_compute")
                or host.get("gpu_query_errors")
                or host.get("other_scaling_processes")
            ):
                return False
        return True

    @staticmethod
    def remote_partial_admission_ready(phase: str, host_id: str, snapshot: dict[str, Any]) -> bool:
        """Admit the unchanged remote matrix only through its deferred supervisor."""
        return bool(
            phase == "c" and host_id == "remote"
            and snapshot.get("deferred_admission_supported") is True
            and (snapshot.get("admission_ready_gpus") or snapshot.get("local_execution_admission_supported") is True)
            and not snapshot.get("issues")
            and not snapshot.get("runtime_admission_disabled")
            and not snapshot.get("other_scaling_processes")
            and not any(int(value) for value in snapshot.get("role_counts", {}).values())
        )

    def launch_phase_job(self, phase: str, hosts: tuple[str, ...]) -> None:
        counts = self.store.state["launch_counts"][phase]
        if any(int(counts[host]) >= self.store.config.max_phase_launches_per_host for host in hosts):
            self.store.halt(f"phase_{phase}_supervisor_launch_budget_exhausted", {"hosts": hosts, "counts": counts})
            return
        attempt = max(int(counts[host]) for host in hosts) + 1
        spec = self.ops.phase_launch_job_spec(phase, hosts, attempt)
        kind = "two_host_phase_launch" if len(hosts) == 2 else "host_phase_recovery_launch"
        key = f"{phase}_{'_'.join(hosts)}_launch_{attempt:03d}"
        payload = {"phase": phase, "hosts": list(hosts), "attempt": attempt, "job": spec}
        self.store.begin_action(key, kind, payload)
        for host in hosts:
            counts[host] = int(counts[host]) + 1
        self.store.record("launch_budget_reserved", phase=phase, hosts=list(hosts), counts=dict(counts))
        self.store.action_started(key)
        try:
            probe = self.ops.launch_job(spec)
        except Exception as exc:
            action = self.store.state["actions"][key]
            action["launch_exception_unresolved"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "observed_at": now(),
            }
            self.store.record(
                "phase_launch_result_unresolved",
                action_key=key,
                error=action["launch_exception_unresolved"]["error"],
            )
            return
        if probe.get("state") in {"complete"}:
            self.store.action_finished(key, {"status": "launch_command_complete", "probe": self.summary(probe) | {"state": "complete"}})
        elif probe.get("state") == "invalid":
            self.store.halt(
                f"phase_{phase}_launch_job_identity_invalid",
                {"action": key, "probe": probe},
            )
        elif probe.get("state") == "failed":
            self.store.action_failed(key, json.dumps(probe, ensure_ascii=False))
        else:
            self.store.state["actions"][key]["last_probe"] = {"state": probe.get("state")}
            self.store.record("phase_launch_job_started", action_key=key, state=probe.get("state"))

    def tick_launch(self, phase: str) -> None:
        if self.reconcile_started_phase_cleanups(phase):
            return
        if self.cleanup_failed_launch_actions(phase):
            return
        launch_state = self.reconcile_launch_jobs(phase)
        if self.store.state["step"] == "halted" or launch_state == "running":
            return
        if launch_state == "failed":
            self.cleanup_failed_launch_actions(phase)
            return
        observation = self.ops.observe_phase(phase)
        self.store.observe(f"{phase}_launch_observation", self.summary(observation))
        if observation.get("status") == "invalid":
            self.store.halt(f"phase_{phase}_prelaunch_invalid", observation.get("alerts"))
            return
        if observation.get("host_probe_errors"):
            return
        phase_hosts = self.ops.phase_hosts(phase)
        host_snapshots = observation.get("hosts", {})
        if not phase_hosts or set(host_snapshots) != set(phase_hosts):
            self.store.observe(
                f"{phase}_prelaunch_idle_gate",
                {"status": "incomplete_host_set", "expected_hosts": list(phase_hosts),
                 "observed_hosts": sorted(host_snapshots)},
            )
            return
        if self.phase_runtime_present(observation) or any(
            int(observation.get("structural_verification", {}).get("counts", {}).get(key, 0))
            for key in ("accepted", "rejected")
        ):
            self.store.transition(f"{phase}_run", evidence=self.summary(observation))
            return
        if launch_state == "complete":
            self.store.transition(f"{phase}_run", evidence={"launch_job": "complete"})
            return
        # The whole immutable matrix and every expected host were observed
        # above.  Admit only clean hosts now; tick_run already waits for foreign
        # compute on the remaining host and launches it when that host is idle.
        ready_hosts = tuple(
            host for host in phase_hosts
            if self.phase_hosts_idle_clean(
                {**observation, "hosts": {host: host_snapshots[host]}}, (host,)
            )
        )
        if not ready_hosts:
            partial_hosts = tuple(
                host for host in phase_hosts
                if self.remote_partial_admission_ready(phase, host, host_snapshots[host])
            )
            if partial_hosts:
                self.launch_phase_job(phase, partial_hosts)
                return
            self.store.observe(f"{phase}_prelaunch_idle_gate", {"status": "not_idle_or_not_clean"})
            return
        self.launch_phase_job(phase, ready_hosts)

    def host_needs_cleanup(self, snapshot: dict[str, Any]) -> bool:
        roles = snapshot.get("role_counts", {})
        return bool(snapshot.get("issues")) and any(int(value) for value in roles.values())

    @staticmethod
    def phase_runtime_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
        processes = snapshot.get("processes", {})
        normalized: dict[str, list[dict[str, Any]]] = {}
        if isinstance(processes, dict):
            for role in ("supervisors", "workers", "guards", "trainers"):
                rows = processes.get(role, [])
                if not isinstance(rows, list):
                    raise ValueError(f"runtime process list is malformed: {role}")
                normalized[role] = sorted(
                    [
                        {
                            key: item.get(key)
                            for key in (
                                "pid",
                                "ppid",
                                "pgid",
                                "sid",
                                "start_ticks",
                                "worker_slot",
                                "run_id",
                            )
                            if key in item
                        }
                        for item in rows
                        if isinstance(item, dict)
                    ],
                    key=lambda item: (
                        str(item.get("start_ticks")), str(item.get("pid"))
                    ),
                )
        identity = {
            "processes": normalized,
            "role_counts": {
                str(key): int(value)
                for key, value in snapshot.get("role_counts", {}).items()
            },
            "supervisor_pid_record": snapshot.get("supervisor_pid_record"),
        }
        if not any(identity["role_counts"].values()):
            raise ValueError("runtime cleanup requested without an owned runtime")
        return identity

    def cleanup_phase_runtime_snapshot(
        self,
        phase: str,
        host_id: str,
        snapshot: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        identity = self.phase_runtime_identity(snapshot)
        fingerprint = canonical_fingerprint(identity)
        prior = [
            (key, action)
            for key, action in self.store.state["actions"].items()
            if action.get("kind") == "identity_safe_phase_cleanup"
            and action.get("payload", {}).get("phase") == phase
            and action.get("payload", {}).get("host_id") == host_id
            and action.get("payload", {}).get("runtime_identity_fingerprint")
            == fingerprint
        ]
        if any(action.get("status") == "complete" for _, action in prior):
            self.store.halt(
                f"phase_{phase}_cleanup_identity_still_live",
                {
                    "host_id": host_id,
                    "runtime_identity_fingerprint": fingerprint,
                    "prior_actions": [key for key, _ in prior],
                },
            )
            return
        unique_cleanups = {
            action.get("payload", {}).get("runtime_identity_fingerprint")
            for action in self.store.state["actions"].values()
            if action.get("kind") == "identity_safe_phase_cleanup"
            and action.get("payload", {}).get("phase") == phase
            and action.get("payload", {}).get("host_id") == host_id
            and action.get("payload", {}).get("runtime_identity_fingerprint")
        }
        if len(unique_cleanups) >= self.store.config.max_phase_launches_per_host + 2:
            self.store.halt(
                f"phase_{phase}_runtime_cleanup_budget_exhausted",
                {"host_id": host_id, "runtime_identities": sorted(unique_cleanups)},
            )
            return
        key = f"{phase}_{host_id}_runtime_cleanup_{fingerprint[:16]}"
        self.run_sync_action(
            key,
            "identity_safe_phase_cleanup",
            {
                "phase": phase,
                "host_id": host_id,
                "reason": reason,
                "runtime_identity_fingerprint": fingerprint,
                "runtime_identity": identity,
            },
            lambda phase=phase, host_id=host_id, action_key=key: self.ops.stop_phase_host(
                phase, host_id, action_key
            ),
        )

    def tick_run(self, phase: str) -> None:
        if self.reconcile_started_phase_cleanups(phase):
            return
        if self.cleanup_failed_launch_actions(phase):
            return
        had_active_launch = bool(self.active_launch_jobs(phase))
        launch_state = self.reconcile_launch_jobs(phase)
        if self.store.state["step"] == "halted" or launch_state == "running":
            return
        # Reconciliation can complete/fail a launcher which created or rolled
        # back runtime after the prior tick.  Always take the runtime/artifact
        # snapshot in a later tick, never in the same side-effect turn.
        if had_active_launch:
            return
        if launch_state == "failed":
            self.cleanup_failed_launch_actions(phase)
            return
        observation = self.ops.observe_phase(phase)
        self.store.observe(f"{phase}_live", self.summary(observation))
        structural = observation.get("structural_verification", {})
        attempts_exhausted = {
            host_id: [
                str(item.get("run_id"))
                for item in snapshot.get("retry_states", [])
                if item.get("status") == "attempts_exhausted"
            ]
            for host_id, snapshot in observation.get("hosts", {}).items()
        }
        attempts_exhausted = {
            host_id: run_ids
            for host_id, run_ids in attempts_exhausted.items()
            if run_ids
        }
        if attempts_exhausted:
            # This is a permanent run-level budget, not a missing-supervisor
            # condition.  A host relaunch cannot legally create another try.
            self.store.halt(
                f"phase_{phase}_run_attempts_exhausted", attempts_exhausted
            )
            return
        if structural.get("matrix_issues") or int(structural.get("counts", {}).get("rejected", 0)):
            self.store.halt(f"phase_{phase}_artifact_invalid", structural)
            return
        if observation.get("host_probe_errors"):
            return
        if observation.get("status") == "ready_for_strict_verification":
            self.store.transition(f"{phase}_verify", evidence=self.summary(observation))
            return
        if observation.get("status") == "invalid":
            # Exact owned orphan groups can be cleaned by the hardened host
            # launcher.  Artifact/retry exhaustion was handled above.
            cleaned = False
            for host_id, snapshot in observation.get("hosts", {}).items():
                if self.host_needs_cleanup(snapshot):
                    self.cleanup_phase_runtime_snapshot(
                        phase,
                        host_id,
                        snapshot,
                        reason="invalid_runtime_observation",
                    )
                    if self.store.state["step"] == "halted":
                        return
                    cleaned = True
            if not cleaned:
                self.store.halt(f"phase_{phase}_runtime_invalid", observation.get("alerts"))
            return
        counts = phase_host_counts(observation, canonical(self.store.config.followup_root) / f"phase_{phase}/matrix.json")
        for host_id, host_counts in counts.items():
            if host_counts["rejected"]:
                self.store.halt(f"phase_{phase}_{host_id}_artifact_rejected", host_counts)
                return
            if not host_counts["pending"]:
                continue
            snapshot = observation.get("hosts", {}).get(host_id, {})
            if snapshot.get("runtime_admission_disabled"):
                continue
            active = any(int(value) for value in snapshot.get("role_counts", {}).values())
            if active:
                continue
            partial_ready = self.remote_partial_admission_ready(phase, host_id, snapshot)
            if not partial_ready and (
                snapshot.get("selected_gpu_compute") or snapshot.get("other_scaling_processes")
                or snapshot.get("gpu_query_errors")
            ):
                continue
            self.launch_phase_job(phase, (host_id,))
            if self.store.state["step"] == "halted":
                return

    def tick_verify_phase(self, phase: str) -> None:
        if self.reconcile_started_phase_cleanups(phase):
            return
        if self.cleanup_failed_launch_actions(phase):
            return
        had_active_launch = bool(self.active_launch_jobs(phase))
        launch_state = self.reconcile_launch_jobs(phase)
        if self.store.state["step"] == "halted" or launch_state == "running":
            return
        if had_active_launch:
            return
        if launch_state == "failed":
            self.cleanup_failed_launch_actions(phase)
            return
        result = self.run_sync_action(
            f"strict_{phase}_{PHASE_EXPECTED[phase]}",
            "strict_checkpoint_verification",
            {"phase": phase, "expected": PHASE_EXPECTED[phase]},
            lambda: self.ops.strict_phase(phase),
        )
        counts = result.get("counts", {})
        if (
            result.get("status") == "complete"
            and result.get("complete") is True
            and int(counts.get("accepted", -1)) == PHASE_EXPECTED[phase]
            and int(counts.get("pending", -1)) == 0
            and int(counts.get("rejected", -1)) == 0
        ):
            self.store.transition(f"{phase}_quiesce", evidence=result)
        else:
            self.store.halt(f"phase_{phase}_strict_verification_failed", result)

    def tick_quiesce_phase(self, phase: str) -> None:
        if self.reconcile_started_phase_cleanups(phase):
            return
        if self.cleanup_failed_launch_actions(phase):
            return
        had_active_launch = bool(self.active_launch_jobs(phase))
        launch_state = self.reconcile_launch_jobs(phase)
        if self.store.state["step"] == "halted" or launch_state == "running":
            return
        if had_active_launch:
            return
        if launch_state == "failed":
            self.cleanup_failed_launch_actions(phase)
            return
        for host in self.ops.phase_hosts(phase):
            self.run_sync_action(
                f"{phase}_final_stop_{host}",
                "identity_safe_phase_cleanup",
                {"phase": phase, "host_id": host, "after_strict": True},
                lambda phase=phase, host=host, action_key=f"{phase}_final_stop_{host}": self.ops.stop_phase_host(
                    phase, host, action_key
                ),
            )
        observation = self.ops.observe_phase(phase)
        self.store.observe(f"{phase}_quiescence", self.summary(observation))
        if observation.get("host_probe_errors"):
            # The stop outcome may be complete, but absence of the remote host
            # is not proof of quiescence.  Keep the state and observe again.
            return
        structural = observation.get("structural_verification", {})
        counts = structural.get("counts", {})
        if (
            structural.get("matrix_issues")
            or int(counts.get("accepted", -1)) != PHASE_EXPECTED[phase]
            or int(counts.get("pending", -1)) != 0
            or int(counts.get("rejected", -1)) != 0
        ):
            self.store.halt(
                f"phase_{phase}_post_strict_artifact_drift",
                self.summary(observation) | {"structural_verification": structural},
            )
            return
        residual_cleaned = False
        for host_id, snapshot in observation.get("hosts", {}).items():
            if any(
                int(value)
                for value in snapshot.get("role_counts", {}).values()
            ):
                self.cleanup_phase_runtime_snapshot(
                    phase,
                    host_id,
                    snapshot,
                    reason="post_strict_residual_runtime",
                )
                if self.store.state["step"] == "halted":
                    return
                residual_cleaned = True
        if residual_cleaned:
            return
        if observation.get("status") != "ready_for_strict_verification":
            self.store.halt(
                f"phase_{phase}_post_stop_status_invalid",
                self.summary(observation),
            )
            return
        if not self.phase_hosts_idle_clean(
            observation, self.ops.phase_hosts(phase)
        ):
            self.store.halt(
                f"phase_{phase}_post_stop_not_clean",
                self.summary(observation),
            )
            return
        self.store.transition("d_preview" if phase == "c" else "union_verify", evidence={"phase": phase, "quiescent": True})

    def tick_union(self) -> None:
        result = self.run_sync_action(
            "strict_union_145",
            "strict_checkpoint_union_verification",
            {"expected": 145, "components": {"b": 60, "c": 60, "d": 25}},
            self.ops.union_145,
        )
        counts = result.get("counts", {})
        design = result.get("design_contract", {})
        if (
            result.get("status") == "complete"
            and result.get("complete") is True
            and int(counts.get("rows", -1)) == 145
            and int(counts.get("accepted", -1)) == 145
            and int(counts.get("pending", -1)) == 0
            and int(counts.get("rejected", -1)) == 0
            and int(design.get("phase_b_cells", -1)) == 60
            and int(design.get("phase_c_cells", -1)) == 60
            and int(design.get("phase_d_cells", -1)) == 25
            and int(design.get("unique_scientific_cells", -1)) == 145
            and int(design.get("unique_run_ids", -1)) == 145
        ):
            completion = canonical(self.store.path.parent / "COMPLETE.json")
            atomic_json(
                completion,
                {
                    "schema_version": COMPLETE_SCHEMA,
                    "status": "complete",
                    "campaign_generation_id": self.store.state["campaign_generation_id"],
                    "counts": counts,
                    "design_contract": design,
                    "union_verification": result["path"],
                    "completed_at": now(),
                },
            )
            self.store.transition("complete", evidence={**result, "completion_marker": str(completion)})
        else:
            self.store.halt("union145_strict_verification_failed", result)


def default_config(args: argparse.Namespace) -> CampaignConfig:
    script = Path(__file__).resolve()
    pipeline = script.parents[2]
    workspace = pipeline.parents[2]
    b_root = workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1"
    followup = workspace / "local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1_followup"
    staged = Path(f"/dev/shm/holomol-{os.getuid()}/scaling/domainwise_dataset_v1/domain_split_manifests_v1/manifest.json")
    return CampaignConfig(
        workspace=str(workspace),
        pipeline_root=str(pipeline),
        python=str(python_launcher(args.python or workspace / "local_runtime/envs/holomol-py311-cu121/bin/python")),
        remote_python=str(python_launcher(args.remote_python or workspace / "local_runtime/envs/holomol-py311-cu121/bin/python")),
        b_campaign_root=str(canonical(args.b_campaign_root or b_root)),
        b_local_matrix=str(canonical(args.b_local_matrix or b_root / "phase_b_matrix_local.json")),
        b_remote_matrix=str(canonical(args.b_remote_matrix or b_root / "phase_b_matrix_remote.json")),
        followup_root=str(canonical(args.followup_root or followup)),
        proposal_json=str(canonical(args.proposal_json or workspace / "scaling_data/reports/domainwise_scaling_matrix_proposal_v1.json")),
        shared_manifest=str(canonical(args.shared_manifest or workspace / "scaling_data/processed/domain_split_manifests_v1/manifest.json")),
        local_staged_manifest=str(canonical(args.local_staged_manifest or staged)),
        remote_staged_manifest=str(canonical(args.remote_staged_manifest or staged)),
        experiment_name=args.experiment_name,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_identity=str(canonical(args.remote_identity)),
        remote_workspace=str(canonical(args.remote_workspace)),
        poll_seconds=args.poll_seconds,
        max_b_launches_per_host=args.max_b_launches_per_host,
        max_phase_launches_per_host=args.max_phase_launches_per_host,
        max_smoke_attempts_per_host=args.max_smoke_attempts_per_host,
        stale_seconds=args.stale_seconds,
    )


def self_identity() -> dict[str, int]:
    return {key: int(value) for key, value in process_identity(os.getpid()).items() if key in {"pid", "ppid", "pgid", "sid", "start_ticks"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--remote-python", type=Path, default=None)
    parser.add_argument("--disable-remote-runtime", action="store_true", help="explicitly suspend remote runtime probes and launches without changing the frozen matrix")
    parser.add_argument("--local-execution-plan", type=Path, default=None, help="execute the frozen logical remote C queue on this physical host")
    parser.add_argument("--restore-venv-launchers", action="store_true",
                        help="restore symlink-preserving venv launchers in wait_b before C/D launches")
    parser.add_argument("--b-campaign-root", type=Path, default=None)
    parser.add_argument("--b-local-matrix", type=Path, default=None)
    parser.add_argument("--b-remote-matrix", type=Path, default=None)
    parser.add_argument("--followup-root", type=Path, default=None)
    parser.add_argument("--proposal-json", type=Path, default=None)
    parser.add_argument("--shared-manifest", type=Path, default=None)
    parser.add_argument("--local-staged-manifest", type=Path, default=None)
    parser.add_argument("--remote-staged-manifest", type=Path, default=None)
    parser.add_argument("--experiment-name", default="domainwise_scaling_fddfs_mf80000_v1_followup")
    parser.add_argument("--remote-host", default="user@compute-host")
    parser.add_argument("--remote-port", type=int, default=10090)
    parser.add_argument("--remote-identity", type=Path, default=Path("~/.ssh/id_rsa"))
    parser.add_argument("--remote-workspace", type=Path, default=Path("/path/to/moltoken-workspace"))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=900.0)
    parser.add_argument("--max-b-launches-per-host", type=int, default=3)
    parser.add_argument("--max-phase-launches-per-host", type=int, default=3)
    parser.add_argument("--max-smoke-attempts-per-host", type=int, default=2)
    parser.add_argument("--state-json", type=Path, default=None)
    parser.add_argument("--pid-json", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument("--generation-nonce", default="manual")
    parser.add_argument("--require-detached", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--idle-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-host-id", default="local", help=argparse.SUPPRESS)
    parser.add_argument("--job-probe-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cleanup-job-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--flock-probe-path", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.idle_probe:
        print(
            json.dumps(
                {
                    "host_id": args.probe_host_id,
                    "observed_at": now(),
                    "scaling_processes": active_scaling_processes(),
                    "compute_processes": running_compute_processes(),
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    if args.flock_probe_path is not None:
        print(
            json.dumps(
                flock_probe(args.flock_probe_path),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    if args.job_probe_json is not None:
        spec = json.loads(args.job_probe_json)
        if not isinstance(spec, dict):
            raise ValueError("detached job probe specification must be an object")
        print(
            json.dumps(
                probe_detached_job(spec, Path(__file__).resolve().with_name("run_domainwise_detached_job.py")),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    if args.cleanup_job_json is not None:
        spec = json.loads(args.cleanup_job_json)
        if not isinstance(spec, dict):
            raise ValueError("detached job cleanup specification must be an object")
        print(
            json.dumps(
                cleanup_detached_job(
                    spec,
                    Path(__file__).resolve().with_name("run_domainwise_detached_job.py"),
                ),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    if (
        args.poll_seconds <= 0
        or args.stale_seconds <= 0
        or args.max_b_launches_per_host <= 0
        or args.max_phase_launches_per_host <= 0
        or args.max_smoke_attempts_per_host <= 0
        or not 0 < args.remote_port < 65536
    ):
        raise ValueError("invalid controller timing/retry/remote settings")
    safe_component(args.generation_nonce, "--generation-nonce")
    config = default_config(args)
    followup = canonical(config.followup_root)
    # The durable operator plan survives detached-controller restarts. Removing
    # this placement / re-enabling SSH requires a later explicit operator change.
    placement = args.local_execution_plan or followup / "phase_c/local_execution_plan.json"
    if args.local_execution_plan is not None or placement.is_file():
        from domainwise_local_execution import load_local_execution_plan

        placement = canonical(placement)
        load_local_execution_plan(followup / "phase_c/matrix.json", placement)
        args.disable_remote_runtime = True
    else:
        placement = None
    controller_root = followup / "controller"
    state_path = canonical(args.state_json or controller_root / "state.json")
    pid_path = canonical(args.pid_json or controller_root / "controller.pid.json")
    lock_path = canonical(args.lock_file or Path(f"/dev/shm/holomol-{os.getuid()}/scaling/domainwise_145_controller.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another campaign controller owns {lock_path}") from exc
    identity = self_identity()
    if args.require_detached and (
        identity["ppid"] != 1
        or identity["pgid"] != identity["pid"]
        or identity["sid"] != identity["pid"]
    ):
        raise RuntimeError(f"controller is not a PID1-reparented session leader: {identity}")
    store = StateStore(state_path, config, restore_venv_launchers=args.restore_venv_launchers)
    atomic_json(
        pid_path,
        {
            "schema_version": PID_SCHEMA,
            **identity,
            "generation_nonce": args.generation_nonce,
            "campaign_generation_id": store.state["campaign_generation_id"],
            "configuration_fingerprint": config.fingerprint,
            "disabled_runtime_hosts": ["remote"] if args.disable_remote_runtime else [],
            "local_execution_plan_json": str(placement) if placement is not None else None,
            "state_json": str(state_path),
            "script": str(Path(__file__).resolve()),
            "started_at": now(),
        },
    )
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_stop)
    controller = CampaignController(store, ProductionOperations(config, controller_root, remote_runtime_disabled=args.disable_remote_runtime, local_execution_plan=placement))
    try:
        while not stop:
            controller.tick()
            print(
                json.dumps(
                    {
                        "time": now(),
                        "step": store.state["step"],
                        "revision": store.state["revision"],
                        "terminal": store.state["terminal"],
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )
            if args.once or store.state["step"] in TERMINAL_STEPS:
                break
            time.sleep(config.poll_seconds)
    finally:
        try:
            current = load_json(pid_path)
            if (
                isinstance(current, dict)
                and int(current.get("pid", -1)) == identity["pid"]
                and int(current.get("start_ticks", -1)) == identity["start_ticks"]
                and current.get("generation_nonce") == args.generation_nonce
            ):
                pid_path.unlink(missing_ok=True)
        except Exception:
            pass
        lock.close()
    return 0 if store.state["step"] != "halted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
