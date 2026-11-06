#!/usr/bin/env python3
"""Durable host lifecycle intent/receipt ledger for Phase-C/D start/stop.

Cleanup intent is published on shared storage *before* a possibly ambiguous
SSH invocation.  The host launcher serializes start/stop with its lifecycle
flock, records a monotonic completion receipt, and treats delayed duplicates
or older cleanup sequences as no-ops.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_domainwise_phase_slot import atomic_json, load_json, safe_component


INTENT_SCHEMA = "domainwise_phase_cleanup_intent_v1"
STATE_SCHEMA = "domainwise_phase_lifecycle_state_v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def lifecycle_root(phase_root: Path, host_id: str) -> Path:
    safe_component(host_id, "host_id")
    return phase_root.expanduser().resolve() / "supervisors" / host_id / "lifecycle"


def _validated_intent(path: Path, phase: str, host_id: str, action_id: str) -> dict[str, Any]:
    value = load_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != INTENT_SCHEMA
        or value.get("phase") != phase
        or value.get("host_id") != host_id
        or value.get("action_id") != action_id
        or int(value.get("sequence", 0)) <= 0
    ):
        raise ValueError(f"invalid cleanup intent: {path}")
    return value


def prepare_cleanup_intent(
    phase_root: Path, phase: str, host_id: str, action_id: str
) -> dict[str, Any]:
    if phase not in {"c", "d"}:
        raise ValueError("phase must be c or d")
    safe_component(action_id, "cleanup action_id")
    root = lifecycle_root(phase_root, host_id)
    intents = root / "cleanup_intents"
    intents.mkdir(parents=True, exist_ok=True)
    path = intents / f"{action_id}.json"
    lock_path = root / "allocation.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if path.is_file():
            return _validated_intent(path, phase, host_id, action_id)
        sequences: list[int] = []
        for candidate in intents.glob("*.json"):
            try:
                value = load_json(candidate)
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == INTENT_SCHEMA
                    and value.get("phase") == phase
                    and value.get("host_id") == host_id
                ):
                    sequences.append(int(value["sequence"]))
            except (KeyError, OSError, TypeError, ValueError):
                # A malformed durable artifact is never overwritten.  Refuse
                # allocation so an operator can inspect it.
                raise ValueError(f"malformed lifecycle intent: {candidate}")
        value = {
            "schema_version": INTENT_SCHEMA,
            "phase": phase,
            "host_id": host_id,
            "action_id": action_id,
            "sequence": max(sequences, default=0) + 1,
            "created_at": now(),
            "creator_pid": os.getpid(),
        }
        atomic_json(path, value)
        return value
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def load_lifecycle_state(phase_root: Path, phase: str, host_id: str) -> dict[str, Any]:
    root = lifecycle_root(phase_root, host_id)
    path = root / "state.json"
    if not path.is_file():
        return {
            "schema_version": STATE_SCHEMA,
            "phase": phase,
            "host_id": host_id,
            "last_completed_cleanup_sequence": 0,
            "last_completed_cleanup_action_id": None,
        }
    value = load_json(path)
    completed = int(value.get("last_completed_cleanup_sequence", -1)) if isinstance(value, dict) else -1
    completed_action = value.get("last_completed_cleanup_action_id") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != STATE_SCHEMA
        or value.get("phase") != phase
        or value.get("host_id") != host_id
        or completed < 0
        or (completed == 0 and completed_action is not None)
        or (completed > 0 and not isinstance(completed_action, str))
    ):
        raise ValueError(f"invalid lifecycle state: {path}")
    if completed > 0:
        safe_component(str(completed_action), "completed cleanup action_id")
        intent_path = root / "cleanup_intents" / f"{completed_action}.json"
        intent = _validated_intent(
            intent_path, phase, host_id, str(completed_action)
        )
        if int(intent["sequence"]) != completed:
            raise ValueError("lifecycle state does not match its completed intent")
    return value


def cleanup_disposition(
    phase_root: Path,
    phase: str,
    host_id: str,
    action_id: str,
    sequence: int,
) -> dict[str, Any]:
    root = lifecycle_root(phase_root, host_id)
    intent = _validated_intent(
        root / "cleanup_intents" / f"{safe_component(action_id, 'cleanup action_id')}.json",
        phase,
        host_id,
        action_id,
    )
    if int(intent["sequence"]) != int(sequence):
        raise ValueError("cleanup sequence differs from durable intent")
    state = load_lifecycle_state(phase_root, phase, host_id)
    completed = int(state["last_completed_cleanup_sequence"])
    return {
        "status": "already_complete_or_superseded" if sequence <= completed else "execute",
        "sequence": sequence,
        "last_completed_cleanup_sequence": completed,
    }


def complete_cleanup(
    phase_root: Path,
    phase: str,
    host_id: str,
    action_id: str,
    sequence: int,
) -> dict[str, Any]:
    disposition = cleanup_disposition(
        phase_root, phase, host_id, action_id, sequence
    )
    state = load_lifecycle_state(phase_root, phase, host_id)
    if sequence > int(state["last_completed_cleanup_sequence"]):
        state.update(
            {
                "last_completed_cleanup_sequence": int(sequence),
                "last_completed_cleanup_action_id": action_id,
                "completed_at": now(),
            }
        )
        root = lifecycle_root(phase_root, host_id)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "state.json", state)
    return {"status": "complete", **state, "previous_disposition": disposition["status"]}


def launch_gate(phase_root: Path, phase: str, host_id: str) -> dict[str, Any]:
    root = lifecycle_root(phase_root, host_id)
    maximum = 0
    pending: list[dict[str, Any]] = []
    seen_sequences: dict[int, str] = {}
    for path in (root / "cleanup_intents").glob("*.json"):
        action_id = path.stem
        safe_component(action_id, "cleanup intent filename")
        value = _validated_intent(path, phase, host_id, action_id)
        sequence = int(value["sequence"])
        if sequence in seen_sequences:
            raise ValueError(
                "duplicate lifecycle cleanup sequence: "
                f"{sequence} ({seen_sequences[sequence]}, {action_id})"
            )
        seen_sequences[sequence] = action_id
        maximum = max(maximum, sequence)
    state = load_lifecycle_state(phase_root, phase, host_id)
    completed = int(state["last_completed_cleanup_sequence"])
    if maximum > completed:
        pending.append({"maximum_intent_sequence": maximum, "completed_sequence": completed})
    return {
        "status": "allowed" if not pending else "blocked_pending_cleanup",
        "maximum_intent_sequence": maximum,
        "last_completed_cleanup_sequence": completed,
        "pending": pending,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", choices=("prepare-cleanup", "cleanup-disposition", "complete-cleanup", "launch-gate"))
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("c", "d"), required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--action-id", default=None)
    parser.add_argument("--sequence", type=int, default=None)
    args = parser.parse_args()
    if args.action == "prepare-cleanup":
        if args.action_id is None:
            raise ValueError("prepare-cleanup requires --action-id")
        result = prepare_cleanup_intent(args.phase_root, args.phase, args.host_id, args.action_id)
    elif args.action == "launch-gate":
        result = launch_gate(args.phase_root, args.phase, args.host_id)
    else:
        if args.action_id is None or args.sequence is None or args.sequence <= 0:
            raise ValueError(f"{args.action} requires --action-id and positive --sequence")
        function = cleanup_disposition if args.action == "cleanup-disposition" else complete_cleanup
        result = function(args.phase_root, args.phase, args.host_id, args.action_id, args.sequence)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if args.action == "launch-gate" and result["status"] != "allowed":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
