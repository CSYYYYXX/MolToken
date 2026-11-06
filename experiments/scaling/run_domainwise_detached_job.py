#!/usr/bin/env python3
"""Run one controller-owned command as an identity-recorded detached job.

This wrapper closes the crash window left by shell ``$!`` bookkeeping: the
actual post-``setsid -f`` process writes its own PID/start-ticks/session identity
before spawning work.  The command and all of its workers stay inside that
recorded session, so a controller can still find and clean the exact owned
session if this wrapper is SIGKILLed.  It holds a host-local flock for its
entire lifetime and writes an atomic terminal result.  It never removes or
overwrites command artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_domainwise_phase_slot import atomic_json, load_json, safe_component


SCHEMA = "domainwise_detached_job_identity_v1"
RESULT_SCHEMA = "domainwise_detached_job_result_v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def self_identity() -> dict[str, int]:
    raw = Path("/proc/self/stat").read_text(encoding="utf-8")
    tail = raw[raw.rfind(")") + 2 :].split()
    if len(tail) < 20:
        raise RuntimeError("cannot parse /proc/self/stat")
    return {
        "pid": os.getpid(),
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
    }


def proc_identity(pid: int) -> dict[str, int]:
    proc = Path("/proc") / str(int(pid))
    if proc.stat().st_uid != os.getuid():
        raise PermissionError(f"pid={pid} is not owned by uid={os.getuid()}")
    raw = (proc / "stat").read_text(encoding="utf-8")
    tail = raw[raw.rfind(")") + 2 :].split()
    if len(tail) < 20:
        raise RuntimeError(f"cannot parse /proc/{pid}/stat")
    return {
        "pid": int(pid),
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
    }


def session_member_pids(sid: int, *, exclude_pid: int | None = None) -> list[int]:
    """Return UID-owned live members of one exact Linux session."""

    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if exclude_pid is not None and pid == exclude_pid:
            continue
        try:
            identity = proc_identity(pid)
            if identity["sid"] == int(sid):
                members.append(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, RuntimeError):
            continue
    return sorted(members)


def signal_session_members(sid: int, signum: int, *, exclude_pid: int) -> list[int]:
    """Signal only revalidated, same-UID members; never signal the wrapper."""

    signalled: list[int] = []
    for pid in session_member_pids(sid, exclude_pid=exclude_pid):
        try:
            current = proc_identity(pid)
            if current["sid"] != int(sid):
                continue
            os.kill(pid, signum)
            signalled.append(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, RuntimeError):
            continue
    return signalled


def command_fingerprint(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--generation-nonce", required=True)
    parser.add_argument("--pid-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--expected-artifact", type=Path, default=None)
    parser.add_argument("--require-detached", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    safe_component(args.job_name, "--job-name")
    safe_component(args.generation_nonce, "--generation-nonce")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command or not Path(command[0]).is_absolute():
        raise ValueError("detached job command must begin with an absolute executable")
    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise FileNotFoundError(cwd)
    pid_path = args.pid_json.expanduser().resolve()
    result_path = args.result_json.expanduser().resolve()
    console_path = args.console_log.expanduser().resolve()
    lock_path = args.lock_file.expanduser().resolve()
    expected_artifact = (
        args.expected_artifact.expanduser().resolve()
        if args.expected_artifact is not None
        else None
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite detached result: {result_path}")
    for path in (pid_path, result_path, console_path, lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another detached job owns {lock_path}") from exc
    identity = self_identity()
    if args.require_detached and (
        identity["ppid"] != 1
        or identity["pgid"] != identity["pid"]
        or identity["sid"] != identity["pid"]
    ):
        raise RuntimeError(f"job is not a PID1-reparented session leader: {identity}")
    fingerprint = command_fingerprint(command)
    started_at = now()
    record = {
        "schema_version": SCHEMA,
        **identity,
        "job_name": args.job_name,
        "generation_nonce": args.generation_nonce,
        "command_fingerprint": fingerprint,
        "command": command,
        "cwd": str(cwd),
        "result_json": str(result_path),
        "expected_artifact": str(expected_artifact) if expected_artifact else None,
        "started_at": started_at,
    }
    atomic_json(pid_path, record)

    child: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None

    def stop(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            signal_session_members(identity["sid"], signal.SIGTERM, exclude_pid=identity["pid"])
            signal_session_members(identity["sid"], signal.SIGCONT, exclude_pid=identity["pid"])

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)

    returncode = 1
    error: str | None = None
    child_pid: int | None = None
    try:
        with console_path.open("ab", buffering=0) as console:
            child = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            child_pid = child.pid
            # The session identity was durably recorded before Popen.  This
            # update is supplemental evidence; cleanup remains possible even
            # if the wrapper dies between Popen and this atomic rewrite.
            try:
                child_identity = proc_identity(child.pid)
                if child_identity["sid"] != identity["sid"]:
                    raise RuntimeError("detached command escaped the recorded job session")
                record["child"] = child_identity
            except FileNotFoundError:
                record["child"] = {"pid": child.pid, "exited_before_identity_read": True}
            atomic_json(pid_path, record)
            while child.poll() is None:
                time.sleep(0.25)
            returncode = int(child.returncode or 0)
        if returncode == 0 and expected_artifact is not None and not expected_artifact.is_file():
            error = f"successful command did not create expected artifact: {expected_artifact}"
            returncode = 1
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        returncode = 1
    finally:
        if child is not None and child.poll() is None:
            signal_session_members(identity["sid"], signal.SIGTERM, exclude_pid=identity["pid"])
            signal_session_members(identity["sid"], signal.SIGCONT, exclude_pid=identity["pid"])
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                signal_session_members(identity["sid"], signal.SIGKILL, exclude_pid=identity["pid"])
                child.wait(timeout=10)
        # Also remove any same-session worker that outlived the direct command
        # (for example a profile subprocess whose coordinator crashed).
        if session_member_pids(identity["sid"], exclude_pid=identity["pid"]):
            signal_session_members(identity["sid"], signal.SIGTERM, exclude_pid=identity["pid"])
            signal_session_members(identity["sid"], signal.SIGCONT, exclude_pid=identity["pid"])
            time.sleep(0.25)
            signal_session_members(identity["sid"], signal.SIGKILL, exclude_pid=identity["pid"])
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete" if returncode == 0 else "failed",
            "job_name": args.job_name,
            "generation_nonce": args.generation_nonce,
            "command_fingerprint": fingerprint,
            "pid": identity["pid"],
            "start_ticks": identity["start_ticks"],
            "child_pid": child_pid,
            "returncode": returncode,
            "received_signal": received_signal,
            "error": error,
            "expected_artifact": str(expected_artifact) if expected_artifact else None,
            "expected_artifact_exists": bool(expected_artifact and expected_artifact.is_file()),
            "started_at": started_at,
            "finished_at": now(),
        }
        try:
            atomic_json(result_path, result)
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
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
