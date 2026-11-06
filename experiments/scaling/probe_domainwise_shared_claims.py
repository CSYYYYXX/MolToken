#!/usr/bin/env python3
"""Bounded, isolated cross-worker claim diagnostic; never touches run claims."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time


def write_new(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("owner", "contender"), required=True)
    parser.add_argument("--hold-seconds", type=int, default=55)
    args = parser.parse_args()
    if not 1 <= args.hold_seconds <= 60:
        raise ValueError("hold-seconds must be between 1 and 60")
    identity = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}
    if args.role == "owner":
        args.root.mkdir(parents=True, exist_ok=False)
        with (args.root / "flock.claim").open("x+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (args.root / "mkdir.claim").mkdir()
            write_new(args.root / "exclusive.claim", identity)
            write_new(args.root / "ready.json", identity)
            print(json.dumps({"state": "holding", **identity}), flush=True)
            time.sleep(args.hold_seconds)
            write_new(args.root / "finished.json", {**identity, "finished": time.time()})
        print(json.dumps({"state": "finished", **identity}), flush=True)
        return

    owner = json.loads((args.root / "ready.json").read_text())
    if owner["host"] == identity["host"]:
        raise ValueError("contender must run on a different worker")
    if (args.root / "finished.json").exists():
        raise RuntimeError("owner finished before contention test")
    with (args.root / "flock.claim").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            flock_excluded = True
        else:
            flock_excluded = False
            fcntl.flock(stream, fcntl.LOCK_UN)
    try:
        (args.root / "mkdir.claim").mkdir()
    except FileExistsError:
        mkdir_excluded = True
    else:
        mkdir_excluded = False
    try:
        with (args.root / "exclusive.claim").open("x"):
            pass
    except FileExistsError:
        exclusive_create_excluded = True
    else:
        exclusive_create_excluded = False
    if (args.root / "finished.json").exists():
        raise RuntimeError("owner finished during contention test")
    result = {
        **identity, "owner": owner,
        "flock_excluded": flock_excluded,
        "mkdir_excluded": mkdir_excluded,
        "exclusive_create_excluded": exclusive_create_excluded,
        "production_claims_touched": False,
    }
    write_new(args.root / f"contender_{identity['host']}.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not mkdir_excluded or not exclusive_create_excluded:
        raise RuntimeError("atomic-create exclusion did not hold across workers")


if __name__ == "__main__":
    main()
