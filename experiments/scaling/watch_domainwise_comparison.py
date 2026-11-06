#!/usr/bin/env python3
"""Update the full-scope comparison when validated result rows change.

Run on a worker. This process reads small result metadata only: it never
launches or signals trainers, reads weights, changes plans, or releases workers.
It owns its comparison output directory; do not run manual writers there while
this updater is live. The separate legacy dashboard may continue unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from collect_domainwise_cluster_results import (
    collect_results, read_plan_list_files, write_tables,
)


def run_cycle(*, baseline_tables, plan_list_file, extra_plans, output_dir,
              previous=None, python=sys.executable):
    plans = [*read_plan_list_files([plan_list_file]), *extra_plans]
    payload = collect_results(
        baseline_completed_csv=baseline_tables / "completed_runs.csv",
        baseline_pending_csv=baseline_tables / "pending_runs.csv",
        plan_jsons=plans,
    )
    # Compare actual rows, not file mtimes, hashes, or per-poll timestamps.
    current = (payload["completed_rows"], payload["pending_rows"])
    changed = current != previous
    if changed:
        tables = output_dir / "tables"
        write_tables(payload, tables)
        subprocess.run([
            python, str(Path(__file__).with_name("plot_domainwise_comparison.py")),
            "--tables", str(tables), "--output-dir", str(output_dir),
        ], check=True)
    return current, {
        "event": "updated" if changed else "unchanged",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed": len(current[0]), "pending": len(current[1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-tables", type=Path, required=True)
    parser.add_argument("--plan-list-file", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 1:
        parser.error("--interval-seconds must be positive")
    print(json.dumps({"event": "start", "pid": os.getpid(),
                      "hostname": socket.gethostname(),
                      "output_dir": str(args.output_dir.resolve()),
                      "interval_seconds": args.interval_seconds}), flush=True)
    previous = None
    while True:
        previous, report = run_cycle(
            baseline_tables=args.baseline_tables,
            plan_list_file=args.plan_list_file,
            extra_plans=args.plan_json,
            output_dir=args.output_dir,
            previous=previous,
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
