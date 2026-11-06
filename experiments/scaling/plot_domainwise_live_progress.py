#!/usr/bin/env python3
"""Read-only, validation-only live campaign plots; never open model/test files.

Completion means a worker completion marker plus a complete final validation,
not the campaign's separate strict acceptance gate. Failed/orphan attempts and
batch-tuning probes are not observations. This monitor owns only its output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import time

import numpy as np


DOMAINS = ("small_molecule", "protein", "dna", "rna", "peptide")
LABELS = dict(zip(DOMAINS, ("小分子", "蛋白质", "DNA", "RNA", "多肽")))
ATTEMPT_RE = re.compile(r"attempt_([0-9]{3,})\Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def final_observation(row, phase):
    root = Path(row["out_dir"]).resolve()
    marker_path = root / "complete.json"
    if not marker_path.exists():
        return None
    marker = read_json(marker_path)
    if (not isinstance(marker, dict) or marker.get("status") != "complete"
            or marker.get("run_id") != row["run_id"]
            or (phase in {"C", "D"} and marker.get("phase") != phase.lower())):
        raise ValueError(f"Invalid completion marker: {marker_path}")
    attempt = Path(marker["attempt_dir"]).resolve()
    summary_path = Path(marker["summary"]).resolve()
    if attempt.parent != root or not ATTEMPT_RE.fullmatch(attempt.name) or summary_path != attempt / "summary.json":
        raise ValueError(f"Completion lineage mismatch: {marker_path}")
    summary = read_json(summary_path)
    validation = summary.get("final_validation", {})
    if not isinstance(validation, dict):
        raise ValueError(f"Invalid final validation: {summary_path}")
    raw_loss = validation.get("loss")
    if not isinstance(raw_loss, (int, float)) or isinstance(raw_loss, bool):
        raise ValueError(f"Invalid validation loss: {summary_path}")
    loss = float(raw_loss)
    expected = int(row["target_input_tokens_with_eos"])
    actual = int(summary.get("actual_input_tokens", -1))
    checks = (
        validation.get("complete_view") is True,
        validation.get("input_token_coverage") == 1.0,
        int(validation.get("tokens", -1)) == int(row["validation_loss_tokens"]),
        int(validation.get("input_tokens", -1)) > 0,
        int(validation.get("input_tokens", -1)) == int(validation.get("expected_input_tokens", -2)),
        math.isfinite(loss) and loss > 0,
        int(summary.get("parameter_count", -1)) == int(row["parameter_count"]),
        int(summary.get("target_input_tokens", -1)) == expected,
        actual >= expected,
        summary.get("termination_reason") == "target_input_tokens",
    )
    if not all(checks):
        raise ValueError(f"Incomplete/mismatched validation summary: {summary_path}")
    return {
        "run_id": row["run_id"], "phase": phase, "domain": row["domain"],
        "group": row["budget_label"], "n": int(row["parameter_count"]),
        "loss": loss, "flops": float(row["target_flops_6nd"]),
        "epochs": float(row["target_epoch_fraction"]),
        "actual_tokens": actual, "summary": str(summary_path),
        "completed_at": marker.get("completed_at"),
    }


def latest_training(row, now):
    paths = sorted(
        (path for path in Path(row["out_dir"]).glob("attempt_*/train_log.jsonl")
         if ATTEMPT_RE.fullmatch(path.parent.name)),
        key=lambda path: int(ATTEMPT_RE.fullmatch(path.parent.name).group(1)),
    )
    if not paths:
        return None
    path = paths[-1]
    # Read only a bounded tail. A final unterminated append is expected on NFS.
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - 65536)
        stream.seek(start)
        lines = stream.read().splitlines(keepends=True)
    if start:
        lines = lines[1:]
    events = []
    for line in lines:
        if not line.endswith(b"\n"):
            continue
        event = json.loads(line)
        if event.get("event") == "train":
            events.append(event)
    if not events:
        return None
    last = events[-1]
    done = int(last["cumulative_input_tokens"])
    target = int(row["target_input_tokens_with_eos"])
    span = events[-1].get("elapsed_sec", 0) - events[0].get("elapsed_sec", 0)
    speed = ((done - events[0]["cumulative_input_tokens"]) / span if span > 0
             else float(last.get("input_tokens_per_sec_global_approx", 0)))
    age = max(0.0, now - path.stat().st_mtime)
    return {
        "run_id": row["run_id"], "domain": row["domain"], "group": row["budget_label"],
        "n": int(row["parameter_count"]), "host": row.get("assigned_host", "local"),
        "step": last["step"], "tokens": done, "target": target,
        "fraction": min(1.0, done / target), "log_age_seconds": round(age, 1),
        "status": "日志活跃" if age < 180 else "日志未更新",
        "eta_minutes": round(max(0, target - done) / speed / 60, 1) if age < 180 and speed > 0 else None,
    }


def fit_curve(points):
    if len(points) < 4 or len({p["n"] for p in points}) != len(points):
        return None
    x = np.log10([p["n"] for p in points])
    y = np.array([p["loss"] for p in points])
    center, scale = float(x.mean()), float(np.ptp(x))
    if not scale > 0:
        return None
    coeff = np.polyfit((x - center) / scale, y, 2)
    predicted = np.polyval(coeff, (x - center) / scale)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - predicted) ** 2)) / total if total > 0 else None
    xs = np.linspace(float(x.min()), float(x.max()), 70)
    ys = np.polyval(coeff, (xs - center) / scale)
    return {"basis": "loss ~ quadratic(log10(parameter_count))", "r2": r2,
            "provisional": len(points) < 5, "convex": bool(coeff[0] > 0),
            "samples": [[round(10.0 ** xx, 2), round(float(yy), 7)] for xx, yy in zip(xs, ys)]}


def snapshot(base, followup):
    now = time.time()
    sources = [("B", base / "phase_b_matrix_local.json"), ("B", base / "phase_b_matrix_remote.json")]
    for phase in ("C", "D"):
        path = followup / f"phase_{phase.lower()}" / "matrix.json"
        if path.exists():
            sources.append((phase, path))
    rows, seen = [], set()
    for phase, path in sources:
        matrix = read_json(path)
        if matrix["metadata"]["status"] != f"runnable_phase_{phase.lower()}":
            raise ValueError(f"Not a formal runnable matrix: {path}")
        for row in matrix["rows"]:
            expected_phase = {"B": "B_bracketing", "C": "C_completion", "D": "D_peptide_surface"}[phase]
            if row.get("execution_phase") != expected_phase:
                raise ValueError(f"Matrix row phase mismatch: {row['run_id']}")
            if row["run_id"] in seen:
                raise ValueError(f"Duplicate formal run: {row['run_id']}")
            seen.add(row["run_id"])
            rows.append((phase, row))
    observations, active = [], []
    for phase, row in rows:
        point = final_observation(row, phase)
        if point is not None:
            observations.append(point)
        else:
            training = latest_training(row, now)
            if training:
                active.append(training)
    panels = []
    for domain in DOMAINS:
        selected = [p for p in observations if p["domain"] == domain]
        grouped = defaultdict(list)
        for point in selected:
            grouped[point["group"]].append(point)
        groups = []
        for label, points in sorted(grouped.items(), key=lambda pair: min(p["flops"] for p in pair[1])):
            points.sort(key=lambda p: p["n"])
            caption = (f"{points[0]['epochs']:g} epoch" if domain == "peptide"
                       else f"{label} · {points[0]['flops']:.2g} FLOPs")
            groups.append({"label": label, "caption": caption, "points": points, "fit": fit_curve(points)})
        panels.append({"domain": domain, "name": LABELS[domain], "count": len(selected),
                       "protocol": "固定数据量（非等 FLOPs）" if domain == "peptide" else "固定 FLOPs",
                       "groups": groups})
    return {
        "schema": "domainwise_validation_progress_v1", "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": dict(Counter(p["phase"] for p in observations)), "total": len(observations), "planned": 145,
        "completion_definition": "worker marker + full final validation; not strict campaign acceptance",
        "active": active, "panels": panels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("interval must be at least 10 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "refresh.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = False

        def request_stop(signum, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        identity = {"pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(), "sid": os.getsid(0),
                    "start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
                    "interval_seconds": args.interval, "status": "running"}
        atomic_text(args.output_dir / "refresh.pid.json", json.dumps(identity, indent=2))
        while not stop:
            data = snapshot(args.base, args.followup)
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
            template = args.template.read_text(encoding="utf-8")
            if template.count("__CAMPAIGN_DATA__") != 1:
                raise ValueError("template must have exactly one data placeholder")
            atomic_text(args.output_dir / "progress.json", encoded)
            atomic_text(args.output_dir / "scaling-progress.html", template.replace("__CAMPAIGN_DATA__", encoded))
            print(f"{data['updated']} trained={data['total']}/145 counts={data['counts']} logs={len(data['active'])}", flush=True)
            if args.once:
                break
            # An eventless sleep here is a detached monitor, not an agent tool wait.
            deadline = time.monotonic() + args.interval
            while not stop and time.monotonic() < deadline:
                time.sleep(min(1, deadline - time.monotonic()))
        identity["status"] = "stopped"
        atomic_text(args.output_dir / "refresh.pid.json", json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
