#!/usr/bin/env python3
"""Publish five cross-domain scaling groups and monitor the live eval lanes.

Each group fixes the training domain. Six panels show union validation loss
and the five target-domain losses. All complete observations are retained and
connected in parameter-count order within each compute budget, without fitting.
Missing checkpoints/unfinished evaluations are not measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import os
from pathlib import Path
import socket
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eval_domainwise_cross_domain import (
    DOMAINS, PROTOCOL, aggregate, check_result, read, stamp, write,
)

LABELS = {"small_molecule": "Small molecule", "peptide": "Peptide", "protein": "Protein",
          "dna": "DNA", "rna": "RNA", "all": "All domains (token weighted)"}
COLORS = dict(zip((f"C{i}" for i in range(6)),
                  ("#3288ff", "#0891b2", "#f58232", "#9467df", "#dc4265", "#43b96d")))


def collect(root):
    plan = read(root / "plan.json")
    rows = []
    for p in plan["rows"]:
        path = root / "results" / f"{p['run_id']}.json"
        if not path.exists():
            continue
        r = read(path)
        check_result(r, p, plan)
        rows.append(dict(run_id=r["run_id"], train_domain=r["domain"], budget=r["budget"],
                         parameter_count=r["parameter_count"], target_flops=r["target_flops"],
                         actual_epoch_fraction=r["actual_epoch_fraction"], checkpoint=r["checkpoint"],
                         representative_first=r["representative_first"],
                         native_diagonal_delta=r["domains"][r["domain"]]["loss"]-p["validation_loss"],
                         elapsed_sec=r["elapsed_sec"], lane=r["lane"],
                         **{d+"_loss": r["domains"][d]["loss"] for d in DOMAINS},
                         **aggregate(r["domains"], r["domain"])))
    return plan, rows


def panel(ax, rows, key, plan_rows):
    diagnostics = []
    for budget in sorted({r["budget"] for r in plan_rows}):
        points = sorted((r for r in rows if r["budget"] == budget),
                        key=lambda r: (r["parameter_count"], r["run_id"]))
        color = COLORS[budget]
        if not points:
            continue
        ax.plot([r["parameter_count"]/1e6 for r in points], [r[key] for r in points],
                color=color, marker="o", markersize=4.5, lw=1.4, zorder=3, alpha=.9)
        diagnostics.append(dict(budget=budget, metric=key, style="observed_polyline_no_fit",
                                point_ids=[r["run_id"] for r in points]))
    ax.set_xscale("log")
    limits = [r["parameter_count"]/1e6 for r in plan_rows]
    ax.set_xlim(min(limits)/1.15, max(limits)*1.15)
    ax.set_xlabel("Parameters (million; log scale)", fontsize=9)
    ax.set_ylabel("Validation loss (nats/token)", fontsize=9)
    ax.grid(alpha=.18)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if not rows:
        ax.text(.5, .5, "No completed cross-domain evaluations yet", ha="center", transform=ax.transAxes)
    return diagnostics


def save(fig, root, name, vector=False):
    for suffix in (("png", "pdf") if vector else ("png",)):
        dest = root / f"{name}.{suffix}"
        tmp = root / f"{name}.{os.getpid()}.tmp"
        fig.savefig(tmp, format=suffix, dpi=160, bbox_inches="tight", facecolor="white")
        os.replace(tmp, dest)
    plt.close(fig)


def publish(root, output):
    plan, rows = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    done = len(rows) == plan["eligible"]
    when = stamp()
    subtitle = f"{'LOCAL BATCH COMPLETE' if done else 'PARTIAL / LIVE'} | {len(rows)}/{plan['eligible']} local checkpoints | {len(plan['unavailable'])} off-host checkpoints unavailable"
    footer = ("Dots: full frozen validation. Lines connect observed points in parameter-count order within each FLOPs budget; NO FITTING.\n"
              "All observations retained; missing checkpoints not imputed. Different training domains have different compute budgets. "
              "Context 512; no test access.")
    diagnostics = {}
    overview, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, d in enumerate(DOMAINS):
        selected = [r for r in rows if r["train_domain"] == d]
        eligible = [r for r in plan["rows"] if r["domain"] == d]
        absent = [r for r in plan["unavailable"] if r["domain"] == d]
        source = eligible + absent
        counts = Counter(r["budget"] for r in selected)
        local_counts = Counter(r["budget"] for r in eligible)
        absent_counts = Counter(r["budget"] for r in absent)
        legends = []
        for b in sorted({r["budget"] for r in source}):
            flops = next(r["target_flops"] for r in source if r["budget"] == b)
            legends.append(Line2D([], [], color=COLORS[b], marker="o", lw=1,
                                 label=f"{b}: {flops:.2g} FLOPs; {counts[b]}/{local_counts[b]} local, {absent_counts[b]} off-host"))
        ax = axes.flat[i]
        panel(ax, selected, "all_domain_loss", source)
        ax.set_title(f"Train: {LABELS[d]} | {len(selected)}/{len(eligible)} local", fontsize=12)
        ax.legend(handles=legends, fontsize=6.5, loc="best", framealpha=.8)
        fig, group_axes = plt.subplots(2, 3, figsize=(16, 9))
        diagnostics[d] = []
        for target, a in zip(("all",)+DOMAINS, group_axes.flat):
            key = "all_domain_loss" if target == "all" else target+"_loss"
            diagnostics[d] += panel(a, selected, key, source)
            a.set_title("Evaluate: " + LABELS[target] + (" [in-domain]" if target == d else ""), fontsize=12)
        fig.suptitle(f"Single-domain training: {LABELS[d]} → validation on all domains\n{subtitle}", fontsize=15)
        fig.legend(handles=legends, loc="lower center", bbox_to_anchor=(.5, .044), ncol=3, fontsize=8)
        fig.text(.5, .005, footer, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .11, 1, .93))
        save(fig, output, f"train_{d}", vector=done)
    axes.flat[5].axis("off")
    axes.flat[5].text(.03, .94, "Reading this figure\n\nOne panel per training domain.\nEach color is a fixed training FLOPs budget.\n"
                     "Y: union of five frozen validation sets,\nweighted by effective prediction-token count.\n\n"
                     "Missing/off-host checkpoints are not evaluated.\nLines connect observed points in increasing N.\n"
                     "No regression, smoothing, or fitted vertices.\n\n"+when,
                     va="top", fontsize=10, linespacing=1.6)
    overview.suptitle("Single → all: scaling of full-validation loss\n"+subtitle, fontsize=16)
    overview.text(.5, .005, footer, ha="center", fontsize=8)
    overview.tight_layout(rect=(0, .06, 1, .93))
    save(overview, output, "single_to_all_overview", vector=done)
    write(output / "polyline_points.json", diagnostics)
    status = dict(protocol=PROTOCOL, at=when, completed=len(rows), eligible=plan["eligible"],
                  off_host_unavailable=len(plan["unavailable"]), all_local_complete=done,
                  plot_style="observed_polyline_no_fit",
                  by_domain={d: dict(complete=sum(r["train_domain"]==d for r in rows),
                                      eligible=sum(r["domain"]==d for r in plan["rows"])) for d in DOMAINS},
                  max_abs_native_diagonal_delta=max((abs(r["native_diagonal_delta"]) for r in rows), default=None))
    write(output / "status.json", status)
    write(output / "summary.json", dict(status, rows=rows))
    if rows:
        tmp = output / f"summary.{os.getpid()}.csv.tmp"
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, output / "summary.csv")
    md = ["# 单域训练 → 全域验证：五组图", "", f"更新时间：{when}；本地完成 **{len(rows)}/{plan['eligible']}**。",
          "", f"另外 {len(plan['unavailable'])} 个外部 checkpoint 不可访问，不能把本批结果称为全部训练点的完整曲线。",
          "", "![总览](single_to_all_overview.png)", "", "每组固定训练领域，包含全域 token 加权 loss 及五个目标领域的 loss。",
          "不同颜色代表固定 FLOPs；点为完整验证实测，同一预算内按模型参数量排序，用折线直接连接。",
          "不做抛物线或其他拟合，不平滑、不补点。单个观测只显示点。横坐标保持对数刻度。", "",
          "| 训练领域 | 本批完成 / 本地可测 | 图 |", "|---|---:|---|"]
    for d in DOMAINS:
        md.append(f"| {LABELS[d]} | {status['by_domain'][d]['complete']}/{status['by_domain'][d]['eligible']} | [六面板](train_{d}.png) |")
    md += ["", "## 首批代表模型（只决定执行顺序，不筛掉其他模型）", "",
           "在各训练领域本地最高计算预算内，取原本域验证 loss 最低者；不同模型参数量和计算预算不同，不能据此作公平的跨训练领域排名。",
           "", "| 训练领域 | 参数量 M | FLOPs | 小分子 | 多肽 | 蛋白质 | DNA | RNA | 全域 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["representative_first"]:
            vals = " | ".join(f"{r[d+'_loss']:.4f}" for d in DOMAINS)
            md.append(f"| {LABELS[r['train_domain']]} | {r['parameter_count']/1e6:.2f} | {r['target_flops']:.3g} | {vals} | {r['all_domain_loss']:.4f} |")
    md += ["", "指标均为 nats/token，越低越好。全域主结果按有效预测 token 加权；另在 summary.csv 保留域等权与排除训练域的加权指标。",
           "不同目标领域有不同的序列熵，跨领域绝对 loss 不直接表示难度或能力排名；优先比较同一验证领域。",
           "", "当前点存在架构宽度/深度差异、单 seed 及外部缺失点；跨域曲线不自动继承本域的凸性或幂律结论。",
           "图只展示本批已完成五域评测的 checkpoint，不展示未完成单域中间结果或 smoke。",
           "原始计划与原子化结果见共享 runs/domainwise_cross_domain_eval_20260909；本批快照不会自动加入后续新训练点。"]
    temp = output / f"README.{os.getpid()}.tmp"
    temp.write_text("\n".join(md)+"\n")
    os.replace(temp, output / "README.md")
    print("PUBLISHED", status, flush=True)
    return status, plan, rows


def lane_health(root, plan, rows, pids):
    completed = {r["run_id"] for r in rows}
    states = []
    for lane, pid in enumerate(pids):
        pending = [r["run_id"] for r in plan["rows"] if r["lane"]==lane and r["run_id"] not in completed]
        proc = Path(f"/proc/{pid}/cmdline")
        try:
            command = proc.read_bytes()
        except FileNotFoundError:
            command = b""  # Expected when a lane naturally exits between two polls.
        alive = b"eval_domainwise_cross_domain.py" in command
        if pending and not alive:
            # A lane may finish after collect() but before the /proc query.
            still_pending = []
            for row in plan["rows"]:
                if row["run_id"] not in pending:
                    continue
                result_path = root / "results" / f"{row['run_id']}.json"
                if result_path.exists():
                    check_result(read(result_path), row, plan)
                else:
                    still_pending.append(row["run_id"])
            pending = still_pending
        progress_path = root / f"progress_lane{lane}.json"
        progress = read(progress_path) if progress_path.exists() else None
        age = time.time()-progress_path.stat().st_mtime if progress_path.exists() else None
        state = dict(lane=lane, pid=pid, process_matches=alive, pending=len(pending), progress=progress,
                     progress_age_seconds=age)
        states.append(state)
        if pending and not alive:
            write(root / "monitor_failure.json", dict(at=stamp(), error="evaluation_process_disappeared_with_pending_results", **state))
            raise RuntimeError(f"lane {lane} PID {pid} disappeared with {len(pending)} pending results")
        if pending and age is not None and age > 300:
            write(root / "monitor_failure.json", dict(at=stamp(), error="evaluation_progress_stale_over_300s", **state))
            raise RuntimeError(f"lane {lane}: progress stale {age:.0f}s; no automatic restart or cancellation")
    write(root / "monitor_health.json", dict(at=stamp(), lanes=states))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--pids", type=int, nargs=2, default=(405, 406))
    p.add_argument("--expected-host", default=None)
    a = p.parse_args()
    if a.expected_host and socket.gethostname() != a.expected_host:
        raise RuntimeError("Plot/monitor must run on the allocated evaluation worker")
    if a.interval < 10:
        p.error("monitor interval must be >=10 seconds")
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output / "publisher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        status, plan, rows = publish(a.root, a.output)
        if not a.watch:
            break
        lane_health(a.root, plan, rows, a.pids)
        if status["all_local_complete"]:
            print("MONITOR_COMPLETE: full local scope validated and plotted; worker retained", flush=True)
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
