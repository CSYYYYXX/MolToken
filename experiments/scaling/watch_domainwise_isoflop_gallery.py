#!/usr/bin/env python3
"""Worker-only five-domain IsoFLOP gallery; excludes historical fixed-D peptide.

Reuse the existing collectors, full-validation checks and all-point fits.
This publisher never launches/stops training or archives files. It must be the
only publisher of its output directory; --once supports a staging preview.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import os
from pathlib import Path
import socket
import time
from zoneinfo import ZoneInfo

from collect_domainwise_cluster_results import collect_results, read_plan_list_files
from domainwise_peptide_isoflop import PROTOCOL, collect as collect_peptide, curves_for
from plot_domainwise_completed_snapshot import (
    DOMAINS, DOMAIN_ZH, BUDGET_COLORS, analyze_snapshot, plt, save_figure,
)
from plot_domainwise_comparison import comparison_panel, matrix_cell
from watch_domainwise_adaptive_results import progress_snapshot

ALL_DOMAINS = (*DOMAINS, "peptide")
LABELS = {**DOMAIN_ZH, "peptide": "多肽"}
NOTE = "点：全量 validation；线：log₁₀N 二次拟合；虚线：R² < 0.8；不足五点不拟合。"


def observed_minimum_support(curve):
    """Count distinct completed sizes, never pending points or fitted values."""
    points = curve["points"]
    if not points:
        return {"left": 0, "right": 0, "two_each_side": False, "dual_supported": False}
    best = min(points, key=lambda p: float(p["validation_loss"]))
    best_n = int(best["parameter_count"])
    sizes = {int(p["parameter_count"]) for p in points}
    left, right = sum(n < best_n for n in sizes), sum(n > best_n for n in sizes)
    two_each_side = left >= 2 and right >= 2
    return {"left": left, "right": right, "two_each_side": two_each_side,
            "dual_supported": two_each_side and curve["visual_category"] == "accepted"}


def compose_snapshot(payload, peptide_completed, peptide_pending):
    completed, pending = [], []
    for source, destination in ((payload["completed_rows"], completed),
                                (payload["pending_rows"], pending)):
        for original in source:
            if original["domain"] == "peptide":
                continue  # Explicitly replace the historical fixed-D protocol.
            if original["domain"] not in DOMAINS:
                raise ValueError(f"unknown standard domain: {original['domain']}")
            row = dict(original)
            row["parameter_count"] = int(row["parameter_count"])
            row["target_flops_6nd"] = float(row["target_flops_6nd"])
            if destination is completed:
                row["validation_loss"] = float(row["validation_loss"])
                row["hidden_size"] = int(row["hidden_size"])
            destination.append(row)
    curves = analyze_snapshot(completed, pending)
    curves.extend(curves_for(peptide_completed, peptide_pending))
    completed.extend(peptide_completed)
    pending.extend(peptide_pending)
    ids = [row["run_id"] for row in completed + pending]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run in five-domain IsoFLOP gallery")
    return {
        "updated_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
        "metric_split": "validation", "test_accessed": False,
        "historical_fixed_d_peptide_included": False,
        "peptide_protocol": PROTOCOL, "curves": curves,
        "completed_rows": completed, "pending_rows": pending,
        "completed": len(completed), "pending": len(pending),
        "accepted_curves": sum(c["visual_category"] == "accepted" for c in curves),
        "dual_supported_curves": sum(observed_minimum_support(c)["dual_supported"] for c in curves),
        "curve_count": len(curves),
        "source_standard_collection": payload["metadata"],
    }


def collect_snapshot(args):
    standard_plans = read_plan_list_files([args.standard_plan_list])
    payload = collect_results(
        baseline_completed_csv=args.baseline_tables / "completed_runs.csv",
        baseline_pending_csv=args.baseline_tables / "pending_runs.csv",
        plan_jsons=standard_plans,
    )
    completed, pending = [], []
    peptide_plans = read_plan_list_files([args.peptide_plan_list])
    if not peptide_plans:
        raise ValueError("peptide plan list is empty")
    for path in peptide_plans:
        plan = json.loads(path.read_text())
        if plan["scientific_protocol"] != PROTOCOL:
            raise ValueError(f"not the approved peptide IsoFLOP protocol: {path}")
        manifest = json.loads(Path(plan["domain_split_manifest"]).read_text())
        done, missing = collect_peptide(plan, manifest)
        completed.extend(done)
        pending.extend(missing)
    report = compose_snapshot(payload, completed, pending)
    report["source_standard_plans"] = [str(p) for p in standard_plans]
    report["source_peptide_plans"] = [str(p) for p in peptide_plans]
    return report


def matrix_rows(curves):
    lookup = {(c["domain"], c["budget"]): c for c in curves}
    return [[LABELS[d]] + [matrix_cell(lookup[d, f"C{i}"])
                          if (d, f"C{i}") in lookup else "—"
                          for i in range(6)] for d in ALL_DOMAINS]


def draw_matrix(ax, report):
    ax.axis("off")
    ax.text(0, 1.24, f"拟合判据 · {report['accepted_curves']}/{report['curve_count']}",
            fontsize=16, transform=ax.transAxes)
    ax.text(0, 1.13, "表格按原拟合判据；实测最低还须左右各 ≥2 点", fontsize=9,
            color="#475569", transform=ax.transAxes)
    table = ax.table(cellText=matrix_rows(report["curves"]),
                     colLabels=["领域"] + [f"C{i}" for i in range(6)],
                     cellLoc="center", bbox=[0, .32, 1, .72])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    lookup = {(c["domain"], c["budget"]): c for c in report["curves"]}
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce2ea")
        cell.set_linewidth(.65)
        if row == 0:
            cell.set_facecolor("#f1f5f9")
            cell.get_text().set_weight("bold")
            if col:
                cell.get_text().set_color(BUDGET_COLORS[f"C{col-1}"])
        elif col == 0:
            cell.set_facecolor("#f8fafc")
        elif (ALL_DOMAINS[row-1], f"C{col-1}") in lookup:
            curve = lookup[ALL_DOMAINS[row-1], f"C{col-1}"]
            if curve["visual_category"] == "accepted":
                cell.set_facecolor("#eaf6ee")
                cell.get_text().set_weight("bold")
    ax.text(0, .23,
            "● 实测  ○ 实测最低  ◇ 已包夹的拟合顶点\n"
            "多肽固定 FLOPs，batch 1024；原实验最多 4 token-epoch。\n"
            "C1/C2 批准的补点最多 5 遍；C3–C5 未设置。\n"
            "全部观测参与拟合；实线不等于包夹，仍需双侧实测支持。\n"
            "单 seed validation；各领域纵轴独立，不比较跨领域绝对 loss。",
            fontsize=9, linespacing=1.8, va="top", color="#475569",
            transform=ax.transAxes)


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def export_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render(report, output):
    output.mkdir(parents=True, exist_ok=True)
    DOMAIN_ZH["peptide"] = "多肽 · 固定 FLOPs"
    timestamp = datetime.fromisoformat(report["updated_at"]).strftime("%Y-%m-%d %H:%M HKT")
    with plt.rc_context({"font.family": ["Droid Sans Fallback", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(3, 2, figsize=(15.8, 14.6))
        fig.subplots_adjust(left=.065, right=.985, bottom=.055, top=.82,
                            hspace=.67, wspace=.24)
        for ax, domain in zip(axes.flat, ALL_DOMAINS):
            comparison_panel(ax, [c for c in report["curves"] if c["domain"] == domain])
        draw_matrix(axes[2, 1], report)
        fig.text(.065, .979, "HoloMol · 五领域固定 FLOPs 对照", fontsize=25, va="top")
        fig.text(.065, .945, "同一领域叠加不同计算预算；多肽使用新 IsoFLOP 实验，不混入旧固定数据量结果。",
                 fontsize=12, color="#334155")
        fig.text(.065, .920,
                 f"完整 validation {report['completed']}/{report['completed']+report['pending']} 点"
                 f"  ·  实测+拟合双满足 {report['dual_supported_curves']}/{report['curve_count']}"
                 f"  ·  原拟合判据 {report['accepted_curves']}/{report['curve_count']}  ·  快照 {timestamp}",
                 fontsize=11, color="#64748b")
        fig.text(.065, .897, NOTE + "  ○ 实测最低；◇ 已包夹拟合顶点。", fontsize=10, color="#64748b")
        save_figure(fig, output, "domainwise_overlay")
        for domain in ALL_DOMAINS:
            fig, ax = plt.subplots(figsize=(10.6, 6.6))
            fig.subplots_adjust(left=.105, right=.975, bottom=.22, top=.79)
            comparison_panel(ax, [c for c in report["curves"] if c["domain"] == domain])
            fig.text(.105, .09, NOTE, fontsize=9.5, color="#475569")
            extra = (
                "多肽：固定 FLOPs / batch 1024；原上限 4 遍，C1/C2 批准的补点上限 5 遍。 "
                if domain == "peptide" else ""
            )
            fig.text(.105, .05, extra + "○ 实测最低；◇ 已包夹拟合顶点。", fontsize=9, color="#64748b")
            fig.text(.105, .015, f"单 seed、validation-only；快照 {timestamp}", fontsize=9, color="#64748b")
            save_figure(fig, output, f"{domain}_overlay")
    fields = ("run_id", "domain", "budget", "model_id", "parameter_count", "hidden_size",
              "num_hidden_layers", "target_flops_6nd", "actual_flops_6nd", "actual_input_tokens",
              "actual_epoch_fraction", "effective_batch_records", "validation_loss")
    export_csv(output / "completed_runs.csv", report["completed_rows"], fields)
    export_csv(output / "pending_runs.csv", report["pending_rows"], fields)
    diagnostics = []
    for c in report["curves"]:
        a = c["analysis"]
        support = observed_minimum_support(c)
        diagnostics.append({"domain": c["domain"], "budget": c["budget"],
                            "target_flops": c["target_flops"], "completed": len(c["points"]),
                            "pending": len(c["pending"]), "status": c["visual_category"],
                            "r_squared": a["quadratic_fit"]["r_squared"] if a else "",
                            "observed_best_N": a["discrete_argmin"]["parameter_count"] if a else "",
                            "observed_left_points": support["left"],
                            "observed_right_points": support["right"],
                            "observed_two_each_side": support["two_each_side"],
                            "observed_and_fitted_supported": support["dual_supported"],
                            "accepted_fitted_N": a["quadratic_fit"]["vertex_parameter_count"]
                            if c["visual_category"] == "accepted" else ""})
    export_csv(output / "bracketing_summary.csv", diagnostics, list(diagnostics[0]))
    atomic_text(output / "snapshot.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    links = " · ".join(f"<a href='{d}_overlay.png'>{LABELS[d]}</a>" for d in ALL_DOMAINS)
    update_note = ("每60秒刷新页面，完整结果变化后重画。" if report["publisher_mode"] == "watch"
                   else "本页为静态预览，数据截至上述快照时间。")
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta http-equiv='refresh' content='60'><title>HoloMol 五领域 IsoFLOP</title>"
            "<style>body{font:16px system-ui;max-width:1550px;margin:24px auto;padding:16px;color:#172033}"
            "img{width:100%;height:auto}p{line-height:1.6}a{color:#1465ba}</style>"
            "<h1>五领域固定 FLOPs 对照</h1>"
            f"<p>完整 validation {report['completed']}/{report['completed']+report['pending']}；"
            f"实测+拟合双满足 {report['dual_supported_curves']}/{report['curve_count']}；"
            f"原拟合判据 {report['accepted_curves']}/{report['curve_count']}；{html.escape(timestamp)}。</p>"
            "<p>双满足：原拟合判据通过，且实测最低点左右各至少2个不同参数量的完整观测；不计在途点。</p>"
            "<p>多肽为新固定 FLOPs 结果，旧固定数据量结果不计入本图。"
            f"{update_note}旧图表归档在 rubbish/。</p>"
            "<p><a href='domainwise_overlay.png'>总览 PNG</a> · <a href='domainwise_overlay.pdf'>PDF</a> · "
            "<a href='domainwise_overlay.svg'>SVG</a> · <a href='completed_runs.csv'>完整指标 CSV</a> · "
            "<a href='bracketing_summary.csv'>包夹状态 CSV</a> · <a href='snapshot.json'>数据快照</a></p>"
            f"<p>{links}</p><img src='domainwise_overlay.png' alt='五领域固定FLOPs总览'></html>")
    atomic_text(output / "index.html", page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tables", type=Path, required=True)
    parser.add_argument("--standard-plan-list", type=Path, required=True)
    parser.add_argument("--peptide-plan-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 30:
        parser.error("interval must be at least 30 seconds")
    print(json.dumps({"event": "start", "pid": os.getpid(), "output": str(args.output_dir)}), flush=True)
    previous = None
    while True:
        report = collect_snapshot(args)
        report["publisher_mode"] = "static_preview" if args.once else "watch"
        signature = (report["completed_rows"], report["pending_rows"])
        changed = signature != previous
        if changed:
            render(report, args.output_dir)
            previous = signature
        plans = [Path(p) for p in report["source_standard_plans"] + report["source_peptide_plans"]]
        status = {"updated_at": report["updated_at"], "pid": os.getpid(),
                  "hostname": socket.gethostname(), "worker_release_authorized": False,
                  "completed_runs": report["completed"], "pending_runs": report["pending"],
                  "accepted_curves": report["accepted_curves"],
                  "dual_supported_curves": report["dual_supported_curves"],
                  "progress": progress_snapshot(plans)}
        atomic_text(args.output_dir / "status.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"event": "updated" if changed else "unchanged",
                          **{k: report[k] for k in ("updated_at", "completed", "pending", "accepted_curves", "curve_count")}}), flush=True)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
