#!/usr/bin/env python3
"""Render same-domain budget overlays and a compact bracketing matrix.

This is a one-shot presentation of an atomic watcher report, not another
training monitor. It neither changes diagnoses nor touches running jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from matplotlib.ticker import FuncFormatter, LogLocator

from plot_domainwise_completed_snapshot import (
    BUDGET_COLORS, DOMAINS, DOMAIN_ZH, analyze_snapshot, budget_overlay_panel,
    format_overlay_axis, load_snapshot, plt, read_csv, save_figure,
)
from plot_domainwise_peptide_supplements import overlay_panel as peptide_overlay_panel


STATUS_LABELS = {
    "accepted": "已包夹", "higher_boundary": "右边界",
    "lower_boundary": "左边界", "sparse_support": "支撑不足",
    "poor_fit": "拟合差", "incomplete": "不足5点",
}


def format_comparison_axis(ax):
    """Use round logarithmic ticks, not interpolated pseudo-model sizes."""
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("参数量 N / 百万（对数刻度）", fontsize=10, labelpad=8)
    ax.set_ylabel("Validation loss ↓（nat/token）", fontsize=10, labelpad=8)


def comparison_panel(ax, curves):
    """Keep all original points/fits; simplify only the presentation."""
    budget_overlay_panel(ax, curves)
    format_comparison_axis(ax)
    ordered = sorted(curves, key=lambda curve: curve["target_flops"])
    handles = list(ax.get_legend().legend_handles)
    labels = [f"{curve['budget']} · {curve['target_flops']:.3g}" for curve in ordered]
    # Matplotlib fills legend columns first; reorder so readers see C0→C5 by row.
    order = [i for column in range(3) for i in range(column, len(ordered), 3)]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3,
              frameon=False, fontsize=9.2, handlelength=1.7, columnspacing=1.5,
              borderaxespad=0, labelspacing=0.65)
    done = sum(len(curve["points"]) for curve in curves)
    total = done + sum(len(curve["pending"]) for curve in curves)
    ax.set_title(f"{DOMAIN_ZH[curves[0]['domain']]} · {done}/{total} 点 · 图例为 FLOPs",
                 loc="left", fontsize=15, pad=51)


def matrix_cell(curve):
    """Only accepted brackets receive a numerical fitted optimum."""
    category = curve["visual_category"]
    if category == "accepted":
        n = curve["analysis"]["quadratic_fit"]["vertex_parameter_count"] / 1e6
        return f"已包夹\n{n:.2f}M"
    count = len(curve["points"])
    planned = count + len(curve["pending"])
    return f"{STATUS_LABELS[category]}\n{count}/{planned}"


def bracket_matrix(ax, curves):
    ax.axis("off")
    lookup = {(c["domain"], c["budget"]): c for c in curves}
    budgets = [f"C{i}" for i in range(6)]
    accepted = sum(c["visual_category"] == "accepted" for c in curves)
    ax.text(0, 1.31, f"当前最低点包夹 · {accepted}/24", fontsize=16,
            transform=ax.transAxes)
    ax.text(0, 1.20, "已包夹：显示拟合最低点 N*；其他：显示实测/计划点数",
            fontsize=9, color="#475569", transform=ax.transAxes)
    cells = [[DOMAIN_ZH[d]] + [matrix_cell(lookup[d, b]) for b in budgets]
             for d in DOMAINS]
    table = ax.table(cellText=cells, colLabels=["领域"] + budgets,
                     cellLoc="center", bbox=[0, 0.42, 1, 0.70])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce2ea")
        cell.set_linewidth(0.65)
        if row == 0:
            cell.set_facecolor("#f1f5f9")
            cell.get_text().set_weight("bold")
            if col:
                cell.get_text().set_color(BUDGET_COLORS[budgets[col - 1]])
        elif col == 0:
            cell.set_facecolor("#f8fafc")
        elif lookup[DOMAINS[row - 1], budgets[col - 1]]["visual_category"] == "accepted":
            cell.set_facecolor("#eaf6ee")
            cell.get_text().set_weight("bold")
    ax.text(0, 0.31,
            "● 全量 validation 实测   ○ 实测最低\n"
            "◇ 包夹通过的拟合顶点；未包夹不标最优参数量。\n"
            "四领域：至少五点才拟合并圈出最低；虚线 R² < 0.8。\n"
            "实线不等于已包夹；包夹状态见表。\n"
            "多肽：连接固定数据量实测点，不是等 FLOPs 抛物线。\n"
            "仅单 seed validation；各领域纵轴独立，不跨领域比较绝对 loss。",
            fontsize=9.3, linespacing=1.8, va="top", color="#475569",
            transform=ax.transAxes)


def snapshot_counts(curves, peptide, peptide_pending=()):
    completed = sum(len(c["points"]) for c in curves) + len(peptide)
    pending = sum(len(c["pending"]) for c in curves) + len(peptide_pending)
    accepted = sum(c["visual_category"] == "accepted" for c in curves)
    return completed, pending, accepted


def render(report, peptide, output, peptide_pending=()):
    curves = report["curves"]
    completed, pending, accepted = snapshot_counts(curves, peptide, peptide_pending)
    timestamp = datetime.fromisoformat(report["updated_at"]).astimezone(
        ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M HKT")
    with plt.rc_context({"font.family": ["Droid Sans Fallback", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(3, 2, figsize=(15.8, 14.4))
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.055, top=0.815,
                            hspace=0.64, wspace=0.24)
        for ax, domain in zip(axes.flat, DOMAINS):
            comparison_panel(ax, [c for c in curves if c["domain"] == domain])
        peptide_overlay_panel(axes[2, 0], peptide, peptide_pending,
                              format_axis=format_overlay_axis)
        format_comparison_axis(axes[2, 0])
        bracket_matrix(axes[2, 1], curves)
        fig.text(0.065, 0.979, "HoloMol · 同领域 Scaling 对照", fontsize=25, va="top")
        fig.text(0.065, 0.944,
                 "每个面板一个领域：叠加 C0 → C5，直接比较 FLOPs 增加后的曲线与最低点变化",
                 fontsize=12.5, color="#334155")
        fig.text(0.065, 0.920,
                 f"完整 validation {completed}/{completed + pending} 点  ·  标准领域已包夹 {accepted}/24  ·  快照 {timestamp}",
                 fontsize=11, color="#64748b")
        fig.text(0.065, 0.898,
                 "● 全量 validation 实测   ○ 实测最低   ◇ 已包夹的拟合顶点；不混入训练中间 loss。",
                 fontsize=10.5, color="#64748b")
        save_figure(fig, output, "domainwise_comparison")
        # Each domain also gets a readable slide-sized export from the same snapshot.
        for domain in (*DOMAINS, "peptide"):
            fig, ax = plt.subplots(figsize=(10.6, 6.6))
            fig.subplots_adjust(left=0.105, right=0.975, bottom=0.21, top=0.79)
            if domain == "peptide":
                peptide_overlay_panel(ax, peptide, peptide_pending,
                                      format_axis=format_overlay_axis)
                note = "多肽按固定数据量分组；连线仅连接实测点，不是等 FLOPs 拟合。"
            else:
                comparison_panel(ax, [c for c in curves if c["domain"] == domain])
                note = "点：全量 validation；线：log₁₀N 二次拟合；虚线：R² < 0.8；不足五点不拟合、不圈最低。"
            format_comparison_axis(ax)
            fig.text(0.105, 0.075, note, fontsize=10, color="#475569")
            fig.text(0.105, 0.035, f"○ 实测最低；◇ 包夹通过的拟合顶点。单 seed；快照 {timestamp}",
                     fontsize=9, color="#64748b")
            save_figure(fig, output, f"{domain}_comparison")
    return {"completed": completed, "pending": pending, "accepted_curves": accepted,
            "snapshot": timestamp}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--tables", type=Path,
                        help="A completed collector snapshot, including peptide supplements")
    parser.add_argument("--peptide-table", type=Path,
                        help="Completed-results CSV containing the fixed-data peptide observations")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.tables is not None:
        if args.peptide_table is not None:
            parser.error("--tables already contains peptide data; do not also pass --peptide-table")
        completed, pending = load_snapshot(args.tables)
        metadata_path = args.tables / "collection_metadata.json"
        # Collector tables have a data timestamp; re-rendering is not a new collection.
        updated_at = (json.loads(metadata_path.read_text())["generated_at"]
                      if metadata_path.is_file()
                      else datetime.fromtimestamp(max(
                          (args.tables / name).stat().st_mtime
                          for name in ("completed_runs.csv", "pending_runs.csv")
                      )).astimezone().isoformat(timespec="seconds"))
        report = {"updated_at": updated_at,
                  "metric_split": "validation", "test_accessed": False,
                  "curves": analyze_snapshot(completed, pending)}
        peptide = [r for r in completed if r["domain"] == "peptide"]
        peptide_pending = [r for r in pending if r["domain"] == "peptide"]
    else:
        if args.peptide_table is None:
            parser.error("--report requires --peptide-table")
        report = json.loads(args.report.read_text())
        if report["metric_split"] != "validation" or report["test_accessed"]:
            raise ValueError("Expected a validation-only bracketing report")
        peptide = [r for r in read_csv(args.peptide_table) if r["domain"] == "peptide"]
        peptide_pending = []
        for row in peptide:
            if row["validation_complete_view"] != "true" or row["metric_split"] != "validation":
                raise ValueError(f"Incomplete peptide validation: {row['run_id']}")
            row["parameter_count"] = int(row["parameter_count"])
            row["validation_loss"] = float(row["validation_loss"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = render(report, peptide, args.output_dir, peptide_pending)
    provenance = {**summary, "source_report": str(args.report) if args.report else None,
                  "source_tables": str(args.tables) if args.tables else None,
                  "source_peptide_table": str(args.peptide_table) if args.peptide_table else None,
                  "curves": report["curves"], "peptide": peptide,
                  "peptide_pending": peptide_pending}
    (args.output_dir / "snapshot.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    page = ("<!doctype html><meta charset='utf-8'><title>HoloMol Scaling 对照</title>"
            "<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:16px}img{width:100%}</style>"
            "<h1>同领域 Scaling 对照</h1>"
            f"<p>静态快照：{html.escape(summary['snapshot'])}；不是实时训练页。</p>"
            "<p><a href='domainwise_comparison.png'>PNG</a> · <a href='domainwise_comparison.pdf'>PDF</a> · "
            "<a href='domainwise_comparison.svg'>SVG</a> · <a href='snapshot.json'>对应数据</a></p>"
            "<p>同一领域共用坐标，图例逐行 C0 → C5，数值为固定 FLOPs。"
            "多肽为固定数据量，不与等 FLOPs 曲线混用。</p>"
            "<a href='domainwise_comparison.png'><img src='domainwise_comparison.svg'></a>"
            "<h2>各领域单图</h2>")
    for domain in (*DOMAINS, "peptide"):
        label = "多肽" if domain == "peptide" else DOMAIN_ZH[domain]
        page += (f"<p>{label}：<a href='{domain}_comparison.png'>PNG</a> · "
                 f"<a href='{domain}_comparison.pdf'>PDF</a> · "
                 f"<a href='{domain}_comparison.svg'>SVG</a></p>")
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
