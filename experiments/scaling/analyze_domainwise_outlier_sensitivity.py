#!/usr/bin/env python3
"""Worker-side point diagnostics; never deletes runs or changes the live gallery.

Screens the currently poor quadratic fits using externally studentized residuals.
Reports every single-case deletion, including the optimistic best-deletion R2,
as sensitivity analysis rather than evidence that the omitted run is invalid.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_domainwise_adaptive_curves import analyze_adaptive_curve


def ols(design, loss):
    coef, _, rank, _ = np.linalg.lstsq(design, loss, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("regression design is rank deficient")
    residual = loss - design @ coef
    leverage = np.sum(design * (design @ np.linalg.inv(design.T @ design)), axis=1)
    sse = float(residual @ residual)
    return dict(coef=coef, residual=residual, leverage=leverage, sse=sse,
                r2=1 - sse / float(np.sum((loss - loss.mean()) ** 2)),
                loo_rmse=float(np.sqrt(np.mean((residual / (1 - leverage)) ** 2))))


def support(points):
    analysis = analyze_adaptive_curve(points)
    ns = [int(p["parameter_count"]) for p in points]
    best = analysis["discrete_argmin"]["parameter_count"]
    boundary = analysis["boundary_evidence"]
    return dict(observed_left=sum(n < best for n in ns),
                observed_right=sum(n > best for n in ns),
                fitted_left=boundary["measured_points_below_vertex"],
                fitted_right=boundary["measured_points_above_vertex"],
                fitted_minimum_n=analysis["quadratic_fit"]["vertex_parameter_count"],
                bracketing_status=analysis["bracketing_status"])


def diagnose_curve(curve, threshold=3.0):
    points = sorted(curve["points"], key=lambda p: int(p["parameter_count"]))
    count = len(points)
    if count < 6:
        raise ValueError("single-deletion bracketing comparison needs at least six points")
    x = np.log10([int(p["parameter_count"]) for p in points])
    y = np.array([float(p["validation_loss"]) for p in points])
    center, scale = float(x.mean()), float(x.std())
    z = (x - center) / scale
    design = np.column_stack((z * z, z, np.ones(count)))
    full = ols(design, y)
    observed_best_n = int(points[int(np.argmin(y))]["parameter_count"])
    fitted_vertex_n = curve["analysis"]["quadratic_fit"]["vertex_parameter_count"]
    mse = full["sse"] / (count - 3)
    cases = []
    for i, point in enumerate(points):
        keep = np.arange(count) != i
        deleted = ols(design[keep], y[keep])
        residual, leverage = float(full["residual"][i]), float(full["leverage"][i])
        t = residual / np.sqrt(deleted["sse"] / (count - 4) * (1 - leverage))
        cases.append(dict(
            domain=curve["domain"], budget=curve["budget"], run_id=point["run_id"],
            parameter_count=int(point["parameter_count"]), hidden_size=int(point["hidden_size"]),
            validation_loss=float(y[i]), residual=residual, leverage=leverage,
            right_of_observed_minimum=int(point["parameter_count"]) > observed_best_n,
            right_of_fitted_vertex=(fitted_vertex_n is not None
                                    and int(point["parameter_count"]) > fitted_vertex_n),
            externally_studentized_residual=float(t), screen_flag=bool(abs(t) > threshold),
            cooks_distance=residual ** 2 * leverage / (3 * mse * (1 - leverage) ** 2),
            deletion_r2=deleted["r2"], delta_r2=deleted["r2"] - full["r2"],
            deletion_coefficients=deleted["coef"].tolist(),
            deletion_support=support([p for j, p in enumerate(points) if j != i]),
        ))
    widths = np.array([int(p["hidden_size"]) for p in points])
    levels = sorted(set(widths.tolist()))
    width_design = np.column_stack([design] + [(widths == h).astype(float) for h in levels[1:]])
    width_fit = ols(width_design, y)
    return dict(
        domain=curve["domain"], budget=curve["budget"], count=count,
        pending=[p["run_id"] for p in curve["pending"]],
        all_point_r2=full["r2"], all_point_loo_rmse=full["loo_rmse"],
        coefficients=full["coef"].tolist(), log_n_center=center, log_n_scale=scale,
        all_point_support=support(points), points=cases,
        screening_threshold_abs_t=threshold,
        screen_flags=[p["run_id"] for p in cases if p["screen_flag"]],
        best_single_deletion=max(cases, key=lambda p: p["deletion_r2"]),
        best_right_single_deletion=max(
            (p for p in cases if p["right_of_observed_minimum"]),
            key=lambda p: p["deletion_r2"], default=None),
        largest_abs_studentized=max(cases, key=lambda p: abs(p["externally_studentized_residual"])),
        width_offset_diagnostic=dict(r2=width_fit["r2"], loo_rmse=width_fit["loo_rmse"],
                                     widths=levels, parameter_count=width_design.shape[1]),
    )


def plot_report(curves, output, *, right_only=False):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullLocator
    from plot_domainwise_completed_snapshot import plt, save_figure

    nrows = (len(curves) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 3.8 * nrows + 2), squeeze=False)
    colors = {64: "#08916d", 128: "#b45309", 256: "#0284c7", 384: "#ea580c", 512: "#8b5cf6"}
    markers = {64: "D", 128: "v", 256: "o", 384: "s", 512: "^"}
    for ax, curve in zip(axes.flat, curves):
        rows = curve["points"]
        ns = np.array([p["parameter_count"] for p in rows]) / 1e6
        grid = np.linspace(np.log10(ns.min() * 1e6), np.log10(ns.max() * 1e6), 200)
        z = (grid - curve["log_n_center"]) / curve["log_n_scale"]
        best = curve["best_right_single_deletion"] if right_only else curve["best_single_deletion"]
        if best is None:
            raise ValueError(f"no measured right-side point: {curve['domain']}/{curve['budget']}")
        ax.plot(10 ** grid / 1e6, np.polyval(curve["coefficients"], z), color="#334155", lw=2)
        ax.plot(10 ** grid / 1e6, np.polyval(best["deletion_coefficients"], z),
                color="#dc2626", ls="--", lw=1.6)
        for point in rows:
            width = point["hidden_size"]
            ax.scatter(point["parameter_count"] / 1e6, point["validation_loss"],
                       color=colors[width], marker=markers[width], s=44, zorder=3)
        ax.scatter(best["parameter_count"] / 1e6, best["validation_loss"],
                   s=160, facecolors="none", edgecolors="#dc2626", lw=1.5, zorder=4)
        label_rows = [p for p in rows if p["right_of_observed_minimum"]] if right_only else rows
        for index, point in enumerate(sorted(label_rows, key=lambda p: abs(p["residual"]), reverse=True)[:3]):
            label = f"{point['parameter_count'] / 1e6:.2f}M"
            ax.annotate(label, (point["parameter_count"] / 1e6, point["validation_loss"]),
                        xytext=(5, 8 + 12 * (index % 2)), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(np.geomspace(ns.min(), ns.max(), 5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.1f}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_title(f"{curve['domain'].upper()} {curve['budget']} | n={len(rows)} | "
                     f"R² {curve['all_point_r2']:.4f} → {best['deletion_r2']:.4f}", loc="left")
        ax.set_xlabel("Total parameters / M (log scale)")
        ax.set_ylabel("Full-validation loss (nat/token)")
        ax.grid(alpha=.18)
    for ax in list(axes.flat)[len(curves):]:
        ax.set_visible(False)
    legend = [Line2D([], [], color=colors[h], marker=markers[h], ls="", label=f"H={h}")
              for h in sorted({p["hidden_size"] for c in curves for p in c["points"]})]
    legend += [Line2D([], [], color="#334155", label="All-point quadratic"),
               Line2D([], [], color="#dc2626", ls="--", label=("Best right-side deletion sensitivity"
                          if right_only else "Best single-deletion sensitivity")),
               Line2D([], [], color="#dc2626", marker="o", markerfacecolor="none", ls="",
                      label="Omitted only in dashed sensitivity fit")]
    fig.suptitle("Right-side influence: sensitivity, not confirmed exclusions" if right_only
                 else "Poor-fit diagnostics: isolated outlier or architecture pattern?", fontsize=17, y=.995)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    fig.text(.05, .015, "All observations preserved. Dashed fits deliberately choose the largest R² gain; "
             "they are NOT confirmed exclusions.\n"
             "Colors show hidden width; width and depth co-vary. No raw run or primary result was removed.",
             fontsize=10, color="#475569")
    fig.tight_layout(rect=(0, .065, 1, .88))
    save_figure(fig, output, "poor_fit_outlier_sensitivity")


def write_report(report, output):
    curves = report["curves"]
    (output / "outlier_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + "\n")
    fields = [k for k in curves[0]["points"][0] if k not in ("deletion_coefficients", "deletion_support")]
    with (output / "point_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(p for c in curves for p in c["points"])
    lines = ["# 拟合差曲线：异常点与删点敏感性", "",
             f"分析时间：{report['generated_at']}；结果快照：{report['source_snapshot_at']}。",
             "分析全点二次拟合 R² < 0.8 的曲线，以及命令显式指定的曲线；原始结果、主总览和包夹判定均未修改。", "",
             "| 曲线 | 点数 | 原 R² | 最有利的单点删除 | 删除后 R² | 最大绝对学生化残差 | 超过3的点数 |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for c in curves:
        b = c["best_single_deletion"]
        lines.append(f"| {c['domain']} {c['budget']} | {c['count']} | {c['all_point_r2']:.6f} | "
                     f"{b['parameter_count']/1e6:.3f}M / `{b['run_id']}` | {b['deletion_r2']:.6f} | "
                     f"{abs(c['largest_abs_studentized']['externally_studentized_residual']):.3f} | {len(c['screen_flags'])} |")
    flags = [p for c in curves for p in c["screen_flags"]]
    lines += ["", "## 本次处理", "",
              f"共检查 {sum(c['count'] for c in curves)} 个点，|t|>3 的筛查标记为 {len(flags)} 个。没有直接删除任何点。",
              "", "## 怎么判断", "",
              "逐点计算外部学生化残差（|t|>3 只作为单点异常的经验筛查，不等于证明训练错误），以及杠杆值、Cook 距离和删除该点后的拟合。",
              "最有利的单点删除是刻意寻找 R² 最大值的敏感性上限，不能据此认定被删点无效；删点后 R² 的分母也会改变。",
              "如果筛查没有标出点，不能排除成组异常或二次模型不合适。单 seed、宽度/深度同时变化，也限制了统计解释。",
              "删除会改变最小值及两侧点数；逐点删除后的完整支撑情况保存在 JSON，不把删后拟合自动当成成功包夹。", "",
              "## 不删点、增加宽度项的诊断", "",
              "保持所有点，比较二次模型与‘二次模型 + 宽度分类偏移项’。额外参数会提高样本内 R²，因此同时报告留一预测 RMSE；这只是架构相关性的诊断，不替代原 scaling law。", "",
              "| 曲线 | 全点二次 R² | 加宽度项 R² | 二次留一 RMSE | 加宽度项留一 RMSE |",
              "|---|---:|---:|---:|---:|"]
    for c in curves:
        w = c["width_offset_diagnostic"]
        lines.append(f"| {c['domain']} {c['budget']} | {c['all_point_r2']:.6f} | {w['r2']:.6f} | "
                     f"{c['all_point_loo_rmse']:.6f} | {w['loo_rmse']:.6f} |")
    lines += ["", "## 右侧测点的敏感性", "",
              "右侧定义为 N 大于实测最低点；CSV 另列是否也在拟合顶点右侧。不是只挑 loss 偏高的点，正负残差均检查。",
              "下表列该侧绝对残差最大的两个点；删除后 R² 是探索性对照，不是剔除建议。", "",
              "| 曲线 | 右侧 N / M | 残差 | 外部学生化残差 t | 删除后 R² | 绝对 t>3 筛查 |",
              "|---|---:|---:|---:|---:|---|"]
    for c in curves:
        right = [p for p in c["points"] if p["right_of_observed_minimum"]]
        for p in sorted(right, key=lambda p: abs(p["residual"]), reverse=True)[:2]:
            lines.append(f"| {c['domain']} {c['budget']} | {p['parameter_count']/1e6:.3f} | "
                         f"{p['residual']:+.6f} | {p['externally_studentized_residual']:+.3f} | "
                         f"{p['deletion_r2']:.6f} | {'标记' if p['screen_flag'] else '否'} |")
    lines += ["", "## 残差最大的点", ""]
    for c in curves:
        lines.append(f"- {c['domain']} {c['budget']}：" + "；".join(
            f"`{p['run_id']}`，H={p['hidden_size']}，残差 {p['residual']:+.6f}"
            for p in sorted(c["points"], key=lambda p: abs(p["residual"]), reverse=True)[:3]) + "。")
    lines += ["", "![逐点诊断](poor_fit_outlier_sensitivity.png)", "",
              ("本次图中红虚线仅在右侧候选中选择使删后 R² 最大的一点；上方总表仍列所有点中的最有利单点删除。两者都是敏感性分析。"
               if report.get("plot_sensitivity_scope") == "right_of_observed_minimum"
               else "图中红虚线为所有候选中的最有利单点删除，仅作敏感性分析。"), "",
              "方法参考：[NIST 回归诊断](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/regrdiag.htm)，"
              "[Penn State 残差与影响点](https://online.stat.psu.edu/stat501/Lesson11)。", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-curve", action="append", default=[], metavar="DOMAIN:BUDGET",
                        help="Additionally inspect a curve even if its R2 is >=0.8.")
    parser.add_argument("--right-only", action="store_true",
                        help="Use the best right-side single deletion in the sensitivity plot; retain all points.")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    available = {f"{c['domain']}:{c['budget']}" for c in snapshot["curves"]}
    unknown = set(args.include_curve) - available
    if unknown:
        raise ValueError(f"unknown requested curves: {sorted(unknown)}")
    curves = [diagnose_curve(c) for c in snapshot["curves"]
              if c["analysis"] and (c["analysis"]["quadratic_fit"]["r_squared"] < .8
                                     or f"{c['domain']}:{c['budget']}" in args.include_curve)]
    if not curves:
        raise ValueError("snapshot contains no poor-fit curves to diagnose")
    report = dict(generated_at=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
                  source_snapshot=str(args.snapshot), source_snapshot_at=snapshot["updated_at"],
                  completed=snapshot["completed"], pending=snapshot["pending"],
                  additional_requested_curves=args.include_curve,
                  plot_sensitivity_scope="right_of_observed_minimum" if args.right_only else "all_points",
                  curves=curves, removed_runs=[], primary_gallery_changed=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(report, args.output_dir)
    plot_report(curves, args.output_dir, right_only=args.right_only)
    print(json.dumps(dict(curves=len(curves), points=sum(c["count"] for c in curves),
                         screen_flags=[p for c in curves for p in c["screen_flags"]],
                         removed_runs=[], output=str(args.output_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
