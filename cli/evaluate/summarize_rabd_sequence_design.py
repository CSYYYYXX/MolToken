#!/usr/bin/env python3
"""Merge disjoint RAbD evaluation ranks; never mix partial and formal coverage."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.rabd_metrics import aggregate_cases
from pipeline.runtime_host import require_compute_host
from pipeline.benchmarks.common import number, write_atomic
from pipeline.benchmarks.common import completed_lineage, checkpoint_path_identity


def summarize(root):
    configs = [json.loads(p.read_text()) for p in sorted(root.glob("rank*/run_config.json"))]
    if not configs:
        raise ValueError("no rank configurations")
    first = configs[0]
    if sorted(c["rank"] for c in configs) != list(range(first["world"])):
        raise ValueError("missing or duplicate evaluation ranks")
    common = ("protocol", "checkpoint", "split", "world", "expected_cases", "draws_per_case",
              "top", "seed", "sampling", "ranking", "dtype")
    for config in configs:
        if any(config[key] != first[key] for key in common):
            raise ValueError("ranks disagree on evaluation protocol")
    cases = []
    for config in configs:
        metrics = json.loads((root / f"rank{config['rank']}" / "metrics.json").read_text())
        if not metrics["complete"] or len(metrics["cases"]) != len(config["case_indices"]):
            raise ValueError("incomplete rank metrics")
        if sorted(c["case_index"] for c in metrics["cases"]) != config["case_indices"]:
            raise ValueError("rank case coverage mismatch")
        cases.extend(metrics["cases"])
    if sorted(c["case_index"] for c in cases) != list(range(first["expected_cases"])):
        raise ValueError("missing or duplicate test case")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    counts = Counter()
    for case in cases:
        if case["all_draw_counts"]["requested"] != first["draws_per_case"]:
            raise ValueError("case sampling budget mismatch")
        counts.update(case["all_draw_counts"])
    aggregate = aggregate_cases(cases)
    aggregate.update(all_draw_counts=dict(counts),
                     raw_validity=counts["decoded"] / counts["requested"],
                     raw_valid_exact_length_rate=counts["exact_requested_length"] / counts["requested"])
    formal = (first["split"] == "test" and len(cases) == 58
              and first["draws_per_case"] == 10000 and first["top"] == 100)
    return {"complete": True, "formal_released58_budget_complete": formal,
            **{key: first[key] for key in common}, "aggregate": aggregate,
            "case_count": len(cases), "cases": sorted(cases, key=lambda c: c["case_index"]),
            "paper_baseline_aar_percent": {"RAbD": 28.53, "LSTM": 22.53,
                                           "AR-GNN": 23.86, "RefineGNN": 35.37},
            "baseline_source": "https://arxiv.org/pdf/2110.04624 (Table1)",
            "comparability": ["Published paper60 cases; released test58; baselines are paper reports, not reruns.",
                              "Graph-token path likelihood ranking is not AA marginal PPL.",
                              "Free EOS vs upstream fixedlength. Failures remain, no hidden replacement.",
                              "Official AAR is null if selected output assumptions fail; fixedslot score is explicitly separate."]}


def big_result_table(result):
    """Transcribe the completed evaluator's counts; never regenerate or rescore."""
    if not result["complete"] or not result["formal_released58_budget_complete"]:
        raise ValueError("RAbD publication requires the completed released58 evaluation")
    a = result["aggregate"]
    raw = a["all_draw_counts"]
    lines = ["### 表8c：Axis Big生成有效性与长度遵从", "",
             "| 候选集合 | 请求数 | 有效肽序列 ↑ | 有效且长度正确 ↑ |",
             "|---|---:|---:|---:|"]
    for label, n, valid, same in (
        ("全部原始生成", raw["requested"], raw["decoded"], raw["exact_requested_length"]),
        ("各抗体按模型自身概率排名前100", a["selected"], a["decoded_valid"], a["valid_exact_length"]),
    ):
        lines.append(f"| {label} | {n:,} | {valid:,}/{n:,} ({number(valid/n, percent=True)}) | "
                     f"{same:,}/{n:,} ({number(same/n, percent=True)}) |")
    lines += ["", f"Big固定候选位AAR = {a['matched_residues']:,}/{a['requested_residues']:,}"
              f" = {number(a['fixed_slot_aar_invalid_or_wrong_length_zero'], percent=True)}。",
              "无效或错长度候选整条计零，分母不变；不预过滤、不补抽、不裁剪。",
              f"满足原作者长度／有效性假设的AAR字段：{number(a['official_aar'], percent=True)}；"
              "—表示假设未全部满足，不把适配分数填作官方AAR。", ""]
    return "\n".join(lines)


def update_big_paper_table(text, result, detail_name="rabd_big_results_20260923.md"):
    table = big_result_table(result)
    before, heading, rest = text.partition("## 8. Antibody：RAbD / CDR-H3 sequence design\n")
    section, following, after = rest.partition("\n## 9.")
    if not heading or not following:
        raise ValueError("cannot locate RAbD section in the paper table")
    # This writer owns the Big row in 8a and table 8c, not later baseline tables.
    boundary = re.search(r"\n### 表8[c-z]：", section)
    later_sections = ""
    if boundary:
        section, later_sections = section[:boundary.start()], section[boundary.start():]
        if later_sections.startswith("\n### 表8c："):
            next_heading = later_sections.find("\n### ", 1)
            later_sections = later_sections[next_heading:] if next_heading >= 0 else ""
    score = 100 * result["aggregate"]["fixed_slot_aar_invalid_or_wrong_length_zero"]
    section, count = re.subn(r"^\| Axis Big \|.*$",
                            f"| Axis Big | R | {score:.2f} | 无效或错长度候选整条计零，固定分母 |",
                            section, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("expected exactly one RAbD Big table row")
    section = section.replace("Axis AAR =", "Axis Small AAR =")
    section = section.replace("\n两阶段为Alignment", "\nSmall两阶段为Alignment")
    section = section.rstrip() + "\n\n" + table
    section += f"\nBig两阶段选点与原始指标路径见[详细结果]({detail_name})。\n"
    section += later_sections
    updated = before + heading + section + following + after
    updated = updated.replace("已收录RAbD Small完整测评", "已收录RAbD Small／Big完整测评")
    return re.sub(r"^\| RAbD / Big \|.*\n", "", updated, flags=re.MULTILINE)


def big_detail(result, training, metrics_path):
    lines = ["# RAbD / CDR-H3：Big完整评测", "", big_result_table(result),
             "## 训练选点", "",
             "| 阶段 | 最佳轮次 | 停止轮次 | 最佳validation loss ↓ |",
             "|---|---:|---:|---:|"]
    for stage in ("alignment", "full"):
        row = training[stage]
        lines.append(f"| {stage} | {row['best_epoch']} | {row['stopping_epoch']} | {number(row['validation_loss'])} |")
    lines += ["", "两阶段max100＋训练集内验证早停；Full从本次Alignment/best开始，未用test选checkpoint。",
              "与Small使用同一数据划分及评测规则：TRAIN8,150／内部验证496／发布test58，训练seed42；",
              f"测评seed{result['seed']}，每抗体{result['draws_per_case']:,}次原始采样，按自身NLL取top{result['top']}。",
              "报告的是生成BPE路径的概率，不是AA序列的边缘概率；没有3D坐标输入。",
              "论文60例与发布58例不一致，已记录的11项裁剪后训练／测试重合仍保留。",
              "基线来源及同集对照见下游总表；不据此声称无泄漏泛化或严格同协议胜负。",
              "论文参考基线与Small对比见[下游总表](axis_downstream_paper_tables_20260923.md#8-antibodyrabd--cdr-h3-sequence-design)。",
              f"原生指标：`{metrics_path}`。", f"测评checkpoint：`{result['checkpoint']}`。", ""]
    return "\n".join(lines)


def publish_big_results(result, output_root, paper_table):
    # Validate completion before touching reports; reuse stage metadata logic.
    big_result_table(result)
    training = completed_lineage(output_root.parent, "rabd_cdrh3_sequence_design")
    if checkpoint_path_identity(result["checkpoint"]) != checkpoint_path_identity(training["full"]["checkpoint"]):
        raise ValueError("RAbD evaluation did not use this run's selected Full/best")
    detail = paper_table.with_name("rabd_big_results_20260923.md")
    updated = update_big_paper_table(paper_table.read_text(), result, detail.name)
    write_atomic(detail, big_detail(result, training, output_root / "metrics.json"))
    write_atomic(paper_table, updated)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--publish-big-to", type=Path,
                   help="report existing completed metrics into this paper table; do not rerun aggregation")
    p.add_argument("--wait-for-metrics", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=1800)
    args = p.parse_args()
    require_compute_host("RAbD result aggregation", socket.gethostname())
    if args.wait_for_metrics and args.publish_big_to is None:
        p.error("--wait-for-metrics requires --publish-big-to")
    if args.poll_seconds < 1:
        p.error("--poll-seconds must be positive")
    if args.publish_big_to is not None:
        metrics_path = args.output_root / "metrics.json"
        while args.wait_for_metrics and not metrics_path.exists():
            print(f"{datetime.now(timezone.utc).isoformat()} WAITING_FOR_RABD_BIG_EVALUATION {metrics_path}", flush=True)
            time.sleep(args.poll_seconds)
        result = json.loads(metrics_path.read_text())
        publish_big_results(result, args.output_root, args.publish_big_to)
        print(f"RABD_BIG_PAPER_TABLE_PUBLISHED {args.publish_big_to}", flush=True)
        return
    result = summarize(args.output_root)
    with (args.output_root / "metrics.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("complete", "formal_released58_budget_complete", "aggregate")}))


if __name__ == "__main__":
    main()
