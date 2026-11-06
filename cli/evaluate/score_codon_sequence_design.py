#!/usr/bin/env python3
"""Decode every completed Codon raw slot, score with author functions and R.

Never repair, truncate or resample an Axis output. Published baseline profiles
are reused at the prepared input-only scope; full/fragment summaries are separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host
from pipeline.benchmarks.codon_generation import PROTOCOL, prompt_only, validate_checkpoint
from pipeline.benchmarks.codon_metrics import decode_coding_slot, score_coding_output, summarize_coding_slots
from pipeline.benchmarks.common import REFERENCE_LABEL
from pipeline.benchmarks.common import number, write_atomic


def model_reporting(summary):
    # Earlier Small summaries did not carry this reporting-only field.
    size = summary.get("model_size", "small")
    if size not in ("small", "big"):
        raise ValueError(f"unknown Codon model size: {size}")
    return size, size.title(), ("9e", "9f") if size == "small" else ("9g", "9h")


def codon_result_tables(summary):
    """Transcribe completed native summaries without rescoring or pooling scopes."""
    if not summary["complete"] or summary["protocol"] != PROTOCOL:
        raise ValueError("Codon tables require completed protocol-matched metrics")
    _, label, (coverage, properties) = model_reporting(summary)
    groups = summary["by_host_scope"]
    lines = [f"### 表{coverage}：Axis {label}请求覆盖与翻译正确性", "",
             "| 宿主 | 范围 | 请求数 | 解码成功 ↑ | 完整密码子框架 ↑ | 精确长度 ↑ | 翻译完全匹配 ↑ |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in groups:
        n = row["requested"]
        cells = [f"{row['decoded']}/{n}"]
        cells += [f"{row[key]['count']}/{n} ({number(row[key]['rate_all_requests'], percent=True)})"
                  for key in ("frame_valid", "exact_length", "translation_matches")]
        lines.append("| " + " | ".join([row["organism"], row["scope"], str(n), *cells]) + " |")
    lines += ["", f"### 表{properties}：Axis {label}有效编码输出的属性均值", "",
              "| 宿主 | 范围 | CSI（计分数／请求数） | GC%（计分数／请求数） | CFD%（计分数／请求数） |",
              "|---|---|---:|---:|---:|"]
    for row in groups:
        cells = [row["organism"], row["scope"]]
        for key in ("CSI", "GC_percent", "CFD_percent"):
            metric = row["all_frame_valid"][key]
            cells.append(f"{number(metric['mean'])} ({metric['scored']}/{row['requested']})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"表{coverage}全部使用原始请求分母，不删除失败或补抽；表{properties}仅统计完整密码子框架的ACGT输出，包含翻译不匹配者，未定义值保留为—。",
              "CSI／GC／CFD不是表达量或统一质量分数，不混排最优值。属性须同时看翻译正确率及计分覆盖率。",
              "同宿主、同范围的baseline数值见[三张参考表](codon_matched_scope_baselines_20260923.md)；",
              "baseline原始生成使用完整蛋白，事后裁剪只统一计分范围，不等于短输入重跑。", ""]
    return "\n".join(lines)


def codon_detail(summary, metrics_path):
    _, label, _ = model_reporting(summary)
    lines = [f"# CodonTransformer：Axis {label}本轮结果", "",
             "输入蛋白＋宿主，单次greedy生成，自由EOS；full_CDS与coding_prefix分开报告。", "",
             codon_result_tables(summary), "## 训练选点", "",
             "| 阶段 | 最佳轮次 | 停止轮次 | 最佳validation loss ↓ |",
             "|---|---:|---:|---:|"]
    for stage in ("alignment", "full"):
        value = summary["training"][stage]
        lines.append(f"| {stage} | {value['best_epoch']} | {value['stopping_epoch']} | {number(value['validation_loss'])} |")
    lines += ["", "## 翻译完全匹配子集的属性", "",
              "| 宿主 | 范围 | 匹配／请求 | CSI（计分数） | GC%（计分数） | CFD%（计分数） |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary["by_host_scope"]:
        cells = [row["organism"], row["scope"], f"{row['translation_matches']['count']}/{row['requested']}"]
        for key in ("CSI", "GC_percent", "CFD_percent"):
            metric = row["translation_matched"][key]
            cells.append(f"{number(metric['mean'])} ({metric['scored']})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Axis与各baseline的原版R曲线距离", "",
              "| 宿主 | 范围 | baseline | 计分对／请求对 | DTW | normalized DTW |",
              "|---|---|---|---:|---:|---:|"]
    for row in summary["DTW_to_each_baseline"]:
        lines.append("| " + " | ".join([row["organism"], row["scope"], row["baseline"],
                     f"{row['scored_pairs']}/{row['requested_pairs']}",
                     number(row["distance"]), number(row["normalized_distance"])]) + " |")
    lines += ["", "DTW是两条密码子使用曲线的距离，不是各模型独立的质量排名；无法计分的配对仍计入请求数。",
              "CSI调用作者get_CSI_value，GC调用get_GC_content，CFD调用get_cfd(threshold=0.3)；参考频率来自固定天然基因组。",
              "MinMax曲线与DTW沿用原版R函数；不使用已发现有问题的历史DTW／CFD列。",
              f"原生指标：`{metrics_path}`。", f"评测checkpoint：`{summary['checkpoint']}`。", ""]
    return "\n".join(lines)


def update_codon_paper_table(text, summary, detail_name=None):
    size, label, (coverage, properties) = model_reporting(summary)
    detail_name = detail_name or f"codon_{size}_current_results.md"
    before, heading, rest = text.partition("## 9. CodonTransformer：蛋白质到宿主适配DNA\n")
    section, following, after = rest.partition("\n## 待完成实验")
    if not heading or not following:
        raise ValueError("cannot locate Codon and pending sections in the paper table")
    section, count = re.subn(rf"^\| Axis {label} \|.*$",
                            f"| Axis {label} | 同范围结果见表{coverage}–{properties}；非全长52蛋白口径 | — | — | — | — | — |",
                            section, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected one Axis {label} row in full-length reference table")
    start, end = f"<!-- codon-{size}-results -->", f"<!-- /codon-{size}-results -->"
    block = start + "\n" + codon_result_tables(summary)
    block += f"\n两阶段选点、翻译匹配子集及DTW见[详细结果]({detail_name})。\n" + end
    if start in section:
        left, _, rest_of_section = section.partition(start)
        _, closing, right = rest_of_section.partition(end)
        if not closing:
            raise ValueError(f"missing end of Codon {size} result section")
        section = left + block + right
    else:
        # Each model owns only its block; preserve the other model and baselines.
        section = section.rstrip() + "\n\n" + block + "\n"
    updated = before + heading + section + following + after
    return re.sub(rf"^\| CodonTransformer / {label} \|.*\n", "", updated, flags=re.MULTILINE)


def publish_codon_results(summary, paper_table, metrics_path):
    size, _, _ = model_reporting(summary)
    detail = paper_table.with_name(f"codon_{size}_current_results.md")
    detail_text = codon_detail(summary, metrics_path)
    updated = update_codon_paper_table(paper_table.read_text(), summary, detail.name)
    write_atomic(detail, detail_text)
    write_atomic(paper_table, updated)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def scoring_input(request):
    """Compare actual prompts and scoring scope, not incidental provenance."""
    return {"prompt": prompt_only(request), "prompt_tokens": request["meta"]["prompt_tokens"],
            **{key: request[key] for key in
               ("max_new_tokens", "organism", "protein", "terminal_stop", "scope")}}


def load_completed_samples(root, requests):
    rows, configs = [], []
    for rank in sorted(root.glob("rank*")):
        config = json.loads((rank / "run_config.json").read_text())
        complete = json.loads((rank / "sampling_complete.json").read_text())
        if config["protocol"] != PROTOCOL or complete["protocol"] != PROTOCOL or not complete["complete"]:
            raise ValueError(f"incomplete/wrong Codon sampling protocol at {rank}")
        validate_checkpoint(config["checkpoint_metadata"])
        inputs = config["input_records"]
        if (len(inputs) != len(config["expected_ids"]) or
                {r["id"] for r in inputs} != set(config["expected_ids"])):
            raise ValueError("recorded inputs do not cover assigned requests")
        if any(scoring_input(r) != scoring_input(requests[r["id"]]) for r in inputs):
            raise ValueError("current input records differ from actual sampling inputs")
        samples = read_rows(rank / "samples.jsonl")
        if (set(config["expected_ids"]) != {r["id"] for r in samples} or
                complete["requests"] != len(samples)):
            raise ValueError("rank output coverage differs from assigned requests")
        rows.extend(samples)
        configs.append(config)
    if not configs or {c["rank"] for c in configs} != set(range(configs[0]["world"])):
        raise ValueError("missing sampling ranks")
    if len({c["checkpoint"] for c in configs}) != 1 or len({c["world"] for c in configs}) != 1:
        raise ValueError("ranks used different checkpoints/layouts")
    if len(rows) != len(requests) or {r["id"] for r in rows} != set(requests):
        raise ValueError("raw outputs have missing/duplicate/unexpected request IDs")
    return rows, configs[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("small", "big"), default="small",
                        help="Reporting label; does not change sampling or metric computation")
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--r-prefix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-table", type=Path,
                        help="After complete scoring, publish native metrics to the existing downstream table")
    args = parser.parse_args()
    require_compute_host("Codon graph decoding and official metrics", socket.gethostname())
    input_rows = read_rows(args.requests)
    requests = {r["id"]: r for r in input_rows}
    if not requests or len(requests) != len(input_rows):
        raise ValueError("empty or duplicate input requests")
    samples, config = load_completed_samples(args.samples_root, requests)
    baseline_requests = read_rows(args.baseline_root / "requests.jsonl")
    if len(baseline_requests) != len(requests) or {r["id"] for r in baseline_requests} != set(requests):
        raise ValueError("baseline requests do not cover Axis requests")
    for reference in baseline_requests:
        request = requests[reference["id"]]
        if any(reference[key] != request[key] for key in ("organism", "protein", "terminal_stop", "scope")):
            raise ValueError("baseline and Axis use different sequence scopes")
    args.output_root.mkdir(parents=True, exist_ok=False)
    sys.setrecursionlimit(1000000)
    sys.path.insert(0, str(args.assets / "CAI_PyPI-2.0.1-py3-none-any.whl"))
    spec = importlib.util.spec_from_file_location("codon_author_metrics", args.assets / "CodonEvaluation.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    references = {r["organism"]: r for r in json.loads(args.references.read_text())}
    from cli.evaluate.eval_axis_b_samples import BPEDecoder, FDDFSVocab
    from cli.evaluate.decode_axis_b_biological_conditions import decode_sequence
    from cli.evaluate.sample_tcr_translate_released_top20_auxiliary import convert_model_sequence
    decoder = BPEDecoder(args.merges, FDDFSVocab.load(args.vocab))
    scored, groups = [], defaultdict(list)
    with (args.output_root / "scored_samples.jsonl").open("x") as dest, \
            (args.output_root / "axis_dna_inputs.tsv").open("x", newline="") as r_input:
        writer = csv.DictWriter(r_input, fieldnames=["case_id", "organism", "model", "dna"], delimiter="\t")
        writer.writeheader()
        for sample in sorted(samples, key=lambda s: s["id"]):
            request = requests[sample["id"]]
            raw, eos, invalid = convert_model_sequence(sample["model_token_ids"])
            if (raw != sample["generated_bpe_ids"] or eos != sample["ended_with_eos"] or
                    invalid != sample["invalid_model_id"] or sample["protocol"] != PROTOCOL or
                    sample["prompt_tokens"] != request["meta"]["prompt_tokens"] or
                    sample["max_new_tokens"] != request["max_new_tokens"]):
                raise ValueError(f"inconsistent raw generation record {sample['id']}")
            dna, status, error = decode_coding_slot(sample, lambda ids: decode_sequence(ids, decoder, "dna"))
            ref = references[REFERENCE_LABEL[request["organism"]]]
            metrics = score_coding_output(dna, protein=request["protein"], terminal_stop=request["terminal_stop"],
                                          official=official, csi_weights=ref["csi_weights"],
                                          codon_frequencies=ref["codon_frequencies"])
            row = {**sample, "organism": request["organism"], "scope": request["scope"],
                   "protein": request["protein"], "generation_status": status,
                   "decode_error": error, "metrics": metrics}
            scored.append(row)
            groups[(row["organism"], row["scope"])].append(row)
            dest.write(json.dumps(row, allow_nan=False) + "\n")
            writer.writerow({"case_id": row["id"], "organism": row["organism"], "model": "Axis",
                             "dna": dna if dna is not None else ""})
    rscript = str(args.r_prefix / "bin/Rscript")
    profiles, self_pairs = args.output_root / "r_profiles.tsv", args.output_root / "r_self_pairs.tsv"
    subprocess.run([rscript, "--vanilla", str(ROOT / "cli/evaluate/codon_minmax_dtw.R"),
                    str(args.r_prefix / "kodonz-source"), str(args.output_root / "axis_dna_inputs.tsv"),
                    str(profiles), str(self_pairs)], check=True)
    dtw_path = args.output_root / "axis_vs_baseline_dtw.tsv"
    subprocess.run([rscript, "--vanilla", str(ROOT / "cli/evaluate/codon_dtw_against_profiles.R"),
                    str(profiles), str(args.baseline_root / "r_profiles.tsv"), str(dtw_path)], check=True)
    pair_groups = defaultdict(list)
    with dtw_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pair_groups[(row["organism"], requests[row["case_id"]]["scope"], row["model2"])].append(row)
    pair_summaries = []
    for (org, scope, baseline), rows in sorted(pair_groups.items()):
        valid = [row for row in rows if row["status"] == "scored"]
        item = {"organism": org, "scope": scope, "baseline": baseline,
                "requested_pairs": len(rows), "scored_pairs": len(valid)}
        for metric in ("distance", "normalized_distance"):
            item[metric] = math.fsum(float(r[metric]) for r in valid) / len(valid) if valid else None
        pair_summaries.append(item)
    summary = {"complete": True, "protocol": PROTOCOL, "checkpoint": config["checkpoint"],
               "training": config["training"], "model_size": args.model_size,
               "requested": len(scored), "aggregate": summarize_coding_slots(scored),
               "by_host_scope": [{"organism": org, "scope": scope, **summarize_coding_slots(rows)}
                                 for (org, scope), rows in sorted(groups.items())],
               "DTW_to_each_baseline": pair_summaries,
               "limitations": ["baseline DNA was generated from full proteins then scored on matched prefixes",
                               "property means carry coverage; translation failures are not silently excluded",
                               "DTW is pairwise similarity, not an overall quality ranking; GC/CSI not expression measures",
                               "official get_cfd/natural-genome references, not historical CSV CFD values"]}
    with (args.output_root / "metrics.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if args.paper_table:
        publish_codon_results(summary, args.paper_table, args.output_root / "metrics.json")
        print(f"CODON_{args.model_size.upper()}_PAPER_TABLE_UPDATED {args.paper_table}", flush=True)
    print(json.dumps({"complete": True, "requested": len(scored), "aggregate": summary["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
