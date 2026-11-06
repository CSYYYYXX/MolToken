#!/usr/bin/env python3
"""汇总 scaling-law 训练结果，按模型规模输出三张表。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .scaling_paths import DEFAULT_RUN_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_RUN_ROOT


def load_json(path: Path) -> dict[str, Any] | None:
    """读取 JSON；不存在时返回 None。"""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def last_train_event(log_path: Path) -> dict[str, Any] | None:
    """从 train_log.jsonl 里取最后一个 train 事件，作为旧 run 的兜底。"""

    if not log_path.exists():
        return None
    last = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "train":
                last = row
    return last


def final_training_loss(out_dir: Path, summary: dict[str, Any] | None) -> tuple[float | None, str]:
    """取最终 training loss：优先取未打印尾段，其次取最后一次日志窗口。"""

    if summary:
        trailing = summary.get("trailing_train_loss")
        if trailing is not None and int(summary.get("trailing_train_loss_tokens") or 0) > 0:
            return float(trailing), "summary.trailing_train_loss"
        logged = summary.get("final_logged_train_loss")
        if logged is not None:
            return float(logged), "summary.final_logged_train_loss"
    row = last_train_event(out_dir / "train_log.jsonl")
    if row and row.get("loss") is not None:
        return float(row["loss"]), "train_log.last_train"
    return None, "missing"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    """写一张 Markdown 表，横坐标字段保留 token budget 和 FLOPs。"""

    headers = [
        "ratio",
        "token_budget_B",
        "flops_6ND",
        "max_steps",
        "global_step",
        "final_training_loss",
        "complete",
    ]
    lines = [f"# {title}", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        vals = [
            f"{row['ratio']:g}",
            f"{row['actual_token_budget_b']:.6f}",
            f"{row['flops_6nd']:.6e}",
            str(row["max_steps"]),
            str(row.get("global_step") or ""),
            "" if row.get("final_training_loss") is None else f"{row['final_training_loss']:.8f}",
            str(bool(row.get("complete"))),
        ]
        lines.append("|" + "|".join(vals) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--experiment-name", default="scaling_fddfs_mf80000_v1")
    ap.add_argument("--matrix-json", type=Path, default=None)
    args = ap.parse_args()

    exp_root = args.run_root / args.experiment_name
    matrix_path = args.matrix_json or (exp_root / "scaling_matrix.json")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    result_root = exp_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in matrix["rows"]:
        out_dir = Path(row["out_dir"])
        summary = load_json(out_dir / "summary.json")
        last_train = last_train_event(out_dir / "train_log.jsonl")
        loss, loss_source = final_training_loss(out_dir, summary)
        if summary:
            global_step = int(summary.get("global_step", 0))
        elif last_train and last_train.get("step") is not None:
            global_step = int(last_train["step"])
        else:
            global_step = 0
        complete = bool(summary and global_step >= int(row["max_steps"]))
        parameter_count = int(summary.get("parameter_count", row["parameter_count"])) if summary else int(row["parameter_count"])
        flops_6nd = float(6.0 * parameter_count * int(row["actual_tokens"]))
        final_loss = loss if complete else None
        current_loss = None if complete else loss
        merged = {
            "model_id": row["model_id"],
            "model_label": row["model_label"],
            "ratio": float(row["ratio"]),
            "parameter_count": parameter_count,
            "target_tokens": int(row["target_tokens"]),
            "actual_tokens": int(row["actual_tokens"]),
            "actual_token_budget_b": float(row["actual_token_budget_b"]),
            "flops_6nd": flops_6nd,
            "tokens_per_step": float(row["tokens_per_step"]),
            "max_steps": int(row["max_steps"]),
            "warmup_steps": int(row["warmup_steps"]),
            "global_step": global_step,
            "final_training_loss": final_loss,
            "current_training_loss": current_loss,
            "loss_source": loss_source,
            "complete": complete,
            "elapsed_sec": None if summary is None else summary.get("elapsed_sec"),
            "out_dir": str(out_dir),
        }
        all_rows.append(merged)
        by_model.setdefault(row["model_id"], []).append(merged)

    write_csv(result_root / "scaling_results_all.csv", all_rows)
    for model_id, rows in by_model.items():
        rows.sort(key=lambda r: float(r["ratio"]))
        write_csv(result_root / f"{model_id}_scaling_results.csv", rows)
        write_markdown(result_root / f"{model_id}_scaling_results.md", rows, f"{model_id} scaling results")

    complete_count = sum(1 for r in all_rows if r["complete"])
    summary = {
        "matrix_json": str(matrix_path),
        "result_root": str(result_root),
        "total_runs": len(all_rows),
        "complete_runs": complete_count,
        "pending_runs": len(all_rows) - complete_count,
    }
    (result_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
