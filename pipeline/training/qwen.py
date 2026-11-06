"""Current FDDFS Qwen3 pretraining entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.paths import PipelinePaths


@dataclass(frozen=True)
class QwenTrainingPaths:
    """当前 Qwen3 预训练和评估脚本路径。"""

    train_script: Path
    worker_launcher: Path
    flagship_launcher: Path
    eval_script: Path


def qwen_training_paths(paths: PipelinePaths | None = None) -> QwenTrainingPaths:
    paths = paths or PipelinePaths()
    return QwenTrainingPaths(
        train_script=paths.pretrain_cli_dir / "train_qwen3_pretrain.py",
        worker_launcher=paths.pretrain_cli_dir / "train_qwen3_pretrain.py",
        flagship_launcher=paths.pretrain_cli_dir / "train_qwen3_pretrain.py",
        eval_script=paths.evaluate_cli_dir / "eval_qwen3_fddfs_decode_strict.py",
    )


def require_qwen_training_tools(paths: PipelinePaths | None = None) -> QwenTrainingPaths:
    """检查 Qwen3 packed training 相关脚本是否存在。"""

    qps = qwen_training_paths(paths)
    missing = [
        path
        for path in (
            qps.train_script,
            qps.worker_launcher,
            qps.flagship_launcher,
            qps.eval_script,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("missing Qwen training tool files: " + ", ".join(str(path) for path in missing))
    return qps
