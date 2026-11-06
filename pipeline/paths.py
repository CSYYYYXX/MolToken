"""统一管理本项目 pipeline 依赖的路径。"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path


PIPELINE_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = Path(
    os.environ.get(
        "MOLTOKEN_DATA_ROOT",
        os.environ.get("HOLOMOL_DATA_ROOT", "./data"),
    )
)

TOKENIZER_ASSET_ROOT = Path(__file__).resolve().parents[1] / "moltoken/assets/fddfs_bpe"
DEFAULT_TEXT_ENCODER = Path(os.environ.get(
    "MOLTOKEN_TEXT_ENCODER", os.environ.get(
        "HOLOMOL_TEXT_ENCODER", str(LOCAL_DATA_ROOT / "pretrained_encoders/all-mpnet-base-v2")
    )
))


@dataclass(frozen=True)
class PipelinePaths:
    """当前项目的关键路径。

    默认值指向当前 ``code/pipeline`` 内已经迁移过来的代码。真正跑大任务时
    只改这里或传参，不要在各个脚本里手写散落路径。
    """

    repo_root: Path = PIPELINE_PROJECT_ROOT
    data_root: Path = LOCAL_DATA_ROOT
    yttm_root: Path = PIPELINE_PROJECT_ROOT / "tools/yttm_int_bpe"

    @property
    def cli_dir(self) -> Path:
        return self.repo_root / "cli"

    @property
    def data_cli_dir(self) -> Path:
        return self.cli_dir / "data"

    @property
    def pretrain_cli_dir(self) -> Path:
        return self.cli_dir / "pretrain"

    @property
    def sft_cli_dir(self) -> Path:
        return self.cli_dir / "sft"

    @property
    def evaluate_cli_dir(self) -> Path:
        return self.cli_dir / "evaluate"

    @property
    def src_dir(self) -> Path:
        return self.repo_root / "src"

    @property
    def graph_tokenizer_root(self) -> Path:
        """兼容旧命名；现在指向本 pipeline 项目根目录。"""

        return self.repo_root

    @property
    def graph_root(self) -> Path:
        return self.data_root / "graphs_heavy_atom_normv2_compact_v1"

    @property
    def fddfs_root(self) -> Path:
        return self.data_root / "fddfs_heavy_atom_normv2_compact_v1"

    @property
    def bpe_binary(self) -> Path:
        return self.yttm_root / "build/int_bpe_bench"

    @property
    def bpe_tools_dir(self) -> Path:
        return self.yttm_root / "tools"

    @property
    def qwen_train_script(self) -> Path:
        return self.pretrain_cli_dir / "train_qwen3_pretrain.py"

    @property
    def fddfs_serialize_script(self) -> Path:
        return self.pretrain_cli_dir / "serialize_fddfs_full.py"


def ensure_legacy_import_paths(paths: PipelinePaths | None = None) -> PipelinePaths:
    """把当前 pipeline 项目根目录加进 import path，并返回路径配置。"""

    paths = paths or PipelinePaths()
    for item in (paths.repo_root, paths.src_dir):
        root = str(item)
        if root not in sys.path:
            sys.path.insert(0, root)
    return paths
