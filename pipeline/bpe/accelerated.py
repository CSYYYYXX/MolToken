"""YouTokenToMe compact-int accelerated BPE tool entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.paths import PipelinePaths


@dataclass(frozen=True)
class BpeToolPaths:
    """当前加速 BPE 主线工具路径。"""

    root: Path
    tools: Path
    binary: Path
    build_script: Path
    shard_builder: Path
    prefix_merge_binary: Path
    prefix_merge_build_script: Path
    launch_tcp: Path
    launch_prefix_tcp: Path
    launch_fddfs: Path


def bpe_tool_paths(paths: PipelinePaths | None = None) -> BpeToolPaths:
    """返回 YouTokenToMe 加速 BPE 相关路径。"""

    paths = paths or PipelinePaths()
    tools = paths.yttm_root / "tools"
    return BpeToolPaths(
        root=paths.yttm_root,
        tools=tools,
        binary=paths.yttm_root / "build/int_bpe_bench",
        build_script=tools / "build_int_bpe_bench.sh",
        shard_builder=tools / "build_fddfs_int_shards.py",
        prefix_merge_binary=paths.yttm_root / "build/prefix_merge_int_shard",
        prefix_merge_build_script=tools / "build_prefix_merge_int_shard.sh",
        launch_tcp=tools / "launch_yttm_int_bpe_tcp.sh",
        launch_prefix_tcp=tools / "launch_yttm_int_bpe_prefix_shards_tcp.sh",
        launch_fddfs=tools / "launch_fddfs_prefix70_then_bpe64k.sh",
    )


def require_bpe_tools(paths: PipelinePaths | None = None) -> BpeToolPaths:
    """检查 BPE 主线工具是否存在。"""

    tool_paths = bpe_tool_paths(paths)
    missing = [
        path
        for path in (
            tool_paths.build_script,
            tool_paths.shard_builder,
            tool_paths.launch_tcp,
            tool_paths.launch_prefix_tcp,
            tool_paths.launch_fddfs,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("missing BPE tool files: " + ", ".join(str(path) for path in missing))
    return tool_paths
