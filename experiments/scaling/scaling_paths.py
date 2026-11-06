"""Local, overridable paths for active scaling experiments.

Historical analysis files intentionally keep their original GPFS paths as
provenance. New matrix builders and workers use the physical-host defaults
defined here instead.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _absolute_path_from_env(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    if not value.is_absolute():
        raise ValueError(f"{name} must be an absolute path, got {value}")
    return value


def safe_path_component(value: str, argument: str) -> str:
    """Return a shell-friendly single path component or fail early."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(
            f"{argument} must be one directory name using letters, digits, '.', "
            f"'_' or '-'; got {value!r}"
        )
    return value


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = _absolute_path_from_env(
    "HOLOMOL_WORKSPACE_ROOT",
    PIPELINE_ROOT,
)


DEFAULT_DATA_ROOT = _absolute_path_from_env(
    "HOLOMOL_SCALING_DATA_ROOT",
    WORKSPACE_ROOT / "scaling_data",
)
DEFAULT_RUN_ROOT = _absolute_path_from_env(
    "HOLOMOL_SCALING_RUN_ROOT",
    WORKSPACE_ROOT / "local_runtime" / "runs" / "scaling",
)
DEFAULT_STATS = _absolute_path_from_env(
    "HOLOMOL_SCALING_STATS_JSON",
    DEFAULT_DATA_ROOT / "reports" / "fddfs_mf80000_encoded_length_stats_v1.json",
)
DEFAULT_HF_DOWNLOAD_ROOT = _absolute_path_from_env(
    "HOLOMOL_SCALING_HF_DOWNLOAD_ROOT",
    DEFAULT_DATA_ROOT / "downloads_raw" / "huggingface",
)


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_HF_DOWNLOAD_ROOT",
    "DEFAULT_RUN_ROOT",
    "DEFAULT_STATS",
    "PIPELINE_ROOT",
    "WORKSPACE_ROOT",
    "safe_path_component",
]
