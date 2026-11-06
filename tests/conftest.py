"""Import paths shared by the lightweight pipeline tests."""

from __future__ import annotations

from pathlib import Path
import sys
import pytest


@pytest.fixture
def root(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def cluster_guard(monkeypatch):
    monkeypatch.setenv("MOLTOKEN_REQUIRE_WORKER", "1")


ROOT = Path(__file__).resolve().parents[1]

for path in (
    ROOT,
    ROOT / "cli" / "data",
    ROOT / "cli" / "pretrain",
    ROOT / "cli" / "sft",
    ROOT / "cli" / "evaluate",
    ROOT / "validation",
    ROOT / "ops" / "download",
    ROOT / "ops" / "environment",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
