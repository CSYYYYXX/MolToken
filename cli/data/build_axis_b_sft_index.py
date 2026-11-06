#!/usr/bin/env python3
"""Build or reuse the Axis-B JSONL random-access index on a worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.finetune.axis_b_data import build_axis_b_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = build_axis_b_index(
        args.data_root,
        index_dir=args.index_dir,
        force=args.force,
    )
    print(json.dumps(index.metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
