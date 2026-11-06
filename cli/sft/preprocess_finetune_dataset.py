#!/usr/bin/env python3
"""Public worker CLI for current MOSES/GuacaMol SFT preprocessing."""

from __future__ import annotations

from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.legacy.preprocess_finetune_dataset_compat import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
