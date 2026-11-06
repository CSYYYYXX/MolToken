#!/usr/bin/env python3
"""Run the real pretrainer and persist process-local CUDA peak memory evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args.pop(0)
    if not trainer_args:
        raise ValueError("trainer arguments are required after --")

    import torch
    from cli.pretrain import train_qwen3_pretrain

    if not torch.cuda.is_available():
        raise RuntimeError("profile smoke requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.reset_peak_memory_stats()
    started = datetime.now(timezone.utc).isoformat()
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(train_qwen3_pretrain.__file__).resolve()), *trainer_args]
        train_qwen3_pretrain.main()
        torch.cuda.synchronize()
    finally:
        sys.argv = old_argv
    device = torch.cuda.current_device()
    atomic_json(
        args.memory_report.expanduser().resolve(),
        {
            "schema_version": "domainwise_profile_smoke_cuda_memory_v1",
            "status": "complete",
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "visible_cuda_device": device,
            "device_name": torch.cuda.get_device_name(device),
            "device_total_mib": torch.cuda.get_device_properties(device).total_memory
            / (1024**2),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
