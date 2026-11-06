#!/usr/bin/env python3
"""Synthetic checks for formal Alignment/best -> Full lineage."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "cli/sft/train_axis_b_repeated_stage_sft.py"
SPEC = importlib.util.spec_from_file_location("axis_b_repeated_stage_resume_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load repeated-stage entry: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RepeatedStageResumeTests(unittest.TestCase):
    def _args(self, checkpoint: Path, root: Path) -> Namespace:
        return Namespace(
            stage="full",
            resume_dir=checkpoint,
            task=["unconditional_moses"],
            lineage_id="unconditional_moses",
            data_root=root / "data",
            pretrained_ckpt=root / "flagship",
            text_encoder=root / "text_encoder",
            molecule_condition_mode="direct_qwen",
            max_context=512,
            strict_no_runtime_truncation=True,
        )

    def _metadata(self, root: Path, *, role: str) -> dict[str, object]:
        audit = {
            "protocol": MODULE.PROTOCOL,
            "stage": "alignment",
            "complete": True,
            "actual_duplicate_records": 0,
            "actual_omitted_selected_records": 0,
            "actual_unexpected_records": 0,
            "actual_processed_records": 100,
            "dataset_records": 100,
        }
        return {
            "checkpoint_role": role,
            "stage": "alignment",
            "epoch": 7,
            "sampling_protocol": MODULE.PROTOCOL,
            "tasks": ["unconditional_moses"],
            "lineage_id": "unconditional_moses",
            "data_root": str((root / "data").resolve()),
            "pretrained_ckpt": str((root / "flagship").resolve()),
            "text_encoder": str((root / "text_encoder").resolve()),
            "molecule_condition_mode": "direct_qwen",
            "max_context": 512,
            "strict_no_runtime_truncation": True,
            "natural_epoch_audit": audit,
            "stats": {"early_stopping": {"best_epoch": 7}},
        }

    def _write_checkpoint(self, path: Path, metadata: dict[str, object]) -> None:
        path.mkdir(parents=True)
        (path / "axis_b_config.json").write_text(
            json.dumps({"metadata": metadata}), encoding="utf-8"
        )

    def test_alignment_best_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "alignment" / "best"
            self._write_checkpoint(checkpoint, self._metadata(root, role="best"))
            MODULE._validate_full_resume(self._args(checkpoint, root))

    def test_alignment_final_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "alignment" / "final"
            self._write_checkpoint(checkpoint, self._metadata(root, role="final"))
            with self.assertRaisesRegex(ValueError, "checkpoint_role"):
                MODULE._validate_full_resume(self._args(checkpoint, root))

    def test_mismatched_best_epoch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "alignment" / "best"
            metadata = self._metadata(root, role="best")
            metadata["stats"] = {"early_stopping": {"best_epoch": 6}}
            self._write_checkpoint(checkpoint, metadata)
            with self.assertRaisesRegex(ValueError, "best_epoch"):
                MODULE._validate_full_resume(self._args(checkpoint, root))

    def test_formal_args_require_full_validation_and_strict_truncation(self) -> None:
        original_parser = MODULE._BASE_PARSE_ARGS
        original_argv = list(sys.argv)
        base = Namespace(
            task=["unconditional_moses"],
            epochs=100,
            steps_per_epoch=0,
            max_steps=0,
            early_stopping_patience=5,
            natural_order_dir=None,
            natural_partition_start=0,
            natural_partition_end=1,
            natural_partition_parts=1,
            save_steps=0,
            molecule_condition_mode="direct_qwen",
            max_context=512,
            eval_batches=0,
            strict_no_runtime_truncation=True,
            stage="alignment",
            resume_dir=None,
            grad_accumulation_steps=1,
        )
        try:
            sys.argv[:] = [
                str(MODULE_PATH),
                "--sampling-protocol",
                MODULE.PROTOCOL,
            ]
            base.eval_batches = 1
            MODULE._BASE_PARSE_ARGS = lambda: base
            with self.assertRaisesRegex(ValueError, "complete train-derived validation"):
                MODULE.parse_args()

            base.eval_batches = 0
            base.strict_no_runtime_truncation = False
            with self.assertRaisesRegex(ValueError, "strict-no-runtime-truncation"):
                MODULE.parse_args()
        finally:
            MODULE._BASE_PARSE_ARGS = original_parser
            sys.argv[:] = original_argv


if __name__ == "__main__":
    unittest.main()
