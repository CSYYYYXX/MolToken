#!/usr/bin/env python3
"""Formal repeated-stage Axis-B SFT for either alignment or full stage.

This entry keeps the base Axis-B model, objective, optimizer,
schedule, checkpoint format, and DDP implementation.  It changes only the
sampler so every training epoch is one complete, disjoint traversal of the
selected single-task train view.  Validation loss selects ``best`` and the
base trainer applies early stopping.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.axis_b_stagewise_epoch import (  # noqa: E402
    StagewiseFullEpochBatchSampler,
    stage_microsteps,
)
from pipeline.finetune.active_scope import ACTIVE_TRAIN_TASK_IDS  # noqa: E402


PROTOCOL = "single-task-repeated-stage-v1"
BASE_TRAINER_PATH = (
    PIPELINE_ROOT / "pipeline/finetune/internal/axis_b_trainer_engine.py"
)
_IMPL: Any | None = None
_BASE_PARSE_ARGS: Any | None = None
_BASE_SAVE_CHECKPOINT: Any | None = None
_BASE_COLLATOR: Any | None = None
_BASE_MOVE_MODEL_BATCH: Any | None = None
_BASE_EVALUATE: Any | None = None
_ACTIVE_SAMPLER: "RepeatedStageEpochBatchSampler | None" = None
_ACCUMULATION_STEPS = 1
_IN_EVALUATION = False
_CURRENT_EPOCH = 0
_FORWARD_LOCAL_INDICES: list[int] = []
_FORWARD_LOCAL_TASKS: list[str] = []
_FORWARD_LOCAL_MICROSTEPS = 0
_FORWARD_LOCAL_FINAL_BATCH_SIZE = 0


def _translate_protocol(argv: list[str]) -> tuple[list[str], bool]:
    translated = list(argv)
    found = False
    for index, value in enumerate(translated):
        if value == "--sampling-protocol":
            if index + 1 >= len(translated):
                raise ValueError("--sampling-protocol requires a value")
            if translated[index + 1] != PROTOCOL:
                raise ValueError(f"this entry accepts only {PROTOCOL}")
            translated[index + 1] = "fixed_steps"
            found = True
        elif value.startswith("--sampling-protocol="):
            supplied = value.split("=", 1)[1]
            if supplied != PROTOCOL:
                raise ValueError(f"this entry accepts only {PROTOCOL}")
            translated[index] = "--sampling-protocol=fixed_steps"
            found = True
    return translated, found


def _alignment_metadata(checkpoint: Path) -> Mapping[str, Any]:
    config_path = checkpoint / "axis_b_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = config.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"alignment checkpoint has no Axis-B metadata: {config_path}")
    return metadata


def _validate_full_resume(args: Any) -> None:
    if args.stage != "full":
        return
    checkpoint = Path(args.resume_dir).expanduser().resolve()
    metadata = _alignment_metadata(checkpoint)
    tasks = tuple(sorted(dict.fromkeys(str(task).strip() for task in args.task)))
    expected = {
        "checkpoint_role": "best",
        "stage": "alignment",
        "sampling_protocol": PROTOCOL,
        "tasks": list(tasks),
        "lineage_id": args.lineage_id or (tasks[0] if len(tasks) == 1 else None),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "pretrained_ckpt": str(Path(args.pretrained_ckpt).expanduser().resolve()),
        "text_encoder": str(Path(args.text_encoder).expanduser().resolve()),
        "molecule_condition_mode": args.molecule_condition_mode,
        "max_context": int(args.max_context),
        "strict_no_runtime_truncation": bool(
            args.strict_no_runtime_truncation
        ),
    }
    from moltoken.checkpoints import checkpoint_path_identity
    metadata = dict(metadata)
    for key in ("data_root", "pretrained_ckpt", "text_encoder"):
        expected[key] = checkpoint_path_identity(expected[key])
        if metadata.get(key) is not None:
            metadata[key] = checkpoint_path_identity(metadata[key])
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    audit = metadata.get("natural_epoch_audit")
    if not isinstance(audit, Mapping):
        mismatches["epoch_audit"] = "missing"
    else:
        audit_expected = {
            "protocol": PROTOCOL,
            "stage": "alignment",
            "complete": True,
            "actual_duplicate_records": 0,
            "actual_omitted_selected_records": 0,
            "actual_unexpected_records": 0,
        }
        for key, value in audit_expected.items():
            if audit.get(key) != value:
                mismatches[f"epoch_audit.{key}"] = {
                    "expected": value,
                    "actual": audit.get(key),
                }
        if audit.get("actual_processed_records") != audit.get("dataset_records"):
            mismatches["epoch_audit.complete_train_view"] = {
                "dataset_records": audit.get("dataset_records"),
                "actual_processed_records": audit.get("actual_processed_records"),
            }
    stats = metadata.get("stats")
    early_stopping = stats.get("early_stopping") if isinstance(stats, Mapping) else None
    if not isinstance(early_stopping, Mapping):
        mismatches["stats.early_stopping"] = "missing"
    else:
        best_epoch = early_stopping.get("best_epoch")
        if best_epoch is None or metadata.get("epoch") != best_epoch:
            mismatches["stats.early_stopping.best_epoch"] = {
                "checkpoint_epoch": metadata.get("epoch"),
                "best_epoch": best_epoch,
            }
    if mismatches:
        raise ValueError(
            "full stage must resume this experiment's completed alignment/best "
            f"checkpoint: {json.dumps(mismatches, sort_keys=True)}"
        )


def parse_args() -> Any:
    global _ACCUMULATION_STEPS
    if _BASE_PARSE_ARGS is None:
        raise RuntimeError("base Axis-B parser is not loaded")
    original = list(sys.argv)
    translated, found = _translate_protocol(original)
    if not found:
        raise ValueError(f"pass --sampling-protocol {PROTOCOL}")
    sys.argv[:] = translated
    try:
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv[:] = original
    args.sampling_protocol = PROTOCOL
    args.formal_active_scope = True
    if not args.task or len(args.task) != 1:
        raise ValueError("repeated-stage SFT requires exactly one task")
    if args.task[0] not in ACTIVE_TRAIN_TASK_IDS:
        raise ValueError(
            f"current formal SFT task must be one of {ACTIVE_TRAIN_TASK_IDS}, "
            f"got {args.task[0]!r}"
        )
    if args.epochs != 100 or args.steps_per_epoch != 0 or args.max_steps != 0:
        raise ValueError(
            "formal repeated-stage SFT requires --epochs 100, "
            "--steps-per-epoch 0, and --max-steps 0"
        )
    if args.early_stopping_patience <= 0:
        raise ValueError("repeated-stage SFT requires positive early-stopping patience")
    if args.natural_order_dir is not None:
        raise ValueError("repeated-stage SFT does not use an order sidecar")
    if (
        args.natural_partition_start,
        args.natural_partition_end,
        args.natural_partition_parts,
    ) != (0, 1, 1):
        raise ValueError("repeated-stage SFT does not use natural partitions")
    if args.save_steps != 0:
        raise ValueError("repeated-stage semantic audit supports epoch checkpoints only")
    if args.molecule_condition_mode != "direct_qwen":
        raise ValueError("repeated-stage SFT requires direct_qwen")
    if int(args.max_context) != 512:
        raise ValueError("formal repeated-stage SFT requires --max-context 512")
    if int(args.eval_batches) != 0:
        raise ValueError(
            "formal repeated-stage SFT requires --eval-batches 0 so checkpoint "
            "selection uses the complete train-derived validation split"
        )
    if not bool(args.strict_no_runtime_truncation):
        raise ValueError(
            "formal repeated-stage SFT requires --strict-no-runtime-truncation"
        )
    if args.stage == "alignment" and args.resume_dir is not None:
        raise ValueError("alignment must start from the selected pretrained flagship")
    if args.stage == "full" and args.resume_dir is None:
        raise ValueError("full must resume alignment/best from this experiment")
    _ACCUMULATION_STEPS = int(args.grad_accumulation_steps)
    _validate_full_resume(args)
    return args


def choose_epoch_microsteps(
    dataset: Any,
    batch_sizes: Mapping[str, int],
    world_size: int,
    requested: int,
) -> int:
    if int(requested) != 0:
        raise ValueError("repeated-stage SFT does not accept fixed steps")
    return stage_microsteps(
        dataset,
        batch_sizes,
        world_size,
        _ACCUMULATION_STEPS,
    )


def _reset_forward_audit(epoch: int) -> None:
    global _CURRENT_EPOCH, _FORWARD_LOCAL_MICROSTEPS
    global _FORWARD_LOCAL_FINAL_BATCH_SIZE
    _CURRENT_EPOCH = int(epoch)
    _FORWARD_LOCAL_INDICES.clear()
    _FORWARD_LOCAL_TASKS.clear()
    _FORWARD_LOCAL_MICROSTEPS = 0
    _FORWARD_LOCAL_FINAL_BATCH_SIZE = 0


class RepeatedStageEpochBatchSampler:
    """Rebuild one exact, disjoint dataset traversal for every epoch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global _ACTIVE_SAMPLER
        self.args = args
        self.kwargs = dict(kwargs)
        self.seed = int(self.kwargs.pop("seed"))
        initial_epoch = int(self.kwargs.pop("epoch"))
        self.delegate: StagewiseFullEpochBatchSampler
        _ACTIVE_SAMPLER = self
        self.set_epoch(initial_epoch)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        self.delegate = StagewiseFullEpochBatchSampler(
            *self.args,
            **self.kwargs,
            seed=self.seed + 1_000_003 * epoch,
            epoch=0,
        )
        _reset_forward_audit(epoch)

    def __len__(self) -> int:
        return len(self.delegate)

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.delegate)


class ForwardTrackingCollator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _BASE_COLLATOR is None:
            raise RuntimeError("base Axis-B collator is not loaded")
        self.delegate = _BASE_COLLATOR(*args, **kwargs)

    def __call__(self, records: Any) -> dict[str, Any]:
        output = self.delegate(records)
        output["_repeated_stage_dataset_indices"] = [
            int(record["_axis_b"]["dataset_index"]) for record in records
        ]
        return output


def move_tracked_model_batch(batch: Mapping[str, Any], device: Any) -> Any:
    global _FORWARD_LOCAL_MICROSTEPS, _FORWARD_LOCAL_FINAL_BATCH_SIZE
    if _BASE_MOVE_MODEL_BATCH is None:
        raise RuntimeError("base move_model_batch is not loaded")
    model_batch = _BASE_MOVE_MODEL_BATCH(batch, device)
    if not _IN_EVALUATION:
        indices = [int(value) for value in batch["_repeated_stage_dataset_indices"]]
        tasks = [str(value) for value in batch["task_names"]]
        if len(indices) != len(tasks):
            raise RuntimeError("forward tracking metadata has inconsistent batch lengths")
        _FORWARD_LOCAL_INDICES.extend(indices)
        _FORWARD_LOCAL_TASKS.extend(tasks)
        _FORWARD_LOCAL_MICROSTEPS += 1
        _FORWARD_LOCAL_FINAL_BATCH_SIZE = len(indices)
    return model_batch


def evaluate_without_training_counts(*args: Any, **kwargs: Any) -> Any:
    global _IN_EVALUATION
    if _BASE_EVALUATE is None:
        raise RuntimeError("base evaluate is not loaded")
    _IN_EVALUATION = True
    try:
        return _BASE_EVALUATE(*args, **kwargs)
    finally:
        _IN_EVALUATION = False


def save_audited_checkpoint(*positional: Any, **keywords: Any) -> None:
    if _BASE_SAVE_CHECKPOINT is None or _ACTIVE_SAMPLER is None or _IMPL is None:
        raise RuntimeError("repeated-stage trainer was not initialized")
    args = keywords["args"]
    rank = int(keywords["rank"])
    cumulative_step = int(keywords["step"])
    delegate = _ACTIVE_SAMPLER.delegate
    local = delegate.local_forward_audit(
        actual_indices=_FORWARD_LOCAL_INDICES,
        actual_tasks=_FORWARD_LOCAL_TASKS,
        actual_microsteps=_FORWARD_LOCAL_MICROSTEPS,
        actual_final_batch_size=_FORWARD_LOCAL_FINAL_BATCH_SIZE,
    )
    model = positional[0] if positional else keywords["model"]
    device = next(model.parameters()).device
    gather = _IMPL.all_gather_int
    local_complete_by_rank = gather(int(local["local_complete"]), device)
    actual_local_records_by_rank = gather(int(local["actual_local_records"]), device)
    actual_microsteps_by_rank = gather(int(local["actual_local_microsteps"]), device)
    actual_final_local_batch_sizes = gather(
        int(local["actual_local_final_batch_size"]), device
    )
    duplicate_records_by_rank = gather(
        int(local["actual_local_duplicate_records"]), device
    )
    omitted_records_by_rank = gather(
        int(local["actual_local_omitted_records"]), device
    )
    unexpected_records_by_rank = gather(
        int(local["actual_local_unexpected_records"]), device
    )
    local_task_counts = Counter(_FORWARD_LOCAL_TASKS)
    actual_task_records = {
        task: sum(gather(int(local_task_counts.get(task, 0)), device))
        for task in sorted(delegate.expected_task_records)
    }
    audit = delegate.plan_audit(
        stage=args.stage,
        accumulation_steps=args.grad_accumulation_steps,
    )
    expected_steps_per_epoch = int(audit["expected_optimizer_steps"])
    expected_cumulative_step = (_CURRENT_EPOCH + 1) * expected_steps_per_epoch
    actual_processed_records = sum(actual_local_records_by_rank)
    actual_duplicate_records = sum(duplicate_records_by_rank)
    actual_omitted_records = sum(omitted_records_by_rank)
    actual_unexpected_records = sum(unexpected_records_by_rank)
    complete = (
        all(value == 1 for value in local_complete_by_rank)
        and actual_processed_records == int(audit["dataset_records"])
        and actual_task_records == audit["expected_task_records"]
        and actual_duplicate_records == 0
        and actual_omitted_records == 0
        and actual_unexpected_records == 0
        and cumulative_step == expected_cumulative_step
    )
    audit.update(
        {
            "protocol": PROTOCOL,
            "semantics": "each completed epoch consumes the selected train view exactly once",
            "complete": complete,
            "epoch_index": _CURRENT_EPOCH,
            "completed_epoch": _CURRENT_EPOCH + 1,
            "maximum_epochs": int(args.epochs),
            "runtime_evidence": "dataset_indices_observed_at_training_forward_boundary",
            "actual_optimizer_steps_this_epoch": expected_steps_per_epoch,
            "actual_cumulative_optimizer_steps": cumulative_step,
            "actual_processed_records": actual_processed_records,
            "actual_unique_records": actual_processed_records if complete else None,
            "actual_task_records": actual_task_records,
            "actual_local_records_by_rank": actual_local_records_by_rank,
            "actual_microsteps_by_rank": actual_microsteps_by_rank,
            "actual_microsteps": actual_microsteps_by_rank[0],
            "actual_final_local_batch_sizes": actual_final_local_batch_sizes,
            "actual_final_global_batch_size": sum(actual_final_local_batch_sizes),
            "actual_duplicate_records": actual_duplicate_records,
            "actual_omitted_selected_records": actual_omitted_records,
            "actual_unexpected_records": actual_unexpected_records,
            "local_forward_checks_by_rank": local_complete_by_rank,
        }
    )
    if not complete:
        raise RuntimeError(
            "training forward did not consume the current epoch exactly once: "
            + json.dumps(audit, sort_keys=True)
        )
    args.natural_epoch_audit = audit
    args.repeated_stage_epoch_audit = audit
    _BASE_SAVE_CHECKPOINT(*positional, **keywords)
    if rank == 0:
        output_dir = Path(
            positional[3] if len(positional) > 3 else keywords["output_dir"]
        )
        text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
        (output_dir / "repeated_stage_epoch_audit.json").write_text(
            text, encoding="utf-8"
        )
        (Path(args.output_dir) / "latest_repeated_stage_epoch_audit.json").write_text(
            text, encoding="utf-8"
        )


def _load_impl() -> Any:
    global _IMPL, _BASE_PARSE_ARGS, _BASE_SAVE_CHECKPOINT
    global _BASE_COLLATOR, _BASE_MOVE_MODEL_BATCH, _BASE_EVALUATE
    if _IMPL is not None:
        return _IMPL
    spec = importlib.util.spec_from_file_location(
        "_axis_b_repeated_stage_base", BASE_TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Axis-B trainer: {BASE_TRAINER_PATH}")
    implementation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = implementation
    spec.loader.exec_module(implementation)
    _BASE_PARSE_ARGS = implementation.parse_args
    _BASE_SAVE_CHECKPOINT = implementation.save_checkpoint
    _BASE_COLLATOR = implementation.AxisBSFTCollator
    _BASE_MOVE_MODEL_BATCH = implementation.move_model_batch
    _BASE_EVALUATE = implementation.evaluate
    implementation.parse_args = parse_args
    implementation.choose_steps_per_epoch = choose_epoch_microsteps
    implementation.HomogeneousTaskBatchSampler = RepeatedStageEpochBatchSampler
    implementation.AxisBSFTCollator = ForwardTrackingCollator
    implementation.move_model_batch = move_tracked_model_batch
    implementation.evaluate = evaluate_without_training_counts
    implementation.save_checkpoint = save_audited_checkpoint
    _IMPL = implementation
    return implementation


def main() -> int:
    return int(_load_impl().main())


if __name__ == "__main__":
    raise SystemExit(main())
