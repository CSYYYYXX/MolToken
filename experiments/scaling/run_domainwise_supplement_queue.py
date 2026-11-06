#!/usr/bin/env python3
"""Fill released domain-wise baseline lanes with approved supplement runs.

The scheduler allocation and its CUDA visibility already exist when this
program starts.  ``--device-indices`` therefore contains allocation-local
CUDA ordinals.  The queue never sets ``CUDA_VISIBLE_DEVICES`` and passes the
selected ordinal to the single-process trainer through ``LOCAL_RANK`` via the
existing :func:`run_one` implementation.

One queue thread is created for every lane in the original 39-run baseline
plan owned by this allocation.  A lane is released only after its original
run has a semantically valid complete marker.  This keeps baseline plus
supplement concurrency at or below the original per-GPU lane count.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, TextIO

try:
    from .domainwise_gpu_admission import probe_gpus
    from .run_domainwise_cluster_completion import (
        COMPLETION_SCHEMA,
        append_jsonl,
        load_json,
        now,
        parse_device_indices,
        run_one,
        single_process_environment,
        validate_completion,
        validate_manifest,
        visible_cuda_device_count,
    )
    from .run_domainwise_cluster_completion_v2 import validate_relaunch_plan
    from .run_domainwise_cluster_profile import run_wave
except ImportError:  # direct script execution
    from domainwise_gpu_admission import probe_gpus
    from run_domainwise_cluster_completion import (
        COMPLETION_SCHEMA,
        append_jsonl,
        load_json,
        now,
        parse_device_indices,
        run_one,
        single_process_environment,
        validate_completion,
        validate_manifest,
        visible_cuda_device_count,
    )
    from run_domainwise_cluster_completion_v2 import validate_relaunch_plan
    from run_domainwise_cluster_profile import run_wave


SUPPLEMENT_SCHEMA = "domainwise_bracketing_supplement_plan_v1"
PROFILE_STEPS = 50
MIN_FREE_MEMORY_MIB = 30 * 1024
MAX_USED_MEMORY_FRACTION = 0.75
MAX_TEMPERATURE_C = 80.0
FORMAL_START_POLL_SECONDS = 2.0
_TIER_ORDER = {"A": 0, "B": 1, "C": 2}
_FORBIDDEN_PLACEMENT_FIELDS = {
    "assigned_gpu",
    "device_index",
    "global_gpu_index",
    "global_lane_index",
    "lane_sequence_index",
    "lane_within_gpu",
    "logical_gpu",
    "worker_slot",
}
_TRAINING_IDENTITY_FIELDS = (
    "phase",
    "domain",
    "budget",
    "run_id",
    "model_id",
    "parameter_count",
    "non_embedding_parameter_count",
    "target_input_tokens_with_eos",
    "target_flops_6nd",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_records",
    "scheduler_total_steps",
    "warmup_ratio",
    "warmup_steps",
    "seed",
    "vocab_size",
    "context_length",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "world_size",
    "random_initialization",
    "init_checkpoint",
    "restart_from_step",
    "source_status",
    "attempt_number",
    "attempt_name",
    "run_dir",
    "attempt_dir",
)


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    plan_path: Path


@dataclass(frozen=True)
class BaselineLane:
    device_index: int
    global_gpu_index: int
    lane_within_gpu: int
    rows: tuple[dict[str, Any], ...]

    @property
    def label(self) -> str:
        return f"g{self.global_gpu_index}_lane{self.lane_within_gpu}"


@dataclass
class FormalOutcome:
    error: BaseException | None = None


@dataclass
class Claim:
    candidate: Candidate
    handle: TextIO
    path: Path
    formal_thread: threading.Thread | None = None
    formal_outcome: FormalOutcome | None = None

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> Claim:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _positive_int(value: Any, field: str, run_id: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer for {run_id}")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field} must be a positive integer for {run_id}")
    return number


def validate_supplement_plan(plan: dict[str, Any], plan_path: Path) -> list[dict[str, Any]]:
    if plan.get("schema_version") != SUPPLEMENT_SCHEMA:
        raise ValueError(
            f"unsupported supplement plan schema in {plan_path}: "
            f"{plan.get('schema_version')!r}"
        )
    if (
        plan.get("status") != "ready_for_50_step_profile"
        or plan.get("phase") != "supplement"
        or plan.get("purpose") != "adaptive_bracketing_supplement"
    ):
        raise ValueError(f"supplement plan is not ready for execution: {plan_path}")
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows or int(plan.get("row_count", -1)) != len(rows):
        raise ValueError(f"supplement rows/row_count are inconsistent: {plan_path}")

    phase_root = Path(str(plan["phase_root"])).expanduser().resolve()
    seen_run_ids: set[str] = set()
    seen_ranks: set[int] = set()
    seen_coordinates: set[tuple[str, str, str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"supplement row is not an object: {plan_path}")
        run_id = str(row.get("run_id", ""))
        if not run_id or "/" in run_id or run_id in seen_run_ids:
            raise ValueError(f"empty, unsafe, or duplicate supplement run_id: {run_id!r}")
        seen_run_ids.add(run_id)
        if _FORBIDDEN_PLACEMENT_FIELDS.intersection(row):
            raise ValueError(f"supplement row must not bind a GPU/lane: {run_id}")
        if (
            row.get("source_status") != "adaptive_supplement"
            or int(row.get("attempt_number", -1)) != 1
            or row.get("attempt_name") != "attempt_001"
            or int(row.get("restart_from_step", -1)) != 0
            or row.get("init_checkpoint") is not None
            or row.get("random_initialization") is not True
            or int(row.get("world_size", -1)) != 1
        ):
            raise ValueError(f"fresh supplement attempt contract changed for {run_id}")
        if (
            row.get("need_profile") is not True
            or int(row.get("profile_optimizer_steps", -1)) != PROFILE_STEPS
            or row.get("profile_status") != "required_before_formal_training"
            or row.get("claim_policy") != "worker_empty_slot"
        ):
            raise ValueError(f"50-step profile gate changed for {run_id}")
        tier = str(row.get("priority_tier", ""))
        rank = _positive_int(row.get("priority_rank"), "priority_rank", run_id)
        if tier not in _TIER_ORDER or rank in seen_ranks:
            raise ValueError(f"invalid tier or duplicate priority rank for {run_id}")
        seen_ranks.add(rank)

        domain = str(row.get("domain", ""))
        budget = str(row.get("budget", ""))
        model_id = str(row.get("model_id", ""))
        seed = int(row.get("seed", -1))
        coordinate = (domain, budget, model_id, seed)
        if not all((domain, budget, model_id)) or seed < 0 or coordinate in seen_coordinates:
            raise ValueError(f"invalid or duplicate scientific coordinate for {run_id}")
        seen_coordinates.add(coordinate)

        micro = _positive_int(row.get("per_device_train_batch_size"), "batch size", run_id)
        accum = _positive_int(row.get("gradient_accumulation_steps"), "grad accum", run_id)
        effective = _positive_int(row.get("effective_batch_records"), "effective batch", run_id)
        if micro * accum != effective:
            raise ValueError(f"effective record batch changed for {run_id}")
        steps = _positive_int(row.get("scheduler_total_steps"), "scheduler steps", run_id)
        warmup_ratio = float(row.get("warmup_ratio", math.nan))
        if (
            not math.isfinite(warmup_ratio)
            or not 0.0 <= warmup_ratio < 1.0
            or int(row.get("warmup_steps", -1)) != max(1, round(steps * warmup_ratio))
        ):
            raise ValueError(f"warmup schedule changed for {run_id}")
        parameters = _positive_int(row.get("parameter_count"), "parameter count", run_id)
        target_tokens = _positive_int(
            row.get("target_input_tokens_with_eos"), "target tokens", run_id
        )
        target_flops = _positive_int(row.get("target_flops_6nd"), "target FLOPs", run_id)
        if abs(6 * parameters * target_tokens - target_flops) > 3 * parameters:
            raise ValueError(f"token and 6ND targets disagree for {run_id}")
        for field in (
            "vocab_size",
            "context_length",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            _positive_int(row.get(field), field, run_id)

        run_dir = Path(str(row.get("run_dir", ""))).expanduser().resolve()
        attempt_dir = Path(str(row.get("attempt_dir", ""))).expanduser().resolve()
        if run_dir != phase_root / "runs" / run_id or attempt_dir != run_dir / "attempt_001":
            raise ValueError(f"supplement output routing changed for {run_id}")
    return rows


def plan_paths_from_args(
    supplement_plan_jsons: Iterable[Path] | None,
    plan_list_json: Path | None,
) -> list[Path]:
    direct = list(supplement_plan_jsons or [])
    if bool(direct) == bool(plan_list_json):
        raise ValueError(
            "provide either repeated --supplement-plan-json or one --plan-list-json"
        )
    if direct:
        paths = direct
    else:
        list_path = plan_list_json.expanduser().resolve()  # type: ignore[union-attr]
        payload = load_json(list_path)
        entries = payload.get("plans") if isinstance(payload, dict) else payload
        if not isinstance(entries, list) or not entries or any(
            not isinstance(item, str) or not item for item in entries
        ):
            raise ValueError(f"plan list must be a non-empty string list: {list_path}")
        paths = [Path(item) if Path(item).is_absolute() else list_path.parent / item for item in entries]
    resolved = [path.expanduser().resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("supplement plan paths contain duplicates")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    return resolved


def load_candidates(plan_paths: Iterable[Path]) -> tuple[list[Candidate], Path]:
    candidates: list[Candidate] = []
    manifests: set[Path] = set()
    seen_run_ids: set[str] = set()
    seen_coordinates: set[tuple[str, str, str, int]] = set()
    for plan_path in plan_paths:
        plan = load_json(plan_path)
        rows = validate_supplement_plan(plan, plan_path)
        manifests.add(Path(str(plan["domain_split_manifest"])).expanduser().resolve())
        for row in rows:
            run_id = str(row["run_id"])
            coordinate = (
                str(row["domain"]),
                str(row["budget"]),
                str(row["model_id"]),
                int(row["seed"]),
            )
            if run_id in seen_run_ids or coordinate in seen_coordinates:
                raise ValueError(f"duplicate supplement row across plans: {run_id}")
            seen_run_ids.add(run_id)
            seen_coordinates.add(coordinate)
            candidates.append(Candidate(row=row, plan_path=plan_path))
    if len(manifests) != 1:
        raise ValueError("all supplement plans must use one domain split manifest")
    candidates.sort(
        key=lambda item: (
            _TIER_ORDER[str(item.row["priority_tier"])],
            int(item.row["priority_rank"]),
            int(item.row["target_input_tokens_with_eos"]),
            str(item.row["run_id"]),
        )
    )
    return candidates, next(iter(manifests))


def build_owned_baseline_lanes(
    baseline_rows: Iterable[dict[str, Any]],
    *,
    device_indices: list[int],
    global_gpu_offset: int,
) -> list[BaselineLane]:
    global_to_device = {
        global_gpu_offset + position: device
        for position, device in enumerate(device_indices)
    }
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in baseline_rows:
        global_gpu = int(row["global_gpu_index"])
        if global_gpu not in global_to_device:
            continue
        lane = int(row["lane_within_gpu"])
        grouped.setdefault((global_gpu, lane), []).append(row)
    lanes = [
        BaselineLane(
            device_index=global_to_device[global_gpu],
            global_gpu_index=global_gpu,
            lane_within_gpu=lane,
            rows=tuple(
                sorted(rows, key=lambda row: int(row.get("lane_sequence_index", 0)))
            ),
        )
        for (global_gpu, lane), rows in sorted(grouped.items())
    ]
    if not lanes:
        raise ValueError("this allocation owns no lanes in the baseline plan")
    return lanes


def validated_complete(row: dict[str, Any], manifest: dict[str, Any]) -> bool:
    run_id = str(row["run_id"])
    run_dir = Path(str(row["run_dir"])).expanduser().resolve()
    attempt_dir = Path(str(row["attempt_dir"])).expanduser().resolve()
    completion_path = run_dir / "complete.json"
    if not completion_path.is_file():
        return False
    marker = load_json(completion_path)
    if (
        marker.get("schema_version") != COMPLETION_SCHEMA
        or marker.get("status") != "complete"
        or marker.get("run_id") != run_id
        or Path(str(marker.get("attempt_dir", ""))).expanduser().resolve() != attempt_dir
    ):
        raise ValueError(f"completion marker identity mismatch for {run_id}")
    validate_completion(attempt_dir=attempt_dir, row=row, manifest=manifest)
    return True


def baseline_lane_released(lane: BaselineLane, manifest: dict[str, Any]) -> bool:
    for row in lane.rows:
        attempt_dir = Path(str(row["attempt_dir"])).expanduser().resolve()
        if (attempt_dir / "failed.json").is_file():
            raise RuntimeError(f"baseline lane run failed: {row['run_id']}")
        if not validated_complete(row, manifest):
            return False
    # run_one writes complete.json only after its trainer subprocess exits and
    # full validation succeeds, so a validated marker is also the lane-release
    # signal; no PID-name heuristic is needed.
    return True


def baseline_completion_counts(
    lanes: Iterable[BaselineLane], manifest: dict[str, Any]
) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {}
    for lane in lanes:
        current = counts.setdefault(
            lane.global_gpu_index,
            {"mapped_runs": 0, "validated_complete_runs": 0, "released_lanes": 0},
        )
        current["mapped_runs"] += len(lane.rows)
        completed = 0
        for row in lane.rows:
            attempt_dir = Path(str(row["attempt_dir"])).expanduser().resolve()
            if (attempt_dir / "failed.json").is_file():
                raise RuntimeError(f"baseline lane run failed: {row['run_id']}")
            completed += int(validated_complete(row, manifest))
        current["validated_complete_runs"] += completed
        current["released_lanes"] += int(completed == len(lane.rows))
    return counts


def admission_block_reason(probe: dict[str, Any]) -> str | None:
    if probe.get("status") != "ok":
        return f"gpu_probe_unavailable: {probe.get('error')}"
    inventory = probe["inventory"]
    total = int(inventory["memory_total_mib"])
    used = int(inventory["memory_used_mib"])
    temperature = float(inventory["temperature_c"])
    if used / total >= MAX_USED_MEMORY_FRACTION:
        return f"gpu_memory_fraction={used / total:.4f} >= {MAX_USED_MEMORY_FRACTION:.2f}"
    if total - used < MIN_FREE_MEMORY_MIB:
        return f"gpu_free_memory_mib={total - used} < {MIN_FREE_MEMORY_MIB}"
    if temperature >= MAX_TEMPERATURE_C:
        return f"gpu_temperature_c={temperature:.1f} >= {MAX_TEMPERATURE_C:.1f}"
    return None


def try_claim(candidate: Candidate, claims_dir: Path, metadata: dict[str, Any]) -> Claim | None:
    claims_dir.mkdir(parents=True, exist_ok=True)
    path = claims_dir / f"{candidate.row['run_id']}.supplement_queue.claim"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "claimed_at": now(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "run_id": candidate.row["run_id"],
                **metadata,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    handle.flush()
    return Claim(candidate=candidate, handle=handle, path=path)


def acquire_allocation_lock(status_dir: Path, global_gpu_offset: int) -> TextIO:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"supplement_queue_g{global_gpu_offset}.allocation.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"another supplement queue already owns global GPU offset "
            f"{global_gpu_offset}: {path}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "locked_at": now(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "global_gpu_offset": global_gpu_offset,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    handle.flush()
    return handle


def has_train_update(attempt_dir: Path) -> bool:
    path = attempt_dir / "train_log.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        return False
    size = path.stat().st_size
    offset = max(0, size - 65536)
    with path.open("rb") as handle:
        if offset:
            handle.seek(offset - 1)
            preceding = handle.read(1)
        else:
            preceding = b"\n"
        handle.seek(offset)
        tail = handle.read()
    lines = tail.splitlines(keepends=True)
    if offset and preceding != b"\n" and lines:
        lines = lines[1:]
    if tail and not tail.endswith(b"\n") and lines:
        lines = lines[:-1]
    found = False
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("event") == "train":
            found = True
    return found


def formal_context_ready(
    *,
    attempt_dir: Path,
) -> str | None:
    if has_train_update(attempt_dir):
        return "first_train_update"
    return None


class SupplementQueue:
    def __init__(
        self,
        *,
        candidates: list[Candidate],
        lanes: list[BaselineLane],
        python: Path,
        pipeline_root: Path,
        manifest: dict[str, Any],
        manifest_path: Path,
        output_dir: Path,
        status_dir: Path,
        poll_seconds: float,
        num_workers: int,
        global_gpu_offset: int,
        plan_list_json: Path | None = None,
    ) -> None:
        self.candidates = candidates
        self.lanes = lanes
        self.python = python
        self.pipeline_root = pipeline_root
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.status_dir = status_dir
        self.poll_seconds = poll_seconds
        self.num_workers = num_workers
        self.plan_list_json = plan_list_json
        self.stop = threading.Event()
        self.status_lock = threading.Lock()
        self.candidate_lock = threading.Lock()
        self.gpu_locks = {lane.device_index: threading.Lock() for lane in lanes}
        suffix = "_".join(str(index) for index in sorted({lane.device_index for lane in lanes}))
        self.status_path = status_dir / f"supplement_queue_g{global_gpu_offset}_d{suffix}.jsonl"
        self.claims_dir = status_dir / "claims"

    def event(self, event: str, **fields: Any) -> None:
        append_jsonl(self.status_path, {"event": event, **fields}, self.status_lock)

    def _candidate_complete(self, candidate: Candidate) -> bool:
        return validated_complete(candidate.row, self.manifest)

    @staticmethod
    def _training_identity(candidate: Candidate) -> tuple[Any, ...]:
        return tuple(candidate.row.get(field) for field in _TRAINING_IDENTITY_FIELDS)

    def refresh_candidates(self) -> list[Candidate]:
        if self.plan_list_json is None:
            return list(self.candidates)
        with self.candidate_lock:
            paths = plan_paths_from_args(None, self.plan_list_json)
            refreshed, manifest_path = load_candidates(paths)
            if manifest_path != self.manifest_path:
                raise ValueError("reloaded supplement plan changed the domain split manifest")
            previous = {str(candidate.row["run_id"]): candidate for candidate in self.candidates}
            current = {str(candidate.row["run_id"]): candidate for candidate in refreshed}
            missing = sorted(set(previous) - set(current))
            if missing:
                raise ValueError(f"reloaded plan list removed known supplement runs: {missing}")
            changed = sorted(
                run_id
                for run_id, old in previous.items()
                if self._training_identity(old)
                != self._training_identity(current[run_id])
            )
            if changed:
                raise ValueError(f"reloaded plan list changed known run identities: {changed}")
            added = sorted(set(current) - set(previous))
            if added:
                validate_manifest(
                    manifest_path,
                    [current[run_id].row for run_id in added],
                )
            self.candidates = refreshed
            if added:
                self.event(
                    "plan_list_reloaded",
                    plan_list_json=str(self.plan_list_json),
                    added_run_ids=added,
                    candidate_count=len(refreshed),
                )
            return list(refreshed)

    def _claim_next(self, lane: BaselineLane) -> Claim | None:
        busy: list[str] = []
        for candidate in self.refresh_candidates():
            if self._candidate_complete(candidate):
                continue
            claim = try_claim(
                candidate,
                self.claims_dir,
                {
                    "device_index": lane.device_index,
                    "global_gpu_index": lane.global_gpu_index,
                    "baseline_lane": lane.lane_within_gpu,
                },
            )
            if claim is None:
                busy.append(str(candidate.row["run_id"]))
                continue
            if self._candidate_complete(candidate):
                claim.close()
                continue
            attempt_dir = Path(str(candidate.row["attempt_dir"])).expanduser().resolve()
            if attempt_dir.exists():
                claim.close()
                raise FileExistsError(
                    f"incomplete supplement attempt already exists: {attempt_dir}"
                )
            if busy:
                self.event(
                    "claim_contention_continued",
                    lane=lane.label,
                    busy_run_ids=busy,
                    claimed_run_id=candidate.row["run_id"],
                )
            return claim
        if busy:
            self.event(
                "claim_contention_no_candidate",
                lane=lane.label,
                busy_run_ids=busy,
            )
        return None

    def _run_profile(self, candidate: Candidate, lane: BaselineLane) -> Path:
        run_id = str(candidate.row["run_id"])
        profile_dir = self.output_dir / "profile" / run_id
        profile_dir.parent.mkdir(parents=True, exist_ok=True)
        self.event(
            "profile_start",
            run_id=run_id,
            lane=lane.label,
            device_index=lane.device_index,
            optimizer_steps=PROFILE_STEPS,
            profile_dir=str(profile_dir),
        )
        args = SimpleNamespace(
            python=self.python,
            pipeline_root=self.pipeline_root,
            manifest=self.manifest_path,
            num_workers=self.num_workers,
            steps=PROFILE_STEPS,
        )
        result = run_wave(
            args,
            self.manifest,
            [candidate.row],
            [lane.device_index],
            1,
            profile_dir,
        )
        runs = result.get("results")
        if (
            result.get("concurrency") != 1
            or not isinstance(runs, list)
            or len(runs) != 1
            or runs[0].get("run_id") != run_id
            or int(runs[0].get("device", -1)) != lane.device_index
            or len(runs[0].get("updates", [])) != PROFILE_STEPS
        ):
            raise RuntimeError(f"50-step profile evidence is incomplete for {run_id}")
        self.event(
            "profile_complete",
            run_id=run_id,
            lane=lane.label,
            profile_dir=str(profile_dir),
        )
        return profile_dir

    def _start_formal(
        self,
        candidate: Candidate,
        lane: BaselineLane,
    ) -> tuple[threading.Thread, FormalOutcome]:
        run_id = str(candidate.row["run_id"])
        outcome = FormalOutcome()

        def target() -> None:
            try:
                run_one(
                    row=candidate.row,
                    device_index=lane.device_index,
                    python=self.python,
                    pipeline_root=self.pipeline_root,
                    manifest=self.manifest,
                    manifest_path=self.manifest_path,
                    num_workers=self.num_workers,
                    plan_path=candidate.plan_path,
                    status_path=self.status_path,
                    status_lock=self.status_lock,
                )
            except BaseException as exc:
                outcome.error = exc

        self.event(
            "formal_start",
            run_id=run_id,
            lane=lane.label,
            device_index=lane.device_index,
        )
        thread = threading.Thread(target=target, name=f"formal-{run_id}")
        thread.start()
        attempt_dir = Path(str(candidate.row["attempt_dir"])).expanduser().resolve()
        started = time.monotonic()
        last_wait_report = 0.0
        while thread.is_alive():
            elapsed = time.monotonic() - started
            reason = formal_context_ready(
                attempt_dir=attempt_dir,
            )
            if reason is not None:
                self.event(
                    "formal_initialized",
                    run_id=run_id,
                    lane=lane.label,
                    evidence=reason,
                    elapsed_seconds=elapsed,
                )
                return thread, outcome
            if elapsed - last_wait_report >= 60.0:
                self.event(
                    "formal_initialization_wait",
                    run_id=run_id,
                    lane=lane.label,
                    elapsed_seconds=elapsed,
                )
                last_wait_report = elapsed
            time.sleep(FORMAL_START_POLL_SECONDS)
        thread.join()
        if outcome.error is not None:
            raise RuntimeError(f"formal trainer failed before initialization: {run_id}") from outcome.error
        self.event(
            "formal_initialized",
            run_id=run_id,
            lane=lane.label,
            evidence="formal_completed_before_readiness_poll",
            elapsed_seconds=time.monotonic() - started,
        )
        return thread, outcome

    def _wait_for_post_profile_admission(
        self, candidate: Candidate, lane: BaselineLane
    ) -> None:
        last_reason: str | None = None
        last_reported = 0.0
        started = time.monotonic()
        while True:
            if self.stop.is_set():
                raise RuntimeError(
                    f"queue stopped before formal start: {candidate.row['run_id']}"
                )
            probe = probe_gpus([lane.device_index])[lane.device_index]
            reason = admission_block_reason(probe)
            if reason is None:
                return
            elapsed = time.monotonic() - started
            if reason != last_reason or elapsed - last_reported >= 60.0:
                self.event(
                    "post_profile_admission_wait",
                    run_id=candidate.row["run_id"],
                    lane=lane.label,
                    reason=reason,
                    elapsed_seconds=elapsed,
                )
                last_reason = reason
                last_reported = elapsed
            self.stop.wait(min(self.poll_seconds, 5.0))

    def _run_claimed(self, claim: Claim, lane: BaselineLane) -> None:
        candidate = claim.candidate
        self._run_profile(candidate, lane)
        self._wait_for_post_profile_admission(candidate, lane)
        formal_thread, outcome = self._start_formal(candidate, lane)
        # The caller releases the per-GPU admission lock after this method
        # returns.  The cross-worker claim remains held while formal training
        # runs and this lane cannot claim another run until join completes.
        claim.formal_thread = formal_thread
        claim.formal_outcome = outcome

    def run_lane(self, lane: BaselineLane) -> None:
        last_wait_state: str | None = None
        released_reported = False
        while not self.stop.is_set():
            if not baseline_lane_released(lane, self.manifest):
                state = "waiting_baseline"
                if state != last_wait_state:
                    self.event(
                        state,
                        lane=lane.label,
                        baseline_run_ids=[row["run_id"] for row in lane.rows],
                    )
                    last_wait_state = state
                self.stop.wait(self.poll_seconds)
                continue

            if not released_reported:
                self.event(
                    "baseline_lane_released",
                    lane=lane.label,
                    baseline_run_ids=[row["run_id"] for row in lane.rows],
                )
                released_reported = True

            with self.gpu_locks[lane.device_index]:
                if self.stop.is_set():
                    return
                probe = probe_gpus([lane.device_index])[lane.device_index]
                blocked = admission_block_reason(probe)
                if blocked is not None:
                    state = f"admission_blocked:{blocked}"
                    if state != last_wait_state:
                        self.event(
                            "admission_blocked",
                            lane=lane.label,
                            device_index=lane.device_index,
                            reason=blocked,
                        )
                        last_wait_state = state
                    claim = None
                else:
                    claim = self._claim_next(lane)
                    if claim is None:
                        state = "queue_empty_watch"
                        if state != last_wait_state:
                            self.event(state, lane=lane.label)
                            last_wait_state = state
                    else:
                        last_wait_state = None
                        try:
                            self._run_claimed(claim, lane)
                        except BaseException as exc:
                            self.event(
                                "supplement_failure",
                                run_id=claim.candidate.row["run_id"],
                                lane=lane.label,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            claim.close()
                            raise

            if claim is None:
                self.stop.wait(self.poll_seconds)
                continue

            formal_thread = claim.formal_thread
            outcome = claim.formal_outcome
            if formal_thread is None or outcome is None:
                claim.close()
                raise RuntimeError(
                    f"formal supplement was not started: {claim.candidate.row['run_id']}"
                )
            formal_thread.join()
            try:
                if outcome.error is not None:
                    raise RuntimeError(
                        f"formal supplement run failed: {claim.candidate.row['run_id']}"
                    ) from outcome.error
                if not self._candidate_complete(claim.candidate):
                    raise RuntimeError(
                        f"formal run returned without valid completion: "
                        f"{claim.candidate.row['run_id']}"
                    )
                self.event(
                    "supplement_complete",
                    run_id=claim.candidate.row["run_id"],
                    lane=lane.label,
                )
            except BaseException as exc:
                self.event(
                    "supplement_failure",
                    run_id=claim.candidate.row["run_id"],
                    lane=lane.label,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                claim.close()

    def run(self) -> None:
        baseline_counts = baseline_completion_counts(self.lanes, self.manifest)
        self.event(
            "queue_start",
            candidate_count=len(self.candidates),
            lane_count=len(self.lanes),
            lanes=[lane.label for lane in self.lanes],
            min_free_memory_mib=MIN_FREE_MEMORY_MIB,
            max_used_memory_fraction=MAX_USED_MEMORY_FRACTION,
            max_temperature_c=MAX_TEMPERATURE_C,
            baseline_by_global_gpu={str(key): value for key, value in baseline_counts.items()},
        )
        threads = [
            threading.Thread(target=self._lane_target, args=(lane,), name=lane.label)
            for lane in self.lanes
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        errors = [getattr(thread, "queue_error", None) for thread in threads]
        first_error = next((error for error in errors if error is not None), None)
        if first_error is not None:
            raise RuntimeError("supplement queue stopped after a lane failure") from first_error

    def _lane_target(self, lane: BaselineLane) -> None:
        thread = threading.current_thread()
        try:
            self.run_lane(lane)
        except BaseException as exc:
            thread.queue_error = exc  # type: ignore[attr-defined]
            self.stop.set()
            self.event(
                "lane_failure",
                lane=lane.label,
                error=f"{type(exc).__name__}: {exc}",
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    plans = parser.add_mutually_exclusive_group(required=True)
    plans.add_argument("--supplement-plan-json", type=Path, action="append")
    plans.add_argument("--plan-list-json", type=Path)
    parser.add_argument("--baseline-plan-json", type=Path, required=True)
    parser.add_argument("--device-indices", required=True)
    parser.add_argument("--global-gpu-offset", type=int, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--pipeline-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--num-workers-per-trainer", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.global_gpu_offset not in {0, 4, 6}:
        raise ValueError("--global-gpu-offset must be one of 0, 4, or 6")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.num_workers_per_trainer < 0:
        raise ValueError("--num-workers-per-trainer must be non-negative")

    python = args.python.expanduser().absolute()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    status_dir = args.status_dir.expanduser().resolve()
    baseline_path = args.baseline_plan_json.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    if not pipeline_root.is_dir():
        raise FileNotFoundError(pipeline_root)
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    device_indices = parse_device_indices(args.device_indices)
    visible_devices = visible_cuda_device_count(python)
    if max(device_indices) >= visible_devices:
        raise RuntimeError(
            f"allocation exposes {visible_devices} CUDA devices, cannot use {device_indices}"
        )

    baseline = load_json(baseline_path)
    baseline_rows = validate_relaunch_plan(baseline)
    placement = baseline["placement"]
    global_gpu_count = int(placement["global_gpu_count"])
    if args.global_gpu_offset + len(device_indices) > global_gpu_count:
        raise ValueError("allocation GPU slice exceeds baseline global GPU count")

    plan_paths = plan_paths_from_args(args.supplement_plan_json, args.plan_list_json)
    candidates, manifest_path = load_candidates(plan_paths)
    baseline_manifest = Path(str(baseline["domain_split_manifest"])).expanduser().resolve()
    if manifest_path != baseline_manifest:
        raise ValueError("supplement and baseline plans use different domain split manifests")
    manifest = validate_manifest(manifest_path, [*baseline_rows, *(c.row for c in candidates)])
    lanes = build_owned_baseline_lanes(
        baseline_rows,
        device_indices=device_indices,
        global_gpu_offset=args.global_gpu_offset,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    queue = SupplementQueue(
        candidates=candidates,
        lanes=lanes,
        python=python,
        pipeline_root=pipeline_root,
        manifest=manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        status_dir=status_dir,
        poll_seconds=args.poll_seconds,
        num_workers=args.num_workers_per_trainer,
        global_gpu_offset=args.global_gpu_offset,
        plan_list_json=(
            args.plan_list_json.expanduser().resolve()
            if args.plan_list_json is not None
            else None
        ),
    )
    allocation_lock = acquire_allocation_lock(status_dir, args.global_gpu_offset)
    try:
        queue.run()
    finally:
        allocation_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
