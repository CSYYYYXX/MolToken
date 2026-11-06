"""Frozen TRAIN-derived length schedules for the exposed PepBench-Gen track.

This is a project reconstruction, not an official PepBenchmark generation
leaderboard. A schedule contains prompts only and is shared by all models.
"""
from collections import Counter
import json
from pathlib import Path
import random

TASK = "pepbench_gen_length_conditioned"
PROTOCOL = "axis-pepbench-train-length-schedule-v1-20260915"
FORMAL_SAMPLES = 10_000
FORMAL_SEEDS = (42, 43, 44)
MIN_LENGTH, MAX_LENGTH = 5, 80


def make_schedule(lengths: list[int], *, seed: int, samples: int) -> dict:
    if not lengths or samples <= 0:
        raise ValueError("nonempty train lengths and positive sample count required")
    if any(type(n) is not int or not MIN_LENGTH <= n <= MAX_LENGTH for n in lengths):
        raise ValueError("train length must be an integer from 5 to 80")
    counts = Counter(lengths)
    # Sorting the empirical distribution makes the schedule independent of
    # dataset file/index iteration order, without changing its probabilities.
    empirical = [n for n in sorted(counts) for _ in range(counts[n])]
    rng = random.Random(seed)
    conditions = [str(rng.choice(empirical)) for _ in range(samples)]
    return {
        "protocol": PROTOCOL, "task": TASK, "source_split": "train",
        "seed": seed, "samples": samples, "train_records": len(lengths),
        "train_length_counts": dict(sorted(counts.items())),
        "conditions": conditions,
        "condition_counts": dict(sorted(Counter(conditions).items(), key=lambda x: int(x[0]))),
        "model_independent": True, "contains_benchmark_targets": False,
        "force_output_length": False, "failed_slots_replaced": False,
        "track": "pretraining-exposed length-matched generation; Axis free EOS",
    }


def load_schedule(path: Path, *, seed: int) -> dict:
    schedule = json.loads(path.read_text())
    if schedule["protocol"] != PROTOCOL or schedule["task"] != TASK:
        raise ValueError("not a PepBench-Gen length schedule")
    if schedule["source_split"] != "train" or schedule["contains_benchmark_targets"]:
        raise ValueError("length conditions must use TRAIN only, without benchmark targets")
    if schedule["seed"] != seed:
        raise ValueError("generation seed does not match the fixed length schedule")
    conditions = schedule["conditions"]
    if not conditions or len(conditions) != schedule["samples"]:
        raise ValueError("length schedule sample count mismatch")
    allowed = {str(n) for n in range(MIN_LENGTH, MAX_LENGTH + 1)}
    if any(condition not in allowed for condition in conditions):
        raise ValueError("condition must be a bare decimal length from 5 to 80")
    return schedule
