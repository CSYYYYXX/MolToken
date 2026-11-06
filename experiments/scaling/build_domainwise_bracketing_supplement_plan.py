#!/usr/bin/env python3
"""Build an explicit round of validation-only IsoFLOP supplements.

The default twelve rows append measurements to eleven already complete curves.
Later rounds accept an explicit candidate list and retain every completed
observation. They never replace an existing run and use a separate output root.
Scientific budgets and seeds come from the completed-results
CSV, model dimensions come from the model registry, and per-domain epoch sizes
come from the frozen split manifest.

This command only writes a small JSON plan and run-id list.  It does not open
training shards, materialize data, claim a GPU, profile, or launch training.
Every proposed batch shape is explicitly gated on a 50-optimizer-step worker
profile before formal execution.

An explicit boundary_probe may use two to four monotone completed observations
to extend a measured boundary. It never fits a parabola or accepts a bracket;
all normal curve fitting and acceptance still require at least five points.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

try:
    from .build_domainwise_cluster_completion_plan import (
        CONTEXT_LENGTH,
        DEFAULT_MANIFEST,
        GPFS_PROJECT_ROOT,
        VOCAB_SIZE,
        WARMUP_RATIO,
        absolute_path,
        atomic_json,
        atomic_text,
        now,
    )
    from .build_domainwise_scaling_matrix import actual_parameter_count, enrich_model_specs
    from .analyze_domainwise_adaptive_curves import analyze_adaptive_curve as analyze_curve
    from .review_phase_b_validation_bracketing import (
        resolve_curve_placement,
    )
except ImportError:  # direct script execution
    from build_domainwise_cluster_completion_plan import (
        CONTEXT_LENGTH,
        DEFAULT_MANIFEST,
        GPFS_PROJECT_ROOT,
        VOCAB_SIZE,
        WARMUP_RATIO,
        absolute_path,
        atomic_json,
        atomic_text,
        now,
    )
    from build_domainwise_scaling_matrix import actual_parameter_count, enrich_model_specs
    from analyze_domainwise_adaptive_curves import analyze_adaptive_curve as analyze_curve
    from review_phase_b_validation_bracketing import (
        resolve_curve_placement,
    )


SCHEMA_VERSION = "domainwise_bracketing_supplement_plan_v1"
PURPOSE = "adaptive_bracketing_supplement"
APPROVAL_SOURCE = "user_goal"
SUPPLEMENT_ROUND = 1
PROFILE_OPTIMIZER_STEPS = 50
MAXIMUM_EPOCH_FRACTION = 4.0
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")

# Local diagnostics extend the model family without changing the frozen matrix.
# DNA C4 and RNA C1: inspect the H256/16-layer low point before the width change.
# Local diagnostics interpolate existing width/depth families; budget-compliant
# probes may extend a curve's measured N range. The original matrix registry,
# vocabulary and model implementation are unchanged.
SUPPLEMENT_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    # User-approved 2026-09-09 peptide C0 boundary extension. Qwen supports
    # a head dimension independent of hidden size; keep the original head128.
    *(
        {
            "model_id": model_id,
            "hidden_size": 64,
            "intermediate_size": 192,
            "num_hidden_layers": layers,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
        }
        for model_id, layers in (
            ("0p0015b_h64", 6), ("0p0016b_h64", 7),
            ("0p0017b_h64", 8), ("0p0018b_h64", 10),
        )
    ),
    # RNA C5's measured minimum moved to H512/L24 on 2026-09-08.
    # Extend that same-width family; the original matrix is unchanged.
    *(
        {
            "model_id": model_id,
            "hidden_size": 512,
            "intermediate_size": 1536,
            "num_hidden_layers": layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
        }
        for model_id, layers in (("0p091b", 26), ("0p097b", 28))
    ),
    # User-approved 2026-09-07 peptide C0/C1 left-boundary extension.
    # Keep head_dim=128 and the frozen vocabulary; do not alter the original
    # 18-model matrix. H128 necessarily has one query and one KV head.
    *(
        {
            "model_id": model_id,
            "hidden_size": 128,
            "intermediate_size": 384,
            "num_hidden_layers": layers,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
        }
        for model_id, layers in (
            ("0p0024b_h128", 1),
            ("0p0026b_h128", 2),
            ("0p0028b_h128", 3),
            ("0p0031b_h128", 4),
            ("0p0035b_h128", 6),
            ("0p0039b_h128", 8),
            ("0p0043b_h128", 10),
            ("0p0048b_h128", 12),
            ("0p0052b_h128", 14),
        )
    ),
    {
        "model_id": "0p029b_deep",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 31,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p084b",
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 24,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
    },
    {
        "model_id": "0p0083b",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 5,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p029b",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 13,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p037b",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 18,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p013b",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 11,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p034b",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 16,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p015b",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 14,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p019b",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 18,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p020b_deep",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 20,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p020b_wide",
        "hidden_size": 384,
        "intermediate_size": 1152,
        "num_hidden_layers": 8,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
    {
        "model_id": "0p007b",
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
    },
)

DEFAULT_PHASE_ROOT = (
    GPFS_PROJECT_ROOT
    / "molscaling"
    / "runs"
    / "domainwise_scaling_fddfs_mf80000_v1_followup"
    / "phase_supplement"
    / "round_001"
)
DEFAULT_ACTIVE_PLAN = (
    GPFS_PROJECT_ROOT
    / "molscaling"
    / "runs"
    / "domainwise_scaling_fddfs_mf80000_v1_followup"
    / "phase_c"
    / "cluster_relaunch_plan_v2_mem70.json"
)


# The order is the approved execution priority.  DNA/C3 deliberately receives
# balanced additions on both sides; its existing residuals remain in the fit.
SUPPLEMENTS: tuple[dict[str, Any], ...] = (
    {
        "priority_tier": "A",
        "domain": "small_molecule",
        "budget": "C0",
        "model_id": "0p017b",
        "expected_status": "ambiguous",
        "expected_direction": "higher_n",
        "reason": "add the second measured point above the fitted vertex",
    },
    {
        "priority_tier": "A",
        "domain": "protein",
        "budget": "C0",
        "model_id": "0p009b",
        "expected_status": "ambiguous",
        "expected_direction": "lower_n",
        "reason": "add 9.1M as the second lower-N support point; do not repeat 13.9M",
    },
    {
        "priority_tier": "A",
        "domain": "protein",
        "budget": "C5",
        "model_id": "0p027b",
        "expected_status": "ambiguous",
        "expected_direction": "lower_n",
        "reason": "add the nearest unused lower-N support point",
    },
    {
        "priority_tier": "A",
        "domain": "dna",
        "budget": "C0",
        "model_id": "0p014b",
        "expected_status": "ambiguous",
        "expected_direction": "higher_n",
        "reason": "fill the 10.7M-to-17.0M gap with a second point above the fitted vertex",
    },
    {
        "priority_tier": "A",
        "domain": "rna",
        "budget": "C0",
        "model_id": "0p026b",
        "expected_status": "ambiguous",
        "expected_direction": "higher_n",
        "reason": "add the nearest unused higher-N support point",
    },
    {
        "priority_tier": "A",
        "domain": "rna",
        "budget": "C5",
        "model_id": "0p071b",
        "expected_status": "ambiguous",
        "expected_direction": "higher_n",
        "reason": "add the nearest registry point above the current boundary minimum",
    },
    {
        "priority_tier": "B",
        "domain": "small_molecule",
        "budget": "C2",
        "model_id": "0p022b",
        "expected_status": "unbracketed_higher_n",
        "expected_direction": "higher_n",
        "reason": "extend one registry step beyond the upper-boundary minimum",
    },
    {
        "priority_tier": "B",
        "domain": "protein",
        "budget": "C2",
        "model_id": "0p014b",
        "expected_status": "ambiguous",
        "expected_direction": "lower_n",
        "reason": "probe the nearest lower-N point; the current concave vertex is a maximum",
    },
    {
        "priority_tier": "B",
        "domain": "rna",
        "budget": "C2",
        "model_id": "0p041b",
        "expected_status": "unbracketed_higher_n",
        "expected_direction": "higher_n",
        "reason": "extend one registry step beyond the upper-boundary minimum",
    },
    {
        "priority_tier": "B",
        "domain": "small_molecule",
        "budget": "C5",
        "model_id": "0p047b",
        "expected_status": "unbracketed_higher_n",
        "expected_direction": "higher_n",
        "reason": "probe the nearest boundary extension, not the spurious 475M extrapolated vertex",
    },
    {
        "priority_tier": "C",
        "domain": "dna",
        "budget": "C3",
        "model_id": "0p009b",
        "expected_status": "ambiguous",
        "expected_direction": "ambiguous",
        "reason": "add the lowest four-epoch-compliant left diagnostic without removing residuals",
    },
    {
        "priority_tier": "C",
        "domain": "dna",
        "budget": "C3",
        "model_id": "0p026b",
        "expected_status": "ambiguous",
        "expected_direction": "ambiguous",
        "reason": "fill the right-side 22.1M-to-30.7M gap without removing residuals",
    },
)


BATCH_CANDIDATES: dict[tuple[str, str], tuple[int, int, str]] = {
    ("small_molecule", "0p015b"): (3072, 4, "small_15m_support_candidate_3072x4"),
    ("small_molecule", "0p017b"): (3072, 4, "small_le31m_3072x4"),
    ("small_molecule", "0p019b"): (3072, 4, "small_19m_support_candidate_3072x4"),
    ("small_molecule", "0p022b"): (3072, 4, "small_le31m_3072x4"),
    ("small_molecule", "0p026b"): (3072, 4, "small_le31m_3072x4"),
    ("small_molecule", "0p027b"): (3072, 4, "small_27m_support_candidate_3072x4"),
    ("small_molecule", "0p031b"): (3072, 4, "small_le31m_3072x4"),
    ("small_molecule", "0p034b"): (3072, 4, "small_34m_support_candidate_3072x4"),
    ("small_molecule", "0p037b"): (3072, 4, "small_37m_right_support_candidate_3072x4"),
    ("small_molecule", "0p041b"): (3072, 4, "small_41m_candidate_3072x4"),
    ("small_molecule", "0p047b"): (1536, 8, "small_47m_candidate_1536x8"),
    ("small_molecule", "0p053b"): (6144, 2, "small_53m_h200_candidate_6144x2"),
    ("small_molecule", "0p065b"): (1024, 12, "small_65m_candidate_1024x12"),
    ("small_molecule", "0p071b"): (1024, 12, "small_71m_candidate_1024x12"),
    ("small_molecule", "0p078b"): (768, 16, "small_78m_support_candidate_768x16"),
    ("protein", "0p009b"): (640, 4, "protein_le31m_640x4"),
    ("protein", "0p011b"): (2560, 1, "protein_11m_h200_candidate_2560x1"),
    ("protein", "0p013b"): (640, 4, "protein_13m_support_candidate_640x4"),
    ("protein", "0p014b"): (640, 4, "protein_le31m_640x4"),
    ("protein", "0p015b"): (1280, 2, "protein_15m_h256_candidate_1280x2"),
    ("protein", "0p019b"): (640, 4, "protein_19m_h256_candidate_640x4"),
    ("protein", "0p020b_deep"): (512, 5, "protein_deep20m_matched_candidate_512x5"),
    ("protein", "0p020b_wide"): (512, 5, "protein_wide20m_matched_candidate_512x5"),
    ("protein", "0p027b"): (640, 4, "protein_le31m_640x4"),
    ("protein", "0p034b"): (2560, 1, "protein_34m_single_h200_candidate_2560x1"),
    ("protein", "0p037b"): (2560, 1, "protein_37m_single_h200_candidate_2560x1"),
    ("dna", "0p009b"): (640, 4, "dna_le31m_640x4"),
    ("dna", "0p013b"): (1280, 2, "dna_13m_h256_candidate_1280x2"),
    ("dna", "0p014b"): (640, 4, "dna_le31m_640x4"),
    ("dna", "0p015b"): (1280, 2, "dna_15m_h200_candidate_1280x2"),
    ("dna", "0p019b"): (640, 4, "dna_19m_candidate_640x4"),
    ("dna", "0p020b_deep"): (640, 4, "dna_deep20m_matched_candidate_640x4"),
    ("dna", "0p020b_wide"): (640, 4, "dna_wide20m_matched_candidate_640x4"),
    ("dna", "0p026b"): (640, 4, "dna_le31m_640x4"),
    ("rna", "0p019b"): (3072, 2, "rna_19m_h200_candidate_3072x2"),
    ("rna", "0p020b_deep"): (3072, 2, "rna_deep20m_h200_candidate_3072x2"),
    ("rna", "0p020b_wide"): (3072, 2, "rna_wide20m_h200_candidate_3072x2"),
    ("rna", "0p026b"): (1536, 4, "rna_le31m_1536x4"),
    ("rna", "0p029b"): (6144, 1, "rna_29m_single_h200_candidate_6144x1"),
    ("rna", "0p029b_deep"): (1536, 4, "rna_deep29m_match_completed_wide29m_1536x4"),
    ("rna", "0p034b"): (1536, 4, "rna_34m_h384_local_candidate_1536x4"),
    ("rna", "0p037b"): (6144, 1, "rna_37m_single_h200_candidate_6144x1"),
    ("rna", "0p041b"): (1024, 6, "rna_ge41m_candidate_1024x6"),
    ("rna", "0p047b"): (6144, 1, "rna_47m_single_h200_candidate_6144x1"),
    ("rna", "0p053b"): (1024, 6, "rna_53m_candidate_1024x6"),
    ("rna", "0p065b"): (1024, 6, "rna_65m_candidate_1024x6"),
    ("rna", "0p071b"): (768, 8, "rna_71m_candidate_768x8"),
    ("rna", "0p078b"): (3072, 2, "rna_78m_h200_candidate_3072x2"),
    ("rna", "0p084b"): (3072, 2, "rna_84m_same_h512_candidate_3072x2"),
    ("rna", "0p091b"): (3072, 2, "rna_91m_same_h512_candidate_3072x2"),
    ("rna", "0p097b"): (1536, 4, "rna_97m_same_h512_candidate_1536x4"),
}

EXPECTED_EFFECTIVE_BATCH = {
    "small_molecule": 12_288,
    "protein": 2_560,
    "dna": 2_560,
    "rna": 6_144,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _positive_int(row: dict[str, str], key: str) -> int:
    text = row.get(key, "")
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}")
    value = int(text)
    if value <= 0:
        raise ValueError(f"{row.get('run_id', '<unknown>')} has non-positive {key}")
    return value


def _positive_float(row: dict[str, str], key: str) -> float:
    text = row.get(key, "")
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}")
    return value


def _integral_decimal(row: dict[str, str], key: str) -> int:
    text = row.get(key, "")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(
            f"{row.get('run_id', '<unknown>')} has invalid {key}: {text!r}"
        ) from exc
    if not value.is_finite() or value <= 0 or value != value.to_integral_value():
        raise ValueError(f"{row.get('run_id', '<unknown>')} has non-integral {key}: {text!r}")
    return int(value)


def _model_specs() -> dict[str, dict[str, Any]]:
    specs = {
        str(spec["model_id"]): spec
        for spec in enrich_model_specs(VOCAB_SIZE)
    }
    for raw in SUPPLEMENT_MODEL_SPECS:
        spec = dict(raw)
        model_id = str(spec["model_id"])
        if model_id in specs:
            raise ValueError(f"supplement model ID collides with registry: {model_id}")
        parameters = actual_parameter_count(spec, VOCAB_SIZE)
        embedding = VOCAB_SIZE * int(spec["hidden_size"])
        spec.update(parameter_count=parameters,
                    non_embedding_parameter_count=parameters - embedding,
                    embedding_parameter_count=embedding)
        specs[model_id] = spec
    return specs


def _manifest_domain_metadata(
    manifest: dict[str, Any], domain: str
) -> tuple[int, int]:
    if (
        manifest.get("schema_version") != "domain_split_raw_indices_v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("split manifest is not complete domain_split_raw_indices_v1")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or int(dataset.get("context_length", -1)) != CONTEXT_LENGTH:
        raise ValueError(f"split manifest context is not {CONTEXT_LENGTH}")
    views = manifest.get("views")
    train = views.get(domain, {}).get("train") if isinstance(views, dict) else None
    if not isinstance(train, dict):
        raise ValueError(f"split manifest has no {domain}/train view")
    records = int(train.get("records", 0))
    input_tokens = int(train.get("effective_input_tokens_with_eos", 0))
    if records <= 0 or input_tokens < records:
        raise ValueError(f"split manifest has invalid {domain}/train accounting")
    return records, input_tokens


def _group_completed_curves(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    seen_run_ids: set[str] = set()
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id or run_id in seen_run_ids:
            raise ValueError(f"empty or duplicate completed run_id: {run_id!r}")
        seen_run_ids.add(run_id)
        if row.get("status") != "complete":
            continue
        key = (row.get("domain", ""), row.get("budget", ""))
        grouped.setdefault(key, []).append(row)
    return grouped


def _curve_source(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    *,
    domain: str,
    budget: str,
    boundary_probe: bool = False,
) -> tuple[list[dict[str, str]], int, int, dict[str, Any]]:
    rows = grouped.get((domain, budget), [])
    minimum = 2 if boundary_probe else 5
    if len(rows) < minimum:
        minimum_label = "two" if boundary_probe else "five"
        raise ValueError(
            f"source curve {domain}/{budget} must have at least {minimum_label} completed points; "
            f"found {len(rows)}"
        )
    observations: list[dict[str, Any]] = []
    budgets: set[int] = set()
    seeds: set[int] = set()
    model_ids: set[str] = set()
    counts: set[int] = set()
    for row in rows:
        if row.get("metric_split") != "validation" or row.get("validation_complete_view") != "true":
            raise ValueError(f"source point is not complete validation: {row['run_id']}")
        model_id = row.get("model_id", "")
        count = _positive_int(row, "parameter_count")
        if not model_id or model_id in model_ids or count in counts:
            raise ValueError(f"source curve {domain}/{budget} repeats a model or N")
        model_ids.add(model_id)
        counts.add(count)
        budgets.add(_integral_decimal(row, "target_flops_6nd"))
        seeds.add(_positive_int(row, "seed"))
        observations.append(
            {
                "run_id": row["run_id"],
                "model_id": model_id,
                "parameter_count": count,
                "validation_loss": _positive_float(row, "validation_loss"),
            }
        )
    if len(budgets) != 1 or len(seeds) != 1:
        raise ValueError(f"source curve {domain}/{budget} mixes FLOP budgets or seeds")
    analysis = (sparse_boundary_analysis(observations) if len(observations) < 5
                else analyze_curve(observations))
    return rows, next(iter(budgets)), next(iter(seeds)), analysis


def sparse_boundary_analysis(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Authorize a measured-boundary probe, never a sparse quadratic fit."""
    if not 2 <= len(observations) < 5:
        raise ValueError("sparse boundary evidence requires two to four observations")
    points = sorted(observations, key=lambda row: row["parameter_count"])
    losses = [row["validation_loss"] for row in points]
    tolerance = max(losses) * 1e-5
    differences = [b - a for a, b in zip(losses, losses[1:])]
    if all(delta < -tolerance for delta in differences):
        direction, best = "higher_n", points[-1]
    elif all(delta > tolerance for delta in differences):
        direction, best = "lower_n", points[0]
    else:
        raise ValueError("sparse probe requires an unambiguous monotone measured boundary")
    return {
        "bracketing_status": "incomplete",
        "valid_run_count": len(points),
        "run_ids": [row["run_id"] for row in points],
        "model_ids": [row["model_id"] for row in points],
        "parameter_counts": [row["parameter_count"] for row in points],
        "validation_losses": losses,
        "discrete_argmin": dict(best),
        "quadratic_fit": None,
        "analysis_scope": "measured_boundary_probe_only_no_fit_or_acceptance",
        "placement_resolution": {
            "status": "incomplete", "direction": direction,
            "rule": "explicit_probe_beyond_monotone_observed_boundary",
            "headline_bracketing_accepted": False,
        },
    }


def _curve_observation(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis["bracketing_status"] == "incomplete":
        return {**analysis, "completed_point_count": analysis["valid_run_count"]}
    fit = analysis["quadratic_fit"]
    boundary = analysis["boundary_evidence"]
    placement = resolve_curve_placement(analysis)
    return {
        "completed_point_count": int(analysis["valid_run_count"]),
        "run_ids": list(analysis["run_ids"]),
        "model_ids": list(analysis["model_ids"]),
        "parameter_counts": list(analysis["parameter_counts"]),
        "validation_losses": list(analysis["validation_losses"]),
        "bracketing_status": analysis["bracketing_status"],
        "placement_resolution": {
            "status": placement["status"],
            "direction": placement["direction"],
            "rule": placement["rule"],
            "headline_bracketing_accepted": placement[
                "headline_bracketing_accepted"
            ],
        },
        "discrete_argmin": dict(analysis["discrete_argmin"]),
        "quadratic_fit": {
            "quadratic_coefficient_a": fit["quadratic_coefficient_a"],
            "curvature_second_derivative": fit["curvature_second_derivative"],
            "vertex_parameter_count": fit["vertex_parameter_count"],
            "r_squared": fit["r_squared"],
            "rmse": fit["rmse"],
        },
        "boundary_evidence": {
            "vertex_relation": boundary["vertex_relation"],
            "vertex_is_fitted_minimum": boundary["vertex_is_fitted_minimum"],
            "measured_points_below_vertex": boundary[
                "measured_points_below_vertex"
            ],
            "measured_points_above_vertex": boundary[
                "measured_points_above_vertex"
            ],
            "fit_meets_r_squared_threshold": boundary[
                "fit_meets_r_squared_threshold"
            ],
        },
    }


def _scheduler_steps(
    target_tokens: int,
    train_records: int,
    train_input_tokens: int,
    effective_batch: int,
) -> int:
    # This is exactly ceil(D / (mean input tokens per record * record batch))
    # without a floating-point boundary ambiguity.
    numerator = target_tokens * train_records
    denominator = train_input_tokens * effective_batch
    return (numerator + denominator - 1) // denominator


def build_plan(
    completed_rows: list[dict[str, str]],
    *,
    completed_csv_path: Path,
    active_plan: dict[str, Any],
    active_plan_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    phase_root: Path,
    supplements: Iterable[dict[str, Any]] | None = None,
    round_number: int = SUPPLEMENT_ROUND,
) -> dict[str, Any]:
    """Build the specified candidates; omitted arguments retain the round-1 plan.

    Candidate dictionaries use the same fields as ``SUPPLEMENTS``. They must
    name a registry model with an existing batch candidate and state the
    expected current curve status/direction, so a stale scientific proposal
    fails visibly instead of silently changing the proposed experiment.
    ``boundary_probe=True`` explicitly permits sparse measured-boundary
    exploration with expected_status="incomplete", without estimating a vertex.
    """

    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("round_number must be a positive integer")
    candidates = list(SUPPLEMENTS if supplements is None else supplements)
    if not candidates:
        raise ValueError("supplements must contain at least one explicit candidate")

    phase_root = absolute_path(phase_root, "--phase-root")
    manifest_path = absolute_path(manifest_path, "--domain-split-manifest")
    active_plan_path = absolute_path(active_plan_path, "--active-plan-json")
    completed_csv_path = absolute_path(completed_csv_path, "--completed-csv")
    active_rows = active_plan.get("rows")
    if not isinstance(active_rows, list) or int(active_plan.get("row_count", -1)) != len(
        active_rows
    ):
        raise ValueError("active completion plan has inconsistent rows/row_count")
    active_ids = {str(row.get("run_id", "")) for row in active_rows}
    if "" in active_ids or len(active_ids) != len(active_rows):
        raise ValueError("active completion plan has empty or duplicate run IDs")
    active_phase_root = Path(str(active_plan.get("phase_root", ""))).expanduser().resolve()
    if phase_root == active_phase_root:
        raise ValueError("supplement phase root must differ from the active completion phase root")

    grouped = _group_completed_curves(completed_rows)
    specs = _model_specs()
    plan_rows: list[dict[str, Any]] = []
    curve_cache: dict[
        tuple[str, str, bool], tuple[int, int, dict[str, Any], set[int]]
    ] = {}
    seen_supplement_ids: set[str] = set()
    for priority_rank, item in enumerate(candidates, start=1):
        domain = str(item["domain"])
        budget = str(item["budget"])
        model_id = str(item["model_id"])
        key = (domain, budget, item.get("boundary_probe", False))
        if key not in curve_cache:
            source_rows, target_flops, seed, analysis = _curve_source(
                grouped, domain=domain, budget=budget,
                boundary_probe=item.get("boundary_probe", False),
            )
            baseline_epoch_values = {
                _positive_int(row, "train_input_tokens_per_epoch")
                for row in source_rows
            }
            curve_cache[key] = (
                target_flops,
                seed,
                analysis,
                baseline_epoch_values,
            )
        target_flops, seed, analysis, baseline_epoch_values = curve_cache[key]
        placement = (analysis["placement_resolution"]
                     if analysis["bracketing_status"] == "incomplete"
                     else resolve_curve_placement(analysis))
        if analysis["bracketing_status"] != item["expected_status"]:
            raise ValueError(
                f"source curve {domain}/{budget} status changed from the approved "
                f"{item['expected_status']} to {analysis['bracketing_status']}"
            )
        if placement["direction"] != item["expected_direction"]:
            raise ValueError(
                f"source curve {domain}/{budget} direction changed from the approved "
                f"{item['expected_direction']} to {placement['direction']}"
            )
        if model_id not in specs:
            raise ValueError(f"supplement selects unknown model: {model_id}")
        spec = specs[model_id]
        if analysis["bracketing_status"] == "incomplete":
            counts = analysis["parameter_counts"]
            n = spec["parameter_count"]
            outside = (n > max(counts) if placement["direction"] == "higher_n"
                       else n < min(counts))
            if not outside:
                raise ValueError("sparse boundary probe must extend the observed N boundary")
        source_model_ids = set(analysis["model_ids"])
        if model_id in source_model_ids:
            raise ValueError(
                f"supplement duplicates an existing point: {domain}/{budget}/{model_id}"
            )

        run_id = f"{domain}_{budget}_{model_id}_s{seed}"
        if SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ValueError(f"unsafe supplement run_id: {run_id!r}")
        if run_id in seen_supplement_ids or run_id in active_ids:
            raise ValueError(f"duplicate supplement or active-plan run_id: {run_id}")
        seen_supplement_ids.add(run_id)

        train_records, train_input_tokens = _manifest_domain_metadata(manifest, domain)
        if baseline_epoch_values and baseline_epoch_values != {train_input_tokens}:
            raise ValueError(
                f"completed CSV and manifest disagree on {domain} train epoch tokens"
            )
        parameters = int(spec["parameter_count"])
        target_tokens = int(round(target_flops / (6.0 * parameters)))
        target_epoch_fraction = target_tokens / train_input_tokens
        if target_epoch_fraction > MAXIMUM_EPOCH_FRACTION:
            raise ValueError(
                f"supplement exceeds four epochs: {run_id}={target_epoch_fraction}"
            )
        microbatch, grad_accum, batch_profile = BATCH_CANDIDATES[(domain, model_id)]
        effective_batch = EXPECTED_EFFECTIVE_BATCH[domain]
        if microbatch * grad_accum != effective_batch:
            raise ValueError(f"candidate batch changes effective records for {run_id}")
        scheduler_steps = _scheduler_steps(
            target_tokens,
            train_records,
            train_input_tokens,
            effective_batch,
        )
        warmup_steps = max(1, round(scheduler_steps * WARMUP_RATIO))
        run_dir = phase_root / "runs" / run_id
        attempt_dir = run_dir / "attempt_001"
        rounded_flops = 6 * parameters * target_tokens
        plan_rows.append(
            {
                "phase": "supplement",
                "execution_phase": PURPOSE,
                "purpose": PURPOSE,
                "round": round_number,
                "priority_tier": item["priority_tier"],
                "priority_rank": priority_rank,
                "domain": domain,
                "budget": budget,
                "run_id": run_id,
                "model_id": model_id,
                "parameter_count": parameters,
                "non_embedding_parameter_count": int(
                    spec["non_embedding_parameter_count"]
                ),
                "target_input_tokens_with_eos": target_tokens,
                "target_flops_6nd": target_flops,
                "target_flops_after_integer_token_rounding": rounded_flops,
                "target_flops_relative_rounding_error": (
                    rounded_flops - target_flops
                )
                / target_flops,
                "target_epoch_fraction": target_epoch_fraction,
                "maximum_epoch_fraction": MAXIMUM_EPOCH_FRACTION,
                "four_epoch_compliant": True,
                "train_records": train_records,
                "train_input_tokens_per_epoch": train_input_tokens,
                "per_device_train_batch_size": microbatch,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_records": effective_batch,
                "scheduler_total_steps": scheduler_steps,
                "warmup_ratio": WARMUP_RATIO,
                "warmup_steps": warmup_steps,
                "seed": seed,
                "vocab_size": VOCAB_SIZE,
                "context_length": CONTEXT_LENGTH,
                "hidden_size": int(spec["hidden_size"]),
                "intermediate_size": int(spec["intermediate_size"]),
                "num_hidden_layers": int(spec["num_hidden_layers"]),
                "num_attention_heads": int(spec["num_attention_heads"]),
                "num_key_value_heads": int(spec["num_key_value_heads"]),
                "head_dim": int(spec["head_dim"]),
                "world_size": 1,
                "random_initialization": True,
                "init_checkpoint": None,
                "restart_from_step": 0,
                "source_status": "adaptive_supplement",
                "attempt_number": 1,
                "attempt_name": "attempt_001",
                "run_dir": str(run_dir),
                "attempt_dir": str(attempt_dir),
                "source_batch_profile": batch_profile,
                "batch_candidate_only": True,
                "need_profile": True,
                "profile_optimizer_steps": PROFILE_OPTIMIZER_STEPS,
                "profile_status": "required_before_formal_training",
                "claim_policy": "worker_empty_slot",
                "supplement_reason": item["reason"],
                "source_curve_observation": _curve_observation(analysis),
                "approval_source": APPROVAL_SOURCE,
            }
        )

    if len(plan_rows) != len(candidates) or len(seen_supplement_ids) != len(
        candidates
    ):
        raise ValueError("supplement plan is incomplete or duplicated")
    if any(
        key in row
        for row in plan_rows
        for key in (
            "global_gpu_index",
            "global_lane_index",
            "lane_within_gpu",
            "lane_sequence_index",
            "device_index",
            "assigned_gpu",
        )
    ):
        raise ValueError("supplement rows must not bind GPUs or lanes")
    tier_counts = Counter(str(row["priority_tier"]) for row in plan_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "status": "ready_for_50_step_profile",
        "phase": "supplement",
        "purpose": PURPOSE,
        "round": round_number,
        "approval_source": APPROVAL_SOURCE,
        "phase_root": str(phase_root),
        "domain_split_manifest": str(manifest_path),
        "source_completed_csv": str(completed_csv_path),
        "source_active_plan": str(active_plan_path),
        "active_plan_row_count": len(active_rows),
        "rows": plan_rows,
        "row_count": len(plan_rows),
        "priority_tier_counts": dict(sorted(tier_counts.items())),
        "execution_policy": {
            "independent_single_process_runs": True,
            "world_size_per_run": 1,
            "random_initialization": True,
            "restart_from_step_zero": True,
            "worker_empty_slot_claim": True,
            "gpu_or_lane_binding_in_plan": False,
            "all_batch_candidates_require_profile": True,
            "profile_optimizer_steps": PROFILE_OPTIMIZER_STEPS,
            "formal_training_requires_successful_profile": True,
            "adaptive_stop_after_each_curve_is_bracketed": True,
            "append_observations_without_deleting_existing_points": True,
            "maximum_epoch_fraction": MAXIMUM_EPOCH_FRACTION,
        },
        "analysis_policy": {
            "metric": "final_validation.loss",
            "test_data_accessed": False,
            "fit_x": "log10(parameter_count)",
            "fit": "ordinary_least_squares_quadratic_over_all_observations",
            "minimum_observations": 5,
            "loss_relative_tolerance": 1e-5,
            "minimum_r_squared": 0.8,
            "vertex_margin_fraction": 0.02,
            "minimum_points_each_side_of_vertex": 2,
            "require_positive_curvature": True,
            "dna_c3_delete_residual_points": False,
            "dna_c3_relax_thresholds": False,
            "registry_boundary_policy": "require_new_same_family_point_design_if_registry_exhausted",
        },
        "training_control": {
            "learning_rate": 2e-4,
            "minimum_learning_rate_ratio": 0.1,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "maximum_gradient_norm": 1.0,
            "optimizer": "adamw-fused",
            "precision": "bf16",
            "model_implementation": "packed-xformers",
            "attention_backend": "xformers_memory_efficient_block_diagonal_causal",
            "gradient_checkpointing": False,
            "torch_compile": False,
            "deepspeed": False,
            "save_every_steps": 0,
            "full_validation_at_end": True,
        },
    }


def main() -> int:
    script = Path(__file__).resolve()
    default_completed = (
        script.parents[4]
        / "doc"
        / "results"
        / "domainwise145_tables_20260906"
        / "completed_runs.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--completed-csv", type=Path, default=default_completed)
    parser.add_argument("--active-plan-json", type=Path, default=DEFAULT_ACTIVE_PLAN)
    parser.add_argument("--domain-split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase-root", type=Path,
                        help="Output run root; defaults to phase_supplement/round_NNN")
    parser.add_argument("--round-number", type=int, default=SUPPLEMENT_ROUND)
    parser.add_argument("--supplements-json", type=Path,
                        help="JSON list of explicit candidate dictionaries; defaults to round-1 candidates")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--run-id-file", type=Path, required=True)
    args = parser.parse_args()

    completed_path = args.completed_csv.expanduser().resolve()
    active_plan_path = args.active_plan_json.expanduser().resolve()
    manifest_path = args.domain_split_manifest.expanduser().resolve()
    output_path = args.out_json.expanduser().resolve()
    if output_path == active_plan_path:
        raise ValueError("--out-json must not overwrite --active-plan-json")
    supplements = load_json(args.supplements_json) if args.supplements_json else None
    if supplements is not None and not isinstance(supplements, list):
        raise ValueError("--supplements-json must contain a JSON list")
    plan = build_plan(
        read_csv(completed_path),
        completed_csv_path=completed_path,
        active_plan=load_json(active_plan_path),
        active_plan_path=active_plan_path,
        manifest=load_json(manifest_path),
        manifest_path=manifest_path,
        phase_root=args.phase_root or DEFAULT_PHASE_ROOT.parent / f"round_{args.round_number:03d}",
        supplements=supplements,
        round_number=args.round_number,
    )
    run_id_path = args.run_id_file.expanduser().resolve()
    atomic_json(output_path, plan)
    atomic_text(
        run_id_path,
        "".join(f"{row['run_id']}\n" for row in plan["rows"]),
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan": str(output_path),
                "run_id_file": str(run_id_path),
                "rows": plan["row_count"],
                "priority_tier_counts": plan["priority_tier_counts"],
                "all_rows_need_50_step_profile": all(
                    row["need_profile"]
                    and row["profile_optimizer_steps"] == PROFILE_OPTIMIZER_STEPS
                    for row in plan["rows"]
                ),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
