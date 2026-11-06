#!/usr/bin/env python3
"""Build the design-stage domain-wise scaling-law matrix.

This builder consumes the exact domain audit and writes a scientific design
matrix.  It intentionally leaves GPU layout, batch size, and max_steps unset;
those fields must be frozen only after per-shape memory/throughput probes and
the domain-filtered loader are available.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

try:
    from .scaling_paths import DEFAULT_DATA_ROOT
except ImportError:  # direct script execution
    from scaling_paths import DEFAULT_DATA_ROOT


VOCAB_SIZE = 17_207
N_PRIOR_COEFFICIENT = 10 ** (-2.1959)
N_PRIOR_EXPONENT = 0.5430
TARGET_MULTIPLIERS = (0.65, 0.82, 1.00, 1.25, 1.60)
STANDARD_DOMAINS = ("small_molecule", "protein", "dna", "rna")


def model_spec(
    model_id: str,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
    attention_heads: int,
    key_value_heads: int,
    legacy_document_params: int,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": key_value_heads,
        "head_dim": 128,
        "legacy_document_parameter_count": legacy_document_params,
    }


MODEL_SPECS = [
    model_spec("0p006b", 256, 768, 2, 2, 1, 5_979_136),
    model_spec("0p008b", 256, 768, 4, 2, 1, 7_553_024),
    model_spec("0p009b", 256, 768, 6, 2, 1, 9_126_912),
    model_spec("0p011b", 256, 768, 8, 2, 1, 10_700_800),
    model_spec("0p014b", 256, 768, 12, 2, 1, 13_848_576),
    model_spec("0p017b", 256, 768, 16, 2, 1, 16_996_352),
    model_spec("0p022b", 384, 1152, 9, 3, 1, 22_097_664),
    model_spec("0p026b", 384, 1152, 11, 3, 1, 25_539_840),
    model_spec("0p027b", 384, 1152, 12, 3, 1, 27_260_928),
    model_spec("0p031b", 384, 1152, 14, 3, 1, 30_703_104),
    model_spec("0p041b", 384, 1152, 20, 3, 1, 41_029_632),
    model_spec("0p047b", 512, 1536, 12, 4, 2, 46_571_520),
    model_spec("0p053b", 512, 1536, 14, 4, 2, 52_865_024),
    model_spec("0p059b", 512, 1536, 16, 4, 2, 59_158_528),
    model_spec("0p065b", 512, 1536, 18, 4, 2, 65_452_032),
    model_spec("0p071b", 512, 1536, 20, 4, 2, 71_745_536),
    model_spec("0p078b", 512, 1536, 22, 4, 2, 78_039_040),
    model_spec("0p098b", 768, 2304, 12, 6, 3, 98_168_832),
]


# The three larger-domain starts are the reviewed draft values.  The small-
# molecule range is shifted to the narrow interval where the existing model
# family can provide two prior-left points at C0 and still keep two prior-left
# points under four epochs at C5.  The common ratio remains exactly 1.5.
BUDGET_STARTS = {
    "small_molecule": 5.25e16,
    "protein": 2.00e17,
    "dna": 8.00e16,
    "rna": 1.60e17,
}

PEPTIDE_MODEL_IDS = ("0p006b", "0p009b", "0p014b", "0p022b", "0p041b")
PEPTIDE_EPOCH_FRACTIONS = (0.25, 0.5, 1.0, 2.0, 4.0)


def actual_parameter_count(spec: dict[str, Any], vocab_size: int) -> int:
    """Count the exact tied-embedding PackedXformersQwen3 parameters."""
    hidden = int(spec["hidden_size"])
    intermediate = int(spec["intermediate_size"])
    layers = int(spec["num_hidden_layers"])
    heads = int(spec["num_attention_heads"])
    kv_heads = int(spec["num_key_value_heads"])
    head_dim = int(spec["head_dim"])
    embedding = vocab_size * hidden
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    projections = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    qk_norms = 2 * head_dim
    mlp = 3 * hidden * intermediate
    decoder_norms = 2 * hidden
    final_norm = hidden
    return int(embedding + layers * (projections + qk_norms + mlp + decoder_norms) + final_norm)


def enrich_model_specs(vocab_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in MODEL_SPECS:
        spec = dict(raw)
        params = actual_parameter_count(spec, vocab_size)
        embedding = vocab_size * int(spec["hidden_size"])
        spec["parameter_count"] = params
        spec["non_embedding_parameter_count"] = params - embedding
        spec["embedding_parameter_count"] = embedding
        spec["parameter_count_delta_vs_legacy_document"] = params - int(
            spec["legacy_document_parameter_count"]
        )
        out.append(spec)
    return out


def prior_parameter_count(flops: float) -> float:
    return N_PRIOR_COEFFICIENT * flops**N_PRIOR_EXPONENT


def choose_models(
    specs: list[dict[str, Any]], flops: float, train_tokens_per_epoch: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = prior_parameter_count(flops)
    minimum = flops / (24.0 * train_tokens_per_epoch)
    eligible = [s for s in specs if int(s["parameter_count"]) >= minimum]
    if len(eligible) < len(TARGET_MULTIPLIERS):
        raise ValueError(f"only {len(eligible)} eligible models for C={flops:g}")
    targets = [prior * multiplier for multiplier in TARGET_MULTIPLIERS]

    def score(combo: tuple[dict[str, Any], ...]) -> float:
        return sum(
            math.log(int(spec["parameter_count"]) / target) ** 2
            for spec, target in zip(combo, targets)
        )

    combos = list(itertools.combinations(eligible, len(TARGET_MULTIPLIERS)))
    bracketed = [
        combo
        for combo in combos
        if sum(int(s["parameter_count"]) < prior for s in combo) >= 2
        and sum(int(s["parameter_count"]) > prior for s in combo) >= 2
    ]
    pool = bracketed if bracketed else combos
    chosen = list(min(pool, key=score))
    left = sum(int(s["parameter_count"]) < prior for s in chosen)
    right = sum(int(s["parameter_count"]) > prior for s in chosen)
    return chosen, {
        "prior_parameter_count": prior,
        "minimum_parameter_count_four_epoch": minimum,
        "target_parameter_counts": targets,
        "prior_left_points": left,
        "prior_right_points": right,
        "prior_bracketed_two_each_side": left >= 2 and right >= 2,
        "selection_objective": "minimum squared log error to ordered multiplier targets, constrained to >=2 points on each side when feasible",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact_number(value: float) -> str:
    return f"{value:.4e}"


def write_markdown(
    path: Path,
    audit_path: Path,
    audit: dict[str, Any],
    curves: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
) -> None:
    lines = [
        "# Domain-wise scaling matrix proposal v1",
        "",
        f"Exact audit: `{audit_path}`",
        "",
        "This is a design matrix, not yet a runnable launch matrix. GPU layout, batch size, gradient accumulation, measured tokens/step and max_steps remain gated on smoke probes.",
        "",
        "## Exact domain audit",
        "",
        "| domain | records | train records | train input tokens U | val loss tokens | test loss tokens | body p50/p95/p99 | truncated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("small_molecule", "peptide", "protein", "dna", "rna"):
        d = audit["domains"][name]
        train = d["splits"]["train"]
        val = d["splits"]["validation"]
        test = d["splits"]["test"]
        length = d["body_length"]
        lines.append(
            f"| {name} | {d['records']:,} | {train['records']:,} | "
            f"{train['effective_input_tokens_with_eos']:,} | {val['supervised_causal_loss_tokens']:,} | "
            f"{test['supervised_causal_loss_tokens']:,} | {length['p50']}/{length['p95']}/{length['p99']} | "
            f"{d['truncated_records']:,} |"
        )

    lines += [
        "",
        "## Standard-domain IsoFLOP curves",
        "",
        "| domain | budget | target FLOPs | N prior | selected total parameters (M) | epoch range | prior L/R | phase |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for curve in curves:
        values = ", ".join(f"{x / 1e6:.3f}" for x in curve["selected_parameter_counts"])
        lines.append(
            f"| {curve['domain']} | {curve['budget_label']} | {curve['target_flops_6nd']:.6e} | "
            f"{curve['prior_parameter_count'] / 1e6:.3f}M | {values} | "
            f"{curve['min_epoch_fraction']:.2f}-{curve['max_epoch_fraction']:.2f} | "
            f"{curve['prior_left_points']}/{curve['prior_right_points']} | {curve['execution_phase']} |"
        )

    peptide_rows = [row for row in rows if row["domain"] == "peptide"]
    lines += [
        "",
        "## Peptide data-limited surface",
        "",
        f"25 runs: models `{', '.join(PEPTIDE_MODEL_IDS)}` crossed with epochs `{', '.join(f'{x:g}' for x in PEPTIDE_EPOCH_FRACTIONS)}`.",
        f"The resulting FLOPs range is {compact_number(min(r['target_flops_6nd'] for r in peptide_rows))} to {compact_number(max(r['target_flops_6nd'] for r in peptide_rows))}.",
        "",
        "## Parameter-count correction",
        "",
        "The historical matrix formula omitted each layer's Qwen3 q_norm and k_norm (2 x head_dim parameters). The matrix uses actual total parameters below.",
        "",
        "| model | actual total | non-embedding | delta vs historical formula |",
        "|---|---:|---:|---:|",
    ]
    for spec in specs:
        lines.append(
            f"| {spec['model_id']} | {spec['parameter_count']:,} | "
            f"{spec['non_embedding_parameter_count']:,} | +{spec['parameter_count_delta_vs_legacy_document']:,} |"
        )
    lines += [
        "",
        "## Freeze gates before launch",
        "",
        "1. Materialize/freeze domain+split raw-index manifests and add loader support for them.",
        "2. Accumulate input tokens (including EOS) separately from causal-loss tokens.",
        "3. Preserve the real final short batch and stop against the token target.",
        "4. Run memory/throughput probes for every distinct shape before filling batch and max_steps.",
        "5. Run Phase B (C0/C2/C5) first and use validation only to verify observed vertex bracketing.",
        "6. Select the eight extra-seed runs only after the middle-budget validation vertices are known.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    default_audit = DEFAULT_DATA_ROOT / "reports" / "domainwise_compact_audit_seed20260902.json"
    default_prefix = DEFAULT_DATA_ROOT / "reports" / "domainwise_scaling_matrix_proposal_v1"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-json", type=Path, default=default_audit)
    ap.add_argument("--out-prefix", type=Path, default=default_prefix)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    audit_path = args.audit_json.expanduser().resolve()
    out_prefix = args.out_prefix.expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["split_assignment"]["status"] != "proposed_freeze_candidate":
        raise ValueError("unexpected split-assignment status")
    specs = enrich_model_specs(args.vocab_size)
    by_id = {str(spec["model_id"]): spec for spec in specs}
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []

    for domain in STANDARD_DOMAINS:
        train_stats = audit["domains"][domain]["splits"]["train"]
        train_tokens = int(train_stats["effective_input_tokens_with_eos"])
        train_records = int(train_stats["records"])
        for budget_index in range(6):
            budget_label = f"C{budget_index}"
            target_flops = BUDGET_STARTS[domain] * (1.5**budget_index)
            chosen, selection = choose_models(specs, target_flops, train_tokens)
            phase = "B_bracketing" if budget_index in {0, 2, 5} else "C_completion"
            curve_rows: list[dict[str, Any]] = []
            for slot, (multiplier, spec) in enumerate(zip(TARGET_MULTIPLIERS, chosen)):
                params = int(spec["parameter_count"])
                target_tokens = max(1, int(round(target_flops / (6.0 * params))))
                rounded_flops = 6.0 * params * target_tokens
                epoch_fraction = target_tokens / train_tokens
                row = {
                    "matrix_status": "design_only_not_runnable",
                    "experiment_family": "domainwise_scaling_fddfs_mf80000_v1",
                    "execution_phase": phase,
                    "domain": domain,
                    "domain_id": int(audit["domains"][domain]["domain_id"]),
                    "budget_label": budget_label,
                    "budget_index": budget_index,
                    "target_flops_6nd": target_flops,
                    "model_slot": slot,
                    "target_multiplier_vs_prior": multiplier,
                    "model_id": spec["model_id"],
                    "parameter_count": params,
                    "non_embedding_parameter_count": int(spec["non_embedding_parameter_count"]),
                    "legacy_document_parameter_count": int(spec["legacy_document_parameter_count"]),
                    "prior_parameter_count": selection["prior_parameter_count"],
                    "minimum_parameter_count_four_epoch": selection[
                        "minimum_parameter_count_four_epoch"
                    ],
                    "actual_multiplier_vs_prior": params / selection["prior_parameter_count"],
                    "target_input_tokens_with_eos": target_tokens,
                    "target_flops_after_integer_token_rounding": rounded_flops,
                    "target_flops_relative_rounding_error": (rounded_flops - target_flops)
                    / target_flops,
                    "target_epoch_fraction": epoch_fraction,
                    "four_epoch_compliant": epoch_fraction <= 4.0,
                    "train_records": train_records,
                    "train_input_tokens_per_epoch": train_tokens,
                    "validation_loss_tokens": int(
                        audit["domains"][domain]["splits"]["validation"][
                            "supervised_causal_loss_tokens"
                        ]
                    ),
                    "test_loss_tokens": int(
                        audit["domains"][domain]["splits"]["test"][
                            "supervised_causal_loss_tokens"
                        ]
                    ),
                    "vocab_size": args.vocab_size,
                    "context_length": int(audit["token_accounting"]["context_length"]),
                    "hidden_size": int(spec["hidden_size"]),
                    "intermediate_size": int(spec["intermediate_size"]),
                    "num_hidden_layers": int(spec["num_hidden_layers"]),
                    "num_attention_heads": int(spec["num_attention_heads"]),
                    "num_key_value_heads": int(spec["num_key_value_heads"]),
                    "head_dim": int(spec["head_dim"]),
                    "gpu_count": None,
                    "per_device_train_batch_size": None,
                    "gradient_accumulation_steps": None,
                    "measured_input_tokens_per_step": None,
                    "max_steps": None,
                    "warmup_steps": None,
                    "seed": 42,
                    "independent_random_initialization": True,
                }
                rows.append(row)
                curve_rows.append(row)

            curves.append(
                {
                    "domain": domain,
                    "budget_label": budget_label,
                    "execution_phase": phase,
                    "target_flops_6nd": target_flops,
                    **selection,
                    "selected_model_ids": [str(x["model_id"]) for x in chosen],
                    "selected_parameter_counts": [int(x["parameter_count"]) for x in chosen],
                    "min_epoch_fraction": min(float(x["target_epoch_fraction"]) for x in curve_rows),
                    "max_epoch_fraction": max(float(x["target_epoch_fraction"]) for x in curve_rows),
                }
            )

    peptide_train = audit["domains"]["peptide"]["splits"]["train"]
    peptide_tokens = int(peptide_train["effective_input_tokens_with_eos"])
    peptide_records = int(peptide_train["records"])
    for model_id in PEPTIDE_MODEL_IDS:
        spec = by_id[model_id]
        params = int(spec["parameter_count"])
        for fraction in PEPTIDE_EPOCH_FRACTIONS:
            target_tokens = max(1, int(round(peptide_tokens * fraction)))
            target_flops = 6.0 * params * target_tokens
            rows.append(
                {
                    "matrix_status": "design_only_not_runnable",
                    "experiment_family": "domainwise_scaling_fddfs_mf80000_v1",
                    "execution_phase": "D_peptide_surface",
                    "domain": "peptide",
                    "domain_id": int(audit["domains"]["peptide"]["domain_id"]),
                    "budget_label": f"E{str(fraction).replace('.', 'p')}",
                    "budget_index": None,
                    "target_flops_6nd": target_flops,
                    "model_slot": None,
                    "target_multiplier_vs_prior": None,
                    "model_id": spec["model_id"],
                    "parameter_count": params,
                    "non_embedding_parameter_count": int(spec["non_embedding_parameter_count"]),
                    "legacy_document_parameter_count": int(spec["legacy_document_parameter_count"]),
                    "prior_parameter_count": prior_parameter_count(target_flops),
                    "minimum_parameter_count_four_epoch": target_flops
                    / (24.0 * peptide_tokens),
                    "actual_multiplier_vs_prior": params / prior_parameter_count(target_flops),
                    "target_input_tokens_with_eos": target_tokens,
                    "target_flops_after_integer_token_rounding": target_flops,
                    "target_flops_relative_rounding_error": 0.0,
                    "target_epoch_fraction": fraction,
                    "four_epoch_compliant": fraction <= 4.0,
                    "train_records": peptide_records,
                    "train_input_tokens_per_epoch": peptide_tokens,
                    "validation_loss_tokens": int(
                        audit["domains"]["peptide"]["splits"]["validation"][
                            "supervised_causal_loss_tokens"
                        ]
                    ),
                    "test_loss_tokens": int(
                        audit["domains"]["peptide"]["splits"]["test"][
                            "supervised_causal_loss_tokens"
                        ]
                    ),
                    "vocab_size": args.vocab_size,
                    "context_length": int(audit["token_accounting"]["context_length"]),
                    "hidden_size": int(spec["hidden_size"]),
                    "intermediate_size": int(spec["intermediate_size"]),
                    "num_hidden_layers": int(spec["num_hidden_layers"]),
                    "num_attention_heads": int(spec["num_attention_heads"]),
                    "num_key_value_heads": int(spec["num_key_value_heads"]),
                    "head_dim": int(spec["head_dim"]),
                    "gpu_count": None,
                    "per_device_train_batch_size": None,
                    "gradient_accumulation_steps": None,
                    "measured_input_tokens_per_step": None,
                    "max_steps": None,
                    "warmup_steps": None,
                    "seed": 42,
                    "independent_random_initialization": True,
                }
            )

    if len(rows) != 145:
        raise AssertionError(f"expected 145 primary rows, got {len(rows)}")
    if not all(bool(row["four_epoch_compliant"]) for row in rows):
        raise AssertionError("matrix contains a row above four epochs")
    if not all(bool(curve["prior_bracketed_two_each_side"]) for curve in curves):
        raise AssertionError("a standard-domain curve does not bracket the prior")

    json_path = Path(str(out_prefix) + ".json")
    csv_path = Path(str(out_prefix) + ".csv")
    md_path = Path(str(out_prefix) + ".md")
    metadata = {
        "matrix_version": "domainwise_scaling_matrix_proposal_v1",
        "status": "design_only_not_runnable",
        "audit_json": str(audit_path),
        "split_assignment": audit["split_assignment"],
        "token_accounting": audit["token_accounting"],
        "vocab_size": args.vocab_size,
        "flops_definition": "6 * actual_total_parameters * actual_nonpadding_input_tokens_including_EOS",
        "n_prior": {
            "coefficient": N_PRIOR_COEFFICIENT,
            "exponent": N_PRIOR_EXPONENT,
            "usage": "placement only; not a fitted domain-wise result",
        },
        "target_multipliers": list(TARGET_MULTIPLIERS),
        "budget_ratio": 1.5,
        "budget_starts": BUDGET_STARTS,
        "small_molecule_budget_note": (
            "C0 shifted from the 4e16 discussion draft to 5.25e16 so the existing model family "
            "has at least two prior-left points at both C0 and C5 while all points remain <=4 epochs"
        ),
        "primary_rows": len(rows),
        "standard_domain_rows": 120,
        "peptide_surface_rows": 25,
        "deferred_extra_seed_rows": 8,
        "launch_gates": [
            "freeze/materialize domain-split manifests",
            "domain-filtered loader with separate test support",
            "actual input-token accumulator including EOS",
            "real final short batch and token-budget stop",
            "per-shape memory and throughput probes",
            "fill gpu_count, batch, grad_accum, measured_tokens_per_step, max_steps, warmup_steps",
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {"metadata": metadata, "model_specs": specs, "curves": curves, "rows": rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, rows)
    write_markdown(md_path, audit_path, audit, curves, rows, specs)
    print(
        json.dumps(
            {
                "matrix_json": str(json_path),
                "matrix_csv": str(csv_path),
                "summary_md": str(md_path),
                "rows": len(rows),
                "curves": len(curves),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
