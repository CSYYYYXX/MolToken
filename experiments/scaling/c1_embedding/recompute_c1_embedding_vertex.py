#!/usr/bin/env python3
"""C1: embedding-convention sensitivity under the *canonical* A1 estimator.

The canonical estimator (A1.3) is: per isoFLOP curve, drop points whose loss
exceeds that curve's minimum by more than 0.05 (this removes the 1.7B/4B
points whose global batch is 2x everyone else's, see C3), fit a parabola in
(log10 N, loss), take the vertex; then OLS log10 N* on log10 C.

That trim is not optional -- with it, `final` loss reproduces the documented
headline a = 0.543046 / R2 = 0.9871 exactly. An earlier version of this script
omitted the trim and reported a = 0.5878 -> 0.7140 (+21.5%); those numbers are
wrong and must not be quoted.

Two variants are reported:
  variant A (primary): x-axis stays the nominal isoFLOP budget C, only the
      reported N changes to N_nonembed. C = 6*N_total*D is the correct measured
      compute because the tied embedding does participate in the output matmul.
  variant B (reference only): fully switch to Kaplan-style C = 6*N_nonembed*D.
      This breaks the isoFLOP grouping (within-budget spread up to 89.5%), so
      it is not a well-posed refit of the same experiment.

Pure CPU. No training, no checkpoint loading.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CSV_PATH = Path(
    "./storage/molscaling"
    "/runs/scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1"
    "/plots/all_fixed_flops_10curves/all_fixed_flops_results.csv"
)

RUN_TO_RUN_SIGMA = 0.00202   # A1.1, measured from base/g1/g3 repeats
OUTLIER_THRESHOLD = 0.05     # A1.3, per-curve loss excess above curve minimum
N_BOOTSTRAP = 4000
RANDOM_SEED = 20260728


def qwen3_non_embedding(cfg: dict) -> tuple[int, int]:
    """Return (non_embedding_params, embedding_params).

    Validated to reproduce run_config parameter_count exactly for all 23 model
    sizes (zero residual) by the sibling check_c1_param_formula.py.
    """
    args = cfg.get("args", cfg)
    vocab = args["vocab_size"]
    hidden = args["hidden_size"]
    inter = args["intermediate_size"]
    layers = args["num_hidden_layers"]
    heads = args["num_attention_heads"]
    kv_heads = args["num_key_value_heads"]
    head_dim = args["head_dim"]

    attn = (
        hidden * heads * head_dim
        + hidden * kv_heads * head_dim
        + hidden * kv_heads * head_dim
        + heads * head_dim * hidden
    )
    qk_norm = 2 * head_dim
    mlp = 3 * hidden * inter
    layer_norms = 2 * hidden

    non_embedding = layers * (attn + qk_norm + mlp + layer_norms) + hidden
    return non_embedding, vocab * hidden


def load_runs() -> list[dict]:
    records: list[dict] = []
    config_cache: dict[str, dict] = {}
    with open(CSV_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            out_dir = row["out_dir"]
            if out_dir not in config_cache:
                cfg_path = Path(out_dir) / "run_config.json"
                if not cfg_path.is_file():
                    raise FileNotFoundError(cfg_path)
                config_cache[out_dir] = json.loads(cfg_path.read_text())
            cfg = config_cache[out_dir]

            non_embedding, embedding = qwen3_non_embedding(cfg)
            n_total = int(row["parameter_count"])
            tokens = int(row["observed_train_loss_tokens_global"])
            records.append(
                {
                    "target_label": row["target_label"],
                    "target_flops": float(row["target_flops_6nd"]),
                    "model_id": row["model_id"],
                    "n_total": n_total,
                    "n_nonembed": non_embedding,
                    "embedding_fraction": embedding / n_total,
                    "tokens_D": tokens,
                    "flops_nonembed": 6.0 * non_embedding * tokens,
                    "epoch_fraction": float(row["epoch_fraction"]),
                    "loss_final": float(row["training_loss"]),
                    "loss_trailing": float(row["trailing_train_loss"]),
                }
            )
    return records


def parabola_vertex(log_n: np.ndarray, loss: np.ndarray) -> float | None:
    design = np.column_stack([np.ones_like(log_n), log_n, log_n**2])
    coefficients, *_ = np.linalg.lstsq(design, loss, rcond=None)
    _, c1, c2 = coefficients
    if not np.isfinite(c2) or c2 <= 0.0:
        return None
    return float(-c1 / (2.0 * c2))


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_centered = x - x.mean()
    slope = float(x_centered @ (y - y.mean()) / (x_centered @ x_centered))
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    total = y - y.mean()
    return slope, intercept, float(1.0 - (residual @ residual) / (total @ total))


def fit_exponent(
    records: list[dict],
    losses: np.ndarray,
    *,
    n_key: str,
    flops_key: str,
    trim: float = OUTLIER_THRESHOLD,
) -> tuple[float, float, dict[str, dict]] | None:
    """Canonical A1 estimator. `losses` is aligned with `records` by index."""
    by_budget: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_budget[record["target_label"]].append(index)

    log_c: list[float] = []
    log_n_star: list[float] = []
    per_curve: dict[str, dict] = {}

    ordered = sorted(by_budget, key=lambda k: records[by_budget[k][0]]["target_flops"])
    for label in ordered:
        indices = by_budget[label]
        log_n = np.array([math.log10(records[i][n_key]) for i in indices])
        curve_loss = losses[indices]

        keep = (curve_loss - curve_loss.min()) <= trim
        if keep.sum() < 3:
            continue

        vertex = parabola_vertex(log_n[keep], curve_loss[keep])
        if vertex is None:
            continue

        if flops_key == "target_flops":
            budget = records[indices[0]]["target_flops"]
        else:
            kept = [indices[j] for j in np.flatnonzero(keep)]
            budget = float(np.mean([records[i][flops_key] for i in kept]))

        log_c.append(math.log10(budget))
        log_n_star.append(vertex)
        per_curve[label] = {
            "vertex_log10": vertex,
            "n_star": 10.0**vertex,
            "n_points": len(indices),
            "n_kept": int(keep.sum()),
            "log_n_min": float(log_n[keep].min()),
            "log_n_max": float(log_n[keep].max()),
        }

    if len(log_c) < 3:
        return None
    slope, _, r_squared = ols(np.array(log_c), np.array(log_n_star))
    return slope, r_squared, per_curve


def summarize(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
    return {
        "median": float(median),
        "ci_low": float(low),
        "ci_high": float(high),
        "sd": float(values.std()),
        "iqr_sd": float((np.percentile(values, 75) - np.percentile(values, 25)) / 1.349),
    }


def main() -> None:
    records = load_runs()
    print(f"loaded {len(records)} runs, "
          f"{len({r['target_label'] for r in records})} budgets, "
          f"{sum(r['epoch_fraction'] > 1.0 for r in records)} over 1 epoch")
    fractions = [r["embedding_fraction"] for r in records]
    print(f"embedding fraction range: {min(fractions):.1%} .. {max(fractions):.1%}")
    print(f"estimator: parabola vertex, per-curve trim = {OUTLIER_THRESHOLD}")

    point: dict[str, dict[str, float]] = {}
    for field, key in (("final", "loss_final"), ("trailing", "loss_trailing")):
        losses = np.array([r[key] for r in records])
        with_emb = fit_exponent(records, losses, n_key="n_total", flops_key="target_flops")
        without_emb = fit_exponent(records, losses, n_key="n_nonembed", flops_key="target_flops")
        variant_b = fit_exponent(records, losses, n_key="n_nonembed", flops_key="flops_nonembed")
        assert with_emb and without_emb and variant_b

        a_with, r2_with, curves_with = with_emb
        a_without, r2_without, curves_without = without_emb
        a_b, r2_b, _ = variant_b
        delta = a_without - a_with

        point[field] = {
            "a_with": a_with, "r2_with": r2_with,
            "a_without": a_without, "r2_without": r2_without,
            "a_variant_b": a_b, "r2_variant_b": r2_b,
            "delta": delta, "delta_percent": 100.0 * delta / a_with,
        }

        print()
        print("=" * 72)
        print(f"loss field = {field}")
        print("=" * 72)
        print(f"  variant A  N with embedding     a = {a_with:.6f}   R2 = {r2_with:.4f}")
        print(f"  variant A  N without embedding  a = {a_without:.6f}   R2 = {r2_without:.4f}")
        print(f"             change               {delta:+.6f}  ({100 * delta / a_with:+.1f}%)")
        print(f"  variant B  full Kaplan switch   a = {a_b:.6f}   R2 = {r2_b:.4f}")

        if field == "final":
            print()
            print(f"  {'budget':>10} {'pts':>5} {'kept':>5} {'N*_total':>10} {'N*_nonemb':>10} {'range':>8}")
            print("  " + "-" * 54)
            for label, info in curves_with.items():
                other = curves_without[label]
                inside = other["log_n_min"] <= other["vertex_log10"] <= other["log_n_max"]
                print(
                    f"  {label:>10} {info['n_points']:>5} {info['n_kept']:>5} "
                    f"{info['n_star'] / 1e6:>9.2f}M {other['n_star'] / 1e6:>9.2f}M "
                    f"{'yes' if inside else 'EXTRAP':>8}"
                )

    # Paired bootstrap on the authoritative loss field.
    print()
    print(f"paired bootstrap: {N_BOOTSTRAP} draws, sigma = {RUN_TO_RUN_SIGMA}, seed = {RANDOM_SEED}")
    rng = np.random.default_rng(RANDOM_SEED)
    base = np.array([r["loss_final"] for r in records])

    draws_with: list[float] = []
    draws_without: list[float] = []
    for index in range(N_BOOTSTRAP):
        perturbed = base + rng.normal(0.0, RUN_TO_RUN_SIGMA, size=base.shape)
        fit_with = fit_exponent(records, perturbed, n_key="n_total", flops_key="target_flops")
        fit_without = fit_exponent(records, perturbed, n_key="n_nonembed", flops_key="target_flops")
        if fit_with is None or fit_without is None:
            continue
        draws_with.append(fit_with[0])
        draws_without.append(fit_without[0])
        if (index + 1) % 1000 == 0:
            print(f"  {index + 1}/{N_BOOTSTRAP}")

    array_with = np.array(draws_with)
    array_without = np.array(draws_without)
    relative = 100.0 * (array_without - array_with) / array_with

    stats_with = summarize(array_with)
    stats_without = summarize(array_without)
    stats_relative = summarize(relative)

    print()
    print("=" * 72)
    print("bootstrap, loss field = final, variant A")
    print("=" * 72)
    print(f"  usable draws         {len(array_with)}/{N_BOOTSTRAP}")
    print(
        f"  N with embedding     median {stats_with['median']:.4f}  "
        f"95% [{stats_with['ci_low']:.4f}, {stats_with['ci_high']:.4f}]  "
        f"sd {stats_with['sd']:.4f}  robust-sd {stats_with['iqr_sd']:.4f}"
    )
    print(
        f"  N without embedding  median {stats_without['median']:.4f}  "
        f"95% [{stats_without['ci_low']:.4f}, {stats_without['ci_high']:.4f}]  "
        f"sd {stats_without['sd']:.4f}  robust-sd {stats_without['iqr_sd']:.4f}"
    )
    print(
        f"  relative change      median {stats_relative['median']:+.1f}%  "
        f"95% [{stats_relative['ci_low']:+.1f}%, {stats_relative['ci_high']:+.1f}%]"
    )
    print(f"  P(without > with)    {float((relative > 0).mean()):.3f}")
    print(f"  P(a_with    > 0.5)   {float((array_with > 0.5).mean()):.3f}")
    print(f"  P(a_without > 0.5)   {float((array_without > 0.5).mean()):.3f}")

    payload = {
        "csv_path": str(CSV_PATH),
        "estimator": {
            "kind": "parabola vertex",
            "per_curve_trim": OUTLIER_THRESHOLD,
            "note": "trim reproduces the documented headline a = 0.543046 on loss field 'final'",
        },
        "n_runs": len(records),
        "sigma": RUN_TO_RUN_SIGMA,
        "n_bootstrap": N_BOOTSTRAP,
        "usable_draws": len(array_with),
        "random_seed": RANDOM_SEED,
        "point_estimates": point,
        "bootstrap_final_variant_a": {
            "a_with": stats_with,
            "a_without": stats_without,
            "relative_change_percent": stats_relative,
            "p_without_gt_with": float((relative > 0).mean()),
            "p_a_with_gt_half": float((array_with > 0.5).mean()),
            "p_a_without_gt_half": float((array_without > 0.5).mean()),
        },
    }
    out_path = Path(__file__).resolve().parent / "c1_embedding_vertex_result.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
