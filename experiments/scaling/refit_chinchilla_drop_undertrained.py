#!/usr/bin/env python
"""Refit the 5-parameter Chinchilla loss law after dropping undertrained runs.

Question: is the alpha/beta imbalance (alpha=0.203, beta=0.816 -> derived a=0.80)
caused by CONTAMINATION from undertrained large-N points, or by the ABSENCE of
usable large-N points?

  - If dropping undertrained points moves alpha toward ~0.69 (self-consistent with
    the IsoFLOP a=0.5430), the contradiction was contamination -> S1 less necessary.
  - If dropping them makes alpha WORSE / less identifiable (wider CI), the problem
    is missing information -> S1 (new well-trained large-N points) is required.

Law:      L = E + A/N^alpha + B/D^beta
Derived:  a = beta / (alpha + beta)   (compute-optimal exponent, N* ~ C^a)
Fit:      minimize mean Huber(ln L_pred - ln L_true, delta=1e-3), L-BFGS-B,
          many random starts, all 5 params as exp() of unconstrained vars.

This mirrors the sibling ``fit_chinchilla_loss.py`` method so numbers are comparable.
Pure CPU, seconds to minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ISOFLOP_A = 0.5430459  # frozen headline (parabola-vertex + OLS), the reference
ISOFLOP_BETA_REF = 0.816274  # baseline fitted beta, used to back-solve target alpha
HUBER_DELTA = 1e-3

DEFAULT_CSV = (
    "./storage/molscaling/runs/"
    "scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1/plots/"
    "chinchilla_loss_fit_44runs/chinchilla_input_44runs.csv"
)


def load_runs(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "model_id": r["model_id"],
                    "N": float(r["parameter_count_N"]),
                    "D": float(r["training_tokens_D"]),
                    "L": float(r["final_training_loss"]),
                    "steps": int(r["global_step"]),
                    "target_flops": float(r["target_flops_6nd"]),
                }
            )
    return rows


def huber(x: np.ndarray, delta: float) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


def fit(rows: list[dict], n_starts: int = 256, seed: int = 20260809) -> dict | None:
    if len(rows) < 6:  # 5 params, need more points than params
        return None
    N = np.array([r["N"] for r in rows])
    D = np.array([r["D"] for r in rows])
    L = np.array([r["L"] for r in rows])
    lnL = np.log(L)
    lnN = np.log(N)
    lnD = np.log(D)

    def objective(theta):
        e, a, b, al, be = theta
        # exp-parametrized so all five stay strictly positive
        pred = np.exp(e) + np.exp(a - np.exp(al) * lnN) + np.exp(b - np.exp(be) * lnD)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1e10
        return float(np.mean(huber(np.log(pred) - lnL, HUBER_DELTA)))

    rng = np.random.default_rng(seed)
    best = None
    # start near the known baseline solution plus wide random starts
    seeds = [np.array([math.log(3.9), math.log(15.0), math.log(7.6e6),
                       math.log(0.20), math.log(0.82)])]
    for _ in range(n_starts):
        seeds.append(
            np.array(
                [
                    rng.uniform(math.log(2.0), math.log(4.6)),
                    rng.uniform(math.log(0.1), math.log(1e3)),
                    rng.uniform(math.log(1.0), math.log(1e9)),
                    rng.uniform(math.log(0.02), math.log(2.0)),
                    rng.uniform(math.log(0.02), math.log(2.0)),
                ]
            )
        )
    for s in seeds:
        try:
            res = minimize(objective, s, method="L-BFGS-B",
                           options={"maxiter": 3000, "ftol": 1e-16, "gtol": 1e-12})
        except Exception:
            continue
        if res.success or np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        return None

    e, a, b, al, be = best.x
    E, A, B, alpha, beta = np.exp(e), np.exp(a), np.exp(b), np.exp(al), np.exp(be)
    pred = E + A / N**alpha + B / D**beta
    resid = pred - L
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    return {
        "n": len(rows),
        "E": float(E),
        "A": float(A),
        "B": float(B),
        "alpha": float(alpha),
        "beta": float(beta),
        "a_derived": float(beta / (alpha + beta)),
        "beta_over_alpha": float(beta / alpha),
        "rmse": float(math.sqrt(ss_res / len(rows))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "obj": float(best.fun),
        "N_min": float(N.min()),
        "N_max": float(N.max()),
        "N_span": float(N.max() / N.min()),
        "n_over_200M": int(np.sum(N > 200e6)),
    }


def bootstrap_alpha(rows, n_boot=300, n_starts=24, seed=20260809):
    """Pairs bootstrap on alpha -> measures IDENTIFIABILITY (CI width)."""
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(rows))
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(rows), replace=True)
        sample = [rows[i] for i in pick]
        r = fit(sample, n_starts=n_starts, seed=int(rng.integers(1 << 30)))
        if r is not None:
            vals.append((r["alpha"], r["a_derived"]))
    if not vals:
        return None
    al = np.array([v[0] for v in vals])
    ad = np.array([v[1] for v in vals])
    return {
        "n_ok": len(vals),
        "alpha_lo": float(np.percentile(al, 2.5)),
        "alpha_hi": float(np.percentile(al, 97.5)),
        "alpha_width": float(np.percentile(al, 97.5) - np.percentile(al, 2.5)),
        "a_lo": float(np.percentile(ad, 2.5)),
        "a_hi": float(np.percentile(ad, 97.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--starts", type=int, default=256)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rows = load_runs(Path(args.csv))
    target_alpha = ISOFLOP_BETA_REF * (1.0 / ISOFLOP_A - 1.0)

    print("=" * 100)
    print("INPUT AUDIT: the large-N points that carry all alpha information")
    print("=" * 100)
    big = sorted([r for r in rows if r["N"] > 200e6], key=lambda r: -r["N"])
    print(f"  total runs = {len(rows)}   N>200M = {len(big)}")
    print(f"  {'model':10s} {'N':>10s} {'D':>12s} {'steps':>8s} {'loss':>8s} {'budget':>10s}")
    for r in big:
        print(f"  {r['model_id']:10s} {r['N']/1e9:9.3f}B {r['D']/1e9:11.3f}B "
              f"{r['steps']:8d} {r['L']:8.4f} {r['target_flops']:10.1e}")
    allsteps = sorted(r["steps"] for r in rows)
    print(f"  step distribution: min={allsteps[0]} p25={allsteps[len(allsteps)//4]} "
          f"median={allsteps[len(allsteps)//2]} max={allsteps[-1]}")

    variants = [
        ("baseline (all 44)", lambda r: True),
        ("drop steps<1000", lambda r: r["steps"] >= 1000),
        ("drop steps<2000", lambda r: r["steps"] >= 2000),
        ("drop steps<3000", lambda r: r["steps"] >= 3000),
        ("drop steps<5000", lambda r: r["steps"] >= 5000),
        ("drop steps<10000", lambda r: r["steps"] >= 10000),
        ("drop ALL N>200M", lambda r: r["N"] <= 200e6),
    ]

    print()
    print("=" * 100)
    print("REFIT RESULTS   (IsoFLOP measured a = %.4f ; self-consistent alpha would be ~%.3f)"
          % (ISOFLOP_A, target_alpha))
    print("=" * 100)
    print(f"  {'variant':22s} {'n':>3s} {'>200M':>5s} {'alpha':>7s} {'beta':>7s} "
          f"{'b/a':>6s} {'a_der':>7s} {'gap_a':>7s} {'E':>7s} {'RMSE':>7s} {'R2':>7s}")
    results = {}
    for name, pred in variants:
        sub = [r for r in rows if pred(r)]
        res = fit(sub, n_starts=args.starts)
        if res is None:
            print(f"  {name:22s} {len(sub):3d}  FIT FAILED / too few points")
            continue
        results[name] = res
        gap = res["a_derived"] - ISOFLOP_A
        print(f"  {name:22s} {res['n']:3d} {res['n_over_200M']:5d} "
              f"{res['alpha']:7.4f} {res['beta']:7.4f} {res['beta_over_alpha']:6.2f} "
              f"{res['a_derived']:7.4f} {gap:+7.4f} {res['E']:7.4f} "
              f"{res['rmse']:7.4f} {res['r2']:7.4f}")

    print()
    print("  N coverage per variant:")
    for name in results:
        r = results[name]
        print(f"    {name:22s} N: {r['N_min']/1e6:7.1f}M .. {r['N_max']/1e6:8.1f}M "
              f"(span {r['N_span']:6.1f}x)")

    # identifiability: does dropping points widen alpha's CI?
    print()
    print("=" * 100)
    print("IDENTIFIABILITY CHECK: bootstrap CI width on alpha (n_boot=%d)" % args.bootstrap)
    print("  wider CI  => alpha LESS identifiable => dropping points destroyed information")
    print("=" * 100)
    boot = {}
    for name in ["baseline (all 44)", "drop steps<3000", "drop ALL N>200M"]:
        if name not in results:
            continue
        pred = dict(variants)[name]
        sub = [r for r in rows if pred(r)]
        b = bootstrap_alpha(sub, n_boot=args.bootstrap, seed=20260809)
        if b is None:
            print(f"  {name:22s} bootstrap failed")
            continue
        boot[name] = b
        print(f"  {name:22s} alpha 95%CI=[{b['alpha_lo']:.4f}, {b['alpha_hi']:.4f}] "
              f"width={b['alpha_width']:.4f}   a_der 95%CI=[{b['a_lo']:.4f}, {b['a_hi']:.4f}] "
              f"(ok {b['n_ok']}/{args.bootstrap})")

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    base = results.get("baseline (all 44)")
    if base:
        print(f"  baseline: alpha={base['alpha']:.4f}  a_derived={base['a_derived']:.4f} "
              f"(IsoFLOP {ISOFLOP_A:.4f}, gap {base['a_derived']-ISOFLOP_A:+.4f})")
        best_name, best_gap = None, None
        for name, r in results.items():
            if name == "baseline (all 44)":
                continue
            g = abs(r["a_derived"] - ISOFLOP_A)
            if best_gap is None or g < best_gap:
                best_name, best_gap = name, g
        if best_name:
            r = results[best_name]
            base_gap = abs(base["a_derived"] - ISOFLOP_A)
            print(f"  best drop-variant: {best_name} -> alpha={r['alpha']:.4f} "
                  f"a_derived={r['a_derived']:.4f} gap={best_gap:+.4f}")
            if best_gap < base_gap * 0.5:
                print("  => dropping undertrained points SUBSTANTIALLY fixes the contradiction")
                print("     (contamination was the main cause; S1's necessity drops)")
            elif best_gap < base_gap:
                print("  => dropping helps only PARTIALLY; residual gap remains")
                print("     (contamination is not the whole story; S1 still needed)")
            else:
                print("  => dropping does NOT help (or makes it worse)")
                print("     (the problem is MISSING large-N information, not bad points;")
                print("      S1 is REQUIRED - no amount of filtering recovers it)")

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps({"results": results, "bootstrap": boot,
                        "isoflop_a": ISOFLOP_A, "target_alpha": target_alpha},
                       indent=2),
            encoding="utf-8",
        )
        print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
