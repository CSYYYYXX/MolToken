#!/usr/bin/env python3
"""Offline, CPU-only native MOSES metric smoke using a staged source checkout.

Set PYTHONPATH to the staged official source before starting Python. This uses
its existing package metadata and does not install or replace the training env.
Synthetic self-comparison values are controls, never model benchmark results.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cli/evaluate"))
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host
from evaluate_axis_b_moses_official import load_official_api, _local_chemnet_fcd
from smoke_guacamol_fcd11_worker import SMILES as GUACAMOL_SMILES

# MOSES counts scaffolds with at least two rings; the smaller GuacaMol control
# alone has none and correctly gives undefined scaffold similarity.
SMILES = GUACAMOL_SMILES + ("c1ccc2ccccc2c1", "c1ccc(-c2ccccc2)cc1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemnet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("official MOSES native metric smoke")
    api = load_official_api({})
    # Official MOSES reference CSVs contain canonical SMILES. Its novelty
    # helper canonicalizes generated strings, but assumes train is canonical.
    reference = api.remove_invalid(list(SMILES), canonize=True, n_jobs=1)
    if len(reference) != len(SMILES):
        raise ValueError("synthetic fixture contains an invalid molecule")
    with _local_chemnet_fcd(api, args.chemnet):
        metrics = api.get_all_metrics(list(SMILES), k=[len(SMILES)],
            test=reference, test_scaffolds=reference, train=reference,
            n_jobs=1, device="cpu", batch_size=32)
    metrics = {k: float(v) for k, v in metrics.items()}
    for key, value in metrics.items():
        if not math.isfinite(value):
            raise ValueError(f"nonfinite synthetic metric: {key}={value}")
    for key in ("FCD/Test", "FCD/TestSF", "logP", "SA", "QED", "weight", "Novelty"):
        if abs(metrics[key]) > 1e-4:
            raise ValueError(f"self-comparison should be zero: {key}={metrics[key]}")
    for key in ("valid", f"unique@{len(SMILES)}", "SNN/Test", "SNN/TestSF", "Frag/Test", "Scaf/Test", "Scaf/TestSF"):
        if abs(metrics[key]-1) > 1e-5:
            raise ValueError(f"self-comparison should be one: {key}={metrics[key]}")
    summary = {"complete": True, "synthetic_control_only": True,
        "sample_size": len(SMILES), "device": "cpu", "metrics": metrics,
        "official_api": api.provenance,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("molsets", "fcd-torch", "torch", "numpy", "rdkit")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "official_api"}), flush=True)
    print("OFFICIAL_MOSES_NATIVE_EXISTING_SOURCE_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
