#!/usr/bin/env python3
"""Native ESMFold scoring for DPLM-aligned generated protein slots.

Only pLDDT/pTM and coverage are produced here. pdb-TM and inner-TM remain
separate missing suite components; pTM must not be relabelled as either.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.pepbench_metrics import distribution_summary
from pipeline.runtime_host import require_compute_host

TASK = "dplm_length_conditioned_protein"
CONDITIONS = ("100", "200", "300", "400")
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_rows(rows, *, per_condition=None):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate raw slot IDs")
    grouped = {condition: [] for condition in CONDITIONS}
    for row in rows:
        if row["task"] != TASK or row["condition"] not in grouped:
            raise ValueError("not a DPLM length-100–400 sample")
        if row["status"] not in {"valid", "invalid_model_id", "missing_eos", "empty_generation", "decode_error"}:
            raise ValueError("unknown DPLM decoder status")
        if row["status"] == "valid":
            seq = row["decoded_sequence"]
            if not isinstance(seq, str) or not seq or set(seq) - AA or not row["ended_with_eos"]:
                raise ValueError("declared-valid slot is not canonical/EOS-complete")
        grouped[row["condition"]].append(row)
    for condition, values in grouped.items():
        if per_condition is not None and len(values) != per_condition:
            raise ValueError(f"{condition}: expected {per_condition} raw slots, got {len(values)}")
        if sorted(row["sample_index"] for row in values) != list(range(len(values))):
            raise ValueError(f"{condition}: missing or duplicate sample indices")
    return grouped


def score_rows(rows, output, fold_one):
    """All exceptions from native folding propagate. Invalid raw slots remain."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache, scored = {}, []
    for row in rows:
        result = {key: row[key] for key in ("id", "condition", "sample_index", "status")}
        result.update(folded=False, exact_length_match=False, plddt=None, ptm_diagnostic=None, pdb=None)
        if row["status"] == "valid":
            seq = row["decoded_sequence"]
            if seq not in cache:
                prediction = fold_one(seq)
                if not math.isfinite(prediction["plddt"]) or not 0 <= prediction["plddt"] <= 100:
                    raise ValueError("invalid native pLDDT")
                if not math.isfinite(prediction["ptm"]) or not 0 <= prediction["ptm"] <= 1:
                    raise ValueError("invalid native pTM")
                pdb_path = output / f"unique-{len(cache):06d}.pdb"
                if not any(line.startswith("ATOM") for line in prediction["pdb"].splitlines()):
                    raise ValueError("native folding output has no PDB atoms")
                pdb_path.write_text(prediction["pdb"])
                cache[seq] = prediction | {"path": str(pdb_path)}
                print(f"FOLDED_UNIQUE count={len(cache)} residues={len(seq)}", flush=True)
            prediction = cache[seq]
            result.update(folded=True, exact_length_match=len(seq) == int(row["condition"]),
                          sequence_length=len(seq), plddt=prediction["plddt"],
                          ptm_diagnostic=prediction["ptm"], pdb=prediction["path"])
        scored.append(result)
    groups = {}
    for condition in CONDITIONS:
        selected = [row for row in scored if row["condition"] == condition]
        folded = [row for row in selected if row["folded"]]
        exact = [row for row in folded if row["exact_length_match"]]
        n = len(selected)
        groups[condition] = {"requested": n, "folded": len(folded),
            "fold_coverage": len(folded) / n if n else None,
            "exact_length_folded": len(exact), "exact_length_coverage": len(exact) / n if n else None,
            "plddt_valid": distribution_summary([row["plddt"] for row in folded]),
            "plddt_exact_length_only": distribution_summary([row["plddt"] for row in exact]),
            "status_counts": dict(Counter(row["status"] for row in selected))}
    means = [groups[condition]["plddt_valid"]["mean"] for condition in CONDITIONS]
    exact_means = [groups[condition]["plddt_exact_length_only"]["mean"] for condition in CONDITIONS]
    summary = {"plddt_scoring_complete": True, "structure_suite_complete": False,
        "missing_suite_metrics": ["pdb-TM", "inner-TM"], "requested": len(rows),
        "folded": sum(row["folded"] for row in scored), "unique_folds_computed": len(cache),
        "fold_coverage": sum(row["folded"] for row in scored) / len(rows), "per_condition": groups,
        "macro_plddt_valid": sum(means) / 4 if all(value is not None for value in means) else None,
        "macro_plddt_exact_length_only": sum(exact_means) / 4 if all(value is not None for value in exact_means) else None,
        "aggregation": "native atom37-weighted pLDDT per protein, then raw valid-slot mean per target length; duplicates retain weight; exact-length subset separately reported",
        "failed_slots_replaced": False, "sequences_repaired_or_truncated": False,
        "ptm_note": "native pTM is diagnostic, NOT pdb-TM or inner-TM"}
    with (output / "per_slot.jsonl").open("x") as handle:
        for row in scored:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return summary


def native_folder(asset_root, *, device, esm_float32=False, cpu_threads=4):
    for name in ("esmfold_3B_v1", "esm2_t36_3B_UR50D", "esm2_t36_3B_UR50D-contact-regression"):
        path = asset_root / "checkpoints" / (name + ".pt")
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"offline ESMFold asset missing: {path}")
    import torch
    import esm
    if cpu_threads <= 0:
        raise ValueError("positive CPU thread count required")
    torch.set_num_threads(cpu_threads)
    torch.hub.set_dir(str(asset_root))
    model = esm.pretrained.esmfold_v1().eval()
    if device == "cpu" or esm_float32:
        model.esm.float()
    if device == "cpu":
        model.cpu()
    else:
        model.cuda()
    model.set_chunk_size(64)
    def fold_one(sequence):
        with torch.inference_mode():
            output = model.infer([sequence], num_recycles=None)
        if not torch.isfinite(output["positions"]).all():
            raise ValueError("native folding produced non-finite coordinates")
        return {"plddt": float(output["mean_plddt"][0]), "ptm": float(output["ptm"][0]),
                "pdb": model.output_to_pdb(output)[0]}
    return fold_one, {"model": "native esmfold_v1", "asset_root": str(asset_root), "device": device,
        "source": esm.__file__, "batch_size": 1, "chunk_size": 64, "num_recycles_argument": None,
        "config_max_recycles": model.cfg.trunk.max_recycles, "esm_dtype": str(next(model.esm.parameters()).dtype),
        "cpu_threads": cpu_threads}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-samples", type=Path, required=True)
    parser.add_argument("--sampling-summary", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--track", choices=("released", "robustness", "diagnostic"), default="released")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute_host("native DPLM generated-protein structure evaluation")
    rows = [json.loads(line) for line in args.decoded_samples.read_text().splitlines()]
    validate_rows(rows, per_condition={"released": 40, "robustness": 100, "diagnostic": None}[args.track])
    sampling = json.loads(args.sampling_summary.read_text()) if args.sampling_summary else None
    if args.track != "diagnostic":
        if sampling is None or sampling["task"] != TASK or not sampling["complete"] or sampling["failed_slots_replaced"]:
            raise ValueError("completed unfiltered DPLM sampling provenance required")
        generation = sampling["generation"]
        if generation["seed"] not in ((42,) if args.track == "released" else (42, 43, 44)):
            raise ValueError("seed outside the selected frozen track")
        if generation["greedy"] or (generation["temperature"], generation["top_p"], generation["top_k"], generation["max_context"]) != (1.0, 1.0, 0, 512):
            raise ValueError("sampling protocol differs from the frozen track")
        if generation["max_new_tokens"] != 512:
            raise ValueError("generation budget differs from the frozen track")
    folder, provenance = native_folder(args.asset_root, device=args.device)
    result = score_rows(rows, args.output, folder)
    result.update(task=TASK, track=args.track, sampling=sampling,
                  decoded_samples=str(args.decoded_samples), evaluator=provenance,
                  scope="DPLM-aligned context/representation-filtered length-100–400 partial protocol; free EOS; no claim of complete DPLM suite")
    with (args.output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
