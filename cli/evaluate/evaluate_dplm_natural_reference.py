#!/usr/bin/env python3
"""Score frozen natural UR50 reference proteins using the same native ESMFold.

This entry point does NOT fabricate model sampling, decoding, EOS, or checkpoint
provenance for natural sequences. Fold coverage is not generation validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cli.evaluate.evaluate_dplm_structure import AA, CONDITIONS, TASK, native_folder, score_rows
from pipeline.runtime_host import require_compute_host


def folding_rows(references, manifest):
    if not manifest["complete"] or manifest["kind"] != "natural_reference":
        raise ValueError("frozen natural reference required")
    if len(references) != manifest["selected_total"] or len({r["id"] for r in references}) != len(references):
        raise ValueError("natural reference record count or IDs differ")
    result = []
    for row in references:
        seq, condition = row["sequence"], row["condition"]
        if condition not in CONDITIONS or len(seq) != int(condition) or set(seq) - AA:
            raise ValueError(f"natural reference has wrong length/alphabet: {row['id']}")
        # Adapter to the shared fold/aggregation API, not a claim of decoding.
        result.append({"id": row["id"], "condition": condition,
                       "sample_index": row["sample_index"], "status": "valid",
                       "decoded_sequence": seq})
    for condition in CONDITIONS:
        indices = sorted(r["sample_index"] for r in result if r["condition"] == condition)
        if indices != list(range(manifest["selected_per_length"])):
            raise ValueError(f"natural reference sample grid differs: {condition}")
    # Interleave lengths so progress and resource costs are visible in all bins.
    return sorted(result, key=lambda row: (row["sample_index"], int(row["condition"])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute_host("natural UR50 native ESMFold evaluation")
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    references = [json.loads(line) for line in (args.reference_dir / "natural_references.jsonl").read_text().splitlines()]
    rows = folding_rows(references, manifest)
    started = time.monotonic()
    folder, evaluator = native_folder(args.asset_root, device=args.device)
    print(f"NATURAL_REFERENCE_MODEL_LOADED device={args.device} seconds={time.monotonic()-started:.1f}", flush=True)
    result = score_rows(rows, args.output, folder)
    result.update(task=TASK, track="local-natural-ur50-reference", kind="natural_reference",
                  is_axis_model_result=False, reference_manifest=manifest,
                  evaluator=evaluator, reference_dir=str(args.reference_dir),
                  status_note="valid means canonical natural input, not a generated/decoded/EOS-complete model output; fold coverage is not generation validity",
                  generation_validity=None, sampling=None, checkpoint=None,
                  elapsed_seconds=time.monotonic()-started)
    with (args.output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("NATURAL_UR50_FOLD_COMPLETE " + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
