#!/usr/bin/env python3
"""Prepare and run the author's unmodified RAbD baseline evaluator offline.

This is a native baseline, not an Axis sampler or a new baseline training.
The author script retains top100 only. Do not infer raw-draw validity from it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
TOP = 100
DRAWS = 10000


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_native_cases(cases, references):
    """Match the author's actual crop/order, not merely the source filenames."""
    if len(cases) != len(references) or not cases:
        raise ValueError("native and Axis case counts differ or are empty")
    if len({r["id"] for r in references}) != len(references):
        raise ValueError("duplicate Axis case IDs")
    for case, ref in zip(cases, references, strict=True):
        target = "".join(aa for aa, cdr in zip(case["seq"], case["cdr"], strict=True) if cdr == "3")
        if (case["pdb"] != ref["pdb"] or case["seq"] != ref["cropped_sequence"]
                or target != ref["target"] or len(target) != ref["requested_length"]):
            raise ValueError(f"native crop/reference mismatch: {ref['id']}")
        if not target or set(target) - AA:
            raise ValueError(f"noncanonical/empty reference: {ref['id']}")


def prepare(args):
    sys.path.insert(0, str(args.native_root.resolve()))
    from structgen import AntibodyDataset

    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.source_root / "test.jsonl.gz", "rb") as source, \
            (output / "test.jsonl").open("xb") as destination:
        shutil.copyfileobj(source, destination)
    refs = read_rows(args.sft_root / "test_references.jsonl")
    cases = AntibodyDataset(str(output / "test.jsonl")).data
    verify_native_cases(cases, refs)
    if len(cases) != 58:
        raise ValueError("formal RAbD baseline requires all58 released test cases")
    dump(output / "test_references.json", refs)

    # Capacity check is chosen only from our TRAIN-derived validation set.
    # Longest CDR first, then largest native context. No test loss/score is used.
    validation = read_rows(args.sft_root / "validation_references.jsonl")
    smoke_ref = max(validation, key=lambda r: (r["requested_length"], len(r["cropped_sequence"])))
    if smoke_ref["source_split"] != "train":
        raise ValueError("capacity smoke must originate in official train")
    smoke = None
    with gzip.open(args.source_root / "train.jsonl.gz", "rt") as source:
        for line, raw in enumerate(source, 1):
            if line == smoke_ref["source_line"]:
                smoke = json.loads(raw)
                break
    if smoke is None:
        raise ValueError("internal-validation source line is missing")
    with (output / "smoke.jsonl").open("x") as handle:
        handle.write(json.dumps(smoke) + "\n")
    verify_native_cases(AntibodyDataset(str(output / "smoke.jsonl")).data, [smoke_ref])
    dump(output / "smoke_references.json", [smoke_ref])
    result = {"complete": True, "test_cases": len(cases), "native_crop": "AntibodyDataset max_len130",
              "test_order_and_context_match_axis": True, "smoke_source": "TRAIN-derived validation",
              "smoke_case_id": smoke_ref["id"], "smoke_cdr_length": smoke_ref["requested_length"],
              "smoke_context_length": len(smoke_ref["cropped_sequence"]),
              "native_root": str(args.native_root), "source_root": str(args.source_root),
              "sft_root": str(args.sft_root), "gpu_inference_run": False}
    dump(output / "prepare_summary.json", result)
    print(json.dumps(result), flush=True)


def score_native_stdout(path, references, top=TOP):
    """Recompute exact author micro-AAR from every printed selected candidate.

    The author's vocabulary includes '#'. Keep such outputs in native AAR,
    and separately expose the conservative canonical-only whole-slot score.
    A wrong output length is an evaluator failure, never a zip denominator cut.
    """
    cases, selected = [], []
    with path.open() as handle:
        for ref in references:
            matched = valid_matched = valid_count = exact = 0
            previous_ppl = -math.inf
            for rank in range(top):
                line = handle.readline()
                if not line:
                    raise ValueError(f"missing selected candidates: {ref['id']}")
                pdb, reference, sequence, ppl_text, flags_text = line.strip().split(maxsplit=4)
                if pdb != ref["pdb"] or reference != ref["target"]:
                    raise ValueError(f"native candidate/reference order mismatch: {ref['id']}")
                if len(sequence) != len(reference):
                    raise ValueError("native fixed-length assumption failed; refuse truncated zip scoring")
                ppl, flags = float(ppl_text), json.loads(flags_text)
                if not math.isfinite(ppl) or ppl < 1 or ppl < previous_ppl:
                    raise ValueError("native PPL must be finite, >=1, and sorted within each case")
                previous_ppl = ppl
                matches = [int(a == b) for a, b in zip(reference, sequence, strict=True)]
                if flags != matches:
                    raise ValueError("native printed matches disagree with sequences")
                valid = not (set(sequence) - AA)
                matched += sum(matches)
                valid_matched += sum(matches) if valid else 0
                valid_count += int(valid)
                exact += int(sequence == reference)
                selected.append({"case_id": ref["id"], "rank": rank + 1, "sequence": sequence,
                                 "ppl": ppl, "canonical": valid, "matches": matches})
            cases.append({"case_id": ref["id"], "selected": top, "canonical_selected": valid_count,
                          "native_matched_residues": matched, "canonical_slot_matched_residues": valid_matched,
                          "requested_residues": top * len(ref["target"]), "exact_sequences": exact})
        footer = handle.readline().strip()
        prefix = "Amino acid recovery rate = "
        if not footer.startswith(prefix) or handle.read().strip():
            raise ValueError("missing native completion line or unexpected trailing output")
    total = sum(c["requested_residues"] for c in cases)
    matches = sum(c["native_matched_residues"] for c in cases)
    valid = sum(c["canonical_selected"] for c in cases)
    n = len(selected)
    native_aar = matches / total
    if footer[len(prefix):] != f"{native_aar:.4f}":
        raise ValueError("exact recomputed AAR disagrees with author's printed rounded score")
    result = {"complete": True, "case_count": len(cases), "selected": n,
              "canonical_selected": valid, "selected_validity": valid / n,
              "native_matched_residues": matches, "requested_residues": total,
              "upstream_aar_exact": native_aar, "upstream_aar_printed": footer[len(prefix):],
              "official_aar_assumptions_satisfied": valid == n,
              "official_aar": native_aar if valid == n else None,
              "fixed_slot_aar_invalid_zero": sum(c["canonical_slot_matched_residues"] for c in cases) / total,
              "raw_draw_validity": None, "raw_draws_saved": False, "cases": cases}
    return result, selected


def run(args):
    preparation = json.loads((args.prepared_root / "prepare_summary.json").read_text())
    if not preparation["complete"] or preparation["test_cases"] != 58:
        raise ValueError("complete58-case preparation required")
    refs = json.loads((args.prepared_root / f"{args.scope}_references.json").read_text())
    args.output_root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(args.native_root / "rabd_test.py"),
               "--data_path", str(args.prepared_root / f"{args.scope}.jsonl"),
               "--load_model", str(args.native_root / "ckpts/RefineGNN-rabd/model.best"),
               "--architecture", "RefineGNN", "--seed", "7"]
    config = {"command": command, "scope": args.scope, "seed": 7, "dtype": "float32",
              "sampling": "unmodified author rabd_test.py: two5000-draw batches per case, top100 ownPPL",
              "draws_per_case": DRAWS, "top": TOP, "expected_cases": len(refs),
              "source_url": "https://github.com/wengong-jin/RefineGNN",
              "source_revision": "f41aa5fea6679d7e4c4dbce776b4bac89115a03a",
              "case_seed_rule": "author global seed once BEFORE model construction; cases in original order",
              "source_modified": False, "is_axis_result": False, "preparation": preparation}
    dump(args.output_root / "run_config.json", config)
    started = time.monotonic()
    print("NATIVE_REFINEGNN_START " + json.dumps(config), flush=True)
    with (args.output_root / "native.stdout.log").open("x") as stdout, \
            (args.output_root / "native.stderr.log").open("x") as stderr:
        subprocess.run(command, cwd=args.native_root, stdout=stdout, stderr=stderr, check=True)
    result, selected = score_native_stdout(args.output_root / "native.stdout.log", refs)
    result.update(protocol=config, elapsed_seconds=time.monotonic() - started,
                  formal_released58_complete=args.scope == "test" and len(refs) == 58,
                  comparability=["Released58 vs paper60: local native rerun, not the paper35.37 value.",
                                 "Fixed-lengthAA output vs Axis freeEOS/BPE-path own-likelihood ranking.",
                                 "Author checkpoint has its own original training/model-selection lineage.",
                                 "Author retains top100 only; raw validity unavailable, never infer it from selected validity."])
    with (args.output_root / "selected.jsonl").open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    dump(args.output_root / "metrics.json", result)
    print("NATIVE_REFINEGNN_COMPLETE " + json.dumps({k: v for k, v in result.items()
                                                   if k not in ("cases", "protocol")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="CPU worker: stage original data and verify native crop/order")
    prep.add_argument("--source-root", type=Path, required=True)
    prep.add_argument("--sft-root", type=Path, required=True)
    infer = sub.add_parser("run", help="GPU worker: call the unmodified author evaluator")
    infer.add_argument("--prepared-root", type=Path, required=True)
    infer.add_argument("--scope", choices=("smoke", "test"), required=True)
    for parser in (prep, infer):
        parser.add_argument("--native-root", type=Path, required=True)
        parser.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    require_compute_host("native RefineGNN preparation/evaluation")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
