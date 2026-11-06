#!/usr/bin/env python3
"""Rescore saved, unchanged RAbD top100 using the approved prefix-L adapter."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.rabd_metrics import score_selected_prefix
from pipeline.runtime_host import require_compute_host

PROTOCOL = "rabd-prefix-L-fixed-denominator-unchanged-top100-v1-20260924"


def load(path):
    return json.loads(Path(path).read_text())


def total(cases):
    keys = [key for key in cases[0] if key not in ("case_id", "prefix_aar")]
    out = {key: sum(row[key] for row in cases) for key in keys}
    out["prefix_aar"] = out["matched_residues"] / out["requested_residues"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require_compute_host("RAbD saved top100 prefix scoring")
    audit = load(args.audit)
    sources = audit["sources"]
    refs = {row["id"]: row for row in map(json.loads, (Path(sources["data_root"]) / "test_references.jsonl").read_text().splitlines())}
    models = {}
    for name in ("small", "big", "native"):
        folder = Path(sources[name + "_metrics"]).parent
        if name == "native":
            by_id = {case_id: [] for case_id in refs}
            with (folder / "selected.jsonl").open() as handle:
                for line in handle:
                    row = json.loads(line)
                    by_id[row["case_id"]].append(dict(decoded_sequence=row["sequence"],
                                                     ended_with_eos=True, invalid_model_id=False))
        else:
            by_id = {}
            for path in sorted(folder.glob("rank*/case*.top100.json")):
                rows = load(path)
                case_id = rows[0]["case_id"]
                assert case_id not in by_id and all(row["case_id"] == case_id for row in rows)
                by_id[case_id] = rows
        assert set(by_id) == set(refs) and len(refs) == 58
        cases = [dict(case_id=case_id, **score_selected_prefix(refs[case_id]["target"], by_id[case_id]))
                 for case_id in refs]
        aggregate = total(cases)
        prior = audit["native"] if name == "native" else audit["models"][name]
        assert aggregate["matched_residues"] == prior["counts"]["positional_matches"]
        assert aggregate["requested_residues"] == prior["counts"]["requested_positions"]
        models[name] = dict(aggregate=aggregate, cases=cases)
    result = dict(complete=True, protocol=PROTOCOL, models=models, sources=sources,
                  source_audit=str(args.audit), sampling_changed=False, ranking_changed=False,
                  raw_outputs_changed=False, official_fixed_length_generation_reproduced=False,
                  scoring="valid AA sequence prefixL vs reference; missing/invalid zero; all requested positions retained",
                  comparability="Same58 and5800 slots; native fixedAA vs Axis freeEOS/BPE-path ranking;11 exact TRAIN overlaps remain.")
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"complete": True, "protocol": PROTOCOL, "models": {name: value["aggregate"] for name, value in models.items()}}), flush=True)


if __name__ == "__main__":
    main()
