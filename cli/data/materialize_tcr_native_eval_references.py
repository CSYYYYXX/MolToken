#!/usr/bin/env python3
"""Attach the released loader's actual reference view to unchanged Axis prompts.

The paired validation materializer is a lossless intermediate, not the native
metric reference view: TCRpMHCdataset.to_seq2seq_dict removes duplicate targets.
No training data, conditions, generated candidates or native list order change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host
from cli.evaluate.sample_tcr_translate_released_top20_auxiliary import load_prompt_groups

REFERENCE_VIEW = "released_native_seq2seq_unique"


def attach_native_references(prompts, native):
    by_key = {row["pmhc"]["official_key"]: row for row in native}
    expected = {row["pmhc"]["official_key"] for row in prompts}
    if len(by_key) != len(native) or set(by_key) != expected:
        raise ValueError("native pMHC groups must match prepared conditions one-to-one")
    result = []
    for row in prompts:
        replacement = by_key[row["pmhc"]["official_key"]]
        refs = replacement["reference_translations"]
        if replacement["pmhc"] != row["pmhc"]:
            raise ValueError("native and prepared pMHC condition metadata differ")
        if not refs or len(set(refs)) != len(refs):
            raise ValueError("native reference mapping is empty or not unique")
        if set(refs) != set(row["reference_translations"]):
            raise ValueError("native references differ beyond the documented deduplication")
        result.append({**row, "reference_translations": list(refs),
                       "reference_view": REFERENCE_VIEW})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-prompts", type=Path, required=True)
    parser.add_argument("--native-reference-groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute_host("TCR native evaluation reference materialization")
    prompts = load_prompt_groups(args.prepared_prompts)
    with args.native_reference_groups.open() as handle:
        native = [json.loads(line) for line in handle]
    result = attach_native_references(prompts, native)
    old_manifest = args.prepared_prompts.with_suffix(args.prepared_prompts.suffix + ".manifest.json")
    manifest = json.loads(old_manifest.read_text())
    counts = [len(row["reference_translations"]) for row in result]
    manifest.update(output=args.output.name, references=sum(counts),
                    reference_count_min=min(counts), reference_count_max=max(counts),
                    reference_view=REFERENCE_VIEW,
                    reference_audit={"native_reference_groups": str(args.native_reference_groups.resolve()),
                                     "prepared_prompts": str(args.prepared_prompts.resolve()),
                                     "prepared_reference_rows": sum(len(row["reference_translations"]) for row in prompts),
                                     "native_unique_reference_entries": sum(counts),
                                     "conditions_unchanged": True,
                                     "training_modified": False,
                                     "reference_order": "exact saved native set-to-list order, PYTHONHASHSEED=42; not author-order reconstruction"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for row in result:
            handle.write(json.dumps(row) + "\n")
    with args.output.with_suffix(args.output.suffix + ".manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    loaded = load_prompt_groups(args.output)
    if loaded != result:
        raise ValueError("published native prompts differ on loader round-trip")
    print("TCR_NATIVE_REFERENCES_COMPLETE", json.dumps(manifest["reference_audit"]), flush=True)


if __name__ == "__main__":
    main()
