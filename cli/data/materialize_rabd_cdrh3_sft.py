#!/usr/bin/env python3
"""Stage the original RefineGNN RAbD sequence-design split for Axis SFT.

Worker only. Preserve the author's 130-residue context crop and held-out
split. Encode left and right flanks as separate molecular components, never
the hidden CDR-H3. The only text condition is its requested residue count.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import socket
import sys
import time

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

TASK = "rabd_cdrh3_sequence_design"
DATASET_ID = "refinegnn_rabd_released_v1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
SPLITS = {"train": "train", "validation": "val", "test": "test"}
CODEC = DECODER = None
TEXT_LENGTHS = {}


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def summarize(values):
    import numpy as np
    values = list(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p99": float(np.quantile(values, .99))}


def official_crop(sequence, cdr):
    """Exact sequence slice in upstream AntibodyDataset(max_len=130)."""
    if not isinstance(sequence, str) or not isinstance(cdr, str):
        raise ValueError("sequence/cdr must be strings")
    if len(sequence) != len(cdr):
        raise ValueError("sequence/cdr annotation lengths differ")
    last = cdr.rindex("3")
    start, end = (last - 130 + 10, last + 10) if last >= 129 else (0, 130)
    return sequence[start:end], cdr[start:end], start, min(end, len(sequence))


def extract_pair(entry, split, line):
    sequence, cdr = entry["seq"], entry["cdr"]
    if cdr is None or "3" not in cdr:
        return None  # Explicit upstream exclusion, counted in source audit.
    cropped, labels, start, end = official_crop(sequence, cdr)
    left, right = labels.index("3"), labels.rindex("3") + 1
    if labels[left:right] != "3" * (right - left):
        raise ValueError(f"{split}:{line}: CDR-H3 is not contiguous")
    if labels.count("3") != cdr.count("3"):
        raise ValueError(f"{split}:{line}: upstream crop truncates CDR-H3")
    return {"id": f"rabd:{split}:{line:06d}:{entry['pdb']}", "split": split,
            "source_line": line, "pdb": entry["pdb"],
            "original_sequence": sequence, "cropped_sequence": cropped,
            "crop_start": start, "crop_end_exclusive": end,
            "left": cropped[:left], "target": cropped[left:right],
            "right": cropped[right:], "cdr_start_in_crop": left,
            "requested_length": right - left}


def read_release(root):
    rows, audits = {}, {}
    for split, stem in SPLITS.items():
        records, missing, invalid, keys = [], [], [], Counter()
        raw_count = 0
        with gzip.open(root / f"{stem}.jsonl.gz", "rt") as handle:
            for line, raw in enumerate(handle, 1):
                entry = json.loads(raw)
                raw_count += 1
                keys.update(entry.keys())
                row = extract_pair(entry, split, line)
                if row is None:
                    missing.append({"line": line, "pdb": entry["pdb"]})
                    continue
                bad = sorted(set(row["cropped_sequence"]) - AA)
                if bad:
                    invalid.append({"id": row["id"], "characters": bad})
                records.append(row)
        rows[split] = records
        audits[split] = {"raw_rows": raw_count, "upstream_usable_rows": len(records),
                         "upstream_no_cdr3_exclusions": missing,
                         "noncanonical_rows": invalid, "source_keys": dict(keys),
                         "empty_left_flank": sum(not r["left"] for r in records),
                         "empty_right_flank": sum(not r["right"] for r in records),
                         "original_residues": summarize(len(r["original_sequence"]) for r in records),
                         "context_crop_residues": summarize(len(r["cropped_sequence"]) for r in records),
                         "target_residues": summarize(len(r["target"]) for r in records)}
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        seq_a = {r["cropped_sequence"] for r in rows[a]}
        seq_b = {r["cropped_sequence"] for r in rows[b]}
        overlaps[f"{a}:{b}:identical_cropped_sequence"] = len(seq_a & seq_b)
    return rows, {"splits": audits, "overlap_counts": overlaps,
                  "source": str(root), "upstream_crop": "AntibodyDataset max_len=130",
                  "official_test_used_for_checkpoint_selection": False}


@lru_cache(maxsize=512)
def encode_component(sequence):
    # Released partial chains can end exactly at the CDR. Preserve that
    # missing flank as an empty side of <SEP>, without inventing residues.
    if not sequence:
        return {"fddfs_tokens": 0}, [], []
    from pipeline.graph.biopolymer_sequence import graph_to_biopolymer_sequence
    from pipeline.serialization.fddfs import deserialize_fddfs
    metrics, tokens, ids = CODEC.encode_sequence("peptide", sequence, "rabd-component")
    _, expanded = DECODER.decode_raw_ids(ids)
    if expanded != tokens:
        raise ValueError("BPE inverse did not restore the FDDFS token stream")
    graph = deserialize_fddfs(expanded, DECODER.vocab.default_edge)
    restored = graph_to_biopolymer_sequence(graph, target_type="peptide")
    if restored != sequence:
        raise ValueError(f"sequence roundtrip mismatch: {sequence!r} != {restored!r}")
    return metrics, tokens, ids


def encode_pair(row):
    left_m, left_t, _ = encode_component(row["left"])
    right_m, right_t, _ = encode_component(row["right"])
    target_m, _, target = encode_component(row["target"])
    joined = [*left_t, "<SEP>", *right_t]
    condition = CODEC.bpe_ids(joined, row["id"] + ":flanks")
    _, expanded = DECODER.decode_raw_ids(condition)
    if expanded != joined:
        raise ValueError(f"{row['id']}: joined condition roundtrip mismatch")
    full = len(condition) + len(target) + TEXT_LENGTHS[row["requested_length"]] + 3
    record = {"id": row["id"], "task": TASK, "target_type": "peptide",
              "condition": {"type": "both", "text": str(row["requested_length"]),
                            "molecule_bpe_ids": condition},
              "target_bpe_ids": target,
              "meta": {"split": row["split"], "pdb": row["pdb"],
                       "source_line": row["source_line"],
                       "requested_length": row["requested_length"],
                       "official_crop_start": row["crop_start"],
                       "official_crop_end_exclusive": row["crop_end_exclusive"],
                       "molecule_prefix_tokens": len(condition),
                       "text_prefix_tokens": TEXT_LENGTHS[row["requested_length"]],
                       "target_bpe_tokens": len(target), "full_context_tokens": full,
                       "condition_fddfs_tokens": len(joined),
                       "target_fddfs_tokens": target_m["fddfs_tokens"],
                       "target_truncated": False, "roundtrip_exact": True}}
    return record


def materialize(args, rows, audit):
    global CODEC, DECODER, TEXT_LENGTHS
    from cli.evaluate.eval_axis_b_samples import BPEDecoder, FDDFSVocab
    from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec
    from pipeline.finetune.axis_b_data import AxisBSFTCollator, build_axis_b_index
    from transformers import AutoTokenizer

    if any(a["noncanonical_rows"] for a in audit["splits"].values()):
        raise ValueError("noncanonical residues found; inspect source_audit.json before choosing a policy")
    if any(audit["overlap_counts"].values()) and not args.encode_only:
        raise ValueError("identical complete examples cross released splits; inspect source_audit.json")
    output = args.output_root
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    CODEC = FrozenAxisCodec(args.guide, args.vocab, args.merges)
    DECODER = BPEDecoder(args.merges, FDDFSVocab.load(args.vocab))
    tokenizer = AutoTokenizer.from_pretrained(str(args.text_encoder), local_files_only=True)
    lengths = {r["requested_length"] for values in rows.values() for r in values}
    TEXT_LENGTHS = {n: len(tokenizer(str(n))["input_ids"]) for n in lengths}
    # Compile/warm the existing C++ encoder before forking CPU workers.
    encode_component("ACD")
    print(json.dumps({"codec_ready": CODEC.codebook, "processes": args.workers}), flush=True)
    collator = AxisBSFTCollator(tokenizer, qwen_max_context=512,
                               condition_token_budget=512, molecule_max_length=512,
                               molecule_condition_mode="direct_qwen", return_tensors="list",
                               strict_no_runtime_truncation=True)
    reports, files, overflow = {}, [], []
    with mp.get_context("fork").Pool(args.workers) as pool, ExitStack() as stack:
        for split, source_rows in rows.items():
            handle = stack.enter_context((staging / f"{split}.jsonl").open("x"))
            references = stack.enter_context((staging / f"{split}_references.jsonl").open("x"))
            encoded = list(pool.imap(encode_pair, source_rows, chunksize=8))
            for record, source in zip(encoded, source_rows, strict=True):
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                references.write(json.dumps(source, separators=(",", ":")) + "\n")
                if record["meta"]["full_context_tokens"] > 512:
                    overflow.append({"id": record["id"], "tokens": record["meta"]["full_context_tokens"]})
            handle.flush()
            if not overflow:
                for offset in range(0, len(encoded), 64):
                    batch = collator(encoded[offset:offset + 64])
                    if any(batch["target_truncated"]) or any(batch["molecule_truncated"]):
                        raise ValueError("collator truncated a RAbD example")
                    actual = [sum(mask) for mask in batch["full_attention_mask"]]
                    expected = [r["meta"]["full_context_tokens"] for r in encoded[offset:offset + 64]]
                    if actual != expected:
                        raise ValueError(f"{split}: collator context accounting mismatch")
            reports[split] = {key: summarize(r["meta"][key] for r in encoded)
                             for key in ("molecule_prefix_tokens", "text_prefix_tokens", "target_bpe_tokens",
                                         "full_context_tokens", "condition_fddfs_tokens", "target_fddfs_tokens")}
            reports[split]["records"] = len(encoded)
            files.append({"dataset": DATASET_ID, "split": split, "relative_source": f"{split}.jsonl",
                          "records": len(encoded), "output_size": (staging / f"{split}.jsonl").stat().st_size})
            print(json.dumps({"split_complete": split, "records": len(encoded),
                              "context": reports[split]["full_context_tokens"]}), flush=True)
    dump_json(staging / "encoding_audit.json", {"splits": reports, "over_context": overflow})
    if overflow:
        raise ValueError(f"{len(overflow)} examples exceed context 512; no truncation/filter policy applied")
    if args.encode_only:
        # Preserve released splits for audit, but never publish an index or
        # formal training manifest for a known-contaminated split.
        audit_manifest = {"encoding_complete": True, "training_ready": False,
                          "reason": "released split overlap; clean split policy pending",
                          "split_counts": {s: len(r) for s, r in rows.items()},
                          "source_audit": audit, "length_statistics": reports,
                          "codec": CODEC.codebook,
                          "all_component_sequence_roundtrips_exact": True,
                          "all_collated_records_fit_512_without_truncation": True}
        dump_json(staging / "encoding_manifest.json", audit_manifest)
        staging.rename(output)
        return audit_manifest
    manifest = {"complete": True, "task": TASK, "dataset_id": DATASET_ID,
                "files": len(files), "records": sum(f["records"] for f in files),
                "file_summaries": files, "split_counts": {s: len(r) for s, r in rows.items()},
                "source_audit": audit, "length_statistics": reports, "codec": CODEC.codebook,
                "protocol": {"name": "axis-refinegnn-rabd-sequence-design-v1-20260923",
                             "serializer": "fddfs", "model_token_offset": 3,
                             "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
                             "condition": "numeric residue count + FDDFS(left flank)<SEP>FDDFS(right flank), frozen BPE",
                             "condition_contains_target_residues": False, "uses_xyz": False,
                             "vocabulary_extension": False, "pooling": False,
                             "context_budget": 512, "target_truncated": False,
                             "validation_source": "released RAbD val.jsonl (training/validation pool); test never used",
                             "stages": "alignment max100+earlystop; full max100+earlystop from alignment/best",
                             "evaluation": "released RAbD test AAR; 10000 candidates each, rank top100; actual test count in split_counts",
                             "evaluation_implementation_ready": False}}
    dump_json(staging / "manifest.json", manifest)
    index = build_axis_b_index(staging)
    for split, alias in (("train", "train"), ("validation", "eval"), ("test", "test")):
        if index.task_counts(alias).get(TASK) != len(rows[split]):
            raise ValueError(f"index count mismatch for {split}")
    index_path = staging / "axis_b_index_v1/index_manifest.json"
    metadata = json.loads(index_path.read_text())
    metadata["data_root"] = str(output)
    dump_json(index_path, metadata)
    staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true",
                        help="encode original overlapping splits for audit, not as training-ready data")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--guide", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--merges", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    args = parser.parse_args()
    from pipeline.runtime_host import require_compute_host
    require_compute_host("RAbD materialization", socket.gethostname())
    sys.setrecursionlimit(100000)
    started = time.monotonic()
    rows, audit = read_release(args.input_root)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(args.audit_output, audit)
    print(json.dumps(audit), flush=True)
    if not args.audit_only:
        result = materialize(args, rows, audit)
        print(json.dumps({"materialized": str(args.output_root), "counts": result["split_counts"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
