#!/usr/bin/env python3
"""Materialize the released CodonTransformer fine-tuning pool, not its pretraining pool.

The CSV's fine_tuned flag selects training sources, NOT a train/test split.
Both Axis stages use the same pool with a protein-grouped train-only holdout.
No published prediction file is read here and no official test split is invented.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import multiprocessing as mp
from pathlib import Path
import random
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host
from pipeline.finetune.codon_context import fit_coding_prefix
from pipeline.finetune.target_priority import pack_target_first
from pipeline.finetune.frozen_axis_errors import RingIdOverflow
from pipeline.finetune import sequence_codec_runtime as runtime
from cli.data.materialize_rabd_cdrh3_sft import dump_json, summarize

TASK = "codontransformer_protein_to_dna"
DATASET = "codontransformer_high_csi_trainpool_v1"
TEXT_LENGTHS = {}


def select_finetuning_row(row):
    flag = row["fine_tuned"]
    if flag not in ("True", "False"):
        raise ValueError(f"unexpected fine_tuned value: {flag!r}")
    return flag == "True"


def split_groups(references, fraction=.05, seed=42):
    # Keep identical retained protein sequences together even across organisms.
    groups = defaultdict(list)
    for ref in references:
        groups[ref["retained_protein"]].append(ref["id"])
    keys = sorted(groups)
    if len(keys) < 2 or not 0 < fraction < 1:
        raise ValueError("need multiple protein groups and a nonempty holdout fraction")
    random.Random(seed).shuffle(keys)
    n = max(1, min(len(keys) - 1, round(len(keys) * fraction)))
    return {rid for key in keys[:n] for rid in groups[key]}


def encode(kind, sequence):
    return runtime.encode_sequence(kind, sequence)[1]


def encode_condition_prefix(protein):
    """Known ring-slot limit: shorten only the condition, never change the codebook."""
    residues = len(protein)
    errors = []
    while True:
        try:
            return encode("protein", protein[:residues]), residues, errors
        except RingIdOverflow as exc:
            errors.append(str(exc))
            if residues == 1:
                raise
            residues = max(1, min(residues - 1, int(residues * exc.ring_capacity / exc.max_ring_id * .9)))


def fit_ring_limited_dna(protein, dna, text_n):
    """Full DNA failed the frozen ring bound; explicitly shorten an aligned source prefix."""
    from pipeline.finetune.codon_context import coding_prefix
    residues = min(256, len(protein) - 1)
    attempts = 0
    while True:
        aa, nt = coding_prefix(protein, dna, residues)
        attempts += 1
        try:
            condition, target = encode("protein", aa), encode("dna", nt)
        except RingIdOverflow:
            if residues == 1:
                raise
            residues = max(1, residues // 2)
            continue
        actual = len(condition) + len(target)
        available = 512 - text_n - 3
        if actual <= available:
            return {"protein": aa, "dna": nt, "condition_bpe_ids": condition,
                    "target_bpe_ids": target, "encoding_attempts": attempts}
        if residues == 1:
            raise ValueError("one AA/codon prefix exceeds context")
        residues = max(1, min(residues-1, int(residues * available / actual * .97)))


def materialize_pair(ref):
    from Bio.Seq import Seq
    protein, dna = ref["protein"], ref["dna"]
    # Author get_codon_table selects NCBI1 or11 for these organisms. Their
    # translation mappings are identical with cds=False (no forced initial M).
    translated = str(Seq(dna).translate(table=1, cds=False))
    if translated != protein + "*":
        raise ValueError(f"{ref['id']}: source DNA does not translate to its protein+stop")
    text_n = TEXT_LENGTHS[ref["organism"]]
    condition, condition_residues, condition_errors = encode_condition_prefix(protein)
    target_error = None
    fitted = None
    try:
        target = encode("dna", dna)
    except RingIdOverflow as exc:
        target_error = str(exc)
        fitted = fit_ring_limited_dna(protein, dna, text_n)
        target = fitted["target_bpe_ids"]
    record = {"id": ref["id"], "task": TASK, "target_type": "dna",
              "condition": {"type": "both", "text": ref["organism"],
                            "molecule_bpe_ids": condition},
              "target_bpe_ids": target,
              "meta": {"source_line": ref["source_line"], "organism": ref["organism"],
                       "source_fine_tuned": True, "sequence_roundtrip_exact": True,
                       "source_translation_exact": True, "source_dna_truncated": False,
                       "condition_ring_limit_errors": condition_errors,
                       "target_ring_limit_error": target_error,
                       "condition_source_residues": condition_residues,
                       "source_protein_residues": len(protein), "source_dna_bases": len(dna),
                       "protein_residues": len(protein), "dna_bases": len(dna),
                       "target_is_coding_fragment": False}}
    packed, budget = pack_target_first(record, text_tokens=text_n)
    retained_protein, retained_dna = protein, dna
    if packed is None and fitted is None:
        try:
            fitted = fit_coding_prefix(protein, dna, text_tokens=text_n,
                                      original_protein_bpe=len(condition), original_dna_bpe=len(target),
                                      encode_sequence=encode)
        except RingIdOverflow as exc:
            record["meta"]["prefix_fit_ring_limit_error"] = str(exc)
            fitted = fit_ring_limited_dna(protein, dna, text_n)
    if fitted is not None:
        retained_protein, retained_dna = fitted["protein"], fitted["dna"]
        record["condition"]["molecule_bpe_ids"] = fitted["condition_bpe_ids"]
        record["target_bpe_ids"] = fitted["target_bpe_ids"]
        record["meta"].update(source_dna_truncated=True, source_protein_truncated=True,
                              protein_residues=len(retained_protein), dna_bases=len(retained_dna),
                              target_is_coding_fragment=True, terminal_stop_codon_added=False,
                              encoding_attempts=fitted["encoding_attempts"])
        packed, budget = pack_target_first(record, text_tokens=text_n)
        if packed is None:
            raise ValueError(f"{ref['id']}: fitted prefix still exceeds context")
    return packed, {**ref, "retained_protein": retained_protein, "retained_dna": retained_dna,
                    "encoded_condition_source_residues": condition_residues,
                    "original_condition_bpe_tokens": None if condition_errors else len(condition),
                    "original_target_bpe_tokens": None if target_error else len(target), "budget": budget}


def main():
    global TEXT_LENGTHS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-root", type=Path, required=True)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--smoke-records", type=int, default=0,
                   help="diagnostic only; never publish a formally registered dataset")
    args = p.parse_args()
    require_compute_host("Codon fine-tuning materialization", socket.gethostname())
    started = time.monotonic()
    sys.setrecursionlimit(100000)
    csv.field_size_limit(10_000_000)
    staging = args.output_root.with_name(args.output_root.name + ".building")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    staging.mkdir(parents=True, exist_ok=False)
    selected, source_counts, species = [], Counter(), Counter()
    with args.source_csv.open() as handle:
        for line, row in enumerate(csv.DictReader(handle), 1):
            source_counts["total"] += 1
            if not select_finetuning_row(row):
                source_counts["pretraining_only_not_selected"] += 1
                continue
            protein = row["protein"]
            if protein.endswith(("_", "*")):
                protein = protein[:-1]
            dna = row["dna"]
            if (not protein or set(protein) - runtime.AA or set(dna) - set("ACGT")
                    or len(dna) != 3 * (len(protein) + 1)):
                raise ValueError(f"source line {line}: invalid coding pair")
            source_counts["fine_tuning_selected"] += 1
            species[row["organism"]] += 1
            selected.append({"id": f"codon:source:{line:07d}", "source_line": line,
                             "organism": row["organism"], "protein": protein,
                             "dna": dna, "GeneID": row["GeneID"], "description": row["description"]})
    if not args.smoke_records and source_counts["fine_tuning_selected"] != 45875:
        raise ValueError(f"published fine-tuning pool count changed: {source_counts}")
    if args.smoke_records:
        rng = random.Random(42)
        selected = sorted(rng.sample(selected, args.smoke_records), key=lambda r: r["source_line"])
    report = {"source_csv": str(args.source_csv), "source_counts": dict(source_counts),
              "fine_tuning_species": dict(species), "selected_records": len(selected),
              "scope": "diagnostic" if args.smoke_records else "released fine_tuned=True pool",
              "official_test_split_created": False, "source_rows_deleted": 0}
    dump_json(staging / "source_selection.json", report)
    print(json.dumps({"source_selection_complete": report}), flush=True)
    from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec
    from cli.evaluate.eval_axis_b_samples import BPEDecoder, FDDFSVocab
    from pipeline.finetune.axis_b_data import (AxisBSFTCollator, build_axis_b_index,
                                              MODEL_TOKEN_OFFSET, EOS_TOKEN_ID)
    from transformers import AutoTokenizer
    shared = args.shared_root
    vocab = shared / "data/fddfs_guide_recount_v1/vocab.json"
    merges = shared / "data/fddfs_bpe_full_prefix70_64k_mf80000_v1/codebook_fddfs_prefix70_mf80000_v1/merges.csv"
    runtime.CODEC = FrozenAxisCodec(shared / "data/fddfs_guide_recount_v1/guide_counts.json", vocab, merges)
    runtime.DECODER = BPEDecoder(merges, FDDFSVocab.load(vocab))
    tokenizer = AutoTokenizer.from_pretrained(str(shared / "data/pretrained_encoders/all-mpnet-base-v2"),
                                             local_files_only=True)
    TEXT_LENGTHS = {org: len(tokenizer(org)["input_ids"]) for org in species}
    encode("dna", "ACGT")
    # Fork only CPU preprocessing state, no GPU model or CUDA context.
    results = []
    with mp.get_context("fork").Pool(args.workers) as pool, (staging / "encoded_pairs.jsonl").open("x") as out:
        for result in pool.imap_unordered(materialize_pair, selected, chunksize=1):
            results.append(result)
            out.write(json.dumps({"record": result[0], "reference": result[1]}, separators=(",", ":")) + "\n")
            if len(results) % 100 == 0:
                out.flush()
                print(json.dumps({"encoded": len(results), "selected": len(selected),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    results.sort(key=lambda item: item[1]["source_line"])
    validation_ids = split_groups([ref for _, ref in results])
    collator = AxisBSFTCollator(tokenizer, qwen_max_context=512, condition_token_budget=512,
                               molecule_condition_mode="direct_qwen", strict_no_runtime_truncation=True,
                               return_tensors="list")
    counts = Counter()
    for start in range(0, len(results), 64):
        part = [record for record, _ in results[start:start+64]]
        batch = collator(part)
        for i, record in enumerate(part):
            n = len(record["target_bpe_ids"])
            expected = [x + MODEL_TOKEN_OFFSET for x in record["target_bpe_ids"]] + [EOS_TOKEN_ID]
            if batch["decoder_labels"][i][2:2+n+1] != expected:
                raise ValueError(f"{record['id']}: target loss labels changed")
            if batch["target_truncated"][i] or batch["molecule_truncated"][i]:
                raise ValueError(f"{record['id']}: runtime truncated a materialized sample")
            if sum(batch["full_attention_mask"][i]) != record["meta"]["full_context_tokens"]:
                raise ValueError(f"{record['id']}: collator context accounting differs")
    summaries = []
    for split in ("train", "validation"):
        path = staging / f"{split}.jsonl"
        with path.open("x") as out, (staging / f"{split}_references.jsonl").open("x") as refs:
            for record, reference in results:
                assigned = "validation" if record["id"] in validation_ids else "train"
                if assigned != split:
                    continue
                record["meta"].update(split=split, source_split="released_fine_tuning_pool")
                reference["split"] = split
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                refs.write(json.dumps(reference, separators=(",", ":")) + "\n")
                counts[split] += 1
        summaries.append({"dataset": "codon_diagnostic_only" if args.smoke_records else DATASET,
                          "split": split, "relative_source": path.name,
                          "records": counts[split], "output_size": path.stat().st_size})
    if sum(counts.values()) != len(selected) or len({r["id"] for r, _ in results}) != len(selected):
        raise ValueError("source coverage or unique IDs changed during materialization")
    shortened = sum(r["meta"]["source_dna_truncated"] for r, _ in results)
    manifest = {"complete": True, "task": TASK, "dataset_id": DATASET,
                "training_ready": not bool(args.smoke_records), "records": len(results),
                "files": 2, "file_summaries": summaries, "split_counts": dict(counts),
                "codec": runtime.CODEC.codebook, "source_selection": report,
                "DNA_targets_unchanged": len(results) - shortened, "DNA_targets_shortened": shortened,
                "condition_ring_limit_prefixes": sum(bool(r["meta"]["condition_ring_limit_errors"]) for r, _ in results),
                "target_ring_limit_prefixes": sum(bool(r["meta"]["target_ring_limit_error"]) for r, _ in results),
                "source_translation_all_exact": True, "sequence_roundtrip_all_exact": True,
                "full_context_tokens": summarize(r["meta"]["full_context_tokens"] for r, _ in results),
                "protein_condition_empty": sum(r["meta"]["protein_condition_empty"] for r, _ in results),
                "elapsed_seconds": time.monotonic() - started,
                "protocol": {"name": "axis-codon-high-csi-two-stage-v1-20260923", "serializer": "fddfs",
                             "model_token_offset": 3, "ids_on_disk": "raw_bpe_before_model_offset_without_bos_eos",
                             "validation_source": "5% retained-protein groups from released fine-tuning pool, seed42",
                             "early_stopping": "both stages max100 epochs; Full from Alignment/best",
                             "context": "512; complete DNA first; extreme DNA complete-codon prefix re-encoding",
                             "official_test_split_created": False,
                             "evaluation_ready": False,
                             "inference": "must not use hidden reference DNA to choose input truncation"}}
    dump_json(staging / "manifest.json", manifest)
    index = build_axis_b_index(staging)
    if index.task_counts("train").get(TASK) != counts["train"] or index.task_counts("eval").get(TASK) != counts["validation"]:
        raise ValueError("published index counts differ")
    index_path = staging / "axis_b_index_v1/index_manifest.json"
    index_meta = json.loads(index_path.read_text())
    index_meta["data_root"] = str(args.output_root)
    dump_json(index_path, index_meta)
    staging.rename(args.output_root)
    print(json.dumps({"CODON_DATA_READY": str(args.output_root), "split_counts": dict(counts),
                      "shortened": shortened, "elapsed_seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
