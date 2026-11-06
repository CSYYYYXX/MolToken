#!/usr/bin/env python3
"""Reconstruct reference weights from the author's released natural genomes.

Compare with published metrics without choosing weights to improve model scores.
This never supplies training data or uses model-generated DNA as a reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.runtime_host import require_compute_host

_OFFICIAL = None
_GENOMES = None
_PROBES = None


def calculate_reference(organism):
    from Bio.Data import CodonTable
    sequences = _GENOMES[organism]
    weights = _OFFICIAL.get_CSI_weights(sequences)
    counts = Counter(codon for dna in sequences for codon in
                     (dna[i:i + 3] for i in range(0, len(dna), 3)))
    table = CodonTable.unambiguous_dna_by_id[11]
    amino_codons = defaultdict(list)
    for codon, amino in table.forward_table.items():
        amino_codons[amino].append(codon)
    amino_codons["_"] = table.stop_codons
    frequencies = {}
    for amino, codons in amino_codons.items():
        codons = sorted(codons)
        total = sum(counts[codon] for codon in codons)
        if not total:
            raise ValueError(f"no {amino} codons in genome {organism}")
        frequencies[amino] = (codons, [counts[codon] / total for codon in codons])
    comparisons = []
    for row in _PROBES[organism]:
        for model in ("natural", "base", "finetune"):
            dna = row[model + "_dna"]
            actual = {"CSI": _OFFICIAL.get_CSI_value(dna, weights),
                      "CFD": _OFFICIAL.get_cfd(dna, frequencies, threshold=0.3)}
            expected = {metric: float(row[model + "_" + metric]) for metric in actual}
            if not all(math.isfinite(x) for x in [*actual.values(), *expected.values()]):
                raise ValueError(f"non-finite reference comparison {organism}/{row['GeneID']}/{model}")
            comparisons.append({"GeneID": row["GeneID"], "model": model,
                                "actual": actual, "published": expected})
    match = {metric: all(math.isclose(row["actual"][metric], row["published"][metric],
                                    rel_tol=1e-9, abs_tol=1e-9) for row in comparisons)
             for metric in ("CSI", "CFD")}
    errors = {metric: max(abs(row["actual"][metric] - row["published"][metric])
                          for row in comparisons) for metric in ("CSI", "CFD")}
    result = {"organism": organism, "natural_gene_count": len(sequences),
              "natural_codon_count": sum(counts.values()), "codon_counts": dict(sorted(counts.items())),
              "csi_weights": weights, "codon_frequencies": frequencies,
              "comparison_count": len(comparisons), "published_metric_matches": match,
              "max_absolute_errors": errors, "comparisons": comparisons}
    print(json.dumps({"reference_built": organism, "genes": len(sequences),
                      "published_matches": match, "errors": errors}), flush=True)
    return result


def main():
    global _OFFICIAL, _GENOMES, _PROBES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--published-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require_compute_host("CodonTransformer natural-genome metric references", socket.gethostname())
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output_root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.assets / "CAI_PyPI-2.0.1-py3-none-any.whl"))
    spec = importlib.util.spec_from_file_location("codon_author_metrics", args.assets / "CodonEvaluation.py")
    _OFFICIAL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_OFFICIAL)
    _GENOMES, _PROBES = defaultdict(list), defaultdict(list)
    started = time.monotonic()
    csv.field_size_limit(10_000_000)
    with args.published_csv.open(newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            dna, organism = row["natural_dna"], row["organism"]
            if not dna or len(dna) % 3 or set(dna) - set("ACGT") or not organism:
                raise ValueError(f"invalid natural reference DNA/host at CSV row {row_number}")
            _GENOMES[organism].append(dna)
            if len(_PROBES[organism]) < 8:
                _PROBES[organism].append(row)
            if row_number % 100000 == 0:
                print(json.dumps({"natural_genes_loaded": row_number}), flush=True)
    print(json.dumps({"stage": "building_references", "organisms": len(_GENOMES),
                      "natural_genes": sum(map(len, _GENOMES.values()))}), flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(calculate_reference, sorted(_GENOMES)))
    results.sort(key=lambda row: row["organism"])
    for row in results:
        row["reference_source"] = "all natural_dna rows for this organism in author Zenodo13262517"
    with (args.output_root / "references.json").open("x") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    summary = {"complete": True, "source_csv": str(args.published_csv),
               "reference_policy": "author natural genes only; no base/finetune/Axis predictions in weights",
               "organisms": len(results), "natural_genes": sum(row["natural_gene_count"] for row in results),
               "comparison_sequences": sum(row["comparison_count"] for row in results),
               "elapsed_seconds": time.monotonic() - started,
               "all_published_matches": {metric: all(row["published_metric_matches"][metric] for row in results)
                                          for metric in ("CSI", "CFD")},
               "by_organism": [{key: row[key] for key in ("organism", "natural_gene_count", "comparison_count",
                                                        "published_metric_matches", "max_absolute_errors")}
                               for row in results],
               "training_data_created": False, "reference_chosen_by_model_performance": False}
    with (args.output_root / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary), flush=True)
    if not all(summary["all_published_matches"].values()):
        print("REFERENCE_COMPARISON_MISMATCH: derived references are NOT a reproduced published scoring setup", flush=True)


if __name__ == "__main__":
    main()
