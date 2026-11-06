#!/usr/bin/env python3
"""Read-only E2a diagnostics; never change the frozen experiment artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import BASE_SIZE, DOMAINS, cpp_encoder, serialize_ids, write_json
from pipeline.runtime_host import require_compute_host

BUDGETS = (0, 20, 70, 200, 500, 1000, 2000, 4000, 8000, 16621)


def merge_rows(root, variant):
    with (root / 'tokenizers' / variant / 'merges.csv').open() as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def quick_tables(root):
    result = {}
    for variant in ('frequency', 'fixed'):
        summary = json.loads((root / 'tokenizers' / variant / 'summary.json').read_text())
        rows = merge_rows(root, variant)
        remaining = summary['input_tokens']
        curve = {'0': remaining}
        for i, row in enumerate(rows, 1):
            remaining -= row['real_merges']
            if i in BUDGETS:
                curve[str(i)] = remaining
        if remaining != summary['final_tokens']:
            raise ValueError('Merge history fails training token accounting')
        domains = {d: Counter() for d in DOMAINS}
        with (root / 'tokenizers' / f'{variant}_test_lengths.csv').open() as f:
            for row in csv.DictReader(f):
                domains[row['domain']].update(records=1, raw=int(row['before_bpe']),
                                              bpe=int(row['after_bpe']))
        result[variant] = dict(train_curve=curve, train_records=summary['input_sequences'],
            initial_unique_pairs=summary['initial_unique_pairs'], test_domains=domains)
    write_json(root / 'diagnostics/e2a_ordering/quick.json', result)
    print(json.dumps(dict(event='quick_tables', results=result)), flush=True)


def curve_task(task):
    root, variant, budget = task
    root = Path(root)
    import numpy as np
    import orjson
    from experiments.diagnostics.native_domain_bpe_probe import read_sequences
    from src.algorithms.compression import _cpp_bpe
    rows = merge_rows(root, variant)[:budget]
    encoder = _cpp_bpe.MinBPEEncoder([r['left'] for r in rows], [r['right'] for r in rows],
        list(range(len(rows))), [r['new_id'] for r in rows])
    metadata = [orjson.loads(x) for x in (root / 'tokenizers/test_records.jsonl').read_bytes().splitlines()]
    counts = {d: Counter() for d in DOMAINS}
    usage = {d: np.zeros(BASE_SIZE + budget, dtype=np.int64) for d in DOMAINS}
    for meta, ids in zip(metadata, read_sequences(root / 'tokenizers' / f'{variant}_test.bin'), strict=True):
        encoded = encoder.encode(ids.tolist())
        domain = meta['domain']
        counts[domain].update(records=1, raw=len(ids), bpe=len(encoded))
        usage[domain] += np.bincount(encoded, minlength=BASE_SIZE + budget)
    result = dict(variant=variant, merges=budget, domains={d: dict(counts[d],
        distinct_tokens=int((usage[d] > 0).sum()),
        base_tokens_emitted=int(usage[d][:BASE_SIZE].sum())) for d in DOMAINS})
    write_json(root / 'diagnostics/e2a_ordering' / f'curve_{variant}_{budget}.json', result)
    return result


def graph_profiles(root, max_per_domain=200):
    import numpy as np
    from cli.pretrain.serialize_fddfs_full import load_count_payload, build_guide
    from pipeline.graph.io import iter_graph_records
    from pipeline.graph.chem_graph_builder import _standard_protein_templates
    from pipeline.serialization.freq_dfs_core import compact_node_label
    vocab = json.loads((root / 'tokenizer_graphs/base_vocab.json').read_text())
    guide = build_guide(load_count_payload(root / 'tokenizer_graphs/frequency_guide.json'))
    encoders = {v: cpp_encoder(guide, vocab, fixed=v == 'fixed') for v in ('frequency', 'fixed')}
    counts = {v: {d: Counter() for d in DOMAINS} for v in encoders}
    roots = {v: {d: Counter() for d in DOMAINS} for v in encoders}
    seen = Counter()
    examples = []
    body_counts = _standard_protein_templates()['body_counts']
    manifest = json.loads((root / 'tokenizer_graphs/manifest.json').read_text())
    for part in manifest['parts']:
        if part['role'] != 'test':
            continue
        for graph, meta in iter_graph_records(Path(part['path'])):
            domain = meta['domain']
            if seen[domain] >= max_per_domain:
                continue
            seen[domain] += 1
            for variant, encoder in encoders.items():
                result = serialize_ids(encoder, graph, vocab['ring_base'])
                c = counts[variant][domain]
                c.update(result['stats'])
                c['records'] += 1
                order = np.asarray(result['old_to_new'])
                root_atom = int(np.argmin(order))
                feature = tuple(map(int, graph['node_features'][root_atom]))
                roots[variant][domain][str(feature)] += 1
                if domain in ('protein', 'peptide'):
                    seq = meta['normalized']
                    sizes = np.asarray([body_counts[aa] for aa in seq])
                    boundaries = np.concatenate(([0], sizes.cumsum()))
                    if int(boundaries[-1]) + 1 != graph['num_nodes']:
                        raise ValueError('Residue offsets do not match cached graph')
                    residue_index = min(int(np.searchsorted(boundaries, root_atom, side='right')) - 1, len(seq)-1)
                    c['root_at_first_residue'] += residue_index == 0
                    c['root_at_last_residue'] += residue_index == len(seq)-1
                    for i, aa in enumerate(seq):
                        if i in (0, len(seq)-1):
                            continue
                        positions = np.sort(order[boundaries[i]:boundaries[i+1]])
                        runs = 1 + int((np.diff(positions) != 1).sum())
                        c['internal_residues'] += 1
                        c['internal_residue_atom_runs'] += runs
                        c['internal_residues_contiguous'] += runs == 1
                    if seen[domain] <= 3:
                        examples.append(dict(sample_id=meta['sample_id'], domain=domain, variant=variant,
                            sequence=seq, root_atom=root_atom, root_feature=feature,
                            root_residue_index=residue_index, root_aa=seq[residue_index], stats=result['stats']))
        if all(seen[d] == max_per_domain for d in DOMAINS):
            break
    result = dict(sample='first 200 supported held-out records per domain; descriptive only',
        aggregation='All per-domain integer statistics are sums over sampled graphs, including max_ring_id and max_active_ring_ids.',
        domains=seen, variants={v: {d: dict(counts[v][d], root_features=roots[v][d]) for d in DOMAINS}
                               for v in encoders}, examples=examples,
        guide_top_nodes=[dict(feature=list(k), label=compact_node_label(k), count=v)
                         for k, v in guide.node_counts.most_common(15)])
    write_json(root / 'diagnostics/e2a_ordering/graph_profiles.json', result)
    print('GRAPH_PROFILES_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    require_compute_host('read-only E2a ordering diagnostics')
    out = args.run / 'diagnostics/e2a_ordering'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    quick_tables(args.run)
    graph_profiles(args.run)
    tasks = [(str(args.run), v, k) for v in ('frequency', 'fixed') for k in BUDGETS if k]
    curves = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(curve_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            curves.append(row)
            print(json.dumps(dict(event='curve_complete', variant=row['variant'], merges=row['merges'],
                domains=row['domains'], elapsed_seconds=time.monotonic()-start)), flush=True)
    write_json(out / 'complete.json', dict(curves=curves, seconds=time.monotonic()-start))
    print('E2A_DIAGNOSTICS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
