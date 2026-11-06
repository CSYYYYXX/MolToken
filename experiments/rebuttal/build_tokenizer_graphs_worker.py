#!/usr/bin/env python3
"""Build paired tokenizer graphs once, preserving row IDs and source groups."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host
from experiments.rebuttal.common import DEFAULT_EDGE, DOMAINS, write_json


def build_chunk(task):
    import numpy as np
    import orjson
    import _rebuttal_fddfs_cpp
    from pipeline.finetune.fddfs_encode import build_graph
    from pipeline.graph.io import write_compact_graph_zst
    role, number, rows, output = task
    counter = _rebuttal_fddfs_cpp.FastStackKCounter()
    records = []
    excluded = []
    alphabets = {'protein': set('ACDEFGHIKLMNPQRSTVWY'), 'peptide': set('ACDEFGHIKLMNPQRSTVWY'),
                 'dna': set('ACGT'), 'rna': set('ACGU')}
    for row in rows:
        domain, value = row['domain'], row['normalized']
        if domain in alphabets and not set(value) <= alphabets[domain]:
            excluded.append(dict(sample_id=row['sample_id'], domain=domain, reason='unsupported_residue_alphabet'))
            continue
        graph = build_graph(target_type=domain, sample_id=row['sample_id'],
                            smiles=value if domain == 'small_molecule' else None,
                            sequence=value if domain != 'small_molecule' else None)
        src, dst = graph['edges']
        counter.add_graph(int(graph['num_nodes']), np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64), np.asarray(graph['node_features'], dtype=np.int64).reshape(-1, 6),
            np.asarray(graph['edge_features'], dtype=np.int64).reshape(-1, 3))
        records.append((graph, row))
    output = Path(output)
    graph_path = output / f'{role}_{number:05d}.cgraph.zst'
    if records:
        write_compact_graph_zst(graph_path, records)
    counts = counter.result()
    counts_path = graph_path.with_suffix('.counts.pkl')
    counts_path.write_bytes(pickle.dumps(counts, protocol=5))
    exclusion_path = graph_path.with_suffix('.exclusions.jsonl')
    with exclusion_path.open('wb') as handle:
        for row in excluded:
            handle.write(orjson.dumps(row) + b'\n')
    return dict(role=role, chunk=number, path=str(graph_path) if records else None,
                counts=str(counts_path), records=len(records), requested=len(rows), excluded=len(excluded),
                domains=dict(Counter(meta['domain'] for _, meta in records)),
                base_nodes=sum(int(g['num_nodes']) for g, _ in records))


def count_payload(parts):
    from pipeline.serialization.stackk_fast import _tuple_counter, _triplet_counter
    from pipeline.serialization.freq_dfs_core import encode_feature
    nodes, edges, triplets, stats = Counter(), Counter(), Counter(), Counter()
    for part in parts:
        raw = pickle.loads(Path(part['counts']).read_bytes())
        nodes.update(_tuple_counter(raw['node_counts']))
        edges.update(_tuple_counter(raw['edge_counts']))
        triplets.update(_triplet_counter(raw['triplet_counts']))
        stats.update(raw['stats'])
    return dict(node_counts={encode_feature('N', k): v for k, v in nodes.items()},
                edge_counts={encode_feature('E', k): v for k, v in edges.items()},
                triplet_counts={'|'.join((encode_feature('N', a), encode_feature('E', b), encode_feature('N', c))): v
                                for (a, b, c), v in triplets.items()},
                default_edge_feature=list(DEFAULT_EDGE), stats=dict(stats))


def main():
    import orjson
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=48)
    args = parser.parse_args()
    require_compute_host('rebuttal tokenizer graphs')
    root = args.run
    manifests = root / 'manifests'
    output = root / 'tokenizer_graphs'
    output.mkdir(exist_ok=False)
    test = [orjson.loads(x) for x in (manifests / 'tokenizer_test.jsonl').read_bytes().splitlines()]
    val = [orjson.loads(x) for x in (manifests / 'model_validation.jsonl').read_bytes().splitlines()]
    test_sources = {r['source_group'] for r in test}
    test_contents = {r['content_key'] for r in test}
    validation = [r for r in val if r['source_group'] not in test_sources and r['content_key'] not in test_contents]
    with (manifests / 'model_validation_disjoint.jsonl').open('wb') as handle:
        for row in validation:
            handle.write(orjson.dumps(row) + b'\n')
    held_sources = test_sources | {r['source_group'] for r in validation}
    held_contents = test_contents | {r['content_key'] for r in validation}
    tasks = []
    requested = Counter()
    for role, filename in (('train', 'tokenizer_train.jsonl'), ('test', 'tokenizer_test.jsonl'),
                           ('validation', 'model_validation_disjoint.jsonl')):
        chunk, number = [], 0
        with (manifests / filename).open('rb') as handle:
            for line in handle:
                row = orjson.loads(line)
                if role == 'train' and (row['source_group'] in held_sources or row['content_key'] in held_contents):
                    raise ValueError(f'Train/holdout overlap: {row["sample_id"]}')
                chunk.append(row)
                requested[role] += 1
                if len(chunk) == 1000:
                    tasks.append((role, number, chunk, str(output)))
                    number += 1
                    chunk = []
            if chunk:
                tasks.append((role, number, chunk, str(output)))
    parts = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_chunk, task) for task in tasks]
        for future in as_completed(futures):
            parts.append(future.result())
            if len(parts) % 20 == 0 or len(parts) == len(tasks):
                print(json.dumps(dict(event='graph_progress', completed=len(parts), total=len(tasks),
                    elapsed_seconds=time.monotonic()-start, encoded=sum(p['records'] for p in parts),
                    excluded=sum(p['excluded'] for p in parts))), flush=True)
    parts.sort(key=lambda p: (p['role'], p['chunk']))
    train = [p for p in parts if p['role'] == 'train']
    frequency = [p for p in train if p['chunk'] < 100]
    write_json(root / 'tokenizer_graphs/graph_guide.json', count_payload(train))
    write_json(root / 'tokenizer_graphs/frequency_guide.json', count_payload(frequency))
    # The base grammar is frozen; learned frequencies and merges are new.
    data = root.parent.parent / 'data'
    vocab = json.loads((data / 'fddfs_guide_recount_v1/vocab.json').read_text())
    if vocab['ring_base'] != 334 or vocab['default_edge_feature'] != list(DEFAULT_EDGE):
        raise ValueError('Unexpected base grammar; inspect semantic compatibility')
    write_json(root / 'tokenizer_graphs/base_vocab.json', vocab)
    write_json(root / 'tokenizer_graphs/manifest.json', dict(parts=parts, requested=requested,
        validation_removed_cross_cohort=len(val)-len(validation), validation_records=len(validation),
        frequency_training_requested=100000, frequency_training_graphs=sum(p['records'] for p in frequency)))
    print('TOKENIZER_GRAPHS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
