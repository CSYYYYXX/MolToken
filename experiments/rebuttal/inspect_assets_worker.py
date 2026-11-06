#!/usr/bin/env python3
"""Inspect E2/E3 inputs on the allocated CPU worker; never load model weights."""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import shutil
import socket
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_compute_host('rebuttal asset inspection')
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.root / 'data'
    result = {'host': socket.gethostname(), 'disk_free': shutil.disk_usage(args.root).free}
    result['data_directories'] = sorted(p.name for p in data.iterdir() if p.is_dir())
    print(json.dumps(result), flush=True)
    import zstandard
    manifest_path = data / 'reports/native_qwen_tokenizer_lengths_20260925/prepared/inputs_full_sharded.json'
    manifest = json.loads(manifest_path.read_text())
    result['native_manifest'] = str(manifest_path)
    result['native_samples'] = []
    counts = Counter()
    selected = set()
    for entry in manifest['scopes']['full']:
        counts[entry['domain']] += entry['records']
        identity = (entry['domain'], '/train/' in entry['path'])
        if identity in selected:
            continue
        selected.add(identity)
        path = data / entry['path']
        with path.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with io.TextIOWrapper(stream) as lines:
                row = json.loads(next(lines))
        sample = {'input': entry, 'row': row}
        result['native_samples'].append(sample)
        print('NATIVE_SAMPLE', json.dumps(sample)[:6000], flush=True)
    result['native_counts'] = dict(counts)
    from pipeline.graph.io import iter_graph_records
    graph_root = data / 'graphs_heavy_atom_normv2_compact_v1'
    result['graph_samples'] = []
    for domain in ('small_molecule', 'peptide', 'protein', 'dna', 'rna'):
        paths = sorted((graph_root / domain).rglob('*.cgraph.zst'))
        if not paths:
            paths = sorted((graph_root / domain).rglob('*.pkl.zst'))
        if not paths:
            raise FileNotFoundError(f'No graph shard for {domain} under {graph_root}')
        records = iter_graph_records(paths[0])
        try:
            graph, meta = next(records)
        finally:
            records.close()
        row = {'domain': domain, 'path': str(paths[0]), 'shards': len(paths),
               'graph_keys': list(graph), 'sample_id': graph.get('sample_id'),
               'properties': graph.get('properties'), 'sequence': graph.get('sequence'),
               'smiles': graph.get('smiles'), 'meta': meta}
        result['graph_samples'].append(row)
        print('GRAPH_SAMPLE', json.dumps(row, default=str)[:6000], flush=True)
    (args.output / 'assets.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print('ASSET_INSPECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
