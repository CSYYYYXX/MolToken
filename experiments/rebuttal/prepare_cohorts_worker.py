#!/usr/bin/env python3
"""Select paired raw records and downstream exclusions on the CPU worker."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host
from experiments.rebuttal.common import DOMAINS, compact_row, iter_native, write_json


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}), flush=True)


def canonical_chunk(rows):
    from rdkit import Chem
    result = []
    for split, index, value in rows:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f'Official MOSES molecule failed parsing: {split}:{index}')
        result.append((split, index, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)))
    return result


def prepare_moses(data, output, workers):
    import orjson
    refs = data / 'benchmark/raw/moses/official_core13_dd7ed6'
    counts = Counter()
    exclusions = set()
    tasks = []
    pending = []
    for split in ('train', 'test', 'test_scaffolds'):
        with (refs / f'{split}.csv').open() as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                pending.append((split, index, row['SMILES']))
                if len(pending) == 10000:
                    tasks.append(pending)
                    pending = []
    if pending:
        tasks.append(pending)
    outputs = {s: (output / f'moses_{s}.jsonl').open('wb') for s in ('train', 'validation')}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part_index, part in enumerate(pool.map(canonical_chunk, tasks)):
                for official_split, index, smiles in part:
                    split = official_split
                    record_id = f'moses:{official_split}:{index}'
                    if official_split == 'train':
                        digest = hashlib.sha256(f'20260827\0unconditional_moses_v1\0{record_id}'.encode()).digest()
                        split = 'validation' if int.from_bytes(digest[:8], 'big') < int(.01 * 2**64) else 'train'
                        outputs[split].write(orjson.dumps(dict(sample_id=record_id, domain='small_molecule',
                            normalized=smiles, official_split=official_split, official_index=index, split=split)) + b'\n')
                    if split != 'train':
                        exclusions.add('smiles:' + hashlib.sha256(smiles.encode()).hexdigest())
                    counts[split] += 1
                if part_index % 10 == 0:
                    emit('moses_normalization', chunks=part_index + 1, total=len(tasks))
    finally:
        for handle in outputs.values():
            handle.close()
    write_json(output / 'moses_exclusion_content_keys.json', sorted(exclusions))
    write_json(output / 'moses_split.json', dict(counts=counts, method='sha256_record_id_fraction_v1',
        seed=20260827, fraction=.01, record_id='moses:{official_split}:{zero_based_csv_row}',
        note='New paired SFT split from official train; not assumed identical to the historical split.'))
    emit('moses_prepared', counts=dict(counts), excluded_contents=len(exclusions))
    return exclusions


def sample_shard(task):
    import numpy as np
    import orjson
    import zstandard
    entry, take, seed, data_root, output = task
    path = Path(data_root) / entry['path']
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(entry['records'], size=min(take, entry['records']), replace=False))
    selected = 0
    destination = Path(output) / f'{seed}.jsonl'
    with path.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with io.BufferedReader(stream) as lines, destination.open('wb') as out:
            for index, line in enumerate(lines):
                if selected == len(indices):
                    break
                if index != int(indices[selected]):
                    continue
                row = orjson.loads(line)
                if row['domain'] != entry['domain']:
                    raise ValueError(f'Domain mismatch: {path}:{index}')
                out.write(orjson.dumps(compact_row(row, entry['path'], index)) + b'\n')
                selected += 1
    if selected != len(indices):
        raise ValueError(f'Source shard shorter than manifest: {path}')
    return dict(path=str(destination), records=selected, domain=entry['domain'])


def collect_sample(entries, count, seed, data, output, workers):
    import numpy as np
    import orjson
    output.mkdir()
    counts = np.array([row['records'] for row in entries], dtype=np.float64)
    # Multinomial allocation followed by uniform within-shard sampling.
    allocation = np.random.default_rng(seed).multinomial(count, counts / counts.sum())
    tasks = [(entry, int(n), seed * 10000 + i, str(data), str(output))
             for i, (entry, n) in enumerate(zip(entries, allocation)) if n]
    parts = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(sample_shard, task) for task in tasks]
        for future in as_completed(futures):
            parts.append(future.result())
            if len(parts) % 10 == 0 or len(parts) == len(tasks):
                emit('sample_scan', directory=output.name, completed=len(parts), total=len(tasks),
                     elapsed_seconds=time.monotonic() - started)
    rows = []
    for part in sorted(parts, key=lambda x: x['path']):
        with Path(part['path']).open('rb') as handle:
            rows.extend(orjson.loads(line) for line in handle)
    np.random.default_rng(seed + 1).shuffle(rows)
    return rows


def write_rows(path, rows):
    import orjson
    with Path(path).open('wb') as handle:
        for row in rows:
            handle.write(orjson.dumps(row) + b'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=48)
    args = parser.parse_args()
    require_compute_host('rebuttal source cohort preparation')
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.root / 'data'
    excluded_contents = prepare_moses(data, args.output, args.workers)
    source_manifest = data / 'reports/native_qwen_tokenizer_lengths_20260925/prepared/inputs_full_sharded.json'
    manifest = json.loads(source_manifest.read_text())
    all_entries = manifest['scopes']['full']
    held_out, model_val = [], []
    held_sources, held_contents = set(), set(excluded_contents)
    for domain_index, domain in enumerate(DOMAINS):
        entries = [r for r in all_entries if r['domain'] == domain and '/train/' not in r['path']]
        rows = collect_sample(entries, min(10000, sum(r['records'] for r in entries)),
                              100 + domain_index, data, args.output / f'heldout_candidates_{domain}', args.workers)
        selected_test, selected_val = [], []
        used_contents = set()
        for row in rows:
            if row['content_key'] in excluded_contents or row['content_key'] in used_contents:
                continue
            if len(selected_test) < 4000:
                selected_test.append(row)
                held_sources.add(row['source_group'])
            elif row['source_group'] not in held_sources and len(selected_val) < 1000:
                selected_val.append(row)
            used_contents.add(row['content_key'])
            if len(selected_test) == 4000 and len(selected_val) == 1000:
                break
        if not selected_test or not selected_val:
            raise ValueError(f'No independent diagnostic/validation support for {domain}')
        held_out.extend(selected_test)
        model_val.extend(selected_val)
        held_sources.update(row['source_group'] for row in selected_val)
        held_contents.update(row['content_key'] for row in selected_test + selected_val)
        emit('heldout_selected', domain=domain, tokenizer_test=len(selected_test), model_validation=len(selected_val))
    write_rows(args.output / 'tokenizer_test.jsonl', held_out)
    write_rows(args.output / 'model_validation.jsonl', model_val)
    write_json(args.output / 'heldout_source_keys.json', sorted(held_sources))
    write_json(args.output / 'excluded_content_keys.json', sorted(held_contents))
    entries = [r for r in all_entries if '/train/' in r['path']]
    rows = collect_sample(entries, 1200000, 42, data, args.output / 'train_candidates', args.workers)
    selected = []
    exclusions = Counter()
    seen = set()
    with (args.output / 'tokenizer_exclusions.jsonl').open('w') as handle:
        for row in rows:
            reason = ('heldout_source' if row['source_group'] in held_sources else
                      'heldout_content' if row['content_key'] in held_contents else
                      'duplicate_content' if row['content_key'] in seen else None)
            if reason:
                exclusions[reason] += 1
                handle.write(json.dumps({'sample_id': row['sample_id'], 'reason': reason}) + '\n')
                continue
            seen.add(row['content_key'])
            selected.append(row)
            if len(selected) == 1000000:
                break
    if len(selected) < 1000000:
        raise ValueError(f'Insufficient eligible tokenizer records: {len(selected)}')
    write_rows(args.output / 'tokenizer_train.jsonl', selected)
    write_rows(args.output / 'frequency_train.jsonl', selected[:100000])
    write_json(args.output / 'summary.json', dict(source_manifest=str(source_manifest),
        tokenizer_train=1000000, frequency_train=100000, tokenizer_test=len(held_out),
        model_validation=len(model_val), train_domains=Counter(r['domain'] for r in selected),
        exclusions=exclusions, native_train_candidates=sum(r['records'] for r in entries),
        source_rule='Original source accession with UniProt/UniRef accession alias normalization; exact normalized content across AA domains',
        homology_clustering=False, full_source_sequence_alias_resolution=False))
    emit('COHORT_PREPARATION_COMPLETE', output=str(args.output))


if __name__ == '__main__':
    main()
