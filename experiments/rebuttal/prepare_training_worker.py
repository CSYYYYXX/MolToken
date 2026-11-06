#!/usr/bin/env python3
"""Index the complete raw pool, then stream paired tokenization in sampled order."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import mmap
from pathlib import Path
import struct
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import (CONTENT_VOCAB, DOMAINS, IntWriter, iter_native,
                                        sequence_key, source_key, write_json)
from pipeline.runtime_host import require_compute_host


def init_catalog(run):
    global ROOT, EXCLUDED, HELD_SOURCES
    ROOT = Path(run)
    EXCLUDED = set(json.loads((ROOT / 'manifests/excluded_content_keys.json').read_text()))
    EXCLUDED.update(json.loads((ROOT / 'manifests/moses_exclusion_content_keys.json').read_text()))
    HELD_SOURCES = set(json.loads((ROOT / 'manifests/heldout_source_keys.json').read_text()))


def catalog_part(task):
    import numpy as np
    index, entry, data = task
    output = ROOT / 'raw_catalog' / f'{index:04d}'
    output.mkdir()
    offsets, original, counts = [], [], Counter()
    total = 0
    with (output / 'strings.bin').open('wb', buffering=1024*1024) as handle:
        for i, row in enumerate(iter_native(Path(data) / entry['path'])):
            total += 1
            if row['domain'] != entry['domain'] or row['split'] != 'train':
                raise ValueError(f'Unexpected input provenance: {entry["path"]}:{i}')
            if sequence_key(row) in EXCLUDED:
                counts['heldout_content'] += 1
                continue
            if source_key(row) in HELD_SOURCES:
                counts['heldout_source'] += 1
                continue
            value = row['normalized'].encode('ascii')
            offsets.append(handle.tell())
            original.append(i)
            handle.write(struct.pack('<I', len(value)))
            handle.write(value)
    if total != entry['records']:
        raise ValueError(f'Raw manifest record mismatch: {entry["path"]}: {total}')
    np.save(output / 'offsets.npy', np.asarray(offsets, dtype='<u8'))
    np.save(output / 'original_indices.npy', np.asarray(original, dtype='<u4'))
    result = dict(index=index, domain=entry['domain'], input_path=entry['path'], directory=str(output),
        input_records=total, records=len(offsets), exclusions=dict(counts))
    write_json(output / 'complete.json', result)
    return result


def init_encode(run):
    global ROOT, TOKENIZER, CATALOG, CUMULATIVE, MAPS
    import numpy as np
    from experiments.rebuttal.tokenizer import PairedTokenizer
    ROOT = Path(run)
    TOKENIZER = PairedTokenizer(ROOT)
    manifest = json.loads((ROOT / 'raw_catalog/manifest.json').read_text())
    CATALOG = manifest['parts']
    CUMULATIVE = np.cumsum([p['records'] for p in CATALOG])
    MAPS = {}


def load_raw(gid):
    import numpy as np
    part_index = int(np.searchsorted(CUMULATIVE, gid, side='right'))
    local = int(gid - (CUMULATIVE[part_index-1] if part_index else 0))
    part = CATALOG[part_index]
    if part_index not in MAPS:
        directory = Path(part['directory'])
        with (directory / 'strings.bin').open('rb') as f:
            strings = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        MAPS[part_index] = (strings, np.load(directory / 'offsets.npy', mmap_mode='r'))
    strings, offsets = MAPS[part_index]
    start = int(offsets[local])
    length, = struct.unpack_from('<I', strings, start)
    return dict(sample_id=f'catalog:{gid}', domain=part['domain'],
                normalized=strings[start+4:start+4+length].decode('ascii'))


def encode_chunk(task):
    import numpy as np
    import orjson
    name, index, ids = task
    directory = ROOT / 'streams' / name / f'{index:06d}'
    directory.mkdir(parents=True, exist_ok=False)
    writers = {k: IntWriter(directory / f'{k}.bin', CONTENT_VOCAB-1) for k in ('graph', 'native')}
    domains, accepted, reasons = [], [], Counter()
    start = time.monotonic()
    with (directory / 'excluded.jsonl').open('wb') as exclusions:
        for gid in ids:
            row = load_raw(int(gid))
            pair, reason = TOKENIZER.encode(row)
            if reason:
                reasons[reason] += 1
                exclusions.write(orjson.dumps(dict(catalog_id=int(gid), domain=row['domain'], reason=reason)) + b'\n')
                continue
            for writer, values in zip(writers.values(), pair):
                writer.write(values)
            accepted.append(int(gid))
            domains.append(DOMAINS.index(row['domain']))
    for writer in writers.values():
        writer.close()
    np.save(directory / 'ids.npy', np.asarray(accepted, dtype='<u8'))
    np.save(directory / 'domains.npy', np.asarray(domains, dtype='u1'))
    result = dict(index=index, candidates=len(ids), records=len(accepted), exclusions=dict(reasons),
        input_tokens={k: w.tokens + 3*w.records for k, w in writers.items()},
        predicted_tokens={k: w.tokens + w.records for k, w in writers.items()},
        seconds=time.monotonic()-start)
    write_json(directory / 'complete.json', result)
    return result


def encode_moses_chunk(task):
    import numpy as np
    import orjson
    role, index, rows, run = task
    from experiments.rebuttal.tokenizer import PairedTokenizer
    tokenizer = PairedTokenizer(run)
    directory = Path(run) / 'moses' / role / f'{index:05d}'
    directory.mkdir(parents=True, exist_ok=False)
    writers = {k: IntWriter(directory / f'{k}.bin', CONTENT_VOCAB-1) for k in ('graph', 'native')}
    accepted, reasons = [], Counter()
    with (directory / 'excluded.jsonl').open('wb') as exclusions:
        for row in rows:
            pair, reason = tokenizer.encode(row)
            if reason:
                reasons[reason] += 1
                exclusions.write(orjson.dumps(dict(sample_id=row['sample_id'], reason=reason)) + b'\n')
                continue
            for writer, values in zip(writers.values(), pair):
                writer.write(values)
            accepted.append(row['sample_id'])
    for writer in writers.values():
        writer.close()
    (directory / 'ids.json').write_text(json.dumps(accepted))
    result = dict(role=role, index=index, directory=str(directory), requested=len(rows), records=len(accepted),
        exclusions=dict(reasons), input_tokens={k: w.tokens + 3*w.records for k,w in writers.items()})
    write_json(directory / 'complete.json', result)
    return result


def prepare_moses(args):
    import orjson
    tasks = []
    for role in ('train', 'validation'):
        rows, index = [], 0
        with (args.run / 'manifests' / f'moses_{role}.jsonl').open('rb') as handle:
            for line in handle:
                rows.append(orjson.loads(line))
                if len(rows) == 10000:
                    tasks.append((role,index,rows,str(args.run)))
                    rows, index = [], index+1
            if rows:
                tasks.append((role,index,rows,str(args.run)))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(encode_moses_chunk, task) for task in tasks]):
            results.append(f.result())
            if len(results) % 10 == 0:
                print(json.dumps(dict(event='moses_encoded', completed=len(results), total=len(tasks))), flush=True)
    results.sort(key=lambda p:(p['role'],p['index']))
    write_json(args.run / 'moses/manifest.json', dict(parts=results))
    print('MOSES_PAIRED_DATA_COMPLETE', flush=True)


def prepare_catalog(args):
    data = args.run.parent.parent / 'data'
    manifest = json.loads((data / 'reports/native_qwen_tokenizer_lengths_20260925/prepared/inputs_full_sharded.json').read_text())
    train = [r for r in manifest['scopes']['full'] if '/train/' in r['path']]
    (args.run / 'raw_catalog').mkdir(exist_ok=False)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_catalog, initargs=(str(args.run),)) as pool:
        for f in as_completed([pool.submit(catalog_part, (i,r,str(data))) for i,r in enumerate(train)]):
            results.append(f.result())
            print(json.dumps(dict(event='catalog_progress', completed=len(results), total=len(train),
                records=sum(p['records'] for p in results))), flush=True)
    results.sort(key=lambda p:p['index'])
    write_json(args.run / 'raw_catalog/manifest.json', dict(parts=results, sampling='Uniform random permutation of all catalog rows; common support rejected identically for A/C.'))
    print('FULL_RAW_CATALOG_COMPLETE', flush=True)


def prepare_stream(args):
    import numpy as np
    manifest = json.loads((args.run / 'raw_catalog/manifest.json').read_text())
    parts = manifest['parts']
    total = sum(p['records'] for p in parts)
    name = f'{args.scope}_{args.seed}'
    directory = args.run / 'streams' / name
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    if args.scope == 'joint':
        indices = rng.permutation(total).astype('<u8')
    else:
        starts = np.cumsum([0]+[p['records'] for p in parts])
        ids = [np.arange(starts[i], starts[i+1], dtype='<u8') for i,p in enumerate(parts) if p['domain']=='small_molecule']
        indices = np.concatenate(ids)
        rng.shuffle(indices)
    np.save(directory / 'candidate_order.npy', indices)
    write_json(directory / 'protocol.json', dict(seed=args.seed, scope=args.scope, candidates=len(indices),
        raw_pool_records=total, max_input_tokens=args.target_tokens, chunk_size=args.chunk_size,
        stop_policy='Materialize enough consecutive chunks for both representations; all candidates belong to the full eligible raw pool.'))
    accumulated = Counter()
    results, next_index = {}, 0
    start = time.monotonic()
    # Keep only a bounded number of tasks in flight; publish each complete chunk atomically.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_encode, initargs=(str(args.run),)) as pool:
        pending, submitted = {}, 0
        chunks = (len(indices)+args.chunk_size-1)//args.chunk_size
        def submit():
            nonlocal submitted
            i = submitted
            future = pool.submit(encode_chunk, (name, i, indices[i*args.chunk_size:(i+1)*args.chunk_size]))
            pending[future] = i
            submitted += 1
        for _ in range(min(args.workers*2, chunks)):
            submit()
        stop = False
        while pending:
            future = next(as_completed(pending))
            index = pending.pop(future)
            results[index] = future.result()
            while next_index in results:
                accumulated.update(results[next_index]['input_tokens'])
                next_index += 1
            stop = stop or min(accumulated.get('graph',0), accumulated.get('native',0)) >= args.target_tokens
            if submitted < chunks and not stop:
                submit()
            if len(results)%20 == 0:
                write_json(directory / 'progress.json', dict(chunks=len(results), contiguous_chunks=next_index,
                    input_tokens=dict(accumulated), seconds=time.monotonic()-start))
                print(json.dumps(dict(event='paired_stream_progress', name=name, chunks=len(results),
                    contiguous_chunks=next_index, input_tokens=dict(accumulated))), flush=True)
    write_json(directory / 'complete.json', dict(chunks=len(results), input_tokens=dict(accumulated),
        candidate_epoch_exhausted=submitted==chunks, materialized_records=sum(r['records'] for r in results.values())))
    print('PAIRED_STREAM_COMPLETE', name, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--mode', choices=['moses','catalog','stream'], required=True)
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--scope', choices=['joint','small'], default='joint')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--chunk-size', type=int, default=10000)
    p.add_argument('--target-tokens', type=int, default=2100000000)
    args = p.parse_args()
    require_compute_host('rebuttal full-pool preparation')
    {'moses':prepare_moses, 'catalog':prepare_catalog, 'stream':prepare_stream}[args.mode](args)


if __name__ == '__main__':
    main()
