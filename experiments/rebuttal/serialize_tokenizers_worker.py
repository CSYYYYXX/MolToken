#!/usr/bin/env python3
"""Serialize the paired cohorts; learn four independent, matched-size BPEs."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import (BASE_SIZE, CONTENT_VOCAB, DEFAULT_EDGE, HEADER,
    NAMESPACE, IntWriter, cpp_encoder, serialize_ids, write_json)
from pipeline.runtime_host import require_compute_host


def initialize(root):
    global ROOT, VOCAB, ENCODERS, ALPHABET, INVERSE
    from cli.pretrain.serialize_fddfs_full import load_count_payload, build_guide
    ROOT = Path(root)
    VOCAB = json.loads((ROOT / 'tokenizer_graphs/base_vocab.json').read_text())
    guide = build_guide(load_count_payload(ROOT / 'tokenizer_graphs/graph_guide.json'))
    freq = build_guide(load_count_payload(ROOT / 'tokenizer_graphs/frequency_guide.json'))
    ENCODERS = {'graph': cpp_encoder(guide, VOCAB), 'frequency': cpp_encoder(freq, VOCAB),
                'fixed': cpp_encoder(freq, VOCAB, fixed=True)}
    ALPHABET = json.loads((ROOT / 'tokenizers/alphabet.json').read_text())
    INVERSE = {v: k for k, v in VOCAB['token_to_id'].items()}
    for i in range(VOCAB['ring_base'], BASE_SIZE):
        INVERSE[i] = 'R' + str(i - VOCAB['ring_base'] + 1)


def serialize_part(part):
    import orjson
    from pipeline.graph.io import iter_graph_records
    from pipeline.serialization.fddfs import FDDFSResult, verify_roundtrip
    role, number = part['role'], part['chunk']
    directory = ROOT / 'tokenizers/parts' / f'{role}_{number:05d}'
    directory.mkdir()
    variants = ['graph', 'native']
    if role == 'test' or (role == 'train' and number < 100):
        variants += ['frequency', 'fixed']
    writers = {k: IntWriter(directory / f'{k}.bin',
                  sum(map(len, ALPHABET.values())) - 1 if k == 'native' else BASE_SIZE - 1)
               for k in variants}
    checks = Counter()
    with (directory / 'records.jsonl').open('wb') as out:
        for graph, row in iter_graph_records(Path(part['path'])):
            for variant in variants:
                if variant == 'native':
                    mapping = ALPHABET[NAMESPACE[row['domain']]]
                    ids = [mapping[c] for c in row['normalized']]
                else:
                    result = serialize_ids(ENCODERS[variant], graph, VOCAB['ring_base'])
                    ids = list(result['ids'])
                    if role == 'test':
                        inverse = {i: int(j) for i, j in enumerate(result['old_to_new'])}
                        decoded = FDDFSResult([INVERSE[int(i)] for i in ids], inverse,
                                              result['stats'], DEFAULT_EDGE)
                        if not verify_roundtrip(graph, decoded):
                            raise ValueError(f'Roundtrip failed: {variant} {row["sample_id"]}')
                        checks[variant] += 1
                writers[variant].write(ids)
            out.write(orjson.dumps(dict(sample_id=row['sample_id'], domain=row['domain'],
                source_group=row['source_group'], content_key=row['content_key'])) + b'\n')
    for writer in writers.values():
        writer.close()
    return dict(role=role, chunk=number, directory=str(directory), checks=dict(checks),
                variants={k: dict(records=v.records, tokens=v.tokens, max_id=v.max_id)
                          for k, v in writers.items()})


def concatenate(parts, output, variant, role):
    selected = sorted([p for p in parts if p['role'] == role and variant in p['variants']], key=lambda p: p['chunk'])
    counts = [sum(p['variants'][variant][key] for p in selected) for key in ('records', 'tokens')]
    maximum = selected[0]['variants'][variant]['max_id']
    with output.open('wb') as out:
        out.write(HEADER.pack(b'YTTMIBPE', 1, *counts, maximum))
        for part in selected:
            with (Path(part['directory']) / f'{variant}.bin').open('rb') as source:
                source.seek(HEADER.size)
                shutil.copyfileobj(source, out, length=8*1024*1024)
    return dict(path=str(output), records=counts[0], tokens=counts[1], max_id=maximum)


def main():
    import orjson
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    require_compute_host('rebuttal serialization and BPE preparation')
    root = args.run
    out = root / 'tokenizers'
    out.mkdir(exist_ok=False)
    (out / 'parts').mkdir()
    chars = {'aa': set('ACDEFGHIKLMNPQRSTVWY'), 'dna': set('ACGT'), 'rna': set('ACGU'), 'smiles': set()}
    with (root / 'manifests/tokenizer_train.jsonl').open('rb') as source:
        for line in source:
            row = orjson.loads(line)
            if row['domain'] == 'small_molecule':
                chars['smiles'].update(row['normalized'])
    symbols = sorted((namespace, c) for namespace, letters in chars.items() for c in letters)
    alphabet = {ns: {} for ns in sorted(chars)}
    for i, (ns, c) in enumerate(symbols):
        alphabet[ns][c] = i
    write_json(out / 'alphabet.json', alphabet)
    manifest = json.loads((root / 'tokenizer_graphs/manifest.json').read_text())
    parts, start = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(root),)) as pool:
        futures = [pool.submit(serialize_part, part) for part in manifest['parts'] if part['records']]
        for future in as_completed(futures):
            parts.append(future.result())
            if len(parts) % 20 == 0 or len(parts) == len(futures):
                print(json.dumps(dict(event='serialize_progress', completed=len(parts), total=len(futures),
                    seconds=time.monotonic()-start)), flush=True)
    corpora = {}
    for variant in ('frequency', 'fixed', 'graph', 'native'):
        corpora[variant] = {}
        for role in ('train', 'test'):
            corpora[variant][role] = concatenate(parts, out / f'{variant}_{role}.bin', variant, role)
    # Test rows have exactly the same order in all four encodings.
    with (out / 'test_records.jsonl').open('wb') as handle:
        for part in sorted([p for p in parts if p['role'] == 'test'], key=lambda p: p['chunk']):
            with (Path(part['directory']) / 'records.jsonl').open('rb') as source:
                shutil.copyfileobj(source, handle)
    write_json(out / 'serialized.json', dict(parts=parts, corpora=corpora,
        graph_roundtrips={v: sum(p['checks'].get(v, 0) for p in parts) for v in ('graph', 'frequency', 'fixed')},
        native_base_size=len(symbols), graph_base_size=BASE_SIZE))
    print('TOKENIZER_SERIALIZATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
