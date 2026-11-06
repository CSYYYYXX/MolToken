#!/usr/bin/env python3
"""Audit BPE fit/apply consistency and ring-ID variation without changing runs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import BASE_SIZE, DOMAINS, cpp_encoder, serialize_ids, write_json
from experiments.rebuttal.tokenizer import BPE
from pipeline.runtime_host import require_compute_host


def train_audit(task):
    import orjson
    from experiments.diagnostics.native_domain_bpe_probe import read_sequences
    root, variant = Path(task[0]), task[1]
    model = BPE(root / 'tokenizers' / variant / 'merges.csv', BASE_SIZE)
    manifest = json.loads((root / 'tokenizers/serialized.json').read_text())
    parts = sorted((p for p in manifest['parts'] if p['role'] == 'train' and variant in p['variants']),
                   key=lambda p: p['chunk'])
    def metadata():
        for part in parts:
            with (Path(part['directory']) / 'records.jsonl').open('rb') as f:
                for line in f:
                    yield orjson.loads(line)
    domains = {d: Counter() for d in DOMAINS}
    for row, ids in zip(metadata(), read_sequences(root / 'tokenizers' / f'{variant}_train.bin'), strict=True):
        encoded = model.encode(ids)
        domains[row['domain']].update(records=1, raw=len(ids), bpe=len(encoded))
    summary = json.loads((root / 'tokenizers' / variant / 'summary.json').read_text())
    total = sum(d['bpe'] for d in domains.values())
    if total != summary['final_tokens']:
        raise ValueError(f'Fit/apply mismatch: {variant}: {total} vs {summary["final_tokens"]}')
    result = dict(variant=variant, domains=domains, reapplied_tokens=total,
                  trained_tokens=summary['final_tokens'], exact_fit_apply_match=True)
    write_json(root / 'diagnostics/e2a_ordering' / f'train_apply_{variant}.json', result)
    print(json.dumps(dict(event='fit_apply_audit', **result)), flush=True)
    return result


def rings(root):
    import numpy as np
    import orjson
    from experiments.diagnostics.native_domain_bpe_probe import read_sequences
    from cli.pretrain.serialize_fddfs_full import load_count_payload, build_guide
    from pipeline.finetune.fddfs_encode import build_graph
    from pipeline.serialization.freq_dfs_core import decode_feature, compact_node_label
    meta = [orjson.loads(x) for x in (root / 'tokenizers/test_records.jsonl').read_bytes().splitlines()]
    native = {r['sample_id']: r for r in (orjson.loads(x) for x in
              (root / 'manifests/tokenizer_test.jsonl').read_bytes().splitlines())}
    vocab = json.loads((root / 'tokenizer_graphs/base_vocab.json').read_text())
    ring_base = vocab['ring_base']
    guide = build_guide(load_count_payload(root / 'tokenizer_graphs/frequency_guide.json'))
    models = {v: BPE(root / 'tokenizers' / v / 'merges.csv', BASE_SIZE) for v in ('frequency', 'fixed')}
    profiles, strata, per_record, token_types = {}, {}, {}, {}
    for variant, model in models.items():
        domains = {d: dict(ring_ids=Counter(), max_ring=Counter()) for d in DOMAINS}
        by_proline = defaultdict(Counter)
        lengths = {}
        with (root / 'tokenizers' / f'{variant}_test_lengths.csv').open() as f:
            for row in csv.DictReader(f):
                lengths[row['sample_id']] = int(row['after_bpe'])
        per_record[variant] = []
        for row, ids in zip(meta, read_sequences(root / 'tokenizers' / f'{variant}_test.bin'), strict=True):
            domain = row['domain']
            ring_ids = ids[ids >= ring_base] - ring_base + 1
            histogram = np.bincount(ring_ids, minlength=1)
            domains[domain]['ring_ids'].update({str(i): int(n) for i, n in enumerate(histogram) if n})
            maximum = int(ring_ids.max()) if len(ring_ids) else 0
            domains[domain]['max_ring'][str(maximum)] += 1
            if domain == 'protein':
                sequence = native[row['sample_id']]['normalized']
                n = sequence.count('P')
                group = '0' if n == 0 else ('1' if n == 1 else ('2-3' if n < 4 else '4+'))
                by_proline[group].update(records=1, residues=len(sequence), prolines=n,
                    raw=len(ids), bpe=lengths[row['sample_id']], sum_max_ring=maximum)
                per_record[variant].append(dict(sample_id=row['sample_id'], residues=len(sequence),
                    prolines=n, max_ring=maximum, bpe=lengths[row['sample_id']]))
        profiles[variant], strata[variant] = domains, dict(by_proline)
        token_types[variant] = {str(threshold): sum(any(x >= ring_base + threshold - 1 for x in expansion)
            for token, expansion in model.expansions.items() if token >= BASE_SIZE) for threshold in (1,2,3,4,8)}
    render = {}
    for symbol, idx in vocab['token_to_id'].items():
        if symbol.startswith('N:'):
            render[idx] = compact_node_label(decode_feature(symbol, 'N'))
        else:
            render[idx] = symbol
    for idx in range(ring_base, BASE_SIZE):
        render[idx] = 'R' + str(idx-ring_base+1)
    examples = []
    for n in (0, 1, 2, 4, 8):
        sequence = 'AF'*4 + 'AP'*n + 'AF'*(12-n)
        graph = build_graph(target_type='protein', sample_id=f'synthetic_p{n}', sequence=sequence)
        for variant, model in models.items():
            result = serialize_ids(cpp_encoder(guide, vocab, fixed=variant == 'fixed'), graph, ring_base)
            ids = list(result['ids'])
            examples.append(dict(sequence=sequence, prolines=n, variant=variant, raw=len(ids),
                bpe=len(model.encode(ids)), stats=result['stats'],
                serialization=' '.join(render[i] for i in ids)))
    result = dict(full_test_ring_profiles=profiles, protein_by_proline_count=strata,
        learned_merges_containing_ring_at_least=token_types, synthetic_examples=examples,
        limitations='Proline strata are descriptive and differ in sequence composition/length; not causal estimates.')
    write_json(root / 'diagnostics/e2a_ordering/rings.json', result)
    write_json(root / 'diagnostics/e2a_ordering/protein_per_record.json', per_record)
    print(json.dumps(dict(event='ring_diagnostics', protein_by_proline_count=strata,
        learned_merges_containing_ring_at_least=token_types,
        synthetic_examples=[{k:v for k,v in r.items() if k not in ('serialization','sequence')} for r in examples])), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    require_compute_host('E2a BPE and ring diagnostics')
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(train_audit, (str(args.run), v)) for v in ('frequency', 'fixed')]
        rings(args.run)
        results = [f.result() for f in futures]
    write_json(args.run / 'diagnostics/e2a_ordering/audit_complete.json', dict(training=results))
    print('E2A_RING_AND_FIT_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
