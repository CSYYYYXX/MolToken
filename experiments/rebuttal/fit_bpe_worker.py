#!/usr/bin/env python3
"""Fit matched vocabularies, report paired held-out E2a and A/C lengths."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import BASE_SIZE, CONTENT_VOCAB, DOMAINS, write_json
from experiments.diagnostics.native_domain_bpe_probe import timed_command, read_sequences, check_namespaces
from pipeline.runtime_host import require_compute_host


def main():
    import numpy as np
    import orjson
    from experiments.rebuttal.tokenizer import BPE
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--threads', type=int, default=48)
    args = p.parse_args()
    require_compute_host('rebuttal BPE fitting')
    root = args.run / 'tokenizers'
    manifest = json.loads((root / 'serialized.json').read_text())
    binary = args.run / 'runtime/int_bpe_compact'
    rows = [orjson.loads(x) for x in (root / 'test_records.jsonl').read_bytes().splitlines()]
    results, lengths = {}, {}
    for variant in ('frequency', 'fixed', 'graph', 'native'):
        base = manifest['native_base_size'] if variant == 'native' else BASE_SIZE
        count = CONTENT_VOCAB - base
        out = root / variant
        # Every fit is fresh; a previous result is never silently substituted.
        if out.exists():
            raise FileExistsError(out)
        measured = timed_command([binary, '--mode', 'single', '--input', root / f'{variant}_train.bin',
            '--out-dir', out, '--num-merges', count, '--min-frequency', 1, '--threads', args.threads,
            '--shards', args.threads, '--no-rle', '--sort-positions', '--build-mode', 'onepass', '--log-every', 1000], out)
        trained = json.loads((out / 'summary.json').read_text())
        if trained['num_merges_performed'] != count:
            raise ValueError(f'Insufficient merges; do not compare unequal vocabularies: {variant}')
        model = BPE(out / 'merges.csv', base)
        values, raw_values = [], []
        for ids in read_sequences(root / f'{variant}_test.bin'):
            encoded = model.encode(ids)
            if model.decode(encoded) != ids.tolist():
                raise ValueError(f'Lossy BPE: {variant}, record {len(values)}')
            values.append(len(encoded))
            raw_values.append(len(ids))
        if len(values) != len(rows):
            raise ValueError('Paired record count mismatch')
        lengths[variant] = np.asarray(values)
        by_domain = {}
        for domain in DOMAINS:
            mask = np.asarray([r['domain'] == domain for r in rows])
            v = lengths[variant][mask]
            by_domain[domain] = dict(records=len(v), total=int(v.sum()), mean=float(v.mean()),
                median=float(np.median(v)), p90=float(np.quantile(v,.9)), p99=float(np.quantile(v,.99)),
                max=int(v.max()), complete_context=int((v <= 509).sum()), bpe_roundtrip_pass=len(v))
        results[variant] = dict(training=trained, resources=measured, by_domain=by_domain)
        if variant == 'native':
            results[variant]['vocabulary_by_namespace'] = check_namespaces(out / 'merges.csv', json.loads((root / 'alphabet.json').read_text()))
        write_json(root / 'fit_progress.json', results)
        with (root / f'{variant}_test_lengths.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'domain', 'before_bpe', 'after_bpe'])
            writer.writerows((r['sample_id'], r['domain'], raw, length) for r, raw, length in zip(rows, raw_values, values))
        print(json.dumps(dict(event='tokenizer_complete', variant=variant, results=results[variant])), flush=True)
        if variant == 'fixed':
            report = {}
            random = np.random.default_rng(20261014)
            for domain in DOMAINS:
                mask = np.asarray([r['domain'] == domain for r in rows])
                a, b = lengths['frequency'][mask], lengths['fixed'][mask]
                boot = []
                for _ in range(2000):
                    sample = random.integers(len(a), size=len(a))
                    boot.append(1 - a[sample].sum()/b[sample].sum())
                report[domain] = dict(frequency=results['frequency']['by_domain'][domain], fixed=results['fixed']['by_domain'][domain],
                    reduction=1-float(a.sum()/b.sum()), paired_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
            write_json(root / 'e2a_results.json', dict(domains=report,
                graph_roundtrips=manifest['graph_roundtrips'], merges_per_variant=count,
                training_records=manifest['corpora']['frequency']['train']['records'],
                interpretation='Conditional on supported graph alphabet; paired same records; positive reduction means frequency is shorter.'))
            print('E2A_RESULTS_READY', flush=True)
    write_json(root / 'fit_complete.json', results)
    print('ALL_TOKENIZERS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
