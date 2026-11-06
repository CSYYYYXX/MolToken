#!/usr/bin/env python3
"""Audit frozen artifacts and publish descriptive E2 results on a worker."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

PIPE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PIPE))
from experiments.rebuttal.common import DOMAINS,write_json
from pipeline.runtime_host import require_compute_host


def main():
    import numpy as np
    import orjson
    from experiments.rebuttal.prepare_training_worker import init_encode,load_raw
    from experiments.rebuttal.tokenizer import PairedTokenizer
    from experiments.diagnostics.native_domain_bpe_probe import read_sequences
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    require_compute_host('E2/E3 artifact audit')
    root=args.run
    fit=json.loads((root/'tokenizers/fit_complete.json').read_text())
    e2a=json.loads((root/'tokenizers/e2a_results.json').read_text())
    raw=json.loads((root/'raw_catalog/manifest.json').read_text())
    counts=Counter()
    for part in raw['parts']:
        counts.update(part['exclusions'])
    moses=json.loads((root/'moses/manifest.json').read_text())
    moses_counts=Counter()
    for part in moses['parts']:
        moses_counts[part['role']]+=part['records']
    init_encode(root)
    tokenizer=PairedTokenizer(root)
    checked=0
    for chunk in range(3):
        directory=root/'streams/joint_42'/f'{chunk:06d}'
        gids=np.load(directory/'ids.npy')
        domains=np.load(directory/'domains.npy')
        a=list(read_sequences(directory/'graph.bin'))
        c=list(read_sequences(directory/'native.bin'))
        assert len(set(map(int,gids)))==len(gids)==len(a)==len(c)==len(domains)
        for i in range(min(128,len(gids))):
            row=load_raw(int(gids[i]))
            expected,reason=tokenizer.encode(row)
            assert reason is None and expected[0]==a[i].tolist() and expected[1]==c[i].tolist()
            assert DOMAINS[int(domains[i])]==row['domain']
            checked+=1
    records=[orjson.loads(x) for x in (root/'tokenizers/test_records.jsonl').read_bytes().splitlines()]
    values={}
    for variant in ('frequency','fixed'):
        with (root/f'tokenizers/{variant}_test_lengths.csv').open() as handle:
            values[variant]=list(csv.DictReader(handle))
    clustered={}
    rng=np.random.default_rng(20261014)
    for domain in DOMAINS:
        groups=defaultdict(lambda:np.zeros(2,dtype=np.int64))
        for i,row in enumerate(records):
            assert row['sample_id']==values['frequency'][i]['sample_id']==values['fixed'][i]['sample_id']
            if row['domain']==domain:
                groups[row['source_group']]+=np.asarray([int(values[k][i]['after_bpe']) for k in ('frequency','fixed')])
        array=np.stack(list(groups.values()))
        estimates=[]
        for _ in range(2000):
            draw=array[rng.integers(len(array),size=len(array))].sum(axis=0)
            estimates.append(1-draw[0]/draw[1])
        clustered[domain]=dict(source_groups=len(array),paired_source_bootstrap_95ci=np.quantile(estimates,[.025,.975]).tolist())
    report=dict(e2a=e2a,source_cluster_bootstrap=clustered,
        tokenizer_fit_records=fit['graph']['training']['input_sequences'],
        raw_pool_records=sum(p['records'] for p in raw['parts']),raw_exclusions=dict(counts),
        moses_records=dict(moses_counts),reencoded_paired_records_checked=checked,
        representation_lengths={d:{v:fit[v]['by_domain'][d] for v in ('graph','native')} for d in DOMAINS},
        native_vocabulary_by_namespace=fit['native']['vocabulary_by_namespace'])
    write_json(root/'initial_results.json',report)
    print(json.dumps(dict(event='ARTIFACT_AUDIT_PASS',checked=checked,raw_pool=report['raw_pool_records'],
        moses_records=dict(moses_counts),source_clusters=clustered,
        native_vocabulary_by_namespace=report['native_vocabulary_by_namespace'],
        mean_lengths={d:{v:fit[v]['by_domain'][d]['mean'] for v in ('graph','native')} for d in DOMAINS})),flush=True)


if __name__=='__main__':
    main()
