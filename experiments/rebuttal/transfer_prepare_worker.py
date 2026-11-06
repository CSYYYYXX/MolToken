#!/usr/bin/env python3
"""Materialize three published prediction splits with the frozen E2/E3 pair."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys
import time
import zipfile

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import write_json
from pipeline.runtime_host import require_compute_host

TASKS = {
    'gb1': dict(domain='protein', problem='regression', metric='spearman',
        source='https://github.com/J-SNACKKB/FLIP/tree/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1',
        split='low_vs_high: published train/validation/test; full construct sequence'),
    'promoter': dict(domain='dna', problem='classification', metric='accuracy',
        source='https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_nontata_promoters/tree/e0003669df0a180c8570ebe091ed91f20fa080ec',
        split='published train/test; stratified sequence-group validation from training, seed 20261017'),
    'mrl': dict(domain='rna', problem='regression', metric='pearson_squared',
        source='https://github.com/pjsample/human_5utr_modeling/blob/d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe/modeling/training_MRL_CNN.ipynb',
        split='egfp_unmod_1: top 280000 by total_reads; first 20000 test; remaining 260000 training pool; 10% group validation'),
}


def raw_rows(task, root):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    s = root / 'sources'
    if task == 'gb1':
        with zipfile.ZipFile(s / 'gb1_splits.zip') as z:
            df = pd.read_csv(z.open('splits/low_vs_high.csv'))
        rows = [dict(id=f'gb1:{i}', sequence=r.sequence, target=float(r.target),
                     split='test' if r.set == 'test' else ('validation' if r.validation is True or r.validation == True else 'train'))
                for i, r in enumerate(df.itertuples(index=False))]
    elif task == 'promoter':
        rows = []
        for split in ('train', 'test'):
            df = pd.read_parquet(s / f'dna_{split}.parquet')
            rows.extend(dict(id=f'promoter:{split}:{i}', sequence=r.seq, target=float(r.label), split=split)
                        for i, r in enumerate(df.itertuples(index=False)))
    elif task == 'mrl':
        df = pd.read_csv(s / 'GSM3130435_egfp_unmod_1.csv.gz')
        if not {'utr', 'rl', 'total_reads'} <= set(df.columns):
            raise ValueError(f'Unexpected MRL columns: {list(df.columns)}')
        # Match the author's default pandas sort; retain source row indices.
        df = df.sort_values('total_reads', ascending=False, kind='quicksort').iloc[:280000]
        if len(df) != 280000:
            raise ValueError(f'MRL requires the complete 280000-record ranked cohort, got {len(df)}')
        rows = [dict(id=f'mrl:{i}', sequence=str(r.utr)[:50].upper().replace('T', 'U'),
                     target=float(r.rl), total_reads=float(r.total_reads),
                     split='test' if rank < 20000 else 'train')
                for rank, (i, r) in enumerate(df.iterrows())]
        if any(len(r['sequence']) != 50 for r in rows):
            raise ValueError('MRL author input is exactly the first 50 nucleotides')
    else:
        raise ValueError(task)
    if not all(np.isfinite(r['target']) for r in rows):
        raise ValueError(f'{task}: nonfinite targets')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source record IDs')

    def group(row):
        seq = row['sequence']
        if task == 'promoter':
            rc = seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
            return min(seq, rc)
        return seq

    original = dict(Counter(r['split'] for r in rows))
    test_groups = {group(r) for r in rows if r['split'] == 'test'}
    # Use input identity only, never test targets, to prevent duplicate leakage.
    removed = [r for r in rows if r['split'] != 'test' and group(r) in test_groups]
    rows = [r for r in rows if r['split'] == 'test' or group(r) not in test_groups]
    if task != 'gb1':
        groups = {}
        for r in rows:
            if r['split'] == 'train':
                groups.setdefault(group(r), []).append(r)
        keys = sorted(groups)
        labels = None
        if task == 'promoter':
            if any(len({r['target'] for r in members}) != 1 for members in groups.values()):
                raise ValueError('Conflicting promoter labels in one sequence/RC group')
            labels = [groups[k][0]['target'] for k in keys]
        _, valid = train_test_split(keys, test_size=.1, random_state=20261017, stratify=labels)
        valid = set(valid)
        for r in rows:
            if r['split'] == 'train' and group(r) in valid:
                r['split'] = 'validation'
    sets = {s: {group(r) for r in rows if r['split'] == s} for s in ('train', 'validation', 'test')}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if sets[a] & sets[b]:
            raise ValueError(f'{task}: input-group overlap {a}/{b}')
    return rows, dict(original_counts=original, prepared_counts=dict(Counter(r['split'] for r in rows)),
        removed_train_validation_test_input_overlap=len(removed), removed_ids=[r['id'] for r in removed],
        grouping='exact sequence and reverse complement' if task == 'promoter' else 'exact sequence',
        split_input_group_overlap=0, pretraining_overlap='not audited; checkpoint pretraining excluded MOSES tests, not these new benchmarks')


def init_encoder(old_run):
    global TOKENIZER
    from experiments.rebuttal.tokenizer import PairedTokenizer
    TOKENIZER = PairedTokenizer(old_run)


def encode_part(job):
    import numpy as np
    task, index, rows, output = job
    directory = Path(output)
    graph, native, go, no, indices, reasons = [], [], [0], [0], [], []
    for i, row in rows:
        pair, reason = TOKENIZER.encode(dict(domain=TASKS[task]['domain'], normalized=row['sequence'], sample_id=row['id']))
        if reason:
            reasons.append(dict(id=row['id'], split=row['split'], reason=reason))
            continue
        a, c = pair
        # Exact native round trip checks the namespace and sequence conversion.
        from experiments.rebuttal.common import NAMESPACE
        expected = [TOKENIZER.alphabet[NAMESPACE[TASKS[task]['domain']]][v] for v in row['sequence']]
        if TOKENIZER.native_bpe.decode(c) != expected:
            raise ValueError(f'Native BPE round trip failed: {row["id"]}')
        graph.extend(a); native.extend(c); go.append(len(graph)); no.append(len(native)); indices.append(i)
    path = directory / f'{index:05d}.npz'
    np.savez(path, graph=np.asarray(graph, dtype=np.uint16), native=np.asarray(native, dtype=np.uint16),
             graph_offsets=np.asarray(go, dtype=np.int64), native_offsets=np.asarray(no, dtype=np.int64),
             indices=np.asarray(indices, dtype=np.int64))
    write_json(directory / f'{index:05d}.json', dict(exclusions=reasons, records=len(indices)))
    return dict(part=index, records=len(indices), excluded=len(reasons))


def main():
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--old-run', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--workers', type=int, default=24)
    args = p.parse_args()
    require_compute_host('transfer benchmark preparation')
    out = args.run / 'data' / args.task
    out.mkdir(exist_ok=False)
    (out / 'parts').mkdir()
    rows, audit = raw_rows(args.task, args.run)
    with gzip.open(out / 'source_rows.jsonl.gz', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    write_json(out / 'split_audit.json', audit)
    jobs = [(args.task, i // 1024, list(enumerate(rows[i:i+1024], start=i)), str(out / 'parts'))
            for i in range(0, len(rows), 1024)]
    start = time.monotonic()
    done = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_encoder, initargs=(str(args.old_run),)) as pool:
        for f in as_completed([pool.submit(encode_part, j) for j in jobs]):
            done.append(f.result())
            print(json.dumps(dict(event='encoded', task=args.task, parts=len(done), total=len(jobs), seconds=time.monotonic()-start)), flush=True)
    indices, exclusions, lengths = [], [], {}
    for i in range(len(jobs)):
        part = np.load(out / 'parts' / f'{i:05d}.npz')
        indices.extend(part['indices'].tolist())
        exclusions.extend(json.loads((out / 'parts' / f'{i:05d}.json').read_text())['exclusions'])
    for rep in ('graph', 'native'):
        sizes = []
        offsets = [0]
        for i in range(len(jobs)):
            with np.load(out / 'parts' / f'{i:05d}.npz') as d:
                sizes.append(len(d[rep]))
                offsets.extend((d[f'{rep}_offsets'][1:] + offsets[-1]).tolist())
        target = np.lib.format.open_memmap(out / f'{rep}.npy', mode='w+', dtype=np.uint16, shape=(sum(sizes),))
        pos = 0
        for i, n in enumerate(sizes):
            with np.load(out / 'parts' / f'{i:05d}.npz') as d:
                target[pos:pos+n] = d[rep]
            pos += n
        target.flush()
        np.save(out / f'{rep}_offsets.npy', np.asarray(offsets, dtype=np.int64))
        lens = np.diff(offsets) + 3
        lengths[rep] = dict(min=int(lens.min()), max=int(lens.max()), mean=float(lens.mean()), input_tokens=int(lens.sum()))
    accepted = [rows[i] for i in indices]
    np.save(out / 'targets.npy', np.asarray([r['target'] for r in accepted], dtype=np.float32))
    split_names = ['train', 'validation', 'test']
    np.save(out / 'splits.npy', np.asarray([split_names.index(r['split']) for r in accepted], dtype=np.uint8))
    (out / 'ids.json').write_text(json.dumps([r['id'] for r in accepted]) + '\n')
    write_json(out / 'exclusions.json', exclusions)
    counts = dict(Counter(r['split'] for r in accepted))
    if any(counts.get(s, 0) == 0 for s in split_names):
        raise ValueError(f'Empty accepted split: {counts}')
    report = dict(task=args.task, **TASKS[args.task], counts=counts, requested=audit['prepared_counts'],
        excluded=len(exclusions), exclusions_by_reason=dict(Counter(r['reason'] for r in exclusions)),
        lengths=lengths, tokenizer_run=str(args.old_run), seconds=time.monotonic()-start,
        context=512, runtime_truncation=False, windowing=False, common_support=True, audit=audit,
        rna_input_conversion='author first 50nt, T to U' if args.task == 'mrl' else None)
    write_json(out / 'complete.json', report)
    print('TRANSFER_DATA_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
