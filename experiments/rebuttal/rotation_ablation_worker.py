#!/usr/bin/env python3
"""Four-way ordering/rotation comparison using the frozen E2a cohorts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time
from unittest.mock import patch

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import BASE_SIZE, CONTENT_VOCAB, DEFAULT_EDGE, DOMAINS, IntWriter, serialize_ids, write_json
from pipeline.runtime_host import require_compute_host

VARIANTS = ('frequency_rotate', 'fixed_rotate', 'frequency_no_rotate', 'fixed_no_rotate')
NEW_VARIANTS = VARIANTS[2:]


def build_runtime(out):
    import pybind11
    runtime = out / 'runtime'
    runtime.mkdir()
    original = (PIPE / 'pipeline/serialization/_stackk_cpp.cpp').read_text()
    source = original
    replacements = [
        ('void set_frequency_ordering(bool enabled) { frequency_ordering_ = enabled; }',
         'void set_frequency_ordering(bool enabled) { frequency_ordering_ = enabled; }\n'
         '    void set_neighbor_rotation(bool enabled) { rotate_neighbors_ = enabled; }'),
        ('bool frequency_ordering_ = true;', 'bool frequency_ordering_ = true;\n    bool rotate_neighbors_ = true;'),
        ('if (neighbors.size() > 1) {\n                std::rotate',
         'if (rotate_neighbors_ && neighbors.size() > 1) {\n                std::rotate'),
        ('.def("set_frequency_ordering", &FastStackKGuide::set_frequency_ordering)',
         '.def("set_frequency_ordering", &FastStackKGuide::set_frequency_ordering)\n'
         '        .def("set_neighbor_rotation", &FastStackKGuide::set_neighbor_rotation)'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f'Rotation patch location is ambiguous or absent: {old}')
        source = source.replace(old, new)
    (runtime / 'original_stackk_cpp.cpp').write_text(original)
    path = runtime / 'rotation_fddfs.cpp'
    path.write_text(source)
    module = runtime / ('_rebuttal_rotation_cpp' + sysconfig.get_config_var('EXT_SUFFIX'))
    command = ['g++', '-O3', '-shared', '-std=c++17', '-fPIC',
        '-I' + pybind11.get_include(), '-I' + sysconfig.get_path('include'),
        '-DFDDFS_MODULE_NAME=_rebuttal_rotation_cpp', str(path), '-o', str(module)]
    with (runtime / 'build.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    sys.path.insert(0, str(runtime))
    print('ISOLATED_ROTATION_RUNTIME_BUILT', flush=True)


def make_encoder(guide, vocab, variant):
    import _rebuttal_rotation_cpp
    from pipeline.serialization.stackk_fast import _items, _triplet_items, _vocab_feature_items
    mapping = vocab['token_to_id']
    encoder = _rebuttal_rotation_cpp.FastStackKGuide(
        _items(guide.node_counts), _items(guide.edge_counts), _triplet_items(guide.triplet_counts),
        DEFAULT_EDGE, _vocab_feature_items(vocab, 'N'), _vocab_feature_items(vocab, 'E'),
        mapping['('], mapping[')'], mapping['<SEP>'], vocab['ring_base'], 1, True)
    encoder.set_frequency_ordering(variant.startswith('frequency'))
    encoder.set_neighbor_rotation(not variant.endswith('no_rotate'))
    return encoder


def smoke(out):
    sys.path.insert(0, str(PIPE / 'tests'))
    from test_fddfs_cpp_equivalence import synthetic_graphs
    from experiments.rebuttal.common import FixedOrderGuide
    from pipeline.serialization import freq_dfs_core, fddfs
    graphs = list(synthetic_graphs().values())
    guide = fddfs.build_frequency_guide(graphs)
    guide.default_edge_feature = DEFAULT_EDGE
    vocab = fddfs.build_vocab(guide.node_counts, guide.edge_counts, DEFAULT_EDGE, True)
    original_plan = freq_dfs_core.make_dfs_plan
    source = inspect.getsource(original_plan)
    rotation = '        if len(neighbors) > 1:\n            neighbors = neighbors[1:] + neighbors[:1]\n'
    if source.count(rotation) != 1:
        raise ValueError('Python reference rotation location is ambiguous or absent')
    namespace = dict(vars(freq_dfs_core))
    exec(compile(source.replace(rotation, ''), '<no-rotation-reference>', 'exec'), namespace)
    reference_no_rotate = namespace['make_dfs_plan']
    output = {}
    for variant in VARIANTS:
        encoder = make_encoder(guide, vocab, variant)
        reference_guide = FixedOrderGuide() if variant.startswith('fixed') else guide
        planner = reference_no_rotate if variant.endswith('no_rotate') else original_plan
        output[variant] = []
        for graph in graphs:
            with patch.object(fddfs, 'make_dfs_plan', planner):
                expected = fddfs.serialize_fddfs(graph, reference_guide)
            actual = serialize_ids(encoder, graph, vocab['ring_base'])
            expected_ids = [fddfs.encode_fddfs_token(t, vocab) for t in expected.sequence]
            if list(actual['ids']) != expected_ids or not fddfs.verify_roundtrip(graph, expected):
                raise ValueError(f'Synthetic reference/roundtrip failure: {variant}')
            output[variant].append(expected_ids)
    for ordering in ('frequency', 'fixed'):
        if output[ordering + '_rotate'] == output[ordering + '_no_rotate']:
            raise ValueError(f'Rotation switch did not affect any synthetic case: {ordering}')
    write_json(out / 'smoke.json', dict(graphs=len(graphs), variants=len(VARIANTS),
        cpp_python_equal=True, roundtrip=True, rotation_changes_examples=True))
    print('FOUR_VARIANT_REFERENCE_SMOKE_PASSED', flush=True)


def initialize(run, out):
    global ROOT, OUT, VOCAB, ENCODERS, INVERSE
    from cli.pretrain.serialize_fddfs_full import load_count_payload, build_guide
    ROOT, OUT = Path(run), Path(out)
    sys.path.insert(0, str(OUT / 'runtime'))
    VOCAB = json.loads((ROOT / 'tokenizer_graphs/base_vocab.json').read_text())
    guide = build_guide(load_count_payload(ROOT / 'tokenizer_graphs/frequency_guide.json'))
    ENCODERS = {v: make_encoder(guide, VOCAB, v) for v in VARIANTS}
    INVERSE = {v: k for k, v in VOCAB['token_to_id'].items()}
    INVERSE.update({i: 'R' + str(i - VOCAB['ring_base'] + 1) for i in range(VOCAB['ring_base'], BASE_SIZE)})


def serialize_part(part):
    import orjson
    from experiments.diagnostics.native_domain_bpe_probe import read_sequences
    from pipeline.graph.io import iter_graph_records
    from pipeline.serialization.fddfs import FDDFSResult, verify_roundtrip
    role, number = part['role'], part['chunk']
    directory = OUT / 'parts' / f'{role}_{number:05d}'
    directory.mkdir()
    old_directory = ROOT / 'tokenizers/parts' / f'{role}_{number:05d}'
    old = {v: iter(read_sequences(old_directory / f'{v}.bin')) for v in ('frequency', 'fixed')}
    writers = {v: IntWriter(directory / f'{v}.bin', BASE_SIZE-1) for v in VARIANTS}
    fields = ['sample_id', 'domain', 'source_group', 'variant', 'raw', 'max_ring_id',
              'branch_pairs', 'ring_tokens', 'node_tokens', 'supported', 'common_support']
    checked = Counter()
    with (directory / 'records.jsonl').open('wb') as records, (directory / 'stats.csv').open('w') as f, \
            (directory / 'exclusions.jsonl').open('wb') as exclusions:
        csv_out = csv.DictWriter(f, fieldnames=fields)
        csv_out.writeheader()
        for graph, row in iter_graph_records(Path(part['path'])):
            variants = {}
            for variant, encoder in ENCODERS.items():
                result = serialize_ids(encoder, graph, VOCAB['ring_base'])
                ids = list(result['ids'])
                variants[variant] = result
                if variant in NEW_VARIANTS:
                    if role == 'test':
                        inverse = {i: int(j) for i, j in enumerate(result['old_to_new'])}
                        tokens = [INVERSE[i] if i < VOCAB['ring_base'] else
                                  'R' + str(i - VOCAB['ring_base'] + 1) for i in ids]
                        decoded = FDDFSResult(tokens, inverse, result['stats'], DEFAULT_EDGE)
                        if not verify_roundtrip(graph, decoded):
                            raise ValueError(f'Graph roundtrip failed: {variant} {row["sample_id"]}')
                        checked[variant + '_roundtrips'] += 1
                else:
                    previous = next(old[variant.removesuffix('_rotate')])
                    if ids != previous.tolist():
                        raise ValueError(f'Rotation-on control changed: {variant} {row["sample_id"]}')
                    checked[variant + '_baseline_equal'] += 1
            support = {v: max(r['ids']) < BASE_SIZE for v,r in variants.items()}
            common = all(support.values())
            checked['requested_records'] += 1
            checked['common_support_records'] += common
            checked['excluded_ring_support'] += not common
            for variant, result in variants.items():
                ids = list(result['ids'])
                checked[variant + '_supported'] += support[variant]
                csv_out.writerow(dict(sample_id=row['sample_id'], domain=row['domain'],
                    source_group=row['source_group'], variant=variant, raw=len(ids),
                    supported=int(support[variant]), common_support=int(common),
                    **{k: result['stats'][k] for k in ('max_ring_id','branch_pairs','ring_tokens','node_tokens')}))
                if common:
                    writers[variant].write(ids)
            if common:
                records.write(orjson.dumps({k: row[k] for k in ('sample_id','domain','source_group')}) + b'\n')
            else:
                exclusions.write(orjson.dumps(dict(sample_id=row['sample_id'], domain=row['domain'],
                    source_group=row['source_group'], reason='ring_id_outside_frozen_249_capacity',
                    max_ring_ids={v:r['stats']['max_ring_id'] for v,r in variants.items()})) + b'\n')
    for iterator in old.values():
        if next(iterator, None) is not None:
            raise ValueError('Unconsumed baseline records')
    for writer in writers.values():
        writer.close()
    return dict(role=role, chunk=number, directory=str(directory), checks=dict(checked),
        variants={v: dict(records=w.records, tokens=w.tokens, max_id=w.max_id) for v,w in writers.items()})


def materialize(root, out, workers):
    from experiments.rebuttal.serialize_tokenizers_worker import concatenate
    manifest = json.loads((root / 'tokenizer_graphs/manifest.json').read_text())
    tasks = [p for p in manifest['parts'] if p['role'] == 'test' or (p['role'] == 'train' and p['chunk'] < 100)]
    (out / 'parts').mkdir()
    parts, start = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(str(root),str(out))) as pool:
        futures = [pool.submit(serialize_part, part) for part in tasks]
        for future in as_completed(futures):
            parts.append(future.result())
            if len(parts) % 5 == 0 or len(parts) == len(tasks):
                print(json.dumps(dict(event='serialize_progress', completed=len(parts), total=len(tasks),
                    seconds=time.monotonic()-start)), flush=True)
    parts.sort(key=lambda p: (p['role'], p['chunk']))
    corpora = {v: {role: concatenate(parts, out / f'{v}_{role}.bin', v, role)
                   for role in ('train','test')} for v in VARIANTS}
    for role in ('train','test'):
        with (out / f'{role}_records.jsonl').open('wb') as handle:
            for part in parts:
                if part['role'] == role:
                    with (Path(part['directory']) / 'records.jsonl').open('rb') as source:
                        shutil.copyfileobj(source, handle)
    for role, expected in (('train',99575), ('test',19937)):
        if sum(p['checks']['requested_records'] for p in parts if p['role'] == role) != expected:
            raise ValueError('Unexpected change in original E2a cohort sizes')
        if len({corpora[v][role]['records'] for v in VARIANTS}) != 1:
            raise ValueError('Four-way common-support cohort sizes differ')
    checks = Counter()
    for part in parts:
        checks.update(part['checks'])
    write_json(out / 'serialization.json', dict(parts=parts, corpora=corpora, checks=checks))
    coverage = {role: {d: dict(original_records=0, common_support_records=0,
                supported={v:0 for v in VARIANTS}, max_ring_id={v:0 for v in VARIANTS})
                for d in DOMAINS} for role in ('train','test')}
    for part in parts:
        with (Path(part['directory'])/'stats.csv').open() as f:
            for row in csv.DictReader(f):
                item = coverage[part['role']][row['domain']]
                v = row['variant']
                if v == VARIANTS[0]:
                    item['original_records'] += 1
                    item['common_support_records'] += int(row['common_support'])
                item['supported'][v] += int(row['supported'])
                item['max_ring_id'][v] = max(item['max_ring_id'][v], int(row['max_ring_id']))
    write_json(out/'coverage.json', coverage)
    print(json.dumps(dict(event='serialization_complete', checks=checks, corpora=corpora)), flush=True)


def fit_and_measure(root, out, threads):
    import numpy as np
    import orjson
    from experiments.diagnostics.native_domain_bpe_probe import timed_command, read_sequences
    from experiments.rebuttal.tokenizer import BPE
    rows = [orjson.loads(x) for x in (out / 'test_records.jsonl').read_bytes().splitlines()]
    saved_rows = [orjson.loads(x) for x in (root / 'tokenizers/test_records.jsonl').read_bytes().splitlines()]
    selected_ids = {r['sample_id'] for r in rows}
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in saved_rows if r['sample_id'] in selected_ids]:
        raise ValueError('Four-way test record order mismatch')
    results, lengths = {}, {}
    for variant in VARIANTS:
        directory = out / variant
        if directory.exists():
            raise FileExistsError(directory)
        measured = timed_command([root/'runtime/int_bpe_compact', '--mode', 'single',
            '--input', out/f'{variant}_train.bin', '--out-dir', directory,
            '--num-merges', CONTENT_VOCAB-BASE_SIZE, '--min-frequency', 1,
            '--threads', threads, '--shards', 48, '--no-rle', '--sort-positions',
            '--build-mode', 'onepass', '--log-every', 1000], directory)
        trained = json.loads((directory/'summary.json').read_text())
        if trained['num_merges_performed'] != CONTENT_VOCAB-BASE_SIZE:
            raise ValueError('Unequal BPE merge budgets')
        model = BPE(directory/'merges.csv', BASE_SIZE)
        values = []
        with (out/f'{variant}_test_lengths.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id','domain','before_bpe','after_bpe'])
            for row, ids in zip(rows, read_sequences(out/f'{variant}_test.bin'), strict=True):
                encoded = model.encode(ids)
                if model.decode(encoded) != ids.tolist():
                    raise ValueError(f'BPE roundtrip failure: {variant} {row["sample_id"]}')
                values.append(len(encoded))
                writer.writerow([row['sample_id'],row['domain'],len(ids),len(encoded)])
        lengths[variant] = np.asarray(values)
        reapplied = sum(len(model.encode(ids)) for ids in read_sequences(out/f'{variant}_train.bin'))
        if reapplied != trained['final_tokens']:
            raise ValueError(f'BPE fit/apply mismatch: {variant}: {reapplied} vs {trained["final_tokens"]}')
        results[variant] = dict(training=trained, resources=measured, fit_apply_exact=True,
            bpe_roundtrips=len(values), merges_containing_ring_at_least_4=sum(
                any(x >= 337 for x in expansion) for token,expansion in model.expansions.items() if token >= BASE_SIZE))
        write_json(out/f'{variant}_measured.json', results[variant])
        print(json.dumps(dict(event='tokenizer_complete', variant=variant,
            test_total=int(lengths[variant].sum()), train_total=reapplied)), flush=True)
    return rows, lengths, results


def report(root, out, rows, lengths, results):
    import numpy as np
    manifest = json.loads((out/'serialization.json').read_text())
    ring_stats = {v: {d: Counter() for d in DOMAINS} for v in VARIANTS}
    ring_max = {v: {d: 0 for d in DOMAINS} for v in VARIANTS}
    for part in manifest['parts']:
        if part['role'] != 'test':
            continue
        with (Path(part['directory'])/'stats.csv').open() as f:
            for row in csv.DictReader(f):
                if not int(row['common_support']):
                    continue
                v,d = row['variant'],row['domain']
                ring_stats[v][d].update(records=1, raw=int(row['raw']),
                    max_ring_sum=int(row['max_ring_id']), branch_pairs=int(row['branch_pairs']))
                ring_max[v][d] = max(ring_max[v][d], int(row['max_ring_id']))
    domains = {}
    comparisons = [('frequency_no_rotate','frequency_rotate'), ('fixed_no_rotate','fixed_rotate'),
                   ('frequency_rotate','fixed_rotate'), ('frequency_no_rotate','fixed_no_rotate')]
    random = np.random.default_rng(20261014)
    for domain in DOMAINS:
        mask = np.asarray([r['domain'] == domain for r in rows])
        ids = np.flatnonzero(mask)
        if not len(ids):
            domains[domain] = dict(records=0, variants=None, comparisons=None,
                reason='No records within the common frozen ring support')
            continue
        groups = sorted({rows[i]['source_group'] for i in ids})
        mapping = {g:i for i,g in enumerate(groups)}
        cluster = np.asarray([mapping[rows[i]['source_group']] for i in ids])
        summary, totals = {}, {}
        for variant in VARIANTS:
            values = lengths[variant][mask]
            summary[variant] = dict(records=len(values), total=int(values.sum()), mean=float(values.mean()),
                median=float(np.median(values)), p90=float(np.quantile(values,.9)), p99=float(np.quantile(values,.99)),
                max=int(values.max()), raw_total=ring_stats[variant][domain]['raw'],
                max_ring_id=ring_max[variant][domain],
                mean_max_ring_id=ring_stats[variant][domain]['max_ring_sum']/len(values))
            totals[variant] = np.bincount(cluster, weights=values, minlength=len(groups))
        contrasts = {}
        for a,b in comparisons:
            samples = []
            for _ in range(2000):
                selected = random.integers(len(groups), size=len(groups))
                samples.append(totals[a][selected].sum()/totals[b][selected].sum()-1)
            contrasts[a+'_vs_'+b] = dict(relative_length_change=float(lengths[a][mask].sum()/lengths[b][mask].sum()-1),
                source_cluster_bootstrap_95ci=np.quantile(samples,[.025,.975]).tolist())
        domains[domain] = dict(variants=summary, comparisons=contrasts, source_groups=len(groups))
    payload = dict(domains=domains, fitting=results, validation=manifest['checks'],
        coverage=json.loads((out/'coverage.json').read_text()),
        interpretation='Positive relative_length_change means the numerator variant is longer. '
            'CIs are source-cluster paired bootstrap, conditional on these four fitted tokenizers. '
            'Only the neighbor-list rotation is toggled; root/ring keys, grammar and guide counts stay fixed. '
            'All four BPEs are freshly trained on the same common-support training subset, with 16621 merges each. '
            'Full original-cohort coverage is reported separately; original full-cohort BPE fits are not reused.')
    write_json(out/'results.json', payload)
    print(json.dumps(dict(event='FOUR_WAY_RESULTS', domains=domains)), flush=True)
    print('ROTATION_ABLATION_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    require_compute_host('E2a rotation ablation')
    args.output.mkdir(exist_ok=False)
    write_json(args.output/'protocol.json', dict(source_run=str(args.run), variants=VARIANTS,
        original_train_records=99575, original_test_records=19937, base_vocab=BASE_SIZE,
        merges=CONTENT_VOCAB-BASE_SIZE, workers=args.workers, bpe_threads=args.threads,
        bpe_shards=48, common_support_policy='Report full-cohort ring-support coverage; '
        'fit all four BPEs afresh on the identical intersection supporting the frozen 249 ring IDs. '
        'Frequency guide remains the original frozen E2a training guide, shared by all variants.',
        no_rotation_definition='Use descending sorted neighbor list directly; '
        'remove only the left rotation, keep parent/visited filtering and last-child continuation unchanged.'))
    shutil.copy2(Path(__file__), args.output/'runner_snapshot.py')
    build_runtime(args.output)
    smoke(args.output)
    materialize(args.run,args.output,args.workers)
    rows, lengths, results = fit_and_measure(args.run,args.output,args.threads)
    report(args.run,args.output,rows,lengths,results)


if __name__ == '__main__':
    main()
