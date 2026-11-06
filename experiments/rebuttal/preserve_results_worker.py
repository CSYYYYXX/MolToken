#!/usr/bin/env python3
"""Snapshot completed rebuttal evidence and inventory retained shared-storage data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host

VARIANTS = ('frequency_rotate', 'fixed_rotate', 'frequency_no_rotate', 'fixed_no_rotate')
ROTATIONS = ('rotation_ablation_20261014_v2', 'rotation_full_cohort_512_20261014_v1')
COPY_LIMIT = 32 * 1024 * 1024
BULK_SUFFIXES = {'.bin', '.zst', '.gz', '.pt', '.pth', '.npy', '.npz', '.safetensors'}


def save_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--docs', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    host = require_compute_host('preserve rebuttal experiment evidence')
    started = datetime.now(timezone.utc).isoformat()
    run, output = args.run.resolve(), args.output.resolve()
    required = [
        'initial_results.json', 'tokenizers/e2a_results.json', 'tokenizers/fit_complete.json',
        'tokenizers/serialized.json', 'tokenizers/test_records.jsonl',
        'tokenizer_graphs/base_vocab.json', 'tokenizer_graphs/frequency_guide.json',
        'tokenizer_graphs/graph_guide.json', 'tokenizer_graphs/manifest.json',
        'provenance/source_snapshot.json', 'diagnostics/e2a_ordering/complete.json',
        'diagnostics/e2a_ordering/audit_complete.json', 'diagnostics/e2a_ordering/rings.json',
        'logs/rotation_ablation.log', 'logs/rotation_ablation_v2.log',
        'logs/rotation_full_cohort_512.log', 'logs/diagnose_e2a.log', 'logs/diagnose_e2a_rings.log',
        'rotation_ablation_20261014_v1/protocol.json',
        'rotation_full_cohort_512_20261014_v1/full_cohort_complete.json',
    ]
    for variant in ('frequency', 'fixed', 'graph', 'native'):
        required += [f'tokenizers/{variant}/{name}' for name in ('merges.csv', 'summary.json')]
        required.append(f'tokenizers/{variant}_test_lengths.csv')
    for experiment in ROTATIONS:
        required += [f'{experiment}/{name}' for name in (
            'protocol.json', 'results.json', 'coverage.json', 'serialization.json',
            'smoke.json', 'runner_snapshot.py', 'train_records.jsonl', 'test_records.jsonl',
            'runtime/original_stackk_cpp.cpp', 'runtime/rotation_fddfs.cpp', 'runtime/build.log')]
        for variant in VARIANTS:
            required += [f'{experiment}/{variant}/{name}' for name in ('merges.csv', 'summary.json')]
            required += [f'{experiment}/{variant}_{name}' for name in ('test_lengths.csv', 'measured.json')]
    retained_required = []
    suite_complete = (run / 'gpu_suite_complete.json').is_file() and (run / 'final_results.json').is_file()
    if args.require_complete:
        if not suite_complete:
            raise ValueError('A final archive requires both suite completion and final results')
        required += ['gpu_suite_complete.json', 'FINAL_RESULTS.md', 'final_results.json']
        final = json.loads((run / 'final_results.json').read_text())
        for method in ('graph_random', 'graph_small', 'graph_joint', 'native_joint'):
            rows = final['groups'][method]['seeds']
            if sorted(row['seed'] for row in rows) != [42, 43, 44]:
                raise ValueError(f'Incomplete final seed coverage: {method}')
            for seed in (42, 43, 44):
                name = f'{method}_sft_{seed}'
                required += [f'models/{name}/{item}' for item in ('complete.json', 'protocol.json')]
                required += [f'evaluation/{name}/{item}' for item in
                             ('metrics.json', 'generation_complete.json', 'raw_attempts.jsonl', 'decoded.jsonl')]
                retained_required.append(f'models/{name}/best.pt')
                if method != 'graph_random':
                    pretrain = f'{method}_pretrain_{seed}'
                    required += [f'models/{pretrain}/{item}' for item in ('complete.json', 'protocol.json')]
                    retained_required.append(f'models/{pretrain}/final.pt')
    for relative in retained_required:
        path = run / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty retained checkpoint: {path}')
    for relative in required:
        path = run / relative
        # Successful compiler commands may legitimately produce an empty log.
        if not path.is_file() or (path.suffix != '.log' and path.stat().st_size == 0):
            raise ValueError(f'Missing or empty required evidence: {path}')
        if path.suffix == '.json':
            json.loads(path.read_text())
    output.mkdir(exist_ok=False)

    inventory = []
    def preserve(source, relative, state):
        before = source.stat()
        copied = before.st_size <= COPY_LIMIT and source.suffix not in BULK_SUFFIXES
        entry = dict(source=str(source), relative_path=str(relative), bytes=before.st_size,
                     mtime_ns=before.st_mtime_ns, state=state,
                     storage='independent_copy' if copied else 'retained_in_place')
        if copied:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            after = source.stat()
            if state != 'live_snapshot' and (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Completed evidence changed while being preserved: {source}')
            if state != 'live_snapshot' and destination.stat().st_size != before.st_size:
                raise RuntimeError(f'Incomplete evidence copy: {source}')
            entry['snapshot_bytes'] = destination.stat().st_size
        inventory.append(entry)

    roots = ['tokenizers', 'tokenizer_graphs', 'manifests', 'diagnostics', 'provenance',
             'preflight', 'runtime', 'rotation_ablation_20261014_v1', *ROTATIONS]
    for name in roots:
        source_root = run / name
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        state = 'failed_capacity_diagnostic' if name == 'rotation_ablation_20261014_v1' else 'completed_evidence'
        for source in sorted(source_root.rglob('*')):
            if source.is_file():
                preserve(source, Path('artifacts') / source.relative_to(run), state)
        print(f'Preserved inventory: {name}', flush=True)

    completed_models, completed_evaluations = [], []
    for family, marker, completed in (
        ('models', 'complete.json', completed_models),
        ('evaluation', 'metrics.json', completed_evaluations),
    ):
        for directory in sorted((run / family).iterdir()):
            if not directory.is_dir():
                continue
            finished = (directory / marker).is_file()
            if finished:
                completed.append(directory.name)
            for source in sorted(directory.rglob('*')):
                if source.is_file():
                    preserve(source, Path('artifacts') / source.relative_to(run),
                             'completed_evidence' if finished else 'live_snapshot')
    for source in sorted((run / 'logs').glob('*')):
        if source.is_file():
            preserve(source, Path('artifacts/logs') / source.name, 'live_snapshot')
    for source in sorted(run.iterdir()):
        if source.is_file() and source.suffix in ('.json', '.md'):
            preserve(source, Path('artifacts') / source.name, 'live_snapshot')
    for source in sorted(Path(__file__).parent.iterdir()):
        if source.is_file() and source.suffix in ('.py', '.sh'):
            preserve(source, Path('source/experiments/rebuttal') / source.name, 'source_at_preservation')
    for name in ('MolToken_E2_E3_run_20261014.md', 'MolToken_rebuttal_experiment_plan_brief_20261014.md',
                 'MolToken_rebuttal_experiment_design_v1.md'):
        preserve(args.docs / name, Path('documents') / name, 'document_at_preservation')
    for relative in required:
        if not (output / 'artifacts' / relative).is_file():
            raise RuntimeError(f'Required evidence missing from snapshot: {relative}')

    save_json(output / 'inventory.json', inventory)
    status_counts = dict(Counter(item['storage'] for item in inventory))
    summary = dict(
        created_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(), host=host,
        source_run=str(run), output=str(output), status='complete',
        required_evidence_files=len(required), inventory_files=len(inventory), storage_counts=status_counts,
        required_retained_checkpoints=len(retained_required), final_suite_archive=args.require_complete,
        independent_copy_bytes=sum(item.get('snapshot_bytes', 0) for item in inventory),
        retained_in_place_bytes=sum(item['bytes'] for item in inventory if item['storage'] == 'retained_in_place'),
        completed_models_at_snapshot=completed_models, completed_evaluations_at_snapshot=completed_evaluations,
        retention='User requested all experimental results be retained. Preserve this snapshot and the source run; '
                  'large raw datasets and model checkpoints remain at the source paths in inventory.json. '
                  'Do not delete or overwrite them without explicit user instruction.',
        copy_policy=dict(max_file_bytes=COPY_LIMIT, bulk_suffixes=sorted(BULK_SUFFIXES)),
        experiments=[
            dict(path='tokenizers/e2a_results.json', status='completed', train=99575, test=19937),
            dict(path='diagnostics/e2a_ordering', status='completed', scope='BPE curves and graph/ring diagnostics'),
            dict(path=ROTATIONS[0], status='completed', train=95983, test=17977, ring_capacity=249),
            dict(path=ROTATIONS[1], status='completed', train=99575, test=19937, ring_capacity=512),
            dict(path='rotation_ablation_20261014_v1', status='failed_capacity_diagnostic',
                 reason='Frequency ordering without rotation requires R257 for DNA, exceeding frozen ring support; no accepted BPE comparison.'),
            dict(path='models', status='completed_suite' if suite_complete else 'ongoing_suite',
                 scope='All four methods and seeds 42/43/44 completed.' if suite_complete else
                 'Only completed-model evidence is final; live snapshots are timestamped observations.'),
        ],
        completeness_scope='Inventory covers tokenizer results/raw corpora, graph caches, cohorts, diagnostics, '
                           'rotation experiments, provenance, runtime, preflight, models, evaluations and logs. '
                           'Live streams, raw_catalog, moses data and tmp remain in the source run without a per-file snapshot.',
    )
    save_json(output / 'manifest.json', summary)
    save_json(output / 'COMPLETE.json', dict(completed_utc=summary['completed_utc'], inventory_files=len(inventory),
                                          required_evidence_files=len(required), status='complete'))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('RESULT_PRESERVATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
