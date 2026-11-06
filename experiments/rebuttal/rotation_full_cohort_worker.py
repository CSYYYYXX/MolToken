#!/usr/bin/env python3
"""Full-cohort rotation diagnostic with 512 ring symbols shared by all arms."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal import rotation_ablation_worker as experiment
from experiments.rebuttal.common import write_json
from pipeline.runtime_host import require_compute_host


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    require_compute_host('full-cohort rotation diagnostic')
    # These globals belong only to this process. Existing experiment runtimes
    # and all source/tokenizer files remain separate.
    experiment.BASE_SIZE = 334 + 512
    experiment.CONTENT_VOCAB = experiment.BASE_SIZE + 16621
    args.output.mkdir(exist_ok=False)
    protocol = dict(source_run=str(args.run), variants=experiment.VARIANTS,
        train_records=99575, test_records=19937, ring_capacity=512,
        base_vocab=experiment.BASE_SIZE, merges=16621,
        content_vocab=experiment.CONTENT_VOCAB, total_vocab_with_three_specials=experiment.CONTENT_VOCAB+3,
        workers=args.workers, threads=args.threads, bpe_shards=48,
        scope='Exploratory full-cohort compression comparison, not a replacement for the frozen model tokenizer.',
        change='All four arms share 512 ring symbols and 16621 freshly fitted merges; '
            'only frequency ordering and neighbor-list left rotation differ. '
            'Root/ring keys, default bond, graph grammar, graph objects, input numbering and guide are retained.')
    write_json(args.output/'protocol.json', protocol)
    shutil.copy2(Path(__file__), args.output/'runner_snapshot.py')
    shutil.copy2(Path(experiment.__file__), args.output/'shared_runner_snapshot.py')
    experiment.build_runtime(args.output)
    experiment.smoke(args.output)
    experiment.materialize(args.run, args.output, args.workers)
    serialization = json.loads((args.output/'serialization.json').read_text())
    for variant in experiment.VARIANTS:
        for role, expected in (('train',99575),('test',19937)):
            if serialization['corpora'][variant][role]['records'] != expected:
                raise ValueError('512 ring symbols do not support the whole frozen cohort; full-cohort claim rejected')
    rows, lengths, results = experiment.fit_and_measure(args.run,args.output,args.threads)
    baseline_audit = {}
    for variant in ('frequency_rotate','fixed_rotate'):
        old_variant = variant.removesuffix('_rotate')
        with (args.run/'tokenizers'/f'{old_variant}_test_lengths.csv').open() as f:
            previous = [int(row['after_bpe']) for row in csv.DictReader(f)]
        if lengths[variant].tolist() != previous:
            raise ValueError(f'Adding unused ring slots changed rotation-on test lengths: {variant}')
        previous_training = json.loads((args.run/'tokenizers'/old_variant/'summary.json').read_text())
        if results[variant]['training']['final_tokens'] != previous_training['final_tokens']:
            raise ValueError(f'Adding unused ring slots changed rotation-on training lengths: {variant}')
        baseline_audit[variant] = dict(test_records_equal=len(previous), train_token_total_equal=True)
    experiment.report(args.run,args.output,rows,lengths,results)
    path = args.output/'results.json'
    payload = json.loads(path.read_text())
    payload['protocol'] = protocol
    payload['original_rotation_on_length_audit'] = baseline_audit
    payload['interpretation'] = ('Full original cohorts, with 512 ring symbols shared by all four arms; '
        'all BPEs freshly trained for 16621 merges. Final total vocabulary is 17470 including three special tokens. '
        'This is an exploratory compression diagnostic, not the frozen 17207-token model setting. '
        'Rotation-on lengths exactly reproduce the original E2a results on every test record and training totals. '
        'Positive relative_length_change means the numerator variant is longer. '
        'CIs are paired source-cluster bootstrap conditional on the fitted tokenizers.')
    write_json(path,payload)
    write_json(args.output/'full_cohort_complete.json', dict(protocol=protocol, baseline_audit=baseline_audit))
    print('FULL_COHORT_ROTATION_ABLATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
