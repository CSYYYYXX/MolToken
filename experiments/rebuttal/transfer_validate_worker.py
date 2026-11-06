#!/usr/bin/env python3
"""Worker-only input, accounting, metric, and checkpoint checks before GPU dispatch."""
import argparse
import json
from pathlib import Path
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host
require_compute_host('transfer CPU validation')
import numpy as np
import torch
from experiments.rebuttal.common import write_json
from experiments.rebuttal.model import VOCAB_SIZE, build_model
from experiments.rebuttal.transfer_model import TransferData, checkpoint_source, metrics, microbatches
from experiments.rebuttal.transfer_train_worker import epoch_audit
from cli.pretrain.train_qwen3_pretrain import build_packed_position_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--old-run', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    checks = []
    for task in ('gb1', 'promoter', 'mrl'):
        pairs = []
        for rep in ('graph', 'native'):
            data = TransferData(args.run, task, rep)
            ids = data.roles['train'][:11]
            groups = list(microbatches(ids, data.lengths, 512))
            if not np.array_equal(np.concatenate(groups), ids):
                raise ValueError('Microbatch order/coverage differs from record order')
            inputs, lengths = data.pack(ids, device='cpu')
            pos = build_packed_position_ids(lengths).squeeze(0).numpy()
            start = 0
            for i, n in zip(ids, lengths.tolist()):
                segment = inputs[0, start:start+n].numpy()
                assert segment[0] == VOCAB_SIZE + data.domain and segment[1] == 1 and segment[-1] == 2
                assert np.array_equal(segment[2:-1] - 3, data.tokens[data.offsets[i]:data.offsets[i+1]])
                assert np.array_equal(pos[start:start+n], np.arange(n))
                start += n
            audit = epoch_audit(ids, np.concatenate(groups))
            try:
                epoch_audit(ids, np.append(ids, ids[0]))
            except ValueError:
                pass
            else:
                raise AssertionError('Duplicate record audit failed to reject duplicate consumption')
            if not data.classification:
                assert abs(float(data.y(data.roles['train'], device='cpu').mean())) < 1e-5
            checks.append(dict(task=task, representation=rep, packed_input=True, positions_reset=True, epoch_audit=audit))
            pairs.append(data)
        assert pairs[0].ids == pairs[1].ids
        assert np.array_equal(pairs[0].splits, pairs[1].splits)
        assert np.array_equal(pairs[0].targets, pairs[1].targets)
    scores = metrics([1, 2, 3, 4], [1, 2, 3, 4], False)
    assert scores['r2'] == 1 and scores['pearson_squared'] == 1 and scores['rmse'] == 0
    scores = metrics([0, 1, 0, 1], [-2, 2, -1, 1], True)
    assert scores['accuracy'] == 1 and scores['auroc'] == 1 and scores['mcc'] == 1
    model = build_model(42)
    sources = []
    for seed in (42, 43, 44):
        for arm in ('graph_small', 'graph_joint', 'native_joint'):
            source = checkpoint_source(args.old_run, arm, seed)
            state = torch.load(source, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            sources.append(dict(arm=arm, seed=seed, source=str(source),
                                parameters=sum(p.numel() for p in model.parameters()), state=state.get('state')))
            del state
            print('SOURCE_CHECKPOINT_VALID', arm, seed, flush=True)
    for file in Path(__file__).parent.glob('transfer_*worker.py'):
        compile(file.read_text(), str(file), 'exec')
    result = dict(status='passed', input_checks=checks, metrics_checked=True, source_checkpoints=sources,
                  gpu_checks='pending: packed isolation, finite gradients, measured memory, BF16 kernel support')
    write_json(args.run / 'runtime' / 'cpu_validation.json', result)
    print('TRANSFER_CPU_VALIDATION_PASSED', flush=True)


if __name__ == '__main__':
    main()
