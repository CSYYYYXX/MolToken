#!/usr/bin/env python3
"""Diagnose cross-record isolation separately from shape-dependent roundoff."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host
import numpy as np
import torch
import xformers.ops as xops
from experiments.rebuttal.common import write_json
from experiments.rebuttal.transfer_model import TransferData, TransferModel, checkpoint_source


@contextmanager
def cpu_reference_attention():
    """Diagnostic only: exact block mask with PyTorch's dense FP32 CPU attention."""
    original = xops.memory_efficient_attention

    def attention(q, k, v, attn_bias=None, p=0., **kwargs):
        if q.device.type != 'cpu' or p != 0.:
            raise ValueError('This diagnostic adapter requires CPU evaluation without dropout')
        mask = attn_bias.materialize((q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
                                     device=q.device, dtype=q.dtype)
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                    v.transpose(1, 2), attn_mask=mask, dropout_p=0.).transpose(1, 2)

    xops.memory_efficient_attention = attention
    try:
        yield
    finally:
        xops.memory_efficient_attention = original


@torch.no_grad()
def isolation_report(model, data, device='cuda', mixed=True):
    """Keep tensor shapes constant and perturb other records, including a positive control."""
    model.eval()
    selected = data.roles['train'][:3]
    inputs, lengths = data.pack(selected, device=device)
    starts = [0] + lengths.cumsum(0).tolist()

    def forward(ids, lens):
        context = torch.autocast(device, dtype=torch.bfloat16) if mixed else nullcontext()
        with context:
            return model.features(ids, lens).float()

    baseline = forward(inputs, lengths)
    alone = torch.cat([forward(*data.pack([int(i)], device=device)) for i in selected])
    shape_difference = (baseline - alone).abs()
    perturbations = []
    changed_batch = None
    for target in range(3):
        changed = inputs.clone()
        for other in range(3):
            if other == target:
                continue
            lo, hi = starts[other]+2, starts[other+1]-1
            body = changed[0, lo:hi]
            changed[0, lo:hi] = (body - 3 + 1) % 17204 + 3
        delta = (forward(changed, lengths)[target] - baseline[target]).abs()
        perturbations.append(dict(target=target, max_delta=float(delta.max()),
                                  unchanged=bool(torch.equal(forward(changed, lengths)[target], baseline[target]))))
        if target == 2:
            changed_batch = changed
    # Removing segment boundaries MUST make the last record sensitive to earlier records.
    hooks = []
    for layer in model.backbone.layers:
        hooks.append(layer.self_attn.register_forward_pre_hook(
            lambda module, args: (args[0], args[1], xops.fmha.attn_bias.LowerTriangularMask())))
    try:
        leaking_delta = (forward(inputs, lengths)[2] - forward(changed_batch, lengths)[2]).abs()
    finally:
        for hook in hooks:
            hook.remove()
    return dict(device=device, precision='bf16_autocast' if mixed else 'fp32',
                lengths=lengths.tolist(), shape_changing_comparison_max_delta=float(shape_difference.max()),
                shape_changing_comparison_rmse=float(torch.sqrt((baseline-alone).square().mean())),
                fixed_shape_perturbations=perturbations,
                positive_control_without_boundaries_max_delta=float(leaking_delta.max()),
                fixed_shape_isolation_pass=all(p['max_delta'] <= 1e-6 for p in perturbations),
                positive_control_detected=bool(leaking_delta.max() > 1e-3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--old-run', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require_compute_host('transfer isolation diagnosis')
    torch.set_num_threads(4 if args.device == 'cpu' else 1)
    cases = [('mrl', 'native_joint', s) for s in (42, 43, 44)] + [('gb1', 'graph_joint', 44)]
    reports = []
    for task, arm, seed in cases:
        data = TransferData(args.run, task, 'native' if arm.startswith('native') else 'graph')
        model = TransferModel(seed, checkpoint_source(args.old_run, arm, seed)).to(args.device)
        context = cpu_reference_attention() if args.device == 'cpu' else nullcontext()
        with context:
            report = isolation_report(model, data, args.device, mixed=args.device == 'cuda')
        report.update(task=task, arm=arm, seed=seed)
        reports.append(report)
        print(json.dumps(report), flush=True)
        write_json(args.output, dict(status='diagnostic_in_progress', reports=reports,
                                    gpu_backend_verified=args.device == 'cuda'))
        del model
    passed = all(r['fixed_shape_isolation_pass'] and r['positive_control_detected'] for r in reports)
    write_json(args.output, dict(status='passed' if passed else 'failed', reports=reports,
                                gpu_backend_verified=args.device == 'cuda'))
    if not passed:
        raise RuntimeError('Semantic isolation diagnostic failed; do not bypass the check')


if __name__ == '__main__':
    main()
