#!/usr/bin/env python3
"""Cached frozen-backbone Alignment, then Full SFT, then one held-out evaluation."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import socket
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
import numpy as np
import torch
from experiments.rebuttal.common import write_json
from experiments.rebuttal.transfer_model import TransferData, TransferModel, checkpoint_source, metrics, microbatches
from pipeline.runtime_host import require_compute_host


def loss_fn(predictions, targets, reduction='mean'):
    if CLASSIFICATION:
        return torch.nn.functional.binary_cross_entropy_with_logits(predictions.float(), targets.float(), reduction=reduction)
    return torch.nn.functional.mse_loss(predictions.float(), targets.float(), reduction=reduction)


def emit(output, event, **values):
    payload = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), event=event, **values)
    print(json.dumps(payload), flush=True)
    with (output / 'events.jsonl').open('a') as f:
        f.write(json.dumps(payload) + '\n')
    write_json(output / 'progress.json', payload)


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def epoch_audit(selected, consumed):
    expected = np.sort(selected)
    actual = np.asarray(consumed, dtype=np.int64)
    unique, counts = np.unique(actual, return_counts=True)
    duplicates = int((counts - 1).sum())
    omitted = int(len(np.setdiff1d(expected, unique, assume_unique=True)))
    extra = int(len(np.setdiff1d(unique, expected, assume_unique=True)))
    if duplicates or omitted or extra or len(actual) != len(expected):
        raise ValueError(f'Epoch record accounting failed: duplicates={duplicates}, omitted={omitted}, extra={extra}')
    return dict(records=len(actual), unique_records=len(unique), duplicates=duplicates, omissions=omitted, unexpected=extra)


@torch.no_grad()
def infer(model, data, indices, token_budget, features=False):
    model.eval()
    parts = []
    for ids in microbatches(indices, data.lengths, token_budget):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = data.pack(ids)
            value = model.features(*inputs) if features else model(*inputs)
        parts.append(value.float().cpu().numpy())
    return np.concatenate(parts)


def smoke(model, data):
    ids = data.roles['train'][:3]
    model.eval()
    together = infer(model, data, ids, 2048, features=True)
    alone = np.concatenate([infer(model, data, [int(i)], 2048, features=True) for i in ids])
    # BF16 kernels may use different reduction orders for packed/single input.
    if not np.allclose(together, alone, atol=.035, rtol=.02):
        raise ValueError(f'Packed record isolation smoke failed: max_delta={np.abs(together-alone).max()}')
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = loss_fn(model(*data.pack(ids)), data.y(ids))
    loss.backward()
    gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or model.head.weight.grad is None:
        raise ValueError('Full-stage finite gradient smoke failed')
    model.zero_grad(set_to_none=True)
    return dict(packed_single_max_delta=float(np.abs(together-alone).max()), full_gradient_smoke=True)


def alignment(model, data, args, output):
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    selected = data.roles['train']
    emit(output, 'alignment_features', records=len(selected), validation_records=len(data.roles['validation']))
    train_features = infer(model, data, selected, args.micro_tokens, features=True)
    validation_features = infer(model, data, data.roles['validation'], args.micro_tokens, features=True)
    # Deterministic backbone, no dropout: cached features are exactly head-only SFT.
    x = torch.from_numpy(train_features).cuda()
    vx = torch.from_numpy(validation_features).cuda()
    y, vy = data.y(selected), data.y(data.roles['validation'])
    del train_features, validation_features
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.alignment_lr, weight_decay=.01)
    best_loss, best_epoch, bad, best_head = math.inf, 0, 0, None
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.head.train()
        order = rng.permutation(len(selected))
        consumed, loss_sum, steps = [], 0., 0
        for start in range(0, len(order), args.batch_size):
            local = order[start:start+args.batch_size]
            gpu_ids = torch.as_tensor(local, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.head(x[gpu_ids]).squeeze(-1), y[gpu_ids])
            loss.backward()
            if any(p.grad is not None for p in model.backbone.parameters()):
                raise ValueError('Backbone acquired gradients during frozen Alignment')
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(local)
            consumed.extend(selected[local].tolist()); steps += 1
        model.head.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model.head(vx).squeeze(-1), vy))
        if not math.isfinite(val_loss):
            raise ValueError('Nonfinite Alignment validation loss')
        if val_loss < best_loss - args.min_delta:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        else:
            bad += 1
        audit = epoch_audit(selected, consumed)
        row = dict(epoch=epoch, loss=loss_sum/len(selected), validation_loss=val_loss,
                   best_epoch=best_epoch, best_validation=best_loss, bad_epochs=bad, optimizer_steps=steps, audit=audit)
        history.append(row)
        emit(output, 'alignment_epoch', **row)
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    model.head.load_state_dict(best_head, strict=True)
    state = dict(stage='Alignment', best_epoch=best_epoch, stopping_epoch=epoch,
                 best_validation=best_loss, max_epochs=args.max_epochs, patience=args.patience,
                 feature_cache_equivalent_to_frozen_backbone=True, history=history)
    path = output / 'alignment_best.pt'
    save_checkpoint(path, dict(model=model.state_dict(), state=state, initialization=args.initialization, args=vars(args)))
    write_json(output / 'alignment.json', state)
    del x, vx, y, vy, optimizer, best_head
    gc.collect(); torch.cuda.empty_cache()
    return state


def full(model, data, args, output, alignment_state):
    # Reload the materialized matching Alignment best, rather than its last epoch.
    lineage = output / 'alignment_best.pt'
    checkpoint = torch.load(lineage, map_location='cpu', weights_only=False)
    if checkpoint['state']['best_epoch'] != alignment_state['best_epoch']:
        raise ValueError('Alignment lineage mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.full_lr, betas=(.9, .95), weight_decay=.1, fused=True)
    selected = data.roles['train']
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = math.ceil(len(selected) / args.batch_size)
    best_loss, best_epoch, bad, step = math.inf, 0, 0, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = selected[rng.permutation(len(selected))]
        consumed, total_loss = [], 0.
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = args.full_lr * min(1., (step + 1) / max(1, 2 * steps_per_epoch))
            for ids in microbatches(batch, data.lengths, args.micro_tokens):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = loss_fn(model(*data.pack(ids)), data.y(ids), reduction='sum')
                (loss / len(batch)).backward()
                total_loss += float(loss.detach())
                consumed.extend(ids.tolist())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError(f'Nonfinite Full gradient at epoch={epoch}, step={step}')
            optimizer.step(); step += 1
            if step % 10 == 0:
                emit(output, 'full_step', epoch=epoch, step=step, epoch_records=len(consumed),
                     memory_reserved_mib=torch.cuda.memory_reserved()/2**20)
        audit = epoch_audit(selected, consumed)
        predictions = infer(model, data, data.roles['validation'], args.micro_tokens)
        val_targets = (data.targets[data.roles['validation']] - data.mean) / data.std
        val_loss = float(loss_fn(torch.from_numpy(predictions), torch.from_numpy(val_targets)))
        if not math.isfinite(val_loss):
            raise ValueError('Nonfinite Full validation loss')
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, bad = val_loss, epoch, 0
        else:
            bad += 1
        row = dict(epoch=epoch, step=step, loss=total_loss/len(selected), validation_loss=val_loss,
                   best_epoch=best_epoch, best_validation=best_loss, bad_epochs=bad,
                   optimizer_steps=steps_per_epoch, audit=audit)
        history.append(row)
        if improved:
            save_checkpoint(output / 'full_best.pt', dict(model=model.state_dict(), state=row, alignment_best=str(lineage),
                            alignment_best_epoch=alignment_state['best_epoch'], args=vars(args)))
        emit(output, 'full_epoch', **row)
        write_json(output / 'full_history.json', history)
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    state = dict(stage='Full', best_epoch=best_epoch, stopping_epoch=epoch, best_validation=best_loss,
                 max_epochs=args.max_epochs, patience=args.patience, alignment_best=str(lineage),
                 alignment_best_epoch=alignment_state['best_epoch'], optimizer_steps=step, history=history)
    write_json(output / 'full.json', state)
    del optimizer
    model.zero_grad(set_to_none=True)
    checkpoint = torch.load(output / 'full_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    gc.collect(); torch.cuda.empty_cache()
    return state


def main():
    global CLASSIFICATION
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--old-run', type=Path, required=True)
    p.add_argument('--task', choices=['gb1', 'promoter', 'mrl'], required=True)
    p.add_argument('--arm', choices=['graph_random', 'graph_small', 'graph_joint', 'native_joint'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--micro-tokens', type=int, required=True)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--min-epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--min-delta', type=float, default=1e-5)
    p.add_argument('--alignment-lr', type=float, default=1e-3)
    p.add_argument('--full-lr', type=float, default=1e-4)
    args = p.parse_args()
    require_compute_host('transfer model training')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('This queue requires exactly the one scheduler-assigned GPU')
    torch.set_num_threads(1)
    output = args.run / 'models' / f'{args.task}_{args.arm}_{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    rep = 'native' if args.arm == 'native_joint' else 'graph'
    data = TransferData(args.run, args.task, rep)
    CLASSIFICATION = data.classification
    source = checkpoint_source(args.old_run, args.arm, args.seed)
    args.initialization = str(source) if source else 'fresh_random_backbone'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(host=socket.gethostname(), representation=rep, target_scaler_train_only=dict(mean=data.mean, std=data.std),
                  protocol='head-only Alignment then Full; both max100/validation-loss early stopping',
                  small_pretrain_interpretation='small-molecule transfer control, NOT same-domain pretraining',
                  primary_metric=data.metadata['metric'], train_records=len(data.roles['train']))
    write_json(output / 'config.json', config)
    emit(output, 'loading', **config)
    start = time.monotonic()
    model = TransferModel(args.seed, source).cuda()
    emit(output, 'smoke_passed', **smoke(model, data))
    a = alignment(model, data, args, output)
    f = full(model, data, args, output, a)
    emit(output, 'test_evaluation', selected_full_epoch=f['best_epoch'])
    test = data.roles['test']
    predictions = infer(model, data, test, args.micro_tokens)
    if not data.classification:
        predictions = predictions * data.std + data.mean
    scores = metrics(data.targets[test], predictions, data.classification)
    if data.classification:
        scores['accuracy_over_requested'] = scores['accuracy'] * len(test) / data.metadata['requested']['test']
    np.savez(output / 'test_predictions.npz', indices=test, targets=data.targets[test], predictions=predictions)
    (output / 'test_ids.json').write_text(json.dumps([data.ids[i] for i in test]) + '\n')
    result = dict(task=args.task, arm=args.arm, seed=args.seed, metrics=scores,
                  test_records=len(test), original_test_records=data.metadata['requested']['test'],
                  test_coverage=len(test)/data.metadata['requested']['test'],
                  alignment_best_epoch=a['best_epoch'], alignment_stopping_epoch=a['stopping_epoch'],
                  full_best_epoch=f['best_epoch'], full_stopping_epoch=f['stopping_epoch'],
                  full_best_validation=f['best_validation'], seconds=time.monotonic()-start,
                  pretraining_overlap=data.metadata['audit']['pretraining_overlap'], config=config)
    write_json(output / 'result.json', result)
    emit(output, 'complete', **result)


if __name__ == '__main__':
    main()
