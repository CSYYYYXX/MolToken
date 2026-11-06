#!/usr/bin/env python3
"""Single-GPU, full-parameter E2/E3 training with exact record/token accounting."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import DOMAINS, write_json
from pipeline.runtime_host import require_compute_host


def emit(event, **values):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **values)), flush=True)


class Stream:
    def __init__(self, root, scope, seed, variant):
        self.root = Path(root) / 'streams' / f'{scope}_{seed}'
        self.variant = variant
        self.chunk = self.offset = self.epoch = 0
        self.data = None
        self.loaded = None
        self.wait_seconds = 0.

    def next(self):
        import numpy as np
        from experiments.diagnostics.native_domain_bpe_probe import read_sequences
        while True:
            if self.loaded != self.chunk:
                directory = self.root / f'{self.chunk:06d}'
                waited = time.monotonic()
                while not (directory / 'complete.json').exists():
                    failure = self.root.parents[1] / 'logs/cpu_suite.failed'
                    if failure.exists():
                        raise RuntimeError(f'CPU preparation failed: {failure.read_text()}')
                    if (self.root / 'complete.json').exists():
                        status = json.loads((self.root / 'complete.json').read_text())
                        if self.chunk >= status['chunks']:
                            if not status['candidate_epoch_exhausted']:
                                raise RuntimeError('Materialized stream exhausted before compute target')
                            self.chunk = 0
                            self.epoch += 1
                            directory = self.root / '000000'
                            break
                    emit('waiting_for_preprocessing', directory=str(directory))
                    time.sleep(10)
                self.wait_seconds += time.monotonic()-waited
                self.data = list(read_sequences(directory / f'{self.variant}.bin'))
                self.ids = np.load(directory / 'ids.npy')
                self.domains = np.load(directory / 'domains.npy')
                if len(self.data) != len(self.ids) or len(self.ids) != len(self.domains):
                    raise ValueError('Stream sidecar counts differ')
                self.loaded = self.chunk
            if self.offset >= len(self.data):
                self.chunk += 1
                self.offset = 0
                continue
            i = self.offset
            self.offset += 1
            return (int(self.domains[i]), self.data[i], int(self.ids[i]))

    def state_dict(self):
        return dict(chunk=self.chunk, offset=self.offset, epoch=self.epoch, wait_seconds=self.wait_seconds)

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self.loaded = None


def moses_records(root, variant):
    from experiments.diagnostics.native_domain_bpe_probe import read_sequences
    manifest = json.loads((Path(root) / 'moses/manifest.json').read_text())
    data = {'train': [], 'validation': []}
    identifiers = {'train': [], 'validation': []}
    for part in manifest['parts']:
        path = Path(part['directory'])
        sequences = list(read_sequences(path / f'{variant}.bin'))
        ids = json.loads((path / 'ids.json').read_text())
        if len(ids) != len(sequences) or len(ids) != part['records']:
            raise ValueError('MOSES row counts differ')
        data[part['role']].extend((0, seq, key) for seq, key in zip(sequences, ids))
        identifiers[part['role']].extend(ids)
    a, b = set(identifiers['train']), set(identifiers['validation'])
    if a & b or len(a) != len(data['train']) or len(b) != len(data['validation']):
        raise ValueError('MOSES records are duplicated or overlap the validation split')
    return data


def microbatches(records, limit):
    batch, size = [], 0
    for row in records:
        length = len(row[1]) + 3
        if batch and size + length > limit:
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += length
    if batch:
        yield batch


def optimize(model, optimizer, records, limit, learning_rate):
    import torch
    from experiments.rebuttal.model import collate
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    predicted = sum(len(row[1])+1 for row in records)
    numerator = torch.zeros((), device='cuda')
    for micro in microbatches(records, limit):
        batch = collate([(row[0],row[1]) for row in micro])
        count = sum(len(row[1])+1 for row in micro)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(**batch)
        if not torch.isfinite(output.loss):
            raise FloatingPointError(f'Nonfinite loss: {float(output.loss)}')
        (output.loss * (count / predicted)).backward()
        numerator += output.loss.detach() * count
        del output, batch
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(numerator/predicted), float(norm)


def validate(model, records, limit):
    import torch
    from experiments.rebuttal.model import collate
    model.eval()
    numerator = torch.zeros((), device='cuda')
    denominator = 0
    with torch.no_grad():
        for micro in microbatches(records, limit):
            batch = collate([(row[0],row[1]) for row in micro])
            count = sum(len(row[1])+1 for row in micro)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(**batch)
            numerator += result.loss * count
            denominator += count
            del result, batch
    model.train()
    value = float(numerator/denominator)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite validation loss')
    return value


def save(path, model, optimizer, state, args, stream=None):
    import torch
    temporary = path.with_suffix('.tmp')
    payload = dict(model=model.state_dict(), config=model.config.to_dict(), state=state,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
        stream=stream.state_dict() if stream else None)
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    import numpy as np
    import torch
    from experiments.rebuttal.model import build_model
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--mode', choices=['pretrain','sft'], required=True)
    p.add_argument('--variant', choices=['graph','native'], required=True)
    p.add_argument('--scope', choices=['joint','small'], default='joint')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--init', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--global-records', type=int, default=12288)
    p.add_argument('--micro-input-tokens', type=int, default=131072)
    p.add_argument('--flops', type=float, default=5e17)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--relative-min-delta', type=float, default=1e-4)
    p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args()
    require_compute_host('E2/E3 Transformer training')
    root = args.run / 'models' / args.name
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    model = build_model(args.seed).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9,.95), weight_decay=.1, fused=True)
    nparams = sum(v.numel() for v in model.parameters())
    target = math.ceil(args.flops/(6*nparams))
    lr = args.learning_rate or (3e-4 if args.mode=='pretrain' else 1e-4)
    state = dict(step=0, epoch=0, input_tokens=0, predicted_tokens=0, records=0,
                 best_validation=None, best_epoch=None, bad_epochs=0, training_seconds=0., domains={})
    stream = Stream(args.run,args.scope,args.data_seed,args.variant) if args.mode=='pretrain' else None
    if args.init:
        checkpoint = torch.load(args.init, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
    if args.resume:
        checkpoint = torch.load(root / 'last.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        state = checkpoint['state']
        torch.set_rng_state(checkpoint['rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        if stream:
            stream.load_state_dict(checkpoint['stream'])
        del checkpoint
    write_json(root / 'protocol.json', dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, parameters=nparams,
        nominal_compute='6 * parameters * nonpad_input_positions_including_TYPE_BOS_EOS',
        target_input_tokens=target if stream else None, learning_rate=lr,
        sft_lr_tuning_budget='one fixed LR=1e-4 for every method; no test-based selection',
        max_epochs=args.max_epochs, early_stopping='train-derived validation; relative improvement threshold',
        initialization=str(args.init) if args.init else 'random', phase='smoke' if args.smoke_steps else 'formal'))
    emit('training_start', name=args.name, parameters=nparams, target_input_tokens=target if stream else None)
    model.train()
    wall_start = time.monotonic()
    if stream:
        # An exact bitset accounts for record reuse without keeping Python sets of millions of IDs.
        catalog = json.loads((args.run / 'raw_catalog/manifest.json').read_text())
        total_raw = sum(p['records'] for p in catalog['parts'])
        seen_path = root / 'seen.npy'
        seen = np.lib.format.open_memmap(seen_path, mode='w+', dtype=np.bool_, shape=(total_raw,))
        seen[:] = False
        if args.resume:
            # Reconstruct from the checkpoint cursor, not a memmap that could
            # contain observations made after the last durable checkpoint.
            if stream.epoch:
                complete = json.loads((stream.root/'complete.json').read_text())
                for i in range(complete['chunks']):
                    seen[np.load(stream.root/f'{i:06d}'/'ids.npy')] = True
            for i in range(stream.chunk):
                seen[np.load(stream.root/f'{i:06d}'/'ids.npy')] = True
            if stream.offset:
                seen[np.load(stream.root/f'{stream.chunk:06d}'/'ids.npy')[:stream.offset]] = True
        unique = int(seen.sum())
        if args.resume and unique != state['unique_records']:
            raise ValueError('Checkpoint cursor and unique-record accounting disagree')
        while state['input_tokens'] < target:
            records, tokens = [], 0
            while len(records) < args.global_records and state['input_tokens']+tokens < target:
                row = stream.next()
                records.append(row)
                tokens += len(row[1])+3
            progress = min(1., (state['input_tokens']+tokens/2)/target)
            factor = progress/.02 if progress < .02 else .1 + .9*(1+math.cos(math.pi*(progress-.02)/.98))/2
            start = time.monotonic()
            loss, norm = optimize(model,optimizer,records,args.micro_input_tokens,lr*factor)
            state['training_seconds'] += time.monotonic()-start
            for domain, ids, gid in records:
                if not seen[gid]:
                    unique += 1
                    seen[gid] = True
                key = DOMAINS[domain]
                item = state['domains'].setdefault(key,dict(records=0,input_tokens=0,predicted_tokens=0))
                item['records'] += 1
                item['input_tokens'] += len(ids)+3
                item['predicted_tokens'] += len(ids)+1
            state['input_tokens'] += tokens
            state['predicted_tokens'] += tokens-2*len(records)
            state['records'] += len(records)
            state['unique_records'] = unique
            state['repeated_record_consumptions'] = state['records']-unique
            state['step'] += 1
            if state['step']%20 == 0 or state['step']==1:
                emit('pretrain_progress', loss=loss, gradient_norm=norm, lr=lr*factor,
                    nominal_flops=6*nparams*state['input_tokens'], **state,
                    gpu_memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader'],text=True).strip())
                write_json(root / 'progress.json',state)
            if state['step']%100 == 0:
                save(root/'last.pt',model,optimizer,state,args,stream)
                seen.flush()
            if args.smoke_steps and state['step'] >= args.smoke_steps:
                break
        seen.flush()
        state['preprocessing_wait_seconds'] = stream.wait_seconds
        state['nominal_flops'] = 6*nparams*state['input_tokens']
        state['input_budget_overshoot'] = state['input_tokens']-target
        save(root/'final.pt',model,optimizer,state,args,stream)
    else:
        data = moses_records(args.run,args.variant)
        write_json(root/'data_counts.json',{k:len(v) for k,v in data.items()})
        count = len(data['train'])
        for epoch in range(state['epoch']+1,args.max_epochs+1):
            order = np.random.default_rng(args.seed+epoch*1000003).permutation(count)
            used = np.zeros(count,dtype=np.bool_)
            loss_sum, pred_sum = 0., 0
            for start_index in range(0,count,args.global_records):
                indices = order[start_index:start_index+args.global_records]
                if used[indices].any():
                    raise RuntimeError('Duplicate record in SFT epoch')
                used[indices] = True
                records = [data['train'][i] for i in indices]
                start = time.monotonic()
                loss, norm = optimize(model,optimizer,records,args.micro_input_tokens,lr)
                state['training_seconds'] += time.monotonic()-start
                tokens = sum(len(r[1])+3 for r in records)
                predicted = tokens-2*len(records)
                loss_sum += loss*predicted
                pred_sum += predicted
                state['input_tokens'] += tokens
                state['predicted_tokens'] += predicted
                state['records'] += len(records)
                state['step'] += 1
                if state['step']%20 == 0 or state['step']==1:
                    emit('sft_progress', epoch=epoch, step=state['step'], records_in_epoch=int(used.sum()),
                        loss=loss, gradient_norm=norm, gpu_memory=subprocess.check_output(['nvidia-smi',
                        '--query-gpu=memory.used,memory.total','--format=csv,noheader'],text=True).strip())
                if args.smoke_steps and state['step'] >= args.smoke_steps:
                    save(root/'smoke.pt',model,optimizer,state,args)
                    write_json(root/'smoke_complete.json',state)
                    return
            if not used.all() or len(order) != count:
                raise RuntimeError('SFT epoch omitted records')
            validation = validate(model,data['validation'],args.micro_input_tokens)
            state['epoch'] = epoch
            previous = state['best_validation']
            significant = previous is None or validation < previous*(1-args.relative_min_delta)
            if significant:
                state['bad_epochs'] = 0
            else:
                state['bad_epochs'] += 1
            if previous is None or validation < previous:
                state['best_validation'], state['best_epoch'] = validation,epoch
                save(root/'best.pt',model,optimizer,state,args)
            state['last_validation'] = validation
            state['epoch_unique_records'] = count
            state['epoch_duplicates'] = state['epoch_omissions'] = 0
            save(root/'last.pt',model,optimizer,state,args)
            write_json(root/'progress.json',state)
            emit('sft_epoch', train_loss=loss_sum/pred_sum, validation_loss=validation, **state)
            if state['bad_epochs'] >= args.patience:
                break
        state['stopping_epoch'] = state['epoch']
    state['wall_seconds'] = time.monotonic()-wall_start
    write_json(root/('smoke_complete.json' if args.smoke_steps else 'complete.json'),state)
    emit('training_complete',name=args.name,**state)


if __name__ == '__main__':
    main()
