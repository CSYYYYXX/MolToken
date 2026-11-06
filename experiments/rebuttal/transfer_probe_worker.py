#!/usr/bin/env python3
"""Real forward/backward probes to size the new one-GPU transfer queue."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
import numpy as np
import torch
from experiments.rebuttal.common import write_json
from experiments.rebuttal.transfer_model import TransferData, TransferModel, microbatches
from pipeline.runtime_host import require_compute_host


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    require_compute_host('transfer GPU memory probe')
    if torch.cuda.device_count() != 1:
        raise ValueError('Expected exactly one scheduler-assigned GPU')
    torch.set_num_threads(1)
    device = torch.cuda.get_device_properties(0)
    total = device.total_memory
    target = .205 * total
    plan = dict(gpu=device.name, total_mib=total/2**20, target_aggregate_fraction=.70,
                max_reserved_aggregate_fraction=.84, configurations={}, attempts=[])
    model = TransferModel(1701).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    # Same effective record batch across arms of one task. Only microbatching varies.
    batch_sizes = dict(gb1=1024, promoter=2048, mrl=8192)
    for task, batch_size in batch_sizes.items():
        for rep in ('graph', 'native'):
            data = TransferData(args.run, task, rep)
            eligible = data.roles['train']
            # Probe the longest training records; no test labels or test-selected tuning.
            indices = eligible[np.argsort(data.lengths[eligible])[::-1][:batch_size]]
            max_tokens = int(data.lengths[indices].sum())
            budget = min(32768, max_tokens)
            good = None
            for attempt in range(8):
                model.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                sample = next(microbatches(indices, data.lengths, budget))
                started = time.monotonic()
                oom = False
                try:
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            prediction = model(*data.pack(sample))
                            target_y = data.y(sample)
                            loss = (torch.nn.functional.binary_cross_entropy_with_logits(prediction.float(), target_y)
                                    if data.classification else torch.nn.functional.mse_loss(prediction.float(), target_y))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_reserved()
                    row = dict(task=task, representation=rep, micro_tokens=budget, records=len(sample),
                               actual_tokens=int(data.lengths[sample].sum()), reserved_mib=peak/2**20,
                               seconds=time.monotonic()-started, oom=False)
                    good = dict(batch_size=batch_size, micro_tokens=budget, probe_records=len(sample),
                                probe_reserved_mib=peak/2**20, estimated_process_mib=(peak/2**20 + 1100)*1.08)
                except torch.cuda.OutOfMemoryError as error:
                    oom = True
                    row = dict(task=task, representation=rep, micro_tokens=budget, oom=True, error=str(error))
                plan['attempts'].append(row)
                print(json.dumps(row), flush=True)
                model.zero_grad(set_to_none=True)
                if oom:
                    if good is None:
                        budget //= 2
                        if budget < 512:
                            raise RuntimeError('No complete-record batch fits the assigned GPU')
                        continue
                    break
                if peak >= target*.92 or budget == max_tokens:
                    break
                # Scale from measured activation memory, preserving optimizer/context margin.
                next_budget = min(max_tokens, int(budget * min(1.8, target / max(peak, 1)) // 1024 * 1024))
                if next_budget <= budget:
                    break
                budget = next_budget
            if good is None:
                raise RuntimeError(f'No successful probe for {task}/{rep}')
            plan['configurations'][f'{task}:{rep}'] = good
            write_json(args.run / 'runtime' / 'gpu_profile_partial.json', plan)
    del optimizer, model
    gc.collect(); torch.cuda.empty_cache()
    plan['nvidia_smi_after_probe'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    write_json(args.run / 'runtime' / 'gpu_profile.json', plan)
    print('TRANSFER_GPU_PROFILE_COMPLETE', json.dumps(plan['configurations']), flush=True)


if __name__ == '__main__':
    main()
