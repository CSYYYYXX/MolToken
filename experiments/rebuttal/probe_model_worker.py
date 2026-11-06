#!/usr/bin/env python3
"""Bounded GPU correctness and real-record memory/throughput preflight."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))


def main():
    import torch
    from experiments.rebuttal.model import build_model, collate, hf_generation_model
    from experiments.diagnostics.native_domain_bpe_probe import read_sequences
    from pipeline.runtime_host import require_compute_host
    from cli.pretrain.train_qwen3_pretrain import build_next_token_labels_from_segments
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--tokens', type=int, nargs='+', default=[65536, 131072, 262144, 393216])
    args = p.parse_args()
    print('GPU_PROBE_HOST', require_compute_host('rebuttal GPU preflight'), flush=True)
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'], text=True), flush=True)
    model = build_model().cuda()
    print('PARAMETERS', sum(p.numel() for p in model.parameters()), flush=True)
    a, b = (0, [3, 4, 5, 6]), (2, [7, 8, 9])
    batch = collate([a, b])
    targets = build_next_token_labels_from_segments(batch['labels'], torch.ones_like(batch['labels'], dtype=torch.bool), batch['segment_lengths'])
    assert int((targets != -100).sum()) == 9
    assert targets.tolist() == [-100, 6, 7, 8, 9, 2, -100, -100, 10, 11, 12, 2, -100]
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        joined = model(**batch)
        single = [model(**collate([r])).loss for r in (a, b)]
        expected = (single[0]*5 + single[1]*4)/9
        torch.testing.assert_close(joined.loss, expected, atol=2e-4, rtol=2e-4)
        # Check the generation implementation against the packed implementation.
        hf = hf_generation_model(model).cuda().eval()
        one = collate([a])
        prefix = model.embed_tokens(one['input_ids'])
        plain = hf(inputs_embeds=prefix, use_cache=True).logits
        packed = model(input_ids=one['input_ids'], segment_lengths=one['segment_lengths']).logits
        torch.testing.assert_close(plain.float(), packed.float(), atol=.04, rtol=.02)
        del hf, prefix, plain, packed
    print('PREFIX_LABEL_ISOLATION_AND_GENERATION_PARITY_PASS', flush=True)
    del joined, single, expected, batch, one
    # Use real serialized training records. This is a bounded runtime test, not a formal run.
    records = []
    parts = sorted((args.run / 'tokenizers/parts').glob('train_*/graph.bin'))
    for path in parts:
        for ids in read_sequences(path):
            if len(ids) <= 509:
                records.append((0, ids))
            if sum(len(x[1])+3 for x in records) >= max(args.tokens):
                break
        if sum(len(x[1])+3 for x in records) >= max(args.tokens):
            break
    if sum(len(x[1])+3 for x in records) < max(args.tokens):
        raise RuntimeError('Not enough real encoded records for the requested memory probe')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    model.train()
    for limit in args.tokens:
        selected, count = [], 0
        for row in records:
            if count + len(row[1])+3 > limit:
                break
            selected.append(row)
            count += len(row[1])+3
        batch = collate(selected)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(**batch)
        result.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print(json.dumps(dict(event='gpu_memory_probe', input_tokens=count, records=len(selected),
            loss=float(result.loss), seconds=time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            memory=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'], text=True).strip())), flush=True)
        del result, batch
    print('GPU_PREFLIGHT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
