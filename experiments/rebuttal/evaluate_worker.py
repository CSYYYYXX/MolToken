#!/usr/bin/env python3
"""Exactly 30k unfiltered draws and the five requested official MOSES metrics."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import BASE_SIZE, DEFAULT_EDGE, write_json
from pipeline.runtime_host import require_compute_host


def init_decode(root, variant):
    global BPE_MODEL, INVERSE, VARIANT
    from experiments.rebuttal.tokenizer import BPE
    root = Path(root)
    VARIANT = variant
    if variant == 'graph':
        vocab = json.loads((root/'tokenizer_graphs/base_vocab.json').read_text())
        INVERSE = {i:t for t,i in vocab['token_to_id'].items()}
        for i in range(vocab['ring_base'],BASE_SIZE):
            INVERSE[i] = 'R'+str(i-vocab['ring_base']+1)
        base = BASE_SIZE
    else:
        alphabet = json.loads((root/'tokenizers/alphabet.json').read_text())
        INVERSE = {i:c for c,i in alphabet['smiles'].items()}
        base = sum(map(len,alphabet.values()))
    BPE_MODEL = BPE(root/'tokenizers'/variant/'merges.csv',base)


def decode_attempt(row):
    from rdkit import Chem, RDLogger
    from pipeline.serialization.fddfs import deserialize_fddfs
    from cli.evaluate.eval_axis_b_samples import graph_to_rdkit_mol
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    row = dict(row, smiles=None, decode_error=None)
    if row['invalid_model_id']:
        row['decode_error'] = 'invalid_model_id'
    elif not row['ended_with_eos']:
        row['decode_error'] = 'missing_eos'
    elif not row['generated_bpe_ids']:
        row['decode_error'] = 'empty_generation'
    else:
        try:
            ids = BPE_MODEL.decode(row['generated_bpe_ids'])
            if VARIANT == 'graph':
                graph = deserialize_fddfs([INVERSE[i] for i in ids],DEFAULT_EDGE)
                molecule, info = graph_to_rdkit_mol(graph,strict=True,aromatic_n_h_completion=True)
                row['aromatic_n_h_completion'] = info.get('aromatic_n_h_completion')
                if not info['sanitize_ok']:
                    raise ValueError('strict RDKit sanitization failed')
            else:
                if any(i not in INVERSE for i in ids):
                    raise ValueError('generated content belongs to a non-SMILES namespace')
                value = ''.join(INVERSE[i] for i in ids)
                molecule = Chem.MolFromSmiles(value)
                if molecule is None:
                    raise ValueError('invalid generated SMILES')
            if molecule.GetNumAtoms() == 0:
                raise ValueError('empty molecule')
            row['smiles'] = Chem.MolToSmiles(molecule,canonical=True,isomericSmiles=True)
        except (ValueError, RuntimeError, RecursionError, IndexError, KeyError) as error:
            # These are malformed generated graph/string cases, retained in
            # the denominator with their exact errors; setup/imports are outside.
            row['decode_error'] = type(error).__name__+': '+str(error)
    return row


def generate(args):
    import numpy as np
    import torch
    import orjson
    from experiments.rebuttal.model import build_model, hf_generation_model
    output = args.output
    output.mkdir(parents=True,exist_ok=False)
    checkpoint = torch.load(args.checkpoint,map_location='cpu')
    model = build_model(args.seed)
    model.load_state_dict(checkpoint['model'],strict=True)
    provenance = checkpoint['state']
    del checkpoint
    hf = hf_generation_model(model).to(device=args.device,dtype=torch.bfloat16).eval()
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    prefix_vector = model.embed_tokens.domain.weight[0].detach().to(device=args.device,dtype=torch.bfloat16)
    del model
    rows, start = [], time.monotonic()
    with torch.inference_mode():
        for offset in range(0,args.samples,args.batch_size):
            count = min(args.batch_size,args.samples-offset)
            bos = hf.model.embed_tokens(torch.ones(count,device=args.device,dtype=torch.long))
            prefix = torch.stack((prefix_vector.expand(count,-1),bos),dim=1)
            sampled = torch.full((count,510),-1,device=args.device,dtype=torch.long)
            done = torch.zeros(count,device=args.device,dtype=torch.bool)
            cache, next_ids = None, None
            for step in range(510):
                kwargs = dict(inputs_embeds=prefix) if step==0 else dict(input_ids=next_ids[:,None])
                result = hf(**kwargs,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache = result.past_key_values
                logits = result.logits[:,-1,:].float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite generation logits')
                # Temperature=1, top_p=1, top_k=0, no validity filtering or retries.
                token = torch.multinomial(torch.softmax(logits,dim=-1),1).squeeze(-1)
                sampled[:,step] = torch.where(done,torch.full_like(token,-1),token)
                done |= (token==2) | (token<2)
                next_ids = torch.where(done,torch.zeros_like(token),token)
                del result,logits,token
                if bool(done.all()):
                    break
            array = sampled.cpu().numpy()
            for i in range(count):
                values = array[i][array[i]>=0].tolist()
                ended = bool(values and values[-1]==2)
                invalid = any(v<2 for v in values)
                body = [v-3 for v in values if v>=3]
                rows.append(dict(attempt_index=offset+i,seed=args.seed,generated_model_ids=values,
                    generated_bpe_ids=body,ended_with_eos=ended,invalid_model_id=invalid))
            print(json.dumps(dict(event='generation_progress',draws=len(rows),target=args.samples,
                seconds=time.monotonic()-start)),flush=True)
            del cache,prefix,sampled,next_ids,bos,done
    del hf,prefix_vector
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    with (output/'raw_attempts.jsonl').open('wb') as handle:
        for row in rows:
            handle.write(orjson.dumps(row)+b'\n')
    counts = Counter()
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,
            initializer=init_decode,initargs=(str(args.run),args.variant)) as pool:
        with (output/'decoded.jsonl').open('wb') as handle:
            for row in pool.map(decode_attempt,rows,chunksize=100):
                handle.write(orjson.dumps(row)+b'\n')
                counts['valid' if row['smiles'] is not None else 'invalid'] += 1
    write_json(output/'generation_complete.json',dict(samples=len(rows),counts=dict(counts),
        variant=args.variant,checkpoint=str(args.checkpoint),checkpoint_provenance=provenance,
        seed=args.seed,seconds=time.monotonic()-start,temperature=1.,top_p=1.,top_k=0,
        retries=0,max_total_context=512,formal=args.samples==30000,
        graph_decoding='existing strict graph decoder with aromatic N-H completion; recorded heavy-graph roundtrip does not imply stereochemical identity'))
    print('GENERATION_COMPLETE',flush=True)


def score(args):
    import orjson
    import torch
    data = args.run.parent.parent/'data'
    assets = data/'downloads_raw/MOSES-dd7ed6ab38e23afd3ef5371d67939a1760bd8599'
    sys.path.insert(0,str(assets/'source'))
    from cli.evaluate.evaluate_axis_b_moses_official import load_official_api,load_official_references,load_official_statistics
    api = load_official_api({})
    refs = data/'benchmark/raw/moses/official_core13_dd7ed6'
    references = load_official_references(refs/'train.csv',refs/'test.csv',refs/'test_scaffolds.csv')
    stats = load_official_statistics(assets/'data/test_stats.npz',assets/'data/test_scaffolds_stats.npz')
    rows = [orjson.loads(line) for line in (args.output/'decoded.jsonl').read_bytes().splitlines()]
    raw = [r['smiles'] or '' for r in rows]
    if len(raw) != args.samples:
        raise ValueError('Generation count differs from the declared denominator')
    torch.set_num_threads(args.workers)
    valid = api.remove_invalid(raw,canonize=True,n_jobs=args.workers)
    result = {'Validity':float(api.fraction_valid(raw,n_jobs=args.workers))}
    undefined = {}
    if valid:
        result['Unique@10k'] = float(api.fraction_unique(valid,k=10000,n_jobs=args.workers))
        result['Novelty'] = float(api.novelty(valid,references.train,n_jobs=args.workers))
    else:
        for key in ('Unique@10k','Novelty'):
            result[key] = None
            undefined[key] = 'no valid molecules'
    if len(valid) >= 2:
        fcd = api.fcd_class(device=args.device,n_jobs=1,batch_size=512,
            model_path=str(assets/'chemnet/ChemNet_v0.13_pretrained.pt'))
        # This is the official get_all_metrics FCD call, caching the identical
        # generated sufficient statistics for its two reference comparisons.
        generated_stats = fcd.precalc(valid)
        for key, reference in (('FCD/Test',stats.test),('FCD/TestSF',stats.test_scaffolds)):
            result[key] = float(fcd(pref=reference['FCD'],pgen=generated_stats))
    else:
        for key in ('FCD/Test','FCD/TestSF'):
            result[key] = None
            undefined[key] = 'fewer than two valid molecules; covariance undefined'
    for key,value in result.items():
        if value is not None and not math.isfinite(value):
            raise FloatingPointError(f'Nonfinite official metric: {key}={value}')
    write_json(args.output/'metrics.json',dict(metrics=result,undefined=undefined,raw_samples=len(raw),
        valid_samples=len(valid),official_api=api.provenance,
        metric_scope='five requested metrics using official MOSES functions; other official metrics not computed',
        generation=json.loads((args.output/'generation_complete.json').read_text())))
    print(json.dumps(dict(event='MOSES_METRICS_COMPLETE',metrics=result,undefined=undefined)),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['generate','score'],required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--variant',choices=['graph','native'],default='graph')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--samples',type=int,default=30000)
    p.add_argument('--batch-size',type=int,default=20000)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cuda')
    args = p.parse_args()
    require_compute_host('rebuttal MOSES generation/scoring')
    short_tmp = PIPE.parents[2]/'.tmp/rebuttal_e2e3'
    short_tmp.mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR'] = str(short_tmp)
    tempfile.tempdir = str(short_tmp)
    {'generate':generate,'score':score}[args.mode](args)


if __name__ == '__main__':
    main()
