#!/usr/bin/env python3
"""Exercise selected official metrics on an explicitly synthetic fixture."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PIPE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PIPE))
sys.path.insert(0,str(PIPE/'ops/environment'))
from pipeline.runtime_host import require_compute_host
from experiments.rebuttal.common import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    require_compute_host('official metric subset integration test')
    from smoke_moses_existing_source_worker import SMILES
    output=args.run/'preflight/metric_subset_control'
    output.mkdir(exist_ok=False)
    with (output/'decoded.jsonl').open('w') as handle:
        for value in SMILES:
            handle.write(json.dumps(dict(smiles=value))+'\n')
    write_json(output/'generation_complete.json',dict(samples=len(SMILES),formal=False,synthetic_control_only=True))
    subprocess.run([sys.executable,'-u',str(PIPE/'experiments/rebuttal/evaluate_worker.py'),'--run',str(args.run),
        '--mode','score','--output',str(output),'--device','cpu','--workers','4','--samples',str(len(SMILES))],check=True)
    result=json.loads((output/'metrics.json').read_text())
    if result['raw_samples']!=len(SMILES) or result['metrics']['Validity']!=1 or result['undefined']:
        raise AssertionError('Official metric subset failed the valid control')
    print('OFFICIAL_METRIC_SUBSET_PASS',flush=True)


if __name__=='__main__':
    main()
