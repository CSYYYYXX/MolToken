#!/usr/bin/env python3
"""Serial single-GPU study; CPU scoring overlaps the next authorized training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PIPE))
from experiments.rebuttal.common import write_json
from pipeline.runtime_host import require_compute_host


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=p.parse_args()
    require_compute_host('rebuttal GPU study controller')
    root=args.run
    pending=[]
    def emit(stage,**values):
        row=dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,**values)
        write_json(root/'gpu_suite_status.json',row)
        print(json.dumps(row),flush=True)
    def check_scores(wait=False):
        for name,process,log in pending:
            code=process.wait() if wait else process.poll()
            if code is not None and code != 0:
                raise RuntimeError(f'MOSES scoring failed: {name}, exit={code}, log={log}')
    def run(name,script,arguments):
        command=[sys.executable,'-u',str(PIPE/'experiments/rebuttal'/script),'--run',str(root),*map(str,arguments)]
        logfile=root/'logs'/f'{name}.log'
        emit(name,command=command,log=str(logfile))
        with logfile.open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(10)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode,command)
        check_scores()
    def train(name,mode,variant,seed,scope='joint',init=None):
        flags=['--name',name,'--mode',mode,'--variant',variant,'--seed',seed,'--scope',scope,'--data-seed',42]
        if init:
            flags+=['--init',init]
        run(name,'train_worker.py',flags)
        if not (root/'models'/name/'complete.json').exists():
            raise RuntimeError(f'Training did not publish completion: {name}')
    def evaluate(name,variant,seed):
        output=root/'evaluation'/name
        run(name+'_generate','evaluate_worker.py',['--mode','generate','--checkpoint',root/'models'/name/'best.pt',
            '--variant',variant,'--seed',seed,'--output',output,'--batch-size',20000])
        command=[sys.executable,'-u',str(PIPE/'experiments/rebuttal/evaluate_worker.py'),'--run',str(root),
            '--mode','score','--output',str(output),'--device','cpu','--workers','8']
        logfile=root/'logs'/f'{name}_score.log'
        with logfile.open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        pending.append((name,process,logfile))
        emit('scoring_started',name=name,pid=process.pid,log=str(logfile))
    try:
        # The first random control was deliberately started during raw-pool setup.
        emit('waiting_for_first_random_control')
        while not (root/'models/graph_random_sft_42/complete.json').exists():
            failure=root/'logs/random_sft.failed'
            if failure.exists():
                raise RuntimeError(failure.read_text())
            time.sleep(10)
        for seed in args.seeds:
            random_name=f'graph_random_sft_{seed}'
            if seed != 42:
                train(random_name,'sft','graph',seed)
            evaluate(random_name,'graph',seed)
            for variant,scope in (('graph','joint'),('native','joint'),('graph','small')):
                pretrain=f'{variant}_{scope}_pretrain_{seed}'
                train(pretrain,'pretrain',variant,seed,scope)
                sft=f'{variant}_{scope}_sft_{seed}'
                train(sft,'sft',variant,seed,scope,root/'models'/pretrain/'final.pt')
                evaluate(sft,variant,seed)
            # The full first-seed pipeline must finish before repetitions.
            check_scores(wait=True)
        emit('complete',seeds=args.seeds)
        write_json(root/'gpu_suite_complete.json',dict(seeds=args.seeds,models=12,pretraining_runs=9))
    except BaseException as error:
        emit('failed',error=type(error).__name__+': '+str(error))
        write_json(root/'gpu_suite_failed.json',dict(error=type(error).__name__+': '+str(error)))
        raise


if __name__=='__main__':
    main()
