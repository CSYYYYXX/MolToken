#!/usr/bin/env python3
"""Run the authorized transfer matrix on one GPU with measured memory reservations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from experiments.rebuttal.common import write_json
from pipeline.runtime_host import require_compute_host


def aggregate(root):
    import numpy as np
    rows = [json.loads(p.read_text()) for p in sorted((root / 'models').glob('*/result.json'))]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['task'], row['arm'])].append(row)
    summary = []
    for (task, arm), results in sorted(grouped.items()):
        metrics = {}
        for name in results[0]['metrics']:
            values = [r['metrics'][name] for r in results]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                metrics[name] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        summary.append(dict(task=task, arm=arm, seeds=[r['seed'] for r in results], metrics=metrics))
    write_json(root / 'results_summary.json', dict(completed=len(rows), total=36, aggregate=summary, results=rows))
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--old-run', type=Path, required=True)
    args = p.parse_args()
    require_compute_host('transfer GPU suite')
    root = args.run
    lock = (root / 'runtime' / 'suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profile = json.loads((root / 'runtime' / 'gpu_profile.json').read_text())
    total = profile['total_mib']
    jobs = []
    for seed in (42, 43, 44):
        for task in ('gb1', 'promoter', 'mrl'):
            for arm in ('graph_joint', 'native_joint', 'graph_small', 'graph_random'):
                name = f'{task}_{arm}_{seed}'
                rep = 'native' if arm == 'native_joint' else 'graph'
                capacity = profile['configurations'][f'{task}:{rep}']
                job = dict(name=name, task=task, arm=arm, seed=seed, **capacity)
                directory = root / 'models' / name
                if directory.exists():
                    raise ValueError(f'Refusing duplicate or implicit restart: {directory}')
                jobs.append(job)
    write_json(root / 'queue.json', dict(host=socket.gethostname(), total=36, jobs=jobs, profile=profile,
        memory_policy='Measured real training memory; reserve before launching concurrent independent runs; no dummy allocations'))
    running, finished, failed = [], [], []
    last_sample = 0.
    while jobs or running:
        for item in list(running):
            code = item['process'].poll()
            if code is None:
                continue
            running.remove(item)
            item['log_handle'].close()
            job = item['job']
            if code != 0 or not (root / 'models' / job['name'] / 'result.json').exists():
                failure = dict(**job, returncode=code, log=str(root / 'logs' / f'{job["name"]}.log'))
                failed.append(failure)
                print('TRANSFER_JOB_FAILED', json.dumps(failure), flush=True)
            else:
                finished.append(job['name'])
                print('TRANSFER_JOB_COMPLETE', job['name'], flush=True)
            aggregate(root)
        reserved = sum(i['job']['estimated_process_mib'] for i in running)
        # Fill toward 70%; a whole new job may cross 70%, but never reserve >84%.
        while jobs and len(running) < 12 and (not running or reserved < total*.70):
            job = jobs[0]
            if running and reserved + job['estimated_process_mib'] > total*.84:
                break
            jobs.pop(0)
            logfile = (root / 'logs' / f'{job["name"]}.log').open('x')
            command = [sys.executable, '-u', str(PIPE / 'experiments/rebuttal/transfer_train_worker.py'),
                       '--run', str(root), '--old-run', str(args.old_run), '--task', job['task'], '--arm', job['arm'],
                       '--seed', str(job['seed']), '--batch-size', str(job['batch_size']),
                       '--micro-tokens', str(job['micro_tokens'])]
            process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running.append(dict(job=job, process=process, log_handle=logfile))
            reserved += job['estimated_process_mib']
            print('TRANSFER_JOB_STARTED', json.dumps(dict(**job, pid=process.pid)), flush=True)
        now = time.monotonic()
        if now - last_sample >= 30 or (not running and not jobs):
            # memory.used is the actual current GPU occupancy, not utilization or an old peak.
            native = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                              '--format=csv,noheader,nounits'], text=True).strip()
            used, available, util = [float(x.strip()) for x in native.split(',')]
            active = []
            for item in running:
                progress = root / 'models' / item['job']['name'] / 'progress.json'
                stage = json.loads(progress.read_text()).get('event') if progress.exists() else 'loading'
                active.append(dict(name=item['job']['name'], pid=item['process'].pid, stage=stage))
            snapshot = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=socket.gethostname(),
                            running=active, completed=finished, failed=failed, queued=len(jobs),
                            memory_used_mib=used, memory_total_mib=available, memory_fraction=used/available,
                            compute_utilization=util, planned_reserved_mib=reserved,
                            transition_or_loading=any(i['stage'] in ('loading', 'alignment_features', 'alignment_epoch', 'test_evaluation') for i in active))
            write_json(root / 'live_status.json', snapshot)
            with (root / 'logs' / 'gpu_memory.jsonl').open('a') as f:
                f.write(json.dumps(snapshot) + '\n')
            print('TRANSFER_SUITE_STATUS', json.dumps(snapshot), flush=True)
            last_sample = now
        if running or jobs:
            time.sleep(5)
    count = aggregate(root)
    write_json(root / 'suite_complete.json', dict(completed=count, total=36, failed=failed, success=count == 36 and not failed))
    if failed or count != 36:
        raise RuntimeError(f'Transfer suite incomplete: completed={count}/36, failed={len(failed)}; see logs')
    print('TRANSFER_SUITE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
