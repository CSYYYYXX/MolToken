#!/usr/bin/env python3
"""Lightweight jump-host watcher for the user's existing tmux79 request only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def command(argv):
    environment = None
    if argv[0] == 'brainctl':
        # New tmux sessions can inherit an external proxy without NO_PROXY.
        # Query the internal cluster API directly for this subprocess only.
        environment = {k: v for k, v in os.environ.items()
                       if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    return subprocess.run(argv, capture_output=True, text=True, timeout=20, env=environment)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True, help='Jump-host mounted run path')
    p.add_argument('--worker-run', required=True)
    p.add_argument('--process', required=True)
    p.add_argument('--namespace', default='ailab-evoagi')
    p.add_argument('--pane', default='%80')
    p.add_argument('--launcher', type=Path, required=True)
    args = p.parse_args()
    lock = (args.run / 'runtime' / 'allocation_watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dispatched = args.run / 'runtime' / 'dispatched.json'
    if dispatched.exists():
        raise RuntimeError('A dispatch record already exists; refusing automatic duplicate launch')
    if not (args.run / 'runtime' / 'cpu_validation.json').exists():
        raise RuntimeError('Worker-side CPU validation must pass before arming dispatch')
    if json.loads((args.run / 'runtime' / 'cpu_validation.json').read_text())['status'] != 'passed':
        raise RuntimeError('Worker-side CPU validation failed')
    query_failures = 0
    while True:
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        try:
            result = command(['brainctl', 'get', 'process', args.process, '-n', args.namespace,
                              '-o', 'jsonpath={.status.phase}{"\\n"}{.status.conditions[*].message}'])
        except subprocess.TimeoutExpired as error:
            query_failures += 1
            write(args.run / 'allocation_status.json', dict(time_utc=now, status='query_timeout', failures=query_failures, error=str(error)))
            print('ALLOCATION_QUERY_TIMEOUT', query_failures, str(error), flush=True)
            if query_failures >= 3:
                raise
            time.sleep(30)
            continue
        if result.returncode:
            write(args.run / 'allocation_status.json', dict(time_utc=now, status='query_error', error=result.stderr[-2000:]))
            if 'NotFound' in result.stderr or 'not found' in result.stderr:
                raise RuntimeError(f'Allocation no longer exists: {args.process}: {result.stderr}')
            print('ALLOCATION_QUERY_ERROR', result.stderr[-1000:], flush=True)
            query_failures += 1
            if query_failures >= 3:
                raise RuntimeError(f'Repeated cluster query failure: {result.stderr}')
            time.sleep(30)
            continue
        query_failures = 0
        fields = result.stdout.strip().splitlines()
        phase = fields[0] if fields else ''
        state = dict(time_utc=now, process=args.process, pane=args.pane, phase=phase,
                     scheduler_message='\n'.join(fields[1:]), dispatched=dispatched.exists())
        pane = command(['tmux', 'capture-pane', '-pt', args.pane, '-S', '-15'])
        if pane.returncode:
            raise RuntimeError(f'Tmux pane disappeared: {pane.stderr}')
        lines = [line.strip() for line in pane.stdout.splitlines() if line.strip()]
        last = lines[-1] if lines else ''
        worker_prompt = re.search(r'@([^ :]*worker[^ :]*):.*[#$]\s*$', last)
        if phase in ('Failed', 'Succeeded', 'Deleted', 'Stopped', 'Completed'):
            write(args.run / 'allocation_status.json', dict(**state, status='allocation_ended'))
            raise RuntimeError(f'User allocation ended: {state}')
        if not dispatched.exists() and phase == 'Running' and worker_prompt:
            exit_path = args.worker_run + '/runtime/bootstrap_exit.txt'
            body = f'bash {shlex.quote(str(args.launcher))}; rc=$?; printf "%s\\n" "$rc" > {shlex.quote(exit_path)}; exit "$rc"'
            payload = f'nohup bash -c {shlex.quote(body)} > {shlex.quote(args.worker_run + "/logs/gpu_bootstrap.log")} 2>&1 < /dev/null &'
            # The server state AND the current worker prompt must agree before sending shell work.
            sent = command(['tmux', 'send-keys', '-t', args.pane, '-l', payload])
            if sent.returncode:
                raise RuntimeError(f'Dispatch failed: {sent.stderr}')
            entered = command(['tmux', 'send-keys', '-t', args.pane, 'Enter'])
            if entered.returncode:
                raise RuntimeError(f'Dispatch Enter failed: {entered.stderr}')
            write(dispatched, dict(time_utc=now, process=args.process, pane=args.pane, host=worker_prompt.group(1), command=payload))
            state['dispatched'] = True
            print('TRANSFER_DISPATCHED', json.dumps(state), flush=True)
        if dispatched.exists():
            exit_file = args.run / 'runtime' / 'bootstrap_exit.txt'
            if exit_file.exists() and int(exit_file.read_text().strip()) != 0:
                write(args.run / 'allocation_status.json', dict(**state, status='worker_failed', returncode=int(exit_file.read_text())))
                raise RuntimeError('Worker bootstrap/suite failed; see logs/gpu_bootstrap.log and per-run logs')
            if (args.run / 'live_status.json').exists():
                write(args.run / 'allocation_status.json', dict(**state, status='gpu_suite_running'))
                print('TRANSFER_SUITE_RUNNING_CONFIRMED', flush=True)
                return
            state['status'] = 'worker_bootstrap_or_profiling'
        else:
            state['status'] = 'waiting_for_allocation'
        write(args.run / 'allocation_status.json', state)
        print('ALLOCATION_STATUS', json.dumps(state), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
