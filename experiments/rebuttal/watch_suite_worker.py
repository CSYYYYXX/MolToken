#!/usr/bin/env python3
"""Read-only experiment monitor: current GPU memory, progress, failures, results."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PIPE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PIPE))
from experiments.rebuttal.common import write_json
from pipeline.runtime_host import require_compute_host


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    require_compute_host('rebuttal progress monitor')
    root=args.run
    telemetry=root/'logs/monitor.jsonl'
    while True:
        now=datetime.now(timezone.utc).isoformat()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).strip()
        used,total,utilization=map(int,gpu.split(','))
        models={}
        for directory in sorted((root/'models').glob('*')):
            if directory.name.startswith('smoke_'):
                continue
            state=read(directory/'complete.json') or read(directory/'progress.json')
            if state:
                models[directory.name]=dict(state,complete=(directory/'complete.json').exists())
        metrics={d.name:read(d/'metrics.json')['metrics'] for d in sorted((root/'evaluation').glob('*'))
                 if (d/'metrics.json').exists()}
        failures={p.name:p.read_text() for p in (root/'logs').glob('*.failed')}
        if (root/'gpu_suite_failed.json').exists():
            failures['gpu_suite']=read(root/'gpu_suite_failed.json')
        streams={d.name:read(d/'complete.json') or read(d/'progress.json') for d in (root/'streams').glob('*')}
        status=dict(utc=now,gpu=dict(memory_used_mib=used,memory_total_mib=total,
            memory_fraction=used/total,compute_utilization_percent=utilization),
            controller=read(root/'gpu_suite_status.json'),models=models,metrics=metrics,
            streams=streams,failures=failures,complete=(root/'gpu_suite_complete.json').exists())
        write_json(root/'live_status.json',status)
        with telemetry.open('a') as out:
            out.write(json.dumps(dict(utc=now,gpu=status['gpu'],controller=status['controller'],failures=failures))+'\n')
        lines=['# E2/E3 实时记录','',f'更新（UTC）：{now}',
            f'GPU 当前显存：{used:,}/{total:,} MiB（{used/total:.1%}）；计算利用率 {utilization}%。',
            '',f'调度状态：`{(status["controller"] or {}).get("stage","starting")}`','',
            '| 模型／阶段 | 状态 | Epoch | 验证 loss（最佳） | 该阶段输入 token |',
            '|---|---|---:|---:|---:|']
        for name,state in models.items():
            lines.append(f'| {name} | {"完成" if state["complete"] else "运行中"} | {state.get("epoch",0)} | {state.get("best_validation")} | {state.get("input_tokens",0):,} |')
        if metrics:
            lines+=['','| 模型 | Validity | Unique@10k | Novelty | FCD/Test | FCD/TestSF |','|---|---:|---:|---:|---:|---:|']
            for name,values in metrics.items():
                lines.append('| '+name+' | '+' | '.join(str(values.get(k)) for k in ('Validity','Unique@10k','Novelty','FCD/Test','FCD/TestSF'))+' |')
        else:
            lines+=['','正式 30,000 条生成评测尚未完成；训练 loss 不替代 MOSES 结果。']
        if failures:
            lines+=['','## 错误','',json.dumps(failures,ensure_ascii=False,indent=2)]
        lines+=['','E2a 已完成，见 `tokenizers/e2a_results.json`；源实体聚类置信区间和 A/C 长度见 `initial_results.json`。',
            '监控只读取状态并写报告，不停止、重启或释放 worker。']
        temporary=root/'LIVE_STATUS.md.tmp'
        temporary.write_text('\n'.join(lines)+'\n')
        temporary.replace(root/'LIVE_STATUS.md')
        if status['complete']:
            print('MONITOR_STUDY_COMPLETE',flush=True)
            return
        time.sleep(30)


if __name__=='__main__':
    main()
