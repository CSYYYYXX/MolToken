#!/usr/bin/env python3
"""Publish per-seed results and three-seed aggregates when the study finishes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import sys
import time

PIPE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PIPE))
from experiments.rebuttal.common import write_json
from pipeline.runtime_host import require_compute_host


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    require_compute_host('rebuttal result aggregation')
    root=args.run
    while not (root/'gpu_suite_complete.json').exists():
        if (root/'gpu_suite_failed.json').exists():
            raise RuntimeError('GPU suite failed; inspect gpu_suite_failed.json and live status')
        time.sleep(30)
    seeds=json.loads((root/'gpu_suite_complete.json').read_text())['seeds']
    keys=('Validity','Unique@10k','Novelty','FCD/Test','FCD/TestSF')
    groups={}
    for method in ('graph_random','graph_small','graph_joint','native_joint'):
        rows=[]
        for seed in seeds:
            name=f'{method}_sft_{seed}'
            result=json.loads((root/'evaluation'/name/'metrics.json').read_text())
            sft=json.loads((root/'models'/name/'complete.json').read_text())
            generation=result['generation']
            if result['raw_samples']!=30000 or not generation['formal'] or generation['seed']!=seed:
                raise ValueError(f'Formal generation contract differs: {name}')
            if generation['checkpoint_provenance']['best_epoch']!=sft['best_epoch']:
                raise ValueError(f'Generation did not use the selected validation checkpoint: {name}')
            pretrain=None if method=='graph_random' else json.loads((root/'models'/f'{method}_pretrain_{seed}'/'complete.json').read_text())
            rows.append(dict(seed=seed,metrics=result['metrics'],undefined=result['undefined'],sft=sft,
                             pretraining=pretrain,generation_seconds=generation['seconds']))
        aggregate={}
        for key in keys:
            values=[row['metrics'][key] for row in rows]
            if any(value is None for value in values):
                aggregate[key]=dict(mean=None,std=None,reason='undefined in at least one seed; no seed dropped')
            else:
                aggregate[key]=dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else None)
        groups[method]=dict(seeds=rows,aggregate=aggregate)
    payload=dict(completed_utc=datetime.now(timezone.utc).isoformat(),groups=groups,
        e2=json.loads((root/'initial_results.json').read_text()),
        limitations=['Downstream conclusion restricted to MOSES distribution learning',
            'Three model initialization/SFT shuffle seeds; one tokenizer fit and one raw pretraining order',
            'Known biological accession aliases and exact sequences excluded; no full homology clustering',
            'Fixed common SFT learning rate; no hyperparameter sweep',
            'Graph roundtrip does not establish stereochemical identity'])
    write_json(root/'final_results.json',payload)
    lines=['# MolToken E2/E3 最终结果','',f'完成时间（UTC）：{payload["completed_utc"]}',
        '',f'每组 {len(seeds)} 个模型种子；每个模型 30,000 条原始生成。数值为均值 ± 样本标准差。',
        '','| 起点 | Validity | Unique@10k | Novelty | FCD/Test | FCD/TestSF |',
        '|---|---:|---:|---:|---:|---:|']
    for method,group in groups.items():
        cells=[]
        for key in keys:
            item=group['aggregate'][key]
            cells.append('未定义' if item['mean'] is None else f'{item["mean"]:.5f} ± {item["std"]:.5f}')
        lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines+=['','FCD 越低越好；其余指标越高越好。未定义值保留在逐种子结果中，没有丢掉失败种子。',
        '','E2a 的频率排序缩短比例（正数表示更短）：', '']
    for domain,row in payload['e2']['e2a']['domains'].items():
        lines.append(f'- {domain}: {row["reduction"]:+.2%}')
    lines+=['','完整逐种子结果、预训练预算／数据接触量、SFT 最佳及停止 epoch 见 `final_results.json`。']
    (root/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('FINAL_RESULTS_PUBLISHED',flush=True)


if __name__=='__main__':
    main()
