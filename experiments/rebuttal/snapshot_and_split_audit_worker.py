#!/usr/bin/env python3
"""Record current code and verify MOSES target-level disjointness, on a worker."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

PIPE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PIPE))
from pipeline.runtime_host import require_compute_host
from experiments.rebuttal.common import write_json


def main():
    import orjson
    from rdkit import Chem
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    require_compute_host('rebuttal provenance and split audit')
    root=args.run
    snapshot=root/'provenance/code'
    snapshot.mkdir(parents=True,exist_ok=False)
    files=list((PIPE/'experiments/rebuttal').glob('*.py'))+list((PIPE/'experiments/rebuttal').glob('*.sh'))
    files += [PIPE/x for x in ('cli/pretrain/train_qwen3_pretrain.py', 'pipeline/serialization/_stackk_cpp.cpp',
        'pipeline/serialization/fddfs.py','pipeline/serialization/freq_dfs_core.py','pipeline/serialization/stackk_fast.py',
        'pipeline/graph/chem_graph_builder.py','pipeline/graph/io.py','pipeline/finetune/fddfs_encode.py',
        'cli/evaluate/eval_axis_b_samples.py','pipeline/graph/aromatic_n_repair.py','cli/evaluate/evaluate_axis_b_moses_official.py')]
    manifest=[]
    for path in files:
        content=path.read_bytes()
        if path.suffix=='.py':
            ast.parse(content,filename=str(path))
        destination=snapshot/path.relative_to(PIPE)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
        manifest.append(dict(source=str(path),snapshot=str(destination),sha256=hashlib.sha256(content).hexdigest()))
    write_json(root/'provenance/source_snapshot.json',dict(files=manifest,hashes_used_as_gate=False))
    contents={}
    counts={}
    for split in ('train','validation'):
        keys=set()
        count=0
        with (root/'manifests'/f'moses_{split}.jsonl').open('rb') as handle:
            for line in handle:
                row=orjson.loads(line)
                keys.add(row['normalized'])
                count+=1
        contents[split]=keys
        counts[split]=dict(records=count,unique_targets=len(keys))
    refs=root.parent.parent/'data/benchmark/raw/moses/official_core13_dd7ed6'
    for split in ('test','test_scaffolds'):
        keys=set()
        with (refs/f'{split}.csv').open() as handle:
            for row in csv.DictReader(handle):
                mol=Chem.MolFromSmiles(row['SMILES'])
                if mol is None:
                    raise ValueError('Invalid official reference')
                keys.add(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True))
        contents[split]=keys
    overlaps={split:len(contents['train']&contents[split]) for split in ('validation','test','test_scaffolds')}
    if any(overlaps.values()):
        raise ValueError(f'MOSES canonical target overlap: {overlaps}')
    write_json(root/'preflight/sft_target_disjointness.json',dict(counts=counts,training_overlap=overlaps))
    print(json.dumps(dict(event='CODE_SNAPSHOT_AND_TARGET_SPLIT_AUDIT_PASS',source_files=len(files),counts=counts,overlaps=overlaps)),flush=True)


if __name__=='__main__':
    main()
