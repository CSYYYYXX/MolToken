"""Exercise production CLIs with synthetic data, one CPU and tiny models."""
import json
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from pipeline.runtime_host import require_compute_host
require_compute_host('synthetic integration test')
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--work-dir', type=Path, required=True)
DEST = parser.parse_args().work_dir
DEST.mkdir(exist_ok=False)
WORK = DEST
os.chdir(ROOT)
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
           TMPDIR=os.environ.get('TMPDIR', __import__('tempfile').gettempdir()), PYTHONDONTWRITEBYTECODE='1', MOLTOKEN_REQUIRE_WORKER=os.environ.get('MOLTOKEN_REQUIRE_WORKER', '0'))
env.pop('PYTHONPATH', None)
results = []
def run(name, module, args):
    command = [sys.executable, '-m', module, *[str(a) for a in args]]
    print('RUN', name, flush=True)
    with (WORK / f'train_test_{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    results.append({'step': name, 'passed': True})
    (WORK / 'training_validation.json').write_text(json.dumps(results, indent=2) + '\n')

run('encode', 'moltoken', ['encode-corpus', '--input', ROOT/'examples/toy_corpus.jsonl', '--output', DEST/'toy.bin'])
run('index', 'cli.data.build_compact_int_index', ['--input-shards', DEST/'toy.bin', '--index-dir', DEST/'index'])
run('pretrain', 'cli.pretrain.train_qwen3_pretrain', [
    '--token-files', DEST/'toy.bin', '--compact-index-dir', DEST/'index',
    '--out-dir', DEST/'pretrain', '--vocab-size', '17207', '--dataset-mode', 'online-pad',
    '--model-implementation', 'hf', '--online-pad-length', '512', '--online-truncate-long-samples',
    '--hidden-size', '32', '--intermediate-size', '64', '--num-hidden-layers', '1',
    '--num-attention-heads', '2', '--num-key-value-heads', '1', '--head-dim', '16',
    '--batch-size', '2', '--grad-accum', '2', '--max-steps', '2', '--num-workers', '0',
    '--eval-every', '0', '--save-every', '0', '--final-eval', '--val-fraction', '0.2',
    '--no-bf16', '--no-gradient-checkpointing'])
run('loss', 'moltoken', ['evaluate-loss', '--checkpoint', DEST/'pretrain/final',
    '--token-files', DEST/'toy.bin', '--index-dir', DEST/'index', '--output', DEST/'test_loss.json',
    '--batch-size', '2', '--truncate'])

# A small, local text encoder has the same 768-dimensional interface as MPNet.
encoder = DEST/'text_encoder'
encoder.mkdir()
(encoder/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngenerate\na\ndna\nsequence\n')
BertTokenizerFast(vocab_file=str(encoder/'vocab.txt')).save_pretrained(encoder)
BertModel(BertConfig(vocab_size=9, hidden_size=768, num_hidden_layers=1,
                     num_attention_heads=12, intermediate_size=64)).save_pretrained(encoder)
task = 'dna_diffusion_celltype_generation'
sources = DEST/'sft_sources'
sources.mkdir()
for split, n in [('train', 5), ('validation', 2), ('test', 2)]:
    rows = [{'id': f'{split}-{i}', 'task': task, 'target_type': 'dna',
             'condition': {'type': 'text', 'text': 'generate a dna sequence', 'molecule_bpe_ids': []},
             'target_bpe_ids': [91,1968], 'meta': {'split': split}}
            for i in range(n)]
    (sources/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
run('sft_materialize', 'cli.data.materialize_encoded_biological_sft_root', [
    '--task', task, '--dataset-id', 'dna_diffusion_official_fourlabel_v1',
    '--train', sources/'train.jsonl', '--validation', sources/'validation.jsonl',
    '--test', sources/'test.jsonl', '--output-root', DEST/'sft_data'])
common = ['--data-root', DEST/'sft_data', '--pretrained-ckpt', DEST/'pretrain/final',
    '--text-encoder', encoder, '--task', task, '--lineage-id', task,
    '--sampling-protocol', 'single-task-repeated-stage-v1', '--epochs', '100',
    '--max-steps', '0', '--steps-per-epoch', '0', '--eval-batches', '0', '--save-steps', '0',
    '--max-context', '512', '--strict-no-runtime-truncation', '--molecule-condition-mode', 'direct_qwen',
    '--batch-size', '2', '--grad-accumulation-steps', '2', '--early-stopping-patience', '1',
    '--early-stopping-min-epochs', '2', '--early-stopping-min-delta', '1000000']
run('alignment', 'cli.sft.train_axis_b_repeated_stage_sft', [*common, '--stage', 'alignment', '--output-dir', DEST/'sft/alignment'])
run('full', 'cli.sft.train_axis_b_repeated_stage_sft', [*common, '--stage', 'full', '--resume-dir', DEST/'sft/alignment/best', '--output-dir', DEST/'sft/full'])
run('sample', 'cli.evaluate.sample_axis_b_sft', ['--checkpoint', DEST/'sft/full/best',
    '--data-root', DEST/'sft_data', '--text-encoder', encoder, '--task', task,
    '--split', 'test', '--n-samples', '2', '--batch-size', '2', '--max-new-tokens', '3',
    '--dtype', 'fp32', '--greedy', '--output', DEST/'samples.jsonl'])
print('SYNTHETIC_TRAINING_VALIDATION_COMPLETE', flush=True)
