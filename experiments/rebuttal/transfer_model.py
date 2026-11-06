"""Record-level prediction on the unchanged packed E2/E3 backbone."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import xformers.ops as xops
from cli.pretrain.train_qwen3_pretrain import build_packed_position_ids
from experiments.rebuttal.common import DOMAINS
from experiments.rebuttal.model import build_model, VOCAB_SIZE


class TransferData:
    def __init__(self, root, task, representation):
        directory = Path(root) / 'data' / task
        self.metadata = json.loads((directory / 'complete.json').read_text())
        self.tokens = np.load(directory / f'{representation}.npy', mmap_mode='r')
        self.offsets = np.load(directory / f'{representation}_offsets.npy', mmap_mode='r')
        self.targets = np.load(directory / 'targets.npy')
        self.splits = np.load(directory / 'splits.npy')
        self.ids = json.loads((directory / 'ids.json').read_text())
        self.lengths = np.diff(self.offsets) + 3
        self.domain = DOMAINS.index(self.metadata['domain'])
        self.classification = self.metadata['problem'] == 'classification'
        if len(self.ids) != len(self.targets) or len(self.offsets) != len(self.targets) + 1:
            raise ValueError('Data index/target length mismatch')
        if self.offsets[-1] != len(self.tokens) or self.lengths.min() < 4 or self.lengths.max() > 512:
            raise ValueError('Invalid complete-context data')
        self.roles = {role: np.flatnonzero(self.splits == i) for i, role in enumerate(('train', 'validation', 'test'))}
        for role, indices in self.roles.items():
            if len(indices) != self.metadata['counts'][role]:
                raise ValueError(f'Split manifest mismatch: {role}')
        y = self.targets[self.roles['train']].astype(np.float64)
        self.mean = 0. if self.classification else float(y.mean())
        self.std = 1. if self.classification else float(y.std())
        if self.std <= 0:
            raise ValueError('Training target variance must be positive')

    def pack(self, indices, device='cuda'):
        lengths = self.lengths[indices]
        flat = np.empty(int(lengths.sum()), dtype=np.int64)
        pos = 0
        for i, length in zip(indices, lengths):
            body = self.tokens[self.offsets[i]:self.offsets[i+1]]
            flat[pos] = VOCAB_SIZE + self.domain
            flat[pos+1] = 1
            flat[pos+2:pos+length-1] = body.astype(np.int64) + 3
            flat[pos+length-1] = 2
            pos += length
        return torch.from_numpy(flat).unsqueeze(0).to(device), torch.as_tensor(lengths.copy(), device=device)

    def y(self, indices, device='cuda'):
        return torch.as_tensor((self.targets[indices] - self.mean) / self.std, device=device)


def microbatches(indices, lengths, max_tokens):
    start, tokens = 0, 0
    for j, i in enumerate(indices):
        n = int(lengths[i])
        if n > max_tokens:
            raise ValueError(f'A complete record ({n}) exceeds microbatch token budget {max_tokens}')
        if tokens and tokens + n > max_tokens:
            yield indices[start:j]
            start, tokens = j, 0
        tokens += n
    if start < len(indices):
        yield indices[start:]


class TransferModel(torch.nn.Module):
    def __init__(self, seed, checkpoint=None):
        super().__init__()
        self.backbone = build_model(seed)
        if checkpoint is not None:
            source = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(source['model'], strict=True)
            del source
        torch.manual_seed(seed + 10000)
        self.head = torch.nn.Linear(self.backbone.config.hidden_size, 1)
        torch.nn.init.normal_(self.head.weight, std=.01)
        torch.nn.init.zeros_(self.head.bias)

    def features(self, ids, lengths):
        positions = build_packed_position_ids(lengths)
        h = self.backbone.embed_tokens(ids)
        rotary = self.backbone.rotary_emb(h, positions)
        bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(lengths.tolist())
        for layer in self.backbone.layers:
            h = layer(h, rotary, bias)
        h = self.backbone.norm(h)
        return h[0, lengths.cumsum(0) - 1]

    def forward(self, ids, lengths):
        return self.head(self.features(ids, lengths).float()).squeeze(-1)


def checkpoint_source(old_run, arm, seed):
    if arm == 'graph_random':
        return None
    return Path(old_run) / 'models' / f'{arm}_pretrain_{seed}' / 'final.pt'


def metrics(targets, predictions, classification):
    from scipy.stats import spearmanr
    from scipy.special import expit
    from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, r2_score
    targets, predictions = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite final predictions')
    if classification:
        probabilities = expit(predictions)
        labels = predictions >= 0.
        return dict(accuracy=float(accuracy_score(targets, labels)), auroc=float(roc_auc_score(targets, probabilities)),
                    mcc=float(matthews_corrcoef(targets, labels)), f1=float(f1_score(targets, labels)))
    nonconstant = np.std(predictions) > 0 and np.std(targets) > 0
    pearson = float(np.corrcoef(targets, predictions)[0, 1]) if nonconstant else None
    return dict(spearman=float(spearmanr(targets, predictions).statistic) if nonconstant else None,
                pearson=pearson, pearson_squared=pearson**2 if pearson is not None else None,
                r2=float(r2_score(targets, predictions)), rmse=float(np.sqrt(np.mean((targets - predictions)**2))),
                correlation_defined=bool(nonconstant))
