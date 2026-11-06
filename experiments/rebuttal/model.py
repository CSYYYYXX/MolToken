"""Shared Qwen3 backbone and external domain-prefix vectors for E2/E3."""
from __future__ import annotations
import torch
from transformers import Qwen3Config
from cli.pretrain.train_qwen3_pretrain import PackedXformersQwen3ForCausalLM
from experiments.rebuttal.common import CONTENT_VOCAB, DOMAINS

VOCAB_SIZE = CONTENT_VOCAB + 3


class DomainEmbedding(torch.nn.Module):
    def __init__(self, token_embedding):
        super().__init__()
        self.tokens = token_embedding
        self.domain = torch.nn.Embedding(len(DOMAINS), token_embedding.embedding_dim)
        torch.nn.init.normal_(self.domain.weight, std=.02)

    @property
    def weight(self):
        return self.tokens.weight

    def forward(self, ids):
        is_domain = ids >= VOCAB_SIZE
        tokens = self.tokens(ids.clamp_max(VOCAB_SIZE - 1))
        prefix = self.domain((ids - VOCAB_SIZE).clamp(0, len(DOMAINS) - 1))
        return torch.where(is_domain.unsqueeze(-1), prefix, tokens)


def build_model(seed=42):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=VOCAB_SIZE, hidden_size=384, intermediate_size=1152,
        num_hidden_layers=20, num_attention_heads=3, num_key_value_heads=1, head_dim=128,
        hidden_act='silu', max_position_embeddings=512, initializer_range=.02,
        rms_norm_eps=1e-6, use_cache=False, tie_word_embeddings=True, rope_theta=1e6,
        attention_dropout=0., attention_bias=False, use_sliding_window=False,
        sliding_window=None, max_window_layers=20, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = PackedXformersQwen3ForCausalLM(config)
    model.embed_tokens = DomainEmbedding(model.embed_tokens)
    return model


def collate(records, device='cuda'):
    # A record is (domain_index, raw BPE IDs). No truncation is permitted.
    flat, labels, lengths = [], [], []
    for domain, body in records:
        if not 0 <= int(domain) < len(DOMAINS):
            raise ValueError(f'Invalid domain index: {domain}')
        if not 0 < len(body) <= 509:
            raise ValueError(f'Content outside complete context support: {len(body)}')
        values = [int(i) + 3 for i in body]
        if min(values) < 3 or max(values) >= VOCAB_SIZE:
            raise ValueError('Content token outside frozen model vocabulary')
        flat.extend([VOCAB_SIZE + int(domain), 1, *values, 2])
        labels.extend([-100, -100, *values, 2])
        lengths.append(len(values) + 3)
    return dict(input_ids=torch.tensor([flat], device=device, dtype=torch.long),
        labels=torch.tensor([labels], device=device, dtype=torch.long),
        segment_lengths=torch.tensor(lengths, device=device, dtype=torch.long))


def hf_generation_model(model):
    from transformers import Qwen3ForCausalLM
    config = Qwen3Config.from_dict(model.config.to_dict())
    config.use_cache = True
    config._attn_implementation = 'sdpa'
    hf = Qwen3ForCausalLM(config)
    state = {}
    for key, value in model.state_dict().items():
        if key == 'embed_tokens.domain.weight':
            continue
        if key == 'embed_tokens.tokens.weight':
            key = 'embed_tokens.weight'
        state[key if key.startswith('lm_head.') else 'model.' + key] = value
    hf.load_state_dict(state, strict=True)
    return hf
