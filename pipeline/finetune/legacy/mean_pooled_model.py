"""Inference-compatible format-v1 model with one soft token per condition.

The v1 checkpoint declares attention-mask mean pooling BEFORE the MLP.
It must not be interpreted as the newer per-position, norm-calibrated model.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pipeline.finetune.axis_b_model import AxisBConditionalModel, masked_mean_pool


class _LegacyProjection(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, values):
        return self.net(values)


class MeanPooledAxisBModel(AxisBConditionalModel):
    def __init__(self, *, legacy_config, **kwargs):
        super().__init__(**kwargs)
        self.legacy_config = dict(legacy_config)
        self.text_projection = _LegacyProjection(self.text_projection.net)
        self.molecule_projection = _LegacyProjection(self.molecule_projection.net)

    def _project_condition_sequence(self, hidden_states, attention_mask,
                                    present, projection, *, normalize):
        values = masked_mean_pool(hidden_states, attention_mask)
        if normalize:
            values = F.normalize(values, p=2, dim=-1)
        projected = projection(values.to(projection.net[0].weight.dtype)).unsqueeze(1)
        projected = projected * present[:, None, None].to(projected.dtype)
        return projected, present.to(torch.long)

    def save_axis_b_pretrained(self, output_dir, *, metadata=None):
        from safetensors.torch import save_file
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.qwen.save_pretrained(output_dir/'qwen3', safe_serialization=True)
        state = {f'{name}.{key}': value.detach().cpu().contiguous()
                 for name in ('text_projection', 'molecule_projection')
                 for key, value in getattr(self, name).state_dict().items()}
        save_file(state, output_dir/'condition_projectors.safetensors')
        cfg = dict(self.legacy_config)
        cfg['stage'] = self.stage
        if metadata is not None:
            cfg['metadata'] = metadata
        (output_dir/'axis_b_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
