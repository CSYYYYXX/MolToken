# Checkpoint formats

## Foundation models

```python
import torch
from moltoken import load_pretrained
model = load_pretrained(
    "checkpoints/moltoken_0.55b", device="cuda", dtype=torch.bfloat16,
)
```

Supported weight containers are single or indexed/sharded `model.safetensors`
and `pytorch_model.bin`, together with `config.json`. Original packed-Qwen keys
`embed_tokens.*`, `layers.*`, `norm.*` are mapped to the standard HF `model.*`
prefix, and tied input/output embeddings are restored. The returned model is
`transformers.Qwen3ForCausalLM`. Architecture dimensions come from the saved
configuration, so the same interface loads both foundation models and smaller
scaling checkpoints. Weight keys and shapes are checked during loading.

Use the included molecular tokenizer, rather than the natural-language Qwen
tokenizer. The following ID convention applies to FDDFS/BPE checkpoints.

## Frozen token interface

| Representation | IDs / rule |
|---|---|
| Base FDDFS vocabulary | 0–333 |
| Reserved FDDFS ring slots | 334–582 |
| Learned BPE IDs | 583–17203, 16,621 ordered merges |
| Raw BPE vocabulary size | 17,204 |
| Model special IDs | PAD=0, BOS=1, EOS=2 |
| Raw molecular ID to model ID | add 3 exactly once |
| Foundation-model vocabulary | 17,207 |
| SFT reserved mask slot | 17,207 |
| Target type: small molecule / peptide / protein / DNA / RNA | 17,208 / 17,209 / 17,210 / 17,211 / 17,212 |
| SFT vocabulary | 17,213 |
| Multi-component FDDFS separator | raw ID 2, model ID 5 (not model EOS) |

The three tokenizer assets, `guide_counts.json`, `vocab.json` and `merges.csv`,
are bundled under `moltoken/assets/fddfs_bpe/`. They define the input vocabulary
used by the checkpoints. Some older asset directory names contain `64k`; the
actual vocabulary size is given above.

## SFT checkpoints and external encoders

An SFT directory contains `axis_b_config.json`, `qwen3/` and
`condition_projectors.safetensors` (or the older `.pt`). Frozen encoders are
not copied into every checkpoint:

```python
from moltoken import load_sft
model, text_tokenizer = load_sft(
    "runs/task/full/best", text_encoder="checkpoints/all-mpnet-base-v2",
    device="cuda", dtype=torch.bfloat16,
)
```

Use `AxisBSFTCollator` with the loaded model's `molecule_condition_mode` and
`condition_token_budget` to prepare conditional inputs. The production
`moltoken sample-sft` CLI targets current `direct_qwen` checkpoints.

For MolBERT-conditioned checkpoints, also provide
`molecule_encoder="checkpoints/molbert/final"`. Format-v1 metadata specifying
`attention_mask_mean` / `attention_mask_mean_then_l2_normalize` selects mean
pooling before projection, with one token per condition. Format-v2 uses the
saved multi-token projection and budget. A standalone MolBERT encoder can be
loaded as `transformers.BertForMaskedLM`.

## Moving runs to a different filesystem

For inference, pass the new checkpoint and encoder locations. For comparisons
against paths recorded by an earlier run, `MOLTOKEN_PATH_MAP` accepts a JSON
file of old-to-new directory prefixes. For example, `paths.json` can contain:

```json
{"runs_previous": "runs"}
```

Set `MOLTOKEN_PATH_MAP=paths.json` when launching the command. Prefixes must
match the saved path spelling; the longest matching prefix takes precedence.
This mapping affects path comparisons, not model weights or checkpoint metadata.
