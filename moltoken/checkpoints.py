"""Explicit, strict loaders for original packed, HF and conditional checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path


def checkpoint_path_identity(path):
    """Resolve an explicitly supplied old-prefix -> new-prefix relocation map.

    MOLTOKEN_PATH_MAP names a JSON object. Longest directory prefix wins.
    This changes path comparisons only, never weights or saved metadata.
    """
    value = str(Path(path).expanduser())
    mapping_file = os.environ.get("MOLTOKEN_PATH_MAP")
    if mapping_file:
        mapping = json.loads(Path(mapping_file).read_text())
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k,v in mapping.items()):
            raise ValueError("MOLTOKEN_PATH_MAP must name a JSON object of path prefixes")
        for old in sorted(mapping, key=len, reverse=True):
            old_prefix = old.rstrip("/")
            if value == old_prefix or value.startswith(old_prefix + "/"):
                value = str(Path(mapping[old]) / value[len(old_prefix):].lstrip("/"))
                break
    return str(Path(value).resolve())


def _weight_files(root):
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if index.is_file():
            mapping = json.loads(index.read_text())["weight_map"]
            files = [root / name for name in sorted(set(mapping.values()))]
            for file in files:
                if file.parent.resolve() != root.resolve():
                    raise ValueError(f"checkpoint shard escapes its directory: {file}")
            return files
    for name in ("model.safetensors", "pytorch_model.bin"):
        file = root / name
        if file.is_file():
            return [file]
    raise FileNotFoundError(f"no supported weight file or shard index in {root}")


def _read_weights(file):
    if file.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(file), device="cpu")
    import torch
    # Do not retry with unrestricted pickle execution.
    return torch.load(file, map_location="cpu", weights_only=True)


def load_pretrained(checkpoint, *, device="cpu", dtype=None, attention="sdpa"):
    """Load Qwen3 weights without renaming token rows or initializing missing weights.

    Original packed checkpoints use embed_tokens/layers/norm at the root;
    HF exports use model.embed_tokens/model.layers/model.norm. Both are
    supported. The returned object is a standard Qwen3ForCausalLM.
    """
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen3Config, Qwen3ForCausalLM
    root = Path(checkpoint)
    if (root / "axis_b_config.json").is_file():
        raise ValueError("conditional checkpoint: use load_sft() or explicitly pass its qwen3 subdirectory")
    config = Qwen3Config.from_pretrained(root, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError(f"expected Qwen3, got {config.model_type}")
    config._attn_implementation = attention
    dtype = torch.float32 if dtype is None else dtype
    # Keep non-persistent rotary buffers real; they are not present in weights.
    with init_empty_weights(include_buffers=False):
        model = Qwen3ForCausalLM(config)
    expected = set(model.state_dict())
    loaded = set()
    for file in _weight_files(root):
        state = _read_weights(file)
        if not isinstance(state, dict):
            raise ValueError(f"not a state dictionary: {file}")
        mapped = {}
        for key, value in state.items():
            if key.startswith(("embed_tokens.", "layers.", "norm.")):
                key = "model." + key
            if key not in expected or key in loaded:
                raise ValueError(f"unexpected or duplicate checkpoint key {key!r} in {file}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"checkpoint value is not a tensor: {key}")
            mapped[key] = value.to(dtype=dtype) if value.is_floating_point() else value
        model.load_state_dict(mapped, strict=False, assign=True)
        loaded.update(mapped)
        del mapped, state
    missing = expected - loaded
    tied = {"model.embed_tokens.weight", "lm_head.weight"}
    if config.tie_word_embeddings and missing.issubset(tied) and loaded.intersection(tied):
        if "model.embed_tokens.weight" in missing:
            model.model.embed_tokens.weight = model.lm_head.weight
        missing = set()
    if missing:
        raise ValueError(f"missing model weights: {sorted(missing)}")
    model.tie_weights()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("checkpoint left meta parameters uninitialized")
    model.to(device)
    model.eval()
    return model


def load_sft(checkpoint, *, text_encoder, molecule_encoder=None, device="cpu", dtype=None):
    """Load a conditional checkpoint plus its separately distributed encoders.

    The condition mode and normalization/budget are read from the checkpoint,
    not replaced by the defaults of the current training CLI. Old MolBERT
    conditions require molecule_encoder explicitly; no network downloads occur.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer, BertForMaskedLM
    from pipeline.finetune.axis_b_model import AxisBConditionalModel
    root = Path(checkpoint)
    cfg = json.loads((root / "axis_b_config.json").read_text())
    meta = cfg.get("metadata", {})
    safe = root / "condition_projectors.safetensors"
    weights = _read_weights(safe if safe.is_file() else root / "condition_projectors.pt")
    has_molecule = any(k.startswith("molecule_projection.") for k in weights)
    mode = cfg.get("molecule_condition_mode", meta.get("molecule_condition_mode"))
    if mode is None:
        mode = "molbert" if has_molecule else "direct_qwen"
    if (mode == "molbert") != has_molecule:
        raise ValueError("checkpoint condition mode and projector weights disagree")
    dtype = torch.float32 if dtype is None else dtype
    qwen = load_pretrained(root / "qwen3", dtype=dtype)
    text = AutoModel.from_pretrained(text_encoder, local_files_only=True, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder, local_files_only=True)
    molecule = None
    if mode == "molbert":
        if molecule_encoder is None:
            raise ValueError("this legacy checkpoint requires the original MolBERT directory")
        molecule = BertForMaskedLM.from_pretrained(molecule_encoder, local_files_only=True,
                                                  torch_dtype=dtype).bert
    encoder_dim = int(weights["text_projection.net.0.weight"].shape[1])
    version = int(cfg.get("format_version", 1))
    if version not in (1, 2):
        raise ValueError(f"unsupported conditional checkpoint format: {version}")
    model_class = AxisBConditionalModel
    extra = {}
    normalize_text = cfg.get("normalize_text", True)
    normalize_molecule = cfg.get("normalize_molecule", True)
    condition_budget = int(cfg.get("condition_token_budget", 512))
    if version == 1:
        from pipeline.finetune.legacy.mean_pooled_model import MeanPooledAxisBModel
        expected = {"molecule": "attention_mask_mean",
                    "text": "attention_mask_mean_then_l2_normalize"}
        if mode != "molbert" or cfg.get("condition_pooling") != expected:
            raise ValueError("unrecognized format-v1 pooling; cannot assume v2 behavior")
        model_class = MeanPooledAxisBModel
        extra = {"legacy_config": cfg}
        normalize_text, normalize_molecule, condition_budget = True, False, 1
    model = model_class(
        qwen=qwen, text_encoder=text, molecule_encoder=molecule, encoder_dim=encoder_dim,
        normalize_text=normalize_text, normalize_molecule=normalize_molecule,
        condition_token_budget=condition_budget,
        molecule_condition_mode=mode,
        **extra,
    )
    allowed_prefixes = ("text_projection.", "molecule_projection.")
    if any(not k.startswith(allowed_prefixes) for k in weights):
        raise ValueError("unrecognized condition projector state keys")
    for name in ("text_projection", "molecule_projection"):
        module = getattr(model, name)
        if module is None:
            continue
        part = {k[len(name)+1:]: v for k,v in weights.items() if k.startswith(name + ".")}
        module.load_state_dict(part, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer
