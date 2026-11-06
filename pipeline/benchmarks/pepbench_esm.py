"""Frozen ESM-2 8M evaluator, independent of the Axis conditioning pipeline.

Embedding = final layer mean over actual residues (no BOS/EOS/padding).
pPPL = exp(mean single-residue masked negative log probability), per sequence.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.benchmarks.pepbench_metrics import sequence_is_valid


class Esm8MFeatures:
    def __init__(self, model_dir: Path, *, device="cpu", batch_size=32, mask_batch_size=32):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if batch_size < 1 or mask_batch_size < 1:
            raise ValueError("positive evaluator batch sizes required")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            str(model_dir), local_files_only=True, use_safetensors=True,
            attn_implementation="eager", torch_dtype=torch.float32,
        ).eval().to(device)
        if self.model.config.num_hidden_layers != 6 or self.model.config.hidden_size != 320:
            raise ValueError("this frozen evaluator requires ESM-2 8M (6 layers, 320 dimensions)")
        self.device, self.batch_size, self.mask_batch_size = device, batch_size, mask_batch_size
        self.model_dir = model_dir
        torch.backends.cuda.matmul.allow_tf32 = False

    def _tokens(self, sequences):
        if any(not sequence_is_valid(s) for s in sequences):
            raise ValueError("ESM evaluator accepts canonical 5–80-AA valid outputs only")
        tokens = self.tokenizer(sequences, padding=True, return_tensors="pt", truncation=False)
        lengths = tokens["attention_mask"].sum(dim=1).tolist()
        if lengths != [len(s) + 2 for s in sequences]:
            raise ValueError("ESM tokenization is not one token per residue plus BOS/EOS")
        return {k: v.to(self.device) for k, v in tokens.items()}

    @torch.inference_mode()
    def embeddings(self, sequences):
        if not sequences:
            return np.empty((0, 320), dtype=np.float32)
        batches = []
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start:start + self.batch_size]
            tokens = self._tokens(batch)
            hidden = self.model.esm(**tokens, return_dict=True).last_hidden_state
            batches.append(torch.stack([hidden[i, 1:len(s) + 1].mean(dim=0)
                                        for i, s in enumerate(batch)]).cpu().numpy())
        result = np.concatenate(batches)
        if not np.isfinite(result).all():
            raise ValueError("ESM returned non-finite embeddings")
        return result

    @torch.inference_mode()
    def pseudo_perplexity(self, sequence):
        tokens = self._tokens([sequence])
        original = tokens["input_ids"]
        losses = []
        for start in range(1, len(sequence) + 1, self.mask_batch_size):
            positions = torch.arange(start, min(start + self.mask_batch_size, len(sequence) + 1), device=self.device)
            n = len(positions)
            masked = original.repeat(n, 1)
            targets = masked[torch.arange(n, device=self.device), positions].clone()
            masked[torch.arange(n, device=self.device), positions] = self.tokenizer.mask_token_id
            output = self.model(input_ids=masked, attention_mask=tokens["attention_mask"].repeat(n, 1))
            logits = output.logits[torch.arange(n, device=self.device), positions]
            losses.extend(F.cross_entropy(logits, targets, reduction="none").double().cpu().tolist())
        result = math.exp(math.fsum(losses) / len(sequence))
        if not math.isfinite(result):
            raise ValueError("ESM returned non-finite pPPL")
        return result

    def provenance(self):
        return {"model": "facebook/esm2_t6_8M_UR50D", "model_dir": str(self.model_dir),
                "layer": 6, "embedding_dimension": 320,
                "pooling": "mean of actual residues only; no special tokens or padding",
                "dtype": "float32", "attention": "eager", "device": self.device,
                "embedding_batch_size": self.batch_size, "mask_batch_size": self.mask_batch_size,
                "ppl_masking": "one residue at a time; full model vocabulary; no truncation"}
