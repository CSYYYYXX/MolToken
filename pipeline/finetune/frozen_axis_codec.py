"""Frozen heavy-graph -> FDDFS -> pretraining-BPE codec for downstream data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from cli.sft.compress_finetune_fddfs_bpe import (
    fddfs_tokens_to_base_ids,
    load_merge_rules,
    load_vocab,
    validate_merge_dependencies,
)
from pipeline.finetune.fddfs_encode import (
    FastCorpusEncoder,
    build_graph,
    load_corpus_guide,
)
from src.algorithms.compression.bpe_engine import BPEEngine


TARGET_FRAME_TOKENS = 3


class FrozenAxisCodec:
    """Encode biological sequences with the exact foundation-model dialect."""

    def __init__(self, guide_path: Path, vocab_path: Path, merges_path: Path) -> None:
        corpus_guide = load_corpus_guide(guide_path)
        self.fddfs = FastCorpusEncoder(corpus_guide)
        self.vocab = load_vocab(vocab_path)
        rules, bpe_vocab_size, first_merge_id = load_merge_rules(merges_path)
        validate_merge_dependencies(rules, initial_max_token=first_merge_id - 1)
        self.first_merge_id = int(first_merge_id)
        self.bpe = BPEEngine.from_codebook_dict(
            {"merge_rules": rules, "vocab_size": bpe_vocab_size},
            encode_backend="cpp",
        )
        self.codebook = {
            "guide": str(guide_path),
            "vocab": str(vocab_path),
            "merges": str(merges_path),
            "merge_rules": len(rules),
            "first_merge_id": int(first_merge_id),
            "max_raw_bpe_id": int(bpe_vocab_size - 1),
        }

    def bpe_ids(self, tokens: Sequence[str], label: str) -> list[int]:
        base = fddfs_tokens_to_base_ids(
            tokens,
            token_to_id=self.vocab["token_to_id"],
            ring_base=self.vocab["ring_base"],
            ring_min=self.vocab["ring_min"],
            first_merge_id=self.first_merge_id,
            sample_label=label,
        )
        return [int(value) for value in self.bpe.encode(base)]

    def encode_sequence(
        self,
        target_type: str,
        sequence: str,
        label: str,
        *,
        text_condition_tokens: int = 0,
        molecule_condition_bpe_tokens: int = 0,
    ) -> tuple[dict[str, int], list[str], list[int]]:
        graph = build_graph(
            target_type=target_type,
            sample_id=label,
            sequence=sequence,
        )
        tokens = self.fddfs.serialize(graph, label)
        bpe = self.bpe_ids(tokens, label)
        context = (
            int(text_condition_tokens)
            + int(molecule_condition_bpe_tokens)
            + len(bpe)
            + TARGET_FRAME_TOKENS
        )
        metrics = {
            "residue_length": len(sequence),
            "heavy_nodes": int(graph["num_nodes"]),
            "fddfs_tokens": len(tokens),
            "bpe_tokens": len(bpe),
            "text_condition_tokens": int(text_condition_tokens),
            "molecule_condition_bpe_tokens": int(molecule_condition_bpe_tokens),
            "full_context_tokens": int(context),
        }
        return metrics, tokens, bpe


__all__ = ["FrozenAxisCodec", "TARGET_FRAME_TOKENS"]
