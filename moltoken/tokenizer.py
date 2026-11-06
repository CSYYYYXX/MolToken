"""Stable graph/FDDFS/BPE API using the original frozen token identifiers."""
from pathlib import Path

from pipeline.paths import TOKENIZER_ASSET_ROOT
from pipeline.finetune.fddfs_encode import build_graph
from pipeline.finetune.frozen_axis_codec import FrozenAxisCodec
from cli.evaluate.eval_axis_b_samples import BPEDecoder, FDDFSVocab
from pipeline.serialization.fddfs import deserialize_fddfs

DOMAINS = ("small_molecule", "peptide", "protein", "dna", "rna")
MODEL_OFFSET = 3
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


class MolTokenizer:
    """Frozen FDDFS + integer BPE; encode() returns RAW ids by default.

    Use model_ids=True to add offset 3 and BOS/EOS. Never apply that offset
    twice. Biological inputs expand directly from residue templates.
    """

    def __init__(self, asset_dir: str | Path | None = None):
        self.asset_dir = Path(asset_dir) if asset_dir else TOKENIZER_ASSET_ROOT
        self.codec = FrozenAxisCodec(self.asset_dir / "guide_counts.json",
                                     self.asset_dir / "vocab.json",
                                     self.asset_dir / "merges.csv")
        self.decoder = BPEDecoder(self.asset_dir / "merges.csv",
                                  FDDFSVocab.load(self.asset_dir / "vocab.json"))
        self.raw_vocab_size = self.codec.codebook["max_raw_bpe_id"] + 1
        self.vocab_size = self.raw_vocab_size + MODEL_OFFSET

    def encode(self, sequence: str, domain: str = "small_molecule", *,
               model_ids: bool = False) -> list[int]:
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
        kwargs = {"smiles": sequence} if domain == "small_molecule" else {"sequence": sequence}
        graph = build_graph(target_type=domain, sample_id="input", **kwargs)
        tokens = self.codec.fddfs.serialize(graph, "input")
        raw = self.codec.bpe_ids(tokens, "input")
        return [BOS_ID, *(i + MODEL_OFFSET for i in raw), EOS_ID] if model_ids else raw

    def _raw_ids(self, ids, model_ids):
        ids = [int(i) for i in ids]
        if model_ids:
            if ids and ids[0] == BOS_ID:
                ids = ids[1:]
            if EOS_ID in ids:
                end = ids.index(EOS_ID)
                if any(i != PAD_ID for i in ids[end + 1:]):
                    raise ValueError("non-padding tokens follow EOS")
                ids = ids[:end]
            ids = [i - MODEL_OFFSET for i in ids]
        if any(i < 0 or i >= self.raw_vocab_size for i in ids):
            raise ValueError("token outside the frozen raw-BPE vocabulary")
        if not ids:
            raise ValueError("empty molecular sequence")
        return ids

    def decode_tokens(self, ids, *, model_ids=False):
        return self.decoder.decode_raw_ids(self._raw_ids(ids, model_ids))[1]

    def decode_graph(self, ids, *, model_ids=False):
        return deserialize_fddfs(self.decode_tokens(ids, model_ids=model_ids),
                                 default_edge=self.decoder.vocab.default_edge)

    def decode_smiles(self, ids, *, model_ids=False):
        from pipeline.graph.to_smiles import graph_to_smiles
        return graph_to_smiles(self.decode_graph(ids, model_ids=model_ids))

    def decode_sequence(self, ids, domain, *, model_ids=False):
        """Recover a canonical linear peptide/protein/DNA/RNA sequence."""
        if domain not in DOMAINS[1:]:
            raise ValueError("decode_sequence requires peptide, protein, dna or rna")
        from pipeline.graph.biopolymer_sequence import graph_to_biopolymer_sequence
        return graph_to_biopolymer_sequence(
            self.decode_graph(ids, model_ids=model_ids), target_type=domain)
