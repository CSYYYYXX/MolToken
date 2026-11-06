"""Process-local frozen codec used by biological data materializers."""
from functools import lru_cache

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
CODEC = DECODER = None


@lru_cache(maxsize=256)
def encode_sequence(kind, sequence):
    from pipeline.serialization.fddfs import deserialize_fddfs
    from pipeline.graph.biopolymer_sequence import graph_to_biopolymer_sequence
    if CODEC is None or DECODER is None:
        raise RuntimeError("initialize process-local CODEC and DECODER before encoding")
    metrics, tokens, ids = CODEC.encode_sequence(kind, sequence, "biopolymer")
    _, expanded = DECODER.decode_raw_ids(ids)
    if tokens != expanded:
        raise ValueError("BPE inverse differs from original FDDFS")
    graph = deserialize_fddfs(expanded, DECODER.vocab.default_edge)
    if graph_to_biopolymer_sequence(graph, target_type=kind) != sequence:
        raise ValueError(f"{kind} sequence roundtrip mismatch")
    return metrics, ids
