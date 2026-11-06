from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from . import _cpp_bpe  # compiled pybind11 module
except Exception as e:  # noqa: E722
    raise ImportError(
        "_cpp_bpe extension not found. Build the C++ backend first (see README)."
    ) from e


class CppBPEBackend:
    def __init__(self, encoder: "_cpp_bpe.MinBPEEncoder", vocab_size: Optional[int] = None) -> None:
        self._encoder = encoder
        self._vocab_size = int(vocab_size) if vocab_size is not None else None

    @classmethod
    def from_bpe_model(cls, model) -> "CppBPEBackend":
        # Prefer model-derived structure if available
        if hasattr(model, "_minbpe_pair_to_rank") and hasattr(model, "_minbpe_pair_to_newid"):
            pair_to_rank = model._minbpe_pair_to_rank
            pair_to_newid = model._minbpe_pair_to_newid
        else:
            pair_to_rank = {}
            pair_to_newid = {}
            for idx, (l, r, new_id) in enumerate(getattr(model, "merge_rules", [])):
                pair_to_rank[(int(l), int(r))] = int(idx)
                pair_to_newid[(int(l), int(r))] = int(new_id)
        lefts, rights, ranks, new_ids = [], [], [], []
        for (l, r), rk in pair_to_rank.items():
            lefts.append(int(l))
            rights.append(int(r))
            ranks.append(int(rk))
            new_ids.append(int(pair_to_newid[(l, r)]))
        # Convert to int32 to match C++ data width
        enc = _cpp_bpe.MinBPEEncoder(
            np.asarray(lefts, dtype=np.int32).tolist(),
            np.asarray(rights, dtype=np.int32).tolist(),
            np.asarray(ranks, dtype=np.int32).tolist(),
            np.asarray(new_ids, dtype=np.int32).tolist(),
        )
        vocab_size = len(getattr(model, "token_to_id", {})) or None
        return cls(enc, vocab_size=vocab_size)

    @classmethod
    def from_codebook(
        cls,
        *,
        merge_rules: Optional[List[Tuple[int, int, int]]],
        pair_to_rank: Optional[Dict[Tuple[int, int], int]],
        pair_to_newid: Optional[Dict[Tuple[int, int], int]],
        vocab_size: Optional[int] = None,
    ) -> "CppBPEBackend":
        if pair_to_rank is None or pair_to_newid is None:
            if merge_rules is None:
                raise ValueError("Provide pair_to_rank/pair_to_newid or merge_rules to build codebook")
            pair_to_rank = {}
            pair_to_newid = {}
            for idx, (l, r, new_id) in enumerate(merge_rules):
                pair_to_rank[(int(l), int(r))] = int(idx)
                pair_to_newid[(int(l), int(r))] = int(new_id)
        lefts, rights, ranks, new_ids = [], [], [], []
        for (l, r), rk in pair_to_rank.items():
            lefts.append(int(l))
            rights.append(int(r))
            ranks.append(int(rk))
            new_ids.append(int(pair_to_newid[(l, r)]))
        enc = _cpp_bpe.MinBPEEncoder(
            np.asarray(lefts, dtype=np.int32).tolist(),
            np.asarray(rights, dtype=np.int32).tolist(),
            np.asarray(ranks, dtype=np.int32).tolist(),
            np.asarray(new_ids, dtype=np.int32).tolist(),
        )
        return cls(enc, vocab_size=vocab_size)

    # -------- Training (minBPE) --------
    @staticmethod
    def train_minbpe_cpp(token_sequences: List[List[int]], num_merges: int, min_frequency: int) -> Dict:
        """Train minBPE using C++ native implementation."""
        results = _cpp_bpe.train_minbpe(token_sequences, int(num_merges), int(min_frequency))
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def train_minbpe_cpp_parallel(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Train minBPE using C++ native implementation with parallel pair counting."""
        if not hasattr(_cpp_bpe, "train_minbpe_parallel"):
            raise RuntimeError("Compiled _cpp_bpe extension does not expose train_minbpe_parallel; rebuild it first.")
        results = _cpp_bpe.train_minbpe_parallel(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            int(num_threads),
        )
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def train_minbpe_cpp_dense_parallel(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Train minBPE with dense pair counters and parallel pair counting."""
        if not hasattr(_cpp_bpe, "train_minbpe_dense_parallel"):
            raise RuntimeError("Compiled _cpp_bpe extension does not expose train_minbpe_dense_parallel; rebuild it first.")
        results = _cpp_bpe.train_minbpe_dense_parallel(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            int(num_threads),
        )
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def train_minbpe_cpp_dense_parallel_apply(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Train minBPE with dense parallel pair counting and experimental parallel apply-merge."""
        if not hasattr(_cpp_bpe, "train_minbpe_dense_parallel_apply"):
            raise RuntimeError("Compiled _cpp_bpe extension does not expose train_minbpe_dense_parallel_apply; rebuild it first.")
        results = _cpp_bpe.train_minbpe_dense_parallel_apply(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            int(num_threads),
        )
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def train_minbpe_cpp_dense_parallel_reuse(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Train minBPE with dense parallel pair counting and reusable serial apply buffer."""
        if not hasattr(_cpp_bpe, "train_minbpe_dense_parallel_reuse"):
            raise RuntimeError("Compiled _cpp_bpe extension does not expose train_minbpe_dense_parallel_reuse; rebuild it first.")
        results = _cpp_bpe.train_minbpe_dense_parallel_reuse(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            int(num_threads),
        )
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def train_minbpe_cpp_dense_parallel_pool(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Train minBPE with dense parallel pair counting using persistent worker threads."""
        if not hasattr(_cpp_bpe, "train_minbpe_dense_parallel_pool"):
            raise RuntimeError("Compiled _cpp_bpe extension does not expose train_minbpe_dense_parallel_pool; rebuild it first.")
        results = _cpp_bpe.train_minbpe_dense_parallel_pool(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            int(num_threads),
        )
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def train_minbpe_cpp_dense_sample_parallel(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Train minBPE by counting/applying within sample shards instead of a separator-flattened stream."""
        if not hasattr(_cpp_bpe, "train_minbpe_dense_sample_parallel"):
            raise RuntimeError("Compiled _cpp_bpe extension does not expose train_minbpe_dense_sample_parallel; rebuild it first.")
        results = _cpp_bpe.train_minbpe_dense_sample_parallel(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            int(num_threads),
        )
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def train_minbpe_cpp_pairmap_sample_parallel(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Compatibility alias for direct pair-frequency sample-level minBPE."""
        return CppBPEBackend.train_minbpe_cpp_direct_pair_sample_parallel(
            token_sequences,
            num_merges,
            min_frequency,
            num_threads,
        )

    @staticmethod
    def train_minbpe_cpp_direct_pair_sample_parallel(
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        num_threads: int = 0,
    ) -> Dict:
        """Train sample-level minBPE with direct freq[left][right] pair counting."""
        if not hasattr(_cpp_bpe, "train_minbpe_direct_pair_sample_parallel"):
            raise RuntimeError("Compiled _cpp_bpe extension does not expose train_minbpe_direct_pair_sample_parallel; rebuild it first.")
        results = _cpp_bpe.train_minbpe_direct_pair_sample_parallel(
            token_sequences,
            int(num_merges),
            int(min_frequency),
            int(num_threads),
        )
        return CppBPEBackend._normalize_train_results(results)

    @staticmethod
    def _normalize_train_results(results) -> Dict:
        # Normalize return structure
        merge_rules_py: List[Tuple[int, int, int]] = []
        for t in results["merge_rules"]:
            l, r, nid = int(t[0]), int(t[1]), int(t[2])
            merge_rules_py.append((l, r, nid))
        return {
            "merge_rules": merge_rules_py,
            "num_merges_performed": int(results["num_merges_performed"]),
            "final_vocab_size": int(results["final_vocab_size"]),
            "base_vocab_size": int(results["base_vocab_size"]),
            "separator_token": int(results["separator_token"]),
            "profile": dict(results["profile"]) if "profile" in results else None,
        }

    def encode(self, seq: List[int]) -> List[int]:
        if not isinstance(seq, (list, tuple)):
            raise TypeError("encode requires List[int]")
        arr = np.asarray(seq, dtype=np.int32)
        return [int(x) for x in self._encoder.encode(arr.tolist())]

    def encode_topk(self, seq: List[int], topk: int) -> List[int]:
        if not isinstance(seq, (list, tuple)):
            raise TypeError("encode_topk requires List[int]")
        arr = np.asarray(seq, dtype=np.int32)
        return [int(x) for x in self._encoder.encode_with_limit(arr.tolist(), int(topk))]

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        if not isinstance(seqs, (list, tuple)):
            raise TypeError("batch_encode requires List[List[int]]")
        # Use ragged interface to minimize Python loop overhead
        offsets = [0]
        flat = []
        for s in seqs:
            arr = np.asarray(s, dtype=np.int32)
            flat.extend(arr.tolist())
            offsets.append(len(flat))
        flat_out, out_offsets = self._encoder.encode_ragged(flat, offsets)
        results: List[List[int]] = []
        for i in range(len(out_offsets) - 1):
            b = out_offsets[i]
            e = out_offsets[i + 1]
            results.append([int(x) for x in flat_out[b:e]])
        return results

    def batch_encode_topk(self, seqs: List[List[int]], topk: int) -> List[List[int]]:
        if not isinstance(seqs, (list, tuple)):
            raise TypeError("batch_encode_topk requires List[List[int]]")
        # Use C++ batch limit interface
        arrs: List[List[int]] = []
        for s in seqs:
            arrs.append(np.asarray(s, dtype=np.int32).tolist())
        outs = self._encoder.batch_encode_with_limit(arrs, int(topk))
        return [[int(x) for x in out] for out in outs]

    # Debug/verification: returns (encoded, ranks_applied)
    def encode_with_limit_trace(self, seq: List[int], rank_limit: int) -> Tuple[List[int], List[int]]:
        arr = np.asarray(seq, dtype=np.int32)
        out, ranks = self._encoder.encode_with_limit_trace(arr.tolist(), int(rank_limit))
        return [int(x) for x in out], [int(r) for r in ranks]
