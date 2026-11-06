from __future__ import annotations

from typing import List, Tuple, Optional, Dict


class BPEEngine:
    """
    Unified high-performance BPE train + encode engine (pluggable backends).

    Training backends: "cpp" | "cpp_parallel" | "cpp_dense_parallel" |
    "cpp_dense_parallel_apply" | "cpp_dense_parallel_reuse" |
    "cpp_dense_parallel_pool" | "cpp_dense_sample_parallel" |
    "cpp_direct_pair_sample_parallel" | "cpp_pairmap_sample_parallel" |
    "numba" | "python"
    Encode backend: "cpp" (C++/pybind11, recommended) or "python" fallback.

    Output is semantically equivalent to minBPE reference:
    - merge_rules: List[Tuple[left_id, right_id, new_id]]
    - vocab_size = base_vocab_size + num_merges_performed
    """

    def __init__(self, *, train_backend: str = "cpp", encode_backend: str = "cpp",
                 train_num_threads: int = 0,
                 encode_rank_mode: str = "all", encode_rank_k: Optional[int] = None,
                 encode_rank_min: Optional[int] = None, encode_rank_max: Optional[int] = None,
                 encode_rank_dist: Optional[str] = None, verbose: bool = False) -> None:
        self.train_backend = train_backend
        self.train_num_threads = int(train_num_threads)
        self.encode_backend = encode_backend
        self.encode_rank_mode = encode_rank_mode
        self.encode_rank_k = encode_rank_k
        self.encode_rank_min = encode_rank_min
        self.encode_rank_max = encode_rank_max
        self.encode_rank_dist = encode_rank_dist
        self.verbose = bool(verbose)
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.vocab_size: Optional[int] = None
        self._encoder = None

    # ---------------- Training ----------------
    def train(self, token_sequences: List[List[int]], *, num_merges: int, min_frequency: int) -> Dict:
        if self.train_backend == "numba":
            return self._train_minbpe_numba(token_sequences, num_merges, min_frequency)
        elif self.train_backend == "python":
            from .main_bpe import StandardBPECompressor
            comp = StandardBPECompressor(num_merges=num_merges, min_frequency=min_frequency, debug=False)
            stats = comp.train(token_sequences)
            self.merge_rules = list(comp.merge_rules)
            self.vocab_size = len(comp.token_to_id)
            self._encoder = None
            return {
                "num_merges_performed": stats.get("num_merges_performed", len(self.merge_rules)),
                "final_vocab_size": self.vocab_size,
            }
        elif self.train_backend == "cpp":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp(token_sequences, int(num_merges), int(min_frequency))
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            return {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
        elif self.train_backend == "cpp_parallel":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_parallel(
                token_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            return {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
        elif self.train_backend == "cpp_dense_parallel":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel(
                token_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            return {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
        elif self.train_backend == "cpp_dense_parallel_apply":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel_apply(
                token_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            return {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
        elif self.train_backend == "cpp_dense_parallel_reuse":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel_reuse(
                token_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            return {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
        elif self.train_backend == "cpp_dense_parallel_pool":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel_pool(
                token_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            return {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
        elif self.train_backend == "cpp_dense_sample_parallel":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_sample_parallel(
                token_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            stats = {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
            if results.get("profile") is not None:
                stats["profile"] = results["profile"]
            return stats
        elif self.train_backend in {"cpp_direct_pair_sample_parallel", "cpp_pairmap_sample_parallel"}:
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_direct_pair_sample_parallel(
                token_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            self._encoder = None
            stats = {
                "num_merges_performed": int(results["num_merges_performed"]),
                "final_vocab_size": self.vocab_size,
            }
            if results.get("profile") is not None:
                stats["profile"] = results["profile"]
            return stats
        else:
            raise ValueError(f"Unsupported train_backend: {self.train_backend}")

    def continue_train(self, token_sequences: List[List[int]], *, num_merges: int, min_frequency: int) -> Dict:
        if self.vocab_size is None:
            raise ValueError("Load or train a codebook before continue_train")

        original_merge_rules = list(self.merge_rules)
        original_vocab_size = int(self.vocab_size)
        next_available_id = (max((nid for _, _, nid in original_merge_rules), default=original_vocab_size - 1) + 1)
        encoded_sequences = self.batch_encode(token_sequences)

        if self.train_backend == "python":
            from .main_bpe import StandardBPECompressor
            comp = StandardBPECompressor(num_merges=num_merges, min_frequency=min_frequency, debug=False)
            stats = comp.train(encoded_sequences)
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in comp.merge_rules:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend == "numba":
            return self._continue_train_minbpe_numba(
                encoded_sequences, num_merges, min_frequency, original_merge_rules, original_vocab_size, next_available_id
            )
        elif self.train_backend == "cpp":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp(encoded_sequences, int(num_merges), int(min_frequency))
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend == "cpp_parallel":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_parallel(
                encoded_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend == "cpp_dense_parallel":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel(
                encoded_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend == "cpp_dense_parallel_apply":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel_apply(
                encoded_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend == "cpp_dense_parallel_reuse":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel_reuse(
                encoded_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend == "cpp_dense_parallel_pool":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_parallel_pool(
                encoded_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend == "cpp_dense_sample_parallel":
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_dense_sample_parallel(
                encoded_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        elif self.train_backend in {"cpp_direct_pair_sample_parallel", "cpp_pairmap_sample_parallel"}:
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp_direct_pair_sample_parallel(
                encoded_sequences,
                int(num_merges),
                int(min_frequency),
                int(self.train_num_threads),
            )
            appended_rules = []
            next_id = next_available_id
            for left_id, right_id, _ in results["merge_rules"]:
                appended_rules.append((int(left_id), int(right_id), int(next_id)))
                next_id += 1
            self.merge_rules = original_merge_rules + appended_rules
            self.vocab_size = max(original_vocab_size, next_id)
            self._encoder = None
            return {
                "num_merges_performed": len(appended_rules),
                "final_vocab_size": self.vocab_size,
                "base_vocab_size": original_vocab_size,
                "appended_merge_rules": appended_rules,
                "encoded_sequence_count": len(encoded_sequences),
            }
        else:
            raise ValueError(f"Unsupported train_backend: {self.train_backend}")

    def _train_minbpe_numba(self, token_sequences: List[List[int]], num_merges: int, min_frequency: int) -> Dict:
        return self._run_minbpe_numba(token_sequences, num_merges, min_frequency, start_id=None, append_to_rules=False)

    def _continue_train_minbpe_numba(
        self,
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        original_merge_rules: List[Tuple[int, int, int]],
        original_vocab_size: int,
        next_available_id: int,
    ) -> Dict:
        result = self._run_minbpe_numba(
            token_sequences, num_merges, min_frequency, start_id=next_available_id, append_to_rules=True
        )
        appended_rules = result.pop("appended_merge_rules")
        self.merge_rules = original_merge_rules + appended_rules
        self.vocab_size = max(original_vocab_size, next_available_id + len(appended_rules))
        self._encoder = None
        result["appended_merge_rules"] = appended_rules
        result["base_vocab_size"] = original_vocab_size
        result["final_vocab_size"] = self.vocab_size
        return result

    def _run_minbpe_numba(
        self,
        token_sequences: List[List[int]],
        num_merges: int,
        min_frequency: int,
        start_id: Optional[int],
        append_to_rules: bool,
    ) -> Dict:
        from .numba_bpe_train import count_pairs_ragged, apply_merge_ragged, select_best_pair
        import numpy as np

        base_vocab: Dict[int, None] = {}
        for s in token_sequences:
            for t in s:
                base_vocab[int(t)] = None

        base_vocab_size = len(base_vocab)
        max_base_id = max(base_vocab.keys()) if base_vocab else -1
        if start_id is None:
            separator_token = max_base_id + 1
            next_id = separator_token + 1
        else:
            next_id = int(start_id)

        offsets = [0]
        flat = []
        for s in token_sequences:
            flat.extend([int(x) for x in s])
            offsets.append(len(flat))
        flat = np.asarray(flat, dtype=np.int32)
        offsets = np.asarray(offsets, dtype=np.int32)

        if len(offsets) > 2:
            warm_n = min(2, len(offsets) - 1)
            w_flat = flat[:offsets[warm_n]]
            w_offsets = offsets[:warm_n + 1].copy()
            _ = count_pairs_ragged(w_flat, w_offsets)
            if w_offsets[-1] >= 3:
                _ = apply_merge_ragged(w_flat, w_offsets, np.int32(w_flat[0]), np.int32(w_flat[1]), np.int32(10))

        pair_keys, pair_counts = count_pairs_ragged(flat, offsets)
        merges_done = 0
        merge_rules: List[Tuple[int, int, int]] = []

        while merges_done < num_merges:
            best_left_i32, best_right_i32, best_freq_i32 = select_best_pair(
                pair_keys, pair_counts, np.int32(min_frequency)
            )
            best_freq = int(best_freq_i32)
            if best_freq < int(min_frequency):
                break

            left_id = int(best_left_i32)
            right_id = int(best_right_i32)
            new_id = next_id
            next_id += 1
            flat, offsets = apply_merge_ragged(flat, offsets, np.int32(left_id), np.int32(right_id), np.int32(new_id))
            pair_keys, pair_counts = count_pairs_ragged(flat, offsets)
            merge_rules.append((left_id, right_id, new_id))
            merges_done += 1

            if self.verbose and (merges_done % 200 == 0 or merges_done <= 5):
                print(f"Merge {merges_done}: ({left_id}, {right_id}) -> {new_id} (freq: {best_freq})")

        if not append_to_rules:
            self.merge_rules = merge_rules
            self.vocab_size = base_vocab_size + merges_done if start_id is None else next_id
            self._encoder = None
            return {"num_merges_performed": merges_done, "final_vocab_size": self.vocab_size}
        return {
            "num_merges_performed": merges_done,
            "final_vocab_size": next_id,
            "appended_merge_rules": merge_rules,
        }

    # ---------------- Encode backend ----------------
    def build_encoder(self) -> None:
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("Train first or load a codebook")
        if self.encode_backend == "cpp":
            from .cpp_bpe_backend import CppBPEBackend
            self._encoder = CppBPEBackend.from_codebook(
                merge_rules=self.merge_rules,
                pair_to_rank=None,
                pair_to_newid=None,
                vocab_size=self.vocab_size,
            )
        elif self.encode_backend == "python":
            self._encoder = None
        else:
            raise ValueError(f"Unsupported encode_backend: {self.encode_backend}")

    def _encode_python(self, seq: List[int]) -> List[int]:
        if self.encode_rank_mode == "none":
            return seq
        id_seq = [int(x) for x in seq]
        pair_to_rank = {(int(l), int(r)): idx for idx, (l, r, _) in enumerate(self.merge_rules)}
        pair_to_newid = {(int(l), int(r)): int(nid) for (l, r, nid) in self.merge_rules}
        while len(id_seq) >= 2:
            stats = {}
            for pair in zip(id_seq, id_seq[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            pair = min(stats, key=lambda p: pair_to_rank.get(p, float("inf")))
            if pair not in pair_to_newid:
                break
            new_id = pair_to_newid[pair]
            new_ids = []
            i = 0
            while i < len(id_seq):
                if i < len(id_seq) - 1 and id_seq[i] == pair[0] and id_seq[i + 1] == pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(id_seq[i])
                    i += 1
            id_seq = new_ids
        return id_seq

    def encode(self, seq: List[int]) -> List[int]:
        if self.encode_backend == "python":
            return self._encode_python(seq)
        if self.encode_rank_mode == "none":
            return seq
        if self.encode_backend == "cpp" and self.encode_rank_mode in ("topk", "random", "gaussian"):
            topk = self._sample_topk()
            from .cpp_bpe_backend import CppBPEBackend
            if isinstance(self._encoder, CppBPEBackend) and topk is not None:
                return self._encoder.encode_topk(seq, int(topk))
        if self._encoder is None:
            self.build_encoder()
        assert self._encoder is not None, "Encoder not built"
        return self._encoder.encode(seq)

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        if self.encode_backend == "python":
            return [self._encode_python(s) for s in seqs]
        if self.encode_rank_mode == "none":
            return seqs
        if self.encode_backend == "cpp" and self.encode_rank_mode in ("topk", "random", "gaussian"):
            topk = self._sample_topk()
            from .cpp_bpe_backend import CppBPEBackend
            if isinstance(self._encoder, CppBPEBackend) and topk is not None:
                return self._encoder.batch_encode_topk(seqs, int(topk))
        if self._encoder is None:
            self.build_encoder()
        assert self._encoder is not None, "Encoder not built"
        if hasattr(self._encoder, "batch_encode"):
            return self._encoder.batch_encode(seqs)
        return [self.encode(s) for s in seqs]

    def _sample_topk(self) -> Optional[int]:
        import random
        if self.encode_rank_mode == 'all':
            return None
        if self.encode_rank_mode == 'topk':
            return int(self.encode_rank_k) if self.encode_rank_k is not None else None
        if self.encode_rank_min is None or self.encode_rank_max is None:
            a = 0
            b = len(self.merge_rules) if self.merge_rules is not None else 0
        else:
            a, b = int(self.encode_rank_min), int(self.encode_rank_max)
        if a < 0:
            a = 0
        if b <= a:
            b = a
        if self.encode_rank_mode == 'random':
            if (self.encode_rank_dist or 'uniform') == 'uniform':
                return random.randint(a, b)
            if self.encode_rank_dist == 'triangular':
                mode = b
                return int(random.triangular(a, b, mode))
            return random.randint(a, b)
        if self.encode_rank_mode == 'gaussian':
            mu = float(b)
            sigma = max(1.0, (b - a) / 3.0)
            for _ in range(8):
                x = int(random.gauss(mu, sigma))
                if a <= x <= b:
                    return x
            return b
        return None

    # ---------------- Codebook IO ----------------
    def to_codebook(self) -> Dict:
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("Codebook not initialized")
        return {"merge_rules": list(self.merge_rules), "vocab_size": int(self.vocab_size)}

    def load_codebook_dict(self, data: Dict) -> None:
        assert isinstance(data, dict) and 'merge_rules' in data and 'vocab_size' in data, (
            "Codebook dict missing required fields: 'merge_rules' or 'vocab_size'"
        )
        self.merge_rules = [tuple(int(x) for x in t) for t in data['merge_rules']]
        self.vocab_size = int(data['vocab_size'])
        self._encoder = None

    def save_codebook(self, path: str) -> None:
        import os
        import pickle
        import json
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("Codebook not initialized")
        suffix = os.path.splitext(path)[1].lower()
        data = self.to_codebook()
        if suffix in ('.pkl', '.pickle'):
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported codebook file extension: {suffix}. Use .pkl or .json.")

    def load_codebook(self, path: str) -> None:
        import os
        import pickle
        import json
        suffix = os.path.splitext(path)[1].lower()
        if suffix in ('.pkl', '.pickle'):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported codebook file extension: {suffix}. Use .pkl or .json.")
        self.load_codebook_dict(data)

    @classmethod
    def from_codebook_dict(cls, data: Dict, *, encode_backend: str = 'cpp', **engine_kwargs) -> "BPEEngine":
        eng = cls(train_backend='python', encode_backend=encode_backend, **engine_kwargs)
        eng.load_codebook_dict(data)
        return eng

    @classmethod
    def load_codebook_to_engine(cls, path: str, *, encode_backend: str = 'cpp', **engine_kwargs) -> "BPEEngine":
        eng = cls(train_backend='python', encode_backend=encode_backend, **engine_kwargs)
        eng.load_codebook(path)
        return eng
