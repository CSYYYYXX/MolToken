#!/usr/bin/env python3
"""用 FDDFS 预训练 checkpoint 做小规模无条件生成和化学检查。

训练时没有显式 BOS；这里按 domain 从真实训练样本里取首个 BPE token
作为起点，之后完全由模型自回归生成到 EOS。这个首 token 只是为了补上
causal LM 空上下文下无法预测第一个 token 的问题，不是条件生成。
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import csv
import json
import os
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from transformers import Qwen3Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.inference.generation import (  # noqa: E402
    graph_summary,
    graph_to_rdkit_mol,
    json_dumps,
    load_torch_state,
    model_ids_to_raw_bpe_ids,
    save_graph_npz,
    save_rdkit_png,
    generate_one,
)
from pipeline.serialization.fddfs import deserialize_fddfs  # noqa: E402
from cli.pretrain.train_qwen3_pretrain import (  # noqa: E402
    PackedXformersQwen3ForCausalLM,
)


sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

DEFAULT_DATA_ROOT = Path(str(_moltoken_paths.LOCAL_DATA_ROOT / ''))
DEFAULT_RUN_ROOT = Path(str(_moltoken_paths.LOCAL_DATA_ROOT.parent / 'runs'))
DEFAULT_RUN = DEFAULT_RUN_ROOT / "qwen3_0p3b_fddfs_mf80000_onlinepad512_shuffle_pretrain_v1_5gpu_b1024_acc2"
DEFAULT_CKPT = DEFAULT_RUN / "final"
DEFAULT_BPE_DIR = DEFAULT_DATA_ROOT / "fddfs_bpe_full_prefix70_64k_mf80000_v1"
DEFAULT_MERGES = _moltoken_paths.TOKENIZER_ASSET_ROOT / "merges.csv"
DEFAULT_VOCAB = _moltoken_paths.TOKENIZER_ASSET_ROOT / "vocab.json"
DEFAULT_RAW_FDDFS_ROOT = DEFAULT_DATA_ROOT / "fddfs_heavy_atom_normv2_compact_v1"
DEFAULT_OUT = DEFAULT_RUN / "generation_unconditional_fddfs_eval_v1" / "samples.jsonl"

SAMPLE_MAGIC = b"YTTMIBPE"
SAMPLE_VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIQQi")
LEN_STRUCT = struct.Struct("<I")


@dataclass
class FDDFSVocab:
    """FDDFS 基础 token id 到可读 token 的映射。"""

    id_to_token: dict[int, str]
    ring_base: int
    ring_min: int
    default_edge: tuple[int, ...]
    extra_id_to_token: dict[int, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "FDDFSVocab":
        """读取 FDDFS 序列化阶段生成的 vocab.json。"""

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        token_to_id = payload["token_to_id"]
        id_to_token = {int(v): str(k) for k, v in token_to_id.items()}
        serializer = str(payload.get("serializer", "")).lower()
        if serializer and serializer != "fddfs":
            raise ValueError(f"vocab serializer is not fddfs: {serializer}")
        return cls(
            id_to_token=id_to_token,
            ring_base=int(payload["ring_base"]),
            ring_min=int(payload["ring_min"]),
            default_edge=tuple(int(x) for x in payload["default_edge_feature"]),
        )

    def token_for_base_id(self, base_id: int, base_vocab_size: int) -> str:
        """把 BPE 展开后的基础 id 转成 FDDFS token 字符串。"""

        base_id = int(base_id)
        if base_id in self.id_to_token:
            return self.id_to_token[base_id]
        if base_id in self.extra_id_to_token:
            return self.extra_id_to_token[base_id]
        if self.ring_base <= base_id < int(base_vocab_size):
            return f"R{base_id - self.ring_base + self.ring_min}"
        raise ValueError(f"base id {base_id} is outside FDDFS vocab size {base_vocab_size}")


class BPEDecoder:
    """把 raw BPE id 递归展开成 FDDFS 基础 token id。"""

    def __init__(self, merges_csv: Path, vocab: FDDFSVocab):
        self.vocab = vocab
        self.merge_map: dict[int, tuple[int, int]] = {}
        min_new_id: int | None = None
        with merges_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                left = int(row["left"])
                right = int(row["right"])
                new_id = int(row["new_id"])
                self.merge_map[new_id] = (left, right)
                min_new_id = new_id if min_new_id is None else min(min_new_id, new_id)
        if min_new_id is None:
            raise ValueError(f"empty merges csv: {merges_csv}")
        self.base_vocab_size = int(min_new_id)
        self.max_token_id = max(self.merge_map)

    @lru_cache(maxsize=None)
    def expand_id(self, raw_id: int) -> tuple[int, ...]:
        """用显式栈展开一个 raw BPE id，避免高阶 token 触发递归深度限制。"""

        raw_id = int(raw_id)
        out: list[int] = []
        stack = [raw_id]
        while stack:
            current = int(stack.pop())
            if current in self.vocab.extra_id_to_token:
                out.append(current)
                continue
            if current < self.base_vocab_size:
                out.append(current)
                continue
            if current not in self.merge_map:
                raise ValueError(f"raw BPE id {current} is not in merge map")
            left, right = self.merge_map[current]
            stack.append(right)
            stack.append(left)
        return tuple(out)

    def decode_raw_ids(self, raw_ids: list[int]) -> tuple[list[int], list[str]]:
        """把一条 raw BPE id 序列展开成基础 id 和 FDDFS token。"""

        base_ids: list[int] = []
        for raw_id in raw_ids:
            base_ids.extend(self.expand_id(int(raw_id)))
        tokens = [self.vocab.token_for_base_id(x, self.base_vocab_size) for x in base_ids]
        return base_ids, tokens

    def encode_base_ids(self, base_ids: list[int]) -> list[int]:
        """按 merges.csv 的顺序把基础 FDDFS id 编成 raw BPE id。"""

        current = [int(x) for x in base_ids]
        if not current:
            return []
        for new_id, (left, right) in sorted(self.merge_map.items()):
            out: list[int] = []
            i = 0
            changed = False
            n = len(current)
            while i < n:
                if i + 1 < n and current[i] == left and current[i + 1] == right:
                    out.append(new_id)
                    i += 2
                    changed = True
                else:
                    out.append(current[i])
                    i += 1
            if changed:
                current = out
        return current


def read_exact(f: BinaryIO, nbytes: int, what: str) -> bytes:
    """读取固定字节数，文件截断时直接报错。"""

    data = f.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"unexpected EOF while reading {what}")
    return data


def read_compact_header(f: BinaryIO, path: Path) -> tuple[int, int, int]:
    """读取并校验 YTTMIBPE compact-int 文件头。"""

    magic, version, nseq, ntokens, max_token = HEADER_STRUCT.unpack(read_exact(f, HEADER_STRUCT.size, f"header {path}"))
    if magic != SAMPLE_MAGIC:
        raise ValueError(f"invalid compact-int magic in {path}: {magic!r}")
    if int(version) != SAMPLE_VERSION:
        raise ValueError(f"unsupported compact-int version={version} in {path}")
    return int(nseq), int(ntokens), int(max_token)


def read_compact_sequence_prefix(f: BinaryIO, path: Path, rec: int, cap: int) -> list[int]:
    """读取一条 compact-int 样本的前缀，并跳过剩余 payload。"""

    (length,) = LEN_STRUCT.unpack(read_exact(f, LEN_STRUCT.size, f"length {rec} from {path}"))
    length = int(length)
    take = length if cap <= 0 else min(length, int(cap))
    raw = read_exact(f, take * 4, f"payload prefix {rec} from {path}")
    remain = length - take
    if remain > 0:
        f.seek(remain * 4, os.SEEK_CUR)
    if take <= 0:
        return []
    return list(struct.unpack(f"<{take}i", raw))


class DomainStartSampler:
    """从指定 domain 的 FDDFS 原始序列中构建首 BPE token 经验池。"""

    def __init__(
        self,
        raw_fddfs_root: Path,
        domain: str,
        split: str,
        decoder: BPEDecoder,
        *,
        token_offset: int,
        pool_size: int,
        max_files: int,
        records_per_file: int,
        max_encode_len: int,
        rng: random.Random,
    ) -> None:
        self.pool: list[int] = []
        token_dir = raw_fddfs_root / "tokens" / domain / split
        if not token_dir.exists():
            raise FileNotFoundError(f"missing domain token dir: {token_dir}")
        token_files: list[Path] = []
        with os.scandir(token_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".fddfs.bin"):
                    token_files.append(Path(entry.path))
                    if max_files > 0 and len(token_files) >= max_files:
                        break
        if not token_files:
            raise FileNotFoundError(f"no .fddfs.bin files found under {token_dir}")
        rng.shuffle(token_files)
        for path in token_files:
            with path.open("rb") as f:
                nseq, _ntokens, _max_token = read_compact_header(f, path)
                for rec in range(min(int(nseq), int(records_per_file))):
                    base_ids = read_compact_sequence_prefix(f, path, rec, max_encode_len)
                    if not base_ids:
                        continue
                    bpe_ids = decoder.encode_base_ids(base_ids)
                    if bpe_ids:
                        self.pool.append(int(bpe_ids[0]) + int(token_offset))
                    if len(self.pool) >= pool_size:
                        return
        if not self.pool:
            raise RuntimeError(f"failed to build start-token pool for domain={domain}")

    def sample(self, rng: random.Random) -> int:
        """从 domain 首 token 池里抽一个模型 token id。"""

        return int(self.pool[rng.randrange(len(self.pool))])


def analyze_protein_like(graph: dict[str, Any]) -> dict[str, Any]:
    """检查生成图里是否有氨基酸骨架和肽键。

    这里不要求 20 种标准氨基酸，只看更底层的化学骨架：
    - peptide bond: C(=O)-N 酰胺键；
    - amino-acid-like residue: alpha carbon 同时连接 N 和 carbonyl C。
    """

    from rdkit import Chem

    info: dict[str, Any] = {
        "protein_like": False,
        "peptide_bond_count": 0,
        "amino_acid_backbone_like_count": 0,
        "carbonyl_c_count": 0,
    }
    try:
        mol, rdkit_info = graph_to_rdkit_mol(graph)
        info["rdkit_sanitize_ok"] = bool(rdkit_info.get("sanitize_ok"))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    carbonyl_c: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
                carbonyl_c.add(atom.GetIdx())
                break
    info["carbonyl_c_count"] = int(len(carbonyl_c))

    peptide_bonds: set[tuple[int, int]] = set()
    for cidx in carbonyl_c:
        atom = mol.GetAtomWithIdx(int(cidx))
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 7:
                continue
            bond = mol.GetBondBetweenAtoms(int(cidx), nbr.GetIdx())
            if bond is not None and bond.GetBondType() in {Chem.BondType.SINGLE, Chem.BondType.UNSPECIFIED}:
                peptide_bonds.add((int(cidx), int(nbr.GetIdx())))
    info["peptide_bond_count"] = int(len(peptide_bonds))

    alpha_carbons: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        has_n = False
        has_carbonyl = False
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 7:
                has_n = True
            if nbr.GetIdx() in carbonyl_c:
                has_carbonyl = True
        if has_n and has_carbonyl:
            alpha_carbons.add(atom.GetIdx())
    info["amino_acid_backbone_like_count"] = int(len(alpha_carbons))
    info["protein_like"] = bool(len(alpha_carbons) >= 2 and len(peptide_bonds) >= 1)
    return info


def load_model(ckpt_dir: Path, dtype_name: str) -> tuple[PackedXformersQwen3ForCausalLM, Qwen3Config, torch.device]:
    """加载训练好的 packed-xformers Qwen3 模型。"""

    cfg = Qwen3Config.from_pretrained(ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("generation must run on a GPU worker")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model = PackedXformersQwen3ForCausalLM(cfg)
    state = load_torch_state(ckpt_dir / "pytorch_model.bin")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg, device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--merges-csv", type=Path, default=DEFAULT_MERGES)
    ap.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--raw-fddfs-root", type=Path, default=DEFAULT_RAW_FDDFS_ROOT)
    ap.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--domains", nargs="+", default=["small_molecule", "peptide", "protein", "dna", "rna"])
    ap.add_argument("--samples-per-domain", type=int, default=3)
    ap.add_argument("--max-attempts-per-domain", type=int, default=12)
    ap.add_argument("--max-total-tokens", type=int, default=256)
    ap.add_argument("--min-total-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--token-offset", type=int, default=3)
    ap.add_argument("--split", default="train")
    ap.add_argument("--domain-start-pool-size", type=int, default=24)
    ap.add_argument("--domain-start-max-files", type=int, default=24)
    ap.add_argument("--domain-start-records-per-file", type=int, default=4)
    ap.add_argument("--domain-start-max-encode-len", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--no-rdkit-visualization", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    vocab = FDDFSVocab.load(args.vocab_json)
    decoder = BPEDecoder(args.merges_csv, vocab)
    model, cfg, device = load_model(args.ckpt_dir, args.dtype)
    pad_token_id = int(cfg.pad_token_id)
    bos_token_id = int(cfg.bos_token_id)
    eos_token_id = int(cfg.eos_token_id)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    graphs_dir = args.out_jsonl.parent / "graphs_npz"
    png_dir = args.out_jsonl.parent / "rdkit_png"
    summary: dict[str, Any] = {
        "ckpt_dir": str(args.ckpt_dir),
        "merges_csv": str(args.merges_csv),
        "vocab_json": str(args.vocab_json),
        "raw_fddfs_root": str(args.raw_fddfs_root),
        "out_jsonl": str(args.out_jsonl),
        "domains": list(args.domains),
        "samples_per_domain": int(args.samples_per_domain),
        "max_attempts_per_domain": int(args.max_attempts_per_domain),
        "max_total_tokens": int(args.max_total_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "dtype": args.dtype,
        "device": str(device),
        "model_vocab_size": int(cfg.vocab_size),
        "bpe_base_vocab_size": int(decoder.base_vocab_size),
        "bpe_max_token_id": int(decoder.max_token_id),
        "fddfs_default_edge": list(vocab.default_edge),
    }
    print(json_dumps({"event": "start", **summary}), flush=True)

    domain_stats: dict[str, dict[str, int]] = {}
    t0 = time.time()
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        out.write(json_dumps({"type": "run_config", **summary}) + "\n")
        for domain in args.domains:
            sampler = DomainStartSampler(
                args.raw_fddfs_root,
                domain,
                args.split,
                decoder,
                token_offset=args.token_offset,
                pool_size=args.domain_start_pool_size,
                max_files=args.domain_start_max_files,
                records_per_file=args.domain_start_records_per_file,
                max_encode_len=args.domain_start_max_encode_len,
                rng=rng,
            )
            stats = {
                "attempts": 0,
                "accepted": 0,
                "bpe_decode_ok": 0,
                "graph_decode_ok": 0,
                "rdkit_sanitize_ok": 0,
                "hit_eos": 0,
                "protein_like": 0,
            }
            domain_stats[domain] = stats
            for attempt_id in range(int(args.max_attempts_per_domain)):
                if stats["accepted"] >= int(args.samples_per_domain):
                    break
                start_token = sampler.sample(rng)
                record: dict[str, Any] = {
                    "type": "sample",
                    "domain": domain,
                    "attempt_id": int(attempt_id),
                    "start_token_id": int(start_token),
                }
                stats["attempts"] += 1
                try:
                    ids, hit_eos = generate_one(
                        model,
                        start_token,
                        device=device,
                        max_total_tokens=args.max_total_tokens,
                        min_total_tokens=args.min_total_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        pad_token_id=pad_token_id,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                    )
                    stats["hit_eos"] += int(hit_eos)
                    raw_ids = model_ids_to_raw_bpe_ids(ids, eos_token_id=eos_token_id, token_offset=args.token_offset)
                    base_ids, fddfs_tokens = decoder.decode_raw_ids(raw_ids)
                    stats["bpe_decode_ok"] += 1
                    graph = deserialize_fddfs(fddfs_tokens, vocab.default_edge)
                    stats["graph_decode_ok"] += 1
                    gsum = graph_summary(graph)
                    accepted_id = stats["accepted"]
                    graph_path = graphs_dir / domain / f"sample_{accepted_id:03d}_attempt_{attempt_id:03d}.npz"
                    save_graph_npz(graph_path, graph)
                    rdkit_info = None
                    if not args.no_rdkit_visualization:
                        rdkit_info = save_rdkit_png(
                            graph,
                            png_dir / domain / f"sample_{accepted_id:03d}_attempt_{attempt_id:03d}.png",
                        )
                        stats["rdkit_sanitize_ok"] += int(bool((rdkit_info or {}).get("sanitize_ok")))
                    protein_info = analyze_protein_like(graph) if domain == "protein" else None
                    if protein_info is not None:
                        stats["protein_like"] += int(bool(protein_info.get("protein_like")))
                    stats["accepted"] += 1
                    record.update(
                        {
                            "status": "ok",
                            "accepted_id": int(accepted_id),
                            "hit_eos": bool(hit_eos),
                            "model_token_len": len(ids),
                            "raw_bpe_len": len(raw_ids),
                            "fddfs_token_len": len(fddfs_tokens),
                            "model_token_ids": ids,
                            "raw_bpe_ids": raw_ids,
                            "fddfs_tokens_prefix": fddfs_tokens[:160],
                            "fddfs_tokens_suffix": fddfs_tokens[-40:],
                            "graph_path": str(graph_path),
                            "graph": gsum,
                            "rdkit": rdkit_info,
                            "protein_check": protein_info,
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                out.write(json_dumps(record) + "\n")
                out.flush()
                print(
                    json_dumps(
                        {
                            "event": "sample_done",
                            "domain": domain,
                            "attempt_id": attempt_id,
                            "status": record["status"],
                            "accepted": stats["accepted"],
                            "hit_eos": record.get("hit_eos"),
                            "model_token_len": record.get("model_token_len"),
                            "fddfs_token_len": record.get("fddfs_token_len"),
                            "num_nodes": (record.get("graph") or {}).get("num_nodes"),
                            "rdkit_sanitize_ok": (record.get("rdkit") or {}).get("sanitize_ok") if record.get("rdkit") else None,
                            "protein_like": (record.get("protein_check") or {}).get("protein_like") if record.get("protein_check") else None,
                            "error": record.get("error"),
                        }
                    ),
                    flush=True,
                )

    done = {
        "event": "done",
        "elapsed_sec": round(time.time() - t0, 3),
        "domain_stats": domain_stats,
        "out_jsonl": str(args.out_jsonl),
        "graphs_dir": str(graphs_dir),
        "png_dir": str(png_dir),
    }
    summary_path = args.out_jsonl.parent / "summary.json"
    summary_path.write_text(json.dumps(done, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json_dumps(done), flush=True)


if __name__ == "__main__":
    main()
