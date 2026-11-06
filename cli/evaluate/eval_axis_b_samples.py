#!/usr/bin/env python3
"""Decode and validate molecular samples emitted by sample_axis_b_sft.py."""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.serialization.fddfs import deserialize_fddfs  # noqa: E402


DATA_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / '')
)
DEFAULT_MERGES = (
    _moltoken_paths.TOKENIZER_ASSET_ROOT / "merges.csv"
)
DEFAULT_VOCAB = _moltoken_paths.TOKENIZER_ASSET_ROOT / "vocab.json"


@dataclass
class FDDFSVocab:
    """Base FDDFS vocabulary used by the project BPE codebook."""

    id_to_token: dict[int, str]
    ring_base: int
    ring_min: int
    default_edge: tuple[int, ...]
    extra_id_to_token: dict[int, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "FDDFSVocab":
        payload = json.loads(path.read_text(encoding="utf-8"))
        token_to_id = payload["token_to_id"]
        serializer = str(payload.get("serializer", "")).lower()
        if serializer and serializer != "fddfs":
            raise ValueError(f"vocab serializer is not fddfs: {serializer}")
        return cls(
            id_to_token={int(value): str(token) for token, value in token_to_id.items()},
            ring_base=int(payload["ring_base"]),
            ring_min=int(payload["ring_min"]),
            default_edge=tuple(int(value) for value in payload["default_edge_feature"]),
        )

    def token_for_base_id(self, base_id: int, base_vocab_size: int) -> str:
        base_id = int(base_id)
        if base_id in self.id_to_token:
            return self.id_to_token[base_id]
        if base_id in self.extra_id_to_token:
            return self.extra_id_to_token[base_id]
        if self.ring_base <= base_id < int(base_vocab_size):
            return f"R{base_id - self.ring_base + self.ring_min}"
        raise ValueError(f"base id {base_id} is outside FDDFS vocab size {base_vocab_size}")


class BPEDecoder:
    """Expand raw merge ids into FDDFS base ids and tokens."""

    def __init__(self, merges_csv: Path, vocab: FDDFSVocab):
        self.vocab = vocab
        self.merge_map: dict[int, tuple[int, int]] = {}
        min_new_id: int | None = None
        with merges_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                left, right, new_id = int(row["left"]), int(row["right"]), int(row["new_id"])
                self.merge_map[new_id] = (left, right)
                min_new_id = new_id if min_new_id is None else min(min_new_id, new_id)
        if min_new_id is None:
            raise ValueError(f"empty merges csv: {merges_csv}")
        self.base_vocab_size = int(min_new_id)

    @lru_cache(maxsize=None)
    def expand_id(self, raw_id: int) -> tuple[int, ...]:
        out: list[int] = []
        stack = [int(raw_id)]
        while stack:
            current = int(stack.pop())
            if current in self.vocab.extra_id_to_token or current < self.base_vocab_size:
                out.append(current)
                continue
            if current not in self.merge_map:
                raise ValueError(f"raw BPE id {current} is not in merge map")
            left, right = self.merge_map[current]
            stack.extend((right, left))
        return tuple(out)

    def decode_raw_ids(self, raw_ids: list[int]) -> tuple[list[int], list[str]]:
        base_ids: list[int] = []
        for raw_id in raw_ids:
            base_ids.extend(self.expand_id(int(raw_id)))
        return base_ids, [
            self.vocab.token_for_base_id(value, self.base_vocab_size) for value in base_ids
        ]


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    node_features = np.asarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
    edge_features = np.asarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
    src, _dst = graph["edges"]
    directed_edges = int(len(src))
    return {
        "num_nodes": int(graph["num_nodes"]),
        "num_directed_edges": directed_edges,
        "num_undirected_edges": directed_edges // 2,
        "node_feature_shape": list(node_features.shape),
        "edge_feature_shape": list(edge_features.shape),
        "atomic_nums": sorted({int(x) for x in node_features[:, 0].tolist()}) if len(node_features) else [],
        "bond_types": sorted({int(x) for x in edge_features[:, 0].tolist()}) if len(edge_features) else [],
    }


def graph_to_rdkit_mol(
    graph: dict[str, Any], *, strict: bool = True,
    aromatic_n_h_completion: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from rdkit import Chem

    node_features = np.asarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
    edge_features = np.asarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
    src, dst = graph["edges"]
    if len(src) != len(dst) or len(src) != len(edge_features):
        raise ValueError("edge arrays have inconsistent lengths")
    if int(graph.get("num_nodes", len(node_features))) != len(node_features):
        raise ValueError("num_nodes disagrees with node_features")
    bond_type_map = {
        0: Chem.BondType.UNSPECIFIED,
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for feat in node_features:
        atom = Chem.Atom(int(feat[0]))
        atom.SetFormalCharge(int(feat[1]))
        atom.SetIsAromatic(bool(int(feat[2])))
        try:
            atom.SetChiralTag(Chem.ChiralType.values[int(feat[3])])
        except Exception as exc:
            if strict:
                raise ValueError(f"unknown chiral tag id: {int(feat[3])}") from exc
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        rw.AddAtom(atom)
    directed_seen: set[tuple[int, int]] = set()
    undirected_counts: Counter[tuple[int, int]] = Counter()
    undirected_features: dict[tuple[int, int], tuple[int, int, int]] = {}
    skipped_self_edges = 0
    for u, v, feat in zip(src, dst, edge_features, strict=False):
        u, v = int(u), int(v)
        if u == v:
            if strict:
                raise ValueError(f"self-edge is not valid in strict graph check: {u}")
            skipped_self_edges += 1
            continue
        if not 0 <= u < len(node_features) or not 0 <= v < len(node_features):
            raise ValueError(f"edge endpoint out of range: {(u, v)}")
        directed_key = (u, v)
        if strict and directed_key in directed_seen:
            raise ValueError(f"duplicate directed edge: {(u, v)}")
        directed_seen.add(directed_key)
        key = (u, v) if u < v else (v, u)
        undirected_counts[key] += 1
        edge_feature = tuple(int(value) for value in feat.tolist())
        previous_feature = undirected_features.setdefault(key, edge_feature)
        if strict and previous_feature != edge_feature:
            raise ValueError(
                f"reverse edge feature mismatch for {key}: "
                f"{previous_feature} != {edge_feature}"
            )
        aromatic = bool(int(feat[1]))
        if int(feat[0]) not in bond_type_map:
            raise ValueError(f"unknown bond type id: {int(feat[0])}")
        bond_type = Chem.BondType.AROMATIC if aromatic else bond_type_map[int(feat[0])]
        if rw.GetBondBetweenAtoms(u, v) is None:
            rw.AddBond(u, v, bond_type)
            bond = rw.GetBondBetweenAtoms(u, v)
            if bond is not None:
                bond.SetIsAromatic(aromatic or bond_type == Chem.BondType.AROMATIC)
                try:
                    bond.SetStereo(Chem.BondStereo.values[int(feat[2])])
                except Exception as exc:
                    if strict:
                        raise ValueError(f"unknown bond stereo id: {int(feat[2])}") from exc
    if strict:
        malformed = {key: count for key, count in undirected_counts.items() if count != 2}
        if malformed:
            raise ValueError(f"undirected edges are not represented twice: {malformed}")
    mol = rw.GetMol()
    info: dict[str, Any] = {
        "mol_build_ok": True,
        "sanitize_ok": False,
        "skipped_self_edges": int(skipped_self_edges),
        "unique_bonds_added": int(len(undirected_counts)),
        "strict": bool(strict),
    }
    try:
        Chem.SanitizeMol(mol)
        info["sanitize_ok"] = True
    except Exception as exc:
        info["sanitize_error"] = f"{type(exc).__name__}: {exc}"
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception as cache_exc:
            info["property_cache_error"] = f"{type(cache_exc).__name__}: {cache_exc}"
    if aromatic_n_h_completion and not info["sanitize_ok"]:
        from pipeline.graph.aromatic_n_repair import (
            compare_molecule_semantics,
            complete_aromatic_n_h,
            materialize_aromatic_n_completion,
        )
        from pipeline.graph.chem_graph_builder import mol_to_graph_data

        completion = complete_aromatic_n_h(graph)
        audit = completion.as_dict()
        audit["accepted"] = False
        info["aromatic_n_h_completion"] = audit
        if completion.repaired:
            selected = next(item for item in completion.completions
                            if item.canonical_smiles == completion.selected_smiles)
            source = materialize_aromatic_n_completion(graph, selected.atom_indices)
            reparsed = Chem.MolFromSmiles(selected.canonical_smiles)
            if reparsed is None or reparsed.GetNumAtoms() == 0:
                audit["rejection_reason"] = "canonical_reparse_failed"
            else:
                comparison = compare_molecule_semantics(
                    source, reparsed, left_heavy_graph=graph,
                    right_heavy_graph=mol_to_graph_data(
                        reparsed, sample_id="aromatic_n_completion_reparse"),
                )
                audit["canonical_roundtrip"] = comparison.as_dict()
                if comparison.equivalent:
                    mol = reparsed
                    info["sanitize_ok"] = True
                    audit["accepted"] = True
                else:
                    audit["rejection_reason"] = comparison.reason
    return mol, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--merges-csv", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def decode_graph(
    raw_ids: list[int], decoder: BPEDecoder, vocab: FDDFSVocab
) -> tuple[dict[str, Any], list[str]]:
    _base_ids, tokens = decoder.decode_raw_ids(raw_ids)
    graph = deserialize_fddfs(tokens, vocab.default_edge)
    return graph, tokens


def main() -> int:
    args = parse_args()
    vocab = FDDFSVocab.load(args.vocab_json)
    decoder = BPEDecoder(args.merges_csv, vocab)
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            counts["records"] += 1
            record = json.loads(line)
            generated = [int(value) for value in record.get("generated_bpe_ids", [])]
            target = [int(value) for value in record.get("target_bpe_ids", [])]
            result: dict[str, Any] = {
                "id": record.get("id"),
                "task": record.get("task"),
                "ended_with_eos": bool(record.get("ended_with_eos")),
                "invalid_model_id": bool(record.get("invalid_model_id")),
                "target_exceeds_max_new_tokens": bool(
                    record.get("target_exceeds_max_new_tokens", False)
                ),
                "target_truncated": bool(record.get("target_truncated", False)),
                "token_exact_comparable": bool(
                    record.get("token_exact_comparable", True)
                ) and not bool(record.get("target_truncated", False)),
            }
            complete_output = bool(
                result["ended_with_eos"]
                and not result["invalid_model_id"]
                and bool(generated)
            )
            result["complete_output"] = complete_output
            counts["ended_with_eos"] += int(result["ended_with_eos"])
            counts["invalid_model_id"] += int(result["invalid_model_id"])
            counts["complete_output"] += int(complete_output)
            counts["token_exact_comparable"] += int(
                result["token_exact_comparable"]
            )
            try:
                if not generated:
                    raise ValueError("empty_generation")
                _generated_base_ids, generated_tokens = decoder.decode_raw_ids(
                    generated
                )
                counts["diagnostic_bpe_decode_ok"] += 1
                generated_graph = deserialize_fddfs(
                    generated_tokens, vocab.default_edge
                )
                counts["diagnostic_graph_decode_ok"] += 1
                if complete_output:
                    counts["bpe_decode_ok"] += 1
                    counts["graph_decode_ok"] += 1
                result["generated_graph"] = graph_summary(generated_graph)
                result["generated_fddfs_tokens"] = len(generated_tokens)
            except Exception as exc:  # noqa: BLE001
                key = f"generated:{type(exc).__name__}:{str(exc)[:160]}"
                failures[key] += 1
                result["generated_error"] = key
                generated_graph = None

            try:
                target_graph, _target_tokens = decode_graph(target, decoder, vocab)
                counts["target_graph_decode_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                key = f"target:{type(exc).__name__}:{str(exc)[:160]}"
                failures[key] += 1
                result["target_error"] = key
                target_graph = None

            if generated_graph is not None:
                try:
                    generated_mol, generated_info = graph_to_rdkit_mol(generated_graph)
                    result["rdkit"] = generated_info
                    sanitize_ok = bool(generated_info.get("sanitize_ok"))
                    counts["rdkit_build_ok"] += int(
                        complete_output and bool(generated_info.get("mol_build_ok"))
                    )
                    counts["rdkit_sanitize_ok"] += int(complete_output and sanitize_ok)
                    generated_smiles = None
                    if sanitize_ok:
                        from rdkit import Chem

                        generated_smiles = Chem.MolToSmiles(
                            generated_mol, canonical=True, isomericSmiles=True
                        )
                        result["generated_smiles"] = generated_smiles
                except Exception as exc:  # noqa: BLE001
                    key = f"rdkit:{type(exc).__name__}:{str(exc)[:160]}"
                    failures[key] += 1
                    result["rdkit_error"] = key
                    generated_smiles = None
            else:
                generated_smiles = None

            if target_graph is not None:
                try:
                    target_mol, target_info = graph_to_rdkit_mol(target_graph)
                    counts["target_rdkit_build_ok"] += int(
                        bool(target_info.get("mol_build_ok"))
                    )
                    counts["target_rdkit_sanitize_ok"] += int(
                        bool(target_info.get("sanitize_ok"))
                    )
                    result["target_rdkit"] = target_info
                    if bool(target_info.get("sanitize_ok")):
                        from rdkit import Chem

                        target_smiles = Chem.MolToSmiles(
                            target_mol, canonical=True, isomericSmiles=True
                        )
                        result["target_smiles"] = target_smiles
                    else:
                        target_smiles = None
                except Exception as exc:  # noqa: BLE001
                    key = f"target_rdkit:{type(exc).__name__}:{str(exc)[:160]}"
                    failures[key] += 1
                    result["target_rdkit_error"] = key
                    target_smiles = None
            else:
                target_smiles = None

            target_summary = graph_summary(target_graph) if target_graph is not None else None
            generated_summary = (
                graph_summary(generated_graph) if generated_graph is not None else None
            )
            structure_match_comparable = bool(
                complete_output and not result["target_exceeds_max_new_tokens"]
                and not result["target_truncated"]
            )
            result["structure_match_comparable"] = structure_match_comparable
            counts["structure_match_comparable"] += int(structure_match_comparable)
            graph_shape_match = bool(
                structure_match_comparable
                and generated_summary is not None
                and target_summary is not None
                and generated_summary["num_nodes"] == target_summary["num_nodes"]
                and generated_summary["num_directed_edges"]
                == target_summary["num_directed_edges"]
            )
            result["graph_shape_match"] = graph_shape_match
            counts["graph_shape_match"] += int(graph_shape_match)

            canonical_comparable = bool(
                structure_match_comparable
                and
                generated_smiles is not None
                and target_smiles is not None
            )
            canonical_exact = bool(canonical_comparable and generated_smiles == target_smiles)
            result["reconstructed_canonical_smiles_comparable"] = canonical_comparable
            result["reconstructed_canonical_smiles_match"] = canonical_exact
            counts["reconstructed_canonical_smiles_comparable"] += int(
                canonical_comparable
            )
            counts["reconstructed_canonical_smiles_match"] += int(canonical_exact)
            output.write(json.dumps(result, separators=(",", ":")) + "\n")

            if (
                len(examples) < max(0, int(args.examples))
                and (
                    result.get("generated_error")
                    or result.get("rdkit_error")
                    or result.get("target_error")
                    or result.get("target_rdkit_error")
                )
            ):
                examples.append({"line": line_number, **result})

    records = int(counts["records"])
    rate_keys = (
        "ended_with_eos",
        "invalid_model_id",
        "complete_output",
        "token_exact_comparable",
        "structure_match_comparable",
        "bpe_decode_ok",
        "graph_decode_ok",
        "rdkit_build_ok",
        "rdkit_sanitize_ok",
        "target_graph_decode_ok",
        "target_rdkit_build_ok",
        "target_rdkit_sanitize_ok",
        "graph_shape_match",
        "reconstructed_canonical_smiles_comparable",
        "reconstructed_canonical_smiles_match",
    )
    summary: dict[str, Any] = {
        "samples": str(args.samples),
        "records": records,
        "counts": dict(counts),
        "rates": {
            key: (float(counts[key]) / records if records else None)
            for key in rate_keys
        },
        "top_failures": failures.most_common(20),
        "failure_examples": examples,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
