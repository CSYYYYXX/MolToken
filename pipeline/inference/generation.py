"""Representation-neutral helpers for Qwen generation and graph inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a state dict across supported torch versions."""

    return torch.load(path, map_location="cpu", weights_only=True)


def json_dumps(value: Any) -> str:
    """Return compact UTF-8 JSON for logs and JSONL output."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a compact summary of one heavy-atom graph."""

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
        "atomic_nums": (
            sorted({int(value) for value in node_features[:, 0].tolist()})
            if len(node_features)
            else []
        ),
        "bond_types": (
            sorted({int(value) for value in edge_features[:, 0].tolist()})
            if len(edge_features)
            else []
        ),
    }


def save_graph_npz(path: Path, graph: dict[str, Any]) -> None:
    """Save a decoded graph for later inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    src, dst = graph["edges"]
    np.savez_compressed(
        path,
        num_nodes=np.asarray([int(graph["num_nodes"])], dtype=np.int64),
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        node_features=np.asarray(graph["node_features"], dtype=np.int64),
        edge_features=np.asarray(graph["edge_features"], dtype=np.int64),
        node_type_ids=np.asarray(graph["node_type_ids"], dtype=np.int64),
        edge_type_ids=np.asarray(graph["edge_type_ids"], dtype=np.int64),
    )


def graph_to_rdkit_mol(graph: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Build an RDKit molecule and report sanitization separately."""

    from rdkit import Chem

    node_features = np.asarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
    edge_features = np.asarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
    src, dst = graph["edges"]
    bond_type_map = {
        0: Chem.BondType.UNSPECIFIED,
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for feature in node_features:
        atom = Chem.Atom(int(feature[0]))
        atom.SetFormalCharge(int(feature[1]))
        atom.SetIsAromatic(bool(int(feature[2])))
        chiral_tag = int(feature[3])
        if 0 <= chiral_tag < len(Chem.ChiralType.values):
            atom.SetChiralTag(Chem.ChiralType.values[chiral_tag])
        else:
            raise ValueError(f"unknown chiral tag id: {chiral_tag}")
        rw.AddAtom(atom)

    seen: set[tuple[int, int]] = set()
    skipped_self_edges = 0
    for u, v, feature in zip(src, dst, edge_features, strict=True):
        u, v = int(u), int(v)
        if u == v:
            skipped_self_edges += 1
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        bond_type_id = int(feature[0])
        if bond_type_id not in bond_type_map:
            raise ValueError(f"unknown bond type id: {bond_type_id}")
        is_aromatic = bool(int(feature[1]))
        bond_type = (
            Chem.BondType.AROMATIC if is_aromatic else bond_type_map[bond_type_id]
        )
        rw.AddBond(u, v, bond_type)
        bond = rw.GetBondBetweenAtoms(u, v)
        if bond is not None:
            bond.SetIsAromatic(is_aromatic or bond_type == Chem.BondType.AROMATIC)
            stereo_id = int(feature[2])
            if 0 <= stereo_id < len(Chem.BondStereo.values):
                bond.SetStereo(Chem.BondStereo.values[stereo_id])
            else:
                raise ValueError(f"unknown bond stereo id: {stereo_id}")

    mol = rw.GetMol()
    info: dict[str, Any] = {
        "mol_build_ok": True,
        "sanitize_ok": False,
        "skipped_self_edges": int(skipped_self_edges),
        "unique_bonds_added": int(len(seen)),
    }
    try:
        Chem.SanitizeMol(mol)
        info["sanitize_ok"] = True
    except Exception as exc:
        info["sanitize_error"] = f"{type(exc).__name__}: {exc}"
        mol.UpdatePropertyCache(strict=False)
    return mol, info


def save_rdkit_png(graph: dict[str, Any], png_path: Path) -> dict[str, Any]:
    """Save a 2D rendering and return the per-sample drawing result."""

    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor

    png_path.parent.mkdir(parents=True, exist_ok=True)
    mol, build_info = graph_to_rdkit_mol(graph)
    rdDepictor.Compute2DCoords(mol)
    Draw.MolToFile(mol, str(png_path), size=(420, 320), kekulize=False)
    Chem.MolToMolFile(mol, str(png_path.with_suffix(".mol")))
    return {
        "png_path": str(png_path),
        "draw_ok": png_path.exists(),
        "mol_path": str(png_path.with_suffix(".mol")),
        **build_info,
    }


def top_k_top_p_filter(
    logits: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to one logit vector."""

    logits = logits.clone()
    if 0 < top_k < logits.numel():
        threshold = torch.topk(logits, k=top_k).values[-1]
        logits[logits < threshold] = -torch.inf
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(probabilities, dim=-1) > top_p
        if remove.numel() > 0:
            remove[1:] = remove[:-1].clone()
            remove[0] = False
        logits[sorted_indices[remove]] = -torch.inf
    return logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    forbid_ids: set[int],
) -> int:
    """Sample one token from the final-position logits."""

    logits = logits.float().clone()
    for token_id in forbid_ids:
        if 0 <= token_id < logits.numel():
            logits[token_id] = -torch.inf
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    logits = top_k_top_p_filter(
        logits / float(temperature), top_k=top_k, top_p=top_p
    )
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0.0:
        raise RuntimeError("invalid sampling probabilities after filtering")
    return int(torch.multinomial(probabilities, num_samples=1).item())


def generate_one(
    model: Any,
    start_token: int,
    *,
    device: torch.device,
    max_total_tokens: int,
    min_total_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> tuple[list[int], bool]:
    """Autoregressively generate one sequence from its initial token."""

    ids = [int(start_token)]
    for _ in range(max(0, max_total_tokens - 1)):
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        forbid = {pad_token_id, bos_token_id}
        if len(ids) < min_total_tokens:
            forbid.add(eos_token_id)
        next_id = sample_next_token(
            output.logits[0, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbid_ids=forbid,
        )
        ids.append(next_id)
        if next_id == eos_token_id:
            return ids, True
    return ids, False


def model_ids_to_raw_bpe_ids(
    ids: list[int], eos_token_id: int, token_offset: int
) -> list[int]:
    """Strip EOS and remove the model-token offset."""

    raw_ids: list[int] = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == eos_token_id:
            break
        raw_id = token_id - token_offset
        if raw_id < 0:
            raise ValueError(
                f"special token {token_id} cannot be converted to raw BPE id"
            )
        raw_ids.append(raw_id)
    return raw_ids
