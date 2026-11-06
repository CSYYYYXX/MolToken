#!/usr/bin/env python3
"""heavy atom graph -> RDKit mol -> SMILES（pipeline 正向 mol_to_graph_data 的逆）。

正向布局（见 pipeline/graph/chem_graph_builder.py::mol_to_graph_data）：

    node_features (6 列) = [atomic_num, formal_charge, is_aromatic,
                            chiral_tag, degree, total_valence]
    edge_features (3 列) = [bond_type_id, aromatic, stereo]
    edges                = (src, dst)，**双向各存一次**

bond_type_id 映射（已由 roundtrip 实测确认）：1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC

逆变换只需 node_features 的 0..3 列：degree/total_valence 是派生量，
由 RDKit 依价规则自行推断隐式氢。

已知信息损失（表示层面，非本模块 bug，均由 roundtrip 实测确认）：

  1. 双键 E/Z 立体：edge_features 只存 stereo 枚举值，不存 stereo atom
     参考原子，故 ``C/C=C/C`` 只能还原成 ``CC=CC``。
  2. 芳香 N-H 杂环（imidazole/pyrrole/indole/腺嘌呤...）：芳香 N 上的
     氢数未被存储，kekulize 时无法定位，抛 KekulizeException。
     **这是正确行为** —— 静默输出饱和环（如 ``C1CNCN1``）会给出化学上
     不同的分子，比报错危险得多。
  3. 手性在经 FDDFS 重排后可能翻转：chiral_tag 是相对邻居顺序定义的，
     而 FDDFS 会重排节点顺序，pipeline 存的是原始 tag、未随置换调整。

因此 SMILES 层面的严格 canonical 相等**不适合**作为 validity 判据；
应使用 graph_signature 比对拓扑（pipeline 自带 verify_roundtrip 即如此）。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# bond_type_id -> RDKit BondType（映射经 roundtrip 实测确认）
_BOND_ID_TO_TYPE: dict[int, Any] = {}


def _bond_table() -> dict[int, Any]:
    global _BOND_ID_TO_TYPE
    if _BOND_ID_TO_TYPE:
        return _BOND_ID_TO_TYPE
    from rdkit import Chem

    _BOND_ID_TO_TYPE = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    return _BOND_ID_TO_TYPE


def graph_to_mol(graph: dict[str, Any], *, sanitize: bool = True) -> Any:
    """把 heavy atom graph 字典重建成 RDKit mol。

    sanitize=False 时只重建拓扑、不做价检查，用于把 validity 拆成
    「图有效」与「化学有效」两层。
    """
    from rdkit import Chem

    nf = np.asarray(graph["node_features"], dtype=np.int64).reshape((-1, 6))
    num_nodes = int(graph.get("num_nodes", nf.shape[0]))
    if nf.shape[0] != num_nodes:
        raise ValueError(f"node_features rows {nf.shape[0]} != num_nodes {num_nodes}")

    rw = Chem.RWMol()
    for row in nf:
        atomic_num = int(row[0])
        if atomic_num <= 1:
            raise ValueError(f"invalid heavy-atom atomic_num={atomic_num}")
        atom = Chem.Atom(atomic_num)
        atom.SetFormalCharge(int(row[1]))
        atom.SetIsAromatic(bool(int(row[2])))
        tag = int(row[3])
        if tag == 1:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        elif tag == 2:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        rw.AddAtom(atom)

    src, dst = graph["edges"]
    src = [int(x) for x in np.asarray(src, dtype=np.int64).reshape((-1,)).tolist()]
    dst = [int(x) for x in np.asarray(dst, dtype=np.int64).reshape((-1,)).tolist()]
    ef = np.asarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3))
    if not (len(src) == len(dst) == ef.shape[0]):
        raise ValueError(f"edge arrays disagree: src={len(src)} dst={len(dst)} feat={ef.shape[0]}")

    table = _bond_table()
    seen: set[tuple[int, int]] = set()
    for k in range(len(src)):
        u, v = src[k], dst[k]
        if u == v:
            raise ValueError(f"self-loop at node {u}")
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"edge endpoint out of range: ({u},{v}) num_nodes={num_nodes}")
        key = (u, v) if u < v else (v, u)
        if key in seen:  # 双向存储 -> 只建一次
            continue
        seen.add(key)
        bt_id = int(ef[k][0])
        bt = table.get(bt_id)
        if bt is None:
            raise ValueError(f"unknown bond_type_id={bt_id}")
        rw.AddBond(key[0], key[1], bt)
        if int(ef[k][1]):
            rw.GetBondBetweenAtoms(key[0], key[1]).SetIsAromatic(True)

    mol = rw.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)  # 失败即抛，由调用方判为化学无效
    return mol


def graph_to_smiles(graph: dict[str, Any], *, canonical: bool = True) -> str:
    """heavy atom graph -> canonical SMILES。sanitize 失败会抛异常。"""
    from rdkit import Chem

    mol = graph_to_mol(graph, sanitize=True)
    return Chem.MolToSmiles(mol, canonical=canonical)


__all__ = ["graph_to_mol", "graph_to_smiles"]
