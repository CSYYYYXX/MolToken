from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

BOND_TYPE_TO_ID = {
    "UNSPECIFIED": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_DNA = set("ACGT")
STANDARD_RNA = set("ACGU")
STANDARD_DNA_ORDER = "ACGT"
STANDARD_RNA_ORDER = "ACGU"
NUCLEIC_ACID_FASTA_FLAVORS = {"dna": 6, "rna": 2}


def is_standard_protein_sequence(sequence: str) -> bool:
    return bool(sequence) and all(ch in STANDARD_AA for ch in sequence.upper())


def is_standard_nucleotide_sequence(sequence: str, target_type: str) -> bool:
    alphabet = STANDARD_DNA if target_type == "dna" else STANDARD_RNA if target_type == "rna" else set()
    return bool(sequence) and all(ch in alphabet for ch in sequence.upper())


def _directed_edge_features(bond_type: int, aromatic: int, stereo: int) -> list[int]:
    return [int(bond_type), int(aromatic), int(stereo)]


def _bond_type_id(bond: Any) -> int:
    name = str(bond.GetBondType()).upper()
    if bond.GetIsAromatic():
        return BOND_TYPE_TO_ID["AROMATIC"]
    return BOND_TYPE_TO_ID.get(name, 0)


def mol_to_graph_data(mol: Any, sample_id: str, properties: Optional[Dict[str, Any]] = None,
                      sequence: Optional[str] = None, smiles: Optional[str] = None) -> Dict[str, Any]:
    from rdkit import Chem

    mol = Chem.RemoveHs(mol, sanitize=False)
    Chem.SanitizeMol(mol)

    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    old_to_new = {atom.GetIdx(): i for i, atom in enumerate(atoms)}
    src = []
    dst = []
    edge_type_ids = []
    edge_features = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in old_to_new or end not in old_to_new:
            continue
        bond_type = _bond_type_id(bond)
        stereo = int(bond.GetStereo())
        aromatic = int(bond.GetIsAromatic())
        for u, v in ((old_to_new[begin], old_to_new[end]), (old_to_new[end], old_to_new[begin])):
            src.append(u)
            dst.append(v)
            edge_type_ids.append(bond_type)
            edge_features.append([bond_type, aromatic, stereo])

    node_type_ids = []
    node_features = []
    for atom in atoms:
        atomic_num = int(atom.GetAtomicNum())
        node_type_ids.append(atomic_num)
        node_features.append([
            atomic_num,
            int(atom.GetFormalCharge()),
            int(atom.GetIsAromatic()),
            int(atom.GetChiralTag()),
            int(atom.GetDegree()),
            int(atom.GetTotalValence()),
        ])

    node_type_arr = np.asarray(node_type_ids, dtype=np.int64)
    edge_type_arr = np.asarray(edge_type_ids, dtype=np.int64)
    graph_data: Dict[str, Any] = {
        "sample_id": sample_id,
        "num_nodes": int(len(node_type_ids)),
        "edges": (src, dst),
        "node_features": np.asarray(node_features, dtype=np.int64).reshape((-1, 6)),
        "edge_features": np.asarray(edge_features, dtype=np.int64).reshape((-1, 3)),
        "node_type_ids": node_type_arr,
        "edge_type_ids": edge_type_arr,
        "node_token_ids": (node_type_arr * 2 + 1).reshape((-1, 1)),
        "edge_token_ids": (edge_type_arr * 2).reshape((-1, 1)),
        "properties": properties or {},
    }
    if sequence is not None:
        graph_data["sequence"] = sequence
    if smiles is not None:
        graph_data["smiles"] = smiles
    return graph_data


def _mol_heavy_graph_no_cleanup(mol: Any) -> Dict[str, Any]:
    """直接从 RDKit mol 读 heavy atom graph，不额外 RemoveHs/Sanitize。"""

    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    old_to_new = {atom.GetIdx(): i for i, atom in enumerate(atoms)}
    src: list[int] = []
    dst: list[int] = []
    edge_type_ids: list[int] = []
    edge_features: list[list[int]] = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in old_to_new or end not in old_to_new:
            continue
        bond_type = _bond_type_id(bond)
        stereo = int(bond.GetStereo())
        aromatic = int(bond.GetIsAromatic())
        for u, v in ((old_to_new[begin], old_to_new[end]), (old_to_new[end], old_to_new[begin])):
            src.append(u)
            dst.append(v)
            edge_type_ids.append(bond_type)
            edge_features.append(_directed_edge_features(bond_type, aromatic, stereo))

    node_type_ids: list[int] = []
    node_features: list[list[int]] = []
    for atom in atoms:
        atomic_num = int(atom.GetAtomicNum())
        node_type_ids.append(atomic_num)
        node_features.append(
            [
                atomic_num,
                int(atom.GetFormalCharge()),
                int(atom.GetIsAromatic()),
                int(atom.GetChiralTag()),
                int(atom.GetDegree()),
                int(atom.GetTotalValence()),
            ]
        )
    return {
        "num_nodes": int(len(node_type_ids)),
        "edges": (src, dst),
        "node_features": np.asarray(node_features, dtype=np.int64).reshape((-1, 6)),
        "edge_features": np.asarray(edge_features, dtype=np.int64).reshape((-1, 3)),
        "node_type_ids": np.asarray(node_type_ids, dtype=np.int64),
        "edge_type_ids": np.asarray(edge_type_ids, dtype=np.int64),
    }


@lru_cache(maxsize=1)
def _standard_protein_templates() -> dict[str, Any]:
    """用 RDKit 短肽生成标准氨基酸模板，供长蛋白快速拼图使用。"""

    from rdkit import Chem

    body_counts: dict[str, int] = {}
    first_templates: dict[str, dict[str, Any]] = {}
    internal_templates: dict[str, dict[str, Any]] = {}

    for aa in STANDARD_AA_ORDER:
        mol = Chem.MolFromFASTA(aa, flavor=0)
        if mol is None:
            raise ValueError(f"RDKit failed to build amino-acid template: {aa}")
        graph = _mol_heavy_graph_no_cleanup(mol)
        # RDKit appends the terminal OXT atom as the last heavy atom for a single residue.
        body_count = int(graph["num_nodes"]) - 1
        if body_count <= 0:
            raise ValueError(f"bad amino-acid template size for {aa}: {graph['num_nodes']}")
        body_counts[aa] = body_count
        src_raw, dst_raw = graph["edges"]
        keep_edges = [
            idx
            for idx, (u, v) in enumerate(zip(src_raw, dst_raw))
            if int(u) < body_count and int(v) < body_count
        ]
        first_templates[aa] = {
            "node_features": graph["node_features"][:body_count].copy(),
            "edge_src": [int(src_raw[idx]) for idx in keep_edges],
            "edge_dst": [int(dst_raw[idx]) for idx in keep_edges],
            "edge_features": graph["edge_features"][keep_edges].copy(),
        }

    gly_body = body_counts["G"]
    for aa in STANDARD_AA_ORDER:
        mol = Chem.MolFromFASTA(f"G{aa}G", flavor=0)
        if mol is None:
            raise ValueError(f"RDKit failed to build internal amino-acid template: {aa}")
        graph = _mol_heavy_graph_no_cleanup(mol)
        start = gly_body
        end = start + body_counts[aa]
        src_raw, dst_raw = graph["edges"]
        keep_edges = [
            idx
            for idx, (u, v) in enumerate(zip(src_raw, dst_raw))
            if start <= int(u) < end and start <= int(v) < end
        ]
        internal_templates[aa] = {
            "node_features": graph["node_features"][start:end].copy(),
            "edge_src": [int(src_raw[idx]) - start for idx in keep_edges],
            "edge_dst": [int(dst_raw[idx]) - start for idx in keep_edges],
            "edge_features": graph["edge_features"][keep_edges].copy(),
        }

    return {
        "body_counts": body_counts,
        "first": first_templates,
        "internal": internal_templates,
        "terminal_o_feature": np.asarray([[8, 0, 0, 0, 1, 2]], dtype=np.int64),
    }


def protein_sequence_to_graph_data_fast(
    sequence: str,
    sample_id: str,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """用标准氨基酸模板直接构建 protein/peptide heavy atom graph。

    该路径只覆盖 20 种标准氨基酸。模板来自 RDKit ``MolFromFASTA`` 的短肽输出；
    对非标准字符仍应回退到 RDKit 全分子构建路径。
    """

    seq = sequence.strip().upper()
    if not is_standard_protein_sequence(seq):
        raise ValueError(f"non-standard protein sequence cannot use fast template path: {sample_id}")
    templates = _standard_protein_templates()
    src: list[int] = []
    dst: list[int] = []
    node_features: list[list[int]] = []
    edge_features: list[list[int]] = []
    offsets: list[int] = []

    cursor = 0
    for idx, aa in enumerate(seq):
        template = templates["first"][aa] if idx == 0 else templates["internal"][aa]
        offsets.append(cursor)
        local_nodes = np.asarray(template["node_features"], dtype=np.int64)
        node_features.extend(local_nodes.tolist())
        for u, v, feat in zip(template["edge_src"], template["edge_dst"], template["edge_features"], strict=True):
            src.append(cursor + int(u))
            dst.append(cursor + int(v))
            edge_features.append([int(x) for x in feat])
        if idx > 0:
            prev_c = offsets[idx - 1] + 2
            cur_n = cursor
            for u, v in ((prev_c, cur_n), (cur_n, prev_c)):
                src.append(int(u))
                dst.append(int(v))
                edge_features.append([1, 0, 0])
        cursor += int(local_nodes.shape[0])

    terminal_o = cursor
    node_features.extend(templates["terminal_o_feature"].tolist())
    last_c = offsets[-1] + 2
    for u, v in ((last_c, terminal_o), (terminal_o, last_c)):
        src.append(int(u))
        dst.append(int(v))
        edge_features.append([1, 0, 0])

    node_type_arr = np.asarray([row[0] for row in node_features], dtype=np.int64)
    edge_type_arr = np.asarray([row[0] for row in edge_features], dtype=np.int64)
    graph_data: Dict[str, Any] = {
        "sample_id": sample_id,
        "num_nodes": int(len(node_features)),
        "edges": (src, dst),
        "node_features": np.asarray(node_features, dtype=np.int64).reshape((-1, 6)),
        "edge_features": np.asarray(edge_features, dtype=np.int64).reshape((-1, 3)),
        "node_type_ids": node_type_arr,
        "edge_type_ids": edge_type_arr,
        "node_token_ids": (node_type_arr * 2 + 1).reshape((-1, 1)),
        "edge_token_ids": (edge_type_arr * 2).reshape((-1, 1)),
        "properties": properties or {},
        "sequence": sequence,
    }
    return graph_data


def _residue_template_from_sequence(seq: str, flavor: int, residue_number: int) -> dict[str, Any]:
    """从短 DNA/RNA RDKit 分子中截取一个残基模板。"""

    from rdkit import Chem

    mol = Chem.MolFromFASTA(seq, flavor=int(flavor))
    if mol is None:
        raise ValueError(f"RDKit failed to build nucleotide template: seq={seq} flavor={flavor}")
    graph = _mol_heavy_graph_no_cleanup(mol)
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    old_to_heavy = {atom.GetIdx(): idx for idx, atom in enumerate(heavy_atoms)}
    residue_global: list[int] = []
    atom_names: dict[int, str] = {}
    for atom in heavy_atoms:
        info = atom.GetMonomerInfo()
        if info is None or int(info.GetResidueNumber()) != int(residue_number):
            continue
        heavy_idx = old_to_heavy[int(atom.GetIdx())]
        local_idx = len(residue_global)
        residue_global.append(heavy_idx)
        atom_names[local_idx] = str(info.GetName()).strip()

    if not residue_global:
        raise ValueError(f"empty nucleotide residue template: seq={seq} residue={residue_number}")
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(residue_global)}
    src_raw, dst_raw = graph["edges"]
    keep_edges = [
        idx
        for idx, (u, v) in enumerate(zip(src_raw, dst_raw))
        if int(u) in global_to_local and int(v) in global_to_local
    ]
    name_to_local = {name: idx for idx, name in atom_names.items()}
    return {
        "node_features": graph["node_features"][residue_global].copy(),
        "edge_src": [global_to_local[int(src_raw[idx])] for idx in keep_edges],
        "edge_dst": [global_to_local[int(dst_raw[idx])] for idx in keep_edges],
        "edge_features": graph["edge_features"][keep_edges].copy(),
        "o3_idx": int(name_to_local["O3'"]),
        "p_idx": int(name_to_local["P"]) if "P" in name_to_local else None,
    }


def _nucleotide_cross_edge_feature(flavor: int, second_base: str) -> list[int]:
    """读取前一个残基 O3' 到当前残基 P 的边特征。"""

    from rdkit import Chem

    mol = Chem.MolFromFASTA(f"A{second_base}", flavor=int(flavor))
    if mol is None:
        raise ValueError(f"RDKit failed to build nucleotide cross template: flavor={flavor}")
    graph = _mol_heavy_graph_no_cleanup(mol)
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    old_to_heavy = {atom.GetIdx(): idx for idx, atom in enumerate(atoms)}
    prev_o3 = None
    curr_p = None
    for atom in atoms:
        info = atom.GetMonomerInfo()
        if info is None:
            continue
        name = str(info.GetName()).strip()
        residue = int(info.GetResidueNumber())
        if residue == 1 and name == "O3'":
            prev_o3 = old_to_heavy[int(atom.GetIdx())]
        if residue == 2 and name == "P":
            curr_p = old_to_heavy[int(atom.GetIdx())]
    if prev_o3 is None or curr_p is None:
        raise ValueError("cannot locate nucleotide cross-bond atoms")
    src_raw, dst_raw = graph["edges"]
    for idx, (u, v) in enumerate(zip(src_raw, dst_raw)):
        if int(u) == int(prev_o3) and int(v) == int(curr_p):
            return [int(x) for x in graph["edge_features"][idx].tolist()]
    raise ValueError("cannot locate nucleotide cross-bond feature")


@lru_cache(maxsize=2)
def _standard_nucleotide_templates(target_type: str) -> dict[str, Any]:
    """用 RDKit 短 DNA/RNA 生成标准碱基模板，供长序列快速拼图使用。"""

    if target_type not in NUCLEIC_ACID_FASTA_FLAVORS:
        raise ValueError(f"unsupported nucleotide target_type: {target_type}")
    flavor = NUCLEIC_ACID_FASTA_FLAVORS[target_type]
    order = STANDARD_DNA_ORDER if target_type == "dna" else STANDARD_RNA_ORDER
    context_base = "A"
    return {
        "single": {
            base: _residue_template_from_sequence(base, flavor, 1)
            for base in order
        },
        "first": {
            base: _residue_template_from_sequence(f"{base}{context_base}", flavor, 1)
            for base in order
        },
        "internal": {
            base: _residue_template_from_sequence(f"{context_base}{base}{context_base}", flavor, 2)
            for base in order
        },
        "last": {
            base: _residue_template_from_sequence(f"{context_base}{base}", flavor, 2)
            for base in order
        },
        "cross_edge_feature": _nucleotide_cross_edge_feature(flavor, context_base),
    }


def nucleotide_sequence_to_graph_data_fast(
    sequence: str,
    target_type: str,
    sample_id: str,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """用标准 DNA/RNA 残基模板直接构建 heavy atom graph。

    该路径只覆盖 DNA 的 A/C/G/T 和 RNA 的 A/C/G/U。模板来自 RDKit
    ``MolFromFASTA`` 的短序列输出；非标准字符继续回退到 RDKit 全分子路径。
    """

    seq = sequence.strip().upper()
    if not is_standard_nucleotide_sequence(seq, target_type):
        raise ValueError(f"non-standard nucleotide sequence cannot use fast template path: {sample_id}")
    templates = _standard_nucleotide_templates(target_type)
    src: list[int] = []
    dst: list[int] = []
    node_features: list[list[int]] = []
    edge_features: list[list[int]] = []
    offsets: list[int] = []
    o3_abs: list[int] = []

    cursor = 0
    for idx, base in enumerate(seq):
        if len(seq) == 1:
            template = templates["single"][base]
        elif idx == 0:
            template = templates["first"][base]
        elif idx == len(seq) - 1:
            template = templates["last"][base]
        else:
            template = templates["internal"][base]
        offsets.append(cursor)
        local_nodes = np.asarray(template["node_features"], dtype=np.int64)
        node_features.extend(local_nodes.tolist())
        for u, v, feat in zip(template["edge_src"], template["edge_dst"], template["edge_features"], strict=True):
            src.append(cursor + int(u))
            dst.append(cursor + int(v))
            edge_features.append([int(x) for x in feat])
        o3_abs.append(cursor + int(template["o3_idx"]))
        if idx > 0:
            p_idx = template["p_idx"]
            if p_idx is None:
                raise ValueError(f"nucleotide template missing phosphate atom: {sample_id} idx={idx}")
            curr_p = cursor + int(p_idx)
            cross_feature = [int(x) for x in templates["cross_edge_feature"]]
            for u, v in ((o3_abs[idx - 1], curr_p), (curr_p, o3_abs[idx - 1])):
                src.append(int(u))
                dst.append(int(v))
                edge_features.append(cross_feature)
        cursor += int(local_nodes.shape[0])

    node_type_arr = np.asarray([row[0] for row in node_features], dtype=np.int64)
    edge_type_arr = np.asarray([row[0] for row in edge_features], dtype=np.int64)
    return {
        "sample_id": sample_id,
        "num_nodes": int(len(node_features)),
        "edges": (src, dst),
        "node_features": np.asarray(node_features, dtype=np.int64).reshape((-1, 6)),
        "edge_features": np.asarray(edge_features, dtype=np.int64).reshape((-1, 3)),
        "node_type_ids": node_type_arr,
        "edge_type_ids": edge_type_arr,
        "node_token_ids": (node_type_arr * 2 + 1).reshape((-1, 1)),
        "edge_token_ids": (edge_type_arr * 2).reshape((-1, 1)),
        "properties": properties or {},
        "sequence": sequence,
    }


def smiles_to_graph_data(smiles: str, sample_id: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {sample_id}: {smiles}")
    return mol_to_graph_data(mol, sample_id=sample_id, properties=properties, smiles=smiles)


def sequence_to_peptide_graph_data(sequence: str, sample_id: str,
                                   properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from rdkit import Chem

    seq = sequence.strip().upper()
    mol = Chem.MolFromFASTA(seq)
    if mol is None:
        raise ValueError(f"RDKit failed to build peptide molecule for {sample_id}")
    return mol_to_graph_data(mol, sample_id=sample_id, properties=properties, sequence=seq)
