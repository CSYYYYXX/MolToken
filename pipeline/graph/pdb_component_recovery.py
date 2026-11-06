"""Reconstruct a declared PDB residue fragment using CCD chemistry and PDB links.

Coordinates are not used to guess bonds. This preserves named modifications,
formal charges and CCD stereochemistry. External covalent attachments are
reported, never silently described as recovered. No sequence-X substitution.
"""
from functools import lru_cache
from pathlib import Path


class StructureRecoveryIssue(ValueError):
    """Expected incomplete/unsupported source chemistry; retain the failed case."""


def cif_rows(data, prefix):
    keys = [key for key in data if key.startswith(prefix)]
    if not keys:
        return []
    lengths = {len(data[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError(f"ragged mmCIF category: {prefix}")
    return [{key[len(prefix):]: data[key][i] for key in keys} for i in range(len(data[keys[0]]))]


def normalized_ins(value):
    return "" if value in ("?", ".", " ") else value


def include_deposited_covalent_decorations(residues, cif):
    """Add observed non-peptide covalent decorations omitted by sequence export.

    Do not add nearby/coordinate-bound ligands, other peptide chains, symmetry
    mates or unobserved amino acids. All additions have explicit PDB covale links.
    """
    result = list(residues)
    def key(r):
        return (r["chain"], r["auth_seq_id"], r["insertion_code"], r["component"])
    selected = {key(r) for r in result}
    types = {r["id"]: r["type"].upper() for r in cif_rows(cif, "_chem_comp.")}
    atoms = {}
    for row in cif_rows(cif, "_atom_site."):
        if row.get("pdbx_PDB_model_num", "1") != "1":
            continue
        k = (row["auth_asym_id"], row["auth_seq_id"], normalized_ins(row.get("pdbx_PDB_ins_code", "?")), row["label_comp_id"])
        atoms.setdefault(k, set()).add(row["label_atom_id"])
    connections = []
    for row in cif_rows(cif, "_struct_conn."):
        if row["conn_type_id"] != "covale" or any(row.get(f"ptnr{n}_symmetry", "1_555") not in ("1_555", ".", "?") for n in (1, 2)):
            continue
        connections.append(tuple((row[f"ptnr{n}_auth_asym_id"], row[f"ptnr{n}_auth_seq_id"],
                                  normalized_ins(row.get(f"pdbx_ptnr{n}_PDB_ins_code", "?")),
                                  row[f"ptnr{n}_label_comp_id"]) for n in (1, 2)))
    added = []
    while True:
        new = set()
        for a, b in connections:
            other = b if a in selected and b not in selected else a if b in selected and a not in selected else None
            if other is not None and other in atoms and "PEPTIDE" not in types[other[3]]:
                new.add(other)
        if not new:
            break
        for k in sorted(new):
            record = {"chain": k[0], "auth_seq_id": k[1], "insertion_code": k[2],
                      "component": k[3], "atom_names": sorted(atoms[k])}
            result.append(record)
            added.append(record)
        selected.update(new)
    return result, added


@lru_cache(maxsize=512)
def component_data(root, component):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    if component in {"UNK", "UNL", "UNX"}:
        raise StructureRecoveryIssue(f"source explicitly unidentified: {component}")
    path = Path(root) / f"{component}.cif"
    if not path.is_file():
        raise StructureRecoveryIssue(f"missing CCD definition: {component}")
    data = MMCIF2Dict(str(path))
    atoms = cif_rows(data, "_chem_comp_atom.")
    bonds = cif_rows(data, "_chem_comp_bond.")
    if not atoms or any(atom["comp_id"] != component for atom in atoms):
        raise ValueError(f"incorrect CCD component file: {path}")
    if any(a["type_symbol"] in {"X", "*", "?", "."} or a["charge"] in {"?", "."} for a in atoms):
        raise StructureRecoveryIssue(f"CCD has unspecified atom identity or charge: {component}")
    return atoms, bonds


def reconstruct_fragment(residues, cif, ccd_root):
    from rdkit import Chem
    bond_types = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
                  "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC}
    residue_keys = [(r["chain"], r["auth_seq_id"], r["insertion_code"], r["component"]) for r in residues]
    if len(set(residue_keys)) != len(residue_keys):
        raise ValueError("duplicate source residue identifiers")
    selected = set(residue_keys)
    templates = {key: component_data(str(ccd_root), key[3]) for key in residue_keys}
    links, external = {}, []
    atom_sites = cif_rows(cif, "_atom_site.")
    polymer_positions = {}
    for atom in atom_sites:
        if atom.get("pdbx_PDB_model_num", "1") != "1":
            continue
        key = (atom["auth_asym_id"], atom["auth_seq_id"],
               normalized_ins(atom.get("pdbx_PDB_ins_code", "?")), atom["label_comp_id"])
        if key in selected and atom["label_seq_id"] not in (".", "?"):
            polymer_positions[key] = (atom["label_asym_id"], int(atom["label_seq_id"]))

    def add_link(a, b, order, source):
        pair = tuple(sorted((a, b)))
        if pair in links and links[pair][0] != order:
            raise ValueError(f"inconsistent PDB link orders: {pair}")
        links[pair] = (order, source)

    for conn in cif_rows(cif, "_struct_conn."):
        if conn["conn_type_id"] not in ("covale", "disulf", "modres"):
            continue  # Coordination/hydrogen bonds are not chemical graph covalent edges.
        ends = []
        for n in (1, 2):
            key = (conn[f"ptnr{n}_auth_asym_id"], conn[f"ptnr{n}_auth_seq_id"],
                   normalized_ins(conn.get(f"pdbx_ptnr{n}_PDB_ins_code", "?")),
                   conn[f"ptnr{n}_label_comp_id"])
            ends.append((key, conn[f"ptnr{n}_label_atom_id"]))
        included = [end[0] in selected for end in ends]
        if all(included):
            if any(conn.get(f"ptnr{n}_symmetry", "1_555") not in ("1_555", ".", "?") for n in (1, 2)):
                raise StructureRecoveryIssue("symmetry-related covalent bond requires assembly reconstruction")
            order = conn.get("pdbx_value_order", "?").upper()
            add_link(*ends, "SING" if order in ("?", ".") else order, conn["conn_type_id"])
        elif any(included):
            external.append({"type": conn["conn_type_id"], "ends": ends})

    by_position = {position: key for key, position in polymer_positions.items()}
    for key, (asym, position) in polymer_positions.items():
        following = by_position.get((asym, position + 1))
        if following is None:
            continue
        # A peptidomimetic can use e.g. C20 rather than C as its backbone
        # carbonyl. Prefer the deposited bond into the next residue's N.
        explicit = [pair for pair, (_, source) in links.items()
                    if source == "covale" and {pair[0][0], pair[1][0]} == {key, following}]
        if any((following, "N") in pair for pair in explicit):
            continue
        own_names = {a["atom_id"] for a in templates[key][0]}
        next_names = {a["atom_id"] for a in templates[following][0]}
        if "C" in own_names and "N" in next_names:
            add_link((key, "C"), (following, "N"), "SING", "consecutive_declared_polymer_positions")
        elif not explicit:
            raise StructureRecoveryIssue(f"non-standard backbone needs explicit bond handling: {key} -> {following}")

    # CCD amino acids are isolated components. Remove leaving heavy atoms
    # only at explicitly joined endpoints, e.g. C-terminal OXT on peptide C.
    removed, condensation_evidence = set(), []
    observed = {key: set(residue.get("atom_names", ())) for key, residue in zip(residue_keys, residues)}
    for pair in links:
        for key, atom_name in pair:
            atoms, bonds = templates[key]
            leaving = {a["atom_id"] for a in atoms if a["pdbx_leaving_atom_flag"] == "Y"
                       and a["type_symbol"] not in ("H", "D")}
            for bond in bonds:
                a, b = bond["atom_id_1"], bond["atom_id_2"]
                other = b if a == atom_name else a if b == atom_name else None
                if other in leaving:
                    removed.add((key, other))
            # Some non-polymer CCD entries (e.g. KI2) omit the leaving flag
            # on the isolated acid's hydroxyl. An explicit C--N linkage,
            # terminal carboxyl-OH chemistry AND absence of that O from the
            # deposited residue jointly identify amide condensation. This
            # is not a generic valence-fixing fallback.
            partner = pair[1] if pair[0] == (key, atom_name) else pair[0]
            atom_specs = {a["atom_id"]: a for a in atoms}
            partner_specs = {a["atom_id"]: a for a in templates[partner[0]][0]}
            if atom_specs[atom_name]["type_symbol"] != "C" or partner_specs[partner[1]]["type_symbol"] != "N":
                continue
            neighbors = [(b["atom_id_2"] if b["atom_id_1"] == atom_name else b["atom_id_1"], b["value_order"].upper())
                         for b in bonds if atom_name in (b["atom_id_1"], b["atom_id_2"])]
            if not any(order == "DOUB" and atom_specs[name]["type_symbol"] == "O" for name, order in neighbors):
                continue
            for oxygen, order in neighbors:
                if order != "SING" or atom_specs[oxygen]["type_symbol"] != "O" or oxygen in observed[key]:
                    continue
                o_neighbors = [b["atom_id_2"] if b["atom_id_1"] == oxygen else b["atom_id_1"]
                               for b in bonds if oxygen in (b["atom_id_1"], b["atom_id_2"])]
                heavy = [n for n in o_neighbors if atom_specs[n]["type_symbol"] not in ("H", "D")]
                has_h = len(heavy) < len(o_neighbors)
                if heavy == [atom_name] and has_h and observed[key] and (key, oxygen) not in removed:
                    removed.add((key, oxygen))
                    condensation_evidence.append({"residue": key, "oxygen": oxygen,
                                                  "evidence": "CCD terminal carboxyl OH; absent in source; declared amide C-N bond"})
    rw, indices, desired_cip = Chem.RWMol(), {}, {}
    for key in residue_keys:
        atoms, bonds = templates[key]
        hydrogens = {a["atom_id"] for a in atoms if a["type_symbol"] in ("H", "D")}
        for spec in atoms:
            name = spec["atom_id"]
            if name in hydrogens or (key, name) in removed:
                continue
            # CCD commonly uses uppercase two-letter elements (ZN/CL/SE),
            # while RDKit requires chemical capitalization (Zn/Cl/Se).
            atom = Chem.Atom(spec["type_symbol"].title())
            atom.SetFormalCharge(int(spec["charge"]))
            if spec["type_symbol"] == "N" and spec["pdbx_aromatic_flag"] == "Y":
                nh = sum((b["atom_id_1"] == name and b["atom_id_2"] in hydrogens)
                         or (b["atom_id_2"] == name and b["atom_id_1"] in hydrogens) for b in bonds)
                atom.SetNumExplicitHs(nh)
            idx = rw.AddAtom(atom)
            indices[(key, name)] = idx
            if spec["pdbx_stereo_config"] in ("R", "S"):
                desired_cip[idx] = spec["pdbx_stereo_config"]
        for spec in bonds:
            a, b = (key, spec["atom_id_1"]), (key, spec["atom_id_2"])
            if a not in indices or b not in indices:
                continue
            if spec["pdbx_stereo_config"] in ("E", "Z"):
                raise StructureRecoveryIssue(f"explicit CCD alkene stereochemistry not implemented: {key[3]}")
            rw.AddBond(indices[a], indices[b], bond_types[spec["value_order"].upper()])
    for (a, b), (order, _) in links.items():
        if a not in indices or b not in indices:
            raise StructureRecoveryIssue(f"PDB link atom absent from CCD reconstruction: {a}, {b}")
        rw.AddBond(indices[a], indices[b], bond_types[order])
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException) as exc:
        raise StructureRecoveryIssue(f"CCD/PDB valence or aromaticity conflict: {exc}") from exc
    for idx in desired_cip:
        mol.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for idx, desired in desired_cip.items():
        atom = mol.GetAtomWithIdx(idx)
        if not atom.HasProp("_CIPCode"):
            raise StructureRecoveryIssue(f"CCD stereocentre absent after fragment construction: {idx}")
        if atom.GetProp("_CIPCode") != desired:
            atom.InvertChirality()
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    if any(mol.GetAtomWithIdx(idx).GetProp("_CIPCode") != label for idx, label in desired_cip.items()):
        raise StructureRecoveryIssue("CCD stereochemistry could not be retained")
    return mol, {"residue_components": [key[3] for key in residue_keys],
                 "explicit_inter_residue_bonds": len(links),
                 "removed_leaving_heavy_atoms": list(removed),
                 "condensation_with_missing_ccd_leaving_flag": condensation_evidence,
                 "external_covalent_links": external,
                 "all_external_covalent_attachments_included": not external,
                 "fragments": len(Chem.GetMolFrags(mol)),
                 "ccd_stereocentres_verified": len(desired_cip),
                 "scope": "declared source residue fragment; no unobserved residue imputation"}
