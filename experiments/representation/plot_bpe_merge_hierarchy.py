#!/usr/bin/env python3
"""Worker-side, codebook-grounded illustration of selected integer-BPE paths.

Only the frozen merge table/vocabulary and tiny constructed sequence examples
are read. No training corpus, checkpoint, codebook update, or model run.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import socket
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
ROOT = PIPELINE.parents[2]
DATA = Path("./storage/molscaling/data")
BOOK = DATA / "fddfs_bpe_full_prefix70_64k_mf80000_v1/codebook_fddfs_prefix70_mf80000_v1/merges.csv"
VOCAB = DATA / "fddfs_guide_recount_v1/vocab.json"
OUTPUT = ROOT / "scaling-csy/doc/paper_materials_20260728/figures/bpe_merge_hierarchy_20260914"
PATHS = {
    "amino_acid": [42, 644, 673, 674, 724],
    "nucleotide": [42, 593, 598, 602, 626],
    "learned_motif": [42, 644, 1292, 7160, 10594],
}
OLD = "#277DA1"
ADDED = "#DD9B43"
CONTEXT = "#C8CDD2"
LANES = ("amino_acid", "nucleotide", "learned_motif")
LANE_NAMES = ("Amino-acid\nfragment", "Nucleotide\nfragment", "Learned\nring motif")
CONTEXT_DOMAINS = {"amino_acid": "protein", "nucleotide": "dna", "learned_motif": "small_molecule"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_book():
    vocabulary = json.loads(VOCAB.read_text())
    labels = {int(i): t for t, i in vocabulary["token_to_id"].items()}
    labels.update({i: f"R{i - 333}" for i in range(334, 583)})
    expansions = {i: (i,) for i in labels}
    rules = {}
    with BOOK.open() as stream:
        for row in csv.DictReader(stream):
            row = {key: int(value) for key, value in row.items()}
            new, left, right = row["new_id"], row["left"], row["right"]
            require(new not in expansions and left in expansions and right in expansions,
                    f"Invalid merge dependency: {row}")
            expansions[new] = expansions[left] + expansions[right]
            rules[new] = row
    for lane, path in PATHS.items():
        for parent, child in zip(path, path[1:]):
            require(parent in (rules[child]["left"], rules[child]["right"]),
                    f"Displayed arrow is not a direct binary merge: {lane}/{parent}->{child}")
    return vocabulary, labels, expansions, rules


def compact(token_id, labels):
    token = labels[token_id]
    if token.startswith("N:"):
        feat = list(map(int, token[2:].split(",")))
        symbol = {6: "C", 7: "N", 8: "O", 15: "P", 16: "S"}[feat[0]]
        return symbol.lower() if feat[2] else symbol
    if token.startswith("E:"):
        return {2: "=", 3: "#", 4: ":"}[int(token[2:].split(",")[0])]
    return token


def abbreviated_sequence(raw_ids, labels):
    """Display every raw symbol in order, abbreviating feature labels only."""
    return " ".join(compact(token_id, labels) for token_id in raw_ids)


def example_contexts(vocabulary, expansions):
    from pipeline.finetune.fddfs_encode import load_corpus_guide, build_graph
    from pipeline.serialization.fddfs import serialize_fddfs
    guide = load_corpus_guide()
    cases = []
    candidates = [("protein", seq) for seq in ("R", "GR", "RG", "GRG", "ARRA", "AGRS", "K", "N")]
    candidates += [(domain, seq) for domain in ("dna", "rna") for seq in ("A", "AA", "AC", "AG", "CG", "GCA", "ACGA")]
    candidates.append(("small_molecule", "c1ccc(N2CCOCC2)cc1"))
    for domain, sequence in candidates:
        field = "smiles" if domain == "small_molecule" else "sequence"
        graph = build_graph(target_type=domain, sample_id=f"illustration_{domain}_{sequence}", **{field: sequence})
        serialized = serialize_fddfs(graph, guide.guide)
        ids = [vocabulary["token_to_id"][token] if token in vocabulary["token_to_id"]
               else 333 + int(token[1:]) for token in serialized.sequence]
        old_by_new = {new: old for old, new in serialized.old_to_new.items()}
        raw_to_atom = {}
        atom_index = 0
        for pos, token in enumerate(serialized.sequence):
            if token.startswith("N:"):
                raw_to_atom[pos] = old_by_new[atom_index]
                atom_index += 1
        hits = {}
        for lane, path in PATHS.items():
            expansion = expansions[path[-1]]
            for start in range(len(ids) - len(expansion) + 1):
                if tuple(ids[start:start + len(expansion)]) == expansion:
                    hits[lane] = start
                    break
        cases.append({"domain": domain, "input_field": field, "reference_input": sequence, "graph": graph,
                      "ids": ids, "raw_to_atom": raw_to_atom, "hits": hits})
        print("CONTEXT", domain, sequence, len(ids), hits, flush=True)
    return cases


def selected_context(cases, lane):
    domain = CONTEXT_DOMAINS[lane]
    matched = [case for case in cases if case["domain"] == domain and lane in case["hits"]]
    require(bool(matched), f"No exact constructed-sequence context for {lane}; do not invent a structure")
    return min(matched, key=lambda case: len(case["ids"]))


def path_intervals(case, lane, expansions, rules):
    """Track the chosen child occurrence back through the actual binary DAG."""
    path = PATHS[lane]
    start = case["hits"][lane]
    intervals = {path[-1]: (start, start + len(expansions[path[-1]]))}
    for parent, child in reversed(list(zip(path, path[1:]))):
        row = rules[child]
        start = intervals[child][0]
        # If both inputs are the same token, follow its left occurrence.
        if parent == row["right"] and parent != row["left"]:
            start += len(expansions[row["left"]])
        end = start + len(expansions[parent])
        require(tuple(case["ids"][start:end]) == expansions[parent], "Context/token expansion mismatch")
        intervals[parent] = (start, end)
    return intervals


def covered_atoms(case, interval):
    return {atom for raw, atom in case["raw_to_atom"].items() if interval[0] <= raw < interval[1]}


def component_count(atoms, edges):
    """Count components using real selected-to-selected edges, not grey context."""
    remaining = set(atoms)
    count = 0
    while remaining:
        count += 1
        reached = {remaining.pop()}
        while True:
            before = len(reached)
            for u, v, _ in edges:
                if u in atoms and v in atoms and ({u, v} & reached):
                    reached.update((u, v))
            if len(reached) == before:
                break
        remaining.difference_update(reached)
    return count


def atom_layout(graph):
    """Coordinates only: chemical identity/edges come from the canonical graph."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    mol = Chem.RWMol()
    for feat in graph["node_features"]:
        atom = Chem.Atom(int(feat[0]))
        atom.SetFormalCharge(int(feat[1]))
        atom.SetIsAromatic(bool(feat[2]))
        mol.AddAtom(atom)
    edges = []
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for u, v, feat in zip(*graph["edges"], graph["edge_features"]):
        if u < v:
            order = int(feat[0])
            mol.AddBond(int(u), int(v), kinds[order])
            edges.append((int(u), int(v), order))
    result = mol.GetMol()
    result.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(result)
    rdDepictor.Compute2DCoords(result)
    pos = result.GetConformer().GetPositions()[:, :2]
    return pos, edges


def molecule_fragment(ax, case, interval, prior_interval, coords, edges, box):
    """Highlight exact atom coverage, leaving one-hop reference context grey.

    Full features are retained in the companion JSON; abbreviated raw sequences
    accompany the graph in each card. These selected examples must be connected;
    this is not asserted for arbitrary vocabulary tokens or standalone molecules.
    """
    selected = covered_atoms(case, interval)
    prior = covered_atoms(case, prior_interval)
    require(prior <= selected and selected, "Atom coverage is not nested")
    require(component_count(selected, edges) == 1, f"Disconnected selected example: {interval}")
    require(prior < selected, f"Selected merge does not add any atom: {interval}")
    visible = set(selected)
    for u, v, _ in edges:
        if u in selected or v in selected:
            visible.update((u, v))
    # Keep reference coordinates stable, but fit the relevant local neighborhood.
    xx, yy, width, height = box
    occupied = coords[sorted(visible)]
    midpoint = (occupied.max(axis=0) + occupied.min(axis=0)) / 2
    extent = np.maximum(np.ptp(occupied, axis=0), .8)
    scale = min((width - .24) / extent[0], (height - .24) / extent[1], .48)
    pts = (coords - midpoint) * scale + np.array([xx + width / 2, yy + height / 2])
    for u, v, order in edges:
        if u not in visible or v not in visible or not ({u, v} & selected):
            continue
        a, b = pts[u], pts[v]
        color = "#5F6B77" if u in selected and v in selected else CONTEXT
        delta = b - a
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta) * .025
        offsets = (0,) if order in (1, 4) else (-1, 1) if order == 2 else (-1.5, 0, 1.5)
        for shift in offsets:
            xy = np.array([a + shift * normal, b + shift * normal])
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=.9,
                    linestyle="--" if order == 4 else "-", zorder=3)
    for atom in sorted(visible):
        point = pts[atom]
        if atom not in selected:
            ax.add_patch(Circle(point, .038, facecolor="white", edgecolor=CONTEXT, lw=.65, zorder=4))
            continue
        color = OLD if atom in prior else ADDED
        ax.add_patch(Circle(point, .088, facecolor="white", edgecolor=color, lw=1.4, zorder=5))
        symbol = {6: "C", 7: "N", 8: "O", 15: "P", 16: "S"}[int(case["graph"]["node_features"][atom][0])]
        ax.text(*point, symbol, color=color, fontsize=8, ha="center", va="center", fontweight="semibold", zorder=6)


def render(cases, labels, expansions, rules, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(15.6, 7.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 15.6), ylim=(0, 7.55))
    ax.axis("off")
    xs, ys = (1.03, 4.0, 7.2, 10.4, 13.6), (6.4, 4.18, 1.96)
    width, height = 2.62, 1.96
    root_y = ys[1]
    ax.add_patch(FancyBboxPatch((.30, root_y - .74), 1.46, 1.48,
                              boxstyle="round,pad=0.03,rounding_size=.12", facecolor="#F7F9FB", edgecolor="#BAC4CF", lw=.8))
    ax.add_patch(Circle((xs[0], root_y + .18), .25, facecolor="white", edgecolor=OLD, lw=1.7))
    ax.text(xs[0], root_y + .18, "C", ha="center", va="center", fontsize=22, color=OLD)
    ax.text(xs[0], root_y - .34, r"$T_{42}$", ha="center", fontsize=11)
    contexts = {}
    sequence_artists = []
    for index, lane in enumerate(LANES):
        y = ys[index]
        path = PATHS[lane]
        ax.text(xs[0], y if index != 1 else y + 1.05, LANE_NAMES[index],
                fontsize=9, color="#495866", ha="center", va="center",
                multialignment="center", fontweight="semibold")
        case = selected_context(cases, lane)
        intervals = path_intervals(case, lane, expansions, rules)
        coords, edges = atom_layout(case["graph"])
        audit = {token: {"atoms": sorted(covered_atoms(case, intervals[token])),
                         "components": component_count(covered_atoms(case, intervals[token]), edges)}
                 for token in path}
        require(all(item["components"] == 1 for item in audit.values()), f"Disconnected path: {lane}/{audit}")
        contexts[lane] = {"type": case["domain"], case["input_field"]: case["reference_input"],
                          "source": "Tiny constructed reference; not claimed to be a sampled training record",
                          "intervals": intervals, "atom_coverage": audit}
        print("CONNECTED_PATH", lane, case["reference_input"], [len(audit[t]["atoms"]) for t in path], flush=True)
        for col, token in enumerate(path[1:], start=1):
            parent = path[col - 1]
            rule = rules[token]
            x = xs[col]
            left_edge = x - width / 2
            right_edge = xs[col - 1] + (width / 2 if col > 1 else .76)
            prior_y = y if col > 1 else root_y
            ax.add_patch(FancyArrowPatch((right_edge + .02, prior_y), (left_edge - .06, y),
                                        arrowstyle="-|>", mutation_scale=10, color="#91A0AE", lw=1.0,
                                        connectionstyle="arc3,rad=0" if col > 1 else f"arc3,rad={-.14 if index == 0 else .14 if index == 2 else 0}",
                                        zorder=1))
            ax.add_patch(FancyBboxPatch((left_edge, y - height / 2), width, height,
                                      boxstyle="round,pad=0.025,rounding_size=.09", facecolor="white",
                                      edgecolor="#D3DCE4", lw=.8, zorder=2))
            molecule_fragment(ax, case, intervals[token], intervals[parent], coords, edges,
                              (left_edge + .10, y - .27, width - .20, 1.17))
            sequence_artist = ax.text(x, y - .49, abbreviated_sequence(expansions[token], labels),
                                     ha="center", va="center", fontsize=6.5, color="#7B8793",
                                     family="DejaVu Sans Mono", zorder=6)
            sequence_artist.set_gid(f"sequence-{lane}-{token}")
            sequence_artists.append((sequence_artist, left_edge + .10, left_edge + width - .10))
            ax.text(x, y - .79, rf"$T_{{{rule['left']}}} + T_{{{rule['right']}}} \rightarrow \mathbf{{T_{{{token}}}}}$",
                    ha="center", fontsize=8.3, zorder=6)
            # Both parents are explicit in every equation; no chemical bond is
            # implied by the plus sign, which means concatenation of token streams.
    stage_names = ("(a) Base token", "(b) Merge 1", "(c) Merge 2", "(d) Merge 3", "(e) Merge 4")
    for x, title in zip(xs, stage_names):
        ax.text(x, .71, title, ha="center", fontsize=9, fontweight="semibold")
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=6.5,
                      markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.3)
               for color in (OLD, ADDED, CONTEXT)]
    # Three evenly spaced centers over the four merge columns, excluding root.
    group_left, group_right = xs[1] - width / 2, xs[-1] + width / 2
    legend_centers = group_left + (np.arange(3) + .5) * (group_right - group_left) / 3
    legends = []
    for handle, label, center in zip(handles, ("previous token", "merging partner", "reference context"), legend_centers):
        legends.append(fig.legend([handle], [label], loc="center", bbox_to_anchor=(center, .23),
                                  bbox_transform=ax.transData, frameon=False, borderaxespad=0,
                                  borderpad=0, handletextpad=.7, fontsize=7.6, labelcolor="#4C5864"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for legend, center in zip(legends, legend_centers):
        box = legend.get_window_extent(renderer).transformed(ax.transData.inverted())
        require(group_left <= box.x0 and box.x1 <= group_right and np.isclose((box.x0 + box.x1) / 2, center),
                "Legend must be centered within the four-column merge group")
    for artist, left, right in sequence_artists:
        box = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        require(box.x0 >= left and box.x1 <= right, f"Sequence label exceeds its card: {artist.get_text()}")
    bounds = fig.get_tightbbox(renderer)
    require(bounds.x0 >= -.01 and bounds.y0 >= -.01 and bounds.x1 <= 15.61 and bounds.y1 <= 7.56,
            f"Clipped artwork: {bounds.bounds}")
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"bpe_merge_hierarchy.{extension}", dpi=450)
    plt.close(fig)
    selected = sorted({token for path in PATHS.values() for token in path})
    provenance = {"revision": "connected_graphs_right_group_legend_v4", "codebook": str(BOOK), "vocabulary": str(VOCAB), "paths": PATHS,
                  "root": {"id": 42, "raw_symbol": labels[42], "definition": "BPE base token, not the FDDFS traversal root"},
                  "new_fits_or_codebook_changes": False,
                  "records": [{"id": token, "raw_ids": expansions[token],
                               "raw_tokens": [labels[t] for t in expansions[token]],
                               "merge_rule": rules.get(token)} for token in selected],
                  "contexts": contexts,
                  "caveats": ["Five displayed columns are selected path depths, not global BPE training rounds.",
                              "All displayed examples have connected exact atom coverage; arbitrary BPE tokens need not be graph-connected or standalone molecules.",
                              "Full atom/edge features and ring/branch syntax remain in raw_tokens; figure sequences preserve raw symbol order but abbreviate atom/edge feature labels.",
                              "Codebook freq/real_merges describe the recorded merge step, not final vocabulary occurrence counts.",
                              "Functional/residue labels are post-hoc descriptions, not an imposed vocabulary or a complete amino-acid/nucleotide token guarantee.",
                              "The learned ring example is a morpholine-related motif; it is not claimed to be unrecognizable to a chemist or unprecedented."]}
    (output / "selected_merges.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print("RENDER_COMPLETE", output, "direct merges", sum(len(path) - 1 for path in PATHS.values()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the confirmed worker")
    vocabulary, labels, expansions, rules = load_book()
    for lane, path in PATHS.items():
        print("PATH", lane, path, flush=True)
        for token in path:
            print(token, rules.get(token), " ".join(compact(t, labels) for t in expansions[token]), flush=True)
    cases = example_contexts(vocabulary, expansions)
    if args.inspect:
        return
    render(cases, labels, expansions, rules, args.output)


if __name__ == "__main__":
    main()
