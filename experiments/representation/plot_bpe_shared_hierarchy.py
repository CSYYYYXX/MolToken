#!/usr/bin/env python3
"""A shared-trunk BPE hierarchy grounded in the frozen merge dependency DAG.

Only three tiny constructed references and the frozen codebook are used.
"""
import argparse
import json
import socket
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

import plot_bpe_merge_hierarchy as p

OUTPUT = p.ROOT / "scaling-csy/doc/paper_materials_20260728/figures/bpe_shared_hierarchy_20260915"
ROOT_FEATURE = "N:7,0,1,0,2,3"
SIZE = (17.2, 9.05)


def build_case(domain, sequence, vocabulary, guide):
    from pipeline.finetune.fddfs_encode import build_graph
    from pipeline.serialization.fddfs import serialize_fddfs
    graph = build_graph(target_type=domain, sample_id=f"shared_hierarchy_{domain}", sequence=sequence)
    serialized = serialize_fddfs(graph, guide.guide)
    ids = [vocabulary["token_to_id"][raw] if raw in vocabulary["token_to_id"] else 333 + int(raw[1:])
           for raw in serialized.sequence]
    old_by_new = {new: old for old, new in serialized.old_to_new.items()}
    raw_to_atom = {}
    for pos, raw in enumerate(serialized.sequence):
        if raw.startswith("N:"):
            raw_to_atom[pos] = old_by_new[len(raw_to_atom)]
    coords, edges = p.atom_layout(graph)
    return {"domain": domain, "reference_input": sequence, "graph": graph, "ids": ids,
            "raw_to_atom": raw_to_atom, "coords": coords, "edges": edges}


def intervals_for(case, path, expansions, rules):
    final = expansions[path[-1]]
    starts = [start for start in range(len(case["ids"]) - len(final) + 1)
              if tuple(case["ids"][start:start + len(final)]) == final]
    p.require(starts, f"No exact occurrence of T{path[-1]} in {case['domain']}/{case['reference_input']}")
    intervals = {path[-1]: (starts[0], starts[0] + len(final))}
    for previous, token in reversed(list(zip(path, path[1:]))):
        rule = rules[token]
        p.require(previous in (rule["left"], rule["right"]), f"Not a real merge: {previous}->{token}")
        start = intervals[token][0]
        if previous == rule["right"] and previous != rule["left"]:
            start += len(expansions[rule["left"]])
        intervals[previous] = (start, start + len(expansions[previous]))
    for token, interval in intervals.items():
        p.require(tuple(case["ids"][slice(*interval)]) == expansions[token], "Raw-symbol mismatch")
        atoms = p.covered_atoms(case, interval)
        p.require(p.component_count(atoms, case["edges"]) == 1, f"Disconnected token T{token}")
    return intervals


def signature(case, interval):
    ordered = [atom for raw, atom in sorted(case["raw_to_atom"].items()) if interval[0] <= raw < interval[1]]
    indices = {atom: i for i, atom in enumerate(ordered)}
    features = [list(map(int, case["graph"]["node_features"][atom])) for atom in ordered]
    edges = sorted((min(indices[u], indices[v]), max(indices[u], indices[v]), order)
                   for u, v, order in case["edges"] if u in indices and v in indices)
    return features, edges


def prepare():
    from pipeline.finetune.fddfs_encode import load_corpus_guide
    vocabulary, labels, expansions, rules = p.load_book()
    root = vocabulary["token_to_id"][ROOT_FEATURE]
    paths = {"histidine": [root, 586, 629, 716, 799, 1140, 13037],
             "dna": [root, 586, 629, 633, 634, 652],
             "rna": [root, 586, 629, 633, 634, 643]}
    guide = load_corpus_guide()
    cases = {key: build_case(domain, seq, vocabulary, guide)
             for key, domain, seq in (("histidine", "protein", "H"), ("dna", "dna", "CA"), ("rna", "rna", "CA"))}
    for key, case in cases.items():
        case["intervals"] = intervals_for(case, paths[key], expansions, rules)
    for tid in (root, 629):
        signatures = [signature(case, case["intervals"][tid]) for case in cases.values()]
        p.require(all(sig == signatures[0] for sig in signatures), f"Shared T{tid} has different induced chemistry")
    p.require(signature(cases["dna"], cases["dna"]["intervals"][634]) ==
              signature(cases["rna"], cases["rna"]["intervals"][634]), "DNA/RNA shared motif mismatch")
    return root, labels, expansions, rules, paths, cases


def design(root):
    # Shared nodes are drawn once. Both branch points therefore remain visible.
    x0, x1, x2, x3, x4 = 1.02, 3.52, 6.62, 9.96, 14.52
    top, middle, bottom = 7.25, 4.60, 1.95
    lower = (middle + bottom) / 2
    trunk = (top + lower) / 2
    nodes = [
        ("root", root, None, "histidine", x0, trunk, 1.38, 1.66, "Shared atom"),
        ("shared", 629, root, "histidine", x1, trunk, 2.18, 2.22, "Aromatic C–N"),
        ("ncn", 716, 629, "histidine", x2, top, 2.72, 2.32, "N–C–N motif"),
        ("amino", 633, 629, "dna", x2, lower, 2.72, 2.32, "Amino-heteroaromatic motif"),
        ("imidazole", 1140, 716, "histidine", x3, top, 2.72, 2.32, "Imidazole motif"),
        ("adenine", 634, 633, "dna", x3, lower, 2.72, 2.32, "Adenine-related motif"),
        ("histidine", 13037, 1140, "histidine", x4, top, 4.40, 2.42, "Histidine fragment"),
        ("dna", 652, 634, "dna", x4, middle, 4.40, 2.42, "DNA nucleotide fragment"),
        ("rna", 643, 634, "rna", x4, bottom, 4.40, 2.42, "RNA nucleotide fragment"),
    ]
    fields = ("key", "token", "prior", "case", "x", "y", "width", "height", "title")
    nodes = {row[0]: dict(zip(fields, row)) for row in nodes}
    edges = [("root", "shared", [root, 586, 629]),
             ("shared", "ncn", [629, 716]), ("shared", "amino", [629, 633]),
             ("ncn", "imidazole", [716, 799, 1140]), ("amino", "adenine", [633, 634]),
             ("imidazole", "histidine", [1140, 13037]), ("adenine", "dna", [634, 652]),
             ("adenine", "rna", [634, 643])]
    return nodes, edges, (x0, x1, x2, x3, x4)


def mark_new(ax, before, prefix):
    for index, artist in enumerate(a for a in ax.get_children() if id(a) not in before):
        artist.set_gid(f"{prefix}__{index}")


def render(output):
    root, labels, expansions, rules, paths, cases = prepare()
    nodes, connections, xs = design(root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, SIZE[0]), ylim=(0, SIZE[1]))
    ax.axis("off")
    groups, texts = [], []
    for source, target, path in connections:
        a, b = nodes[source], nodes[target]
        for previous, token in zip(path, path[1:]):
            p.require(previous in (rules[token]["left"], rules[token]["right"]), "Invented dependency")
        start = (a["x"] + a["width"] / 2 + .05, a["y"])
        end = (b["x"] - b["width"] / 2 - .07, b["y"])
        before = {id(artist) for artist in ax.get_children()}
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                    color="#91A0AE", lw=1.25,
                                    linestyle=(0, (3, 2)) if len(path) > 2 else "-", zorder=1))
        if len(path) > 2:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .17,
                    f"{len(path) - 1} merges", ha="center", color="#7B8793", fontsize=6.2)
        mark_new(ax, before, f"arrow-{source}-{target}")
        groups.append({"id": f"arrow-{source}-{target}", "name": f"Merge path T{a['token']} to T{b['token']}"})
    for key, node in nodes.items():
        x, y, width, height = (node[k] for k in ("x", "y", "width", "height"))
        left, bottom = x - width / 2, y - height / 2
        before = {id(artist) for artist in ax.get_children()}
        shared = key in ("root", "shared")
        ax.add_patch(FancyBboxPatch((left, bottom), width, height,
                                   boxstyle="round,pad=0.025,rounding_size=.09", zorder=2,
                                   facecolor="#F5F9FB" if shared else "white",
                                   edgecolor="#B4CCD8" if shared else "#D3DCE4", lw=1.1 if shared else .8))
        title = ax.text(x, bottom + height - .23, node["title"], ha="center", va="center",
                        fontsize=8.3 if width > 2 else 8.5, fontweight="semibold", color="#495866", zorder=6)
        texts.append((title, left + .06, left + width - .06))
        case = cases[node["case"]]
        if key == "root":
            ax.add_patch(Circle((x, y + .06), .25, facecolor="white", edgecolor=p.OLD, lw=1.7, zorder=5))
            ax.text(x, y + .06, "N", fontsize=22, color=p.OLD, ha="center", va="center", zorder=6)
        else:
            current = case["intervals"][node["token"]]
            previous = case["intervals"][node["prior"]]
            p.require(p.covered_atoms(case, previous) < p.covered_atoms(case, current), "Displayed merge must grow")
            p.molecule_fragment(ax, case, current, previous, case["coords"], case["edges"],
                                (left + .12, bottom + .69, width - .24, height - 1.07))
            # Preserve all abbreviated symbols; long final fragments wrap, never truncate.
            sequence = p.abbreviated_sequence(expansions[node["token"]], labels)
            lines = textwrap.wrap(sequence, width=int((width - .28) * 72 / 3.50),
                                  break_long_words=False, break_on_hyphens=False)
            p.require(len(lines) <= 2, f"Sequence needs a larger card: T{node['token']}")
            label = ax.text(x, bottom + .47, "\n".join(lines), ha="center", va="center",
                            family="DejaVu Sans Mono", fontsize=5.8, color="#7B8793", linespacing=1.3, zorder=6)
            texts.append((label, left + .10, left + width - .10))
        ax.text(x, bottom + .16, rf"$\mathbf{{T_{{{node['token']}}}}}$", fontsize=9,
                ha="center", color="#273744", zorder=6)
        mark_new(ax, before, "card-" + key)
        groups.append({"id": "card-" + key, "name": f"{node['title']} / T{node['token']}"})
    stages = ("Shared root", "Shared fragment", "Chemical motifs", "Larger motifs", "Biomolecular fragments")
    for index, (x, stage) in enumerate(zip(xs, stages)):
        ax.text(x, .48, f"({chr(97 + index)}) {stage}", ha="center", va="center", fontsize=8.4,
                fontweight="semibold", color="#344250", gid=f"annotation-stage{index}")
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=5.5,
                      markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.2)
               for color in (p.OLD, p.ADDED, p.CONTEXT)]
    handles += [Line2D([], [], color="#91A0AE", lw=1.2, linestyle="--")]
    fig.legend(handles, ["inherited fragment", "added by merges", "reference context", "multiple BPE merges"],
               loc="center", bbox_to_anchor=(9.4, .14), bbox_transform=ax.transData, ncol=4,
               frameon=False, columnspacing=2.7, handletextpad=.6, fontsize=7, labelcolor="#4C5864")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, left, right in texts:
        box = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        p.require(box.x0 >= left and box.x1 <= right, f"Card text exceeds bounds: {artist.get_text()}")
    box = fig.get_tightbbox(renderer)
    p.require(box.x0 >= -.01 and box.y0 >= -.01 and box.x1 <= SIZE[0] + .01 and box.y1 <= SIZE[1] + .01,
              f"Clipped drawing: {box.bounds}")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"bpe_merge_hierarchy.{suffix}", dpi=350)
    plt.close(fig)
    tokens = sorted({tid for path in paths.values() for tid in path})
    report = {"revision": "shared_trunk_v5", "codebook": str(p.BOOK), "vocabulary": str(p.VOCAB),
              "root": {"id": root, "raw_symbol": labels[root]}, "paths": paths, "layout_groups": groups,
              "display_nodes": list(nodes.values()),
              "display_edges": [{"source": a, "target": b, "actual_path": path} for a, b, path in connections],
              "records": [{"id": tid, "raw_ids": expansions[tid], "raw_tokens": [labels[t] for t in expansions[tid]],
                           "merge_rule": rules.get(tid)} for tid in tokens],
              "contexts": {key: {"domain": c["domain"], "reference_input": c["reference_input"],
                                  "intervals": c["intervals"],
                                  "atom_counts": {tid: len(p.covered_atoms(c, v)) for tid, v in c["intervals"].items()}}
                           for key, c in cases.items()},
              "checks": {"genuine_merge_paths": True, "shared_induced_chemistry_identical": True,
                         "connected_selected_atom_sets": True, "all_abbreviated_symbols_retained": True},
              "caveats": ["One selected shared base token, not the only vocabulary root or an FDDFS traversal root.",
                          "Chemical labels describe selected graph fragments, not complete standalone molecules or mandatory residue-aligned BPE units.",
                          "Arrows are codebook dependency paths, not chemical reactions. Dashed arrows contain two actual merges, including syntax-only merges.",
                          "Reference inputs H and CA are small constructed examples, not randomly sampled training records.",
                          "Source graph provides grey neighboring context. Chemical labels are post-hoc, not imposed by training.",
                          "No token-frequency or cross-domain occurrence-frequency claim is made."]}
    (output / "selected_merges.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("SHARED_HIERARCHY_COMPLETE", output, "root", root, "nodes", len(nodes), "arrows", len(connections), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--output", type=p.Path, default=OUTPUT)
    args = parser.parse_args()
    p.require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the confirmed worker")
    render(args.output)
