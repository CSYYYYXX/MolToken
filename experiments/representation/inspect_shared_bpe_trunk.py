#!/usr/bin/env python3
"""Inspect genuine shared BPE ancestors in small constructed molecular examples.

No training data, checkpoints, or vocabulary fitting. Run on a CPU worker.
"""
import argparse
from collections import defaultdict, deque
import json
import socket

import plot_bpe_merge_hierarchy as p


def references(vocabulary):
    from pipeline.finetune.fddfs_encode import build_graph, load_corpus_guide
    from pipeline.serialization.fddfs import serialize_fddfs
    guide = load_corpus_guide()
    amino = "ACDEFGHIKLMNPQRSTVWY"
    examples = [("protein", seq) for seq in amino]
    examples += [("protein", "G" + seq + "G") for seq in amino]
    examples += [("peptide", seq) for seq in ("SS", "RS", "KK", "GSG", "GKG")]
    examples += [(domain, seq) for domain, alphabet in (("dna", "ACGT"), ("rna", "ACGU"))
                 for seq in list(alphabet) + [a + b for a in alphabet for b in alphabet] + [alphabet]]
    examples += [("small_molecule", seq) for seq in (
        "CCO", "CCCO", "OCCO", "CCN", "CCCN", "CC(=O)O", "CC(=O)N", "CC(=O)NCCO",
        "NCCO", "NCC(=O)O", "N[C@@H](CO)C(=O)O", "NCCC(=O)O", "NCCCN",
        "CCOCC", "OCCCO", "CCN(CC)CCO", "c1ccc(N2CCOCC2)cc1", "C1COCCN1",
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O", "O=C1NC=CC(=O)N1",
        "c1c[nH]cn1", "Cc1c[nH]cn1", "Nc1ncnc2[nH]cnc12", "c1ncnc2[nH]cnc12" )]
    cases = []
    for domain, sequence in examples:
        field = "smiles" if domain == "small_molecule" else "sequence"
        graph = build_graph(target_type=domain, sample_id="shared_trunk_illustration", **{field: sequence})
        serialized = serialize_fddfs(graph, guide.guide)
        ids = [vocabulary["token_to_id"][raw] if raw in vocabulary["token_to_id"] else 333 + int(raw[1:])
               for raw in serialized.sequence]
        old_by_new = {new: old for old, new in serialized.old_to_new.items()}
        raw_to_atom = {}
        for pos, raw in enumerate(serialized.sequence):
            if raw.startswith("N:"):
                raw_to_atom[pos] = old_by_new[len(raw_to_atom)]
        edges = [(int(u), int(v), int(feat[0])) for u, v, feat in zip(*graph["edges"], graph["edge_features"]) if u < v]
        cases.append({"domain": domain, "input_field": field, "reference_input": sequence,
                      "graph": graph, "ids": ids, "raw_to_atom": raw_to_atom, "edges": edges})
    return cases


def matches(cases, expansions, labels):
    trie = {}
    for token, raw in expansions.items():
        if len(raw) > 120 or not any(labels[t].startswith("N:") for t in raw):
            continue
        node = trie
        for symbol in raw:
            node = node.setdefault(symbol, {})
        node.setdefault("end", []).append(token)
    found = defaultdict(list)
    for case_index, case in enumerate(cases):
        for start in range(len(case["ids"])):
            node = trie
            for stop in range(start, min(start + 120, len(case["ids"]))):
                if case["ids"][stop] not in node:
                    break
                node = node[case["ids"][stop]]
                for token in node.get("end", ()):
                    atoms = p.covered_atoms(case, (start, stop + 1))
                    if p.component_count(atoms, case["edges"]) != 1:
                        continue
                    found[token].append((case_index, start, stop + 1, len(atoms)))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--output", type=p.Path, required=True)
    args = parser.parse_args()
    p.require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the confirmed worker")
    vocabulary, labels, expansions, rules = p.load_book()
    cases = references(vocabulary)
    found = matches(cases, expansions, labels)
    children = defaultdict(set)
    for token, rule in rules.items():
        children[rule["left"]].add(token)
        children[rule["right"]].add(token)
    summaries = []
    for token, hits in sorted(found.items()):
        domains = {cases[h[0]]["domain"] for h in hits}
        atom_count = hits[0][-1]
        if not (2 <= atom_count <= 6 and domains & {"protein", "peptide"} and domains & {"dna", "rna"}):
            continue
        descendants = defaultdict(list)
        queue = deque([(token,)])
        while queue:
            path = queue.popleft()
            tid = path[-1]
            if tid != token:
                for case_index, start, end, count in found.get(tid, ()):
                    domain = cases[case_index]["domain"]
                    if atom_count < count <= 26:
                        descendants[domain].append({"path": list(path), "atoms": count,
                                                   "case_index": case_index, "interval": [start, end]})
            if len(path) < 8:
                for child in sorted(children[tid]):
                    if child in found:
                        queue.append(path + (child,))
        for domain in descendants:
            descendants[domain].sort(key=lambda x: (-x["atoms"], len(x["path"]), len(cases[x["case_index"]]["ids"])))
            descendants[domain] = descendants[domain][:40]
        summary = {"id": token, "atoms": atom_count, "raw": p.abbreviated_sequence(expansions[token], labels),
                   "raw_features": [labels[t] for t in expansions[token]], "rule": rules.get(token),
                   "domains": sorted(domains),
                   "examples": {domain: [cases[h[0]]["reference_input"] for h in hits if cases[h[0]]["domain"] == domain][:5]
                                for domain in sorted(domains)}, "descendants": descendants}
        summaries.append(summary)
        print("SHARED", token, atom_count, sorted(domains), summary["raw"], flush=True)
        print("FEATURES", summary["raw_features"], flush=True)
        for domain, choices in descendants.items():
            for choice in choices[:3]:
                print("DESC", domain, cases[choice["case_index"]]["reference_input"], choice["path"], choice["atoms"],
                      p.abbreviated_sequence(expansions[choice["path"][-1]], labels), flush=True)
    for tid in (590, 629, 716, 799, 1140, 2278, 2949, 13037, 633, 634, 652, 885):
        print("TOKEN_DETAIL", tid, rules.get(tid), [labels[t] for t in expansions[tid]], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"candidates": summaries,
                                      "references": [{"domain": c["domain"], "input": c["reference_input"]} for c in cases]},
                                     indent=2) + "\n")
    print("SHARED_TRUNK_INSPECTION_COMPLETE", len(cases), "references", len(summaries), "shared tokens", flush=True)


if __name__ == "__main__":
    main()
