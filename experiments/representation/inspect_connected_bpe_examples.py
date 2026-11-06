#!/usr/bin/env python3
"""Find connected illustrative four-merge paths in tiny reference molecules.

Run only on the retained worker. Does not load or alter a training corpus.
"""
import argparse
from collections import defaultdict
import socket

import plot_bpe_merge_hierarchy as p


def connected(graph, atoms):
    if not atoms:
        return False
    reached = {next(iter(atoms))}
    while True:
        before = len(reached)
        for u, v in zip(*graph["edges"]):
            if u in atoms and v in atoms and ({u, v} & reached):
                reached.update((u, v))
        if before == len(reached):
            return reached == atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    args = parser.parse_args()
    p.require("worker" in args.expected_host and socket.gethostname() == args.expected_host,
              "Use the confirmed worker")
    v, labels, expansions, rules = p.load_book()
    children = defaultdict(set)
    for tid, rule in rules.items():
        for parent in (rule["left"], rule["right"]):
            children[parent].add(tid)
    paths = [(42,)]
    for _ in range(4):
        paths = [path + (child,) for path in paths for child in sorted(children[path[-1]])
                 if len(expansions[child]) <= 55]
    by_raw = defaultdict(list)
    for path in paths:
        by_raw[expansions[path[-1]]].append(path)
    sizes = sorted({len(raw) for raw in by_raw})
    print("CANDIDATE_PATHS", len(paths), flush=True)

    from pipeline.finetune.fddfs_encode import build_graph, load_corpus_guide
    from pipeline.serialization.fddfs import serialize_fddfs
    guide = load_corpus_guide()
    examples = [
        "c1ccc2OCOc2c1", "c1ccc2c(c1)OCO2", "COc1ccc2c(c1)OCO2",
        "CCOc1ccc2c(c1)OCO2", "CCN1CCOCC1", "c1ccc(N2CCOCC2)cc1",
        "O=C(NCCOCC)c1ccccc1", "CCOCCOC(=O)c1ccccc1",
        "O=C(NCCNCC)c1ccccc1", "CCNCCOCC", "CCCOCCOC",
        "CCOC(=O)NCCCOC", "COC1CCCCC1", "c1ccc(CCOCC)cc1",
        "O=C1NCCO1", "O=C1NCCCC1", "C1CNCCN1", "C1COCCN1",
        "C1CCCCC1", "C1CCCCCC1", "CCCCCCCC", "CCN(CC)CCO",
    ]
    for smiles in examples:
        graph = build_graph(target_type="small_molecule", sample_id="connected_illustration", smiles=smiles)
        serialized = serialize_fddfs(graph, guide.guide)
        ids = [v["token_to_id"][raw] if raw in v["token_to_id"] else 333 + int(raw[1:])
               for raw in serialized.sequence]
        old_by_new = {new: old for old, new in serialized.old_to_new.items()}
        raw_to_atom = {}
        for raw_idx, raw in enumerate(serialized.sequence):
            if raw.startswith("N:"):
                raw_to_atom[raw_idx] = old_by_new[len(raw_to_atom)]
        candidates = []
        for size in sizes:
            for begin in range(len(ids) - size + 1):
                for path in by_raw.get(tuple(ids[begin:begin + size]), ()):
                    intervals = {path[-1]: (begin, begin + size)}
                    for parent, child in reversed(list(zip(path, path[1:]))):
                        rule = rules[child]
                        start = intervals[child][0]
                        if parent == rule["right"] and parent != rule["left"]:
                            start += len(expansions[rule["left"]])
                        intervals[parent] = (start, start + len(expansions[parent]))
                    atoms_by_stage = [{atom for raw, atom in raw_to_atom.items() if intervals[t][0] <= raw < intervals[t][1]}
                                      for t in path]
                    if not all(connected(graph, atoms) for atoms in atoms_by_stage):
                        continue
                    counts = [len(atoms) for atoms in atoms_by_stage]
                    if not all(a < b for a, b in zip(counts, counts[1:])):
                        continue
                    candidates.append((path, counts))
        if candidates:
            print("SMILES", smiles, "RAW", len(ids), flush=True)
            for path, counts in candidates[:14]:
                print("PATH", path, "ATOMS", counts, "FINAL", " ".join(p.compact(t, labels) for t in expansions[path[-1]]), flush=True)


if __name__ == "__main__":
    main()
