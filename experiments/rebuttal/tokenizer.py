"""Frozen rebuttal tokenizer loaders; no data-dependent fitting here."""
import csv
import json
from pathlib import Path
from experiments.rebuttal.common import BASE_SIZE, CONTENT_VOCAB, NAMESPACE, cpp_encoder, serialize_ids


class BPE:
    def __init__(self, path, base_size):
        from src.algorithms.compression import _cpp_bpe
        with Path(path).open() as f:
            rows = list(csv.DictReader(f))
        self.rules = [(int(r['left']), int(r['right']), int(r['new_id'])) for r in rows]
        self.expansions = {i: (i,) for i in range(base_size)}
        for rank, (a, b, c) in enumerate(self.rules):
            if c != base_size + rank or a not in self.expansions or b not in self.expansions:
                raise ValueError(f'Invalid merge dependency: {path}:{rank}')
            self.expansions[c] = self.expansions[a] + self.expansions[b]
        self.encoder = _cpp_bpe.MinBPEEncoder([r[0] for r in self.rules], [r[1] for r in self.rules],
            list(range(len(self.rules))), [r[2] for r in self.rules])

    def encode(self, ids):
        return list(self.encoder.encode(list(map(int, ids))))

    def decode(self, ids):
        return [v for i in ids for v in self.expansions[int(i)]]


class PairedTokenizer:
    def __init__(self, run):
        from cli.pretrain.serialize_fddfs_full import load_count_payload, build_guide
        root = Path(run)
        self.vocab = json.loads((root / 'tokenizer_graphs/base_vocab.json').read_text())
        self.guide = build_guide(load_count_payload(root / 'tokenizer_graphs/graph_guide.json'))
        self.graph_encoder = cpp_encoder(self.guide, self.vocab)
        self.alphabet = json.loads((root / 'tokenizers/alphabet.json').read_text())
        self.graph_bpe = BPE(root / 'tokenizers/graph/merges.csv', BASE_SIZE)
        self.native_bpe = BPE(root / 'tokenizers/native/merges.csv', sum(map(len, self.alphabet.values())))
        if any(len(b.expansions) != CONTENT_VOCAB for b in (self.graph_bpe, self.native_bpe)):
            raise ValueError('A/C final vocabulary mismatch')

    def encode(self, row):
        from pipeline.finetune.fddfs_encode import build_graph
        value, domain = row['normalized'], row['domain']
        alphabets = {'protein': set('ACDEFGHIKLMNPQRSTVWY'), 'peptide': set('ACDEFGHIKLMNPQRSTVWY'),
                     'dna': set('ACGT'), 'rna': set('ACGU')}
        if domain in alphabets and not set(value) <= alphabets[domain]:
            return None, 'unsupported_residue_alphabet'
        mapping = self.alphabet[NAMESPACE[domain]]
        if not set(value) <= mapping.keys():
            return None, 'native_symbol_outside_fitted_alphabet'
        graph = build_graph(target_type=domain, sample_id=row.get('sample_id', row.get('id', 'encode')),
            smiles=value if domain == 'small_molecule' else None,
            sequence=value if domain != 'small_molecule' else None)
        # Features outside the frozen base grammar are an explicit common-support exclusion.
        try:
            ids = serialize_ids(self.graph_encoder, graph, self.vocab['ring_base'])['ids']
        except RuntimeError as error:
            if str(error) in ('node feature missing from vocab', 'edge feature missing from vocab'):
                return None, 'graph_feature_outside_base_grammar'
            raise
        if max(ids) >= BASE_SIZE:
            return None, 'ring_id_outside_base_grammar'
        a = self.graph_bpe.encode(ids)
        c = self.native_bpe.encode([mapping[v] for v in value])
        if not 0 < len(a) <= 509 or not 0 < len(c) <= 509:
            return None, 'outside_common_complete_context'
        return (a, c), None
