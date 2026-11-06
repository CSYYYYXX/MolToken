"""Shared E2/E3 metadata and representation definitions."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import struct

DOMAINS = ('small_molecule', 'peptide', 'protein', 'dna', 'rna')
NAMESPACE = {'small_molecule': 'smiles', 'peptide': 'aa', 'protein': 'aa',
             'dna': 'dna', 'rna': 'rna'}
HEADER = struct.Struct('<8sIQQi')
BASE_SIZE = 583
CONTENT_VOCAB = 17204
DEFAULT_EDGE = (1, 0, 0)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def iter_native(path):
    import orjson
    import zstandard
    with Path(path).open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with io.BufferedReader(stream) as lines:
            for line in lines:
                yield orjson.loads(line)


def sequence_key(row):
    # Same amino-acid string is comparable across peptide/protein; DNA/RNA
    # remain chemically distinct even when their letter strings match.
    return NAMESPACE[row['domain']] + ':' + hashlib.sha256(row['normalized'].encode()).hexdigest()


def source_key(row):
    domain = row['domain']
    if domain in ('small_molecule', 'peptide'):
        return sequence_key(row)
    source = row['source']
    accession = row['original_id']
    if domain == 'protein' and ('UniProt' in source or 'UniRef' in source):
        accession = re.sub(r'^UniRef(?:50|90|100)_', '', accession)
        source = 'uniprot'
    return NAMESPACE[domain] + ':' + source + ':' + accession


def compact_row(row, input_path, input_index):
    return dict(sample_id=row['id'], domain=row['domain'], normalized=row['normalized'],
                source=row['source'], original_id=row['original_id'],
                source_group=source_key(row), content_key=sequence_key(row),
                metadata=row.get('metadata', {}), source_version=row.get('source_version'),
                original_split=row['split'], input_path=input_path, input_index=input_index)


class IntWriter:
    def __init__(self, path, max_id):
        self.file = Path(path).open('wb')
        self.max_id = int(max_id)
        self.records = self.tokens = 0
        self.file.write(HEADER.pack(b'YTTMIBPE', 1, 0, 0, self.max_id))

    def write(self, values):
        import numpy as np
        values = np.asarray(values, dtype='<i4')
        if not len(values) or values.min() < 0 or values.max() > self.max_id:
            raise ValueError('Invalid integer token sequence')
        self.file.write(struct.pack('<I', len(values)))
        self.file.write(values.tobytes())
        self.records += 1
        self.tokens += len(values)

    def close(self):
        self.file.seek(0)
        self.file.write(HEADER.pack(b'YTTMIBPE', 1, self.records, self.tokens, self.max_id))
        self.file.close()


def guide_from_payload(payload):
    from pipeline.serialization.freq_dfs_core import FrequencyGuide
    guide = FrequencyGuide.__new__(FrequencyGuide)
    guide.node_counts = Counter(payload['node_counts'])
    guide.edge_counts = Counter(payload['edge_counts'])
    guide.triplet_counts = Counter(payload['triplet_counts'])
    guide.default_edge_feature = DEFAULT_EDGE
    return guide


class FixedOrderGuide:
    """Remove frequency terms without altering guide counts or graph grammar."""
    default_edge_feature = DEFAULT_EDGE

    def root_key(self, graph, node, degree):
        from pipeline.serialization.freq_dfs_core import compact_node_label, node_features
        return (degree, compact_node_label(node_features(graph)[node]), -node)

    def transition_key_from_features(self, src_feat, edge_feat, dst_feat, dst):
        from pipeline.serialization.freq_dfs_core import compact_node_label
        return (compact_node_label(dst_feat), -dst)

    def transition_key(self, graph, src, edge, dst):
        from pipeline.serialization.freq_dfs_core import node_features
        return self.transition_key_from_features(None, None, node_features(graph)[dst], dst)


def cpp_encoder(guide, vocab, *, fixed=False):
    import _rebuttal_fddfs_cpp
    from pipeline.serialization.stackk_fast import _items, _triplet_items, _vocab_feature_items
    mapping = vocab['token_to_id']
    encoder = _rebuttal_fddfs_cpp.FastStackKGuide(
        _items(guide.node_counts), _items(guide.edge_counts), _triplet_items(guide.triplet_counts),
        DEFAULT_EDGE, _vocab_feature_items(vocab, 'N'), _vocab_feature_items(vocab, 'E'),
        mapping['('], mapping[')'], mapping['<SEP>'], vocab['ring_base'], 1, True)
    encoder.set_frequency_ordering(not fixed)
    return encoder


def serialize_ids(encoder, graph, ring_base):
    import numpy as np
    return encoder.serialize_fddfs(
        int(graph['num_nodes']),
        np.ascontiguousarray(graph['edges'][0], dtype=np.int64),
        np.ascontiguousarray(graph['edges'][1], dtype=np.int64),
        np.ascontiguousarray(graph['node_features'], dtype=np.int64).reshape(-1, 6),
        np.ascontiguousarray(graph['edge_features'], dtype=np.int64).reshape(-1, 3),
        int(ring_base), 1)
