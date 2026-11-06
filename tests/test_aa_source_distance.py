import json
import random

import pytest
from rapidfuzz.distance import Levenshtein
from pipeline.benchmarks.aa_source_index import build_index
from pipeline.benchmarks.aa_source_distance import nearest_source_distances


def index_fixture(tmp_path, references, limit=400):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"normalized": s, "length": len(s)}) + "\n" for s in references))
    catalog = {"sources": [{"source_path": str(source), "records": len(references)}],
               "source_records_total": len(references), "reference_scope": "fixture"}
    index = tmp_path / "index"
    build_index(catalog, index, maximum_length=limit)
    return index


def test_exact_pruned_distance_matches_exhaustive_native(tmp_path):
    rng = random.Random(17)
    references = ["".join(rng.choices("ACDEFGH", k=rng.randint(5, 70))) for _ in range(100)]
    references += ["XXXXX", "ACDEF", "ACDEF"]
    queries = ["".join(rng.choices("ACDEFGH", k=rng.randint(5, 80))) for _ in range(30)] + ["ACDEF", "ACDEF"]
    nearest, result = nearest_source_distances(queries, index_fixture(tmp_path, references), references[:5],
                                              workers=2, reference_batch=7, query_batch=3)
    for seq in queries:
        assert nearest[seq] == pytest.approx(min(Levenshtein.normalized_distance(seq, ref) for ref in references), abs=1e-12)
    assert nearest["ACDEF"] == 0
    assert result["complete"]


def test_omitted_long_reference_cannot_be_silently_ignored(tmp_path):
    references = ["XXXXX", "A" * 15]
    index = index_fixture(tmp_path, references, limit=10)
    with pytest.raises(ValueError, match="could improve"):
        nearest_source_distances(["A" * 5], index, references[:1])


def test_proven_long_bound_and_zero_query(tmp_path):
    references = ["AAAAA", "A" * 15]
    index = index_fixture(tmp_path, references, limit=10)
    nearest, result = nearest_source_distances(["AAAAA"], index, references[:1])
    assert nearest == {"AAAAA": 0}
    assert result["unindexed_lengths_proven_unable_to_improve"]
    assert nearest_source_distances([], index, references)[0] == {}
