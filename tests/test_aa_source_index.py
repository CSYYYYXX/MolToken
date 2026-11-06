import json
import subprocess

import pytest

from pipeline.benchmarks.aa_source_index import build_index, exact_source_matches, source_rows


def catalog_fixture(tmp_path):
    sequences = ["ACDEF", "ACDEF", "XXXXX", "ACDEFG", "A" * 15]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"normalized": s, "length": len(s)}) + "\n" for s in sequences))
    return {"sources": [{"source_path": str(source), "records": len(sequences)}],
            "source_records_total": len(sequences), "reference_scope": "fixture conservative source"}


def test_index_preserves_duplicates_alphabet_and_long_counts(tmp_path):
    catalog = catalog_fixture(tmp_path)
    index = tmp_path / "index"
    result = build_index(catalog, index, maximum_length=10)
    assert result["source_records_total"] == 5
    assert result["stored_records"] == 4
    assert result["omitted_records"] == 1
    assert result["minimum_omitted_length"] == 15
    assert (index / "length_0005.txt").read_text().splitlines() == ["ACDEF", "ACDEF", "XXXXX"]
    matched, summary = exact_source_matches(["ACDEF", "ACDEF", "XXXXX", "GGGGG"], index)
    assert matched == {"ACDEF", "XXXXX"}
    assert summary["per_record_gradient_exposure_proven"] is False


def test_index_length_coverage_is_not_silent(tmp_path):
    index = tmp_path / "index"
    build_index(catalog_fixture(tmp_path), index, maximum_length=10)
    with pytest.raises(ValueError, match="outside"):
        exact_source_matches(["A" * 15], index)
    assert exact_source_matches([], index)[0] == set()
    assert exact_source_matches(["ACDEFGH"], index)[0] == set()


def test_index_incomplete_sources_fail_without_summary(tmp_path):
    catalog = catalog_fixture(tmp_path)
    catalog["sources"][0]["records"] += 1
    with pytest.raises(ValueError, match="catalog expected"):
        build_index(catalog, tmp_path / "index")
    assert not (tmp_path / "index/summary.json").exists()


def test_index_truncated_bucket_fails_loudly(tmp_path):
    index = tmp_path / "index"
    build_index(catalog_fixture(tmp_path), index)
    (index / "length_0005.txt").write_text("ACDEF\n")
    with pytest.raises(ValueError, match="index has"):
        exact_source_matches(["ACDEF"], index)


def test_actual_system_zstd_stream(tmp_path):
    source = tmp_path / "one.jsonl"
    source.write_text('{"normalized": "ACDEF", "length": 5}\n')
    compressed = tmp_path / "one.jsonl.zst"
    subprocess.run(["zstd", "-q", str(source), "-o", str(compressed)], check=True)
    assert list(source_rows(compressed)) == [{"normalized": "ACDEF", "length": 5}]
