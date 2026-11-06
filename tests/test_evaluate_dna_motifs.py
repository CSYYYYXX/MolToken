from collections import Counter

import pytest

from cli.evaluate.evaluate_dna_motifs import CONDITIONS, TASK, score_rows


class Scanner:
    matrix_ids = ["A", "B", "C"]

    def __init__(self):
        self.calls = []

    def counts(self, sequence):
        self.calls.append(sequence)
        return Counter(A=2, B=1)

    def provenance(self):
        return {"scope": "synthetic test scanner"}


def rows():
    return [{"id": f"{c}:{i}", "task": TASK, "condition": c, "sample_index": i,
             "status": "valid" if i < 2 else "missing_eos",
             "ended_with_eos": i < 2, "decoded_sequence": "A" * 200 if i < 2 else None}
            for c in CONDITIONS for i in range(3)]


def test_raw_slots_duplicates_and_zero_count_matrix():
    scanner = Scanner()
    result, slots = score_rows(rows(), scanner, samples_per_condition=3)
    assert len(scanner.calls) == 1
    assert result["requested"] == len(slots) == 9
    assert result["scan_coverage"] == pytest.approx(2 / 3)
    for group in result["per_condition"].values():
        assert group["counts_by_matrix"] == {"A": 4, "B": 2, "C": 0}
        assert group["total_motif_hits"] == 6
    assert result["official_headline_complete"] is False


def test_no_hits_is_undefined_not_perfect_diversity():
    scanner = Scanner()
    scanner.counts = lambda _seq: Counter()
    result, _ = score_rows(rows(), scanner, samples_per_condition=3)
    assert result["scan_coverage"] == pytest.approx(2 / 3)
    assert result["diagnostic_macro_plain_gini_diversity"] is None


def test_invalid_group_is_retained_not_replaced():
    data = rows()
    for row in data[:3]:
        row.update(status="decode_error", ended_with_eos=True, decoded_sequence=None)
    result, slots = score_rows(data, Scanner(), samples_per_condition=3)
    assert result["scan_coverage"] == pytest.approx(4 / 9)
    assert result["per_condition"][CONDITIONS[0]]["scan_coverage"] == 0
    assert result["diagnostic_macro_plain_gini_diversity"] is None
    assert len(slots) == 9


def test_bad_slots_fail_loudly():
    data = rows()
    data[0]["decoded_sequence"] = "A" * 199
    with pytest.raises(ValueError, match="200-bp"):
        score_rows(data, Scanner(), samples_per_condition=3)
    data = rows()
    data[1]["sample_index"] = 0
    with pytest.raises(ValueError, match="indices"):
        score_rows(data, Scanner(), samples_per_condition=3)
    with pytest.raises(ValueError, match="expected 1000"):
        score_rows(rows(), Scanner())


def test_native_failure_is_not_converted_to_invalid_generation():
    scanner = Scanner()
    def fail(_seq):
        raise RuntimeError("native scan failed")
    scanner.counts = fail
    with pytest.raises(RuntimeError, match="native scan failed"):
        score_rows(rows(), scanner, samples_per_condition=3)
