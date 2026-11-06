import pytest

from pipeline.benchmarks.dplm_tm import parse_tmalign, score_inner_tm

CONDITIONS = ("100", "200", "300", "400")


def rows():
    return [{"id": f"{c}:{i}", "condition": c, "sample_index": i,
             "status": "valid" if i < 3 else "decode_error", "folded": i < 3,
             "pdb": ("A.pdb" if i < 2 else "B.pdb") if i < 3 else None,
             "exact_length_match": i < 2}
            for c in CONDITIONS for i in range(4)]


def align(a, b):
    return {"Chain_1": 1.0 if a == b else (.2 if a == "A.pdb" else .4), "Chain_2": .9}


def test_both_diagonals_directed_scores_and_duplicate_multiplicity():
    calls = []
    def counted(a, b):
        calls.append((a, b))
        return align(a, b)
    result, slots, pairs = score_inner_tm(rows(), counted, conditions=CONDITIONS)
    assert len(calls) == len(pairs) == 4  # Both directions and both self-pairs.
    assert result["requested"] == len(slots) == 16
    assert result["fold_coverage"] == .75
    for group in result["per_condition"].values():
        metric = group["all_folded"]
        assert metric["ordered_pairs_including_self"] == 9
        assert metric["ordered_pairs_excluding_self"] == 6
        assert metric["released_inner_tm_including_self"] == pytest.approx(6.2 / 9)
        assert metric["paper_inner_tm_excluding_self"] == pytest.approx(3.2 / 6)
        assert group["exact_length_only"]["paper_inner_tm_excluding_self"] == 1
    assert result["structure_suite_complete"] is False


def test_singleton_and_missing_groups_do_not_create_perfect_diversity():
    data = rows()
    for row in data[:3]:
        row.update(folded=False, status="decode_error", pdb=None)
    data[4 + 1].update(folded=False, status="decode_error", pdb=None)
    data[4 + 2].update(folded=False, status="decode_error", pdb=None)
    result, _, _ = score_inner_tm(data, align, conditions=CONDITIONS)
    assert result["per_condition"]["100"]["all_folded"]["released_inner_tm_including_self"] is None
    assert result["per_condition"]["200"]["all_folded"]["released_inner_tm_including_self"] == 1
    assert result["per_condition"]["200"]["all_folded"]["paper_inner_tm_excluding_self"] is None
    assert result["macro"]["all_folded"]["released_inner_tm_including_self"] is None


def test_native_parser_selects_chain_one_and_fails_on_missing_result():
    output = "TM-score= 0.2 (if normalized by length of Chain_1)\nTM-score= 0.4 (if normalized by length of Chain_2)\n"
    assert parse_tmalign(output) == {"Chain_1": .2, "Chain_2": .4}
    with pytest.raises(ValueError, match="both chain"):
        parse_tmalign("WARNING: no valid structure")
    with pytest.raises(ValueError, match="outside"):
        parse_tmalign(output.replace("0.2", "1.2"))


def test_native_error_is_not_silently_averaged_away():
    def fail(a, b):
        raise RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        score_inner_tm(rows(), fail, conditions=CONDITIONS)
