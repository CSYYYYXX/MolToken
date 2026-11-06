import unittest
from pipeline.benchmarks.rabd_metrics import rank_candidates, score_selected, aggregate_cases, score_selected_prefix


def sample(sequence, score=1., index=0):
    return {"decoded_sequence": sequence, "ended_with_eos": True,
            "invalid_model_id": False, "nll_per_requested_residue": score,
            "sample_index": index}


class RAbDMetricsTests(unittest.TestCase):
    def test_same_length_matches_upstream_accumulator(self):
        ref = "ACDE"
        rows = [sample("ACDE"), sample("AAAA")]
        # Exact accumulator in upstream rabd_test.py, with its fixed-length assumption.
        succ = tot = 0
        for row in rows:
            matches = [int(a == b) for a, b in zip(ref, row["decoded_sequence"])]
            succ += sum(matches)
            tot += len(matches)
        result = score_selected(ref, rows, 2)
        self.assertEqual(result["official_aar"], succ / tot)

    def test_short_output_never_shrinks_denominator(self):
        result = score_selected("ACDE", [sample("A"), sample("ACDE")], 2)
        self.assertIsNone(result["official_aar"])
        self.assertEqual(result["requested_residues"], 8)
        self.assertEqual(result["fixed_slot_aar_invalid_or_wrong_length_zero"], .5)

    def test_invalid_selected_slot_is_retained(self):
        rows = [sample(None), sample("AA")]
        result = score_selected("AA", rows, 2)
        self.assertEqual(result["selected"], 2)
        self.assertEqual(result["decoded_valid"], 1)
        self.assertEqual(result["fixed_slot_aar_invalid_or_wrong_length_zero"], .5)

    def test_ranking_has_no_reference_and_does_not_filter_invalid(self):
        rows = [sample("AA", 2., 0), sample(None, 1., 1), sample("AA", 1., 2)]
        self.assertEqual([r["sample_index"] for r in rank_candidates(rows, 3, 2)], [1, 2])

    def test_micro_not_macro_aggregation(self):
        a = score_selected("AA", [sample("AA")], 1)
        b = score_selected("CCCCCC", [sample("AAAAAA")], 1)
        self.assertEqual(aggregate_cases([a, b])["official_aar"], .25)

    def test_prefix_long_output_not_whole_zero(self):
        rows = [sample("ACDEFG")]
        result = score_selected_prefix("ACDE", rows, 1)
        self.assertEqual(result["prefix_aar"], 1)
        self.assertEqual(result["ignored_suffix_residues"], 2)
        self.assertEqual(rows[0]["decoded_sequence"], "ACDEFG")

    def test_prefix_short_and_invalid_do_not_shrink_denominator(self):
        result = score_selected_prefix("ACDE", [sample("AC"), sample(None)], 2)
        self.assertEqual(result["matched_residues"], 2)
        self.assertEqual(result["requested_residues"], 8)
        self.assertEqual(result["prefix_aar"], .25)

    def test_prefix_equal_length_is_original_aar(self):
        rows = [sample("ACDE"), sample("AAAA")]
        self.assertEqual(score_selected_prefix("ACDE", rows, 2)["prefix_aar"],
                         score_selected("ACDE", rows, 2)["official_aar"])

    def test_prefix_requires_valid_complete_decoding(self):
        bad_eos = sample("ACDE")
        bad_eos["ended_with_eos"] = False
        rows = [bad_eos, sample("AXDE"), sample("ACDE")]
        result = score_selected_prefix("ACDE", rows, 3)
        self.assertEqual(result["decoded_valid"], 1)
        self.assertEqual(result["prefix_aar"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
