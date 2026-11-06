from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from cli.evaluate.evaluate_tcr_translate_released_top20_auxiliary import (
    PREDICTION_PROTOCOL,
    compute_official_sequence_metrics,
    load_predictions,
)
from cli.evaluate.sample_tcr_translate_released_top20_auxiliary import (
    OFFICIAL_K,
    candidate_payload,
    convert_model_sequence,
    official_beam_generation_kwargs,
    official_greedy_generation_kwargs,
)
from cli.data.materialize_tcr_translate_released_top20_auxiliary import (
    materialize_auxiliary_groups,
)


HLA_A = "A" * 34
HLA_C = "C" * 34


class FakeCodec:
    def __init__(self) -> None:
        self.codebook = {"name": "fake-frozen-codebook"}

    def encode_sequence(
        self,
        target_type,
        sequence,
        label,
        *,
        text_condition_tokens=0,
        molecule_condition_bpe_tokens=0,
    ):
        del label, text_condition_tokens, molecule_condition_bpe_tokens
        tokens = [f"{target_type}:{residue}" for residue in sequence]
        return {"full_context_tokens": len(sequence) + 3}, tokens, [10] * len(sequence)

    def bpe_ids(self, tokens, label):
        del label
        return [100 + index for index, _token in enumerate(tokens)]


class Decoder:
    def decode(self, raw_ids):
        if raw_ids == [9]:
            raise ValueError("synthetic decode failure")
        return "CASS"


class FakeReleasedEvaluator:
    @staticmethod
    def _precision_at_k(translations, references, k=None):
        del k
        return sum(value in references for value in translations) / len(translations)

    @staticmethod
    def _recall_at_k(translations, references, k):
        hits = {value for value in translations if value in references}
        return len(hits) / min(k, len(references))

    @classmethod
    def _f1_at_k(cls, translations, references, k=None):
        precision = cls._precision_at_k(translations, references, k)
        recall = cls._recall_at_k(translations, references, k)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    @staticmethod
    def _sequence_bleu(translation, references, max_references, max_ngram):
        assert max_references == 20
        assert max_ngram == 4
        return float(translation in references)

    @staticmethod
    def _dataset_bleu(translations, references, max_references, max_ngram):
        assert max_references == 20
        assert max_ngram == 4
        return sum(value in refs for value, refs in zip(translations, references)) / len(translations)

    @staticmethod
    def _mean_edit_distance(translations, references):
        del references
        return sum(not value for value in translations) / len(translations)

    @staticmethod
    def _mean_sequence_recovery(translations, references):
        return sum(value in references for value in translations) / len(translations)


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(value + "\n" for value in values), encoding="utf-8")


def prediction_record() -> dict:
    translations = ["CASS", "CASS"] + [""] * (OFFICIAL_K - 2)
    candidates = [
        {
            "rank": rank,
            "official_translation": translation,
            "status": "valid" if translation else "decode_error",
        }
        for rank, translation in enumerate(translations, start=1)
    ]
    return {
        "protocol": PREDICTION_PROTOCOL,
        "track": "released_top20_validation_auxiliary",
        "headline": False,
        "id": "group-1",
        "pmhc": {"official_key": "AC_HLA-A*01:01"},
        "generation": {
            "mode": "beam",
            "k": 100,
            "num_beams": 300,
            "max_new_tokens": 25,
            "no_repeat_ngram_size": 5,
            "do_sample": False,
            "invalid_or_duplicate_filtering": False,
            "replacement_sampling": False,
        },
        "beam_candidates": candidates,
        "translations": translations,
        "greedy_translation": "CASR",
        "reference_translations": ["CASS", "CASR", "CASS"],
    }


class TcrTranslateReleasedTop20AuxiliaryTests(unittest.TestCase):
    def test_group_materializer_preserves_reference_order_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            target = root / "target.txt"
            hla_map = root / "hla.csv"
            topk = root / "topk.csv"
            output = root / "groups.jsonl"
            write_lines(
                source,
                [f"AC {HLA_A}", f"AC {HLA_A}", f"CD {HLA_C}", f"CD {HLA_C}"],
            )
            write_lines(target, ["CASS", "CASS", "CASR", "CAST"])
            with hla_map.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["allele", "pseudo-sequence"])
                writer.writeheader()
                writer.writerow({"allele": "HLA-A*01:01", "pseudo-sequence": HLA_A})
                writer.writerow({"allele": "HLA-C*01:01", "pseudo-sequence": HLA_C})
            with topk.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Epitope", "Allele"])
                writer.writeheader()
                writer.writerow({"Epitope": "AC", "Allele": "HLA-A*01:01"})
                writer.writerow({"Epitope": "CD", "Allele": "HLA-C*01:01"})

            manifest = materialize_auxiliary_groups(
                source_path=source,
                target_path=target,
                topk_table=topk,
                hla_map_path=hla_map,
                output_path=output,
                codec=FakeCodec(),
                expected_rows=4,
                expected_groups=2,
                expected_duplicate_rows=1,
            )
            rows = [json.loads(raw) for raw in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["references"], 4)
            self.assertFalse(manifest["headline"])
            self.assertEqual(rows[0]["reference_translations"], ["CASS", "CASS"])
            self.assertEqual(
                set(rows[0]["condition"]), {"type", "molecule_bpe_ids"}
            )
            self.assertEqual(rows[0]["pmhc"]["official_key"], "AC_HLA-A*01:01")

    def test_generation_contract_is_exact_official_beam_and_greedy(self) -> None:
        beam = official_beam_generation_kwargs()
        self.assertEqual(beam["num_beams"], 300)
        self.assertEqual(beam["num_return_sequences"], 100)
        self.assertEqual(beam["max_new_tokens"], 25)
        self.assertEqual(beam["no_repeat_ngram_size"], 5)
        self.assertFalse(beam["do_sample"])
        greedy = official_greedy_generation_kwargs()
        self.assertEqual(greedy["num_beams"], 1)
        self.assertEqual(greedy["num_return_sequences"], 1)
        self.assertEqual(greedy["max_new_tokens"], 25)

    def test_invalid_candidate_keeps_rank_as_empty_official_hypothesis(self) -> None:
        invalid = candidate_payload(
            [9],
            ended_with_eos=True,
            invalid_model_id=False,
            decoder=Decoder(),
            rank=7,
        )
        self.assertEqual(invalid["rank"], 7)
        self.assertEqual(invalid["status"], "decode_error")
        self.assertEqual(invalid["official_translation"], "")
        self.assertEqual(invalid["bpe_ids"], [9])
        raw, ended, bad_id = convert_model_sequence([13, 14, 2, 0])
        self.assertEqual(raw, [10, 11])
        self.assertTrue(ended)
        self.assertFalse(bad_id)

    def test_metric_adapter_preserves_duplicate_and_invalid_slots(self) -> None:
        record = prediction_record()
        result = compute_official_sequence_metrics(
            [record], FakeReleasedEvaluator
        )
        self.assertAlmostEqual(result["dataset_metrics"]["precision"], 0.02)
        self.assertAlmostEqual(result["dataset_metrics"]["recall"], 1 / 3)
        self.assertAlmostEqual(result["dataset_metrics"]["diversity"], 0.02)
        self.assertIsNone(result["dataset_metrics"]["perplexity"])
        self.assertEqual(
            result["denominators"]["reference_rows_including_duplicates"], 3
        )

    def test_prediction_loader_accepts_duplicate_outputs_but_not_missing_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(prediction_record()) + "\n", encoding="utf-8")
            loaded = load_predictions(path, expected_groups=1)
            self.assertEqual(loaded[0]["translations"][:2], ["CASS", "CASS"])
            broken = prediction_record()
            broken["translations"].pop()
            path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 100 translations"):
                load_predictions(path, expected_groups=1)


if __name__ == "__main__":
    unittest.main()
