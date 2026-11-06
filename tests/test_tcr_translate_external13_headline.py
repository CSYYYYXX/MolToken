from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from cli.evaluate.evaluate_tcr_translate_external13_headline import (
    PREDICTION_PROTOCOL,
    compute_reference_metrics,
    load_predictions,
    native_sequence_recovery,
    parse_giana_reference_clustered,
    write_giana_input,
)
from cli.evaluate.sample_tcr_translate_external13_headline import (
    HEADLINE_K,
    headline_beam_generation_kwargs,
)
from cli.data.materialize_tcr_translate_external13_headline import (
    EXCLUDED_DEEP_SIMULATION_PMHC,
    materialize_external13_groups,
)


HLA_A = "A" * 34


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


def prediction_record() -> dict:
    translations = ["CASS", "CASS", "CASR"] + [""] * (HEADLINE_K - 3)
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
        "track": "external13_sparse_headline",
        "headline": True,
        "id": "external-group-1",
        "pmhc": {"official_key": "TEST_A*01:01"},
        "generation": {
            "mode": "beam",
            "k": 1000,
            "num_beams": 1000,
            "max_new_tokens": 25,
            "no_repeat_ngram_size": 5,
            "do_sample": False,
            "invalid_or_duplicate_filtering": False,
            "replacement_sampling": False,
            "post_hoc_sequence_repair": False,
        },
        "beam_candidates": candidates,
        "translations": translations,
        "reference_translations": ["CASS", "CATS"],
    }


class TcrTranslateExternal13HeadlineTests(unittest.TestCase):
    def test_external_materializer_excludes_deep_simulation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "benchmark.csv"
            immprep = root / "IMMPREP_test_set.csv"
            hla_map = root / "hla.csv"
            output = root / "external13.jsonl"
            with table.open("w", encoding="utf-8", newline="") as handle:
                fields = [
                    "Epitope",
                    "Allele",
                    "reference_translations",
                    "Dataset",
                    "Year",
                    "closest_training",
                    "pMHC",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "Epitope": "ACDE",
                        "Allele": "HLA-A*01:01",
                        "reference_translations": repr(["CASS"]),
                        "Dataset": "IMMREP23",
                        "Year": "2023",
                        "closest_training": "OTHER",
                        "pMHC": "ACDE_A*01:01",
                    }
                )
                writer.writerow(
                    {
                        "Epitope": "RVRAYTYSK",
                        "Allele": "HLA-A*03:01",
                        "reference_translations": repr(["CASR", "CAST"]),
                        "Dataset": "IEDB",
                        "Year": "2024",
                        "closest_training": "OTHER",
                        "pMHC": EXCLUDED_DEEP_SIMULATION_PMHC,
                    }
                )
            immprep.write_text("non-empty provenance\n", encoding="utf-8")
            with hla_map.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["allele", "pseudo-sequence"])
                writer.writeheader()
                writer.writerow({"allele": "HLA-A*01:01", "pseudo-sequence": HLA_A})
                writer.writerow({"allele": "HLA-A*03:01", "pseudo-sequence": HLA_A})

            manifest = materialize_external13_groups(
                benchmark_table=table,
                immprep_source=immprep,
                hla_map_path=hla_map,
                output_path=output,
                codec=FakeCodec(),
                expected_released_groups=2,
                expected_headline_groups=1,
                expected_released_references=3,
                expected_excluded_references=2,
                expected_headline_references=1,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(manifest["headline"])
            self.assertEqual(manifest["track"], "external13_sparse_headline")
            self.assertEqual(manifest["groups"], 1)
            self.assertEqual(rows[0]["pmhc"]["official_key"], "ACDE_A*01:01")
            self.assertEqual(rows[0]["reference_translations"], ["CASS"])

    def test_headline_generation_contract_is_1000_way_beam(self) -> None:
        kwargs = headline_beam_generation_kwargs()
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["num_beams"], 1000)
        self.assertEqual(kwargs["num_return_sequences"], 1000)
        self.assertEqual(kwargs["max_new_tokens"], 25)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 5)

    def test_exact_and_recovery_keep_duplicate_and_invalid_slots(self) -> None:
        metrics = compute_reference_metrics(prediction_record())
        self.assertEqual(metrics["generation_slots"], 1000)
        self.assertEqual(metrics["exact_reference_match_count"], 2)
        self.assertEqual(metrics["sequence_recovery_ge_0_90_count"], 2)
        self.assertEqual(metrics["canonical_nonempty_count"], 3)
        self.assertEqual(metrics["unique_nonempty_count"], 2)
        self.assertEqual(metrics["invalid_or_empty_count"], 997)
        self.assertEqual(
            native_sequence_recovery("CAAAAAAAAA", ["CAAAAAAAAT"]), 0.9
        )

    def test_giana_adapter_counts_generation_occurrences_in_reference_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "combined.tsv"
            record = prediction_record()
            record["reference_translations"] = ["CASSAAAAAA", "CASRAAAAAA"]
            record["translations"] = [
                "CASSAAAAAA",
                "CASSAAAAAA",
                "CATTAAAAAA",
                *([""] * (HEADLINE_K - 3)),
            ]
            for candidate, translation in zip(
                record["beam_candidates"], record["translations"]
            ):
                candidate["official_translation"] = translation
            audit = write_giana_input(record, input_path)
            self.assertEqual(audit["generation_slots"], 1000)
            self.assertEqual(audit["giana_eligible_generation_slots"], 3)
            # GIANA itself accepts CDR3 lengths 10--24.  Use a synthetic
            # output to test the occurrence-level cluster parser directly.
            output = root / "clusters.tsv"
            output.write_text(
                "##Column Info: CDR3 aa sequence, cluster id, other information\n"
                "CASSAAAAAA\t1\treference\tref:0000\n"
                "CASSAAAAAA\t1\tgeneration\tgen:0000\n"
                "CASSAAAAAA\t1\tgeneration\tgen:0001\n"
                "CASRAAAAAA\t2\tgeneration\tgen:0002\n",
                encoding="utf-8",
            )
            parsed = parse_giana_reference_clustered(output)
            self.assertEqual(parsed["giana_4_1_reference_clustered_count"], 2)
            self.assertEqual(parsed["giana_reference_clusters"], 1)

    def test_native_recovery_uses_edit_distance_and_stable_reference_ties(self) -> None:
        query = "ACDEFGHIKLMNPQRSTVWY"
        rotated = query[1:] + query[:1]
        two_substitutions = "YY" + query[2:]
        # Both references have edit distance 2. Official stable sorting keeps
        # the first, even though maximizing positional identity would not.
        self.assertEqual(native_sequence_recovery(query, [rotated, two_substitutions]), 0.0)
        self.assertEqual(native_sequence_recovery(query, [two_substitutions, rotated]), 0.9)

    def test_native_recovery_length_fallback_is_not_clipped(self) -> None:
        self.assertEqual(native_sequence_recovery("AAAA", ["A"]), -2.0)
        self.assertEqual(native_sequence_recovery("", ["CASS"]), 0.0)

    def test_prediction_loader_requires_every_headline_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(prediction_record()) + "\n", encoding="utf-8")
            loaded = load_predictions(path, expected_groups=1)
            self.assertEqual(len(loaded[0]["translations"]), 1000)
            broken = prediction_record()
            broken["translations"].pop()
            path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 1000 translations"):
                load_predictions(path, expected_groups=1)


if __name__ == "__main__":
    unittest.main()
