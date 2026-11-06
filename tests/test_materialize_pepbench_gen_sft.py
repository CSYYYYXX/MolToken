from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cli.data.materialize_pepbench_gen_sft import (
    CONTEXT_BUDGET,
    DATASET_ID,
    EXPECTED_ELIGIBLE_RECORDS,
    EXPECTED_EXCLUDED_RECORDS,
    EXPECTED_FULL_COMPACT_ENCODED,
    EXPECTED_FULL_COMPACT_FAILED,
    EXPECTED_SOURCE_RECORDS,
    ExcludedSourceRecord,
    HISTORICAL_SAMPLED_PEPBENCH_RECORDS,
    SourceIndexRecord,
    SourceScan,
    audit_flagship_pretraining,
    classify_source_row,
    exact_length_stratified_split,
    make_sft_record,
    parse_provenance_path_mappings,
    reconcile_source_scan_with_full_graph,
    scan_source_index,
)


class PepBenchSourceAndSplitTest(unittest.TestCase):
    def test_source_row_is_classified_without_sequence_repair(self) -> None:
        payload = {
            "id": "PepBenchmark:data/example.csv:4:sequence",
            "domain": "peptide",
            "source": "PepBenchmark",
            "raw": "ACDEFG",
            "normalized": "ACDEFG",
            "length": 6,
        }
        record = classify_source_row(
            payload, path=Path("fixture.jsonl"), line_number=1
        )
        self.assertIsInstance(record, SourceIndexRecord)
        assert isinstance(record, SourceIndexRecord)
        self.assertEqual(record.length, 6)
        self.assertEqual(record.sequence, "ACDEFG")

        excluded = classify_source_row(
            {
                **payload,
                "normalized": "acdeXg",
                "raw": "acdeXg",
            },
            path=Path("fixture.jsonl"),
            line_number=2,
        )
        self.assertIsInstance(excluded, ExcludedSourceRecord)
        assert isinstance(excluded, ExcludedSourceRecord)
        self.assertEqual(excluded.original_sequence, "acdeXg")
        self.assertEqual(excluded.source_raw_sequence, "acdeXg")
        self.assertEqual(
            excluded.reasons, ("contains_noncanonical_amino_acid",)
        )
        self.assertEqual(
            excluded.noncanonical_symbols, ("X", "a", "c", "d", "e", "g")
        )

    def test_scan_audits_exclusions_before_splitting(self) -> None:
        sequences = ("ACDEF", "CDEFGH", "ACDEX", "ACDE", "A" * 81)
        rows = [
            {
                "id": f"PepBenchmark:fixture:{index}",
                "domain": "peptide",
                "source": "PepBenchmark",
                "raw": sequence,
                "normalized": sequence,
                "length": len(sequence),
            }
            for index, sequence in enumerate(sequences)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            scan = scan_source_index(
                path,
                expected_source_records=5,
                expected_eligible_records=2,
                expected_excluded_records=3,
            )
        self.assertEqual(scan.raw_records, 5)
        self.assertEqual(len(scan.eligible_records), 2)
        self.assertEqual(len(scan.excluded_records), 3)
        self.assertEqual(
            scan.exclusion_reason_counts,
            {
                "contains_noncanonical_amino_acid": 1,
                "length_above_maximum": 1,
                "length_below_minimum": 1,
            },
        )
        invalid_reference = scan.excluded_records[0].reference()
        self.assertEqual(invalid_reference["original_sequence"], "ACDEX")
        self.assertEqual(
            invalid_reference["reasons"],
            ["contains_noncanonical_amino_acid"],
        )

    def test_scan_counts_reconcile_with_full_graph_aggregate(self) -> None:
        self.assertEqual(EXPECTED_ELIGIBLE_RECORDS, EXPECTED_FULL_COMPACT_ENCODED)
        self.assertEqual(EXPECTED_EXCLUDED_RECORDS, EXPECTED_FULL_COMPACT_FAILED)
        scan = SourceScan(
            raw_records=5,
            eligible_records=(None, None),  # type: ignore[arg-type]
            excluded_records=(None, None, None),  # type: ignore[arg-type]
            exclusion_reason_counts={},
        )
        reconciliation = reconcile_source_scan_with_full_graph(
            scan,
            {
                "full_compact": {
                    "pepbenchmark_source_records_seen": 5,
                    "pepbenchmark_graph_records_encoded": 2,
                    "pepbenchmark_graph_records_failed": 3,
                }
            },
        )
        self.assertTrue(reconciliation["aggregate_counts_reconciled"])
        self.assertEqual(
            reconciliation["eligible_canonical_records"],
            2,
        )

    def test_split_is_exact_per_length_deterministic_and_order_invariant(self) -> None:
        records = [
            SourceIndexRecord(
                source_id=f"source-{length}-{index}",
                record_id=f"record-{length}-{index}",
                sequence="A" * length,
                length=length,
            )
            for length in (5, 6)
            for index in range(20)
        ]
        first, per_length = exact_length_stratified_split(
            records,
            seed=42,
            test_fraction=0.15,
            validation_fraction=0.15,
        )
        reversed_result, _ = exact_length_stratified_split(
            list(reversed(records)),
            seed=42,
            test_fraction=0.15,
            validation_fraction=0.15,
        )
        self.assertEqual(first, reversed_result)
        self.assertEqual(
            per_length,
            {
                5: {"train": 14, "validation": 3, "test": 3},
                6: {"train": 14, "validation": 3, "test": 3},
            },
        )
        self.assertEqual(set(first), {record.record_id for record in records})

    def test_record_condition_is_only_the_bare_decimal_length(self) -> None:
        row = SourceIndexRecord("source", "record", "A" * 20, 20)
        record = make_sft_record(
            row,
            split="train",
            condition_text_tokens=3,
            metrics={"fddfs_tokens": 90, "full_context_tokens": 23},
            target_bpe=[7, 8, 9],
        )
        self.assertEqual(record["condition"], {
            "type": "text",
            "text": "20",
            "molecule_bpe_ids": [],
        })
        self.assertEqual(record["target_type"], "peptide")
        self.assertEqual(record["target_bpe_ids"], [7, 8, 9])
        self.assertEqual(record["meta"]["dataset_id"], DATASET_ID)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            make_sft_record(
                row,
                split="train",
                condition_text_tokens=3,
                metrics={
                    "fddfs_tokens": 900,
                    "full_context_tokens": CONTEXT_BUDGET + 1,
                },
                target_bpe=[7],
            )


class PepBenchFlagshipExposureAuditTest(unittest.TestCase):
    def test_flagships_used_full_compact_not_historical_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "full_compact"
            compact.mkdir()
            done = compact / "done.json"
            done.write_text(
                json.dumps(
                    {
                        "format": "compact_int_shards",
                        "records": 323_402_911,
                        "tokens": 9_902_656_060,
                    }
                ),
                encoding="utf-8",
            )
            configs = []
            for name in ("small", "big"):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "args": {
                                "token_files": [str(compact / "shard_*.bin")],
                                "dataset_mode": "online-pad",
                                "compact_index_dir": str(root / "index"),
                                "val_fraction": 0.001,
                                "split_strategy": "stride",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                configs.append(path)

            graph_root = root / "graphs"
            counts = {
                "train": (70_436, 69_434, 1_002),
                "valid": (362, 357, 5),
                "test": (339, 326, 13),
            }
            for split, (seen, encoded, failed) in counts.items():
                directory = graph_root / "peptide" / split
                directory.mkdir(parents=True)
                (directory / "part-input017-00000.done.json").write_text(
                    json.dumps(
                        {
                            "counts": {
                                "seen": seen,
                                "ok": encoded,
                                "failed": failed,
                            },
                            "sources": {"PepBenchmark": encoded},
                            "input": (
                                f"/shared/data/shards_normv2_all/peptide/{split}/"
                                "part-input017-00000.jsonl.zst"
                            ),
                        }
                    ),
                    encoding="utf-8",
                )

            audit = audit_flagship_pretraining(
                small_run_config=configs[0],
                big_run_config=configs[1],
                full_compact_done=done,
                graph_root=graph_root,
            )
        self.assertEqual(audit["flagship_input_selection"], "full compact corpus")
        self.assertFalse(audit["historical_sampled_corpus_used_by_flagships"])
        self.assertEqual(
            audit["historical_sampled_pepbench_records"],
            HISTORICAL_SAMPLED_PEPBENCH_RECORDS,
        )
        compact_audit = audit["full_compact"]
        self.assertEqual(
            compact_audit["pepbenchmark_source_records_seen"],
            EXPECTED_SOURCE_RECORDS,
        )
        self.assertEqual(
            compact_audit["pepbenchmark_graph_records_encoded"],
            EXPECTED_FULL_COMPACT_ENCODED,
        )
        self.assertEqual(
            compact_audit["pepbenchmark_graph_records_failed"],
            EXPECTED_FULL_COMPACT_FAILED,
        )
        self.assertFalse(audit["unseen_generalization_claim_allowed"])

    def test_sampled_token_path_is_rejected_as_wrong_flagship_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "full_compact"
            compact.mkdir()
            done = compact / "done.json"
            done.write_text(
                json.dumps({"format": "compact_int_shards", "records": 1, "tokens": 1}),
                encoding="utf-8",
            )
            sampled = root / "sampled"
            sampled.mkdir()
            config = root / "run.json"
            config.write_text(
                json.dumps(
                    {
                        "args": {
                            "token_files": [str(sampled / "shard_*.bin")],
                            "dataset_mode": "online-pad",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "flagship token source"):
                audit_flagship_pretraining(
                    small_run_config=config,
                    big_run_config=config,
                    full_compact_done=done,
                    graph_root=root / "unused",
                )


class PepBenchProvenancePathMappingTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(
        root: Path,
        *,
        token_parent: Path,
    ) -> tuple[Path, Path, Path, Path]:
        compact = root / "local" / "full_compact"
        compact.mkdir(parents=True)
        done = compact / "done.json"
        done.write_text(
            json.dumps(
                {
                    "format": "compact_int_shards",
                    "records": 323_402_911,
                    "tokens": 9_902_656_060,
                }
            ),
            encoding="utf-8",
        )
        config = root / "run_config.json"
        config.write_text(
            json.dumps(
                {
                    "args": {
                        "token_files": [str(token_parent / "shard_*.bin")],
                        "dataset_mode": "online-pad",
                        "compact_index_dir": "/legacy/index",
                        "val_fraction": 0.001,
                        "split_strategy": "stride",
                    }
                }
            ),
            encoding="utf-8",
        )
        graph_root = root / "graphs"
        counts = {
            "train": (70_436, 69_434, 1_002),
            "valid": (362, 357, 5),
            "test": (339, 326, 13),
        }
        for split, (seen, encoded, failed) in counts.items():
            directory = graph_root / "peptide" / split
            directory.mkdir(parents=True)
            (directory / "part-input017-00000.done.json").write_text(
                json.dumps(
                    {
                        "counts": {"seen": seen, "ok": encoded, "failed": failed},
                        "sources": {"PepBenchmark": encoded},
                        "input": (
                            f"/legacy/shards_normv2_all/peptide/{split}/"
                            "part-input017-00000.jsonl.zst"
                        ),
                    }
                ),
                encoding="utf-8",
            )
        return compact, done, config, graph_root

    def test_old_absolute_path_fails_without_explicit_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_compact = Path("/legacy/gpfs/molscaling/full_compact")
            _compact, done, config, graph_root = self._write_fixture(
                root,
                token_parent=old_compact,
            )
            with self.assertRaisesRegex(ValueError, "flagship token source"):
                audit_flagship_pretraining(
                    small_run_config=config,
                    big_run_config=config,
                    full_compact_done=done,
                    graph_root=graph_root,
                )

    def test_explicit_mapping_passes_and_preserves_original_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_compact = Path("/legacy/gpfs/molscaling/full_compact")
            compact, done, config, graph_root = self._write_fixture(
                root,
                token_parent=old_compact,
            )
            original_sidecar = config.read_bytes()
            mappings = parse_provenance_path_mappings(
                [[str(old_compact), str(compact)]]
            )
            audit = audit_flagship_pretraining(
                small_run_config=config,
                big_run_config=config,
                full_compact_done=done,
                graph_root=graph_root,
                provenance_path_mappings=mappings,
            )
            self.assertEqual(config.read_bytes(), original_sidecar)

        semantics = audit["provenance_path_semantic_comparison"]
        self.assertEqual(semantics["mode"], "explicit_old_prefix_to_local_prefix")
        self.assertTrue(semantics["comparison_only"])
        self.assertFalse(semantics["sidecar_content_rewritten"])
        self.assertEqual(
            semantics["mappings"],
            [{"old_prefix": str(old_compact), "local_prefix": str(compact)}],
        )
        small = audit["flagships"]["small_0p55b"]
        self.assertEqual(
            small["token_files"], [str(old_compact / "shard_*.bin")]
        )
        self.assertEqual(small["token_file_parent_original"], str(old_compact))
        self.assertEqual(small["token_file_parent_compared"], str(compact))

    def test_mapping_does_not_match_out_of_bounds_or_prefix_lookalike(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_parent = Path("/legacy/gpfs/corpus-copy/full_compact")
            compact, done, config, graph_root = self._write_fixture(
                root,
                token_parent=token_parent,
            )
            mappings = parse_provenance_path_mappings(
                [["/legacy/gpfs/corpus", str(compact)]]
            )
            with self.assertRaisesRegex(ValueError, "outside every configured"):
                audit_flagship_pretraining(
                    small_run_config=config,
                    big_run_config=config,
                    full_compact_done=done,
                    graph_root=graph_root,
                    provenance_path_mappings=mappings,
                )

    def test_ambiguous_or_overbroad_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = str(Path(temporary) / "local")
            with self.assertRaisesRegex(ValueError, "ambiguous/overbroad"):
                parse_provenance_path_mappings([["/", local]])
            with self.assertRaisesRegex(ValueError, "contains '\\.\\.'"):
                parse_provenance_path_mappings(
                    [["/legacy/gpfs/corpus/../other", local]]
                )

    def test_overlapping_multi_match_mappings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "local"
            with self.assertRaisesRegex(ValueError, "ambiguous/multi-match"):
                parse_provenance_path_mappings(
                    [
                        ["/legacy/gpfs", str(local)],
                        ["/legacy/gpfs/corpus", str(local / "corpus")],
                    ]
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
