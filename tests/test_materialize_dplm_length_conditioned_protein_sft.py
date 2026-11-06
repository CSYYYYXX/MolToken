from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from cli.data.materialize_dplm_length_conditioned_protein_sft import (
    DATASET_ID,
    TARGET_FRAME_TOKENS,
    TARGET_LENGTHS,
    TASK,
    StreamingIntegerStats,
    assign_split,
    encode_candidate_shards,
    iter_gzip_fasta,
    materialize_uniref50,
    scan_uniref50_candidates,
)
from pipeline.finetune.frozen_axis_errors import (
    RingIdOverflow,
    UnknownFeatureError,
)


class FakeCodec:
    codebook = {"name": "fake-frozen-heavy-graph-fddfs-bpe"}

    def __init__(
        self,
        *,
        forced_bpe_length: int | None = None,
        overflow_label: str | None = None,
        ring_overflow_label: str | None = None,
        unknown_feature_label: str | None = None,
        unexpected_error_label: str | None = None,
    ) -> None:
        self.forced_bpe_length = forced_bpe_length
        self.overflow_label = overflow_label
        self.ring_overflow_label = ring_overflow_label
        self.unknown_feature_label = unknown_feature_label
        self.unexpected_error_label = unexpected_error_label
        self.calls: list[tuple[str, str, str, int]] = []

    def encode_sequence(
        self,
        target_type: str,
        sequence: str,
        label: str,
        *,
        text_condition_tokens: int = 0,
        molecule_condition_bpe_tokens: int = 0,
    ):
        self.calls.append((target_type, sequence, label, text_condition_tokens))
        if label == self.ring_overflow_label:
            raise RingIdOverflow(label, 360, 249, 583)
        if label == self.unknown_feature_label:
            raise UnknownFeatureError({"N:9,9,9,9,9,9", "E:9,9,9"})
        if label == self.unexpected_error_label:
            raise RuntimeError(f"unexpected codec failure: {label}")
        if self.overflow_label is not None and label == self.overflow_label:
            bpe_length = 510
        else:
            bpe_length = (
                self.forced_bpe_length
                if self.forced_bpe_length is not None
                else (len(sequence) + 1) // 2
            )
        bpe = [100 + (index % 100) for index in range(bpe_length)]
        full_context = text_condition_tokens + len(bpe) + TARGET_FRAME_TOKENS
        return (
            {
                "full_context_tokens": full_context,
                "bpe_tokens": len(bpe),
            },
            ["fake-fddfs"] * len(sequence),
            bpe,
        )


@dataclass(frozen=True)
class FakeCodecFactory:
    forced_bpe_length: int | None = None
    overflow_label: str | None = None
    ring_overflow_label: str | None = None
    unknown_feature_label: str | None = None
    unexpected_error_label: str | None = None

    def __call__(self) -> FakeCodec:
        return FakeCodec(
            forced_bpe_length=self.forced_bpe_length,
            overflow_label=self.overflow_label,
            ring_overflow_label=self.ring_overflow_label,
            unknown_feature_label=self.unknown_feature_label,
            unexpected_error_label=self.unexpected_error_label,
        )


class FakeIndex:
    def __init__(self, counts: dict[str, int]) -> None:
        self.files = ({"dataset": DATASET_ID},)
        self.counts = counts

    def task_counts(self, split: str):
        return {TASK: self.counts[split]}


def fake_index_builder(root: Path, *, allow_non_worker: bool):
    if not allow_non_worker:
        raise AssertionError("unit test must explicitly allow non-worker indexing")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["split_counts"]
    index_counts = {
        "train": counts["train"],
        "eval": counts["validation"],
        "test": counts["test"],
    }
    index_dir = root / "axis_b_index_v1"
    index_dir.mkdir()
    (index_dir / "index_manifest.json").write_text(
        json.dumps({"complete": True}) + "\n", encoding="utf-8"
    )
    return FakeIndex(index_counts)


def write_gzip_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with gzip.open(path, "wt", encoding="ascii", newline="") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            midpoint = len(sequence) // 2
            handle.write(sequence[:midpoint] + "\n")
            handle.write(sequence[midpoint:] + "\n")


def records_covering_every_split(per_split: int = 2) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for length in TARGET_LENGTHS:
        found = {"train": 0, "validation": 0, "test": 0}
        candidate = 0
        while min(found.values()) < per_split:
            accession = f"UniRef50_TEST_L{length}_{candidate:06d}"
            record_id = f"uniref50:{accession}"
            split = assign_split(
                record_id,
                seed=17,
                inner_validation_fraction=0.30,
                heldout_test_fraction=0.30,
            )
            if found[split] < per_split:
                residue = "ACDEFGHIKLMNPQRSTVWY"[candidate % 20]
                records.append(
                    (
                        f"{accession} synthetic record",
                        "A" * (length - 1) + residue,
                    )
                )
                found[split] += 1
            candidate += 1
    return records


class DplmLengthConditionedMaterializerTests(unittest.TestCase):
    def test_split_is_deterministic_and_uses_distinct_namespaces(self) -> None:
        record_ids = [f"uniref50:UniRef50_TEST_{index}" for index in range(1000)]
        first = [
            assign_split(
                value,
                seed=42,
                inner_validation_fraction=0.10,
                heldout_test_fraction=0.10,
            )
            for value in record_ids
        ]
        second = [
            assign_split(
                value,
                seed=42,
                inner_validation_fraction=0.10,
                heldout_test_fraction=0.10,
            )
            for value in record_ids
        ]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "test"})

    def test_streaming_stats_matches_linear_interpolated_quantiles(self) -> None:
        stats = StreamingIntegerStats()
        for value in (1, 2, 3, 4):
            stats.add(value)
        summary = stats.as_dict()
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["p50"], 2.5)
        self.assertEqual(summary["max"], 4)

    def test_fasta_parser_requires_uniref50_headers_and_nonempty_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.fasta.gz"
            write_gzip_fasta(
                valid,
                [("UniRef50_A0A000 description", "ACDEFG")],
            )
            records = list(iter_gzip_fasta(valid))
            self.assertEqual(records[0].accession, "UniRef50_A0A000")
            self.assertEqual(records[0].sequence, "ACDEFG")

            invalid = root / "invalid.fasta.gz"
            write_gzip_fasta(invalid, [("sp|A0A000| description", "ACDEFG")])
            with self.assertRaisesRegex(ValueError, "non-UniRef50"):
                list(iter_gzip_fasta(invalid))

    def test_end_to_end_streams_filters_encodes_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uniref50.fasta.gz"
            records = records_covering_every_split()
            records.extend(
                [
                    ("UniRef50_INVALID canonical-filter", "A" * 99 + "X"),
                    ("UniRef50_OTHER ignored-length", "A" * 99),
                ]
            )
            write_gzip_fasta(source, records)
            output = root / "materialized"
            manifest = materialize_uniref50(
                source_path=source,
                output_root=output,
                staging_root=root / "staging",
                codec_factory=FakeCodecFactory(),
                condition_tokens_by_length={length: 3 for length in TARGET_LENGTHS},
                candidate_shards=4,
                encode_workers=2,
                decompressor="python-gzip",
                read_threads=1,
                mp_start_method="spawn",
                inner_validation_fraction=0.30,
                heldout_test_fraction=0.30,
                split_seed=17,
                progress_every=10_000,
                allow_non_worker_index=True,
                index_builder=fake_index_builder,
            )

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["task"], TASK)
            self.assertEqual(manifest["dataset_id"], DATASET_ID)
            self.assertTrue(manifest["scope_disclosure"]["pretraining_exposed"])
            self.assertFalse(
                manifest["scope_disclosure"]["source_snapshots_identical"]
            )
            self.assertEqual(
                manifest["selection_audit"]["rejected_noncanonical_records"]["100"],
                1,
            )
            self.assertEqual(
                manifest["selection_audit"]["ignored_nonrequested_length_records"],
                1,
            )
            self.assertTrue((output / "generation_audit.json").is_file())
            self.assertTrue((output / "axis_b_index_v1/index_manifest.json").is_file())

            observed = []
            for split in ("train", "validation", "test"):
                rows = [
                    json.loads(line)
                    for line in (output / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(len(rows), manifest["split_counts"][split])
                observed.extend(rows)
                for row in rows:
                    length = row["meta"]["sequence_length"]
                    self.assertIn(length, TARGET_LENGTHS)
                    self.assertEqual(row["condition"]["text"], str(length))
                    self.assertEqual(row["condition"]["type"], "text")
                    self.assertEqual(row["target_type"], "protein")
                    self.assertFalse(row["meta"]["target_truncated"])
                    self.assertLessEqual(row["meta"]["full_context_tokens"], 512)
            self.assertEqual(len(observed), len(records) - 2)

            generation_audit = json.loads(
                (output / "generation_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(generation_audit["supported_lengths"], [100, 200, 300, 400])
            self.assertFalse(
                generation_audit["generation_acceptance"]["post_generation_repair"]
            )
            self.assertFalse(
                generation_audit["generation_acceptance"]
                ["test_references_used_for_checkpoint_selection"]
            )

    def test_context_overflow_fails_loudly_without_publishing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uniref50.fasta.gz"
            write_gzip_fasta(source, records_covering_every_split(per_split=1))
            output = root / "materialized"
            with self.assertRaisesRegex(ValueError, "exceeding 512"):
                materialize_uniref50(
                    source_path=source,
                    output_root=output,
                    staging_root=root / "staging",
                    codec_factory=FakeCodecFactory(forced_bpe_length=510),
                    condition_tokens_by_length={
                        length: 3 for length in TARGET_LENGTHS
                    },
                    candidate_shards=4,
                    encode_workers=2,
                    decompressor="python-gzip",
                    read_threads=1,
                    mp_start_method="fork",
                    inner_validation_fraction=0.30,
                    heldout_test_fraction=0.30,
                    split_seed=17,
                    progress_every=10_000,
                    allow_non_worker_index=True,
                    index_builder=fake_index_builder,
                )
            self.assertFalse(output.exists())

    def test_record_exclude_audits_overflow_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uniref50.fasta.gz"
            records = records_covering_every_split(per_split=2)
            overflow_header, _overflow_sequence = next(
                record for record in records if len(record[1]) == 400
            )
            overflow_accession = overflow_header.split(None, 1)[0]
            overflow_id = f"uniref50:{overflow_accession}"
            write_gzip_fasta(source, records)
            output = root / "materialized"
            manifest = materialize_uniref50(
                source_path=source,
                output_root=output,
                staging_root=root / "staging",
                codec_factory=FakeCodecFactory(overflow_label=overflow_id),
                condition_tokens_by_length={length: 3 for length in TARGET_LENGTHS},
                candidate_shards=4,
                encode_workers=2,
                decompressor="python-gzip",
                read_threads=1,
                mp_start_method="fork",
                inner_validation_fraction=0.30,
                heldout_test_fraction=0.30,
                split_seed=17,
                progress_every=10_000,
                encode_progress_every=0,
                context_overflow_policy="record-exclude",
                allow_non_worker_index=True,
                index_builder=fake_index_builder,
            )
            audit = manifest["context_overflow_audit"]
            self.assertEqual(audit["policy"], "record-exclude")
            self.assertEqual(audit["candidate_records"], len(records))
            self.assertEqual(audit["fit_records"], len(records) - 1)
            self.assertEqual(audit["excluded_records"], 1)
            self.assertEqual(audit["excluded_records_by_length"]["400"], 1)
            self.assertFalse(manifest["all_canonical_candidates_fit_512"])
            exclusions = [
                json.loads(line)
                for line in (output / "excluded_records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(exclusions), 1)
            self.assertEqual(exclusions[0]["id"], overflow_id)
            self.assertEqual(exclusions[0]["length"], 400)
            self.assertGreater(exclusions[0]["full_context_tokens"], 512)
            self.assertFalse(exclusions[0]["target_truncated"])
            self.assertFalse(exclusions[0]["target_repaired"])
            materialized_ids = {
                json.loads(line)["id"]
                for split in ("train", "validation", "test")
                for line in (output / f"{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            self.assertNotIn(overflow_id, materialized_ids)

    def test_representation_limit_defaults_to_fail_loud(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uniref50.fasta.gz"
            records = records_covering_every_split(per_split=1)
            header, _sequence = next(record for record in records if len(record[1]) == 400)
            record_id = f"uniref50:{header.split(None, 1)[0]}"
            write_gzip_fasta(source, records)
            output = root / "materialized"
            with self.assertRaises(RingIdOverflow):
                materialize_uniref50(
                    source_path=source,
                    output_root=output,
                    staging_root=root / "staging",
                    codec_factory=FakeCodecFactory(ring_overflow_label=record_id),
                    condition_tokens_by_length={length: 3 for length in TARGET_LENGTHS},
                    candidate_shards=4,
                    encode_workers=2,
                    decompressor="python-gzip",
                    read_threads=1,
                    mp_start_method="fork",
                    inner_validation_fraction=0.30,
                    heldout_test_fraction=0.30,
                    split_seed=17,
                    progress_every=10_000,
                    allow_non_worker_index=True,
                    index_builder=fake_index_builder,
                )
            self.assertFalse(output.exists())

    def test_record_exclude_audits_typed_representation_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uniref50.fasta.gz"
            records = records_covering_every_split(per_split=2)
            ring_header, _sequence = next(
                record for record in records if len(record[1]) == 400
            )
            unknown_header, _sequence = next(
                record for record in records if len(record[1]) == 300
            )
            ring_id = f"uniref50:{ring_header.split(None, 1)[0]}"
            unknown_id = f"uniref50:{unknown_header.split(None, 1)[0]}"
            write_gzip_fasta(source, records)
            output = root / "materialized"
            manifest = materialize_uniref50(
                source_path=source,
                output_root=output,
                staging_root=root / "staging",
                codec_factory=FakeCodecFactory(
                    ring_overflow_label=ring_id,
                    unknown_feature_label=unknown_id,
                ),
                condition_tokens_by_length={length: 3 for length in TARGET_LENGTHS},
                candidate_shards=4,
                encode_workers=2,
                decompressor="python-gzip",
                read_threads=1,
                mp_start_method="fork",
                inner_validation_fraction=0.30,
                heldout_test_fraction=0.30,
                split_seed=17,
                progress_every=10_000,
                representation_limit_policy="record-exclude",
                allow_non_worker_index=True,
                index_builder=fake_index_builder,
            )
            audit = manifest["representation_limit_audit"]
            self.assertEqual(audit["policy"], "record-exclude")
            self.assertEqual(audit["excluded_records"], 2)
            self.assertEqual(
                audit["excluded_records_by_reason"]
                ["ring_id_exceeds_frozen_vocab_capacity"],
                1,
            )
            self.assertEqual(
                audit["excluded_records_by_length_reason"]["400"]
                ["ring_id_exceeds_frozen_vocab_capacity"],
                1,
            )
            exclusions = {
                row["id"]: row
                for row in (
                    json.loads(line)
                    for line in (output / "excluded_records.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }
            self.assertEqual(exclusions[ring_id]["exception_type"], "RingIdOverflow")
            self.assertEqual(exclusions[ring_id]["max_ring_id"], 360)
            self.assertEqual(exclusions[ring_id]["ring_capacity"], 249)
            self.assertIsNone(exclusions[ring_id]["full_context_tokens"])
            self.assertEqual(
                exclusions[unknown_id]["exception_type"], "UnknownFeatureError"
            )
            self.assertEqual(exclusions[unknown_id]["unknown_feature_count"], 2)
            self.assertFalse(exclusions[ring_id]["target_truncated"])
            self.assertFalse(exclusions[unknown_id]["target_repaired"])

    def test_record_exclude_does_not_swallow_unexpected_codec_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uniref50.fasta.gz"
            records = records_covering_every_split(per_split=1)
            header, _sequence = records[0]
            record_id = f"uniref50:{header.split(None, 1)[0]}"
            write_gzip_fasta(source, records)
            with self.assertRaisesRegex(RuntimeError, "unexpected codec failure"):
                materialize_uniref50(
                    source_path=source,
                    output_root=root / "materialized",
                    staging_root=root / "staging",
                    codec_factory=FakeCodecFactory(unexpected_error_label=record_id),
                    condition_tokens_by_length={length: 3 for length in TARGET_LENGTHS},
                    candidate_shards=4,
                    encode_workers=2,
                    decompressor="python-gzip",
                    read_threads=1,
                    mp_start_method="fork",
                    inner_validation_fraction=0.30,
                    heldout_test_fraction=0.30,
                    split_seed=17,
                    progress_every=10_000,
                    context_overflow_policy="record-exclude",
                    representation_limit_policy="record-exclude",
                    allow_non_worker_index=True,
                    index_builder=fake_index_builder,
                )

    def test_scan_and_encoded_shards_resume_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uniref50.fasta.gz"
            write_gzip_fasta(source, records_covering_every_split(per_split=1))
            staging = root / "staging"
            scan = scan_uniref50_candidates(
                source_path=source,
                staging_root=staging,
                candidate_shards=4,
                inner_validation_fraction=0.30,
                heldout_test_fraction=0.30,
                split_seed=17,
                decompressor="python-gzip",
                read_threads=1,
                progress_every=10_000,
                resume=False,
            )
            scan_mtime = (staging / "scan/manifest.json").stat().st_mtime_ns
            resumed_scan = scan_uniref50_candidates(
                source_path=source,
                staging_root=staging,
                candidate_shards=4,
                inner_validation_fraction=0.30,
                heldout_test_fraction=0.30,
                split_seed=17,
                decompressor="python-gzip",
                read_threads=1,
                progress_every=10_000,
                resume=True,
            )
            self.assertEqual(scan, resumed_scan)
            self.assertEqual(
                scan_mtime,
                (staging / "scan/manifest.json").stat().st_mtime_ns,
            )

            encoded = encode_candidate_shards(
                scan_manifest=scan,
                staging_root=staging,
                codec_factory=FakeCodecFactory(),
                condition_tokens_by_length={length: 3 for length in TARGET_LENGTHS},
                context_budget=512,
                encode_workers=2,
                encode_progress_every=0,
                mp_start_method="fork",
                context_overflow_policy="fail",
                resume=False,
            )
            mtimes = {
                int(item["shard"]): (
                    staging
                    / "encoded"
                    / f"shard-{int(item['shard']):05d}"
                    / "encoded.jsonl"
                ).stat().st_mtime_ns
                for item in encoded["summaries"]
            }
            incomplete = staging / "encoded/.shard-00000.building.99999"
            incomplete.mkdir()
            (incomplete / "partial.txt").write_text("partial\n", encoding="utf-8")
            resumed_encoded = encode_candidate_shards(
                scan_manifest=resumed_scan,
                staging_root=staging,
                codec_factory=FakeCodecFactory(),
                condition_tokens_by_length={length: 3 for length in TARGET_LENGTHS},
                context_budget=512,
                encode_workers=2,
                encode_progress_every=0,
                mp_start_method="fork",
                context_overflow_policy="record-exclude",
                representation_limit_policy="record-exclude",
                resume=True,
            )
            self.assertEqual(encoded["records"], resumed_encoded["records"])
            self.assertEqual(
                resumed_encoded["context_overflow_policy"], "record-exclude"
            )
            self.assertEqual(
                resumed_encoded["representation_limit_policy"], "record-exclude"
            )
            self.assertFalse(incomplete.exists())
            for shard, mtime in mtimes.items():
                path = (
                    staging
                    / "encoded"
                    / f"shard-{shard:05d}"
                    / "encoded.jsonl"
                )
                self.assertEqual(mtime, path.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
