#!/usr/bin/env python3
"""Dependency-free synthetic tests for the official MOSES Axis adapter."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest import mock

from pipeline.benchmarks.moses_numeric import count_cosine_float64

from evaluate_axis_b_moses_official import (
    ArtifactSpec,
    BIOMATRIX_METRIC_KEYS,
    FORMAL_GENERATION,
    FORMAL_SEEDS,
    GenerationAttempt,
    OFFICIAL_METRIC_KEYS,
    OfficialAPI,
    ReferenceBundle,
    StatisticsBundle,
    StoredMosesSampler,
    TASK_ID,
    aggregate_seed_results,
    configure_offline_worker_environment,
    load_reference_csv,
    parse_args,
    require_worker,
    run_protocol,
    score_seed,
    validate_chemnet,
    validate_dependency_lock,
    validate_generation_contract,
)


class FakeOfficialMetrics:
    def __init__(self, *, omit_key: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.omit_key = omit_key
        self.cos_similarity = lambda _ref, _gen: -1.0

    def get_all_metrics(self, generated: Sequence[str], **kwargs: Any) -> dict[str, float]:
        assert self.cos_similarity is count_cosine_float64
        assert list(generated) == ["C", "", "CC", "C", ""]
        assert kwargs["k"] == [1000, 10000]
        assert kwargs["test"] == ["N"]
        assert kwargs["test_scaffolds"] == ["O"]
        assert kwargs["train"] == ["C"]
        assert kwargs["ptest"] == {"synthetic": "test"}
        assert kwargs["ptest_scaffolds"] == {"synthetic": "test_sf"}
        self.calls.append({"generated": list(generated), "kwargs": kwargs})
        call_offset = len(self.calls) * 0.01
        result = {
            key: 0.1 + call_offset + index * 0.001
            for index, key in enumerate(OFFICIAL_METRIC_KEYS)
        }
        if self.omit_key is not None:
            result.pop(self.omit_key)
        return result


def make_api(fake: FakeOfficialMetrics) -> OfficialAPI:
    def fraction_valid(values: Sequence[str], n_jobs: int = 1) -> float:
        assert n_jobs == 1
        return sum(bool(value) for value in values) / len(values)

    def remove_invalid(
        values: Sequence[str], canonize: bool = True, n_jobs: int = 1
    ) -> list[str]:
        assert canonize and n_jobs == 1
        return [value for value in values if value]

    def fraction_unique(
        values: Sequence[str], k: int | None = None, n_jobs: int = 1
    ) -> float:
        assert k is None and n_jobs == 1
        return len(set(values)) / len(values)

    def novelty(values: Sequence[str], train: Sequence[str], n_jobs: int = 1) -> float:
        assert n_jobs == 1
        unique = set(values)
        return len(unique.difference(train)) / len(unique)

    def internal_diversity(
        values: Sequence[str], n_jobs: int = 1, device: str = "cpu", p: int = 1
    ) -> float:
        assert values and n_jobs == 1 and device == "cuda:0"
        return 0.8 - p * 0.1

    return OfficialAPI(
        get_all_metrics=fake.get_all_metrics,
        fraction_valid=fraction_valid,
        remove_invalid=remove_invalid,
        fraction_unique=fraction_unique,
        novelty=novelty,
        internal_diversity=internal_diversity,
        get_mol=lambda value: value or None,
        metrics_module=fake,
        fcd_class=None,
        provenance={"synthetic_official_api": True},
    )


def attempts_for(seed: int) -> list[GenerationAttempt]:
    values = (
        ((11, 12), True, False, "eos", "C", None),
        ((21, 22), False, False, "dynamic_context_or_generation_limit", None, "missing_eos"),
        ((31,), True, False, "eos", "CC", None),
        ((41,), True, False, "eos", "C", None),
        ((51,), False, True, "invalid_model_id", None, "invalid_model_id"),
    )
    return [
        GenerationAttempt(
            attempt_id=f"moses:seed{seed}:attempt{index:05d}",
            seed=seed,
            attempt_index=index,
            generated_bpe_ids=tuple(raw),
            ended_with_eos=eos,
            invalid_model_id=invalid,
            stop_reason=stop,
            prompt_token_count=2,
            effective_max_new_tokens=510,
            smiles=smiles,
            decode_error=error,
        )
        for index, (raw, eos, invalid, stop, smiles, error) in enumerate(values)
    ]


class FakeSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def sample_seed(self, seed: int, number_samples: int) -> list[GenerationAttempt]:
        self.calls.append((seed, number_samples))
        assert number_samples == 5
        return attempts_for(seed)


def test_dependency_lock(root: Path) -> None:
    lock = root / "dependencies.json"
    versions = {
        "molsets": "1.0",
        "fcd-torch": "1.2.3",
        "torch": "2.0.0",
        "numpy": "1.0.0",
        "scipy": "1.0.0",
        "pandas": "2.0.0",
        "rdkit": "2025.01",
    }
    lock.write_text(
        json.dumps(
            {"schema_version": "axis-moses-dependency-lock-v1", "distributions": versions},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validated = validate_dependency_lock(lock, version_resolver=versions.__getitem__)
    assert validated["all_exact_matches"]
    assert validated["sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()

    bad_versions = dict(versions)
    bad_versions["torch"] = "different"
    mismatched = validate_dependency_lock(
        lock, version_resolver=bad_versions.__getitem__
    )
    assert not mismatched["all_exact_matches"]
    assert mismatched["version_matches"]["torch"] is False
    assert mismatched["used_as_acceptance_gate"] is False

    missing = root / "missing_dependency.json"
    incomplete = dict(versions)
    incomplete.pop("rdkit")
    missing.write_text(json.dumps({"distributions": incomplete}), encoding="utf-8")
    incomplete_provenance = validate_dependency_lock(
        missing, version_resolver=incomplete.__getitem__
    )
    assert ["rdkit", "rdkit-pypi"] in incomplete_provenance[
        "missing_required_groups"
    ]
    assert incomplete_provenance["used_as_acceptance_gate"] is False


def test_reference_hash_and_rows(root: Path) -> None:
    reference = root / "tiny.csv"
    reference.write_text("SMILES\nC\nCC\n", encoding="utf-8")
    spec = ArtifactSpec(
        name="tiny.csv",
        sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        bytes=reference.stat().st_size,
        records=2,
    )
    values, provenance = load_reference_csv(reference, spec)
    assert values == ["C", "CC"]
    assert provenance["records"] == 2
    values, mismatched = load_reference_csv(
        reference, replace(spec, sha256="0" * 64, bytes=1)
    )
    assert values == ["C", "CC"]
    assert mismatched["sha256_matches_declared"] is False
    assert mismatched["sha256_used_as_gate"] is False
    assert mismatched["bytes_match_declared"] is False
    assert mismatched["bytes_used_as_gate"] is False


def test_chemnet_hash_is_provenance(root: Path) -> None:
    chemnet = root / "synthetic_chemnet.pt"
    chemnet.write_bytes(b"synthetic model bytes")
    provenance = validate_chemnet(chemnet, "0" * 64)
    assert provenance["sha256_matches_declared"] is False
    assert provenance["sha256_used_as_gate"] is False
    without_declared_hash = validate_chemnet(chemnet)
    assert without_declared_hash["sha256_matches_declared"] is None

    try:
        validate_chemnet(root / "missing.pt")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing ChemNet model was accepted")


def test_guards() -> None:
    assert require_worker("synthetic-gpu-worker-0") == "synthetic-gpu-worker-0"
    try:
        require_worker("jump-host")
    except RuntimeError as exc:
        assert "worker-only" in str(exc)
    else:
        raise AssertionError("jump host was accepted as a formal worker")

    args = argparse.Namespace(
        max_context=512,
        max_new_tokens=512,
        condition_token_budget=512,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        molecule_condition_mode="direct_qwen",
        greedy=False,
        batch_size=4,
        metric_batch_size=8,
        n_jobs=1,
        cpu_threads=2,
    )
    validate_generation_contract(args)
    args.temperature = 0.9
    try:
        validate_generation_contract(args)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("generation protocol drift was accepted")

    os.environ["HTTPS_PROXY"] = "http://must-not-survive.invalid"
    offline = configure_offline_worker_environment()
    assert "HTTPS_PROXY" in offline["removed_proxy_variables"]
    assert "HTTPS_PROXY" not in os.environ
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_cli_hash_and_lock_inputs_are_optional() -> None:
    with mock.patch.object(
        sys,
        "argv",
        [
            "evaluate_axis_b_moses_official.py",
            "--checkpoint",
            "/checkpoint",
            "--output-root",
            "/output",
            "--chemnet",
            "/chemnet.pt",
        ],
    ):
        args = parse_args()
    assert args.chemnet_sha256 is None
    assert args.dependency_lock is None


def test_protocol(root: Path) -> None:
    sampler = FakeSampler()
    fake = FakeOfficialMetrics()
    api = make_api(fake)
    references = ReferenceBundle(
        train=["C"],
        test=["N"],
        test_scaffolds=["O"],
        provenance={"synthetic_references": True},
    )
    stats = StatisticsBundle(
        test={"synthetic": "test"},
        test_scaffolds={"synthetic": "test_sf"},
        provenance={"synthetic_statistics": True},
    )
    output = root / "moses_synthetic"
    summary = run_protocol(
        sampler=sampler,
        output_root=output,
        api=api,
        references=references,
        reference_statistics=stats,
        chemnet_path=root / "synthetic_chemnet.pt",
        chemnet_provenance={"synthetic_chemnet": True},
        axis_provenance={"synthetic_axis": True},
        offline_provenance={"synthetic_offline": True},
        device="cuda:0",
        batch_size=8,
        n_jobs=1,
        seeds=(7, 8),
        samples_per_seed=5,
        biomatrix_prefix_size=4,
        formal=False,
    )
    assert sampler.calls == [(7, 5), (8, 5)]  # exactly once; no retry
    assert len(fake.calls) == 2
    assert summary["task_id"] == "moses"
    assert summary["retry_filter_repair"] == "none"
    assert summary["aggregate"]["reported_metrics"]["MW"]["n"] == 2
    assert summary["aggregate"]["reported_metrics"]["MW"]["sd_ddof"] == 0
    assert set(summary["aggregate"]["biomatrix_first10k"]) == set(
        BIOMATRIX_METRIC_KEYS
    )
    first_view = summary["seed_results"][0]["biomatrix_first10k"]
    assert first_view["raw_prefix_records"] == 4
    assert first_view["valid_records"] == 3
    assert first_view["metrics"] == {
        "Validity": 0.75,
        "Unique": 2 / 3,
        "Novelty": 0.5,
        "IntDiv1": 0.7000000000000001,
        "IntDiv2": 0.6000000000000001,
    }

    for seed in (7, 8):
        raw_path = output / f"seed_{seed}" / "raw_attempts.jsonl"
        decoded_path = output / f"seed_{seed}" / "decoded_attempts.jsonl"
        raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        decoded_rows = [
            json.loads(line) for line in decoded_path.read_text().splitlines()
        ]
        assert len(raw_rows) == len(decoded_rows) == 5
        assert len({row["id"] for row in raw_rows}) == 5
        assert [row["id"] for row in raw_rows] == [row["id"] for row in decoded_rows]
        assert raw_rows[1]["generated_bpe_ids"] == [21, 22]
        assert decoded_rows[1]["smiles"] is None
        assert not decoded_rows[1]["valid"]
        assert decoded_rows[1]["decode_error"] == "missing_eos"
    success = json.loads((output / "_SUCCESS.json").read_text())
    assert success["raw_attempts"] == 10
    assert success["one_for_one_invalid_accounting"]

    # The official metric schema is a gate, not a best-effort subset.
    incomplete_fake = FakeOfficialMetrics(omit_key="Filters")
    incomplete_api = make_api(incomplete_fake)
    try:
        score_seed(
            attempts_for(9),
            seed=9,
            api=incomplete_api,
            references=references,
            reference_statistics=stats,
            chemnet_path=root / "synthetic_chemnet.pt",
            chemnet_provenance={"synthetic": True},
            device="cuda:0",
            batch_size=8,
            n_jobs=1,
            biomatrix_prefix_size=4,
        )
    except RuntimeError as exc:
        assert "Filters" in str(exc)
    else:
        raise AssertionError("incomplete official metric result was accepted")


def test_single_seed_aggregate() -> None:
    result = aggregate_seed_results([{
        "seed": 42, "official_metrics": {"valid": .8},
        "reported_metrics": {"valid": .8},
        "biomatrix_first10k": {"metrics": {"Validity": .8}},
    }])
    assert result["seeds"] == [42]
    assert result["official_metrics"]["valid"]["mean"] == .8
    assert result["official_metrics"]["valid"]["sd"] is None
    assert result["official_metrics"]["valid"]["n"] == 1


def test_saved_draws_use_explicit_checkpoint_relocation(root, monkeypatch) -> None:
    from cli.evaluate import eval_axis_b_samples as decode

    samples = root / "samples"
    samples.mkdir()
    saved_checkpoint = "runs_previous/task/full/best"
    (samples / "protocol.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "generation": FORMAL_GENERATION,
        "axis": {"checkpoint": saved_checkpoint},
        "seeds": [42],
    }))
    mapping = root / "paths.json"
    mapping.write_text(json.dumps({"runs_previous": str(root / "runs")}))
    monkeypatch.setenv("MOLTOKEN_PATH_MAP", str(mapping))
    args = SimpleNamespace(
        raw_samples_root=samples, checkpoint=root / "runs/task/full/best",
        seeds=[42], vocab_json=root / "vocab.json", merges_csv=root / "merges.csv",
    )
    with mock.patch.object(decode.FDDFSVocab, "load"), mock.patch.object(decode, "BPEDecoder"):
        sampler = StoredMosesSampler(args)
    assert sampler.source_axis["checkpoint"] == saved_checkpoint

    args.checkpoint = root / "runs/other-task/full/best"
    try:
        StoredMosesSampler(args)
    except ValueError as exc:
        assert "different checkpoint" in str(exc)
    else:
        raise AssertionError("saved draws from a different checkpoint were accepted")


def test_formal_aggregate() -> None:
    seed_results: list[Mapping[str, Any]] = []
    for offset, seed in enumerate(FORMAL_SEEDS):
        official = {key: float(offset) for key in OFFICIAL_METRIC_KEYS}
        reported = {
            ("MW" if key == "weight" else key): value
            for key, value in official.items()
        }
        seed_results.append(
            {
                "seed": seed,
                "official_metrics": official,
                "reported_metrics": reported,
                "biomatrix_first10k": {
                    "metrics": {key: float(offset) for key in BIOMATRIX_METRIC_KEYS}
                },
            }
        )
    aggregated = aggregate_seed_results(seed_results)
    assert aggregated["official_metrics"]["valid"]["mean"] == 1.0
    assert math.isclose(
        aggregated["official_metrics"]["valid"]["sd"], math.sqrt(2 / 3)
    )
    assert aggregated["official_metrics"]["valid"]["sd_ddof"] == 0


def main() -> int:
    test_guards()
    test_cli_hash_and_lock_inputs_are_optional()
    test_single_seed_aggregate()
    test_formal_aggregate()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        test_dependency_lock(root)
        test_reference_hash_and_rows(root)
        test_chemnet_hash_is_provenance(root)
        test_protocol(root)
    print("official MOSES Axis adapter synthetic smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
