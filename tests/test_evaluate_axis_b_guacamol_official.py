#!/usr/bin/env python3
"""Synthetic smoke tests for the official GuacaMol Axis adapter."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any

from evaluate_axis_b_guacamol_official import (
    BENCHMARK_KEYS,
    OfficialAPI,
    SampleBatch,
    make_official_generator,
    run_official_distribution_suite,
    validate_official_asset_files,
    validate_official_source_checkout,
    validate_official_training_reference,
)


class FakeOfficialDistributionMatchingGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate(self, number_samples: int) -> list[str]:
        raise NotImplementedError


@dataclass
class FakeResult:
    benchmark_name: str
    score: float
    sampling_time: float
    metadata: dict[str, Any]


class FakeBenchmark:
    def __init__(self, name: str, call_sizes: list[int], score: float) -> None:
        self.name = name
        self.call_sizes = call_sizes
        self.score = score

    def assess_model(self, model: FakeOfficialDistributionMatchingGenerator) -> FakeResult:
        observed: list[int] = []
        for size in self.call_sizes:
            observed.append(len(model.generate(size)))
        metadata: dict[str, Any] = {"observed_call_sizes": observed}
        if self.name == "Frechet ChemNet Distance":
            metadata.update(
                {
                    "FCD": 4.25,
                    "number_reference_molecules": 7,
                    "number_generated_molecules": 7,
                }
            )
        return FakeResult(self.name, self.score, 0.01, metadata)


def fake_suite_factory(
    *, chembl_file_path: str, version_name: str, number_samples: int
) -> list[FakeBenchmark]:
    assert Path(chembl_file_path).is_file()
    assert version_name == "v1"
    assert number_samples == 7
    return [
        FakeBenchmark("Validity", [7], 0.1),
        FakeBenchmark("Uniqueness", [7, 3, 1], 0.2),
        FakeBenchmark("Novelty", [7, 6], 0.3),
        FakeBenchmark("KL divergence", [7, 2], 0.4),
        FakeBenchmark("Frechet ChemNet Distance", [7, 4, 1], 0.5),
    ]


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_provenance = validate_official_source_checkout(root)
        assert source_provenance["commit"] is None
        assert source_provenance["source_file_names_used_as_gate"] is False
        assert all(
            item["present"] is False
            and item["sha256_used_as_gate"] is False
            for item in source_provenance["verified_files"].values()
        )
        training_file = root / "guacamol_v1_train.smiles"
        training_file.write_text("C\nCC\nCCC\n", encoding="utf-8")
        expected_md5 = hashlib.md5(
            training_file.read_bytes(), usedforsecurity=False
        ).hexdigest()
        provenance = validate_official_training_reference(
            training_file, expected_md5=expected_md5
        )
        assert provenance["byte_identical_official_reference"]
        assert provenance["nonempty_records"] == 3
        mismatched_reference = validate_official_training_reference(
            training_file, expected_md5="0" * 32, expected_sha256="0" * 64
        )
        assert mismatched_reference["md5_matches_declared"] is False
        assert mismatched_reference["md5_used_as_gate"] is False
        assert mismatched_reference["sha256_matches_declared"] is False
        assert mismatched_reference["sha256_used_as_gate"] is False

        guacamol_wheel = root / "guacamol.whl"
        fcd_wheel = root / "fcd.whl"
        chemnet = root / "chemnet.h5"
        guacamol_wheel.write_bytes(b"synthetic guacamol wheel")
        fcd_wheel.write_bytes(b"synthetic fcd wheel")
        chemnet.write_bytes(b"synthetic ChemNet")
        staged = validate_official_asset_files(
            guacamol_wheel=guacamol_wheel,
            fcd_wheel=fcd_wheel,
            chemnet_h5=chemnet,
        )
        assert all(
            item["sha256_matches_declared"] is False
            and item["sha256_used_as_gate"] is False
            for item in staged.values()
        )
        try:
            validate_official_asset_files(
                guacamol_wheel=root / "missing.whl",
                fcd_wheel=fcd_wheel,
                chemnet_h5=chemnet,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing staged evaluator asset was accepted")

        sampler_calls: list[tuple[int, int]] = []

        def sampler(number_samples: int, seed: int) -> SampleBatch:
            sampler_calls.append((number_samples, seed))
            values = ["C" if index % 2 == 0 else "" for index in range(number_samples)]
            return SampleBatch(values, {"synthetic_seed": seed})

        api = OfficialAPI(
            version="0.5.5",
            generator_base=FakeOfficialDistributionMatchingGenerator,
            suite_factory=fake_suite_factory,
            code_provenance={"synthetic": True},
        )
        mismatched_provenance = dict(provenance)
        mismatched_provenance["path"] = str(root / "different.smiles")
        provenance_only_run = run_official_distribution_suite(
            sampler=sampler,
            training_file=training_file,
            reference_provenance=mismatched_provenance,
            base_seed=100,
            official_api=api,
            number_samples=7,
        )
        assert provenance_only_run["protocol"]["reference"]["path"] == str(
            root / "different.smiles"
        )

        direct = make_official_generator(
            api, sampler, benchmark_key="validity", benchmark_seed=10
        )
        assert isinstance(direct, FakeOfficialDistributionMatchingGenerator)
        assert len(direct.generate(2)) == 2
        assert len(direct.generate(1)) == 1
        direct_audit = direct.audit()
        assert direct_audit["raw_attempts"] == 3
        rng = random.Random(10)
        assert [item["seed"] for item in direct_audit["generate_calls"]] == [rng.getrandbits(63) for _ in range(2)]

        sampler_calls.clear()
        payload = run_official_distribution_suite(
            sampler=sampler,
            training_file=training_file,
            reference_provenance=provenance,
            base_seed=100,
            official_api=api,
            number_samples=7,
            benchmark_seed_stride=1_000,
        )
        assert tuple(payload["results"]) == BENCHMARK_KEYS
        assert payload["official_scores"] == {
            "validity": 0.1,
            "uniqueness": 0.2,
            "novelty": 0.3,
            "kl": 0.4,
            "fcd": 0.5,
        }
        expected_calls = {
            "validity": ([7], 100),
            "uniqueness": ([7, 3, 1], 1100),
            "novelty": ([7, 6], 2100),
            "kl": ([7, 2], 3100),
            "fcd": ([7, 4, 1], 4100),
        }
        for key, (sizes, benchmark_seed) in expected_calls.items():
            rng = random.Random(benchmark_seed)
            seeds = [rng.getrandbits(63) for _ in sizes]
            audit = payload["results"][key]["generator_audit"]
            assert [call["requested_raw_attempts"] for call in audit["generate_calls"]] == sizes
            assert [call["seed"] for call in audit["generate_calls"]] == seeds
            assert audit["raw_attempts"] == sum(sizes)

        fcd = payload["results"]["fcd"]
        assert fcd["official_score"] == 0.5
        assert fcd["raw_fcd_distance_metadata"] == {
            "value": 4.25,
            "headline_score": False,
            "note": "Official headline FCD score is exp(-0.2 * raw distance).",
        }
        assert "FCD" not in fcd["official_metadata"]
        assert "raw Frechet ChemNet distance is metadata only" in payload["protocol"]["raw_fcd_policy"]
        json.dumps(payload)

    print("official GuacaMol Axis adapter synthetic smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
