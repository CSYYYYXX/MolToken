#!/usr/bin/env python3
"""Run Axis-B through the official GuacaMol distribution-learning suite.

The five GuacaMol metrics invoke ``DistributionMatchingGenerator.generate``
with different retry/stopping policies.  This adapter therefore performs live
Axis sampling for every official call instead of scoring one pre-generated
file five times. Invalid Axis attempts retain empty SMILES in the decoded
ledger, but reach the official API as explicitly unparsable SMILES: upstream
canonicalize('') returns '', which its unique helper otherwise counts as a
molecule. Official GuacaMol code, not this wrapper, owns retry semantics.
Optional graph-preserving aromatic N-H completion is applied before returning
each draw; it never changes requested counts or performs adapter-side retries.

Formal execution is worker-only and fixed to GuacaMol 0.5.5, suite v1 and
10,000 samples.  File hashes are recorded only as non-blocking provenance.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence, TextIO


SCRIPT_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_ROOT.parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

DATA_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / '')
)
DEFAULT_TRAINING_FILE = (
    DATA_ROOT / "downloads_raw/guacamol_official_v1/guacamol_v1_train.smiles"
)
DEFAULT_TEXT_ENCODER = _moltoken_paths.DEFAULT_TEXT_ENCODER
DEFAULT_MERGES = _moltoken_paths.TOKENIZER_ASSET_ROOT / "merges.csv"
DEFAULT_VOCAB = _moltoken_paths.TOKENIZER_ASSET_ROOT / "vocab.json"

OFFICIAL_GUACAMOL_COMMIT = "60ebe1f6a396f16e08b834dce448e9343d259feb"
DEFAULT_OFFICIAL_SOURCE_ROOT = (
    DATA_ROOT
    / "downloads_raw/upstream_repos"
    / f"guacamol-{OFFICIAL_GUACAMOL_COMMIT}"
)
DEFAULT_OFFICIAL_ASSET_ROOT = (
    DATA_ROOT
    / "downloads_raw"
    / f"guacamol_official_evaluator_{OFFICIAL_GUACAMOL_COMMIT}"
)
DEFAULT_OFFICIAL_GUACAMOL_WHEEL = (
    DEFAULT_OFFICIAL_ASSET_ROOT / "wheels/guacamol-0.5.5-py3-none-any.whl"
)
DEFAULT_OFFICIAL_FCD_WHEEL = (
    DEFAULT_OFFICIAL_ASSET_ROOT / "wheels/FCD-1.1-py3-none-any.whl"
)
DEFAULT_OFFICIAL_CHEMNET = (
    DEFAULT_OFFICIAL_ASSET_ROOT
    / "chemnet_fcd_1p1/ChemNet_v0.13_pretrained.h5"
)

OFFICIAL_GUACAMOL_VERSION = "0.5.5"
OFFICIAL_FCD_VERSION = "1.1"
OFFICIAL_SUITE_VERSION = "v1"
OFFICIAL_NUMBER_SAMPLES = 10_000
INVALID_SMILES_PLACEHOLDER = "INVALID_SMILES"
OFFICIAL_TRAINING_MD5 = "05ad85d871958a05c02ab51a4fde8530"
OFFICIAL_TRAINING_SHA256 = (
    "3c67ee945f351dbbdc02d9016da22efaffc32a39d882021b6f213d5cd60b6a80"
)
OFFICIAL_GUACAMOL_WHEEL_SHA256 = (
    "9fea8922fa11513d605a2f97b72aa1b4b82c2109f6366b0ccb0bbdeabddaa131"
)
OFFICIAL_FCD_WHEEL_SHA256 = (
    "68ab17039ccf41d5c5e4df8db4cff90da51f487e0a7d5ed5094a1297e4fe26bf"
)
OFFICIAL_CHEMNET_H5_SHA256 = (
    "330339a30f96316c8ac9180f6dc6d9c5d507775fcc0af0eb7964d48a9438e554"
)
OFFICIAL_FCD_SOURCE_SHA256 = {
    "FCD.py": "e9e43ff7b4080637313a2f7a5532aeffdfc6a15875c084074fefb2a94ed1df13",
    "__init__.py": "18523af83512803f822e0fa98e533b411f8ded81cd8f01c86a372dbd2f5beda6",
}
OUTPUT_SCHEMA_VERSION = "axisb-guacamol-official-distribution-v1"
BENCHMARK_KEYS = (
    "validity",
    "uniqueness",
    "novelty",
    "kl",
    "fcd",
)
OFFICIAL_NAMES = {
    "Validity": "validity",
    "Uniqueness": "uniqueness",
    "Novelty": "novelty",
    "KL divergence": "kl",
    "Frechet ChemNet Distance": "fcd",
}
OFFICIAL_STOPPING_POLICIES = {
    "validity": "one generate(number_samples) call",
    "uniqueness": (
        "official sample_valid_molecules max_tries=10 while-loop; the final "
        "remaining-to-sample request may cross the nominal 10*N guard"
    ),
    "novelty": (
        "official sample_unique_molecules max_tries=2 while-loop; the final "
        "remaining-to-sample request may cross the nominal 2*N guard"
    ),
    "kl": (
        "official sample_unique_molecules max_tries=2 while-loop; the final "
        "remaining-to-sample request may cross the nominal 2*N guard"
    ),
    "fcd": (
        "official sample_valid_molecules max_tries=10 while-loop; the final "
        "remaining-to-sample request may cross the nominal 10*N guard"
    ),
}
OFFICIAL_MAX_TRIES = {
    "validity": 1,
    "uniqueness": 10,
    "novelty": 2,
    "kl": 2,
    "fcd": 10,
}
OFFICIAL_CODE_MODULES = (
    "guacamol.distribution_matching_generator",
    "guacamol.distribution_learning_benchmark",
    "guacamol.utils.sampling_helpers",
    "guacamol.utils.chemistry",
    "guacamol.utils.data",
    "guacamol.standard_benchmarks",
    "guacamol.benchmark_suites",
    "guacamol.frechet_benchmark",
)
OFFICIAL_0P5P5_SOURCE_SHA256 = {
    "guacamol.distribution_matching_generator": "4e34921ede5b145d15e59e8222b3293b9be0bf3f64203515a7a0def2b96bf880",
    "guacamol.distribution_learning_benchmark": "c528fa1be9d21edfc00adab87ce25a7652482a3c465c9cfdd960a7ff7604890a",
    "guacamol.utils.sampling_helpers": "d276d6e516bbb96ddfbcfcea21415b137a90082bd2aa296a087f303729748425",
    "guacamol.utils.chemistry": "f933811fadc3004dd7c1a709d173f8f2a377f01308dbefd1f8e45bffb46538f3",
    "guacamol.utils.data": "e9064b58413c86afa2197da186eb6591d961870428787af38eecbd44e6f1c00c",
    "guacamol.standard_benchmarks": "02782338b20dedb5dd49b077dd9c6ce3322559eea5b7b050bcec4e5a98f3e085",
    "guacamol.benchmark_suites": "5a837b25b468e0007093df88af231f608dc4d5dea8ce8df64a3d5dab259027a0",
    "guacamol.frechet_benchmark": "b6bbc43396111b9641448477444499006fde4c0aafe9d7fd9eab0b643b87ec81",
}
OFFICIAL_CHECKOUT_SOURCE_SHA256 = {
    **{
        module_name.replace(".", "/") + ".py": digest
        for module_name, digest in OFFICIAL_0P5P5_SOURCE_SHA256.items()
    },
    "setup.py": "f10a09c67ae657d60dc41614c142610b103e82586585a1946bbced170039ba91",
    "guacamol/__init__.py": "efc99fa4b7b0295922e9cf5438d4947655667bf26c37d67021fa781dfe23986d",
    "README.md": "1e6ff39862cc8ce985d9b519343a7c5644b24bd4fc74403cd4359d939473ccc6",
}


@dataclass(frozen=True)
class SampleBatch:
    """One raw Axis draw returned to an official ``generate`` call."""

    smiles: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OfficialAPI:
    version: str
    generator_base: type
    suite_factory: Callable[..., Sequence[Any]]
    code_provenance: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_official_source_checkout(root: Path) -> dict[str, Any]:
    """Observe an optional source checkout without using its identity as a gate."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"official GuacaMol source root is missing: {resolved}")
    has_git = (resolved / ".git").is_dir()
    commit = _git_output(resolved, "rev-parse", "HEAD") if has_git else None
    dirty = (
        _git_output(resolved, "status", "--porcelain", "--untracked-files=no")
        if has_git
        else None
    )
    files: dict[str, Any] = {}
    for relative, expected_sha256 in OFFICIAL_CHECKOUT_SOURCE_SHA256.items():
        path = resolved / relative
        if not path.is_file():
            files[relative] = {
                "path": str(path),
                "present": False,
                "declared_sha256": expected_sha256,
                "sha256_used_as_gate": False,
            }
            continue
        actual_sha256 = _sha256_file(path)
        files[relative] = {
            "path": str(path),
            "present": True,
            "sha256": actual_sha256,
            "declared_sha256": expected_sha256,
            "sha256_used_as_gate": False,
        }
    return {
        "path": str(resolved),
        "repository": "https://github.com/BenevolentAI/guacamol.git",
        "commit": commit,
        "expected_commit": OFFICIAL_GUACAMOL_COMMIT,
        "git_tree": (
            _git_output(resolved, "rev-parse", "HEAD^{tree}") if has_git else None
        ),
        "declared_commit": OFFICIAL_GUACAMOL_COMMIT,
        "commit_used_as_gate": False,
        "tracked_worktree_clean": None if dirty is None else not bool(dirty),
        "worktree_clean_used_as_gate": False,
        "source_file_names_used_as_gate": False,
        "verified_files": files,
    }


def validate_official_asset_files(
    *, guacamol_wheel: Path, fcd_wheel: Path, chemnet_h5: Path
) -> dict[str, Any]:
    """Require staged assets and record hashes as non-blocking provenance."""

    expected = {
        "guacamol_wheel": (
            guacamol_wheel,
            OFFICIAL_GUACAMOL_WHEEL_SHA256,
        ),
        "fcd_1p1_wheel": (fcd_wheel, OFFICIAL_FCD_WHEEL_SHA256),
        "chemnet_v0p13_h5": (chemnet_h5, OFFICIAL_CHEMNET_H5_SHA256),
    }
    payload: dict[str, Any] = {}
    for label, (raw_path, expected_sha256) in expected.items():
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned GuacaMol asset {label}: {path}")
        actual_sha256 = _sha256_file(path)
        payload[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
            "declared_sha256": expected_sha256,
            "sha256_matches_declared": actual_sha256 == expected_sha256,
            "sha256_used_as_gate": False,
        }
    return payload


def _install_scipy_histogram_compatibility_alias() -> dict[str, Any]:
    """Restore the removed SciPy alias used verbatim by GuacaMol 0.5.5.

    Old ``scipy.histogram`` was the NumPy histogram function re-exported from
    SciPy.  New SciPy releases removed that alias.  Supplying the same NumPy
    function lets the byte-identical official source import without changing
    its evaluator code or numerical operation.
    """

    import numpy
    import scipy

    installed = False
    if not hasattr(scipy, "histogram"):
        scipy.histogram = numpy.histogram  # type: ignore[attr-defined]
        installed = True
    return {
        "name": "scipy.histogram=numpy.histogram",
        "installed": installed,
        "reason": "compatibility for removed SciPy top-level re-export",
        "semantics": "the historical scipy.histogram alias delegated to numpy.histogram",
    }


def _load_official_api(
    *,
    official_source_root: Path = DEFAULT_OFFICIAL_SOURCE_ROOT,
    guacamol_wheel: Path = DEFAULT_OFFICIAL_GUACAMOL_WHEEL,
    fcd_wheel: Path = DEFAULT_OFFICIAL_FCD_WHEEL,
    chemnet_h5: Path = DEFAULT_OFFICIAL_CHEMNET,
) -> OfficialAPI:
    """Import and verify the pinned official package used for scoring."""

    source_checkout = validate_official_source_checkout(official_source_root)
    staged_assets = validate_official_asset_files(
        guacamol_wheel=guacamol_wheel,
        fcd_wheel=fcd_wheel,
        chemnet_h5=chemnet_h5,
    )

    try:
        fcd_installed = importlib.metadata.version("FCD")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "formal GuacaMol evaluation requires the staged FCD==1.1 runtime; "
            "it is not installed. Do not let pip resolve this to FCD 1.2.x, "
            "whose PyTorch .pt resource is incompatible with GuacaMol 0.5.5's "
            "hard-coded ChemNet_v0.13_pretrained.h5 loader."
        ) from exc
    if fcd_installed != OFFICIAL_FCD_VERSION:
        raise RuntimeError(
            f"formal GuacaMol evaluation requires FCD=={OFFICIAL_FCD_VERSION}; "
            f"installed={fcd_installed!r}"
        )

    scipy_compatibility = _install_scipy_histogram_compatibility_alias()
    try:
        fcd_module = importlib.import_module("fcd")
    except Exception as exc:  # noqa: BLE001 - expose dependency failure clearly
        raise RuntimeError(
            "FCD 1.1 import failed; its TensorFlow/Keras/h5py runtime must be "
            "available in the pre-staged offline runtime"
        ) from exc
    fcd_init_source = inspect.getsourcefile(fcd_module)
    if fcd_init_source is None:
        raise RuntimeError("cannot locate installed FCD 1.1 source")
    fcd_package_root = Path(fcd_init_source).resolve().parent
    fcd_files: dict[str, Any] = {}
    for relative, expected_sha256 in OFFICIAL_FCD_SOURCE_SHA256.items():
        path = fcd_package_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"installed FCD 1.1 file is missing: {path}")
        actual_sha256 = _sha256_file(path)
        fcd_files[relative] = {
            "path": str(path),
            "sha256": actual_sha256,
            "declared_sha256": expected_sha256,
            "sha256_used_as_gate": False,
        }
    installed_chemnet = fcd_package_root / "ChemNet_v0.13_pretrained.h5"
    if not installed_chemnet.is_file():
        raise FileNotFoundError(
            "installed FCD 1.1 is missing ChemNet_v0.13_pretrained.h5"
        )
    installed_chemnet_sha256 = _sha256_file(installed_chemnet)

    guacamol = importlib.import_module("guacamol")
    installed = importlib.metadata.version("guacamol")
    declared = str(getattr(guacamol, "__version__", ""))
    if installed != OFFICIAL_GUACAMOL_VERSION or declared != OFFICIAL_GUACAMOL_VERSION:
        raise RuntimeError(
            "formal GuacaMol evaluation requires guacamol=="
            f"{OFFICIAL_GUACAMOL_VERSION}; installed={installed!r}, declared={declared!r}"
        )

    generator_module = importlib.import_module(
        "guacamol.distribution_matching_generator"
    )
    suites_module = importlib.import_module("guacamol.benchmark_suites")
    installed_package_root = Path(guacamol.__file__).resolve().parent
    checkout_package_root = Path(source_checkout["path"]) / "guacamol"
    installed_tree_files: dict[str, Any] = {}
    for checkout_path in sorted(checkout_package_root.rglob("*.py")):
        relative = checkout_path.relative_to(checkout_package_root)
        installed_path = installed_package_root / relative
        installed_tree_files[str(relative)] = {
            "path": str(installed_path),
            "present": installed_path.is_file(),
            "sha256": (
                _sha256_file(installed_path) if installed_path.is_file() else None
            ),
            "checkout_sha256": _sha256_file(checkout_path),
            "sha256_used_as_gate": False,
            "file_name_used_as_gate": False,
        }
    code_files: dict[str, Any] = {}
    for module_name in OFFICIAL_CODE_MODULES:
        module = importlib.import_module(module_name)
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"cannot locate source for official module {module_name}")
        source_path = Path(source).resolve()
        actual_sha256 = _sha256_file(source_path)
        expected_sha256 = OFFICIAL_0P5P5_SOURCE_SHA256[module_name]
        code_files[module_name] = {
            "path": str(source_path),
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256,
            "sha256_used_as_gate": False,
        }
    dependency_versions: dict[str, str | None] = {}
    for distribution_name in (
        "FCD",
        "tensorflow",
        "tensorflow-cpu",
        "keras",
        "h5py",
        "rdkit",
        "rdkit-pypi",
        "numpy",
        "scipy",
        "joblib",
        "tqdm",
    ):
        try:
            dependency_versions[distribution_name] = importlib.metadata.version(
                distribution_name
            )
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution_name] = None
    return OfficialAPI(
        version=declared,
        generator_base=generator_module.DistributionMatchingGenerator,
        suite_factory=suites_module.distribution_learning_benchmark_suite,
        code_provenance={
            "distribution_name": "guacamol",
            "distribution_version": installed,
            "declared_version": declared,
            "installed_package_root": str(installed_package_root),
            "installed_package_tree": installed_tree_files,
            "modules": code_files,
            "official_source_checkout": source_checkout,
            "staged_assets": staged_assets,
            "fcd_runtime": {
                "distribution_version": fcd_installed,
                "package_root": str(fcd_package_root),
                "source_files": fcd_files,
                "chemnet": {
                    "path": str(installed_chemnet),
                    "bytes": installed_chemnet.stat().st_size,
                    "sha256": installed_chemnet_sha256,
                    "declared_sha256": OFFICIAL_CHEMNET_H5_SHA256,
                    "sha256_matches_declared": (
                        installed_chemnet_sha256 == OFFICIAL_CHEMNET_H5_SHA256
                    ),
                    "sha256_used_as_gate": False,
                    "load_validation": (
                        "official FCD benchmark loads this model during scoring"
                    ),
                },
            },
            "compatibility": {"scipy_histogram_alias": scipy_compatibility},
            "metric_dependency_versions": dependency_versions,
        },
    )


def validate_official_training_reference(
    path: Path,
    *,
    expected_md5: str = OFFICIAL_TRAINING_MD5,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a readable training reference and return byte provenance."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"GuacaMol training reference is missing: {resolved}")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    byte_count = 0
    records = 0
    blank_records = 0
    with resolved.open("rb") as handle:
        for raw_line in handle:
            md5.update(raw_line)
            sha256.update(raw_line)
            byte_count += len(raw_line)
            if raw_line.strip():
                records += 1
            else:
                blank_records += 1
    actual_md5 = md5.hexdigest()
    actual_sha256 = sha256.hexdigest()
    if expected_sha256 is None and expected_md5 == OFFICIAL_TRAINING_MD5:
        expected_sha256 = OFFICIAL_TRAINING_SHA256
    if records == 0:
        raise ValueError(f"GuacaMol training reference has no non-empty records: {resolved}")
    return {
        "path": str(resolved),
        "kind": "official_raw_guacamol_v1_train_smiles",
        "expected_md5": expected_md5,
        "md5": actual_md5,
        "md5_matches_declared": actual_md5 == expected_md5,
        "md5_used_as_gate": False,
        "expected_sha256": expected_sha256,
        "sha256": actual_sha256,
        "sha256_matches_declared": (
            None if expected_sha256 is None else actual_sha256 == expected_sha256
        ),
        "sha256_used_as_gate": False,
        "bytes": byte_count,
        "nonempty_records": records,
        "blank_records": blank_records,
        "byte_identical_official_reference": (
            actual_md5 == expected_md5
            and (expected_sha256 is None or actual_sha256 == expected_sha256)
        ),
        "byte_identity_used_as_gate": False,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite benchmark value: {value}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"cannot serialize benchmark value of type {type(value).__name__}")


class _AxisGeneratorCore:
    """Stateful implementation mixed with GuacaMol's official ABC at runtime."""

    def __init__(
        self,
        sampler: Callable[[int, int], SampleBatch | Sequence[str]],
        *,
        benchmark_key: str,
        benchmark_seed: int,
    ) -> None:
        if benchmark_key not in BENCHMARK_KEYS:
            raise ValueError(f"unknown benchmark key: {benchmark_key}")
        self._sampler = sampler
        self.benchmark_key = benchmark_key
        self.benchmark_seed = int(benchmark_seed)
        # A private stream avoids aliasing seed42/call1 with seed43/call0.
        # The sampler reseeds global Python/Torch RNGs, never this instance.
        self._call_seed_rng = random.Random(self.benchmark_seed)
        self.raw_attempts = 0
        self.generate_calls: list[dict[str, Any]] = []

    def generate(self, number_samples: int) -> list[str]:
        """Return exactly one raw string per requested attempt.

        No validity or uniqueness filtering is done here. Empty decoded
        failures map one-for-one to unparsable placeholders at this boundary;
        official GuacaMol helpers own retry and stopping behavior.
        """

        requested = int(number_samples)
        if requested <= 0:
            raise ValueError(f"number_samples must be positive, got {requested}")
        call_index = len(self.generate_calls)
        call_seed = self._call_seed_rng.getrandbits(63)
        start_attempt = self.raw_attempts
        started = time.monotonic()
        sampled = self._sampler(requested, call_seed)
        if isinstance(sampled, SampleBatch):
            smiles = list(sampled.smiles)
            backend_metadata = dict(sampled.metadata)
        else:
            smiles = list(sampled)
            backend_metadata = {}
        if len(smiles) != requested:
            raise RuntimeError(
                f"Axis sampler returned {len(smiles)} strings for {requested} raw attempts"
            )
        if any(not isinstance(smiles_value, str) for smiles_value in smiles):
            raise TypeError("Axis sampler must return SMILES strings; use '' for invalid attempts")
        self.raw_attempts += requested
        self.generate_calls.append(
            {
                "call_index": call_index,
                "seed": call_seed,
                "requested_raw_attempts": requested,
                "returned_strings": len(smiles),
                "empty_invalid_placeholders": sum(value == "" for value in smiles),
                "raw_attempt_start_inclusive": start_attempt,
                "raw_attempt_stop_exclusive": self.raw_attempts,
                "elapsed_seconds": time.monotonic() - started,
                "backend_metadata": _jsonable(backend_metadata),
            }
        )
        return [value if value else INVALID_SMILES_PLACEHOLDER for value in smiles]

    def audit(self) -> dict[str, Any]:
        initial_request = (
            int(self.generate_calls[0]["requested_raw_attempts"])
            if self.generate_calls
            else 0
        )
        nominal_guard = initial_request * OFFICIAL_MAX_TRIES[self.benchmark_key]
        return {
            "benchmark_key": self.benchmark_key,
            "benchmark_seed": self.benchmark_seed,
            "seed_policy": "private Python Random(benchmark_seed).getrandbits(63) per generate call; actual seeds recorded",
            "evaluator_invalid_placeholder": INVALID_SMILES_PLACEHOLDER,
            "empty_invalid_count_scope": "backend decoded ledger, before boundary mapping",
            "raw_attempts": self.raw_attempts,
            "official_max_tries": OFFICIAL_MAX_TRIES[self.benchmark_key],
            "nominal_while_guard_attempts": nominal_guard,
            "raw_attempts_crossed_nominal_while_guard": self.raw_attempts > nominal_guard,
            "nominal_guard_note": (
                "The official helper tests the guard before requesting all remaining "
                "molecules, so its final generate call can cross this nominal product."
            ),
            "generate_call_count": len(self.generate_calls),
            "generate_calls": _jsonable(self.generate_calls),
        }


_OFFICIAL_ADAPTER_CLASSES: dict[type, type] = {}


def _official_adapter_class(generator_base: type) -> type:
    """Create a concrete subclass of the installed official interface."""

    adapter = _OFFICIAL_ADAPTER_CLASSES.get(generator_base)
    if adapter is None:
        adapter = type(
            "AxisDistributionMatchingGenerator",
            (_AxisGeneratorCore, generator_base),
            {
                "__module__": __name__,
                "__doc__": "Axis adapter implementing official DistributionMatchingGenerator.",
            },
        )
        _OFFICIAL_ADAPTER_CLASSES[generator_base] = adapter
    return adapter


def make_official_generator(
    official_api: OfficialAPI,
    sampler: Callable[[int, int], SampleBatch | Sequence[str]],
    *,
    benchmark_key: str,
    benchmark_seed: int,
) -> _AxisGeneratorCore:
    adapter_class = _official_adapter_class(official_api.generator_base)
    generator = adapter_class(
        sampler,
        benchmark_key=benchmark_key,
        benchmark_seed=benchmark_seed,
    )
    if not isinstance(generator, official_api.generator_base):
        raise AssertionError("Axis adapter does not implement the official generator interface")
    return generator


def _normalise_result(result: Any, benchmark_key: str) -> dict[str, Any]:
    metadata = dict(getattr(result, "metadata", {}) or {})
    raw_fcd = None
    if benchmark_key == "fcd":
        if "FCD" not in metadata:
            raise RuntimeError("official Frechet result is missing raw FCD metadata")
        raw_fcd = _jsonable(metadata.pop("FCD"))
    payload = {
        "official_benchmark_name": str(result.benchmark_name),
        "official_score": _jsonable(result.score),
        "sampling_time_seconds": _jsonable(result.sampling_time),
        "official_metadata": _jsonable(metadata),
    }
    if benchmark_key == "fcd":
        payload["raw_fcd_distance_metadata"] = {
            "value": raw_fcd,
            "headline_score": False,
            "note": "Official headline FCD score is exp(-0.2 * raw distance).",
        }
    return payload


def run_official_distribution_suite(
    *,
    sampler: Callable[[int, int], SampleBatch | Sequence[str]],
    training_file: Path,
    reference_provenance: Mapping[str, Any],
    base_seed: int,
    official_api: OfficialAPI,
    number_samples: int = OFFICIAL_NUMBER_SAMPLES,
    benchmark_seed_stride: int = 1_000_003,
) -> dict[str, Any]:
    """Call all five official benchmark objects with independent adapters."""

    resolved_path = training_file.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"GuacaMol training reference is missing: {resolved_path}")
    resolved_training_file = str(resolved_path)
    benchmarks = list(
        official_api.suite_factory(
            chembl_file_path=resolved_training_file,
            version_name=OFFICIAL_SUITE_VERSION,
            number_samples=int(number_samples),
        )
    )
    if len(benchmarks) != len(BENCHMARK_KEYS):
        raise RuntimeError(
            f"official suite returned {len(benchmarks)} benchmarks, expected {len(BENCHMARK_KEYS)}"
        )

    results: dict[str, Any] = {}
    official_scores: dict[str, float] = {}
    for index, benchmark in enumerate(benchmarks):
        name = str(getattr(benchmark, "name", ""))
        key = OFFICIAL_NAMES.get(name)
        expected_key = BENCHMARK_KEYS[index]
        if key != expected_key:
            raise RuntimeError(
                f"unexpected official benchmark order/name at index {index}: {name!r}"
            )
        benchmark_seed = int(base_seed) + index * int(benchmark_seed_stride)
        generator = make_official_generator(
            official_api,
            sampler,
            benchmark_key=key,
            benchmark_seed=benchmark_seed,
        )
        result = benchmark.assess_model(generator)
        if str(result.benchmark_name) != name:
            raise RuntimeError(
                f"official result name mismatch: benchmark={name!r}, result={result.benchmark_name!r}"
            )
        normalized = _normalise_result(result, key)
        normalized["generator_audit"] = generator.audit()
        normalized["official_stopping_policy"] = OFFICIAL_STOPPING_POLICIES[key]
        results[key] = normalized
        official_scores[key] = float(result.score)

    if tuple(results) != BENCHMARK_KEYS:
        raise AssertionError(f"incomplete official results: {tuple(results)}")
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "task_id": "guacamol_distribution",
        "protocol_id": "axis-core13-official-v1-20260821",
        "formal": int(number_samples) == OFFICIAL_NUMBER_SAMPLES,
        "official_scores": official_scores,
        "results": results,
        "protocol": {
            "guacamol_version": official_api.version,
            "suite_version": OFFICIAL_SUITE_VERSION,
            "number_samples_per_benchmark": int(number_samples),
            "reference": _jsonable(reference_provenance),
            "base_seed": int(base_seed),
            "benchmark_seed_stride": int(benchmark_seed_stride),
            "benchmark_seed_policy": (
                "benchmark_seed = base_seed + official_suite_index * benchmark_seed_stride"
            ),
            "generator_policy": (
                "fresh DistributionMatchingGenerator per benchmark; raw Axis attempts are "
                "returned unfiltered, and official GuacaMol code owns retries/stopping"
            ),
            "invalid_placeholder_mapping": {
                "decoded_ledger": "", "official_generator": INVALID_SMILES_PLACEHOLDER,
                "reason": "upstream unique helper accepts canonicalize('') as an empty molecule",
                "one_for_one": True,
            },
            "raw_fcd_policy": (
                "raw Frechet ChemNet distance is metadata only; official_scores.fcd is the headline score"
            ),
            "official_code": _jsonable(official_api.code_provenance),
        },
    }


class AxisUnconditionalSampler:
    """Live worker-side Axis sampler with strict FDDFS-to-SMILES decoding."""

    def __init__(self, args: argparse.Namespace, *, raw_handle: TextIO,
                 decoded_handle: TextIO) -> None:
        if not __import__("pipeline.runtime_host", fromlist=["is_compute_host"]).is_compute_host(socket.gethostname().lower()):
            raise RuntimeError("live Axis GuacaMol sampling must run on an allocated worker")
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("live Axis GuacaMol sampling requires a scheduler-assigned GPU")
        torch.set_num_threads(args.cpu_threads)
        from pipeline.finetune.axis_b_data import AxisBSFTCollator
        from rdkit import Chem
        import sample_axis_b_sft as axis_sample
        import eval_axis_b_samples as axis_decode

        self.torch = torch
        self.Chem = Chem
        self.axis_sample = axis_sample
        self.axis_decode = axis_decode
        self.args = args
        self.raw_handle = raw_handle
        self.decoded_handle = decoded_handle
        self.call_index = 0
        source_objects = {
            "axis_sampler": axis_sample,
            "axis_decoder": axis_decode,
            "axis_collator": AxisBSFTCollator,
        }
        source_provenance: dict[str, Any] = {}
        for label, source_object in source_objects.items():
            source = inspect.getsourcefile(source_object)
            if source is None:
                raise RuntimeError(f"cannot locate source for {label}")
            source_path = Path(source).resolve()
            source_provenance[label] = {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
            }
        self.source_provenance = source_provenance
        self.device = torch.device(args.device)
        args.molecule_condition_mode = axis_sample.resolve_molecule_condition_mode(
            args.checkpoint, args.molecule_condition_mode
        )
        self.model, tokenizer = axis_sample.load_model(args, self.device)
        self.collator = AxisBSFTCollator(
            tokenizer,
            qwen_max_context=int(args.max_context),
            return_tensors="pt",
            condition_token_budget=int(args.condition_token_budget),
            molecule_condition_mode=args.molecule_condition_mode,
        )
        vocab = axis_decode.FDDFSVocab.load(args.vocab_json)
        self.vocab = vocab
        self.decoder = axis_decode.BPEDecoder(args.merges_csv, vocab)

    @staticmethod
    def _record(attempt_id: int) -> dict[str, Any]:
        # The placeholder target is discarded after the type/BOS generation
        # prompt is built.  It never enters model sampling or evaluation.
        return {
            "id": f"guacamol_live_{attempt_id}",
            "task": "guacamol_distribution",
            "target_type": "small_molecule",
            "condition": {
                "type": "none",
                "text": None,
                "molecule_bpe_ids": [],
                "modality": "none",
                "role": None,
                "instruction": None,
            },
            "meta": {
                "benchmark": "guacamol_official_live",
                "protocol_id": "axis-core13-official-v1-20260821",
            },
            "target_bpe_ids": [0],
        }

    def __call__(self, number_samples: int, seed: int) -> SampleBatch:
        torch = self.torch
        random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        call_index = self.call_index
        self.call_index += 1
        output: list[str] = []
        counts: Counter[str] = Counter()
        failure_types: Counter[str] = Counter()
        for start in range(0, int(number_samples), int(self.args.batch_size)):
            size = min(int(self.args.batch_size), int(number_samples) - start)
            records = [self._record(start + row) for row in range(size)]
            build, _collated = self.axis_sample.model_batch_from_records(
                records, self.collator, self.device
            )
            generated, ended, invalid_ids = self.axis_sample.generate_batch(
                self.model, build, self.args
            )
            raw_rows = [
                {"id": f"guacamol:call{call_index}:attempt{start + local}",
                 "call_index": call_index, "seed": int(seed),
                 "attempt_index": start + local,
                 "generated_bpe_ids": [int(value) for value in raw_ids],
                 "ended_with_eos": bool(ended_with_eos),
                 "invalid_model_id": bool(invalid_model_id)}
                for local, (raw_ids, ended_with_eos, invalid_model_id) in enumerate(
                    zip(generated, ended, invalid_ids, strict=True))
            ]
            if len(raw_rows) != size:
                raise RuntimeError(f"Axis generated {len(raw_rows)} rows for batch size {size}")
            # Persist the entire generated batch BEFORE decoding. Unexpected
            # decoder errors leave the original draws available for diagnosis.
            for row in raw_rows:
                self.raw_handle.write(json.dumps(row) + "\n")
            self.raw_handle.flush()
            for row in raw_rows:
                raw_ids = row["generated_bpe_ids"]
                ended_with_eos = row["ended_with_eos"]
                invalid_model_id = row["invalid_model_id"]
                counts["raw_attempts"] += 1
                counts["ended_with_eos"] += int(ended_with_eos)
                counts["invalid_model_id"] += int(invalid_model_id)
                smiles, error = "", None
                completion_audit = None
                if invalid_model_id:
                    error = "invalid_model_id"
                elif not ended_with_eos:
                    error = "missing_eos"
                elif not raw_ids:
                    error = "empty_generation"
                else:
                    try:
                        graph, _tokens = self.axis_decode.decode_graph(
                            raw_ids, self.decoder, self.vocab
                        )
                        counts["graph_decode_ok"] += 1
                        mol, info = self.axis_decode.graph_to_rdkit_mol(
                            graph, strict=True,
                            aromatic_n_h_completion=getattr(self.args, "aromatic_n_h_completion", False),
                        )
                        completion_audit = info.get("aromatic_n_h_completion")
                        if completion_audit is not None and completion_audit["accepted"]:
                            counts["aromatic_n_h_repaired"] += 1
                        if not bool(info.get("sanitize_ok")) or mol.GetNumAtoms() <= 0:
                            raise ValueError("strict RDKit sanitization failed")
                        smiles = self.Chem.MolToSmiles(
                            mol, canonical=True, isomericSmiles=True
                        )
                        if not smiles:
                            raise ValueError("empty canonical SMILES")
                    except ValueError as exc:
                        # The FDDFS parser and strict graph checks use ValueError
                        # for malformed draws. Runtime/import/schema errors are
                        # not benchmark events and must propagate.
                        error = f"{type(exc).__name__}:{str(exc)[:240]}"
                        failure_types[error] += 1
                output.append(smiles)
                if smiles:
                    counts["valid_smiles"] += 1
                else:
                    counts["invalid_placeholders"] += 1
                self.decoded_handle.write(json.dumps(
                    {**row, "smiles": smiles, "decode_error": error,
                     **({"aromatic_n_h_completion": completion_audit}
                        if completion_audit is not None else {})}) + "\n")
            self.decoded_handle.flush()
        if len(output) != int(number_samples):
            raise AssertionError("live Axis sampler lost raw attempts")
        return SampleBatch(
            smiles=output,
            metadata={
                "global_call_index": call_index,
                "raw_samples_file": str(self.raw_handle.name),
                "decoded_samples_file": str(self.decoded_handle.name),
                "counts": dict(counts),
                "top_decode_failures": failure_types.most_common(10),
            },
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _force_offline_worker_environment() -> dict[str, Any]:
    removed: list[str] = []
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        if key in os.environ:
            removed.append(key)
            os.environ.pop(key, None)
    forced = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    os.environ.update(forced)
    return {
        "proxy_variables_removed": removed,
        "offline_variables": forced,
        "network_downloads_allowed": False,
        "asset_policy": "all evaluator, data and model dependencies are supplied as local paths",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-file", type=Path, default=DEFAULT_TRAINING_FILE)
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--merges-csv", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--official-source-root",
        type=Path,
        default=DEFAULT_OFFICIAL_SOURCE_ROOT,
        help=(
            "local immutable checkout of the official GuacaMol repository at "
            f"commit {OFFICIAL_GUACAMOL_COMMIT}"
        ),
    )
    parser.add_argument(
        "--official-guacamol-wheel",
        type=Path,
        default=DEFAULT_OFFICIAL_GUACAMOL_WHEEL,
    )
    parser.add_argument(
        "--official-fcd-wheel",
        type=Path,
        default=DEFAULT_OFFICIAL_FCD_WHEEL,
    )
    parser.add_argument(
        "--official-chemnet-h5",
        type=Path,
        default=DEFAULT_OFFICIAL_CHEMNET,
    )
    parser.add_argument(
        "--molecule-condition-mode",
        choices=("direct_qwen",),
        default="direct_qwen",
        help="current formal GuacaMol pipeline uses the frozen FDDFS/BPE vocabulary directly",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Explicit Axis generation device; no CUDA-to-CPU fallback")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--condition-token-budget", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--aromatic-n-h-completion", action="store_true",
                        help="Graph-preserving, reference-free aromatic N-H completion before official scoring")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not __import__("pipeline.runtime_host", fromlist=["is_compute_host"]).is_compute_host(socket.gethostname().lower()):
        raise RuntimeError("formal GuacaMol evaluation must run on an allocated worker")
    offline_environment = _force_offline_worker_environment()
    for label in (
        "checkpoint",
        "training_file",
        "text_encoder",
        "merges_csv",
        "vocab_json",
        "official_source_root",
        "official_guacamol_wheel",
        "official_fcd_wheel",
        "official_chemnet_h5",
    ):
        value = Path(getattr(args, label))
        if not value.exists():
            raise FileNotFoundError(f"missing --{label.replace('_', '-')}: {value}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.greedy:
        raise ValueError("formal GuacaMol distribution learning requires stochastic sampling")

    official_api = _load_official_api(
        official_source_root=args.official_source_root,
        guacamol_wheel=args.official_guacamol_wheel,
        fcd_wheel=args.official_fcd_wheel,
        chemnet_h5=args.official_chemnet_h5,
    )
    reference = validate_official_training_reference(args.training_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite GuacaMol results: {args.output}")
    raw_path = Path(str(args.output) + ".raw.jsonl")
    decoded_path = Path(str(args.output) + ".decoded.jsonl")
    with raw_path.open("x") as raw_handle, decoded_path.open("x") as decoded_handle:
        sampler = AxisUnconditionalSampler(
            args, raw_handle=raw_handle, decoded_handle=decoded_handle)
        payload = run_official_distribution_suite(
            sampler=sampler,
            training_file=args.training_file,
            reference_provenance=reference,
            base_seed=args.seed,
            official_api=official_api,
        )
    payload["sample_artifacts"] = {
        "raw": str(raw_path.resolve()), "decoded": str(decoded_path.resolve()),
        "policy": "one row per raw attempt including invalid draws; no adapter filtering/resampling; completion policy recorded in axis",
    }
    payload["axis"] = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "merges_csv": str(args.merges_csv.expanduser().resolve()),
        "vocab_json": str(args.vocab_json.expanduser().resolve()),
        "generation": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context,
            "condition_token_budget": args.condition_token_budget,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "molecule_condition_mode": args.molecule_condition_mode,
        },
        "invalid_attempt_representation": "empty SMILES passed to official evaluator",
        "graph_to_smiles_adapter": (
            "strict FDDFS decode, strict graph reconstruction, RDKit sanitization, and canonical SMILES; "
            "failures remain one-for-one empty invalid placeholders"
        ),
        "post_generation_selection_or_repair": (
            "graph-preserving aromatic N-H completion; reference-free lexicographic choice; no filtering/resampling"
            if args.aromatic_n_h_completion else "none"
        ),
        "aromatic_n_h_completion": args.aromatic_n_h_completion,
        "offline_environment": offline_environment,
        "adapter_sources": {
            "guacamol_axis_adapter": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            **sampler.source_provenance,
        },
    }
    _atomic_write_json(args.output, payload)
    print(json.dumps(payload["official_scores"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
