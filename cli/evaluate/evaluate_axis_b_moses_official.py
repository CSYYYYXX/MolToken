#!/usr/bin/env python3
"""Run the complete official MOSES benchmark for Axis-B.

The protocol draws exactly 30,000 unconditional molecules per requested seed.
Use --seeds 42 for a first pass, or 42 43 44 for repeated sampling.
Every model draw occupies one output row.  Incomplete,
out-of-range, or undecodable draws become an empty SMILES for the official
MOSES evaluator; this adapter never retries or validity-filters a draw.
Optional aromatic N-H completion restores missing hydrogen annotations while
preserving the represented heavy graph. Raw BPE ids and decoded records are
retained in separate JSONL files; saved draws can be re-decoded without sampling.

Official references, MOSES statistics, ChemNet weights and the model checkpoint
are supplied as local paths. Evaluation runs without network downloads.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_ROOT.parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from pipeline.benchmarks.moses_numeric import (
    COUNT_COSINE_COMPATIBILITY, float64_count_cosine,
)

DATA_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / '')
)
RUN_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT.parent / 'runs')
)
OFFICIAL_COMMIT = "dd7ed6ab38e23afd3ef5371d67939a1760bd8599"
DEFAULT_MOSES_REFERENCE_ROOT = (
    DATA_ROOT / "benchmark/raw/moses/official_core13_dd7ed6"
)
DEFAULT_MOSES_PACKAGE_DATA = (
    DATA_ROOT / "downloads_raw" / f"MOSES-{OFFICIAL_COMMIT}" / "data"
)
DEFAULT_TRAIN = DEFAULT_MOSES_REFERENCE_ROOT / "train.csv"
DEFAULT_TEST = DEFAULT_MOSES_REFERENCE_ROOT / "test.csv"
DEFAULT_TEST_SCAFFOLDS = DEFAULT_MOSES_REFERENCE_ROOT / "test_scaffolds.csv"
DEFAULT_TEST_STATS = DEFAULT_MOSES_PACKAGE_DATA / "test_stats.npz"
DEFAULT_TEST_SCAFFOLDS_STATS = DEFAULT_MOSES_PACKAGE_DATA / "test_scaffolds_stats.npz"
DEFAULT_TEXT_ENCODER = _moltoken_paths.DEFAULT_TEXT_ENCODER
DEFAULT_MERGES = _moltoken_paths.TOKENIZER_ASSET_ROOT / "merges.csv"
DEFAULT_VOCAB = _moltoken_paths.TOKENIZER_ASSET_ROOT / "vocab.json"

TASK_ID = "moses"
PROTOCOL_ID = "axis-core13-official-v1-20260821"
OUTPUT_SCHEMA_VERSION = "axisb-moses-official-full-v1"
OFFICIAL_REPOSITORY = "molecularsets/moses"
OFFICIAL_DISTRIBUTION = "molsets"
OFFICIAL_DISTRIBUTION_VERSION = "1.0"
FORMAL_SEEDS = (42, 43, 44)
FORMAL_SAMPLES_PER_SEED = 30_000
BIOMATRIX_PREFIX = 10_000
FORMAL_GENERATION = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "beam": 1,
    "do_sample": True,
    "max_context": 512,
    "max_new_tokens_request": 512,
    "stop_policy": "min(max_new_tokens_request, max_context - active_prompt_length)",
}

OFFICIAL_METRIC_KEYS = (
    "valid",
    "unique@1000",
    "unique@10000",
    "FCD/Test",
    "SNN/Test",
    "Frag/Test",
    "Scaf/Test",
    "FCD/TestSF",
    "SNN/TestSF",
    "Frag/TestSF",
    "Scaf/TestSF",
    "IntDiv",
    "IntDiv2",
    "Filters",
    "logP",
    "SA",
    "QED",
    "weight",
    "Novelty",
)
BIOMATRIX_METRIC_KEYS = ("Validity", "Unique", "Novelty", "IntDiv1", "IntDiv2")

HIGHER_IS_BETTER = {
    "valid",
    "unique@1000",
    "unique@10000",
    "SNN/Test",
    "Frag/Test",
    "Scaf/Test",
    "SNN/TestSF",
    "Frag/TestSF",
    "Scaf/TestSF",
    "IntDiv",
    "IntDiv2",
    "Filters",
    "Novelty",
}
LOWER_IS_BETTER = {"FCD/Test", "FCD/TestSF", "logP", "SA", "QED", "weight"}


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    sha256: str
    bytes: int | None = None
    records: int | None = None


OFFICIAL_REFERENCE_SPECS = {
    "train": ArtifactSpec(
        "train.csv",
        "49fe1aae29604ec0f5023ea34edc8789415f954138f2903dabd005a4b888a961",
        67_916_778,
        1_584_663,
    ),
    "test": ArtifactSpec(
        "test.csv",
        "257165e8f1e1be2d5712db51c93d2b1544564b782dc303d7d19c887c21a858e6",
        7_370_473,
        176_074,
    ),
    "test_scaffolds": ArtifactSpec(
        "test_scaffolds.csv",
        "8bc942dd807147e07b506151fe8675f58780ed8e494801b705c4bdadb3df3440",
        9_195_363,
        176_225,
    ),
}
OFFICIAL_STATS_SPECS = {
    "test": ArtifactSpec(
        "test_stats.npz",
        "ded25073225a895d4ed17bf1d4223d27560e87ff4cb85786346ca1cc8efa03d3",
        16_781_400,
    ),
    "test_scaffolds": ArtifactSpec(
        "test_scaffolds_stats.npz",
        "7b7aba6c1474634a019bebf16f389d554cb9dd1021dfc8c1d1e54430a9f0e7f8",
        15_949_706,
    ),
}

# Byte hashes from the immutable upstream commit.  These cover the aggregator,
# helpers, dataset loader, filter definitions, and synthetic-accessibility data.
OFFICIAL_SOURCE_SHA256 = {
    "moses": "40f8222224a6a640a5698118b9784d2384a526e5e673d7b50c0059f15a5bf018",
    "moses.metrics": "dc944f30446986c1b91d649ffb95eea0022d6a5e097820837e5efe77a9d9a4f0",
    "moses.metrics.metrics": "143dace725619134950d840037b290efd93c045ae1d363ba59bfd051e7cb1ec9",
    "moses.metrics.utils": "94f1f4a4083d355de819c5e224688ab5f9307f1e16b92083a37e06f638a586a1",
    "moses.dataset": "1fbe72fcae62ef255d95d195ede10051f353c6b5e334efe84060b7bd6484b701",
    "moses.dataset.dataset": "a7a68cf828dc98d35e9b22c6193c851b206c1432426a2c396b1acdb6a9fff64a",
    "moses.utils": "6edce274b106b3fc6e45c481f171b059f33cd6d3bd6bd08db888e3f60055596f",
    "moses.metrics.SA_Score.sascorer": "ac80f7aef88c42fcf8d8419303f0a00369b643ef2666c0fe6f5c2ddb43c1df67",
}
OFFICIAL_RESOURCE_SHA256 = {
    "moses.metrics/mcf.csv": "f4a2bfacb4cb8877295eb2a4b4e61ec9770a26388f1331e5dc9c193d22dd5f89",
    "moses.metrics/wehi_pains.csv": "d762e5420ec70f06d7e41d336cae6f67e2feb0eddf05c6171192a332726d296f",
    "moses.metrics.SA_Score/fpscores.pkl.gz": "10dcef9340c873e7b987924461b0af5365eb8dd96be607203debe8ddf80c1e73",
}
REQUIRED_STATS_KEYS = {"FCD", "SNN", "Frag", "Scaf", "logP", "SA", "QED", "weight"}
REQUIRED_DEPENDENCY_GROUPS = (
    ("molsets",),
    ("fcd-torch", "fcd_torch"),
    ("torch",),
    ("numpy",),
    ("scipy",),
    ("pandas",),
    ("rdkit", "rdkit-pypi"),
)


@dataclass(frozen=True)
class GenerationAttempt:
    attempt_id: str
    seed: int
    attempt_index: int
    generated_bpe_ids: tuple[int, ...]
    ended_with_eos: bool
    invalid_model_id: bool
    stop_reason: str
    prompt_token_count: int
    effective_max_new_tokens: int
    smiles: str | None
    decode_error: str | None
    aromatic_n_h_completion: Mapping[str, Any] | None = None

    @property
    def valid(self) -> bool:
        return self.smiles is not None

    @property
    def metric_smiles(self) -> str:
        return self.smiles if self.smiles is not None else ""


@dataclass(frozen=True)
class ReferenceBundle:
    train: Sequence[str]
    test: Sequence[str]
    test_scaffolds: Sequence[str]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class StatisticsBundle:
    test: Mapping[str, Any]
    test_scaffolds: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class OfficialAPI:
    get_all_metrics: Callable[..., Mapping[str, Any]]
    fraction_valid: Callable[..., Any]
    remove_invalid: Callable[..., Sequence[str]]
    fraction_unique: Callable[..., Any]
    novelty: Callable[..., Any]
    internal_diversity: Callable[..., Any]
    get_mol: Callable[[str], Any]
    metrics_module: Any
    fcd_class: type | None
    provenance: Mapping[str, Any]


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric value: {value}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _as_finite_float(value: Any, label: str) -> float:
    number = float(value.item() if hasattr(value, "item") else value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite: {number}")
    return number


def configure_offline_worker_environment() -> dict[str, Any]:
    """Disable library network fallbacks and inherited proxy routes."""

    forced = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "WANDB_DISABLED": "true",
    }
    for key, value in forced.items():
        os.environ[key] = value
    proxy_keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    removed = [key for key in proxy_keys if os.environ.pop(key, None) is not None]
    return {
        "forced_offline_variables": forced,
        "removed_proxy_variables": sorted(removed),
        "network_fallbacks_in_adapter": False,
        "staging_policy": "network CPU partition or jump host; GPU worker reads local/shared files only",
    }


def require_worker(hostname: str | None = None) -> str:
    observed = hostname or socket.gethostname()
    if not __import__("pipeline.runtime_host", fromlist=["is_compute_host"]).is_compute_host(observed.lower()):
        raise RuntimeError(
            f"formal MOSES generation/evaluation is worker-only; refusing {observed!r}. "
            "Do not run the workload on the jump host."
        )
    return observed


def validate_generation_contract(args: argparse.Namespace) -> None:
    expected = {
        "max_context": 512,
        "max_new_tokens": 512,
        "condition_token_budget": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "molecule_condition_mode": "direct_qwen",
    }
    for name, wanted in expected.items():
        observed = getattr(args, name)
        if observed != wanted:
            raise ValueError(
                f"formal MOSES protocol fixes --{name.replace('_', '-')}={wanted!r}; "
                f"got {observed!r}"
            )
    if bool(args.greedy):
        raise ValueError("formal MOSES protocol requires stochastic sampling")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.metric_batch_size) <= 0:
        raise ValueError("--metric-batch-size must be positive")
    if int(args.n_jobs) <= 0:
        raise ValueError("--n-jobs must be positive")
    if int(args.cpu_threads) <= 0:
        raise ValueError("--cpu-threads must be positive")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def _atomic_write_jsonl(path: Path, rows: Iterator[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    records = 0
    ids: set[str] = set()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                record_id = str(row.get("id") or "")
                if not record_id:
                    raise ValueError("artifact row is missing id")
                if record_id in ids:
                    raise ValueError(f"duplicate artifact id: {record_id}")
                ids.add(record_id)
                handle.write(
                    json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                records += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(path.resolve()),
        "records": records,
        "unique_ids": len(ids),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_dependency_lock(
    path: Path | None,
    *,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    """Inspect a pre-staged dependency manifest without using it as a gate.

    The manifest is provenance only.  Runtime compatibility is established by
    importing the official evaluator and exercising its API, not by requiring
    every transitive package version (or this file's bytes) to match a lock.
    """

    if path is None:
        return {
            "present": False,
            "used_as_acceptance_gate": False,
            "reason": "no dependency manifest supplied; runtime API imports are authoritative",
            "python": sys.version,
            "platform": platform.platform(),
        }
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"dependency lock is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("dependency provenance must be a JSON object")
    distributions = payload.get("distributions")
    if not isinstance(distributions, Mapping):
        distributions = {}
    expected = {str(key): str(value) for key, value in distributions.items()}
    lowered = {name.lower().replace("_", "-"): name for name in expected}
    missing_groups: list[list[str]] = []
    for alternatives in REQUIRED_DEPENDENCY_GROUPS:
        canonical = {name.lower().replace("_", "-") for name in alternatives}
        if not canonical.intersection(lowered):
            missing_groups.append(list(alternatives))
    installed: dict[str, str] = {}
    missing_installed: list[str] = []
    version_matches: dict[str, bool] = {}
    for name, pinned in expected.items():
        try:
            observed = str(version_resolver(name))
        except importlib.metadata.PackageNotFoundError:
            missing_installed.append(name)
            continue
        installed[name] = observed
        version_matches[name] = observed == pinned
    lock_sha256 = sha256_file(resolved)
    return {
        "path": str(resolved),
        "sha256": lock_sha256,
        "sha256_used_as_gate": False,
        "schema_version": payload.get("schema_version"),
        "schema_version_used_as_gate": False,
        "expected": expected,
        "installed": installed,
        "missing_required_groups": missing_groups,
        "missing_installed_distributions": missing_installed,
        "version_matches": version_matches,
        "all_exact_matches": (
            not missing_groups
            and not missing_installed
            and all(version_matches.values())
        ),
        "used_as_acceptance_gate": False,
        "python": sys.version,
        "platform": platform.platform(),
    }


def _patch_pandas_for_moses() -> bool:
    """Restore the one pandas API used by pinned MOSES on pandas 2.x."""

    import pandas as pd

    if hasattr(pd.DataFrame, "append"):
        return False

    def _append(
        frame: Any,
        other: Any,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
    ) -> Any:
        if not isinstance(other, pd.DataFrame):
            other = pd.DataFrame([other])
        return pd.concat(
            [frame, other],
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort,
        )

    pd.DataFrame.append = _append  # type: ignore[attr-defined]
    return True


def load_official_api(dependency_provenance: Mapping[str, Any]) -> OfficialAPI:
    """Import the official MOSES implementation and verify its public API."""

    installed_distribution_version = importlib.metadata.version(
        OFFICIAL_DISTRIBUTION
    )
    if installed_distribution_version != OFFICIAL_DISTRIBUTION_VERSION:
        raise RuntimeError(
            f"formal MOSES requires {OFFICIAL_DISTRIBUTION}=="
            f"{OFFICIAL_DISTRIBUTION_VERSION}; installed="
            f"{installed_distribution_version!r}"
        )

    pandas_shim = _patch_pandas_for_moses()
    modules: dict[str, Any] = {}
    module_provenance: dict[str, Any] = {}
    for name, expected_sha256 in OFFICIAL_SOURCE_SHA256.items():
        module = importlib.import_module(name)
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"cannot locate official MOSES source for {name}")
        source_path = Path(source).resolve()
        observed = sha256_file(source_path)
        modules[name] = module
        module_provenance[name] = {
            "path": str(source_path),
            "sha256": observed,
            "expected_sha256": expected_sha256,
            "sha256_used_as_gate": False,
        }

    metrics_dir = Path(modules["moses.metrics.metrics"].__file__).resolve().parent
    sa_dir = Path(modules["moses.metrics.SA_Score.sascorer"].__file__).resolve().parent
    resource_paths = {
        "moses.metrics/mcf.csv": metrics_dir / "mcf.csv",
        "moses.metrics/wehi_pains.csv": metrics_dir / "wehi_pains.csv",
        "moses.metrics.SA_Score/fpscores.pkl.gz": sa_dir / "fpscores.pkl.gz",
    }
    resources: dict[str, Any] = {}
    for name, resource_path in resource_paths.items():
        if not resource_path.is_file():
            raise FileNotFoundError(f"official MOSES metric resource is missing: {resource_path}")
        observed = sha256_file(resource_path)
        expected = OFFICIAL_RESOURCE_SHA256[name]
        resources[name] = {
            "path": str(resource_path),
            "bytes": int(resource_path.stat().st_size),
            "sha256": observed,
            "declared_sha256": expected,
            "sha256_matches_declared": observed == expected,
            "sha256_used_as_gate": False,
        }

    metrics = modules["moses.metrics.metrics"]
    return OfficialAPI(
        get_all_metrics=metrics.get_all_metrics,
        fraction_valid=metrics.fraction_valid,
        remove_invalid=metrics.remove_invalid,
        fraction_unique=metrics.fraction_unique,
        novelty=metrics.novelty,
        internal_diversity=metrics.internal_diversity,
        get_mol=metrics.get_mol,
        metrics_module=metrics,
        fcd_class=metrics.FCDMetric,
        provenance={
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "distribution": OFFICIAL_DISTRIBUTION,
            "distribution_version": installed_distribution_version,
            "source_modules": module_provenance,
            "metric_resources": resources,
            "dependency_lock": _jsonable(dependency_provenance),
            "pandas_dataframe_append_compatibility_shim_applied": pandas_shim,
            "pandas_shim_scope": "DataFrame.append used only to concatenate pinned MCF and PAINS tables",
        },
    )


def _validate_artifact(path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"official artifact is missing: {resolved}")
    observed_bytes = int(resolved.stat().st_size)
    observed_sha256 = sha256_file(resolved)
    return {
        "path": str(resolved),
        "bytes": observed_bytes,
        "declared_bytes": spec.bytes,
        "bytes_match_declared": spec.bytes is None or observed_bytes == spec.bytes,
        "bytes_used_as_gate": False,
        "sha256": observed_sha256,
        "declared_sha256": spec.sha256,
        "sha256_matches_declared": observed_sha256 == spec.sha256,
        "sha256_used_as_gate": False,
    }


def load_reference_csv(path: Path, spec: ArtifactSpec) -> tuple[list[str], dict[str, Any]]:
    provenance = _validate_artifact(path, spec)
    smiles: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "SMILES" not in (reader.fieldnames or []):
            raise ValueError(
                f"official MOSES CSV {path} is missing the case-sensitive SMILES column"
            )
        for row_index, row in enumerate(reader):
            value = row.get("SMILES")
            if value is None or value == "":
                raise ValueError(f"blank SMILES at {path}:{row_index + 2}")
            smiles.append(value)
    if spec.records is not None and len(smiles) != spec.records:
        raise ValueError(
            f"official row-count mismatch for {spec.name}: expected {spec.records}, "
            f"got {len(smiles)}"
        )
    provenance["records"] = len(smiles)
    provenance["expected_records"] = spec.records
    provenance["raw_values_preserved"] = True
    return smiles, provenance


def load_official_references(
    train_path: Path,
    test_path: Path,
    test_scaffolds_path: Path,
) -> ReferenceBundle:
    train, train_provenance = load_reference_csv(
        train_path, OFFICIAL_REFERENCE_SPECS["train"]
    )
    test, test_provenance = load_reference_csv(
        test_path, OFFICIAL_REFERENCE_SPECS["test"]
    )
    test_scaffolds, scaffold_provenance = load_reference_csv(
        test_scaffolds_path, OFFICIAL_REFERENCE_SPECS["test_scaffolds"]
    )
    return ReferenceBundle(
        train=train,
        test=test,
        test_scaffolds=test_scaffolds,
        provenance={
            "source": f"https://github.com/{OFFICIAL_REPOSITORY}/tree/{OFFICIAL_COMMIT}/data",
            "train": train_provenance,
            "test": test_provenance,
            "test_scaffolds": scaffold_provenance,
        },
    )


def load_official_statistics(
    test_stats_path: Path,
    test_scaffolds_stats_path: Path,
) -> StatisticsBundle:
    import numpy as np

    loaded: dict[str, Mapping[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for split, path in (
        ("test", test_stats_path),
        ("test_scaffolds", test_scaffolds_stats_path),
    ):
        spec = OFFICIAL_STATS_SPECS[split]
        provenance[split] = _validate_artifact(path, spec)
        with np.load(path, allow_pickle=True) as archive:
            if "stats" not in archive:
                raise ValueError(f"official statistics archive lacks 'stats': {path}")
            stats = archive["stats"].item()
        if not isinstance(stats, Mapping):
            raise TypeError(f"official statistics payload is not a mapping: {path}")
        missing = REQUIRED_STATS_KEYS.difference(stats)
        if missing:
            raise ValueError(f"official statistics {path} missing keys: {sorted(missing)}")
        loaded[split] = stats
        provenance[split]["statistics_keys"] = sorted(str(key) for key in stats)
    provenance["source"] = (
        f"https://github.com/{OFFICIAL_REPOSITORY}/tree/{OFFICIAL_COMMIT}/data"
    )
    return StatisticsBundle(
        test=loaded["test"],
        test_scaffolds=loaded["test_scaffolds"],
        provenance=provenance,
    )


def validate_chemnet(
    path: Path, declared_sha256: str | None = None
) -> dict[str, Any]:
    """Require a local ChemNet file; record any legacy hash as provenance."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ChemNet model is missing: {resolved}")
    observed_sha256 = sha256_file(resolved)
    declared = str(declared_sha256).strip().lower() if declared_sha256 else None
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": observed_sha256,
        "declared_sha256": declared,
        "sha256_matches_declared": (
            None if declared is None else observed_sha256 == declared
        ),
        "sha256_used_as_gate": False,
        "role": "local_fcd_torch_ChemNet_model",
        "load_validation": "FCDMetric loads this path during official scoring",
    }


def validate_attempts(
    attempts: Sequence[GenerationAttempt],
    *,
    seed: int,
    expected_count: int,
) -> None:
    if len(attempts) != expected_count:
        raise RuntimeError(
            f"seed {seed} returned {len(attempts)} attempts; expected {expected_count}"
        )
    ids: set[str] = set()
    for index, attempt in enumerate(attempts):
        expected_id = f"moses:seed{seed}:attempt{index:05d}"
        if attempt.seed != seed or attempt.attempt_index != index:
            raise RuntimeError(f"seed/index mismatch at attempt {index}: {attempt}")
        if attempt.attempt_id != expected_id:
            raise RuntimeError(
                f"attempt id mismatch at seed {seed} index {index}: "
                f"{attempt.attempt_id!r} != {expected_id!r}"
            )
        if attempt.attempt_id in ids:
            raise RuntimeError(f"duplicate attempt id: {attempt.attempt_id}")
        ids.add(attempt.attempt_id)
        if attempt.valid and attempt.decode_error is not None:
            raise RuntimeError(f"valid attempt has decode_error: {attempt.attempt_id}")
        if not attempt.valid and not attempt.decode_error:
            raise RuntimeError(f"invalid attempt lacks decode_error: {attempt.attempt_id}")


def write_attempt_artifacts(
    output_root: Path,
    attempts: Sequence[GenerationAttempt],
    *,
    seed: int,
) -> dict[str, Any]:
    seed_root = output_root / f"seed_{seed}"
    raw_path = seed_root / "raw_attempts.jsonl"
    decoded_path = seed_root / "decoded_attempts.jsonl"

    def raw_rows() -> Iterator[Mapping[str, Any]]:
        for item in attempts:
            yield {
                "id": item.attempt_id,
                "task": TASK_ID,
                "seed": item.seed,
                "attempt_index": item.attempt_index,
                "generated_bpe_ids": list(item.generated_bpe_ids),
                "generated_token_count": len(item.generated_bpe_ids),
                "ended_with_eos": item.ended_with_eos,
                "invalid_model_id": item.invalid_model_id,
                "stop_reason": item.stop_reason,
                "prompt_token_count": item.prompt_token_count,
                "effective_max_new_tokens": item.effective_max_new_tokens,
                "max_context": FORMAL_GENERATION["max_context"],
            }

    def decoded_rows() -> Iterator[Mapping[str, Any]]:
        for item in attempts:
            yield {
                "id": item.attempt_id,
                "raw_record_id": item.attempt_id,
                "raw_artifact": raw_path.name,
                "task": TASK_ID,
                "seed": item.seed,
                "attempt_index": item.attempt_index,
                "smiles": item.smiles,
                "valid": item.valid,
                "decode_error": item.decode_error,
                "generated_token_count": len(item.generated_bpe_ids),
                **({"aromatic_n_h_completion": item.aromatic_n_h_completion}
                   if item.aromatic_n_h_completion is not None else {}),
            }

    raw = _atomic_write_jsonl(raw_path, raw_rows())
    decoded = _atomic_write_jsonl(decoded_path, decoded_rows())
    if raw["records"] != decoded["records"] or raw["records"] != len(attempts):
        raise AssertionError("raw/decoded one-for-one artifact count mismatch")
    return {
        "raw": raw,
        "decoded": decoded,
        "one_for_one_records": True,
        "invalid_placeholder_policy": "decoded null becomes empty string only at evaluator boundary",
    }


@contextmanager
def _local_chemnet_fcd(api: OfficialAPI, chemnet_path: Path) -> Iterator[None]:
    if api.fcd_class is None:
        yield
        return
    original = api.fcd_class

    class FCDWithPinnedLocalModel(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["model_path"] = str(chemnet_path.expanduser().resolve())
            super().__init__(*args, **kwargs)

    previous_attribute = getattr(api.metrics_module, "FCDMetric", None)
    globals_dict = getattr(api.get_all_metrics, "__globals__", {})
    previous_global = globals_dict.get("FCDMetric")
    setattr(api.metrics_module, "FCDMetric", FCDWithPinnedLocalModel)
    globals_dict["FCDMetric"] = FCDWithPinnedLocalModel
    try:
        yield
    finally:
        setattr(api.metrics_module, "FCDMetric", previous_attribute)
        if previous_global is None:
            globals_dict.pop("FCDMetric", None)
        else:
            globals_dict["FCDMetric"] = previous_global


def biomatrix_first10k_view(
    attempts: Sequence[GenerationAttempt],
    *,
    api: OfficialAPI,
    train: Sequence[str],
    device: str,
    prefix_size: int = BIOMATRIX_PREFIX,
) -> dict[str, Any]:
    if len(attempts) < prefix_size:
        raise ValueError(
            f"BioMatrix view requires {prefix_size} raw attempts, got {len(attempts)}"
        )
    prefix = list(attempts[:prefix_size])
    raw_smiles = [item.metric_smiles for item in prefix]
    validity = _as_finite_float(api.fraction_valid(raw_smiles, n_jobs=1), "Validity")
    canonical_valid = list(api.remove_invalid(raw_smiles, canonize=True, n_jobs=1))
    if not canonical_valid:
        raise RuntimeError("BioMatrix prefix contains no valid molecule")
    unique = _as_finite_float(
        api.fraction_unique(canonical_valid, k=None, n_jobs=1), "Unique"
    )
    mols = [api.get_mol(smiles) for smiles in canonical_valid]
    if any(molecule is None for molecule in mols):
        raise AssertionError("remove_invalid returned a value that get_mol rejected")
    novelty = _as_finite_float(api.novelty(mols, train, n_jobs=1), "Novelty")
    intdiv1 = _as_finite_float(
        api.internal_diversity(mols, n_jobs=1, device=device, p=1), "IntDiv1"
    )
    intdiv2 = _as_finite_float(
        api.internal_diversity(mols, n_jobs=1, device=device, p=2), "IntDiv2"
    )
    return {
        "raw_prefix_records": prefix_size,
        "valid_records": len(canonical_valid),
        "invalid_records": prefix_size - len(canonical_valid),
        "validity_denominator": "first 10,000 raw attempts",
        "unique_novelty_diversity_denominator": "valid canonical molecules among the same raw prefix",
        "metrics": {
            "Validity": validity,
            "Unique": unique,
            "Novelty": novelty,
            "IntDiv1": intdiv1,
            "IntDiv2": intdiv2,
        },
    }


def score_seed(
    attempts: Sequence[GenerationAttempt],
    *,
    seed: int,
    api: OfficialAPI,
    references: ReferenceBundle,
    reference_statistics: StatisticsBundle,
    chemnet_path: Path,
    chemnet_provenance: Mapping[str, Any],
    device: str,
    batch_size: int,
    n_jobs: int,
    biomatrix_prefix_size: int = BIOMATRIX_PREFIX,
) -> dict[str, Any]:
    raw_smiles = [item.metric_smiles for item in attempts]
    if any(not isinstance(value, str) for value in raw_smiles):
        raise TypeError("metric input must contain strings only")
    with _local_chemnet_fcd(api, chemnet_path), float64_count_cosine(api.metrics_module):
        result = api.get_all_metrics(
            raw_smiles,
            k=[1000, 10000],
            n_jobs=int(n_jobs),
            device=device,
            batch_size=int(batch_size),
            pool=None,
            test=list(references.test),
            test_scaffolds=list(references.test_scaffolds),
            ptest=reference_statistics.test,
            ptest_scaffolds=reference_statistics.test_scaffolds,
            train=list(references.train),
        )
    observed_keys = set(result)
    expected_keys = set(OFFICIAL_METRIC_KEYS)
    if observed_keys != expected_keys:
        raise RuntimeError(
            f"official MOSES metric key mismatch: missing={sorted(expected_keys-observed_keys)}, "
            f"extra={sorted(observed_keys-expected_keys)}"
        )
    official_metrics = {
        key: _as_finite_float(result[key], f"seed {seed} {key}")
        for key in OFFICIAL_METRIC_KEYS
    }
    reported_metrics = {
        ("MW" if key == "weight" else key): value
        for key, value in official_metrics.items()
    }
    failures = Counter(
        item.decode_error or "valid" for item in attempts if not item.valid
    )
    return {
        "seed": seed,
        "raw_attempts": len(attempts),
        "valid_decodes": sum(item.valid for item in attempts),
        "invalid_placeholders": sum(not item.valid for item in attempts),
        "decode_failure_counts": dict(sorted(failures.items())),
        "official_metrics": official_metrics,
        "reported_metrics": reported_metrics,
        "official_weight_key_display_name": "MW",
        "numeric_compatibility": COUNT_COSINE_COMPATIBILITY,
        "chemnet": _jsonable(chemnet_provenance),
        "biomatrix_first10k": biomatrix_first10k_view(
            attempts,
            api=api,
            train=references.train,
            device=device,
            prefix_size=biomatrix_prefix_size,
        ),
    }


def aggregate_seed_results(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds = tuple(int(item["seed"]) for item in seed_results)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("aggregation requires at least one distinct seed")

    def aggregate(metric_maps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        keys = tuple(metric_maps[0])
        if any(tuple(item) != keys for item in metric_maps[1:]):
            raise ValueError("metric key/order mismatch across seeds")
        output: dict[str, Any] = {}
        for key in keys:
            values = [_as_finite_float(item[key], key) for item in metric_maps]
            output[key] = {
                "per_seed": {str(seed): value for seed, value in zip(seeds, values, strict=True)},
                "mean": statistics.fmean(values),
                "sd": statistics.pstdev(values) if len(values) > 1 else None,
                "sd_ddof": 0 if len(values) > 1 else None,
                "n": len(values),
            }
        return output

    return {
        "seeds": list(seeds),
        "official_metrics": aggregate(
            [item["official_metrics"] for item in seed_results]
        ),
        "reported_metrics": aggregate(
            [item["reported_metrics"] for item in seed_results]
        ),
        "biomatrix_first10k": aggregate(
            [item["biomatrix_first10k"]["metrics"] for item in seed_results]
        ),
        "standard_deviation_definition": (
            "not estimated from a single seed" if len(seeds) == 1
            else "population SD across the requested seeds (ddof=0)"),
    }


def _aggregate_metric_maps(
    metric_maps: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    """Aggregate an arbitrary synthetic seed set with population SD."""

    if not metric_maps:
        raise ValueError("cannot aggregate an empty metric sequence")
    keys = tuple(metric_maps[0])
    if any(tuple(item) != keys for item in metric_maps[1:]):
        raise ValueError("metric key/order mismatch across seeds")
    return {
        key: {
            "per_seed": {
                str(seed): _as_finite_float(values[key], key)
                for seed, values in zip(seeds, metric_maps, strict=True)
            },
            "mean": statistics.fmean(
                [_as_finite_float(values[key], key) for values in metric_maps]
            ),
            "sd": statistics.pstdev(
                [_as_finite_float(values[key], key) for values in metric_maps]
            ),
            "sd_ddof": 0,
            "n": len(metric_maps),
        }
        for key in keys
    }


class AxisMosesSampler:
    """Live Axis sampler; all heavyweight imports remain worker-side and lazy."""

    def __init__(self, args: argparse.Namespace) -> None:
        require_worker()
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("formal MOSES sampling requires a scheduler-assigned GPU")
        torch.set_num_threads(args.cpu_threads)
        from pipeline.finetune.axis_b_data import AxisBSFTCollator
        from rdkit import Chem, RDLogger
        import eval_axis_b_samples as axis_decode
        import sample_axis_b_sft as axis_sample

        RDLogger.DisableLog("rdApp.*")
        self.torch = torch
        self.Chem = Chem
        self.axis_decode = axis_decode
        self.axis_sample = axis_sample
        self.args = args
        self.device = torch.device(args.device)
        self.model, tokenizer = axis_sample.load_model(args, self.device)
        if args.molecule_condition_mode != "direct_qwen":
            raise RuntimeError("loaded checkpoint did not retain direct_qwen mode")
        # ``generate_batch`` supplies the stochastic controls explicitly.  Pin
        # beam count here as well so a checkpoint-local generation_config
        # cannot silently override the formal beam=1 contract.
        self.model.qwen.generation_config.num_beams = 1
        self.model.qwen.generation_config.num_beam_groups = 1
        self.generation_config_provenance = {
            "num_beams": int(self.model.qwen.generation_config.num_beams),
            "num_beam_groups": int(
                self.model.qwen.generation_config.num_beam_groups
            ),
            "source": "explicitly pinned by official MOSES adapter",
        }
        self.collator = AxisBSFTCollator(
            tokenizer,
            qwen_max_context=int(args.max_context),
            return_tensors="pt",
            condition_token_budget=int(args.condition_token_budget),
            molecule_condition_mode="direct_qwen",
        )
        self.vocab = axis_decode.FDDFSVocab.load(args.vocab_json)
        self.decoder = axis_decode.BPEDecoder(args.merges_csv, self.vocab)

    @staticmethod
    def _record(seed: int, attempt_index: int) -> dict[str, Any]:
        # The collator requires a target placeholder to build type/BOS.  It is
        # sliced away before generation and never scored.
        return {
            "id": f"moses:seed{seed}:attempt{attempt_index:05d}",
            "task": TASK_ID,
            "target_type": "small_molecule",
            "condition": {
                "type": "none",
                "text": None,
                "molecule_bpe_ids": [],
                "modality": "none",
                "role": None,
                "instruction": None,
            },
            "target_bpe_ids": [0],
            "meta": {"benchmark": "MOSES", "protocol_id": PROTOCOL_ID},
        }

    def sample_seed(self, seed: int, number_samples: int) -> list[GenerationAttempt]:
        torch = self.torch
        random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        attempts: list[GenerationAttempt] = []
        for start in range(0, int(number_samples), int(self.args.batch_size)):
            size = min(int(self.args.batch_size), int(number_samples) - start)
            records = [self._record(seed, start + row) for row in range(size)]
            build, _collated = self.axis_sample.model_batch_from_records(
                records, self.collator, self.device
            )
            # Unconditional prompts contain only target-type and BOS tokens.
            # Deriving the active length from the actual model input keeps the
            # recorded dynamic budget honest if the shared collator changes.
            qwen_inputs = self.model.build_qwen_inputs(**build)
            prompt_lengths = [
                int(value)
                for value in qwen_inputs["attention_mask"].to(dtype=torch.long).sum(dim=1).tolist()
            ]
            generated, ended, invalid_ids = self.axis_sample.generate_batch(
                self.model, build, self.args
            )
            for local, (raw_ids, ended_with_eos, invalid_model_id) in enumerate(
                zip(generated, ended, invalid_ids, strict=True)
            ):
                attempt_index = start + local
                prompt_length = prompt_lengths[local]
                effective_budget = min(
                    int(self.args.max_new_tokens),
                    max(1, int(self.args.max_context) - prompt_length),
                )
                smiles: str | None = None
                decode_error: str | None = None
                completion_audit = None
                if invalid_model_id:
                    stop_reason = "invalid_model_id"
                    decode_error = "invalid_model_id"
                elif not ended_with_eos:
                    stop_reason = "dynamic_context_or_generation_limit"
                    decode_error = "missing_eos"
                elif not raw_ids:
                    stop_reason = "eos"
                    decode_error = "empty_generation"
                else:
                    stop_reason = "eos"
                    try:
                        graph, _tokens = self.axis_decode.decode_graph(
                            list(raw_ids), self.decoder, self.vocab
                        )
                        molecule, info = self.axis_decode.graph_to_rdkit_mol(
                            graph, strict=True,
                            aromatic_n_h_completion=getattr(self.args, "aromatic_n_h_completion", False),
                        )
                        completion_audit = info.get("aromatic_n_h_completion")
                        if not bool(info.get("sanitize_ok")) or molecule.GetNumAtoms() <= 0:
                            raise ValueError("strict RDKit sanitization failed")
                        candidate = self.Chem.MolToSmiles(
                            molecule, canonical=True, isomericSmiles=True
                        )
                        if not candidate:
                            raise ValueError("empty canonical SMILES")
                        smiles = candidate
                    except ValueError as exc:
                        # Malformed FDDFS/strict graph draws are expected;
                        # runtime, dependency and memory failures must abort.
                        decode_error = f"{type(exc).__name__}:{str(exc)[:240]}"
                attempts.append(
                    GenerationAttempt(
                        attempt_id=f"moses:seed{seed}:attempt{attempt_index:05d}",
                        seed=int(seed),
                        attempt_index=attempt_index,
                        generated_bpe_ids=tuple(int(value) for value in raw_ids),
                        ended_with_eos=bool(ended_with_eos),
                        invalid_model_id=bool(invalid_model_id),
                        stop_reason=stop_reason,
                        prompt_token_count=prompt_length,
                        effective_max_new_tokens=effective_budget,
                        smiles=smiles,
                        decode_error=decode_error,
                        aromatic_n_h_completion=completion_audit,
                    )
                )
            print(
                f"moses seed={seed} sampled={len(attempts)}/{number_samples}",
                flush=True,
            )
        validate_attempts(attempts, seed=seed, expected_count=number_samples)
        return attempts


class StoredMosesSampler:
    """Re-decode exactly the saved 30k attempts; never generate replacement slots."""

    def __init__(self, args):
        from rdkit import Chem
        from cli.evaluate import eval_axis_b_samples as decode
        self.args, self.Chem, self.decode = args, Chem, decode
        source_protocol = json.loads((args.raw_samples_root / "protocol.json").read_text())
        if source_protocol["task_id"] != TASK_ID or source_protocol["generation"] != FORMAL_GENERATION:
            raise ValueError("saved MOSES draws have a different task or sampling protocol")
        self.source_axis = source_protocol["axis"]
        from moltoken.checkpoints import checkpoint_path_identity as canonical_path
        if canonical_path(self.source_axis["checkpoint"]) != canonical_path(args.checkpoint):
            raise ValueError("saved draws belong to a different checkpoint")
        if not set(args.seeds).issubset(source_protocol["seeds"]):
            raise ValueError("requested seed was not part of the source evaluation")
        self.vocab = decode.FDDFSVocab.load(args.vocab_json)
        self.decoder = decode.BPEDecoder(args.merges_csv, self.vocab)
        self.generation_config_provenance = {
            "source": "saved raw attempts; no model generation",
            "raw_samples_root": str(args.raw_samples_root.resolve()),
            "original_axis": self.source_axis,
        }

    def sample_seed(self, seed, number_samples):
        source = self.args.raw_samples_root / f"seed_{seed}" / "raw_attempts.jsonl"
        attempts = []
        with source.open() as handle:
            for line in handle:
                row = json.loads(line)
                smiles, error, audit = None, None, None
                if row["invalid_model_id"]:
                    error = "invalid_model_id"
                elif not row["ended_with_eos"]:
                    error = "missing_eos"
                elif not row["generated_bpe_ids"]:
                    error = "empty_generation"
                else:
                    try:
                        graph, _ = self.decode.decode_graph(row["generated_bpe_ids"], self.decoder, self.vocab)
                        mol, info = self.decode.graph_to_rdkit_mol(
                            graph, strict=True, aromatic_n_h_completion=self.args.aromatic_n_h_completion)
                        audit = info.get("aromatic_n_h_completion")
                        if not info["sanitize_ok"] or mol.GetNumAtoms() == 0:
                            raise ValueError("strict RDKit sanitization failed")
                        smiles = self.Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                        if not smiles:
                            raise ValueError("empty canonical SMILES")
                    except ValueError as exc:
                        smiles, error = None, f"{type(exc).__name__}:{str(exc)[:240]}"
                attempts.append(GenerationAttempt(
                    attempt_id=row["id"], seed=row["seed"], attempt_index=row["attempt_index"],
                    generated_bpe_ids=tuple(row["generated_bpe_ids"]),
                    ended_with_eos=row["ended_with_eos"], invalid_model_id=row["invalid_model_id"],
                    stop_reason=row["stop_reason"], prompt_token_count=row["prompt_token_count"],
                    effective_max_new_tokens=row["effective_max_new_tokens"],
                    smiles=smiles, decode_error=error, aromatic_n_h_completion=audit,
                ))
                if len(attempts) % 5000 == 0:
                    print(f"MOSES saved seed={seed} redecoded={len(attempts)}/{number_samples}", flush=True)
        validate_attempts(attempts, seed=seed, expected_count=number_samples)
        return attempts


def run_protocol(
    *,
    sampler: Any,
    output_root: Path,
    api: OfficialAPI,
    references: ReferenceBundle,
    reference_statistics: StatisticsBundle,
    chemnet_path: Path,
    chemnet_provenance: Mapping[str, Any],
    axis_provenance: Mapping[str, Any],
    offline_provenance: Mapping[str, Any],
    device: str,
    batch_size: int,
    n_jobs: int,
    seeds: Sequence[int] = FORMAL_SEEDS,
    samples_per_seed: int = FORMAL_SAMPLES_PER_SEED,
    biomatrix_prefix_size: int = BIOMATRIX_PREFIX,
    formal: bool = True,
) -> dict[str, Any]:
    seeds_tuple = tuple(int(seed) for seed in seeds)
    if formal and (
        samples_per_seed != FORMAL_SAMPLES_PER_SEED
        or biomatrix_prefix_size != BIOMATRIX_PREFIX
    ):
        raise ValueError("formal MOSES sample/BioMatrix-prefix contract was changed")
    if not seeds_tuple or len(seeds_tuple) != len(set(seeds_tuple)):
        raise ValueError("request at least one distinct seed")
    if samples_per_seed < biomatrix_prefix_size or biomatrix_prefix_size <= 0:
        raise ValueError("sample count must cover the positive BioMatrix prefix")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite MOSES output root: {output_root}")
    output_root.mkdir(parents=True)

    protocol_payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "formal": formal,
        "seeds": list(seeds_tuple),
        "repetition_scope": "single-seed first pass" if len(seeds_tuple) == 1 else "multiple sampling seeds",
        "raw_attempts_per_seed": int(samples_per_seed),
        "generation": FORMAL_GENERATION,
        "retry_filter_repair": (
            "aromatic_n_h_completion_only; no filtering or resampling"
            if axis_provenance.get("aromatic_n_h_completion", False) else "none"
        ),
        "invalid_evaluator_placeholder": "empty SMILES, retained one-for-one",
        "official_moses": _jsonable(api.provenance),
        "numeric_compatibility": COUNT_COSINE_COMPATIBILITY,
        "references": _jsonable(references.provenance),
        "reference_statistics": _jsonable(reference_statistics.provenance),
        "chemnet": _jsonable(chemnet_provenance),
        "axis": _jsonable(axis_provenance),
        "offline_execution": _jsonable(offline_provenance),
        "metric_directions": {
            key: ("higher" if key in HIGHER_IS_BETTER else "lower")
            for key in OFFICIAL_METRIC_KEYS
        },
        "molecular_weight_official_key": "weight",
        "molecular_weight_report_label": "MW",
    }
    _atomic_write_json(output_root / "protocol.json", protocol_payload)

    seed_results: list[dict[str, Any]] = []
    for seed in seeds_tuple:
        started = time.time()
        attempts = list(sampler.sample_seed(seed, samples_per_seed))
        validate_attempts(attempts, seed=seed, expected_count=samples_per_seed)
        artifacts = write_attempt_artifacts(output_root, attempts, seed=seed)
        result = score_seed(
            attempts,
            seed=seed,
            api=api,
            references=references,
            reference_statistics=reference_statistics,
            chemnet_path=chemnet_path,
            chemnet_provenance=chemnet_provenance,
            device=device,
            batch_size=batch_size,
            n_jobs=n_jobs,
            biomatrix_prefix_size=biomatrix_prefix_size,
        )
        result["artifacts"] = artifacts
        result["elapsed_seconds"] = time.time() - started
        _atomic_write_json(output_root / f"seed_{seed}" / "metrics.json", result)
        seed_results.append(result)
        del attempts

    if formal:
        aggregate = aggregate_seed_results(seed_results)
    else:
        # Synthetic tests can use arbitrary seeds while exercising identical
        # one-for-one sampling and artifact logic.
        if seeds_tuple == FORMAL_SEEDS:
            aggregate = aggregate_seed_results(seed_results)
        else:
            biomatrix_maps = [
                item["biomatrix_first10k"]["metrics"] for item in seed_results
            ]
            aggregate = {
                "seeds": list(seeds_tuple),
                "official_metrics": _aggregate_metric_maps(
                    [item["official_metrics"] for item in seed_results], seeds_tuple
                ),
                "reported_metrics": _aggregate_metric_maps(
                    [item["reported_metrics"] for item in seed_results], seeds_tuple
                ),
                "biomatrix_first10k": _aggregate_metric_maps(
                    biomatrix_maps, seeds_tuple
                ),
                "standard_deviation_definition": (
                    "population SD across requested synthetic seeds (ddof=0)"
                ),
            }

    summary = {
        **protocol_payload,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "complete": True,
    }
    _atomic_write_json(output_root / "summary.json", summary)
    _atomic_write_json(
        output_root / "_SUCCESS.json",
        {
            "complete": True,
            "protocol_id": PROTOCOL_ID,
            "task_id": TASK_ID,
            "seed_count": len(seed_results),
            "raw_attempts": len(seed_results) * samples_per_seed,
            "summary_sha256": sha256_file(output_root / "summary.json"),
            "summary_sha256_used_as_gate": False,
            "one_for_one_invalid_accounting": True,
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS),
                        help="Use 42 for a first pass; sample count remains 30000 per seed")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--test-scaffolds", type=Path, default=DEFAULT_TEST_SCAFFOLDS)
    parser.add_argument("--test-stats", type=Path, default=DEFAULT_TEST_STATS)
    parser.add_argument(
        "--test-scaffolds-stats", type=Path, default=DEFAULT_TEST_SCAFFOLDS_STATS
    )
    parser.add_argument("--chemnet", type=Path, required=True)
    parser.add_argument(
        "--chemnet-sha256",
        help="legacy provenance only; never used to accept or reject the model",
    )
    parser.add_argument(
        "--dependency-lock",
        type=Path,
        help="optional dependency provenance; contents never decide acceptance",
    )
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--merges-csv", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--molecule-condition-mode",
        choices=("direct_qwen",),
        default="direct_qwen",
        help="current formal MOSES pipeline uses the frozen FDDFS/BPE vocabulary directly",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Explicit device for generation and official MOSES metrics; no fallback")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--metric-batch-size", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--condition-token-budget", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--aromatic-n-h-completion", action="store_true")
    parser.add_argument("--raw-samples-root", type=Path,
                        help="Reuse existing seed_N/raw_attempts.jsonl instead of generating new samples")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    offline = configure_offline_worker_environment()
    hostname = require_worker()
    validate_generation_contract(args)
    for label in ("checkpoint", "text_encoder", "merges_csv", "vocab_json"):
        value = Path(getattr(args, label))
        if not value.exists():
            raise FileNotFoundError(f"missing --{label.replace('_', '-')}: {value}")

    dependency_provenance = validate_dependency_lock(args.dependency_lock)
    api = load_official_api(dependency_provenance)
    references = load_official_references(args.train, args.test, args.test_scaffolds)
    reference_statistics = load_official_statistics(
        args.test_stats, args.test_scaffolds_stats
    )
    chemnet_provenance = validate_chemnet(args.chemnet, args.chemnet_sha256)
    sampler = StoredMosesSampler(args) if args.raw_samples_root is not None else AxisMosesSampler(args)
    axis_provenance = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "text_encoder": str(args.text_encoder.expanduser().resolve()),
        "merges_csv": str(args.merges_csv.expanduser().resolve()),
        "merges_sha256": sha256_file(args.merges_csv),
        "vocab_json": str(args.vocab_json.expanduser().resolve()),
        "vocab_sha256": sha256_file(args.vocab_json),
        "molecule_condition_mode": args.molecule_condition_mode,
        "dtype": args.dtype,
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "worker": hostname,
        "generation_batch_size": args.batch_size,
        "metric_batch_size": args.metric_batch_size,
        "metric_n_jobs": args.n_jobs,
        "hf_generation_config": sampler.generation_config_provenance,
        "aromatic_n_h_completion": args.aromatic_n_h_completion,
        "raw_samples_root": str(args.raw_samples_root.resolve()) if args.raw_samples_root else None,
    }
    if args.raw_samples_root is not None:
        axis_provenance.update(
            device=sampler.source_axis["device"],
            dtype=sampler.source_axis["dtype"],
            generation_batch_size=sampler.source_axis["generation_batch_size"],
            original_sampling_worker=sampler.source_axis["worker"],
            scoring_device=args.device,
        )
    summary = run_protocol(
        sampler=sampler,
        output_root=args.output_root,
        api=api,
        references=references,
        reference_statistics=reference_statistics,
        chemnet_path=args.chemnet,
        chemnet_provenance=chemnet_provenance,
        axis_provenance=axis_provenance,
        offline_provenance=offline,
        device="cuda:0" if args.device == "cuda" else "cpu",
        batch_size=args.metric_batch_size,
        n_jobs=args.n_jobs,
        seeds=args.seeds,
    )
    headline = {
        key: {
            "mean": value["mean"],
            "sd": value["sd"],
        }
        for key, value in summary["aggregate"]["reported_metrics"].items()
    }
    print(json.dumps(headline, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
