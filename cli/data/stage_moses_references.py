#!/usr/bin/env python3
"""Partition the released MOSES CSV into its train/test/scaffold splits.

The input is an already downloaded ``dataset_v1.csv``. Source row bytes and
order are preserved. This command does not download data or rewrite SMILES.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from typing import Any, BinaryIO, Mapping

from pipeline.paths import LOCAL_DATA_ROOT
from pipeline.runtime_host import require_compute_host


DATA_ROOT = LOCAL_DATA_ROOT
DEFAULT_COMBINED = DATA_ROOT / "benchmark/raw/moses/dataset_v1.csv"
DEFAULT_OUTPUT = DATA_ROOT / "benchmark/raw/moses/official_core13_dd7ed6"

PROTOCOL_ID = "axis-core13-official-v1-20260821"
SCHEMA_VERSION = "axis-core13-moses-reference-staging-v1"
OFFICIAL_REPOSITORY = "molecularsets/moses"
OFFICIAL_COMMIT = "dd7ed6ab38e23afd3ef5371d67939a1760bd8599"
OFFICIAL_COMBINED_URL = (
    "https://media.githubusercontent.com/media/molecularsets/moses/"
    f"{OFFICIAL_COMMIT}/data/dataset_v1.csv"
)


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    sha256: str
    size_bytes: int
    records: int


COMBINED_SPEC = ArtifactSpec(
    filename="dataset_v1.csv",
    sha256="bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b",
    size_bytes=84_482_588,
    records=1_936_962,
)

# These values are deliberately duplicated from
# evaluate_axis_b_moses_official.py.  The companion unit test checks that the
# two files remain in lockstep.  They are byte partitions of dataset_v1.csv,
# not pandas re-serializations and not decompressed package CSVs.
OUTPUT_SPECS: Mapping[str, ArtifactSpec] = {
    "train": ArtifactSpec(
        filename="train.csv",
        sha256="49fe1aae29604ec0f5023ea34edc8789415f954138f2903dabd005a4b888a961",
        size_bytes=67_916_778,
        records=1_584_663,
    ),
    "test": ArtifactSpec(
        filename="test.csv",
        sha256="257165e8f1e1be2d5712db51c93d2b1544564b782dc303d7d19c887c21a858e6",
        size_bytes=7_370_473,
        records=176_074,
    ),
    "test_scaffolds": ArtifactSpec(
        filename="test_scaffolds.csv",
        sha256="8bc942dd807147e07b506151fe8675f58780ed8e494801b705c4bdadb3df3440",
        size_bytes=9_195_363,
        records=176_225,
    ),
}

EXPECTED_HEADER = b"SMILES,SPLIT\n"


def enforce_offline_cpu_runtime() -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PIP_NO_INDEX"] = "1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def require_worker(hostname: str | None = None) -> str:
    observed = hostname or socket.gethostname()
    require_compute_host("MOSES reference staging", observed)
    return observed


class _DigestWriter:
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, value: bytes) -> None:
        self.handle.write(value)
        self.digest.update(value)
        self.size += len(value)

    def finish(self) -> str:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self.digest.hexdigest()


def _split_from_raw_row(raw_line: bytes, *, line_number: int) -> str:
    if not raw_line.endswith(b"\n"):
        raise ValueError(f"source line {line_number} lacks its final LF byte")
    try:
        text = raw_line[:-1].decode("utf-8")
        row = next(csv.reader([text]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot parse source CSV line {line_number}: {exc}") from exc
    if len(row) != 2:
        raise ValueError(
            f"source line {line_number} must have exactly SMILES,SPLIT; got {len(row)} fields"
        )
    split = row[1]
    if split not in OUTPUT_SPECS:
        raise ValueError(f"unknown MOSES split {split!r} at source line {line_number}")
    if not row[0]:
        raise ValueError(f"blank SMILES at source line {line_number}")
    return split


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_references(
    source: Path,
    output_root: Path,
    *,
    source_spec: ArtifactSpec = COMBINED_SPEC,
    output_specs: Mapping[str, ArtifactSpec] = OUTPUT_SPECS,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically partition one immutable combined release."""

    worker = require_worker(hostname)
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"combined MOSES source is missing: {source}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")

    observed_source_sha256, observed_source_size = sha256_and_size(source)
    if observed_source_size != source_spec.size_bytes:
        raise ValueError(
            f"combined source size mismatch: expected {source_spec.size_bytes}, "
            f"got {observed_source_size}"
        )
    if observed_source_sha256 != source_spec.sha256:
        raise ValueError(
            f"combined source SHA-256 mismatch: expected {source_spec.sha256}, "
            f"got {observed_source_sha256}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.partial.", dir=str(output_root.parent)
        )
    )
    handles: dict[str, BinaryIO] = {}
    writers: dict[str, _DigestWriter] = {}
    counts = {split: 0 for split in output_specs}
    try:
        for split, spec in output_specs.items():
            handle = (temporary_root / spec.filename).open("wb")
            handles[split] = handle
            writers[split] = _DigestWriter(handle)

        with source.open("rb") as source_handle:
            header = source_handle.readline()
            if header != EXPECTED_HEADER:
                raise ValueError(
                    f"official header mismatch: expected {EXPECTED_HEADER!r}, got {header!r}"
                )
            for writer in writers.values():
                writer.write(header)
            for line_number, raw_line in enumerate(source_handle, start=2):
                split = _split_from_raw_row(raw_line, line_number=line_number)
                writers[split].write(raw_line)
                counts[split] += 1

        output_details: dict[str, Any] = {}
        for split, writer in writers.items():
            observed_sha = writer.finish()
            spec = output_specs[split]
            if counts[split] != spec.records:
                raise ValueError(
                    f"{split} row-count mismatch: expected {spec.records}, got {counts[split]}"
                )
            if writer.size != spec.size_bytes:
                raise ValueError(
                    f"{split} byte-count mismatch: expected {spec.size_bytes}, got {writer.size}"
                )
            if observed_sha != spec.sha256:
                raise ValueError(
                    f"{split} SHA-256 mismatch: expected {spec.sha256}, got {observed_sha}"
                )
            output_details[split] = {
                "path": str((output_root / spec.filename).resolve()),
                "filename": spec.filename,
                "records": counts[split],
                "bytes": writer.size,
                "sha256": observed_sha,
                "expected": asdict(spec),
                "byte_identical_to_adapter_reference": True,
            }

        for handle in handles.values():
            handle.close()
        handles.clear()

        if sum(counts.values()) != source_spec.records:
            raise ValueError(
                f"combined logical row-count mismatch: expected {source_spec.records}, "
                f"got {sum(counts.values())}"
            )
        # Each output repeats the source header.  This invariant catches any
        # silently omitted or duplicated raw bytes independently of SHA-256.
        expected_partition_bytes = source_spec.size_bytes + (
            len(output_specs) - 1
        ) * len(EXPECTED_HEADER)
        if sum(item["bytes"] for item in output_details.values()) != expected_partition_bytes:
            raise AssertionError("partition byte conservation invariant failed")

        provenance = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "worker": worker,
            "offline": True,
            "network_access": "none",
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "source_url": OFFICIAL_COMBINED_URL,
            "source": {
                "path": str(source),
                "filename": source_spec.filename,
                "records": source_spec.records,
                "bytes": observed_source_size,
                "sha256": observed_source_sha256,
                "expected": asdict(source_spec),
            },
            "partition_semantics": (
                "raw source-order rows partitioned by exact SPLIT field; original "
                "SMILES,SPLIT header and row bytes retained; no normalization, "
                "deduplication, reserialization, filtering, or repair"
            ),
            "outputs": output_details,
            "materializer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        _atomic_write_json(temporary_root / "provenance.json", provenance)
        provenance_sha = sha256_file(temporary_root / "provenance.json")
        _atomic_write_json(
            temporary_root / "_SUCCESS.json",
            {
                "complete": True,
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "commit": OFFICIAL_COMMIT,
                "source_sha256": observed_source_sha256,
                "provenance_sha256": provenance_sha,
                "all_adapter_specs_match": True,
            },
        )
        os.replace(temporary_root, output_root)
        return provenance
    except BaseException:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    enforce_offline_cpu_runtime()
    args = parse_args()
    provenance = materialize_references(args.combined, args.output_root)
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
