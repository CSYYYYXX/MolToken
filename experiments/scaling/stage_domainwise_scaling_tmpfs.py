#!/usr/bin/env python3
"""Stage compact payload/index/manifests to node-local tmpfs and relocate paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from .scaling_paths import DEFAULT_DATA_ROOT
except ImportError:
    from scaling_paths import DEFAULT_DATA_ROOT


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rsync_directory(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-aS", "--delete", f"{source}/", f"{destination}/"],
        check=True,
    )


def main() -> None:
    default_dataset = (
        DEFAULT_DATA_ROOT
        / "downloads_raw"
        / "huggingface"
        / "HoloMol-Pretrain-Compact"
    )
    default_manifest = (
        DEFAULT_DATA_ROOT
        / "processed"
        / "domain_split_manifests_v1"
        / "manifest.json"
    )
    default_destination = Path(f"/dev/shm/holomol-{os.getuid()}/scaling/domainwise_dataset_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--manifest-json", type=Path, default=default_manifest)
    parser.add_argument("--destination", type=Path, default=default_destination)
    args = parser.parse_args()

    source_root = args.dataset_root.expanduser().resolve()
    source_manifest = args.manifest_json.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    compact_name = "encoded_compact_int_prefix70_mf80000_11w_v3"
    index_name = "encoded_compact_int_prefix70_mf80000_11w_v3_index"
    staged_compact = destination / compact_name
    staged_index = destination / index_name
    staged_indices = destination / "domain_split_manifests_v1"

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "domain_split_raw_indices_v1":
        raise ValueError(f"unsupported manifest: {source_manifest}")
    rsync_directory(source_root / compact_name, staged_compact)
    rsync_directory(source_root / index_name, staged_index)
    rsync_directory(source_manifest.parent, staged_indices)

    relocated = json.loads(json.dumps(manifest))
    relocated["dataset"]["compact_dir"] = str(staged_compact)
    relocated["dataset"]["index_dir"] = str(staged_index)
    relocated["dataset"]["staged_from_manifest"] = str(source_manifest)
    for domain_views in relocated["views"].values():
        for view in domain_views.values():
            for row in view["shards"]:
                shard_name = Path(row["token_file"]).name
                index_name_for_shard = Path(row["raw_indices_file"]).name
                row["token_file"] = str(staged_compact / shard_name)
                row["raw_indices_file"] = str(
                    staged_indices
                    / row["domain"]
                    / row["split"]
                    / index_name_for_shard
                )
    staged_manifest = staged_indices / "manifest.json"
    atomic_json(staged_manifest, relocated)

    # Validate every relocated file and all inexpensive size invariants.  This
    # deliberately avoids content hashes: the source release has already been
    # audited and the user requested no repeated SHA verification.
    token_bytes = 0
    index_bytes = 0
    for path in sorted(staged_compact.glob("shard_*.bin")):
        token_bytes += path.stat().st_size
    for path in sorted(staged_index.iterdir()):
        if path.is_file():
            index_bytes += path.stat().st_size
    for domain_views in relocated["views"].values():
        for view in domain_views.values():
            for row in view["shards"]:
                token_path = Path(row["token_file"])
                raw_indices_path = Path(row["raw_indices_file"])
                if not token_path.is_file() or not raw_indices_path.is_file():
                    raise FileNotFoundError(f"incomplete tmpfs stage: {token_path} / {raw_indices_path}")
                expected = int(row["records"]) * 4
                if raw_indices_path.stat().st_size != expected:
                    raise RuntimeError(f"raw-index byte count mismatch: {raw_indices_path}")

    ready = {
        "status": "ready",
        "source_manifest": str(source_manifest),
        "staged_manifest": str(staged_manifest),
        "compact_bytes": token_bytes,
        "index_bytes": index_bytes,
    }
    atomic_json(destination / "READY.json", ready)
    print(json.dumps(ready, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
