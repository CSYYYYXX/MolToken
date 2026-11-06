#!/usr/bin/env python3
"""Build bounded-memory AA source buckets on an allocated worker, offline."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.benchmarks.aa_source_index import build_index
from pipeline.runtime_host import require_compute_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-length", type=int, default=400)
    args = parser.parse_args()
    require_compute_host("full AA source streaming index")
    start = time.time()
    result = build_index(json.loads(args.catalog.read_text()), args.output,
                         maximum_length=args.maximum_length)
    print(json.dumps({"output": str(args.output), "records": result["source_records_total"],
                      "stored": result["stored_records"], "elapsed_seconds": time.time() - start}), flush=True)


if __name__ == "__main__":
    main()
