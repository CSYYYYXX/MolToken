#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/build/prefix_merge_int_shard"
mkdir -p "$(dirname "$OUT")"

CXX="${CXX:-g++}"
"$CXX" -std=c++17 -O3 \
  "$ROOT/tools/prefix_merge_int_shard.cpp" \
  -o "$OUT"

echo "$OUT"
