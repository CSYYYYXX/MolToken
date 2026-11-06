#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/build/int_bpe_bench"
mkdir -p "$(dirname "$OUT")"

CXX="${CXX:-g++}"
EXTRA_CXXFLAGS=${EXTRA_CXXFLAGS:-}
"$CXX" -std=c++17 -O3 -pthread $EXTRA_CXXFLAGS \
  "$ROOT/tools/int_bpe_bench.cpp" \
  -o "$OUT"

echo "$OUT"
