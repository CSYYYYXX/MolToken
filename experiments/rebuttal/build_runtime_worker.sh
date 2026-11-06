#!/usr/bin/env bash
set -euo pipefail
PIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOL=${MOL:?Set MOL to the local molscaling asset directory}
RUN=${RUN:-$MOL/rebuttal_experiments/e2e3_20261014_v1}
PY=${PYTHON:-python}

export PYTHONPATH="$RUN/runtime:$PIPE"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$RUN/tmp" HOLOMOL_DATA_ROOT="$MOL/data"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN/runtime" "$RUN/tmp" "$RUN/logs"
"$PY" -u -c 'from pipeline.runtime_host import require_compute_host; print(require_compute_host("rebuttal build")); import pybind11,numpy,orjson,zstandard; print("BUILD_IMPORTS_OK")'
mapfile -t includes < <("$PY" -c 'import os,pybind11,sysconfig; print("-I"+pybind11.get_include()); print("-I"+(os.environ.get("PYTHON_INCLUDE_DIR") or sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_path("include")))')
suffix=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
g++ -O3 -shared -std=c++17 -fPIC "${includes[@]}" -DFDDFS_MODULE_NAME=_rebuttal_fddfs_cpp \
  "$PIPE/pipeline/serialization/_stackk_cpp.cpp" -o "$RUN/runtime/_rebuttal_fddfs_cpp$suffix"
"$PY" -u "$PIPE/tests/test_rebuttal_ordering.py" -v
g++ -O3 -std=c++17 -pthread -DINT_BPE_COMPACT_POSITION \
  "$PIPE/tools/yttm_int_bpe/tools/int_bpe_bench.cpp" -o "$RUN/runtime/int_bpe_compact"
"$PY" -u - "$RUN" <<'PY'
import sys
from pathlib import Path
from experiments.diagnostics.native_domain_bpe_probe import smoke
root = Path(sys.argv[1])
smoke(root / 'runtime/int_bpe_compact', root / 'runtime')
print('RUNTIME_BUILD_AND_TESTS_COMPLETE', flush=True)
PY
