#!/usr/bin/env bash
set -euo pipefail
PIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOL=${MOL:?Set MOL to the local molscaling asset directory}
RUN=${RUN:-$MOL/rebuttal_experiments/e2e3_20261014_v1}

export PYTHONPATH="$RUN/runtime:$PIPE"

export PYTHONDONTWRITEBYTECODE=1 HOLOMOL_DATA_ROOT="$MOL/data"
export TMPDIR="${TMPDIR:-$RUN/tmp}"
export XDG_CACHE_HOME="$RUN/runtime/cache" HF_HOME="$RUN/runtime/cache/huggingface" TORCH_HOME="$RUN/runtime/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec "${PYTHON:-python}" -u "$PIPE/experiments/rebuttal/run_gpu_suite.py" --run "$RUN" --seeds 42 43 44
