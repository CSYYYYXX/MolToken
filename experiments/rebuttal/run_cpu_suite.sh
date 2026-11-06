#!/usr/bin/env bash
set -euo pipefail
PIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOL=${MOL:?Set MOL to the local molscaling asset directory}
RUN=${RUN:-$MOL/rebuttal_experiments/e2e3_20261014_v1}
PY=${PYTHON:-python}

export PYTHONPATH="$RUN/runtime:$PIPE"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$RUN/tmp" HOLOMOL_DATA_ROOT="$MOL/data"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'status=$?; if (( status != 0 )); then printf "CPU_SUITE_FAILED exit=%s line=%s\n" "$status" "$LINENO" > "$RUN/logs/cpu_suite.failed"; fi' EXIT
for ((attempt=0; attempt<360; attempt++)); do
  if [[ -f "$RUN/tokenizers/fit_complete.json" ]]; then break; fi
  sleep 10
done
test -f "$RUN/tokenizers/fit_complete.json"
"$PY" -u "$PIPE/experiments/rebuttal/prepare_training_worker.py" --run "$RUN" --mode moses --workers "${WORKERS:-4}" > "$RUN/logs/prepare_moses.log" 2>&1
"$PY" -u "$PIPE/experiments/rebuttal/prepare_training_worker.py" --run "$RUN" --mode catalog --workers "${WORKERS:-4}" > "$RUN/logs/prepare_catalog.log" 2>&1
"$PY" -u "$PIPE/experiments/rebuttal/prepare_training_worker.py" --run "$RUN" --mode stream --scope joint --seed 42 --workers "${WORKERS:-4}" > "$RUN/logs/prepare_joint_stream.log" 2>&1
"$PY" -u "$PIPE/experiments/rebuttal/prepare_training_worker.py" --run "$RUN" --mode stream --scope small --seed 42 --workers "${WORKERS:-4}" > "$RUN/logs/prepare_small_stream.log" 2>&1
printf 'CPU_SUITE_COMPLETE\n'
