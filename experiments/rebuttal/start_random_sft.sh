#!/usr/bin/env bash
set -euo pipefail
PIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOL=${MOL:?Set MOL to the local molscaling asset directory}
RUN=${RUN:-$MOL/rebuttal_experiments/e2e3_20261014_v1}
PY=${PYTHON:-python}

export PYTHONPATH="$RUN/runtime:$PIPE"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$RUN/tmp" HOLOMOL_DATA_ROOT="$MOL/data"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'status=$?; if (( status != 0 )); then printf "RANDOM_SFT_FAILED exit=%s line=%s\n" "$status" "$LINENO" > "$RUN/logs/random_sft.failed"; fi' EXIT
for ((attempt=0; attempt<720; attempt++)); do
  test ! -f "$RUN/logs/cpu_suite.failed"
  if [[ -f "$RUN/moses/manifest.json" ]]; then break; fi
  sleep 10
done
test -f "$RUN/moses/manifest.json"
"$PY" -u "$PIPE/experiments/rebuttal/train_worker.py" --run "$RUN" --name smoke_graph_sft_42 --mode sft --variant graph --seed 42 --smoke-steps 2 > "$RUN/logs/smoke_graph_sft_42.log" 2>&1
test -f "$RUN/models/smoke_graph_sft_42/smoke_complete.json"
"$PY" -u "$PIPE/experiments/rebuttal/train_worker.py" --run "$RUN" --name graph_random_sft_42 --mode sft --variant graph --seed 42 > "$RUN/logs/graph_random_sft_42.log" 2>&1
printf 'RANDOM_SFT_COMPLETE\n'
