#!/usr/bin/env bash
set -euo pipefail
PIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOL=${MOL:?Set MOL to the local molscaling asset directory}
RUN=${RUN:-$MOL/rebuttal_experiments/e2e3_20261014_v1}
ASSETS=$MOL/data/downloads_raw/MOSES-dd7ed6ab38e23afd3ef5371d67939a1760bd8599
PY=${PYTHON:-python}

export PYTHONPATH="$ASSETS/source:$RUN/runtime:$PIPE"
export PYTHONDONTWRITEBYTECODE=1 HOLOMOL_DATA_ROOT="$MOL/data"
export TMPDIR="${TMPDIR:-$RUN/tmp}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR"
"$PY" -u "$PIPE/ops/environment/smoke_moses_existing_source_worker.py" \
  --chemnet "$ASSETS/chemnet/ChemNet_v0.13_pretrained.pt" --output "$RUN/preflight/moses_official_smoke.json" > "$RUN/logs/moses_official_smoke.log" 2>&1
"$PY" -u "$PIPE/experiments/rebuttal/evaluate_worker.py" --run "$RUN" --mode generate \
  --checkpoint "$RUN/models/smoke_graph_sft_42/smoke.pt" --variant graph --samples 16 --batch-size 16 \
  --output "$RUN/preflight/generation_smoke" --workers 2 > "$RUN/logs/generation_smoke.log" 2>&1
printf 'EVALUATION_PREFLIGHT_COMPLETE\n'
