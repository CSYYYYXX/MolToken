#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_LEGACY_RLAUNCH:-0}" != "1" ]; then
  echo "legacy rlaunch entry point is disabled on the physical host; set ALLOW_LEGACY_RLAUNCH=1 only for historical cluster replay" >&2
  exit 2
fi

# 在跳板机侧提交 scaling-law 训练矩阵。
# 每个模型规模一个 worker；单个模型只做单节点 DDP，不跨节点。

PROJECT_ROOT=${PROJECT_ROOT:-.}
RUN_ROOT=${RUN_ROOT:-./storage/molscaling/runs}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scaling_fddfs_mf80000_v1}
MATRIX_JSON=${MATRIX_JSON:-$RUN_ROOT/$EXPERIMENT_NAME/scaling_matrix.json}
CHARGED_GROUP=${CHARGED_GROUP:-evoagi_gpu}
MAX_WAIT=${MAX_WAIT:-6h}
MODELS=${MODELS:-"0p06b 0p10b 0p30b"}
PREDICT_ONLY=${PREDICT_ONLY:-0}
REBUILD_MATRIX=${REBUILD_MATRIX:-1}

mkdir -p "$RUN_ROOT/$EXPERIMENT_NAME"

if [ "$REBUILD_MATRIX" = "1" ] || [ ! -s "$MATRIX_JSON" ]; then
  python3 "$PROJECT_ROOT/experiments/scaling/build_scaling_matrix.py" \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-root "$RUN_ROOT"
fi

model_field() {
  local model_id=$1
  local field=$2
  python3 - "$MATRIX_JSON" "$model_id" "$field" <<'PY'
import json
import sys
from pathlib import Path
matrix = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_id = sys.argv[2]
field = sys.argv[3]
rows = [r for r in matrix["rows"] if r["model_id"] == model_id]
if not rows:
    raise SystemExit(f"no rows for model_id={model_id}")
print(rows[0][field])
PY
}

for model_id in $MODELS; do
  gpu_count=$(model_field "$model_id" gpu_count)
  cpu_count=$(model_field "$model_id" cpu_count)
  memory_mib=$(model_field "$model_id" memory_mib)
  model_dir="$RUN_ROOT/$EXPERIMENT_NAME/$model_id"
  mkdir -p "$model_dir"
  client_log="$model_dir/rlaunch_client.log"

  echo "model_id=$model_id gpu=$gpu_count cpu=$cpu_count memory=$memory_mib predict_only=$PREDICT_ONLY"
  if [ "$PREDICT_ONLY" = "1" ]; then
    rlaunch \
      --gpu="$gpu_count" \
      --cpu="$cpu_count" \
      --memory="$memory_mib" \
      --charged-group="$CHARGED_GROUP" \
      --private-machine=yes \
      --mount="${HOLOMOL_CLUSTER_MOUNT:?Set the scheduler storage mount}" \
      --predict-only \
      -- bash
  else
    rlaunch \
      --gpu="$gpu_count" \
      --cpu="$cpu_count" \
      --memory="$memory_mib" \
      --charged-group="$CHARGED_GROUP" \
      --private-machine=yes \
      --detach \
      --max-wait-duration="$MAX_WAIT" \
      --mount="${HOLOMOL_CLUSTER_MOUNT:?Set the scheduler storage mount}" \
      -e ZSH_VERSION= \
      -- bash --noprofile --norc -lc "
set -euo pipefail
cd '$PROJECT_ROOT'
MATRIX_JSON='$MATRIX_JSON' \
MODEL_ID='$model_id' \
EXPERIMENT_NAME='$EXPERIMENT_NAME' \
RUN_ROOT='$RUN_ROOT' \
ALLOW_LEGACY_SCALING_PATHS=1 \
bash experiments/scaling/run_scaling_model_worker.sh
" >"$client_log" 2>&1
    echo "submitted model_id=$model_id client_log=$client_log"
    sleep 0.5
  fi
done
