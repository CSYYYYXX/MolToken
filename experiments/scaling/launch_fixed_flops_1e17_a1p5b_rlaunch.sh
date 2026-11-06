#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_LEGACY_RLAUNCH:-0}" != "1" ]; then
  echo "legacy rlaunch entry point is disabled on the physical host; set ALLOW_LEGACY_RLAUNCH=1 only for historical cluster replay" >&2
  exit 2
fi

# A1.5b out-of-sample extrapolation check at C = 1e17.
# Three bracket points around the pre-registered vertex N* = 10.86M, all reusing
# existing pure-depth shapes (hidden=256, L=4/8/16).  One 1-GPU worker per model.
# Predictions were FROZEN before this launch:
#   doc/paper_materials_20260728/preregistration_A1p5b/PREDICTIONS_FROZEN.json

PROJECT_ROOT=${PROJECT_ROOT:-.}
RUN_ROOT=${RUN_ROOT:-./storage/molscaling/runs}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scaling_fddfs_mf80000_fixed_flops1e17_a1p5b_domainlog_v1}
MATRIX_JSON=${MATRIX_JSON:-$RUN_ROOT/$EXPERIMENT_NAME/scaling_matrix.json}
CHARGED_GROUP=${CHARGED_GROUP:-evoagi_gpu}
MAX_WAIT=${MAX_WAIT:-6h}
WORKER_GC_TIME=${WORKER_GC_TIME:-7d}
RLAUNCH_CPU=${RLAUNCH_CPU:-12}
RLAUNCH_MEMORY=${RLAUNCH_MEMORY:-72000}

if [ ! -s "$MATRIX_JSON" ]; then
  echo "missing matrix: $MATRIX_JSON" >&2
  exit 2
fi

log_dir="$RUN_ROOT/$EXPERIMENT_NAME/launcher_logs"
mkdir -p "$log_dir"

entries=(
  "m008|0p008b|fx1e17"
  "m011|0p011b|fx1e17"
  "m017|0p017b|fx1e17"
)

for entry in "${entries[@]}"; do
  IFS='|' read -r tag model_id target_labels <<<"$entry"
  client_log="$log_dir/${tag}_rlaunch_$(date +%Y%m%d_%H%M%S).log"
  echo "submitting tag=$tag model=$model_id labels=$target_labels log=$client_log"
  rlaunch \
    --gpu=1 \
    --cpu="$RLAUNCH_CPU" \
    --memory="$RLAUNCH_MEMORY" \
    --charged-group="$CHARGED_GROUP" \
    --private-machine=yes \
    --detach \
    --max-wait-duration="$MAX_WAIT" \
    --worker-garbage-collection-time="$WORKER_GC_TIME" \
    --mount="${HOLOMOL_CLUSTER_MOUNT:?Set the scheduler storage mount}" \
    -e ZSH_VERSION= \
    -- bash --noprofile --norc -lc "
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true
cd '$PROJECT_ROOT'
MATRIX_JSON='$MATRIX_JSON' \\
MODEL_ID='$model_id' \\
TARGET_LABELS='$target_labels' \\
EXPERIMENT_NAME='$EXPERIMENT_NAME' \\
RUN_ROOT='$RUN_ROOT' \\
ALLOW_LEGACY_SCALING_PATHS=1 \\
LOG_DOMAIN_LOSS=1 \\
EVAL_EVERY=0 \\
SAVE_EVERY=0 \\
bash experiments/scaling/run_flops_aligned_model_worker.sh
" >"$client_log" 2>&1
  sleep 1
done

echo "submitted ${#entries[@]} one-GPU workers for A1.5b (C=1e17)"
