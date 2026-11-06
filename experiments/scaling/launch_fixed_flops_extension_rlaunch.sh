#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_LEGACY_RLAUNCH:-0}" != "1" ]; then
  echo "legacy rlaunch entry point is disabled on the physical host; set ALLOW_LEGACY_RLAUNCH=1 only for historical cluster replay" >&2
  exit 2
fi

# Jump-host controller.  Each entry gets an isolated worker and runs one or
# more independent fixed-FLOPs rows sequentially on its scheduler-assigned GPU.

PROJECT_ROOT=${PROJECT_ROOT:-.}
RUN_ROOT=${RUN_ROOT:-./storage/molscaling/runs}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1}
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
  "m009|0p009b|fx5e17"
  "m014|0p014b|fx5e17,fx7e17"
  "m022|0p022b|fx5e17,fx7e17,fx1p3e18"
  "m027|0p027b|fx7e17,fx1p6e18,fx3p5e18"
  "m031|0p031b|fx5e17,fx1p3e18,fx2p4e18"
  "m041a|0p041b|fx5e17,fx7e17,fx1p3e18"
  "m041b|0p041b|fx1p6e18,fx2p9e18,fx3p5e18"
  "m047|0p047b|fx2e18,fx2p4e18,fx3p5e18"
  "m053a|0p053b|fx1p6e18,fx2e18"
  "m053b|0p053b|fx2p9e18,fx3p5e18"
  "m059|0p059b|fx7e17,fx1p3e18,fx2p4e18"
  "m065|0p065b|fx2e18,fx2p9e18"
  "m071|0p071b|fx1p6e18,fx2e18,fx2p4e18,fx3p5e18"
  "m078|0p078b|fx2p9e18"
  "m098|0p098b|fx1p3e18,fx1p6e18,fx2p4e18,fx2p9e18"
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

echo "submitted ${#entries[@]} one-GPU workers"
