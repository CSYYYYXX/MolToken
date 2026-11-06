#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_LEGACY_RLAUNCH:-0}" != "1" ]; then
  echo "legacy rlaunch entry point is disabled on the physical host; set ALLOW_LEGACY_RLAUNCH=1 only for historical cluster replay" >&2
  exit 2
fi

# Jump-host controller for C4 holdout-loss evaluation.
# Shards the 67 completed runs by checkpoint size (make_c4_shards.py, LPT balance)
# and launches one detached 1-GPU worker per shard. Each worker imports the
# training script directly and recomputes holdout loss on the clean stride-val
# set under the identical training protocol. Idempotent: runs that already have
# holdout_loss.json are skipped by the sharder.

PROJECT_ROOT=${PROJECT_ROOT:-.}
RUN_ROOT=${RUN_ROOT:-./storage/molscaling/runs}
C4_ROOT=${C4_ROOT:-$RUN_ROOT/c4_holdout_eval}
SHARD_DIR=${SHARD_DIR:-$C4_ROOT/shards}
RESULTS_CSV=${RESULTS_CSV:-$RUN_ROOT/scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1/plots/all_fixed_flops_10curves/all_fixed_flops_results.csv}
NUM_SHARDS=${NUM_SHARDS:-10}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
MAX_BATCHES=${MAX_BATCHES:--1}
OUT_NAME=${OUT_NAME:-holdout_loss.json}
CHARGED_GROUP=${CHARGED_GROUP:-evoagi_gpu}
MAX_WAIT=${MAX_WAIT:-6h}
WORKER_GC_TIME=${WORKER_GC_TIME:-7d}
RLAUNCH_CPU=${RLAUNCH_CPU:-12}
RLAUNCH_MEMORY=${RLAUNCH_MEMORY:-72000}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/envs/scaling-csy}

log_dir="$C4_ROOT/launcher_logs"
mkdir -p "$log_dir" "$SHARD_DIR"

# Build size-balanced shards (only runs still missing holdout_loss.json).
"${CONDA_ENV_PATH}/bin/python" "$PROJECT_ROOT/experiments/scaling/make_c4_shards.py" \
  --csv-path "$RESULTS_CSV" --num-shards "$NUM_SHARDS" \
  --out-dir "$SHARD_DIR" --out-name "$OUT_NAME"

shopt -s nullglob
shard_files=("$SHARD_DIR"/shard_*.txt)
if [ "${#shard_files[@]}" -eq 0 ]; then
  echo "no shard files produced (nothing to evaluate)"
  exit 0
fi

for shard_file in "${shard_files[@]}"; do
  tag="c4_$(basename "$shard_file" .txt)"
  client_log="$log_dir/${tag}_rlaunch_$(date +%Y%m%d_%H%M%S).log"
  echo "submitting $tag shard=$shard_file log=$client_log"
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
export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-}
cd '$PROJECT_ROOT'
'${CONDA_ENV_PATH}/bin/python' experiments/scaling/eval_c4_holdout_loss.py \\
  --runs-file '$shard_file' \\
  --eval-batch-size '$EVAL_BATCH_SIZE' \\
  --max-batches '$MAX_BATCHES' \\
  --out-name '$OUT_NAME'
" >"$client_log" 2>&1
  sleep 1
done

echo "submitted ${#shard_files[@]} C4 holdout-eval workers"
