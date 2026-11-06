#!/usr/bin/env bash
set -euo pipefail

# Start one allocation-local completion supervisor in the background.  Run
# this inside the allocated worker; the Python runner maps allocation-local
# CUDA ordinals without changing CUDA_VISIBLE_DEVICES.

if [ "$#" -lt 2 ]; then
  echo "usage: $0 LOG_FILE RUNNER_ARGUMENT..." >&2
  exit 2
fi

LOG_FILE=$1
shift
PYTHON=${HOLOMOL_SCALING_PYTHON:-/data/envs/holomol-py311-cu121/bin/python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/run_domainwise_cluster_completion.py"

mkdir -p "$(dirname "$LOG_FILE")"
nohup "$PYTHON" "$RUNNER" "$@" >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
echo "started pid=$PID log=$LOG_FILE"
