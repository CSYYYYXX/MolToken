#!/usr/bin/env bash
# Run inside an existing worker. Does not request GPUs or stop baseline runs.
set -euo pipefail
devices="$1"
offset="$2"
worker_tag="$3"
pipeline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${pipeline_root}/ops/cluster/domainwise_h200_environment.sh"
storage="${HOLOMOL_CLUSTER_STORAGE}"
logs="${storage}/logs/domainwise_scaling_20260906"
mkdir -p "${logs}"
log="${logs}/supplement_queue_${worker_tag}_20260906.log"
nohup "${HOLOMOL_SCALING_PYTHON}" \
  "${pipeline_root}/experiments/scaling/run_domainwise_supplement_queue.py" \
  --plan-list-json "${pipeline_root}/ops/cluster/domainwise_supplement_plan_list_20260906.json" \
  --baseline-plan-json "${storage}/runs/domainwise_scaling_fddfs_mf80000_v1_followup/phase_c/cluster_relaunch_plan_v2_mem70.json" \
  --device-indices "${devices}" --global-gpu-offset "${offset}" \
  --python "${HOLOMOL_SCALING_PYTHON}" --pipeline-root "${pipeline_root}" \
  --output-dir "${storage}/runs/domainwise145_h200_20260906/adaptive_supplement_queue" \
  --status-dir "${logs}/adaptive_supplement_queue" \
  --poll-seconds 60 --num-workers-per-trainer 2 \
  >> "${log}" 2>&1 </dev/null &
printf 'supplement queue started pid=%s log=%s\n' "$!" "${log}"
