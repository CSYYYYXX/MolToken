#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(cd "$PIPELINE_ROOT/../../.." && pwd)
HOST_LAUNCHER=$SCRIPT_DIR/launch_domainwise_phase_host.sh
ACTION=${1:-status}
PHASE=${2:-}
[[ "$PHASE" == "c" || "$PHASE" == "d" ]] || {
  echo "phase must be c or d" >&2
  exit 2
}

REMOTE_HOST=${HOLOMOL_SCALING_REMOTE_HOST:-user@compute-host}
REMOTE_PORT=${HOLOMOL_SCALING_REMOTE_PORT:-10090}
REMOTE_IDENTITY=${HOLOMOL_SCALING_REMOTE_IDENTITY:-$HOME/.ssh/id_rsa}
REMOTE_WORKSPACE=${HOLOMOL_SCALING_REMOTE_WORKSPACE:-/path/to/moltoken-workspace}
LOCAL_PYTHON=${HOLOMOL_SCALING_ENV_ROOT:-$WORKSPACE_ROOT/local_runtime/envs/holomol-py311-cu121}/bin/python
REMOTE_PYTHON=${HOLOMOL_SCALING_REMOTE_ENV_ROOT:-$REMOTE_WORKSPACE/local_runtime/envs/holomol-py311-cu121}/bin/python
CAMPAIGN=${HOLOMOL_SCALING_EXPERIMENT_NAME:-domainwise_scaling_fddfs_mf80000_v1_followup}
RUN_ROOT=${HOLOMOL_SCALING_RUN_ROOT:-$WORKSPACE_ROOT/local_runtime/runs/scaling}
MATRIX_JSON=${MATRIX_JSON:-$RUN_ROOT/$CAMPAIGN/phase_$PHASE/matrix.json}
DEFAULT_STAGED_MANIFEST=/dev/shm/holomol-$(id -u)/scaling/domainwise_dataset_v1/domain_split_manifests_v1/manifest.json
LOCAL_MANIFEST=${HOLOMOL_SCALING_LOCAL_DOMAIN_MANIFEST:-${HOLOMOL_SCALING_DOMAIN_MANIFEST:-$DEFAULT_STAGED_MANIFEST}}
REMOTE_MANIFEST=${HOLOMOL_SCALING_REMOTE_DOMAIN_MANIFEST:-/dev/shm/holomol-$(id -u)/scaling/domainwise_dataset_v1/domain_split_manifests_v1/manifest.json}
REMOTE_LAUNCHER=$REMOTE_WORKSPACE/HoloMol/code/pipeline/experiments/scaling/launch_domainwise_phase_host.sh
REMOTE_MATRIX=${HOLOMOL_SCALING_REMOTE_MATRIX_JSON:-$REMOTE_WORKSPACE/local_runtime/runs/scaling/$CAMPAIGN/phase_$PHASE/matrix.json}
LIFECYCLE_TOOL=$SCRIPT_DIR/domainwise_phase_lifecycle.py
LAUNCH_ACTION_ID=${HOLOMOL_SCALING_LAUNCH_ACTION_ID:-twohost_${PHASE}_$$_$(date +%Y%m%dT%H%M%S)}
PHASE_ROOT=$(
  "$LOCAL_PYTHON" -c 'import json,pathlib,sys; print(pathlib.Path(json.loads(pathlib.Path(sys.argv[1]).read_text())["metadata"]["output_base"]).expanduser().resolve())' "$MATRIX_JSON" 2>/dev/null || true
)
PHASE_ROOT=${PHASE_ROOT:-$RUN_ROOT/$CAMPAIGN/phase_$PHASE}

[[ -f "$REMOTE_IDENTITY" ]] || { echo "missing SSH identity: $REMOTE_IDENTITY" >&2; exit 2; }
[[ "$REMOTE_PORT" =~ ^[0-9]+$ ]] && (( REMOTE_PORT > 0 && REMOTE_PORT < 65536 )) || {
  echo "invalid remote port: $REMOTE_PORT" >&2
  exit 2
}
ssh_args=(-i "$REMOTE_IDENTITY" -p "$REMOTE_PORT" -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST")

run_local() {
  local requested_action=$1
  local lifecycle_action_id=${2:-}
  local lifecycle_sequence=${3:-}
  HOLOMOL_SCALING_LIFECYCLE_ACTION_ID="$lifecycle_action_id" \
    HOLOMOL_SCALING_LIFECYCLE_SEQUENCE="$lifecycle_sequence" \
    HOLOMOL_SCALING_DOMAIN_MANIFEST="$LOCAL_MANIFEST" MATRIX_JSON="$MATRIX_JSON" \
    bash "$HOST_LAUNCHER" "$requested_action" "$PHASE" local
}

run_remote() {
  local requested_action=$1
  local lifecycle_action_id=${2:-}
  local lifecycle_sequence=${3:-}
  local remote_command
  printf -v remote_command \
    'HOLOMOL_SCALING_ENV_ROOT=%q HOLOMOL_SCALING_EXPERIMENT_NAME=%q HOLOMOL_SCALING_RUN_ROOT=%q HOLOMOL_SCALING_DOMAIN_MANIFEST=%q HOLOMOL_SCALING_LIFECYCLE_ACTION_ID=%q HOLOMOL_SCALING_LIFECYCLE_SEQUENCE=%q MATRIX_JSON=%q bash %q %q %q remote' \
    "$(dirname "$(dirname "$REMOTE_PYTHON")")" "$CAMPAIGN" "$RUN_ROOT" "$REMOTE_MANIFEST" "$lifecycle_action_id" "$lifecycle_sequence" "$REMOTE_MATRIX" "$REMOTE_LAUNCHER" "$requested_action" "$PHASE"
  ssh "${ssh_args[@]}" "$remote_command"
}

prepare_cleanup() {
  local host_id=$1
  local label=$2
  local action_id="${LAUNCH_ACTION_ID}_${label}_${host_id}"
  local intent_json sequence
  intent_json=$(
    "$LOCAL_PYTHON" "$LIFECYCLE_TOOL" prepare-cleanup \
      --phase-root "$PHASE_ROOT" --phase "$PHASE" --host-id "$host_id" \
      --action-id "$action_id"
  )
  sequence=$(
    "$LOCAL_PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["sequence"])' "$intent_json"
  )
  printf '%s|%s\n' "$action_id" "$sequence"
}

stop_with_receipt() {
  local host_id=$1
  local label=$2
  local prepared action_id sequence
  prepared=$(prepare_cleanup "$host_id" "$label")
  IFS='|' read -r action_id sequence <<<"$prepared"
  if [[ "$host_id" == "local" ]]; then
    run_local stop "$action_id" "$sequence"
  else
    run_remote stop "$action_id" "$sequence"
  fi
}

local_matrix_identity() {
  "$LOCAL_PYTHON" - "$MATRIX_JSON" <<'PY'
import json, pathlib, sys
metadata = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["metadata"]
print(f"{metadata['generation_id']}|{metadata['semantic_fingerprint']['value']}")
PY
}

remote_matrix_identity() {
  local remote_command
  printf -v remote_command \
    '%q -c %q %q' \
    "$REMOTE_PYTHON" \
    'import json,pathlib,sys; m=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["metadata"]; print(f"{m['"'"'generation_id'"'"']}|{m['"'"'semantic_fingerprint'"'"']['"'"'value'"'"']}")' \
    "$REMOTE_MATRIX"
  ssh "${ssh_args[@]}" "$remote_command"
}

assert_same_global_matrix() {
  local local_identity remote_identity
  local_identity=$(local_matrix_identity)
  remote_identity=$(remote_matrix_identity)
  [[ -n "$local_identity" && "$local_identity" == "$remote_identity" ]] || {
    echo "local/remote matrix identity mismatch: local=$local_identity remote=$remote_identity" >&2
    return 1
  }
}

case "$ACTION" in
  start)
    # Both hosts must pass every data/environment/GPU/handoff check before
    # either host is allowed to create a formal worker.
    assert_same_global_matrix
    run_local preflight
    run_remote preflight
    run_local start
    if ! run_remote start; then
      echo "remote launch failed; rolling back the identity-verified local supervisor" >&2
      stop_with_receipt local remote_start_failure || true
      exit 1
    fi
    if ! run_remote status; then
      echo "remote detached identity check failed; rolling back both hosts" >&2
      stop_with_receipt remote remote_status_failure || true
      stop_with_receipt local remote_status_failure || true
      exit 1
    fi
    ;;
  preflight)
    assert_same_global_matrix
    run_local preflight
    run_remote preflight
    ;;
  stop)
    local_rc=0
    remote_rc=0
    stop_with_receipt remote requested_stop || remote_rc=$?
    stop_with_receipt local requested_stop || local_rc=$?
    if [[ "$local_rc" -ne 0 || "$remote_rc" -ne 0 ]]; then
      exit 1
    fi
    ;;
  status)
    assert_same_global_matrix
    local_rc=0
    remote_rc=0
    run_local status || local_rc=$?
    run_remote status || remote_rc=$?
    if [[ "$local_rc" -ne 0 || "$remote_rc" -ne 0 ]]; then
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 {start|status|preflight|stop} {c|d}" >&2
    exit 2
    ;;
esac
