#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(cd "$PIPELINE_ROOT/../../.." && pwd)
PYTHON=${HOLOMOL_SCALING_ENV_ROOT:-$WORKSPACE_ROOT/local_runtime/envs/holomol-py311-cu121}/bin/python
CAMPAIGN_ROOT=${HOLOMOL_SCALING_CAMPAIGN_ROOT:-${HOLOMOL_SCALING_RUN_ROOT:-$WORKSPACE_ROOT/local_runtime/runs/scaling}/domainwise_scaling_fddfs_mf80000_v1}
ACTION=${1:-start}
HOST_ID=${2:-${HOLOMOL_SCALING_HOST_ID:-local}}
[[ "$HOST_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "unsafe host id: $HOST_ID" >&2; exit 2; }
MATRIX_JSON=${MATRIX_JSON:-$CAMPAIGN_ROOT/phase_b_matrix_${HOST_ID}.json}
SUPERVISOR_LOG=$CAMPAIGN_ROOT/supervisor_${HOST_ID}_console.log
SUPERVISOR_STATUS=$CAMPAIGN_ROOT/supervisor_${HOST_ID}_status.jsonl
PID_FILE=$CAMPAIGN_ROOT/supervisor_${HOST_ID}.pid
DOMAIN_SPLIT_MANIFEST=${HOLOMOL_SCALING_DOMAIN_MANIFEST:-}
LAUNCH_AUTHORITY=${HOLOMOL_SCALING_LAUNCH_AUTHORITY:-}
LAUNCH_NONCE=${HOLOMOL_SCALING_LAUNCH_NONCE:-}

authority_allowed() {
  [[ -n "$LAUNCH_AUTHORITY" && -n "$LAUNCH_NONCE" ]] || return 1
  "$PYTHON" - "$LAUNCH_AUTHORITY" "$HOST_ID" "$LAUNCH_NONCE" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).expanduser().resolve()
value = json.loads(path.read_text(encoding="utf-8"))
valid = (
    value.get("schema_version") == "domainwise_phase_b_launch_authority_v1"
    and value.get("host_id") == sys.argv[2]
    and value.get("allowed") is True
    and value.get("launch_nonce") == sys.argv[3]
)
raise SystemExit(0 if valid else 3)
PY
}

mkdir -p "$CAMPAIGN_ROOT"

status() {
  if [[ -s "$PID_FILE" ]]; then
    pid=$(tr -d '[:space:]' < "$PID_FILE")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "running host=$HOST_ID pid=$pid matrix=$MATRIX_JSON"
      tail -n 20 "$SUPERVISOR_STATUS" 2>/dev/null || true
      return 0
    fi
  fi
  echo "not running host=$HOST_ID matrix=$MATRIX_JSON"
  tail -n 20 "$SUPERVISOR_STATUS" 2>/dev/null || true
  return 1
}

case "$ACTION" in
  status)
    status
    ;;
  start)
    if [[ -n "$LAUNCH_AUTHORITY" || -n "$LAUNCH_NONCE" ]]; then
      [[ -n "$LAUNCH_AUTHORITY" && -n "$LAUNCH_NONCE" ]] || {
        echo "incomplete Phase-B controller launch authority" >&2
        exit 3
      }
      authority_lock=${LAUNCH_AUTHORITY%.json}.lock
      [[ "$authority_lock" != "$LAUNCH_AUTHORITY" ]] || {
        echo "Phase-B launch authority must end in .json" >&2
        exit 3
      }
      mkdir -p "$(dirname "$authority_lock")"
      exec 9>>"$authority_lock"
      flock -x 9
      authority_allowed || {
        echo "Phase-B launch authority is absent, revoked, or stale" >&2
        exit 3
      }
    fi
    if status >/dev/null 2>&1; then
      echo "Phase-B supervisor is already running"
      status
      exit 0
    fi
    [[ -x "$PYTHON" ]] || { echo "missing environment Python: $PYTHON" >&2; exit 2; }
    [[ -s "$MATRIX_JSON" ]] || { echo "missing runnable Phase-B matrix: $MATRIX_JSON" >&2; exit 2; }
    matrix_host_id=$(
      "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"]["host_id"])' "$MATRIX_JSON"
    )
    [[ "$matrix_host_id" == "$HOST_ID" ]] || {
      echo "matrix host_id mismatch: requested=$HOST_ID matrix=$matrix_host_id" >&2
      exit 2
    }
    manifest_args=()
    if [[ -n "$DOMAIN_SPLIT_MANIFEST" ]]; then
      manifest_args+=(--domain-split-manifest "$DOMAIN_SPLIT_MANIFEST")
    fi
    if [[ -n "$LAUNCH_AUTHORITY" ]]; then
      # Re-check while holding the same shared-storage lock used by controller
      # revocation. A delayed SSH start cannot cross the B60 handoff gate.
      authority_allowed || {
        echo "Phase-B launch authority was revoked before spawn" >&2
        exit 3
      }
    fi
    nohup setsid "$PYTHON" "$SCRIPT_DIR/supervise_domainwise_phase_b_local.py" \
      --matrix-json "$MATRIX_JSON" \
      --pipeline-root "$PIPELINE_ROOT" \
      --python "$PYTHON" \
      "${manifest_args[@]}" \
      9>&- \
      >"$SUPERVISOR_LOG" 2>&1 </dev/null &
    launched_pid=$!
    echo "$launched_pid" > "$PID_FILE"
    sleep 2
    if ! kill -0 "$launched_pid" 2>/dev/null; then
      echo "Phase-B supervisor failed during preflight; see $SUPERVISOR_LOG" >&2
      tail -n 80 "$SUPERVISOR_LOG" >&2 || true
      exit 1
    fi
    echo "started Phase-B supervisor host=$HOST_ID pid=$launched_pid log=$SUPERVISOR_LOG"
    ;;
  *)
    echo "usage: $0 [start|status] [local|remote]" >&2
    exit 2
    ;;
esac
