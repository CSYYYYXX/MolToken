#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(cd "$PIPELINE_ROOT/../../.." && pwd)
PYTHON=${HOLOMOL_SCALING_ENV_ROOT:-$WORKSPACE_ROOT/local_runtime/envs/holomol-py311-cu121}/bin/python
CONTROLLER=$SCRIPT_DIR/orchestrate_domainwise_145_campaign.py
FOLLOWUP_ROOT=${HOLOMOL_SCALING_FOLLOWUP_ROOT:-$WORKSPACE_ROOT/local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1_followup}
CONTROLLER_ROOT=$FOLLOWUP_ROOT/controller
STATE_JSON=$CONTROLLER_ROOT/state.json
PID_JSON=$CONTROLLER_ROOT/controller.pid.json
CONSOLE_LOG=$CONTROLLER_ROOT/console.log
LOCK_FILE=/dev/shm/holomol-$(id -u)/scaling/domainwise_145_controller.lock
ACTION=${1:-status}
shift || true

[[ -x "$PYTHON" ]] || { echo "missing environment Python: $PYTHON" >&2; exit 2; }
[[ -f "$CONTROLLER" ]] || { echo "missing controller: $CONTROLLER" >&2; exit 2; }

validate_identity() {
  local expected_nonce=${1:-}
  "$PYTHON" - "$PID_JSON" "$CONTROLLER" "$STATE_JSON" "$expected_nonce" <<'PY'
import json
import os
import pathlib
import sys

pid_path, script_raw, state_raw, nonce = sys.argv[1:]
record = json.loads(pathlib.Path(pid_path).read_text(encoding="utf-8"))
if record.get("schema_version") != "domainwise_145_controller_identity_v1":
    raise SystemExit(1)
pid = int(record["pid"])
proc = pathlib.Path("/proc") / str(pid)
if proc.stat().st_uid != os.getuid():
    raise SystemExit(1)
raw = (proc / "stat").read_text(encoding="utf-8")
tail = raw[raw.rfind(")") + 2:].split()
cmd = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
actual = {
    "pid": pid,
    "ppid": int(tail[1]),
    "pgid": int(tail[2]),
    "sid": int(tail[3]),
    "start_ticks": int(tail[19]),
}
if any(int(record.get(key, -1)) != value for key, value in actual.items()):
    raise SystemExit(1)
if actual["ppid"] != 1 or actual["pgid"] != pid or actual["sid"] != pid:
    raise SystemExit(1)
script = pathlib.Path(script_raw).resolve()
if not any(pathlib.Path(token).name == script.name and pathlib.Path(token).resolve() == script for token in cmd):
    raise SystemExit(1)
def one(flag):
    indexes = [i for i, value in enumerate(cmd) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(cmd):
        raise SystemExit(1)
    return cmd[indexes[0] + 1]
if pathlib.Path(one("--state-json")).resolve() != pathlib.Path(state_raw).resolve():
    raise SystemExit(1)
if pathlib.Path(one("--pid-json")).resolve() != pathlib.Path(pid_path).resolve():
    raise SystemExit(1)
if "--require-detached" not in cmd:
    raise SystemExit(1)
if nonce and (record.get("generation_nonce") != nonce or one("--generation-nonce") != nonce):
    raise SystemExit(1)
print(pid)
PY
}

state_summary() {
  if [[ -s "$STATE_JSON" ]]; then
    "$PYTHON" - "$STATE_JSON" <<'PY'
import json, pathlib, sys
x=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({"step":x.get("step"),"revision":x.get("revision"),"terminal":x.get("terminal"),"halt":x.get("halt"),"updated_at":x.get("updated_at")},ensure_ascii=False))
PY
  else
    echo '{"step":"not_initialized"}'
  fi
}

case "$ACTION" in
  status)
    if [[ -s "$PID_JSON" ]] && pid=$(validate_identity 2>/dev/null); then
      echo "running pid=$pid state=$STATE_JSON"
      state_summary
      exit 0
    fi
    echo "not_running state=$STATE_JSON"
    state_summary
    exit 1
    ;;
  start)
    for token in "$@"; do
      case "$token" in
        --generation-nonce|--generation-nonce=*|--state-json|--state-json=*|--pid-json|--pid-json=*|--lock-file|--lock-file=*|--python|--python=*|--followup-root|--followup-root=*|--require-detached|--require-detached=*|--once|--once=*|--idle-probe|--idle-probe=*|--probe-host-id|--probe-host-id=*|--job-probe-json|--job-probe-json=*|--cleanup-job-json|--cleanup-job-json=*|--flock-probe-path|--flock-probe-path=*)
          echo "launcher owns reserved controller option: $token" >&2
          exit 2
          ;;
      esac
    done
    if [[ -s "$PID_JSON" ]] && validate_identity >/dev/null 2>&1; then
      echo "controller is already running"
      "$0" status
      exit 0
    fi
    if [[ -s "$STATE_JSON" ]]; then
      step=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$STATE_JSON")
      if [[ "$step" == "complete" || "$step" == "halted" ]]; then
        echo "refusing to start terminal controller state=$step; inspect $STATE_JSON" >&2
        exit 1
      fi
    fi
    mkdir -p "$CONTROLLER_ROOT" "$(dirname "$LOCK_FILE")"
    nonce="controller_$(date +%Y%m%dT%H%M%S)_$$_$RANDOM"
    nohup setsid -f "$PYTHON" "$CONTROLLER" \
      --python "$PYTHON" \
      --followup-root "$FOLLOWUP_ROOT" \
      --state-json "$STATE_JSON" \
      --pid-json "$PID_JSON" \
      --lock-file "$LOCK_FILE" \
      --generation-nonce "$nonce" \
      --require-detached \
      "$@" \
      >>"$CONSOLE_LOG" 2>&1 </dev/null
    for _ in $(seq 1 120); do
      if [[ -s "$PID_JSON" ]] && pid=$(validate_identity "$nonce" 2>/dev/null); then
        echo "started pid=$pid nonce=$nonce state=$STATE_JSON log=$CONSOLE_LOG"
        exit 0
      fi
      if [[ -s "$STATE_JSON" ]]; then
        step=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("step"))' "$STATE_JSON" 2>/dev/null || true)
        if [[ "$step" == "halted" || "$step" == "complete" || "$step" == "manual_review_required" ]]; then
          echo "controller reached non-running gate during startup: $step" >&2
          state_summary >&2
          exit 1
        fi
      fi
      sleep 0.5
    done
    echo "timed out waiting for detached controller identity" >&2
    tail -n 100 "$CONSOLE_LOG" >&2 || true
    exit 1
    ;;
  stop)
    if [[ ! -s "$PID_JSON" ]] || ! pid=$(validate_identity 2>/dev/null); then
      echo "controller is not running; no signal sent"
      state_summary
      exit 0
    fi
    kill -TERM "$pid"
    for _ in $(seq 1 120); do
      if ! validate_identity >/dev/null 2>&1; then
        echo "controller stopped pid=$pid; training/smoke jobs were not blindly signalled"
        exit 0
      fi
      sleep 0.5
    done
    echo "controller did not exit after TERM; refusing unverified escalation" >&2
    exit 1
    ;;
  *)
    echo "usage: $0 {start|status|stop} [controller options]" >&2
    exit 2
    ;;
esac
