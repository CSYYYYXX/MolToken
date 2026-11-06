#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(cd "$PIPELINE_ROOT/../../.." && pwd)
PYTHON=${HOLOMOL_SCALING_ENV_ROOT:-$WORKSPACE_ROOT/local_runtime/envs/holomol-py311-cu121}/bin/python
ACTION=${1:-status}
PHASE=${2:-}
HOST_ID=${3:-${HOLOMOL_SCALING_HOST_ID:-local}}

[[ "$PHASE" == "c" || "$PHASE" == "d" ]] || {
  echo "phase must be c or d" >&2
  exit 2
}
[[ "$HOST_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo "unsafe host id: $HOST_ID" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "missing environment Python: $PYTHON" >&2; exit 2; }

CAMPAIGN=${HOLOMOL_SCALING_EXPERIMENT_NAME:-domainwise_scaling_fddfs_mf80000_v1_followup}
RUN_ROOT=${HOLOMOL_SCALING_RUN_ROOT:-$WORKSPACE_ROOT/local_runtime/runs/scaling}
MATRIX_JSON=${MATRIX_JSON:-$RUN_ROOT/$CAMPAIGN/phase_$PHASE/matrix.json}
SUPERVISOR=$SCRIPT_DIR/supervise_domainwise_phase_host.py
DEFAULT_STAGED_MANIFEST=/dev/shm/holomol-$(id -u)/scaling/domainwise_dataset_v1/domain_split_manifests_v1/manifest.json
DOMAIN_SPLIT_MANIFEST=${HOLOMOL_SCALING_DOMAIN_MANIFEST:-$DEFAULT_STAGED_MANIFEST}

matrix_field() {
  "$PYTHON" - "$MATRIX_JSON" "$PHASE" "$HOST_ID" "$SCRIPT_DIR" "$1" <<'PY'
import json
import pathlib
import sys

matrix_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
phase = sys.argv[2]
host = sys.argv[3]
sys.path.insert(0, sys.argv[4])
from supervise_domainwise_phase_host import validate_global_matrix

matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
validate_global_matrix(matrix, phase, host)
field = sys.argv[5]
if field == "root":
    print(pathlib.Path(matrix["metadata"]["output_base"]).expanduser().resolve())
elif field == "generation":
    print(matrix["metadata"]["generation_id"])
elif field == "fingerprint":
    print(matrix["metadata"]["semantic_fingerprint"]["value"])
else:
    raise SystemExit(f"unknown field: {field}")
PY
}

if [[ "$ACTION" != "stop" ]]; then
  [[ -s "$MATRIX_JSON" ]] || { echo "missing formal matrix: $MATRIX_JSON" >&2; exit 2; }
fi
if [[ "$ACTION" == "stop" ]]; then
  # Stop must remain possible when a frozen evidence/review artifact becomes
  # temporarily unreadable.  Read only output_base from the matrix, with the
  # canonical campaign path as a recovery fallback; exact live identity is
  # revalidated from the supervisor PID record below.
  PHASE_ROOT=$(
    "$PYTHON" - "$MATRIX_JSON" <<'PY' 2>/dev/null || true
import json, pathlib, sys
print(pathlib.Path(json.loads(pathlib.Path(sys.argv[1]).read_text())["metadata"]["output_base"]).expanduser().resolve())
PY
  )
  PHASE_ROOT=${PHASE_ROOT:-$RUN_ROOT/$CAMPAIGN/phase_$PHASE}
  MATRIX_GENERATION=""
  MATRIX_FINGERPRINT=""
else
  PHASE_ROOT=$(matrix_field root)
  MATRIX_GENERATION=$(matrix_field generation)
  MATRIX_FINGERPRINT=$(matrix_field fingerprint)
fi
SUPERVISOR_ROOT=$PHASE_ROOT/supervisors/$HOST_ID
SUPERVISOR_LOG=$SUPERVISOR_ROOT/console.log
SUPERVISOR_STATUS=$SUPERVISOR_ROOT/status.jsonl
PID_JSON=$SUPERVISOR_ROOT/supervisor.pid.json
LIFECYCLE_TOOL=$SCRIPT_DIR/domainwise_phase_lifecycle.py
LIFECYCLE_ROOT=$SUPERVISOR_ROOT/lifecycle
LIFECYCLE_LOCK=$LIFECYCLE_ROOT/host_lifecycle.lock
LIFECYCLE_ACTION_ID=${HOLOMOL_SCALING_LIFECYCLE_ACTION_ID:-}
LIFECYCLE_SEQUENCE=${HOLOMOL_SCALING_LIFECYCLE_SEQUENCE:-}

if [[ "$ACTION" == "stop" ]]; then
  if [[ -z "$LIFECYCLE_ACTION_ID" && -z "$LIFECYCLE_SEQUENCE" ]]; then
    LIFECYCLE_ACTION_ID="manual_${PHASE}_${HOST_ID}_$(date +%Y%m%dT%H%M%S)_$$_$RANDOM"
    intent_json=$(
      "$PYTHON" "$LIFECYCLE_TOOL" prepare-cleanup \
        --phase-root "$PHASE_ROOT" --phase "$PHASE" --host-id "$HOST_ID" \
        --action-id "$LIFECYCLE_ACTION_ID"
    )
    LIFECYCLE_SEQUENCE=$(
      "$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["sequence"])' "$intent_json"
    )
  elif [[ -z "$LIFECYCLE_ACTION_ID" || ! "$LIFECYCLE_SEQUENCE" =~ ^[1-9][0-9]*$ ]]; then
    echo "stop requires a complete lifecycle action id/sequence pair" >&2
    exit 2
  fi
fi

if [[ "$ACTION" == "start" || "$ACTION" == "stop" ]]; then
  mkdir -p "$LIFECYCLE_ROOT"
  exec 8>>"$LIFECYCLE_LOCK"
  flock -x 8
fi

if [[ "$ACTION" == "stop" && -s "$PID_JSON" ]]; then
  readarray -t persisted_identity < <(
    "$PYTHON" - "$PID_JSON" <<'PY' 2>/dev/null || true
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value.get("matrix_generation_id", ""))
print(value.get("matrix_semantic_fingerprint", ""))
PY
  )
  MATRIX_GENERATION=${persisted_identity[0]:-}
  MATRIX_FINGERPRINT=${persisted_identity[1]:-}
fi

matching_runtime_pids() {
  local role=$1
  "$PYTHON" - "$role" "$SCRIPT_DIR" "$MATRIX_JSON" "$PHASE" "$HOST_ID" "$PID_JSON" <<'PY'
import json
import os
import pathlib
import sys

role, script_dir_raw, matrix_raw, phase, host, pid_json_raw = sys.argv[1:]
script_dir = pathlib.Path(script_dir_raw).resolve()
expected_script = script_dir / {
    "worker": "run_domainwise_phase_slot.py",
    "guard": "guard_domainwise_phase_thermal.py",
    "supervisor": "supervise_domainwise_phase_host.py",
    "trainer": "../../cli/pretrain/train_qwen3_pretrain.py",
}[role]
expected_script = expected_script.resolve()
matrix = pathlib.Path(matrix_raw).resolve()
matrix_payload = None
if role == "trainer":
    try:
        matrix_payload = json.loads(matrix.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        matrix_payload = {}
allowed_run_dirs = {
    pathlib.Path(row["out_dir"]).expanduser().resolve()
    for row in (matrix_payload or {}).get("rows", [])
    if row.get("assigned_host") == host
}
if role == "trainer" and not allowed_run_dirs:
    try:
        identity = json.loads(pathlib.Path(pid_json_raw).read_text(encoding="utf-8"))
        if (
            identity.get("phase") == phase
            and identity.get("host_id") == host
            and pathlib.Path(identity.get("matrix_json", "")).expanduser().resolve() == matrix
        ):
            allowed_run_dirs = {
                pathlib.Path(value).expanduser().resolve()
                for value in identity.get("owned_run_dirs", [])
            }
    except (OSError, ValueError, TypeError):
        pass
def one(cmd, flag):
    positions = [i for i, value in enumerate(cmd) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(cmd):
        raise ValueError(flag)
    return cmd[positions[0] + 1]
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        if entry.stat().st_uid != os.getuid():
            continue
        raw = (entry / "stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        pid = int(entry.name)
        if tail[0] in {"Z", "X", "x"}:
            continue
        if role != "trainer" and (int(tail[2]) != pid or int(tail[3]) != pid):
            continue
        cmd = [part.decode(errors="replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        if not any(pathlib.Path(value).name == expected_script.name and pathlib.Path(value).resolve() == expected_script for value in cmd):
            continue
        if role == "trainer":
            attempt = pathlib.Path(one(cmd, "--out-dir")).expanduser().resolve()
            if attempt.parent not in allowed_run_dirs or not attempt.name.startswith("attempt_"):
                continue
            print(int(tail[2]))
        else:
            if pathlib.Path(one(cmd, "--matrix-json")).resolve() != matrix or one(cmd, "--host-id") != host:
                continue
            if role in {"guard", "supervisor"} and one(cmd, "--phase") != phase:
                continue
            print(pid)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
        continue
PY
}

validate_pid_identity() {
  local expected_nonce=${1:-}
  "$PYTHON" - "$PID_JSON" "$SUPERVISOR" "$MATRIX_JSON" "$PHASE" "$HOST_ID" "$expected_nonce" "$MATRIX_GENERATION" "$MATRIX_FINGERPRINT" <<'PY'
import json
import os
import pathlib
import sys

pid_path, supervisor_raw, matrix_raw, phase, host, nonce, generation, fingerprint = sys.argv[1:]
record = json.loads(pathlib.Path(pid_path).read_text(encoding="utf-8"))
pid = int(record["pid"])
proc = pathlib.Path("/proc") / str(pid)
if proc.stat().st_uid != os.getuid():
    raise SystemExit(1)
raw = (proc / "stat").read_text(encoding="utf-8")
tail = raw[raw.rfind(")") + 2:].split()
cmd = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
def one(flag):
    positions = [i for i, value in enumerate(cmd) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(cmd):
        raise SystemExit(1)
    return cmd[positions[0] + 1]
def has_script(expected):
    expected = pathlib.Path(expected).resolve()
    return any(pathlib.Path(value).name == expected.name and pathlib.Path(value).resolve() == expected for value in cmd)
identity = {
    "pid": pid,
    "ppid": int(tail[1]),
    "pgid": int(tail[2]),
    "sid": int(tail[3]),
    "start_ticks": int(tail[19]),
}
if any(int(record[key]) != value for key, value in identity.items()):
    raise SystemExit(1)
if identity["ppid"] != 1 or identity["pgid"] != pid or identity["sid"] != pid:
    raise SystemExit(1)
if record.get("phase") != phase or record.get("host_id") != host:
    raise SystemExit(1)
if record.get("matrix_generation_id") != generation or record.get("matrix_semantic_fingerprint") != fingerprint:
    raise SystemExit(1)
if nonce and record.get("generation_nonce") != nonce:
    raise SystemExit(1)
if not has_script(supervisor_raw):
    raise SystemExit(1)
if pathlib.Path(one("--matrix-json")).resolve() != pathlib.Path(matrix_raw).resolve():
    raise SystemExit(1)
if one("--phase") != phase or one("--host-id") != host:
    raise SystemExit(1)
admission_policy = record.get("admission_policy", "strict")
if admission_policy not in {"strict", "defer_unavailable_gpus_v1"}:
    raise SystemExit(1)
if cmd.count("--defer-unavailable-gpus") != int(admission_policy == "defer_unavailable_gpus_v1"):
    raise SystemExit(1)
execution_plan = record.get("execution_plan_json")
if execution_plan:
    if host != "remote" or record.get("physical_host_id") != "local" or admission_policy != "defer_unavailable_gpus_v1":
        raise SystemExit(1)
    if pathlib.Path(one("--local-execution-plan")).resolve() != pathlib.Path(execution_plan).resolve():
        raise SystemExit(1)
elif "--local-execution-plan" in cmd:
    raise SystemExit(1)
print(pid)
PY
}

validate_host_completion() {
  "$PYTHON" - "$MATRIX_JSON" "$PHASE" "$HOST_ID" "$SCRIPT_DIR" "$DOMAIN_SPLIT_MANIFEST" <<'PY'
import pathlib
import sys

matrix_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
phase, host, script_dir, manifest_raw = sys.argv[2:]
sys.path.insert(0, script_dir)
from run_domainwise_phase_slot import completed_attempt, load_json, validate_manifest_for_matrix
from supervise_domainwise_phase_host import validate_global_matrix

matrix = load_json(matrix_path)
rows, _, _, root = validate_global_matrix(matrix, phase, host)
manifest_path = pathlib.Path(manifest_raw).expanduser().resolve()
manifest = validate_manifest_for_matrix(matrix, manifest_path, check_files=True)
marker_path = root / "host_completion" / f"{host}.json"
marker = load_json(marker_path)
metadata = matrix["metadata"]
if not isinstance(marker, dict) or any((
    marker.get("schema_version") != "domainwise_phase_host_completion_v1",
    marker.get("status") != "complete",
    marker.get("phase") != phase,
    marker.get("host_id") != host,
    pathlib.Path(marker.get("matrix_json", "")).expanduser().resolve() != matrix_path,
    marker.get("matrix_generation_id") != metadata["generation_id"],
    marker.get("matrix_semantic_fingerprint") != metadata["semantic_fingerprint"]["value"],
    pathlib.Path(marker.get("domain_split_manifest", "")).expanduser().resolve() != manifest_path,
    int(marker.get("rows", -1)) != len(rows),
)):
    raise SystemExit(1)
completed_ids = []
for row in rows:
    complete = completed_attempt(
        pathlib.Path(row["out_dir"]), row, manifest, matrix_path, manifest_path
    )
    if complete is None:
        raise SystemExit(1)
    completed_ids.append(row["run_id"])
marker_ids = [item.get("run_id") for item in marker.get("completed_runs", [])]
if len(marker_ids) != len(set(marker_ids)) or set(marker_ids) != set(completed_ids):
    raise SystemExit(1)
print(marker_path)
PY
}

status() {
  if [[ -s "$PID_JSON" ]] && pid=$(validate_pid_identity 2>/dev/null); then
    guard_pids=$(matching_runtime_pids guard | paste -sd, -)
    echo "running phase=$PHASE host=$HOST_ID pid=$pid guard_pids=${guard_pids:-none} matrix_generation=$MATRIX_GENERATION"
    tail -n 20 "$SUPERVISOR_STATUS" 2>/dev/null || true
    return 0
  fi
  if completion=$(validate_host_completion 2>/dev/null); then
    echo "already_complete phase=$PHASE host=$HOST_ID completion=$completion matrix_generation=$MATRIX_GENERATION"
    return 0
  fi
  echo "not running phase=$PHASE host=$HOST_ID matrix=$MATRIX_JSON"
  tail -n 20 "$SUPERVISOR_STATUS" 2>/dev/null || true
  return 1
}

manifest_args=(--domain-split-manifest "$DOMAIN_SPLIT_MANIFEST")
disjoint_overlap_args=()
deferred_admission_args=()
local_execution_plan_args=()
if [[ "$ACTION" == "start" || "$ACTION" == "preflight" ]]; then
  case "${HOLOMOL_SCALING_DEFER_UNAVAILABLE_GPUS:-0}" in
    0) ;;
    1) deferred_admission_args=(--defer-unavailable-gpus) ;;
    *)
      echo "HOLOMOL_SCALING_DEFER_UNAVAILABLE_GPUS must be exactly 0 or 1" >&2
      exit 2
      ;;
  esac
  if [[ -n "${HOLOMOL_SCALING_LOCAL_EXECUTION_PLAN:-}" ]]; then
    if [[ "$HOST_ID" != "remote" || "${HOLOMOL_SCALING_DEFER_UNAVAILABLE_GPUS:-0}" != "1" ]]; then
      echo "local execution plan requires logical remote host and deferred admission" >&2
      exit 2
    fi
    [[ -s "$HOLOMOL_SCALING_LOCAL_EXECUTION_PLAN" ]] || { echo "missing local execution plan" >&2; exit 2; }
    local_execution_plan_args=(--local-execution-plan "$HOLOMOL_SCALING_LOCAL_EXECUTION_PLAN")
  fi
  case "${HOLOMOL_SCALING_ALLOW_EXISTING_SCALING_ON_OTHER_GPUS:-0}" in
    0)
      ;;
    1)
      disjoint_overlap_args=(--allow-existing-scaling-processes-on-other-gpus)
      ;;
    *)
      echo "HOLOMOL_SCALING_ALLOW_EXISTING_SCALING_ON_OTHER_GPUS must be exactly 0 or 1" >&2
      exit 2
      ;;
  esac
fi

case "$ACTION" in
  status)
    status
    ;;
  preflight)
    "$PYTHON" "$SUPERVISOR" \
      --matrix-json "$MATRIX_JSON" \
      --phase "$PHASE" \
      --host-id "$HOST_ID" \
      --pipeline-root "$PIPELINE_ROOT" \
      --python "$PYTHON" \
      --preflight-only \
      "${disjoint_overlap_args[@]}" \
      "${deferred_admission_args[@]}" \
      "${local_execution_plan_args[@]}" \
      "${manifest_args[@]}"
    ;;
  start)
    "$PYTHON" "$LIFECYCLE_TOOL" launch-gate \
      --phase-root "$PHASE_ROOT" --phase "$PHASE" --host-id "$HOST_ID"
    if completion=$(validate_host_completion 2>/dev/null); then
      echo "already_complete phase=$PHASE host=$HOST_ID completion=$completion matrix_generation=$MATRIX_GENERATION"
      exit 0
    fi
    if status >/dev/null 2>&1; then
      echo "supervisor is already running"
      status
      exit 0
    fi
    mkdir -p "$SUPERVISOR_ROOT"
    # Complete all expensive and destructive-resource preflight synchronously
    # before creating a detached process.
    "$PYTHON" "$SUPERVISOR" \
      --matrix-json "$MATRIX_JSON" \
      --phase "$PHASE" \
      --host-id "$HOST_ID" \
      --pipeline-root "$PIPELINE_ROOT" \
      --python "$PYTHON" \
      --preflight-only \
      "${disjoint_overlap_args[@]}" \
      "${deferred_admission_args[@]}" \
      "${local_execution_plan_args[@]}" \
      "${manifest_args[@]}"
    nonce="$(date +%Y%m%dT%H%M%S)_$$_$RANDOM"
    launch_log="$SUPERVISOR_ROOT/console.$nonce.log"
    # setsid -f performs the extra fork: the actual supervisor is reparented
    # to PID1, leads its own session, and writes the authoritative PID JSON.
    nohup setsid -f "$PYTHON" "$SUPERVISOR" \
      --matrix-json "$MATRIX_JSON" \
      --phase "$PHASE" \
      --host-id "$HOST_ID" \
      --pipeline-root "$PIPELINE_ROOT" \
      --python "$PYTHON" \
      --generation-nonce "$nonce" \
      --require-detached \
      "${disjoint_overlap_args[@]}" \
      "${deferred_admission_args[@]}" \
      "${local_execution_plan_args[@]}" \
      "${manifest_args[@]}" \
      8>&- \
      >"$launch_log" 2>&1 </dev/null
    for _ in $(seq 1 240); do
      if [[ -s "$PID_JSON" ]] && pid=$(validate_pid_identity "$nonce" 2>/dev/null); then
        guard_pids=$(matching_runtime_pids guard | paste -sd, -)
        if [[ -n "$guard_pids" ]]; then
          echo "started phase=$PHASE host=$HOST_ID pid=$pid guard_pids=$guard_pids nonce=$nonce log=$launch_log"
          exit 0
        fi
      fi
      if completion=$(validate_host_completion 2>/dev/null); then
        echo "already_complete phase=$PHASE host=$HOST_ID completion=$completion matrix_generation=$MATRIX_GENERATION"
        exit 0
      fi
      if grep -qE 'Traceback|RuntimeError|ValueError|FileNotFoundError' "$launch_log" 2>/dev/null; then
        echo "supervisor failed during detached startup" >&2
        tail -n 100 "$launch_log" >&2 || true
        exit 1
      fi
      sleep 1
    done
    echo "timed out waiting for identity-safe detached supervisor startup" >&2
    tail -n 100 "$launch_log" >&2 || true
    exit 1
    ;;
  stop)
    cleanup_disposition=$(
      "$PYTHON" "$LIFECYCLE_TOOL" cleanup-disposition \
        --phase-root "$PHASE_ROOT" --phase "$PHASE" --host-id "$HOST_ID" \
        --action-id "$LIFECYCLE_ACTION_ID" --sequence "$LIFECYCLE_SEQUENCE"
    )
    if [[ $("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "$cleanup_disposition") == "already_complete_or_superseded" ]]; then
      echo "cleanup already complete/superseded phase=$PHASE host=$HOST_ID action=$LIFECYCLE_ACTION_ID sequence=$LIFECYCLE_SEQUENCE"
      exit 0
    fi
    supervisor_pid=""
    if [[ -s "$PID_JSON" ]]; then
      supervisor_pid=$(validate_pid_identity 2>/dev/null || true)
    fi
    if [[ -n "$supervisor_pid" ]]; then
      kill -TERM "$supervisor_pid"
    fi
    for old_supervisor_pid in $(matching_runtime_pids supervisor); do
      kill -TERM "$old_supervisor_pid" 2>/dev/null || true
    done
    # Also handle the narrow case where the supervisor died before cleaning
    # its independently-sessioned guard.
    for guard_pid in $(matching_runtime_pids guard); do
      kill -TERM "$guard_pid" 2>/dev/null || true
    done
    for _ in $(seq 1 60); do
      [[ -z "$(matching_runtime_pids guard)" ]] && break
      sleep 1
    done
    # Exact worker generations from this matrix/host are first continued, so
    # no thermal STOP can survive the handoff, then terminated gracefully.
    for worker_pid in $(matching_runtime_pids worker); do
      kill -TERM -- "-$worker_pid" 2>/dev/null || true
      kill -CONT -- "-$worker_pid" 2>/dev/null || true
    done
    for trainer_pgid in $(matching_runtime_pids trainer | sort -u); do
      kill -TERM -- "-$trainer_pgid" 2>/dev/null || true
      kill -CONT -- "-$trainer_pgid" 2>/dev/null || true
    done
    for _ in $(seq 1 120); do
      supervisor_alive=""
      if [[ -s "$PID_JSON" ]]; then
        supervisor_alive=$(validate_pid_identity 2>/dev/null || true)
      fi
      if [[ -z "$supervisor_alive" && -z "$(matching_runtime_pids supervisor)" && -z "$(matching_runtime_pids guard)" && -z "$(matching_runtime_pids worker)" && -z "$(matching_runtime_pids trainer)" ]]; then
        "$PYTHON" "$LIFECYCLE_TOOL" complete-cleanup \
          --phase-root "$PHASE_ROOT" --phase "$PHASE" --host-id "$HOST_ID" \
          --action-id "$LIFECYCLE_ACTION_ID" --sequence "$LIFECYCLE_SEQUENCE"
        echo "stopped phase=$PHASE host=$HOST_ID; no owned supervisor/guard/worker remains"
        exit 0
      fi
      sleep 1
    done
    echo "graceful stop timed out; escalating only exact matrix/host process groups" >&2
    for owned_pid in $(matching_runtime_pids guard) $(matching_runtime_pids worker) $(matching_runtime_pids trainer) $(matching_runtime_pids supervisor); do
      kill -CONT -- "-$owned_pid" 2>/dev/null || true
      kill -KILL -- "-$owned_pid" 2>/dev/null || true
    done
    for _ in $(seq 1 20); do
      if [[ -z "$(matching_runtime_pids supervisor)" && -z "$(matching_runtime_pids guard)" && -z "$(matching_runtime_pids worker)" && -z "$(matching_runtime_pids trainer)" ]]; then
        "$PYTHON" "$LIFECYCLE_TOOL" complete-cleanup \
          --phase-root "$PHASE_ROOT" --phase "$PHASE" --host-id "$HOST_ID" \
          --action-id "$LIFECYCLE_ACTION_ID" --sequence "$LIFECYCLE_SEQUENCE"
        echo "stopped phase=$PHASE host=$HOST_ID after identity-safe escalation"
        exit 0
      fi
      sleep 1
    done
    echo "identity-safe stop could not prove an empty runtime" >&2
    exit 1
    ;;
  *)
    echo "usage: $0 {start|status|preflight|stop} {c|d} {local|remote}" >&2
    exit 2
    ;;
esac
