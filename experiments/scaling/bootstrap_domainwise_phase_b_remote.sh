#!/usr/bin/env bash
set -euo pipefail

# Run on gamma-4090 after the host-local tmpfs stage has been started.  This
# process is designed to live under nohup+setsid: it waits for the stage,
# exercises the exact first four formal workloads, then starts the detached
# remote Phase-B supervisor.

WORKSPACE_ROOT=/path/to/moltoken-workspace
PIPELINE_ROOT=$WORKSPACE_ROOT/HoloMol/code/pipeline
SCRIPT_DIR=$PIPELINE_ROOT/experiments/scaling
PYTHON=$WORKSPACE_ROOT/local_runtime/envs/holomol-py311-cu121/bin/python
CAMPAIGN_ROOT=$WORKSPACE_ROOT/local_runtime/runs/scaling/domainwise_scaling_fddfs_mf80000_v1
MATRIX_JSON=$CAMPAIGN_ROOT/phase_b_matrix_remote.json
STAGE_ROOT=/dev/shm/holomol-1046/scaling
STAGE_PID_FILE=$STAGE_ROOT/domainwise_stage.pid
STAGED_MANIFEST=$STAGE_ROOT/domainwise_dataset_v1/domain_split_manifests_v1/manifest.json
READY_JSON=$STAGE_ROOT/domainwise_dataset_v1/READY.json
BOOTSTRAP_READY=$STAGE_ROOT/remote_phase_b_bootstrap.READY

echo "$(date -Is) bootstrap_start host=$(hostname) pid=$$"
while [[ ! -s "$READY_JSON" ]]; do
  if [[ -s "$STAGE_PID_FILE" ]]; then
    stage_pid=$(tr -d '[:space:]' < "$STAGE_PID_FILE")
    if [[ ! "$stage_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$stage_pid" 2>/dev/null; then
      echo "$(date -Is) ERROR stage exited before READY.json" >&2
      exit 10
    fi
  fi
  du -sh "$STAGE_ROOT/domainwise_dataset_v1" 2>/dev/null || true
  sleep 15
done

"$PYTHON" - "$READY_JSON" "$STAGED_MANIFEST" <<'PY'
import json
import pathlib
import sys

ready_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
ready = json.loads(ready_path.read_text(encoding="utf-8"))
if ready.get("status") != "ready":
    raise SystemExit(f"stage status is not ready: {ready}")
if pathlib.Path(ready.get("staged_manifest", "")) != manifest_path:
    raise SystemExit(f"staged manifest mismatch: {ready}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "complete":
    raise SystemExit("staged manifest is incomplete")
PY

echo "$(date -Is) stage_ready manifest=$STAGED_MANIFEST"

smoke_root=$CAMPAIGN_ROOT/smoke_remote_first_v1
if [[ -s "$smoke_root/report.json" ]]; then
  "$PYTHON" - "$smoke_root/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("failures"):
    raise SystemExit(f"existing remote smoke failed: {report['failures']}")
PY
  echo "$(date -Is) smoke_already_passed root=$smoke_root"
else
  if [[ -e "$smoke_root" ]]; then
    smoke_root=$CAMPAIGN_ROOT/smoke_remote_first_v1_retry_$(date +%Y%m%d_%H%M%S)
  fi
  "$PYTHON" -u "$SCRIPT_DIR/smoke_domainwise_phase_b_slots.py" \
    --matrix-json "$MATRIX_JSON" \
    --python "$PYTHON" \
    --output-root "$smoke_root" \
    --selection first
  echo "$(date -Is) smoke_passed root=$smoke_root"
fi

HOLOMOL_SCALING_DOMAIN_MANIFEST=$STAGED_MANIFEST \
  bash "$SCRIPT_DIR/launch_domainwise_phase_b_local.sh" start remote

# Cold imports from the shared environment can take around two minutes on this
# host.  Do not call the launch successful until all four formal workers have
# emitted a real training step.
for poll in $(seq 1 60); do
  supervisor_pid=""
  if [[ -s "$CAMPAIGN_ROOT/supervisor_remote.pid" ]]; then
    supervisor_pid=$(tr -d '[:space:]' < "$CAMPAIGN_ROOT/supervisor_remote.pid")
  fi
  if [[ -n "$supervisor_pid" ]] && ! kill -0 "$supervisor_pid" 2>/dev/null; then
    echo "$(date -Is) ERROR remote supervisor exited during startup" >&2
    tail -n 100 "$CAMPAIGN_ROOT/supervisor_remote_console.log" >&2 || true
    exit 20
  fi
  trained=0
  for status in "$CAMPAIGN_ROOT"/workers/remote_gpu*_slot*/status.jsonl; do
    [[ -f "$status" ]] || continue
    attempt_dir=$("$PYTHON" - "$status" <<'PY'
import json
import sys

attempt = ""
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    if row.get("event") == "run_start":
        attempt = row["attempt_dir"]
print(attempt)
PY
)
    if [[ -n "$attempt_dir" && -s "$attempt_dir/train_log.jsonl" ]] && \
       grep -q '"event": "train"' "$attempt_dir/train_log.jsonl"; then
      trained=$((trained + 1))
    fi
  done
  echo "$(date -Is) formal_startup_poll=$poll trained_slots=$trained"
  if [[ "$trained" -eq 4 ]]; then
    printf 'status=ready\nhost=%s\nsupervisor_pid=%s\ntime=%s\n' \
      "$(hostname)" "$supervisor_pid" "$(date -Is)" > "$BOOTSTRAP_READY"
    echo "$(date -Is) bootstrap_ready supervisor_pid=$supervisor_pid"
    exit 0
  fi
  sleep 15
done

echo "$(date -Is) ERROR timed out waiting for four remote train steps" >&2
exit 30
