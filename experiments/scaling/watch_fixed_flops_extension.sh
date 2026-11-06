#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_LEGACY_RLAUNCH:-0}" != "1" ]; then
  echo "legacy GPFS watcher is disabled on the physical host; set ALLOW_LEGACY_RLAUNCH=1 only for historical cluster replay" >&2
  exit 2
fi

# Lightweight jump-host watcher.  It never controls workers; after all rows
# finish, it validates artifacts and renders the combined ten-budget figures.

PROJECT_ROOT=${PROJECT_ROOT:-.}
RUN_ROOT=${RUN_ROOT:-./storage/molscaling/runs}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1}
EXPERIMENT_ROOT="$RUN_ROOT/$EXPERIMENT_NAME"
MATRIX_JSON="$EXPERIMENT_ROOT/scaling_matrix.json"
POLL_SECONDS=${POLL_SECONDS:-600}

OLD_MATRIX_3P5E17="$RUN_ROOT/scaling_fddfs_mf80000_fixed_flops3p5e17_smallmodels_domainlog_v1/scaling_matrix.json"
OLD_MATRIX_SWEEP="$RUN_ROOT/scaling_fddfs_mf80000_fixed_flops_sweep_overepoch_domainlog_v2/scaling_matrix.json"
OLD_MATRIX_3P5E18="$RUN_ROOT/scaling_fddfs_mf80000_fixed_flops3p5e18_domainlog_v1/scaling_matrix.json"
PLOT_DIR="$EXPERIMENT_ROOT/plots/all_fixed_flops_10curves"

while true; do
  status=$(python3 - "$MATRIX_JSON" <<'PY'
import json
import math
import time
import sys
from pathlib import Path

matrix = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
domain_fields = {
    "domain_samples", "domain_sample_fraction", "domain_input_tokens",
    "domain_input_token_fraction", "domain_loss_tokens", "domain_loss_token_fraction",
    "domain_loss_sum", "domain_loss_contribution", "domain_loss",
}
complete = 0
running = 0
pending = 0
stale = []
invalid = []
domain_invalid = []
active_steps = {}
completed_optimizer_steps = 0
total_optimizer_steps = sum(int(row["max_steps"]) for row in matrix["rows"])
completed_target_flops = 0.0
total_target_flops = sum(float(row["target_flops_6nd"]) for row in matrix["rows"])
now = time.time()
for row in matrix["rows"]:
    out = Path(row["out_dir"])
    key = f"{row['model_id']}/{row['flops_target_label']}"
    max_steps = int(row["max_steps"])
    target_flops = float(row["target_flops_6nd"])
    summary_path = out / "summary.json"
    final_path = out / "final" / "pytorch_model.bin"
    train_log = out / "train_log.jsonl"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            global_step = int(summary.get("global_step", -1))
            loss = summary.get("final_logged_train_loss")
            done = global_step >= max_steps and isinstance(loss, (int, float)) and math.isfinite(float(loss))
        except Exception:
            global_step = 0
            done = False
        if done and final_path.exists():
            complete += 1
            completed_optimizer_steps += max_steps
            completed_target_flops += target_flops
        else:
            invalid.append(key)
    elif train_log.exists():
        running += 1
        last = None
        with train_log.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "train":
                    last = event
        if last is not None:
            step = min(int(last.get("step", 0)), max_steps)
            completed_optimizer_steps += step
            completed_target_flops += target_flops * step / max_steps
            active_steps[key] = f"{step}/{max_steps}"
            loss = last.get("loss")
            missing = sorted(domain_fields - set(last))
            if missing or not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
                domain_invalid.append({"run": key, "missing": missing, "loss": loss})
        else:
            domain_invalid.append({"run": key, "missing": ["train_event"], "loss": None})
        if now - train_log.stat().st_mtime > 1800:
            stale.append(key)
    else:
        pending += 1
print(json.dumps({
    "complete": complete,
    "running": running,
    "pending": pending,
    "stale": stale,
    "invalid": invalid,
    "domain_invalid": domain_invalid,
    "optimizer_step_progress": completed_optimizer_steps / total_optimizer_steps,
    "target_flops_work_progress": completed_target_flops / total_target_flops,
    "active_steps": active_steps,
    "total": len(matrix["rows"]),
}, separators=(",", ":")))
PY
)
  printf '%s %s\n' "$(date -Is)" "$status"

  complete=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["complete"])' "$status")
  total=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["total"])' "$status")
  if [ "$complete" -eq "$total" ]; then
    python3 "$PROJECT_ROOT/experiments/scaling/verify_fixed_flops_extension.py" \
      --matrix-json "$MATRIX_JSON" \
      --expected-rows 39 \
      --output-json "$EXPERIMENT_ROOT/completion_verification.json"
    python3 "$PROJECT_ROOT/experiments/scaling/plot_all_fixed_flops_results.py" \
      --matrix-json "$OLD_MATRIX_3P5E17" \
      --matrix-json "$OLD_MATRIX_SWEEP" \
      --matrix-json "$OLD_MATRIX_3P5E18" \
      --matrix-json "$MATRIX_JSON" \
      --output-dir "$PLOT_DIR"
    python3 "$PROJECT_ROOT/experiments/scaling/plot_optimal_scaling_fits.py" \
      --matrix-json "$OLD_MATRIX_3P5E17" \
      --matrix-json "$OLD_MATRIX_SWEEP" \
      --matrix-json "$OLD_MATRIX_3P5E18" \
      --matrix-json "$MATRIX_JSON" \
      --require-budget-count 10 \
      --output-dir "$PLOT_DIR"
    python3 "$PROJECT_ROOT/experiments/scaling/audit_optimal_scaling_goal.py" \
      --experiment-root "$EXPERIMENT_ROOT" \
      --plot-dir "$PLOT_DIR"
    echo "$(date -Is) completed and plotted: $PLOT_DIR"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
