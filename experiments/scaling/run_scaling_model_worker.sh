#!/usr/bin/env bash
set -euo pipefail

# 在一个 GPU worker 上顺序跑某个模型规模的 scaling-law 实验。
# 单个模型只使用当前 worker 内部的 GPU，不做跨节点并行。

PROJECT_ROOT=${HOLOMOL_PIPELINE_ROOT:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
WORKSPACE_ROOT=${HOLOMOL_WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../../.." && pwd)}
DATA_ROOT=${HOLOMOL_SCALING_DATA_ROOT:-${DATA_ROOT:-$WORKSPACE_ROOT/scaling_data}}
RUN_ROOT=${HOLOMOL_SCALING_RUN_ROOT:-${RUN_ROOT:-$WORKSPACE_ROOT/local_runtime/runs/scaling}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scaling_fddfs_mf80000_v1}
MATRIX_JSON=${MATRIX_JSON:-$RUN_ROOT/$EXPERIMENT_NAME/scaling_matrix.json}
MODEL_ID=${MODEL_ID:?MODEL_ID is required, for example 0p06b}
CONDA_ENV_PATH=${HOLOMOL_SCALING_ENV_ROOT:-${CONDA_ENV_PATH:-$WORKSPACE_ROOT/local_runtime/envs/holomol-py311-cu121}}
SCALING_TMP_ROOT=${SCALING_TMP_ROOT:-/dev/shm/holomol-${UID}/scaling}
SCALING_LOG_ROOT=${SCALING_LOG_ROOT:-$WORKSPACE_ROOT/local_runtime/logs/scaling}
ALLOW_LEGACY_SCALING_PATHS=${ALLOW_LEGACY_SCALING_PATHS:-0}
ALLOW_OUT_DIR_OUTSIDE_RUN_ROOT=${ALLOW_OUT_DIR_OUTSIDE_RUN_ROOT:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-0.0002}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
LOG_EVERY=${LOG_EVERY:-20}
EVAL_EVERY=${EVAL_EVERY:-0}
EVAL_BATCHES=${EVAL_BATCHES:-20}
SAVE_EVERY=${SAVE_EVERY:-0}
OPTIMIZER=${OPTIMIZER:-adamw-fused}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
TORCH_COMPILE=${TORCH_COMPILE:-0}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29610}
LOG_DOMAIN_LOSS=${LOG_DOMAIN_LOSS:-0}
COMPACT_DIR=${COMPACT_DIR:-$DATA_ROOT/fddfs_bpe_full_prefix70_64k_mf80000_v1/encoded_compact_int_prefix70_mf80000_11w_v3}
COMPACT_INDEX_DIR=${COMPACT_INDEX_DIR:-$DATA_ROOT/fddfs_bpe_full_prefix70_64k_mf80000_v1/encoded_compact_int_prefix70_mf80000_11w_v3_index}
COMPACT_DOMAIN_ID_DIR=${COMPACT_DOMAIN_ID_DIR:-$DATA_ROOT/fddfs_bpe_full_prefix70_64k_mf80000_v1/encoded_compact_int_prefix70_mf80000_11w_v3_domain_ids}
BPE_MERGES_CSV=${BPE_MERGES_CSV:-$DATA_ROOT/fddfs_bpe_full_prefix70_64k_mf80000_v1/codebook_fddfs_prefix70_mf80000_v1/merges.csv}
DOMAIN_NAMES=${DOMAIN_NAMES:-small_molecule,peptide,protein,dna,rna}

require_safe_component() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "$name must be one directory name using letters, digits, '.', '_' or '-': $value" >&2
    exit 2
  fi
}

require_safe_component EXPERIMENT_NAME "$EXPERIMENT_NAME"
require_safe_component MODEL_ID "$MODEL_ID"

if [ "$ALLOW_LEGACY_SCALING_PATHS" != "1" ]; then
  case "$PROJECT_ROOT|$DATA_ROOT|$RUN_ROOT|$CONDA_ENV_PATH|$MATRIX_JSON|$COMPACT_DIR|$COMPACT_INDEX_DIR|$COMPACT_DOMAIN_ID_DIR|$BPE_MERGES_CSV|$SCALING_TMP_ROOT|$SCALING_LOG_ROOT" in
    *"./scaling-csy"*|*"/data/envs/scaling-csy"*)
      echo "refusing legacy cluster scaling path; set ALLOW_LEGACY_SCALING_PATHS=1 only for historical replay" >&2
      exit 2
      ;;
  esac
fi

mkdir -p "$SCALING_TMP_ROOT" "$SCALING_LOG_ROOT" "$RUN_ROOT/$EXPERIMENT_NAME/$MODEL_ID"
export TMPDIR="$SCALING_TMP_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

if [ ! -s "$MATRIX_JSON" ]; then
  echo "missing scaling matrix: $MATRIX_JSON" >&2
  exit 2
fi

ENV_DIR="$SCALING_TMP_ROOT/${EXPERIMENT_NAME}_${MODEL_ID}_envs_$(hostname)_$$"
rm -rf "$ENV_DIR"
mkdir -p "$ENV_DIR"

"${CONDA_ENV_PATH}/bin/python" - "$MATRIX_JSON" "$MODEL_ID" "$ENV_DIR" "$RUN_ROOT" "$ALLOW_OUT_DIR_OUTSIDE_RUN_ROOT" <<'PY'
import json
import shlex
import sys
from pathlib import Path

matrix = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_id = sys.argv[2]
env_dir = Path(sys.argv[3])
run_root = Path(sys.argv[4]).expanduser().resolve()
allow_external = sys.argv[5] == "1"
rows = [r for r in matrix["rows"] if r["model_id"] == model_id]
if not rows:
    raise SystemExit(f"no rows for model_id={model_id}")
rows.sort(key=lambda r: float(r["ratio"]))
for i, row in enumerate(rows):
    raw_out_dir = Path(row["out_dir"]).expanduser()
    if not raw_out_dir.is_absolute():
        raise SystemExit(f"matrix out_dir must be absolute: {raw_out_dir}")
    out_dir = raw_out_dir.resolve()
    if not allow_external and not out_dir.is_relative_to(run_root):
        raise SystemExit(
            f"matrix out_dir escapes RUN_ROOT: out_dir={out_dir} run_root={run_root}"
        )
    env_path = env_dir / f"{i:02d}_{row['model_id']}_r{row['ratio']:g}.env"
    values = {
        "ROW_INDEX": i,
        "RUN_NAME": row["run_name"],
        "OUT_DIR": str(out_dir),
        "LOG_PATH": str(out_dir / "console.log"),
        "NPROC_PER_NODE": row["gpu_count"],
        "BATCH_SIZE": row["per_device_train_batch_size"],
        "GRAD_ACCUM": row["gradient_accumulation_steps"],
        "MAX_STEPS": row["max_steps"],
        "SEED": row["seed"],
        "VOCAB_SIZE": row["vocab_size"],
        "TOKEN_OFFSET": row["token_offset"],
        "ONLINE_PAD_LENGTH": row["online_pad_length"],
        "VAL_FRACTION": row["val_fraction"],
        "HIDDEN_SIZE": row["hidden_size"],
        "INTERMEDIATE_SIZE": row["intermediate_size"],
        "NUM_HIDDEN_LAYERS": row["num_hidden_layers"],
        "NUM_ATTENTION_HEADS": row["num_attention_heads"],
        "NUM_KEY_VALUE_HEADS": row["num_key_value_heads"],
        "HEAD_DIM": row["head_dim"],
        "SCALING_RATIO": row["ratio"],
        "SCALING_PARAMETER_COUNT": row["parameter_count"],
        "SCALING_TARGET_TOKENS": row["target_tokens"],
        "SCALING_ACTUAL_TOKENS": row["actual_tokens"],
        "SCALING_FLOPS_6ND": row["flops_6nd"],
    }
    with env_path.open("w", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={shlex.quote(str(value))}\n")
print(len(rows))
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; run this script inside a GPU worker" >&2
  exit 2
fi

STATUS_JSONL="$RUN_ROOT/$EXPERIMENT_NAME/$MODEL_ID/worker_status.jsonl"
echo "{\"event\":\"worker_start\",\"model_id\":\"$MODEL_ID\",\"time\":\"$(date -Is)\",\"matrix\":\"$MATRIX_JSON\"}" | tee -a "$STATUS_JSONL"

for env_file in "$ENV_DIR"/*.env; do
  # shellcheck disable=SC1090
  source "$env_file"
  mkdir -p "$OUT_DIR"
  MASTER_PORT=$((MASTER_PORT_BASE + ROW_INDEX))

  if "${CONDA_ENV_PATH}/bin/python" - "$OUT_DIR/summary.json" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
need = int(sys.argv[2])
if not p.exists():
    raise SystemExit(1)
d = json.loads(p.read_text(encoding="utf-8"))
raise SystemExit(0 if int(d.get("global_step", -1)) >= need else 1)
PY
  then
    echo "{\"event\":\"skip_completed\",\"run_name\":\"$RUN_NAME\",\"time\":\"$(date -Is)\"}" | tee -a "$STATUS_JSONL"
    continue
  fi

  cat > "$OUT_DIR/scaling_config.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "model_id": "$MODEL_ID",
  "ratio": $SCALING_RATIO,
  "parameter_count": $SCALING_PARAMETER_COUNT,
  "target_tokens": $SCALING_TARGET_TOKENS,
  "actual_tokens": $SCALING_ACTUAL_TOKENS,
  "flops_6nd": $SCALING_FLOPS_6ND,
  "matrix_json": "$MATRIX_JSON"
}
EOF

  echo "{\"event\":\"run_start\",\"run_name\":\"$RUN_NAME\",\"ratio\":$SCALING_RATIO,\"time\":\"$(date -Is)\",\"out_dir\":\"$OUT_DIR\"}" | tee -a "$STATUS_JSONL"
  LOG_PATH="$LOG_PATH" \
  OUT_DIR="$OUT_DIR" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  BATCH_SIZE="$BATCH_SIZE" \
  GRAD_ACCUM="$GRAD_ACCUM" \
  MAX_STEPS="$MAX_STEPS" \
  SEED="$SEED" \
  VOCAB_SIZE="$VOCAB_SIZE" \
  TOKEN_OFFSET="$TOKEN_OFFSET" \
  ONLINE_PAD_LENGTH="$ONLINE_PAD_LENGTH" \
  VAL_FRACTION="$VAL_FRACTION" \
  HIDDEN_SIZE="$HIDDEN_SIZE" \
  INTERMEDIATE_SIZE="$INTERMEDIATE_SIZE" \
  NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
  NUM_ATTENTION_HEADS="$NUM_ATTENTION_HEADS" \
  NUM_KEY_VALUE_HEADS="$NUM_KEY_VALUE_HEADS" \
  HEAD_DIM="$HEAD_DIM" \
  LR="$LR" \
  MIN_LR_RATIO="$MIN_LR_RATIO" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  LOG_EVERY="$LOG_EVERY" \
  EVAL_EVERY="$EVAL_EVERY" \
  EVAL_BATCHES="$EVAL_BATCHES" \
  SAVE_EVERY="$SAVE_EVERY" \
  NUM_WORKERS="$NUM_WORKERS" \
  MASTER_PORT="$MASTER_PORT" \
  OPTIMIZER="$OPTIMIZER" \
  GRADIENT_CHECKPOINTING="$GRADIENT_CHECKPOINTING" \
  TORCH_COMPILE="$TORCH_COMPILE" \
  LOG_DOMAIN_LOSS="$LOG_DOMAIN_LOSS" \
  DATA_ROOT="$DATA_ROOT" \
  RUN_ROOT="$RUN_ROOT" \
  CONDA_ENV_PATH="$CONDA_ENV_PATH" \
  SCALING_TMP_ROOT="$SCALING_TMP_ROOT" \
  SCALING_LOG_ROOT="$SCALING_LOG_ROOT" \
  ALLOW_LEGACY_SCALING_PATHS="$ALLOW_LEGACY_SCALING_PATHS" \
  ALLOW_OUT_DIR_OUTSIDE_RUN_ROOT="$ALLOW_OUT_DIR_OUTSIDE_RUN_ROOT" \
  COMPACT_DIR="$COMPACT_DIR" \
  COMPACT_INDEX_DIR="$COMPACT_INDEX_DIR" \
  COMPACT_DOMAIN_ID_DIR="$COMPACT_DOMAIN_ID_DIR" \
  BPE_MERGES_CSV="$BPE_MERGES_CSV" \
  DOMAIN_NAMES="$DOMAIN_NAMES" \
  bash ops/cluster/run_qwen3_fddfs_onlinepad512_pretrain_worker.sh
  echo "{\"event\":\"run_done\",\"run_name\":\"$RUN_NAME\",\"ratio\":$SCALING_RATIO,\"time\":\"$(date -Is)\",\"out_dir\":\"$OUT_DIR\"}" | tee -a "$STATUS_JSONL"
done

echo "{\"event\":\"worker_done\",\"model_id\":\"$MODEL_ID\",\"time\":\"$(date -Is)\"}" | tee -a "$STATUS_JSONL"
