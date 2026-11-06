#!/usr/bin/env bash
set -euo pipefail
# Set DATA, OUT and batch/accumulation for the local machine. No scheduler calls.
: "${DATA:?Directory containing compact-int shards}"
: "${INDEX:?Compact-int index directory}"
: "${OUT:?Output checkpoint directory}"
torchrun --standalone --nproc_per_node="${GPUS:-1}" \
  -m cli.pretrain.train_qwen3_pretrain \
  --token-files "$DATA/shard_*.bin" --compact-index-dir "$INDEX" \
  --out-dir "$OUT" --vocab-size 17207 \
  --dataset-mode online-pad --online-pad-length 512 --online-truncate-long-samples \
  --model-implementation packed-xformers --hidden-size 1024 \
  --intermediate-size 3072 --num-hidden-layers 34 --num-attention-heads 16 \
  --num-key-value-heads 8 --head-dim 128 \
  --batch-size "${BATCH:-8}" --grad-accum "${ACCUM:-4}" \
  --max-steps "${STEPS:-1000}" --final-eval "$@"
