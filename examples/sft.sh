#!/usr/bin/env bash
set -euo pipefail
: "${DATA:?Indexed SFT root}"
: "${BASE:?Base checkpoint directory}"
: "${ENCODER:?Local frozen text encoder directory}"
: "${TASK:?Record task ID}"
: "${OUT:?Run output directory}"
common=(--data-root "$DATA" --pretrained-ckpt "$BASE" --text-encoder "$ENCODER"
  --task "$TASK" --lineage-id "$TASK" --sampling-protocol single-task-repeated-stage-v1
  --epochs 100 --steps-per-epoch 0 --max-steps 0 --eval-batches 0 --save-steps 0
  --max-context 512 --molecule-condition-mode direct_qwen --strict-no-runtime-truncation
  --batch-size "${BATCH:-8}" --protein-batch-size "${BATCH:-8}"
  --grad-accumulation-steps "${ACCUM:-1}" --early-stopping-patience "${PATIENCE:-10}")
python -m cli.sft.train_axis_b_repeated_stage_sft "${common[@]}" \
  --stage alignment --output-dir "$OUT/alignment" "$@"
python -m cli.sft.train_axis_b_repeated_stage_sft "${common[@]}" \
  --stage full --resume-dir "$OUT/alignment/best" --output-dir "$OUT/full" "$@"
