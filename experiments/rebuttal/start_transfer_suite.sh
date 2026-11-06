#!/usr/bin/env bash
set -euo pipefail
if [[ "${MOLTOKEN_REQUIRE_WORKER:-0}" == 1 && "$(hostname)" != *worker* ]]; then
    echo "Compute worker required" >&2; exit 2
fi
PIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOL=${MOL:?Set MOL to the local molscaling asset directory}
RUN=${RUN:-$MOL/rebuttal_experiments/e2e3_transfer_20261017_v1}
OLD_RUN=${OLD_RUN:-$MOL/rebuttal_experiments/e2e3_20261014_v1}


export PYTHONPATH="$OLD_RUN/runtime:$PIPE"
export PYTHONDONTWRITEBYTECODE=1 HOLOMOL_DATA_ROOT="$MOL/data"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="$RUN/runtime/tmp" XDG_CACHE_HOME="$RUN/runtime/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" TORCH_HOME="$XDG_CACHE_HOME/torch"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# No installation or network access on the GPU worker; reuse its known environment.
command -v "${PYTHON:-python}" >/dev/null
"${PYTHON:-python}" -u -c '
from pipeline.runtime_host import require_compute_host
require_compute_host("transfer bootstrap")
import socket, torch, transformers, xformers
assert torch.cuda.device_count() == 1, "Expected one scheduler-assigned GPU"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
print("TRANSFER_GPU_PREFLIGHT", socket.gethostname(), torch.__version__, transformers.__version__, xformers.__version__, torch.cuda.get_device_name(0), flush=True)
'
date -u '+%F %T UTC'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
"${PYTHON:-python}" -u "$PIPE/experiments/rebuttal/transfer_probe_worker.py" --run "$RUN" > "$RUN/logs/gpu_profile.log" 2>&1
exec "${PYTHON:-python}" -u "$PIPE/experiments/rebuttal/transfer_suite_worker.py" --run "$RUN" --old-run "$OLD_RUN"
