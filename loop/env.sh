# Sourced by every loop/ script.
# MAIN is the primary repo clone; TREE is the current checkout (main or a git worktree).
set -a
LOOP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAIN=$(cd "$LOOP/.." && pwd)
TREE=$(git rev-parse --show-toplevel 2>/dev/null || echo "$MAIN")
STATE=${KERN_LOOP_STATE:-$MAIN/state}
GPU=${KERN_LOOP_GPU:-0}
PY=${PY:-/workspace/kern-qwen38/.venv/bin/python}
KERN=${KERN:-/workspace/kern/target/release/kern}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
WEIGHTS=${KERN_LOOP_WEIGHTS:-/workspace/weights/Inkling-2-Mini}
SMOKE_REF=${KERN_LOOP_SMOKE_REF:-/workspace/baselines/inkling-2-mini/smoke_reference.parquet}
STD_REF=${KERN_LOOP_STD_REF:-/workspace/baselines/inkling-2-mini/standard_reference.parquet}
KV_SCALES=${KERN_LOOP_KV_SCALES:-$MAIN/results/kv-scales.json}
set +a
mkdir -p "$STATE"
