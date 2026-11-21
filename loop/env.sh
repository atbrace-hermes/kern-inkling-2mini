# Sourced by every loop/ script.
# MAIN is always the primary repo clone (shared across all git worktrees via --git-common-dir);
# TREE is the current checkout (main or a git worktree).
set -a
_THIS_LOOP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_COMMON_GIT=$(git -C "$_THIS_LOOP" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [[ -n "$_COMMON_GIT" ]]; then
  MAIN=$(cd "$_COMMON_GIT/.." && pwd)
else
  MAIN=$(cd "$_THIS_LOOP/.." && pwd)
fi
LOOP=$MAIN/loop
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
unset _THIS_LOOP _COMMON_GIT
set +a
mkdir -p "$STATE"
