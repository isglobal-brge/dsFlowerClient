#!/bin/sh
# Public pod2 campaign only; fresh v3 roots must be prepared before invocation.
set -eu
export F_SEG_RUNNER_PARENT=/workspace/segmentation/runtime
export PYTHONPATH=/workspace/segmentation/runtime
export F_SEG_GATES_JSON=/workspace/segmentation/mechanism-gates.json
export DSFLOWER_VENV_ROOT=/workspace/segmentation/server-venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/segmentation/client-venv
export R_LIBS_USER=/workspace/segmentation/rlib
export TORCH_HOME=/workspace/segmentation/torch
export XDG_CACHE_HOME=/workspace/segmentation
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PATH=/workspace/segmentation/venv/bin:$PATH
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=/workspace/segmentation/v3
[ -f "$root/launch-manifest.json" ]
[ ! -e "$root/runs-batch16" ]
[ ! -e "$root/runs-batch64" ]
/workspace/segmentation/venv/bin/python "$script_dir/run_matrix.py" --root "$root" --workers 1 --batch-size 16 > /workspace/logs/segmentation-v3-batch16-driver.log 2>&1 &
p16=$!
/workspace/segmentation/venv/bin/python "$script_dir/run_matrix.py" --root "$root" --workers 1 --batch-size 64 > /workspace/logs/segmentation-v3-batch64-driver.log 2>&1 &
p64=$!
printf 'batch16_pid=%s batch64_pid=%s\n' "$p16" "$p64"
status=0
wait "$p16" || status=1
wait "$p64" || status=1
exit "$status"
