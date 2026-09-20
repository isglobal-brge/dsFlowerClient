#!/bin/sh
# One detached supervisor; all cohort data are public fixtures.
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
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PATH=/workspace/segmentation/venv/bin:$PATH
unset F_SEG_SYNTHETIC F_SEG_INNER_SPLIT F_SEG_V4_CONFIG
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /workspace/segmentation/venv/bin/python "$script_dir/run_v4.py" --root /workspace/segmentation/v4
