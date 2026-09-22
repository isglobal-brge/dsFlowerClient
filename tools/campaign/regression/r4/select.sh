#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/cells
export R_LIBS="$ROOT/Rlib" DSFLOWER_VENV_ROOT="$ROOT/venvs" DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_NODE_SECRET_FILE="$ROOT/smoke/parent-node-secret"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S="$ROOT/r4-client/tools/campaign/regression/r4"
for seed in 20260820 20260821 20260822; do
  "$ROOT/venvs/pytorch/bin/python" "$S/stage_selection.py" --seed "$seed"
  for candidate in sgd_lr003_e5_b64 sgd_lr001_e5_b32; do
    Rscript "$S/select_federated.R" "$ROOT" "$seed" "$candidate"
  done
done
