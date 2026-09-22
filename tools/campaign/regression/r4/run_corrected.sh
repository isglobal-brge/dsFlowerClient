#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/cells
export R_LIBS="$ROOT/Rlib" DSFLOWER_VENV_ROOT="$ROOT/venvs" DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_NODE_SECRET_FILE="$ROOT/smoke/parent-node-secret"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S="$ROOT/r4-client/tools/campaign/regression/r4"
for epsilon in 1 4 8; do
  Rscript "$S/run_corrected.R" "$ROOT" "$epsilon" "$ROOT/r4/corrected_evidence"
done
