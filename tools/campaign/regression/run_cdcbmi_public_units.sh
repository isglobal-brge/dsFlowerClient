#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/workspace/cells}
export R_LIBS="$ROOT/Rlib"
export DSFLOWER_VENV_ROOT="$ROOT/venvs"
export DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_NODE_SECRET_FILE="$ROOT/smoke/parent-node-secret"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT/dsFlowerClient"
for epsilon in 1 4 8; do
  Rscript tools/campaign/regression/run_cdcbmi_public_units.R "$ROOT" "$epsilon" inst/extdata/campaign/regression
done
