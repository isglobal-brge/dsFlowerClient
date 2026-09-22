#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/workspace/cells}
export R_LIBS="$ROOT/Rlib"
export DSFLOWER_VENV_ROOT="$ROOT/venvs"
export DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_NODE_SECRET_FILE="$ROOT/parent-node-secret"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT/dsFlowerClient"
Rscript tools/campaign/multiclass/check_cdcgenhlth.R
Rscript tools/campaign/multiclass/prepare_cdcgenhlth.R "$ROOT"
Rscript tools/campaign/multiclass/run_cdcgenhlth.R "$ROOT" inst/extdata/campaign/multiclass
