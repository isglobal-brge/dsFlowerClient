#!/bin/bash
set -euo pipefail
# Fresh provisioned pod only. Diagnostic helpers refuse existing preparation/run.
ROOT=/workspace/cells-sequence
TOOLS="$ROOT/src/dsFlowerClient/tools/campaign/sequence"
PYTHON=/opt/cells-sequence/venvs/pytorch-gpu/bin/python
export R_LIBS_USER=/opt/cells-sequence/Rlib
export DSFLOWER_VENV_ROOT=/opt/cells-sequence/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/opt/cells-sequence/client
export PYTHONPATH=/opt/cells-sequence/Rlib/dsFlowerClient/flower_app
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$ROOT/r4/logs"
mkdir -m 700 -p /tmp/dsflower-sequence-r4-parent
export DSFLOWER_NODE_SECRET_FILE=/tmp/dsflower-sequence-r4-parent/secret
export DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET=1
"$PYTHON" "$TOOLS/install_public_observer.py" > "$ROOT/r4/logs/observer-install.json"
"$PYTHON" "$TOOLS/r4/prepare.py" --root "$ROOT" > "$ROOT/r4/logs/prepare.log"
timeout 1800 Rscript "$TOOLS/r4/run_federated.R" "$ROOT" > "$ROOT/r4/logs/real-contract.log" 2>&1
timeout 3600 "$PYTHON" -u "$TOOLS/r4/emulate.py" --root "$ROOT" > "$ROOT/r4/logs/emulation.log" 2>&1
"$PYTHON" "$TOOLS/r4/verify.py" --root "$ROOT" > "$ROOT/r4/logs/verification.log" 2>&1
"$PYTHON" "$TOOLS/r4/accountant_table.py" --root "$ROOT" > "$ROOT/r4/logs/accountant-table.log" 2>&1
