#!/bin/sh
set -eu
umask 077
cd /workspace/survival
export R_LIBS_USER=/workspace/survival/runtime/rlib
export TMPDIR=/workspace/survival/runtime/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_root=$(readlink -f runtime/venv)
exec "$python_root/bin/python" -u dsFlowerClient/tools/campaign/survival/run_hazard_v2.py /workspace/survival
