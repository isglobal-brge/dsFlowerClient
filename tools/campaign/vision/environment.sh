#!/bin/bash
# Source after provisioning, before invoking the public campaign drivers.
export R_LIBS_USER=/workspace/cells-vision/Rlib
export DSFLOWER_VENV_ROOT=/workspace/cells-vision/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/cells-vision/client
export TORCH_HOME=/workspace/cells-vision/torch
export XDG_CACHE_HOME=/workspace/cells-vision
export PYTHONPATH=/workspace/cells-vision/src/dsFlowerClient/inst/flower_app
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
