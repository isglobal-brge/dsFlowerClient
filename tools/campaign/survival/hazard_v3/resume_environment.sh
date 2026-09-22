#!/bin/sh
# Resume only the dedicated, already provisioned hazard workspace.
set -eu
cd /workspace/hazard
export R_LIBS_USER=/workspace/hazard/runtime/rlib
export PATH=/workspace/hazard/runtime/venv/bin:/root/.local/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runtime/rlib logs
Rscript - <<'R'
.libPaths(c(Sys.getenv('R_LIBS_USER'),.libPaths()))
options(repos=c(CRAN='https://cloud.r-project.org'),Ncpus=8)
if (!requireNamespace('dsBase',quietly=TRUE)) {
  install.packages(c('Matrix','lme4','jomo','mitml','mice'),lib=.libPaths()[1])
  remotes::install_github('datashield/dsBase@6.3.5',lib=.libPaths()[1],upgrade='never',dependencies=NA)
}
stopifnot(requireNamespace('dsBase',quietly=TRUE))
R
sh dsFlowerClient/tools/campaign/survival/install_and_freeze.sh
