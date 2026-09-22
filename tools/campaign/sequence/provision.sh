#!/bin/bash
set -euo pipefail
ROOT=/workspace/cells-sequence
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$ROOT"/{data_cache,logs}
# Local POSIX storage avoids the release's short process-readiness deadline
# being consumed by Python imports from the RunPod network volume.
mkdir -p /opt/cells-sequence/{Rlib,venvs,client}
for part in Rlib venvs client; do
    if [ ! -e "$ROOT/$part" ]; then ln -s "/opt/cells-sequence/$part" "$ROOT/$part"; fi
done
curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc -o /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
printf '%s\n' 'deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/' > /etc/apt/sources.list.d/cran_r.list
curl -fsSL https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc -o /etc/apt/trusted.gpg.d/cranapt_key.asc
printf '%s\n' 'deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu jammy main' > /etc/apt/sources.list.d/cranapt.list
printf '%s\n' 'Package: *' 'Pin: release o=CRAN-Apt Project' 'Pin-Priority: 700' > /etc/apt/preferences.d/99cranapt
apt-get update -qq
apt-get install -y --no-install-recommends r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev cmake pkgconf r-cran-arrow r-cran-digest r-cran-filelock r-cran-jsonlite r-cran-processx r-cran-ps r-cran-r6 r-cran-httr r-cran-openssl r-cran-remotes r-cran-dsi r-cran-dslite
python -m pip install uv
export R_LIBS_USER="$(realpath "$ROOT/Rlib")"
export DSFLOWER_VENV_ROOT="$(realpath "$ROOT/venvs")"
export DSFLOWER_CLIENT_VENV_ROOT="$(realpath "$ROOT/client")"
export DSFLOWER_TORCH_BACKEND=cu124
Rscript -e 'stopifnot(getRversion() >= "4.4"); library(DSI); library(DSLite); library(arrow)'
R CMD INSTALL -l "$ROOT/Rlib" "$ROOT/src/dsFlower"
R CMD INSTALL -l "$ROOT/Rlib" "$ROOT/src/dsFlowerClient"
uv pip freeze --python "$ROOT/venvs/pytorch-gpu/bin/python" > "$ROOT/python-runtime-requirements.txt"
uv pip install --python "$ROOT/client/venv/bin/python" --torch-backend cu124 -r "$ROOT/python-runtime-requirements.txt"
"$ROOT/venvs/pytorch/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
Rscript -e 'library(dsFlower); library(dsFlowerClient); stopifnot(packageVersion("dsFlower") == "0.5.1", packageVersion("dsFlowerClient") == "0.5.0"); a <- dsFlowerClient:::.compute_local_runner_hash(); b <- dsFlower:::.compute_harness_hash(); stopifnot(identical(a,b)); cat("RUNNER_SHA256",a,"\n")'
printf '%s\n' R_STACK_DONE
