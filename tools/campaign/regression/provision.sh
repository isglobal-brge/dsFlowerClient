#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/cells
mkdir -p "$ROOT"/{Rlib,logs,data_cache,smoke}
date -u +%FT%TZ > "$ROOT/logs/provision-start.txt"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg build-essential gfortran pkg-config git rsync unzip python3-pip libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev libfontconfig1-dev libfreetype6-dev libpng-dev libjpeg-dev libtiff5-dev libharfbuzz-dev libfribidi-dev libglpk-dev libgmp-dev cmake
curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc -o /etc/apt/trusted.gpg.d/cran.asc
echo 'deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/' > /etc/apt/sources.list.d/cran-r.list
apt-get update
apt-get install -y --no-install-recommends r-base r-base-dev r-cran-jsonlite r-cran-digest r-cran-processx r-cran-ps r-cran-filelock r-cran-r6 r-cran-remotes r-cran-httr r-cran-bit64 r-cran-purrr r-cran-tidyselect r-cran-vctrs r-cran-cpp11
python3 -m pip install uv
export R_LIBS="$ROOT/Rlib"
export LIBARROW_BINARY=true
export NOT_CRAN=true
Rscript "$ROOT/dsFlowerClient/tools/campaign/regression/install_dependencies.R"
export DSFLOWER_VENV_ROOT="$ROOT/venvs"
export DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_TORCH_BACKEND=cpu
R CMD INSTALL --library="$ROOT/Rlib" "$ROOT/dsFlower"
R CMD INSTALL --library="$ROOT/Rlib" "$ROOT/dsFlowerClient"
uv pip install --python "$ROOT/client/venv/bin/python" --torch-backend cpu \
  'torch>=2.0.0,<3.0.0' 'numpy>=1.21.0' 'pandas>=1.3.0' \
  'pyarrow>=10.0.0' 'opacus>=1.4.0,<2.0.0' 'cryptography>=42.0.0'
date -u +%FT%TZ > "$ROOT/logs/provision-end.txt"
Rscript -e 'stopifnot(getRversion()>="4.4", packageVersion("dsFlower")=="0.5.0", packageVersion("dsFlowerClient")=="0.5.0"); cat(R.version.string,"\n"); cat(dsFlowerClient:::.compute_local_runner_hash(),"\n")'
