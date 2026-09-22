#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/cells
mkdir -p "$ROOT"/{Rlib,venvs,client,data_cache,logs,smoke}
date -u +%FT%TZ > "$ROOT/logs/provision-start.txt"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl gnupg build-essential gfortran python3-pip rsync git libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev cmake
curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc -o /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
echo 'deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/' > /etc/apt/sources.list.d/cran_r.list
curl -fsSL https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc -o /etc/apt/trusted.gpg.d/cranapt_key.asc
echo 'deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu jammy main' > /etc/apt/sources.list.d/cranapt.list
printf 'Package: *\nPin: release o=CRAN-Apt Project\nPin-Priority: 700\n' > /etc/apt/preferences.d/99cranapt
apt-get update -qq
apt-get install -y --no-install-recommends r-base-core r-base-dev r-cran-arrow r-cran-digest r-cran-filelock r-cran-jsonlite r-cran-processx r-cran-ps r-cran-dsi r-cran-dslite
python3 -m pip install uv
export R_LIBS="$ROOT/Rlib"
export DSFLOWER_VENV_ROOT="$ROOT/venvs"
export DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_TORCH_BACKEND=cpu
Rscript -e 'stopifnot(getRversion() >= "4.4")'
R CMD INSTALL -l "$ROOT/Rlib" "$ROOT/dsFlower"
R CMD INSTALL -l "$ROOT/Rlib" "$ROOT/dsFlowerClient"
uv pip install --python "$ROOT/client/venv/bin/python" scikit-learn
curl -fsSL https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data -o "$ROOT/data_cache/breast-cancer-wisconsin.data"
curl -fsSL https://archive.ics.uci.edu/static/public/891/data.csv -o "$ROOT/data_cache/cdc_diabetes_health_indicators.csv"
cd "$ROOT/data_cache"
sha256sum breast-cancer-wisconsin.data cdc_diabetes_health_indicators.csv > CHECKSUMS.sha256
date -u +%FT%TZ > "$ROOT/logs/provision-end.txt"
