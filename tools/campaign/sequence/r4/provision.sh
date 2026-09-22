#!/bin/bash
set -euo pipefail
# Fresh Ubuntu 22.04 CUDA pod; R3 versions, released source, TRAIN data only.
ROOT=/workspace/cells-sequence
TOOLS="$ROOT/src/dsFlowerClient/tools/campaign/sequence"
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SECONDS=0
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$ROOT"/{data_cache,logs,r4} /opt/cells-sequence/{Rlib,venvs,client}
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
python -m pip install uv==0.11.33
export R_LIBS_USER=/opt/cells-sequence/Rlib
export DSFLOWER_VENV_ROOT=/opt/cells-sequence/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/opt/cells-sequence/client
export DSFLOWER_TORCH_BACKEND=cu124
PYTHON="$DSFLOWER_VENV_ROOT/pytorch-gpu/bin/python"
uv venv --python /usr/bin/python3.11 "$DSFLOWER_VENV_ROOT/pytorch-gpu"
uv pip install --python "$PYTHON" --torch-backend cu124 -r "$TOOLS/r4/requirements-r3.txt"
ln -s pytorch-gpu "$DSFLOWER_VENV_ROOT/pytorch"
uv venv --python /usr/bin/python3.11 "$DSFLOWER_CLIENT_VENV_ROOT/venv"
uv pip install --python "$DSFLOWER_CLIENT_VENV_ROOT/venv/bin/python" --torch-backend cu124 -r "$TOOLS/r4/requirements-r3.txt" optuna==4.8.0
"$PYTHON" -c 'import torch, opacus; assert torch.cuda.is_available(); assert torch.__version__ == "2.6.0+cu124"; assert opacus.__version__ == "1.6.0"; print("CUDA_READY", torch.__version__, torch.cuda.get_device_name())'
Rscript -e 'stopifnot(getRversion() >= "4.4"); library(DSI); library(DSLite); library(arrow)'
# Explicit frozen provisioning above avoids unrelated runtime environments.
DSFLOWER_SKIP_PYTHON_SETUP=1 R CMD INSTALL -l "$R_LIBS_USER" "$ROOT/src/dsFlower"
DSFLOWER_SKIP_PYTHON_SETUP=1 R CMD INSTALL -l "$R_LIBS_USER" "$ROOT/src/dsFlowerClient"
Rscript -e 'library(dsFlower); library(dsFlowerClient); stopifnot(packageVersion("dsFlower") == "0.5.1", packageVersion("dsFlowerClient") == "0.5.1"); a <- dsFlowerClient:::.compute_local_runner_hash(); b <- dsFlower:::.compute_harness_hash(); stopifnot(identical(a,b), a == "2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724"); cat("RUNNER_SHA256",a,"\n")'
test "$(sha256sum "$ROOT/src/dsFlower/inst/python/sitecustomize.py" | cut -d ' ' -f 1)" = 3ae7c9ce6750c81c00e8c70e618d0bc52978486d7736587f39c716ff401fe98d
curl -fL --retry 2 https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip -o "$ROOT/data_cache/uci-har-240.zip"
printf '%s\n' 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031  /workspace/cells-sequence/data_cache/uci-har-240.zip' | sha256sum -c -
"$PYTHON" "$TOOLS/prepare_public_data.py" --root "$ROOT"
"$PYTHON" "$TOOLS/r4/verify_provision.py"
uv pip freeze --python "$PYTHON" > "$ROOT/r4/python-runtime-requirements.txt"
uv pip freeze --python "$DSFLOWER_CLIENT_VENV_ROOT/venv/bin/python" > "$ROOT/r4/client-runtime-requirements.txt"
dpkg-query -W > "$ROOT/r4/dpkg-versions.txt"
printf '{"started_at":"%s","completed_at":"%s","elapsed_seconds":%s,"test_accessed":false}\n' "$STARTED" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SECONDS" > "$ROOT/r4/provisioning-timing.json"
printf '%s\n' PROVISIONING_DONE
