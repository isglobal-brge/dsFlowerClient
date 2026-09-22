#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /workspace/hazard
mkdir -p runtime/rlib runtime/tmp runtime/runs runtime/server logs data/survival
chmod 700 runtime/tmp runtime/runs
curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc -o /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
printf '%s\n' 'deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/' > /etc/apt/sources.list.d/cran.list
curl -fsSL https://r2u.stat.illinois.edu/ubuntu/dirk_eddelbuettel_pubkey.asc -o /etc/apt/trusted.gpg.d/r2u.asc
printf '%s\n' 'deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu jammy main' > /etc/apt/sources.list.d/r2u.list
printf '%s\n' 'Package: *' 'Pin: release o=CRAN-Apt Project' 'Pin-Priority: 700' > /etc/apt/preferences.d/99cranapt
apt-get update -qq
apt-get install -y --no-install-recommends r-base r-base-dev r-cran-arrow r-cran-digest r-cran-filelock r-cran-jsonlite r-cran-processx r-cran-ps r-cran-dsi r-cran-dslite r-cran-resourcer r-cran-remotes
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/hazard-uv.sh
sh /tmp/hazard-uv.sh
export PATH="/root/.local/bin:$PATH" UV_CACHE_DIR=/workspace/hazard/runtime/uv-cache
uv venv --python 3.11.10 runtime/venv
uv pip install --python runtime/venv/bin/python torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python runtime/venv/bin/python -r dsFlowerClient/tools/campaign/survival/hazard_v3/requirements-cpu.txt
ln -s ../venv runtime/server/pytorch
uv pip freeze --python runtime/venv/bin/python > runtime/pip-freeze.txt
Rscript dsFlowerClient/tools/campaign/survival/hazard_v3/install_r.R
export R_LIBS_USER=/workspace/hazard/runtime/rlib
export PATH="/workspace/hazard/runtime/venv/bin:$PATH"
sh dsFlowerClient/tools/campaign/survival/install_and_freeze.sh
