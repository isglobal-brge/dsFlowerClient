#!/usr/bin/env bash
# Continue the regression pod's interrupted installation, using the archived tags.
set -euo pipefail
ROOT=/workspace/cells
export R_LIBS="$ROOT/Rlib"
export DSFLOWER_VENV_ROOT="$ROOT/venvs"
export DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_TORCH_BACKEND=cpu DSFLOWER_FORCE_GPU=0
mkdir -p "$ROOT/logs"
date -u +%FT%TZ > "$ROOT/logs/r4-provision-resume.txt"
uv python install 3.11
test "$(git get-tar-commit-id < "$ROOT/regression-r4-server-v050.tar")" = 408f08c539329e2711260050ab40a6567aa4d89e
test "$(git get-tar-commit-id < "$ROOT/regression-r4-client-v050.tar")" = 50dda000a32ffcbdd039c2b74c909df451392bfb
if [ ! -f "$ROOT/wheels.tar" ]; then
  cat "$ROOT"/wheel-chunks/wheels.part-* > "$ROOT/wheels.tar"
fi
echo "828008601d2717d4da5f616209e97b76beb652a6d2713e2626ec3c6b3d7dbdf6  $ROOT/wheels.tar" | sha256sum -c
tar -xf "$ROOT/wheels.tar" -C "$ROOT/wheels"
(cd "$ROOT/wheels" && sha256sum -c CHECKSUMS.sha256)
export UV_NO_INDEX=1 UV_OFFLINE=1 UV_FIND_LINKS="$ROOT/wheels"
export UV_CONSTRAINT="$ROOT/dsFlowerClient/tools/campaign/regression/runtime-constraints.txt"
Rscript -e 'p <- c("rlang","cli","glue","magrittr","vctrs","purrr","tidyselect","cpp11","bit","bit64","jsonlite","digest","processx","ps","filelock","curl","openssl","arrow","DSI","DSLite","resourcer"); stopifnot(all(vapply(p,requireNamespace,logical(1),quietly=TRUE))); f<-tempfile(fileext=".parquet"); arrow::write_parquet(data.frame(x=1:3),f); stopifnot(identical(as.data.frame(arrow::read_parquet(f))$x,1:3)); unlink(f)'
R CMD INSTALL --library="$ROOT/Rlib" "$ROOT/dsFlower" > "$ROOT/logs/r4-server-install.log" 2>&1
tail -60 "$ROOT/logs/r4-server-install.log"
R CMD INSTALL --library="$ROOT/Rlib" "$ROOT/dsFlowerClient" > "$ROOT/logs/r4-client-install.log" 2>&1
tail -25 "$ROOT/logs/r4-client-install.log"
uv pip install --python "$ROOT/client/venv/bin/python" --torch-backend cpu \
  'torch>=2.0.0,<3.0.0' 'numpy>=1.21.0' 'pandas>=1.3.0' \
  'pyarrow>=10.0.0' 'opacus>=1.4.0,<2.0.0' 'cryptography>=42.0.0'
Rscript -e 'stopifnot(getRversion()>="4.4", packageVersion("dsFlower")=="0.5.0", packageVersion("dsFlowerClient")=="0.5.0", dsFlower:::.venv_is_healthy(file.path(Sys.getenv("DSFLOWER_VENV_ROOT"),"native-tree"),"native-tree"), dsFlower:::.venv_is_healthy(file.path(Sys.getenv("DSFLOWER_VENV_ROOT"),"pytorch"),"pytorch"), dsFlowerClient:::.client_venv_is_healthy()); cat(R.version.string,"\n"); cat(dsFlowerClient:::.compute_local_runner_hash(),"\n"); print(installed.packages()[c("dsFlower","dsFlowerClient","DSI","DSLite","arrow"),c("Package","Version")])'
uv pip freeze --python "$ROOT/venvs/pytorch/bin/python" > "$ROOT/logs/r4-pytorch-freeze.txt"
uv pip freeze --python "$ROOT/client/venv/bin/python" > "$ROOT/logs/r4-client-freeze.txt"
date -u +%FT%TZ > "$ROOT/logs/r4-provision-end.txt"
