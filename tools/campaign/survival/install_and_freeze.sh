#!/bin/sh
# Run from the authorized workspace; never installs to a global R library.
set -eu
workspace=$(pwd)
mkdir -p runtime/rlib logs
for repo in dsFlower dsFlowerClient; do
  git -C "$repo" diff --quiet -- R inst/flower_app inst/python
  DSFLOWER_SKIP_PYTHON_SETUP=1 R CMD INSTALL --library=runtime/rlib "$repo" > "logs/install-frozen-$repo.log" 2>&1
done
Rscript dsFlowerClient/tools/campaign/survival/verify_runtime.R > logs/runtime-ready.log 2>&1
python3 dsFlowerClient/tools/check-runner-sync.py
Rscript - <<'R'
.libPaths(c(normalizePath('runtime/rlib'),.libPaths()))
library(dsFlower)
library(dsFlowerClient)
commits <- setNames(lapply(c('dsFlower','dsFlowerClient'),function(repo)
  trimws(system2('git',c('-C',repo,'rev-parse','HEAD'),stdout=TRUE))),c('dsFlower','dsFlowerClient'))
result <- list(commits=commits,runner_sha256=dsFlowerClient:::.compute_local_runner_hash(),
  built_utc=format(Sys.time(),'%Y-%m-%dT%H:%M:%SZ',tz='UTC'),
  package_versions=lapply(c('dsFlower','dsFlowerClient'),function(x)as.character(packageVersion(x))))
jsonlite::write_json(result,'runtime/build.json',auto_unbox=TRUE,pretty=TRUE)
cat('FROZEN_INSTALL_READY\n')
R
