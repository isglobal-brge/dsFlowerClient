# Two-node Flower smoke

`dslite-multinode-smoke.R` is a Linux-oriented, real integration smoke for
dsFlower and dsFlowerClient. It starts two isolated local DSLite data nodes,
two Flower SuperNodes, and one SuperLink, then fits one round of
`pytorch_logreg` on small deterministic synthetic data.

GitHub Actions workflows are removed in 0.7.1. A real federation is verified by
this local multi-node integration harness. Run only one harness session at a
time, using a private R library and private work/secret directories. The command
below preserves the 280-second timeout plus 20-second hard-kill grace.

For a local run, install the current dsFlower and dsFlowerClient sources, use a
CPU PyTorch environment provisioned by dsFlower, and run:

```sh
smoke_dir="$(mktemp -d)"
DSFLOWER_VENV_ROOT=/path/to/dsflower/venvs \
DSFLOWER_CLIENT_VENV_ROOT=/path/to/dsflower-client \
DSFLOWER_NODE_SECRET_FILE="$smoke_dir/parent-node-secret" \
DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET=1 \
DSFLOWER_SMOKE_WORKDIR="$smoke_dir/run" \
timeout --signal=TERM --kill-after=20s 280s \
  Rscript tools/integration/dslite-multinode-smoke.R
```

The test chooses five dynamic localhost ports; no data leave localhost.
Node secrets are ephemeral local test files; the only persisted artifact is
derived from synthetic public data. Success requires exactly two SuperNodes,
one completed history round with no failures, a non-empty model artifact, and
a clean shutdown of the SuperLink and both SuperNodes.

## Full local suites

Install both packages into an explicitly selected private library, leaving the
shared/system library unchanged. From a workspace containing both package
checkouts and the release `run_tests.sh` entry point:

```sh
private_rlib="$(pwd)/.tmp/local-tests/rlib"
mkdir -p "$private_rlib"
R_LIBS_USER="$private_rlib" R_LIBS="$private_rlib" \
  DSF_RLIB="$private_rlib" bash run_tests.sh local
python3 dsFlowerClient/tools/check-runner-sync.py --server dsFlower
```

The entry point installs both packages, runs their testthat suites, executes every
Python test file in the relevant node runtimes and checks runner synchronization.
Keep native source/build verifiers and the dependency requirement files; workflow
removal does not remove test coverage. Known-answer fixtures depend on the final
v3 runner/runtime identity and must be regenerated deliberately after review.
