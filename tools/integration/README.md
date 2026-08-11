# Two-node Flower smoke

`dslite-multinode-smoke.R` is a Linux-oriented, real integration smoke for
dsFlower and dsFlowerClient. It starts two isolated local DSLite data nodes,
two Flower SuperNodes, and one SuperLink, then fits one round of
`pytorch_logreg` on small deterministic synthetic data.

The smoke is intentionally not part of push or pull-request CI. The
`multinode-smoke.yml` workflow runs only on manual dispatch or its weekly
schedule. It has a 280-second timeout, a 20-second hard-kill grace, and an
always-run process reaper, keeping the hard ceiling at 300 seconds.

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
