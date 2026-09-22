# Fresh sequence GPU provisioning

The fresh `pod-flower-sequence-2` is running Ubuntu 22.04.5 on one NVIDIA A40
(46,068 MiB reported by `nvidia-smi`), with driver 595.91.07 and the supplied
RunPod PyTorch/CUDA image. Provisioning began with an empty `/workspace`.
No other pod was accessed. The pod remains running.

The source checkout and bootstrap began at approximately 06:16 UTC on
2026-09-22. The main provisioner ran from 06:19:19 to 06:21:04 UTC (105 seconds);
the separate final import/health verification completed at the timestamp in
`provisioning/provisioning-runtime.json`. This is below the 45-minute budget.

## Sources and runtime

- dsFlower `v0.5.1`, tag object `7cc54b976d920a7f91379f7e0381d6ef31a46911`,
  commit `12247417c15d1844064ca382d016c2a8793240f7`; installed version 0.5.1.
- dsFlowerClient `main`, commit `91dab75bb1c210bf5f520e7fbbcba90f5d659758`;
  installed version 0.5.1. Campaign tools were overlaid without changing package
  source, canonical runner, R code, configure scripts or metadata.
- Both installed canonical runner hashes:
  `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
- The required recurrent import guard is included in the server tag. Installed
  `dsFlower/python/sitecustomize.py` SHA-256:
  `3ae7c9ce6750c81c00e8c70e618d0bc52978486d7736587f39c716ff401fe98d`, exactly
  the R3 fixed guard.
- R 4.6.1; DSI 1.8.0; DSLite 1.4.1; Python 3.11.10. All 70 Python package
  versions from the retained R3 runtime were pinned in `requirements-r3.txt`
  and individually checked after installation. This includes
  torch 2.6.0+cu124, torchvision 0.21.0+cu124, Opacus 1.6.0,
  Flower 1.31.0, NumPy 2.4.6, pandas 3.0.6 and SciPy 1.17.1.
- The client environment also contains its required Optuna 4.8.0 and its two
  additional dependencies; exact freezes and all OS package versions are in
  `provisioning/`.

The original track provisioner used package configure to resolve Python ranges.
To reproduce all R3 versions exactly and avoid provisioning unrelated runtimes,
this provisioner installs the frozen CUDA and client environments explicitly,
then uses `DSFLOWER_SKIP_PYTHON_SETUP=1 R CMD INSTALL`. After successful imports,
`verify_provision.py` writes the release's own dependency-derived readiness
markers and asserts both release health checks. No package source was patched.

## Commands executed

From the laptop workspace, only the requested wrappers were used:

```sh
POD=~/Documents/GitHub/dsflower-cells/pods/pod-flower-sequence-2
PODCP=~/Documents/GitHub/dsflower-cells/pods/pod-flower-sequence-2cp
"$POD" 'mkdir -p /workspace/cells-sequence/src /workspace/cells-sequence/r4 /workspace/cells-sequence/logs'
"$POD" 'git clone --depth 1 --branch v0.5.1 https://github.com/isglobal-brge/dsFlower.git /workspace/cells-sequence/src/dsFlower'
"$POD" 'git clone --depth 1 --branch main https://github.com/isglobal-brge/dsFlowerClient.git /workspace/cells-sequence/src/dsFlowerClient'
"$POD" 'apt-get update -qq && apt-get install -y --no-install-recommends rsync'
"$PODCP" -rz tools/campaign/sequence/ pod:/workspace/cells-sequence/src/dsFlowerClient/tools/campaign/sequence/
"$POD" 'set -o pipefail; bash /workspace/cells-sequence/src/dsFlowerClient/tools/campaign/sequence/r4/provision.sh 2>&1 | tee /workspace/cells-sequence/logs/provision-r4.log'
"$PODCP" -rz tools/campaign/sequence/r4/verify_provision.py pod:/workspace/cells-sequence/src/dsFlowerClient/tools/campaign/sequence/r4/
"$POD" '/opt/cells-sequence/venvs/pytorch-gpu/bin/python /workspace/cells-sequence/src/dsFlowerClient/tools/campaign/sequence/r4/verify_provision.py > /workspace/cells-sequence/logs/verify-provision-r4.log 2>&1'
```

The first rsync attempt found that the image lacked rsync, so the OS bootstrap
above was performed and the transfer retried. A provision invocation before
that successful transfer found no script and performed no work. The final
`provision.sh` includes the separately executed verification command for a
single-command fresh reproduction. Logs retain the successful main provision
and final verifier. Foreground SSH commands were used throughout.

## State and data boundary

The `/workspace/cells-sequence/{Rlib,venvs,client}` paths are aliases to local
POSIX storage under `/opt/cells-sequence`. This avoids network-volume imports
consuming the release's 15-second process-readiness deadline. The GPU Python is
`/opt/cells-sequence/venvs/pytorch-gpu/bin/python`; `venvs/pytorch` aliases that
same CUDA environment. The client Python is
`/opt/cells-sequence/client/venv/bin/python`.

The official UCI URL was downloaded directly:
`https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`.
Archive SHA-256 was checked before preparation:
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
The track's `prepare_public_data.py` opened TRAIN members only and wrote raw
C-order token-major windows to `prepared/train.npz` and `prepared/train.csv`.
The archive remains in `data_cache/uci-har-240.zip`; TEST members have not been
opened. Preparation verified 7,352 windows, 21 training subjects and original
site counts 2,553 / 2,397 / 2,402. The retained prepared audit contains the old
empirical bounds; R4 diagnostics use the explicit R3 public bounds instead.
There was no synthetic preflight, evaluation-cell declaration, test scoring,
or campaign matrix run during provisioning.

For R launchers use:

```sh
export R_LIBS_USER=/opt/cells-sequence/Rlib
export DSFLOWER_VENV_ROOT=/opt/cells-sequence/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/opt/cells-sequence/client
export PYTHONPATH=/opt/cells-sequence/Rlib/dsFlowerClient/flower_app
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
