# UCI HAR sequence utility cell

The original 0.5.0 execution was blocked before training. Its unchanged records
are preserved under `inst/extdata/campaign/sequence/blocked-0.5.0/`.
The continuation uses dsFlower 0.5.1 (`c4eaaf153db4a7204878bf1d8995c16faa615110`)
and dsFlowerClient 0.5.0. The patch verifies one PyTorch-generated internal
module's origin and exact template; it leaves the canonical runner, mechanisms
and defaults unchanged. See `SEQUENCE_DIAGNOSIS.md` in the evidence directory.

This driver evaluates the unchanged
`pytorch_lstm` contract on the official UCI HAR subject-disjoint split.
`protocol.json` is the binding design recorded before scoring. No test members
are read during preparation or training. The final scorer refuses an existing
scoring marker, and the training driver refuses existing run directories.

There are 7,352 training windows from 21 subjects. Sorted training subjects are
dealt round-robin into three sites of seven subjects: 2,553, 2,397 and 2,402
windows. Each window has 128 time steps and nine inertial channels, flattened in
time-major order. Bounds are training-only channel minima/maxima expanded by
10%, repeated across time steps. Labels are the six activities encoded 0–5.

## Release limitation

The released generic patient path averages all feature vectors within a subject
and assigns its modal class (lowest class breaks ties), then applies per-unit
DP-SGD. Consequently each site trains on seven pooled sequences. It does not
compute a subject gradient from losses over the original individual windows.
The central twin uses exactly the same bounded, pooled tensors and initial
model arrays. This comparison characterizes that released behavior; it is not
a claim about a conventional window-trained HAR classifier. Held-out metrics
are window-level macro OVR AUC, accuracy and log-loss.

Defaults remain hidden size 32, SGD 0.001, batch size 32, one local epoch and
no scheduler. Five rounds give five full-subject-batch updates per site. Epsilon
1, 8 and 4 execute in that order with delta 1e-6 and clipping norm 1. Seeds
20260922–20260924 set public initialization. Node secrets and cryptographic DP
randomness retain release behavior; these seeds do not make DP noise publicly
replayable. The trivial predictor uses training window frequencies and their
majority class. No pooled-DP twin is scheduled.

## Reproduction

The work root is `/workspace/cells-sequence` on Ubuntu 22.04 with an NVIDIA GPU.
Rsync dsFlower at the patch commit and dsFlowerClient at its 0.5.0 release
commit into `src/dsFlower` and `src/dsFlowerClient`; overlay this campaign's
tooling onto the client source. Exact commits and hashes are in
`release-source.json`.
Use `rsync -rz` on the RunPod FUSE volume, which cannot preserve laptop owners.
The provisioner keeps environments and the R library on local POSIX storage
under `/opt/cells-sequence`, with aliases in the work root. Importing Flower
from the network volume exceeded the release's fixed 15-second SuperLink
readiness deadline in an initial attempt, before any model initialization.
For the observed recovery, existing environments were copied and their console
interpreter paths relocated with `relocate_console_scripts.py`; Python bodies
were checked unchanged. Both the startup failure and subsequent guard failure
are retained. Fresh guarded synthetic probes confirmed the same import failure for LSTM and
GRU, and regression DP updates pass for both with the patch. The HAR matrix
uses LSTM only.

```sh
ROOT=/workspace/cells-sequence
TOOLS=$ROOT/src/dsFlowerClient/tools/campaign/sequence
bash "$TOOLS/provision.sh"
export R_LIBS_USER=$ROOT/Rlib
export DSFLOWER_VENV_ROOT=$ROOT/venvs
export DSFLOWER_CLIENT_VENV_ROOT=$ROOT/client
export PYTHONPATH=$ROOT/Rlib/dsFlowerClient/flower_app
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=$(realpath "$ROOT/venvs")/pytorch-gpu/bin/python
"$PYTHON" "$TOOLS/prepare_public_data.py" --root "$ROOT"
"$PYTHON" "$TOOLS/preflight.py" "$ROOT"
"$PYTHON" "$TOOLS/install_public_observer.py"
"$PYTHON" "$TOOLS/capture_runtime.py" "$ROOT"
"$PYTHON" "$TOOLS/run_matrix.py" --root "$ROOT"
"$PYTHON" "$TOOLS/score_and_assemble.py" --root "$ROOT" --out "$ROOT/evidence"
```

On the existing prepared pod, retain the old `runs/`, runtime/preflight JSONs
and `logs/pytorch_lstm-*` under `attempts/blocked-0.5.0/` before creating a fresh
`runs/`. Do not move or reset a scoring marker. Install the server patch without
reprovisioning the already-frozen Python environment:

```sh
DSFLOWER_SKIP_PYTHON_SETUP=1 R CMD INSTALL -l "$ROOT/Rlib" "$ROOT/src/dsFlower"
"$PYTHON" "$TOOLS/trace_guard.py" --root "$ROOT" --out "$ROOT/diagnosis-0.5.1"
```

Then rerun preflight, runtime capture, matrix, and the single final scorer in
the order above. The scorer opens `test-scoring-started.json` exclusively. The execution driver
also refuses further training once that marker exists. An unscored tooling
interruption can be resumed with `run_matrix.py --root "$ROOT" --resume-unscored`;
it retains completed phases only after checking their model hashes and cleanup
status. Preserve the failed phase's empty directory/log under `attempts/` first.

The first completed federation exposed a checker-only label-dtype mismatch:
node captures hash float32 targets, while the initial checker used int64.
Matching the runner's dtype verified every feature and target hash without
changing any data values, model settings or completed federation. The first
central attempt had stopped before training. A subsequent TRAIN-only prediction
smoke check found that the central checkpoint omitted shared recurrent aliases;
central export now uses stock `state_dict`, exactly like the released ServerApp.
The central fit was replayed unscored with identical settings, and its unique
parameter tensors are checked against the retained first fit. Both failure logs
are archived; the completed federation was never repeated.
`trace_guard.py` is synthetic and does not open the HAR archive. The guard
regression lives in dsFlower's `inst/python/tests/test_sitecustomize.py`.

The observed blocked archive can be reassembled without running or scoring:

```sh
python assemble_blocked.py --raw /path/to/archived/raw-jsons --out /path/to/evidence
```

Run in the foreground. `logs/` retains provisioning and per-replicate output;
`runs/` retains models, initialization, node policy/capabilities, node-round
accountant observations, and status. Publish only public JSON evidence, not
secrets or data/model caches. The runtime recorder verifies both installed
runner hashes against `release-source.json`. All instrumentation is isolated
under this campaign directory and its dedicated virtual environment, adapted
from the segmentation campaign. The node observer attaches only after the
mandatory runner integrity verifier and delegates unchanged training calls.

The scorer verifies every site's five rounds, tensor hashes, source census,
noise and full-horizon accounting before opening the test split. It scores
unchanged released artifacts with the bundled prediction helper. Mean and
sample SD refer to three training replicates on the same fixed split. The gap
is federated-DP minus central macro AUC. Whether macro AUC exceeds 0.5 and epsilon-8 accuracy exceeds the
training-majority predictor's test accuracy are annotations only, never
pass/fail or rerun criteria; per-seed annotations are also reported. No setting is changed after
scoring. A shared package failure stops execution and is documented explicitly.
The synthetic preflight must pass the unchanged GPU DP training path before
starting any HAR replicate; it does not score or train on HAR.

## Dataset provenance

[UCI dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
provides the [official archive](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip).
Its SHA-256 is
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
The UCI repository identifies its licence as CC BY 4.0. Dataset DOI:
[10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013).
*A Public Domain Dataset for Human Activity Recognition Using Smartphones.*
European Symposium on Artificial Neural Networks, Computational Intelligence
and Machine Learning (ESANN).
