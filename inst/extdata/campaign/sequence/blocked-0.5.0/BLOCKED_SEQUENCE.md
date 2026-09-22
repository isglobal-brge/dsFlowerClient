# Sequence classification blocked on the released node import guard

Date: 2026-09-22. Pod: `pod-flower-sequence` (`5abpdvx54g6uz2`, NVIDIA A40).

The three-site `pytorch_lstm` federation for UCI HAR, epsilon 1, seed 20260922,
failed before any successful node training update or model release. All three
nodes reported exactly:

```text
DSFLOWER SECURITY: package '_remote_module_non_scriptable' is not in pinned_packages.json (default-deny).
Aborting process.
```

The R API reported:

```text
Federated training failed (status 1); no model was accepted or saved.
```

`dsFlower/inst/python/sitecustomize.py` applies its default-deny foreign-module
pin check. The installed torch runtime generates and imports the named module
in `torch/distributed/nn/jit/instantiator.py`; its non-scriptable template is
instantiated during the shared `torch.distributed.nn.api.remote_module` import.
This is a runtime import boundary before LSTM training. GRU was not attempted
because changing the recurrent model does not address this shared import path.
No pin-map addition, guard bypass, dependency experiment after this failure,
package change, or mechanism change was attempted. This records the observed
environment; it does not assert that all permitted dependency versions fail.

Both R packages are 0.5.0. Installed and release runner SHA-256:
`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
Environment: R 4.6.1, Python 3.11.10, torch 2.6.0+cu124, Opacus 1.6.0,
Flower 1.31.0. The CUDA backend was pinned to cu124 for the supplied CUDA 12.4
image. A synthetic direct GPU DP update passed; it did not exercise the full
node import guard and is not a HAR utility result.

The official archive SHA-256 is
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
Only TRAIN was read: 7,352 windows, 21 subjects, three sites of seven subjects
with 2,553 / 2,397 / 2,402 windows. Input is 128 time steps × nine channels,
time-major flattened, six activities, subject privacy, clipping norm 1,
delta 1e-6, five planned rounds, and unmodified LSTM defaults. The release's
generic patient path averages subject features and uses modal activity labels;
this material limitation is recorded in the protocol.

An earlier startup attempt timed out at the unchanged 15-second SuperLink
readiness limit, before initialization. Its error and 157.598 seconds are
preserved. Python environments and the R library were copied to local POSIX
storage, with work-root aliases; only generated console interpreter paths were
relocated, preserving their Python bodies. The retry passed startup but hit
the guard above after 123.300 seconds. Both attempts cleaned up successfully.

No held-out features or labels were read; no held-out metrics, central/trivial
scores, epsilon-8 diagnostic, or gap mean/SD exist. Remaining seeds and epsilon
4/8 were not executed. The pod remains running, with no campaign jobs running.

Evidence on the laptop: `evidence/sequence/`. Committed evidence:
`sequence-client/inst/extdata/campaign/sequence/`. Tooling and reproduction:
`sequence-client/tools/campaign/sequence/`. The shared branch is
`evidence/representative-cells`; the frozen design commit is
`409a62e3ce0e09e9882395134ab3bb78e7b0ed46`, and execution tooling was
`20078cefef0d053072b4f107138472feb15d491b`.

Final evidence commit pushed to `evidence/representative-cells`:
`6d5b558126fe7b16ff47a695ac5b72e532373116`.
