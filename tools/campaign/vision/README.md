# BUS-BRA frozen-backbone classification

Execution is **blocked before DP training** on the fresh A40 pod. Both
installed 0.5.0 runner hashes and all patient split hashes passed verification;
the archive was freshly rehashed and dsImaging admission passed. The first
epsilon-1 / seed-20260919 federation then aborted on all three nodes:

```text
DSFLOWER SECURITY: package '_remote_module_non_scriptable' is not in pinned_packages.json (default-deny).
Aborting process.
```

The installed gate independently reproduces exit 99 during public model
construction with the benchmark observer disabled and no data staged.
`params.load_user_model` imports Opacus; its Torch dependency imports a
generated module outside the trusted installation directories. The unchanged
gate rejects that module. A plain ClientApp import passes. The tested runtime
is torch 2.6.0+cu124, torchvision 0.21.0+cu124, Flower 1.31.0 and R 4.6.1;
the complete resolved dependencies are in the evidence's `provisioning.json`.
No alternate dependency stack or gate change was attempted.

The failed federation took 735.409 seconds, including startup checks, and
cleaned up its Flower processes successfully. No DP optimizer steps completed,
no trained model was released, and no central/pooled twin or test scoring ran.
All metric and gap fields remain null. The epsilon-8 diagnostic is unassessed.
The drivers below are implemented, but training/twin/scoring execution has
not been validated past this blocker; only admission, preflight and synthetic
metric checks passed. The pod remains running.

This driver evaluates `pytorch_resnet18` from dsFlower/dsFlowerClient 0.5.0:
a frozen ImageNet ResNet-18 and a 1,026-parameter linear classification head.
The [frozen protocol](protocol.json) declares three patient-disjoint sites,
five rounds, three seeds, epsilon 1/8/4, delta 1e-6, patient clipping norm 1,
and unchanged model defaults (SGD 0.001, batch 32, one local epoch).

Every seed reuses the byte-identical segmentation split: 852 training and
212 test patients, with 284 training patients per site. The released runner
averages image features within each patient and uses the modal patient label.
Metrics are evaluated per image, with malignant as the positive class.
The epsilon-8 diagnostic requires AUC > 0.5 and accuracy above the held-out
majority rate; report both individual replicates and the three-seed means.
No schedule search or post-score configuration changes are permitted. No
alternative cohort or contract is predeclared.

The central reference uses the identical initial head, patient pooling,
optimizer and five-round epoch schedule, with pooled Poisson sampling and
no clipping/noise. It is a finite-schedule noiseless reference, not an
optimized upper bound. The pooled-DP twin calls the unchanged released
training function. The trivial classifier predicts the training-majority
class and uses training prevalence for probability metrics. The same central
fit is reused across epsilon values for each seed.

## Reproduction

The commands below reproduce the frozen campaign and its current blocker;
they do not establish a working scored cell.

Use only the designated `pod-flower-vision` pod and `/workspace/cells-vision`.
A fresh Ubuntu 22.04 GPU image needs `rsync` installed before source transfer.
Use `rsync -rlzt` with the supplied wrapper: the volume does not support
preserving laptop ownership. Exclude `.git`, `*.o`, `*.so` and `__pycache__`.
The client install uses `--preclean` to rebuild native objects for Linux.
Place unchanged v0.5.0 package sources at
`src/dsFlower` and `src/dsFlowerClient`. Also place released dependency sources
at `src/dsHPC` (v0.2.5, commit
`2917ad168e2f1c6f191a9d964632744e52cc2d4b`) and `src/dsImaging` (v0.3.8,
commit `a5f218273cbf44cd3d4401672bd508299828284c`).

```sh
ROOT=/workspace/cells-vision
TOOLS="$ROOT/src/dsFlowerClient/tools/campaign/vision"
bash "$TOOLS/provision.sh"
python "$TOOLS/prepare.py" --root "$ROOT"
source "$TOOLS/environment.sh"
PY="$ROOT/venvs/pytorch-gpu/bin/python"
python "$TOOLS/verify_runtime.py" --root "$ROOT" --library "$ROOT/Rlib" \
  --runner-sha256 2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724 \
  > "$ROOT/runtime_preflight.json"
Rscript "$TOOLS/check_admission.R" "$ROOT"
"$PY" "$TOOLS/install_public_observer.py" > "$ROOT/observer-install.json"
"$PY" "$TOOLS/test_metrics.py"
"$PY" "$TOOLS/run_matrix.py" --root "$ROOT" --epsilons 1 8 4
```

After the failed first matrix attempt, record the independent check and
blocked evidence with:

```sh
"$PY" "$TOOLS/verify_import.py" --root "$ROOT" > "$ROOT/import-check.json"
python "$TOOLS/record_blocked.py" --root "$ROOT" \
  --out "$ROOT/src/dsFlowerClient/inst/extdata/campaign/vision"
```

The following downstream commands were not executed because training failed:

```sh
"$PY" "$TOOLS/score.py" --root "$ROOT" --epsilons 1 8 4
"$PY" "$TOOLS/assemble_evidence.py" --root "$ROOT" \
  --out "$ROOT/src/dsFlowerClient/inst/extdata/campaign/vision" --epsilons 1 8 4
```

All commands run in the foreground. The matrix stops on any failed run.
Training and twin verification never read held-out images or summarize
held-out labels. `score.py` requires all trained artifacts and writes an
exclusive scoring marker with their hashes before test access; it refuses
a second invocation. Retain failed attempts and their diagnostics.

`prepare.py` calls the segmentation campaign's `fetch` function for BUS-BRA,
the checkpoint and provenance, then its unchanged `prepare_busbra` and
`split_subjects` functions. Mechanical preparation performs the original
metadata and mask audit but computes no model scores. All three generated
split hashes must equal the archived hashes. Each training site is admitted
through an actual dsImaging resource with image assets, sample manifests,
content hashes, the pathology vocabulary and the patient roster. No legacy
table-backed image shortcut is used.

The public benchmark observer is copied from the segmentation/sequence
machinery. Analyst-side instrumentation seeds and captures public initial
arrays. Custodian-side instrumentation is installed only in the isolated
campaign venv, attaches after the mandatory runner integrity verifier, and
records node privacy settings, accountant steps and tensor hashes. It calls
the original training/calibration functions without replacing their results.
Node keys live under POSIX `/tmp`, because the volume ignores permissions.
They are not copied into evidence. Seeds alone do not reconstruct node-owned
DP noise. Patient data and model artifacts stay on the benchmark pod; only
public-cohort aggregate evidence is committed.

## Provenance

[BUS-BRA v1.0](https://zenodo.org/records/8231412), 1,875 images from 1,064
patients. The task uses `Pathology` (benign/malignant); privacy units use the
released `Case` identifier. Archive SHA-256:
`ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff`.
Source: `https://zenodo.org/api/records/8231412/files/BUSBRA.zip/content`.

Gómez-Flores W, Gregorio-Calas MJ, Pereira WCA (2024). *BUS-BRA: A Breast
Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems*.
Medical Physics 51:3110–3123. [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812).
Dataset DOI: [10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412).
Zenodo declares CC BY 4.0; the ZIP also contains an attribution licence
requiring the paper citation. Both provenance and the archive licence are
retained under the pod's `data/` and `prepared/busbra/` directories.

The pinned ImageNet checkpoint is `resnet18-f37072fd.pth`, SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Evidence is written to `inst/extdata/campaign/vision/`; see its README and
`summary.json` for execution status and measured outcomes.

`release_integrity_check.json` additionally verifies that the installed and
pod-source integrity gates match the v0.5.0 Git tag byte for byte.
