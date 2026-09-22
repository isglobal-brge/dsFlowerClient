# BUS-BRA frozen-backbone classification

This campaign resumes the frozen protocol after installing the server-only
import-guard patch from `fix/import-guard-torch-generated-modules`, commit
`c4eaaf153db4a7204878bf1d8995c16faa615110`. It leaves the client at 0.5.0 and
both canonical runners unchanged. The 0.5.0 failure evidence and original
protocol are retained under `inst/extdata/campaign/vision/blocked-0.5.0/`.
See the evidence README and summary for the measured execution outcome.

This driver evaluates `pytorch_resnet18` with dsFlower 0.5.1 and dsFlowerClient 0.5.0:
a frozen ImageNet ResNet-18 and a 1,026-parameter linear classification head.
The [frozen protocol](protocol.json) declares three patient-disjoint sites,
five rounds, three seeds, epsilon 1/8/4, delta 1e-6, patient clipping norm 1,
and unchanged model defaults (SGD 0.001, batch 32, one local epoch).

Every seed reuses the byte-identical segmentation split: 852 training and
212 test patients, with 284 training patients per site. The released runner
averages image features within each patient and uses the modal patient label.
Metrics are evaluated per image, with malignant as the positive class.
The epsilon-8 annotation compares AUC with 0.5 and accuracy with the held-out
majority rate, for individual replicates and the three-seed means. These
comparisons do not determine execution status or trigger configuration changes.
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

Use only `pod-flower-vision` and `/workspace/cells-vision`. The prepared BUS-BRA
collections and frozen checkpoint are reused. For a fresh pod, provision
0.5.0 and run `prepare.py` as documented in the archived blocked README.
Then fetch the server patch, verify its commit, and transfer it with the pod
rsync wrapper (`-rlzt`, excluding `.git`, `*.o`, `*.so`, and `__pycache__`).
Transfer this vision driver directory separately; no client reinstall is needed.
From the client checkout, also run the CI source comparison:
`python3 tools/check-runner-sync.py --server /path/to/dsFlower`.

```sh
ROOT=/workspace/cells-vision
TOOLS="$ROOT/src/dsFlowerClient/tools/campaign/vision"
source "$TOOLS/environment.sh"
# The prepared runtime is retained; no dependency setup is needed.
DSFLOWER_SKIP_PYTHON_SETUP=true R CMD INSTALL --preclean \
  --library="$ROOT/Rlib" "$ROOT/src/dsFlower"
PY="$ROOT/venvs/pytorch-gpu/bin/python"
mkdir -p "$ROOT/verification-0.5.1"
python3 "$TOOLS/verify_runtime.py" --root "$ROOT" --library "$ROOT/Rlib" \
  --runner-sha256 2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724 \
  > "$ROOT/runtime_preflight.json"
"$PY" "$TOOLS/verify_import.py" --root "$ROOT" \
  > "$ROOT/verification-0.5.1/import_check.json"
"$PY" "$ROOT/src/dsFlower/inst/python/tests/test_sitecustomize.py"
Rscript "$TOOLS/check_admission.R" "$ROOT"
"$PY" "$TOOLS/test_metrics.py"
# Needed on a fresh runtime; already installed on the prepared pod.
"$PY" "$TOOLS/install_public_observer.py" > "$ROOT/observer-install.json"
```

Capture the remaining data-free evidence inputs before running the matrix:

```sh
"$PY" - <<'PY' > "$ROOT/verification-0.5.1/python-packages.json"
import importlib.metadata as m, json, platform
print(json.dumps(dict(python=platform.python_version(), packages=dict(sorted(
    (d.metadata['Name'], d.version) for d in m.distributions()))), indent=2))
PY
"$ROOT/client/venv/bin/python" - <<'PY' > "$ROOT/verification-0.5.1/prediction-numerics.json"
import json, os, torch
fixed = dict(matmul_allow_tf32=False, cudnn_allow_tf32=False, deterministic_algorithms=True)
print(json.dumps(dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
    canonical_predictor_runtime=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE')),
    node_and_twin_training=fixed, direct_twin_test_features=fixed,
    policy='Retain the pushed numerical controls; test feature bitwise parity is not asserted.',
    test_data_read=False), indent=2))
PY
```

Before training, require the installed guard SHA-256 to equal the fetched
commit's guard (`3ae7c9ce6750c81c00e8c70e618d0bc52978486d7736587f39c716ff401fe98d`),
and recheck the archive, checkpoint and all three split hashes. If resuming
the original failure, move its run directory into `runs/blocked-0.5.0/` after
confirming its status is failed and no scoring marker exists. Retain it in full.

```sh
"$PY" "$TOOLS/run_matrix.py" --root "$ROOT" --epsilons 1 8 4
"$PY" "$TOOLS/score.py" --root "$ROOT" --epsilons 1 8 4
"$PY" "$TOOLS/assemble_evidence.py" --root "$ROOT" \
  --out "$ROOT/evidence-0.5.1" --epsilons 1 8 4
```

All commands run in the foreground. The matrix stops on any failed run.
Training and twin verification never read held-out images or summarize
held-out labels. `score.py` requires all trained artifacts and writes an
exclusive scoring marker with their hashes before test access; it refuses
a second invocation. The scorer resolves each saved model from the federation
status `output_dir`, including its model-named subdirectory; this path fix
was validated on a synthetic image before any held-out data access. Retain
failed attempts and their diagnostics. To bound scoring wall-clock time, the
three independent epsilon predictions for a seed run concurrently within the
foreground scoring process, through the same canonical predictor, once each.
Model settings, training order and metric definitions are unchanged. Do not
rerun this scored pod or change any scored configuration.

The recorded prediction numerical controls preserve the pushed routes: the
canonical predictor uses its default cuDNN TF32 setting, while direct twin
feature extraction disables TF32 and enables deterministic algorithms.
The checkpoint and transforms agree; exact tensor parity is verified for
training. Test feature bitwise parity between these inference routes is not
asserted. See `verification-0.5.1/prediction-numerics.json` in the evidence.

`prepare.py` reuses segmentation preparation and requires byte-identical
patient split hashes. Each site is admitted through an actual dsImaging
resource with image assets, pathology vocabulary and patient roster.
The released runner pools features and labels by patient before clipping.

The public benchmark observer seeds and captures analyst-side public initial
arrays. The observer in the isolated campaign venv attaches only after the
mandatory runner integrity verifier and records node privacy settings,
accountant steps and tensor hashes. It calls the original functions without
replacing their results. Node keys live under POSIX `/tmp` and are never
copied into evidence. Seeds alone do not reconstruct node-owned DP noise.
Patient data and model artifacts stay on the pod; only aggregate evidence
for this public cohort is committed.

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
