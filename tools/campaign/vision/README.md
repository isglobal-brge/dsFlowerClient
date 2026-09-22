# Representative vision classification cell

Execution is **blocked before training**. On 2026-09-22, the designated
`thesis-dsflower-2` pod (`7g8bohjp2o5ufh`) exposed an idle NVIDIA A40, but
`/workspace` filesystem operations timed out. The installed R packages,
prepared BUS-BRA data and split files could not be read. This is an
infrastructure failure; no package incompatibility has been established.

The [protocol](protocol.json) records the planned `pytorch_resnet18` cell:
three sites, five rounds, seeds 20260919–20260921, epsilon 1/8/4 in execution
order, delta 1e-6, patient clipping norm 1, and unchanged release defaults.
Each archived segmentation split has 852 training and 212 test patients,
with 284 training patients per site. All image labels belonging to a patient
remain on the same side of the split. The node pools frozen image features
within patient before training the private head. Planned evaluation is
per image, with malignant as the positive class.

The epsilon-8 diagnostic is AUC above 0.5 and accuracy above the held-out
majority-class rate. No diagnostic was assessed. There was no training,
schedule selection, test-label/image inspection, scoring or alternative run.
Central, pooled-DP and trivial results are absent, rather than estimated.

Only the bounded, read-only runtime verifier has been implemented. A working
federation/twin/scoring driver has **not** been validated or delivered. After
volume recovery, continue by adapting the three-worker orchestration in
`../segmentation/run_federated.R`, using actual dsImaging admission for the
image asset and metadata label. Keep new execution files under
`/workspace/cells-vision`; leave `/workspace/segmentation` unchanged.

## Reproduce the blocker

From this repository on the laptop, the following sends the verifier as a
command argument, so it does not require writing to the inaccessible pod
volume. It runs in the foreground and returns JSON; exit 2 means blocked.

```sh
python3 - <<'PY'
from pathlib import Path
import shlex
import subprocess

source = Path('tools/campaign/vision/verify_runtime.py').read_text()
wrapper = Path.home() / 'Documents/GitHub/dsflower-cells/pods/thesis-dsflower-2'
command = shlex.join([
    'python3', '-c', source, '--runner-sha256',
    '2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724'])
raise SystemExit(subprocess.run([str(wrapper), command], timeout=115).returncode)
PY
```

The verifier bounds each filesystem probe to ten seconds. If access works,
it checks the installed dsFlower and dsFlowerClient versions and both
canonical runner hashes. A mismatch requires installing unchanged v0.5.0
sources into `/workspace/cells-rlib`, then rerunning with
`--library /workspace/cells-rlib`. Successful preflight verifies release and
filesystem access only; it does not establish that training or dsImaging
admission works.

The local dsFlower and dsFlowerClient package code matches v0.5.0; the
canonical runner hash is the value above. Their release tag commits are
`408f08c539329e2711260050ab40a6567aa4d89e` and
`50dda000a32ffcbdd039c2b74c909df451392bfb`, respectively. Installed pod
versions and hashes remain unknown. No reinstall was possible.

## Dataset provenance

[BUS-BRA v1.0](https://zenodo.org/records/8231412) contains 1,875 images from
1,064 patients according to the archived segmentation audit. Classification
uses the released `Pathology` column and the released `Case` patient ID.
The archive URL is
`https://zenodo.org/api/records/8231412/files/BUSBRA.zip/content`; its archived
SHA-256 is
`ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff`.
The inaccessible pod archive was not rehashed. Dataset provenance and split
hashes come from `inst/extdata/campaign/segmentation/provenance/`.

Citation: Gómez-Flores, W., Gregorio-Calas, M. J., and Pereira, W. C. A.
(2024). *BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided
Diagnosis Systems*. Medical Physics 51, 3110–3123.
[doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812).
Dataset DOI: [10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412).
The archived Zenodo record declares CC BY 4.0. The ZIP also contains a
permissive attribution licence requiring citation, preserved in
`inst/extdata/campaign/segmentation/provenance/BUS-BRA-LICENSE.txt`.

## Evidence

`inst/extdata/campaign/vision/` contains the observed preflight, one blocked
record per epsilon, and `summary.json`. Every planned replicate has null
metrics and null training wall time; no seed was executed. Dataset sizes are
planned sizes from the archived audit, not a fresh pod census. The records
separate requested privacy settings from absent node-reported settings.

The pod was left running. No package code, runner, registry, privacy defaults,
DESCRIPTION, or segmentation files were changed.
