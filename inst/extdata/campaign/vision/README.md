# BUS-BRA vision classification: runtime blocker

The fresh A40 pod passed release, CUDA, dataset checksum, patient split and
dsImaging admission checks. The first `pytorch_resnet18` federation
(epsilon 1, seed 20260919) failed before DP training. All three nodes aborted:

```text
DSFLOWER SECURITY: package '_remote_module_non_scriptable' is not in pinned_packages.json (default-deny).
Aborting process.
```

`import_check.json` reproduces exit 99 during public model construction with
no data staged and the benchmark observer disabled. The installed default-deny
gate is unchanged. Torch 2.6.0+cu124 generates the rejected remote-module code
outside the trusted installation directories. No gate/pin-map exception,
package change, or alternate dependency stack was attempted.

| Epsilon | Federation attempts | Scored replicates | Central / federated-DP / pooled-DP / trivial | AUC gap |
|---|---:|---:|---|---|
| 1 | 1, failed before training | 0 | null | null |
| 8 | 0 | 0 | null | null |
| 4 | 0 | 0 | null | null |

The failed attempt took 735.409 seconds, including startup checks. Cleanup
succeeded. The epsilon-8 diagnostic is unassessed. No model was released,
no optimizer step completed, and no test scoring or schedule search ran.
The pod (`0nk8si7zupfczc`) remains running. The previous volume blocker is
superseded by this observed runtime-integrity blocker.

The three seeded splits are byte-identical to segmentation: 852 training
and 212 test patients, three sites of 284 patients. Dataset sizes in the
records come from preparation, not a scored model. `first_attempt.json`
contains the actual node privacy policies/capabilities, public initial
configuration, and node diagnostics. `runtime_preflight.json` verifies both
installed 0.5.0 runner hashes; `provisioning.json` retains resolved dependency
versions and the recovered ownership/native-build installation errors.

[BUS-BRA v1.0](https://zenodo.org/records/8231412): 1,875 ultrasound images
from 1,064 patients. Pathology provides benign/malignant labels; the released
`Case` column identifies patients. Fresh archive SHA-256:
`ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff`.
Zenodo declares CC BY 4.0; the archive attribution licence also requires
citation of Gómez-Flores W, Gregorio-Calas MJ, Pereira WCA (2024),
*BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis
Systems*, Medical Physics 51:3110–3123,
[doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812).

See [the reproduction instructions](../../../../tools/campaign/vision/README.md).
Downstream twin/scoring/aggregation tooling is implemented but remains
unvalidated on a completed cell because execution stopped at the gate.

`release_integrity_check.json` additionally verifies that the installed and
pod-source integrity gates match the v0.5.0 Git tag byte for byte.
