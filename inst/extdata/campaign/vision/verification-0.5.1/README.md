# Patch and pre-score verification

`source.json` identifies the fetched server commit, transfer and install command.
`install.log` records the successful dsFlower 0.5.1 installation; the client was
not reinstalled. The installer also created default auxiliary environments in
`/var/lib/dsflower/venvs`, while the campaign explicitly retained its prepared
`/workspace/cells-vision/venvs` runtime. `python-packages.json` matches the prior
campaign dependency versions. The documented reproduction command skips this
unneeded environment setup.

`runtime_preflight.json` verifies dsFlower 0.5.1, dsFlowerClient 0.5.0, CUDA and
both installed canonical runner hashes. `source-runner-sync.log` is the CI
file-by-file source comparison. `guard-sha256.txt` matches the installed and
transferred guard to the fetched Git source. `import_check.json` records a
fresh data-free vision-head construction under the active default-deny guard,
with the canonical runner pin and benchmark observer disabled.

Nine guard tests passed without skips, three synthetic metric tests passed,
and actual dsImaging resource admission passed. `input-sha256.txt` rechecks the
BUS-BRA archive, frozen checkpoint and all three segmentation split files.
`blocked-archive-check.json` verifies preservation of every original evidence
file; `protocol-parity.json` verifies unchanged statistical settings and core
training/prediction drivers.

Two pre-score execution issues were recovered. An early preflight ran during
staged installation, when the final package path was temporarily absent; its
output is retained as `preflight-during-install.json`, and the completed-install
verification is authoritative. The synthetic prediction probe initially used
the parent output folder, while `ds.flower.fit` saves into a named subdirectory.
The scorer now uses the returned `output_dir`; the corrected synthetic probe
passed. Both logs are retained. No test images or labels were read by these
checks, and no trained configuration was changed.

The single scoring lock hashes model files, model metadata, public initial
configuration, twins, protocol, scoring driver and canonical R prediction
wrapper before held-out access. To bound elapsed time, its three independent
epsilon predictions run concurrently for each seed through the same canonical
predictor, once each. Training order and metric definitions remain unchanged.

`prediction-numerics.json` records a data-free check of the existing prediction
runtime defaults. The canonical predictor permits cuDNN TF32; direct twin
feature extraction disables it and enables deterministic algorithms. Both use
the pinned backbone and transforms. Exact training tensor parity was verified;
bitwise parity of test features across the two pushed inference paths is not
asserted. These original numerical controls were preserved.

`first-independent-accounting.json` independently checks the first completed
federated-site and pooled-DP schedules with PRV accounting. The cell records
repeat this check for every actual noise multiplier and schedule. Opacus emits
its generic `secure_mode` warning on PrivacyEngine construction; the unchanged
canonical `dp_harness.make_private_dpsgd` requires and installs the node-derived
ChaCha20 sampling and noise streams instead. It does not use the default Torch
noise generator. The RDP order warning is retained in the execution logs; no
accounting settings were changed.

`provenance/` contains the downloaded source manifest, publisher metadata and
dataset/checkpoint licence texts. Patient data, model parameters and node
secrets remain on the designated pod.
