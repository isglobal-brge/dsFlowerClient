# Training-only window-level HAR diagnosis

This directory contains diagnostic tooling and evidence, not an evaluation-cell
declaration. Official TEST members are never opened. The existing R3 evidence
and scored cells are unchanged. The report is `SEQUENCE_DIAGNOSIS_R4.md`.

Provision the fresh sequence pod using [PROVISIONING.md](PROVISIONING.md).
Overlay this track directory, `tools/campaign/campaign_lib.R`, and the existing
`inst/extdata/campaign/sequence/har_window_pytorch_lstm_eps8.json` onto the
dsFlowerClient main checkout. The archived JSON supplies only the already
recorded R-built model configuration; no test data or new test predictions are
used. Then `bash run_diagnosis.sh` executes the fresh diagnostic workflow in
the foreground. Preparation and the real-contract launcher refuse existing
destinations. `emulate.py` retains completed arms and resumes only missing
diagnostic arms from `r4/emulation.json`.

The additional overlay commands, run from the laptop `sequence-client` checkout,
were:

```sh
POD=~/Documents/GitHub/dsflower-cells/pods/pod-flower-sequence-2
PODCP=~/Documents/GitHub/dsflower-cells/pods/pod-flower-sequence-2cp
"$PODCP" -rz tools/campaign/campaign_lib.R pod:/workspace/cells-sequence/src/dsFlowerClient/tools/campaign/
"$POD" 'mkdir -p /workspace/cells-sequence/src/dsFlowerClient/inst/extdata/campaign/sequence'
"$PODCP" -rz inst/extdata/campaign/sequence/har_window_pytorch_lstm_eps8.json pod:/workspace/cells-sequence/src/dsFlowerClient/inst/extdata/campaign/sequence/
```

- `prepare.py`: checks raw TRAIN arrays against official archive members,
  preserves original site membership, holds out TRAIN subjects 1/8/17/25/30,
  and supplies the fixed R3 public bounds.
- `run_federated.R`: executes the unchanged released epsilon-8 window contract
  on the inner training data, with node-owned randomness and the prior public
  observer after the mandatory integrity guard.
- `emulate.py`: central controls and equal-weight FedAvg diagnostics. The
  clipped arm calls the installed `_dp_fit` with noise multiplier zero. Paired
  noiseless arms use public ChaCha sampling streams, the actual expected-batch
  divisor and fresh optimizers each round. These arms are not DP releases.
- `verify.py`: checks all 15 node-round captures, tensor and initialization
  hashes, full-horizon accounting, and local prediction/class/preprocessing
  parity on inner validation only.
- `accountant_table.py`: ranks eight completed schedules by final noiseless
  clipped AUC and calibrated per-step gradient noise, and queries the actual
  accountant for the fixed-q epoch comparison.
- `source_audit.md`: source locations for the contract and runner semantics.

The execution used the same commands as `run_diagnosis.sh`, except the real
contract and `emulate.py --mode controls` overlapped as separate foreground
SSH jobs. `--mode candidates` followed completed controls and reused the
clipped baseline. This avoids concurrent writes to the diagnostic summary.
The R job had a 1,800-second timeout, controls 2,400 seconds, and candidate
diagnostics 3,600 seconds. No sleep-poll loops or detached training jobs were
used. Only `pod-flower-sequence-2` was accessed, and it remains running.

A preparation assertion initially expected bounds to contain only `lower` and
`upper`; the recorded R object also contains the public feature names. It was
corrected to check those names separately, without changing bounds or data.
The initial partial preparation remains at `r4/prepared-config-check` on the
pod. It preceded every diagnostic fit. There was no released-code change or
training replay caused by this tooling correction.

Committed evidence is under `diagnosis/` and `provisioning/`. Raw data, node
secrets, DSLite staging directories and model checkpoints stay on the pod.
No scoring script, test-scoring marker or evaluation-cell index is created.
