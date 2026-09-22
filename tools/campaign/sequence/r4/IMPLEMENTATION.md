# R4 window schedule selection and evaluation

`implementation.py --mode select` compares the diagnosis's two highest
noiseless-clipped endpoint candidates using real epsilon-8 federations at all
three original initialization seeds. The frozen subject-disjoint inner split
and the two-candidate rule are recorded in `selection/plan.json` before fitting.
The five-round endpoint is ranked by mean inner-validation macro OVR AUC.
`run_implementation.R` uses the released DSLite/Flower contract, node-owned
secrets and unchanged public observer. `implementation_verify.py` retains the
R3 staged-tensor, training-pin and full-horizon independent-accounting checks.

After selection, the track README and `implementation_protocol.json` declare
the single corrected window schedule before full-data training. The training
mode refuses a scoring marker, existing preparation, or existing run directory.
It trains one non-private federated twin and one clipped noiseless twin per
seed using `emulate.py`, then the nine real federations (three epsilons by three
seeds). Twins use final round five, equal site weights and public paired
Poisson streams; no checkpoints are chosen by monitoring scores. Their only
training monitor is every eighth TRAIN window. These controls are non-private.

The central comparator reuses the immutable R3 central model identities and
already-recorded metrics from `har_window_pytorch_lstm_eps8.json`. Verification
requires identical archive, prepared TRAIN bytes, subject/site membership,
features, public bounds and per-seed initial tensors, and the nominal 20-epoch,
Adam 0.01, batch-256, reset-every-four-epochs, 580-update schedule. No other pod
is contacted. Central checkpoints are not downloaded or rescored.

`score_implementation.py` checks every model hash, all 135 node-round captures,
all three central initialization identities and all six new twin identities.
A TRAIN-only prediction smoke test precedes an exclusive scoring marker. The
scorer loads official TEST once, predicts each of the 15 new models once and
writes the three `har_r4_window_` records using the existing record schema
with additional comparator branches. Diagnostic annotations never trigger a
rerun. Central and noiseless twin results are shared across epsilon values.

The optional pooled-DP arm is omitted to prioritize the real three-site matrix
and both matched noiseless controls within the session budget. No package code
is changed. Only `pod-flower-sequence-2` is used, and it remains running.

Reproduction after overlaying the tooling and declaring the frozen protocol:

```sh
export PYTHONPATH=/opt/cells-sequence/Rlib/dsFlowerClient/flower_app
PYTHON=/opt/cells-sequence/venvs/pytorch-gpu/bin/python
TOOLS=/workspace/cells-sequence/src/dsFlowerClient/tools/campaign/sequence/r4
ROOT=/workspace/cells-sequence
# Selection must precede the declaration and training.
timeout 3600 "$PYTHON" -u "$TOOLS/implementation.py" --root "$ROOT" --mode select
# Freeze selection and commit the declaration before this command.
timeout 5400 "$PYTHON" -u "$TOOLS/implementation.py" --root "$ROOT" --mode train
"$PYTHON" "$TOOLS/score_implementation.py" --root "$ROOT" --verify-only
"$PYTHON" "$TOOLS/score_implementation.py" --root "$ROOT"
```

HAR provenance follows the track README, with thesis keys `anguita_har_2013`
and `uci_har`. This is a window-level mechanism measurement on subject-disjoint
sites and provides no subject-level protection. Training-only public schedule
selection is not an end-to-end private selection guarantee; the separate
budgets/seeds do not establish a campaign-wide composition guarantee.

## Completed execution

The declaration was committed and pushed as `8c3e4451bc3338d169c59bbc7f435e05da69ad4a` before full-data training. All six selection federations, six noiseless twins and nine evaluation federations completed without a failed attempt or retry. The exclusive scoring pass completed at 2026-09-22T07:44:09.853045+00:00. No subsequent training or alternate cell was run.

The selected R4 schedule did **not** improve held-out ε=8 AUC over R3:
0.787685 versus 0.801364.
At ε=8, the paired ordered AUC contrasts are
-0.060526 ± 0.007338
for finite schedule plus federation,
-0.003144 ± 0.009037 for clipping, and
-0.118309 ± 0.014971
for added noise plus independent sampling variation. These are annotations,
not an exact causal allocation. No training or alternative followed scoring.

Public outcome records and the execution audit are under `inst/extdata/campaign/sequence/`. The shared release checkout contained unrelated regression edits, so this work used an isolated clone of the same evidence branch; those edits were left untouched.
