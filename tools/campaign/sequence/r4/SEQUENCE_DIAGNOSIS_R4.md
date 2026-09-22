# Sequence diagnosis R4: window-level UCI HAR

2026-09-22. Training-only diagnosis; no new evaluation cell and no official TEST
members opened. The fresh `pod-flower-sequence-2` remains running. Package code,
privacy policy and the existing scored evidence are unchanged.

The full federated-DP gap is not the marginal cost of Gaussian noise. On the
matched inner split, clipping lowers AUC by **0.076769**, and the marginal
epsilon-8 noise estimate is a further **0.086579 AUC** decrease. The full central-to-DP inner gap is
**−0.198954**. Schedule and federation estimates depend materially on the
sampling/optimizer-reset comparison. These are one-seed, ordered counterfactual
contrasts on TRAIN subjects; they do **not** exactly allocate the historical
held-out gap of −0.168301.

## Data and comparison design

The unchanged r3 inner split holds out TRAIN subjects **1, 8, 17, 25, 30**
(1,788 windows). Fit subjects are **3, 5, 6, 7, 11, 14, 15, 16, 19, 21, 22,
23, 26, 27, 28, 29** (5,564 windows). The original subject-disjoint sites are
filtered, not repartitioned:

| Site | Fit subjects | Windows | Steps/epoch | Total steps | Poisson q | Expected-batch divisor |
|---|---|---:|---:|---:|---:|---:|
| 1 | 6, 11, 16, 21, 28 | 1,797 | 8 | 160 | 1/8 | 224 |
| 2 | 3, 7, 14, 22, 26, 29 | 2,029 | 8 | 160 | 1/8 | 253 |
| 3 | 5, 15, 19, 23, 27 | 1,738 | 7 | 140 | 1/7 | 248 |

The r3 full-data DP schedule actually used **200 steps/site**, not approximately
190: `ceil(N/256)=10` at all three original sites, and `10×4×5=200`.
The full-data central reference used 580 steps. On this inner split, the full
central reference uses 440 steps. The primary finite central twin uses 160
steps, matching two sites; the 140-step twin covers the third site's horizon.

All baseline arms use the exact released DPLSTM(9,32), last hidden state,
six-logit head, public bounds, initialization seed 20260922, Adam 0.01, batch
256, five optimizer-reset blocks and no penalties. The full central resets
Adam every 88 steps (four epochs); finite central resets every 32 or 28 steps,
matching each local round budget. Each federated node resets Adam every round.
Fixed site weights are 1/1/1, irrespective of site population.

## Numerical decomposition

All numbers below score only the 1,788 inner-validation windows. Differences
are later minus earlier, so negative values are losses.

| Arm | Optimizer steps | Macro-AUC | Accuracy | Log-loss | Ordered AUC difference |
|---|---|---:|---:|---:|---:|
| Full central, ordinary shuffled batches | 440 | 0.989000 | 0.937360 | 0.240914 | Reference |
| (a) Finite central, ordinary shuffled batches | 160 | 0.918602 | 0.595078 | 0.863285 | **−0.070398** schedule |
| (b) Unclipped, noiseless FedAvg, matched Poisson batches | 160/160/140 | 0.953394 | 0.724832 | 0.618490 | **+0.034791** net federation-path change |
| (c) Coordinate-clamped and norm-clipped, noiseless FedAvg | 160/160/140 | 0.876624 | 0.615213 | 1.275110 | **−0.076769** clipping |
| (d) Real released federated-DP, epsilon 8 | 160/160/140 | 0.790045 | 0.446868 | 1.198721 | **−0.086579** marginal noise estimate |

The signed differences sum to **−0.198954**. Only the final contrast estimates
the added Gaussian-noise cost. Real node-owned sampling/noise streams are
independent of the public replay streams used for noiseless controls; therefore
(d)−(c) also contains one-run sampling variation. It is not a coupled-draw or
multi-seed estimate, and no confidence interval is claimed.

The positive net (b)−(a) must not be read as evidence that subject heterogeneity
helps. A central Poisson control with the same 160-step budget scores
**0.961505**, giving a **+0.042903** shuffled-to-Poisson/normalization contrast
and then **−0.008112** for the remaining federation contrast. Together they
equal +0.034791. Site-specific expected batch sizes, unequal local horizons,
equal-site weighting and subject heterogeneity remain part of that federation
comparison; this does not identify heterogeneity alone. An ordinary-shuffled
FedAvg bridge scores **0.941030**.

The 140-step ordinary central twin scores **0.961459**, versus 0.918602 at
160 steps. This nonmonotonic one-seed outcome exposes sensitivity to sampling
and optimizer-reset boundaries. Consequently the schedule term is a measured
contrast, not a universal per-update penalty. The fully sampler-aligned
alternative chain assigns **−0.027494** to finite schedule plus central sampling,
then −0.008112 to federation, with the same clipping and noise terms.

The full central control reproduces all three r3 inner-validation metrics
exactly. The clipped baseline peaks at 0.916357 AUC after round four and falls
to 0.876624 after round five. The reported and ranked endpoint is always round
five; no early checkpoint or held-out result selects a schedule.

## Verified execution and clipping behavior

- **Layout and bounds:** prepared raw float32 windows, labels and subjects equal
  a fresh read of official TRAIN archive members. C-order rows reshape to
  `[N,128,9]`; a recurrent pre-hook verifies shape and exact values. Public
  per-channel bounds are `(1,1,1,1,1,1,2,2,2)`, with the contract applying
  `clip(x,-b,b)/b` exactly once. These are design constants, not archive extrema.
- **Prediction:** the installed local predictor matches the direct contract
  model's probabilities and metrics on all 1,788 inner-validation windows;
  response mode equals probability argmax on 64 checked windows. Column order
  is class 0 through 5 (original activity code minus one). Prediction
  preprocessing equals training preprocessing exactly.
- **Model and gradients:** initialization hashes match the real ServerApp, and
  server-built versus node-built forward logits match exactly. The recurrent
  model is Opacus **DPLSTM**, not `torch.nn.LSTM` silently rewritten. The actual
  trainable leaves are two `RNNLinear` modules and one `Linear`, all with native
  Opacus gradient samplers. `force_functorch=False`; no unsupported trainable
  leaf requires functorch fallback. At initialization, batched per-window
  gradients match four individual autograd calculations within **3.73e−8**
  absolute error.
- **Clipping semantics:** `dp_harness.py:65–88,402–404` first replaces nonfinite
  gradient coordinates with finite values and clamps each coordinate to ±1,
  then Opacus clips the combined per-window gradient to global L2 norm 1.
  The batch update divides the sum by `floor(N/ceil(N/B))`, not the realized
  Poisson batch size. The noiseless clipped arm calls the installed `_dp_fit`
  with sigma 0; no alternate gradient implementation or package patch is used.
- **Federation and noise:** (b)/(c) share public ChaCha Poisson masks and the
  expected-batch divisor. The real contract retains node secrets and its normal
  cryptographic streams. All 15 captures pass staged-tensor hashes, parameter
  pins, q, step counts and accountant-history checks; all five rounds succeed.

Norm distributions cover **all 5,564 fit windows**. Round-one measurements use
the globally averaged model after that round, before further training.

| Model snapshot | Raw norm median | p90 | p99 | Maximum | Windows norm-clipped | Coordinates clamped | Aggregate norm retained | Aggregate cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initialization | 1.100 | 1.218 | 1.281 | 1.382 | 100.00% | 0% | 85.33% | 0.9821 |
| After one unclipped round | 2.822 | 3.744 | 4.937 | 5.332 | 81.33% | 0.00130% | 36.79% | 0.2420 |
| After one clipped round | 2.769 | 3.423 | 5.397 | 6.084 | 81.06% | 0% | 16.90% | 0.5834 |

At initialization the norm bound is only moderately active, so hidden32 and
128 steps do not intrinsically produce enormous initial gradients. Later,
global norm clipping substantially reduces and redirects the aggregate
gradient: at the clipped round-one snapshot it removes about 83% of its norm.
The coordinate clamp is negligible at these measured snapshots. Aggregate
retention measures the norm of the sum after clipping divided by before
clipping; it is not the average per-window multiplier or a causal utility
percentage. No nonfinite gradient coordinates were found.

## Accountant and candidate search

The privacy policy remains window-level, replace-one epsilon 8, delta 1e−6,
clipping norm 1 and five rounds. Calibration converts to add/remove epsilon 4
and delta `1e−6/(1+exp(4)) ≈ 1.798621e−8`, then composes the full local horizon.
The actual nodes report sigma **2.4267578125** at q=1/8 and 160 steps, and
**2.578125** at q=1/7 and 140 steps. Independent PRV checks give replace-one
epsilon **7.986844 / 7.986844 / 7.981442**, with delta below 1e−6.

The supported primary search surface is **Adam learning rate 0.003 or 0.01,
local epochs 4 or 8, batch 256 or 512**, with five rounds, no scheduler, weight
decay 0 and L1 penalty 0. The diagnostic sweep also tests Adam 0.001 and SGD
0.1. SGD/Adam/AdamW/RMSprop are contract options, but this sweep supports
prioritizing Adam; AdamW duplicates Adam here when weight decay is zero.
`hidden` is explicitly exposed by `R/model_registry.R:1018–1020`: retain **32**
for the ranked schedules, with **16** an allowed secondary search dimension
that was not measured here. Do not change sequence length, privacy unit, delta,
clipping norm or round count. No new schedule has been declared as an official
cell.

Eight measured schedules, ranked by final **noiseless clipped** macro-AUC:

| AUC rank | Optimizer LR / local epochs / batch | Inner AUC | Worst-site gradient noise SD, inner | Same SD, full TRAIN | Noise rank (ties; inner and full) |
|---:|---|---:|---:|---:|---:|
| 1 | adam 0.01 / 4 / 512 | 0.919101 | 0.007650 | 0.006259 | 1 |
| 2 | adam 0.003 / 8 / 256 | 0.908255 | 0.014561 | 0.012279 | 7 |
| 3 | adam 0.01 / 4 / 256 | 0.876624 | 0.010834 | 0.009194 | 4 |
| 4 | adam 0.003 / 4 / 256 | 0.860266 | 0.010834 | 0.009194 | 4 |
| 5 | adam 0.003 / 8 / 512 | 0.859591 | 0.010441 | 0.008502 | 3 |
| 6 | adam 0.001 / 4 / 256 | 0.790618 | 0.010834 | 0.009194 | 4 |
| 7 | adam 0.003 / 4 / 512 | 0.785540 | 0.007650 | 0.006259 | 1 |
| 8 | sgd 0.1 / 8 / 256 | 0.737378 | 0.014561 | 0.012279 | 7 |

The noise quantity is `sigma / expected_batch_size`, before the optimizer,
with C=1. It measures exposure to per-coordinate Gaussian gradient noise, not
predicted AUC or Adam's parameter displacement. Learning rate and optimizer do
not change accountant sigma at fixed batch/epochs, but they do change the
response to noise. All measured models have 5,702 unique parameters; multiplying
this SD by sqrt(5,702) gives the noise vector's RMS L2 norm. For the baseline it
is 0.8181 on the worst inner site, versus 0.5777 for batch512/epochs4.

At batch512 the inner sites have q=1/4, expected divisors 449/507/434, and sigma
**3.3203125** for 80 steps (four local epochs), or **4.53125** for 160 steps
(eight). Batch256/eight epochs uses 320/320/280 steps and sigma
**3.26171875 / 3.26171875 / 3.4716796875**. Thus larger sigma alone does not
imply greater mean-gradient noise: the expected-batch divisor also increases.

The actual released accountant gives the following on the original full TRAIN
sites at **fixed q=0.1**, batch256, five rounds, epsilon8/delta1e-6:

| Local epochs | Steps/site | Required sigma |
|---:|---:|---:|
| 1 | 50 | 1.408691406 |
| 2 | 100 | 1.713867188 |
| 4 | 200 | 2.197265625 |
| 8 | 400 | 2.934570312 |
| 12 | 600 | 3.520507812 |

Doubling local epochs from four to eight raises sigma from 2.197266 to 2.934570
at the same q. The longer-training candidate's better noiseless result therefore
does not establish better DP utility.

**Recommend these five candidates, in noiseless-AUC order:** Adam
**0.01/4/512**, **0.003/8/256**, **0.01/4/256** (baseline control),
**0.003/4/256**, and **0.003/8/512** (LR/epochs/batch). Their expected
mean-gradient-noise ordering is first, fifth, third=fourth, second. The first
candidate combines the best observed noiseless clipped AUC and the lowest
calibrated gradient-noise exposure in the measured set. Prioritize it, while
retaining the longer-training candidate as a utility-versus-noise comparison.
The remaining three screened schedules are retained above, not promoted on the
basis of unmeasured DP behavior.

The implementation objective is maximum **real epsilon-8 inner-validation
macro-AUC**, not the smallest central gap or sigma. Compare the recommended
schedules under the real contract, preferably across multiple initializations
given the observed finite-schedule variation, before choosing an implementation
schedule. This diagnosis stops before those DP candidate runs. Detailed ranks,
per-site geometry and accountant output are in `diagnosis/candidate-ranking.json`.

## Provisioning and retained state

Only the requested SSH/rsync wrappers for `pod-flower-sequence-2` were used,
including the rsync `pod:` prefix. The pod began with empty `/workspace`, has
one NVIDIA A40 (46,068 MiB), and runs the supplied RunPod PyTorch/CUDA image on
Ubuntu 22.04.5. Its base torch reports 2.4.1+cu124 / CUDA 12.4; the isolated
diagnostic environment uses the pinned r3 torch version below.
Provisioning, including clones/bootstrap/final checks, completed
in about 5½ minutes; the main provision script took 105 seconds.

- dsFlower **v0.5.1**, commit `12247417c15d1844064ca382d016c2a8793240f7`.
- dsFlowerClient **main**, commit `91dab75bb1c210bf5f520e7fbbcba90f5d659758`,
  installed metadata **0.5.1**.
- Both canonical runner hashes:
  `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
- Required import-guard hash:
  `3ae7c9ce6750c81c00e8c70e618d0bc52978486d7736587f39c716ff401fe98d`.
- R 4.6.1, DSI 1.8.0, DSLite 1.4.1, Python 3.11.10. All 70 r3 Python versions
  match, including torch 2.6.0+cu124, Opacus 1.6.0 and Flower 1.31.0.

The work root is `/workspace/cells-sequence`. Its `Rlib`, `venvs` and `client`
aliases point to local POSIX storage under `/opt/cells-sequence`, avoiding slow
volume imports. GPU Python is
`/opt/cells-sequence/venvs/pytorch-gpu/bin/python`; the client venv is
`/opt/cells-sequence/client/venv`. Exact provisioning commands, health markers,
source identity, OS inventory and package freezes are in
[PROVISIONING.md](PROVISIONING.md) and `provisioning/`.

The official archive is retained at `data_cache/uci-har-240.zip`, SHA-256
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
Raw full-TRAIN preparation is under `prepared/`; inner fit/validation are under
`r4/prepared/`; the completed real model, node captures and status are under
`r4/real-contract/`; noiseless checkpoints remain under `r4/emulation/`.
Raw data, model checkpoints and node secrets are not committed.
All diagnostic foreground jobs have finished. The real federation's cleanup
checks pass, and the final GPU check reports 0% utilization and 0 MiB in use.
The pod itself is still running; no other pod was accessed.

## Reproduction, scope and readiness

Tooling is on `evidence/representative-cells` in
`tools/campaign/sequence/r4/`. See [README.md](README.md) for execution order,
[source_audit.md](source_audit.md) for file/line references, `diagnosis/` for
numerical evidence and `run_diagnosis.sh` for a fresh-pod foreground workflow.
The preexisting branch was pulled with `git pull --rebase origin
evidence/representative-cells` before changes, and is refreshed before pushing.

The preparation assertion corrected here concerned an extra `features` key in
the recorded bounds object. It was a diagnostic-tool check, fixed before any
fit; public bounds and prepared values did not change. The partial preparation
is retained on the pod. No released-code defect was found, no package code
changed, and no new official evaluation cell, scoring marker or test prediction
was created.

The observed decompositions are order-dependent and based on one initialization
and one inner split. The original 0.970/0.801 numbers are three-seed held-out
means on different training populations and cannot be subtracted from these
inner controls to claim exact historical causal shares. Public non-DP inner
selection does not supply an end-to-end private selection guarantee. No
campaign-wide composition guarantee is claimed.

Primary external references: [UCI HAR archive and dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
[Opacus recurrent layers](https://opacus.ai/api/dp_rnn.html), and
[Opacus DP optimizer](https://opacus.ai/api/optim/dp_optimizer.html). Runtime
claims above are checked against the pinned installed source and local evidence.

PROCEED: yes
