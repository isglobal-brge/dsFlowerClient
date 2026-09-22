# Discrete-time hazard v3

## Diagnosis recorded before new training

Packaged SUPPORT2 full epsilon-8 means (seeds 1101/1102/1103):

| Contract | Federated-DP | Pooled-DP | Pooled non-private | Pooled-DP minus federated | Non-private minus pooled-DP |
|---|---:|---:|---:|---:|---:|
| Hazard v2 h06 | 0.5951121874 | 0.6209146899 | 0.6218393185 | 0.0258025026 | 0.0009246285 |
| Lognormal AFT | 0.6339464325 | 0.6386643651 | 0.6486589680 | 0.0047179325 | 0.0099946029 |
| Weibull AFT | 0.6224505587 | 0.6274792577 | 0.6386557207 | 0.0050286990 | 0.0111764630 |

This is an observed gap decomposition, not identification of separate causal effects.
Hazard's evaluated non-private ceiling is 0.0268196495 below lognormal and
0.0168164023 below Weibull; it is a ceiling for this fit/schedule, not a proven
limit of the entire discrete-hazard model family.

| Epsilon 8 mechanism | Site hazard | Pooled hazard | Site AFT | Pooled AFT |
|---|---:|---:|---:|---:|
| Patients | 2428 | 7284 | 2428 | 7284 |
| Nominal batch / expected divisor | 64 / 63 | 64 / 63 | 128 / 127 | 128 / 127 |
| Poisson probability | 1/38 | 1/114 | 1/19 | 1/57 |
| Total epochs | 40 | 40 | 20 | 20 |
| Total steps | 1520 | 4560 | 380 | 1140 |
| Noise multiplier | 1.52587890625 | 1.03027343750 | 1.56738281250 | 1.060791015625 |
| Per-coordinate gradient noise SD | 0.02422030 | 0.01635355 | 0.01234160 | 0.00835269 |

All use the unchanged audited PRV calibration with RDP fallback and independent
PRV verification. Replace-one epsilon=8, delta=1e-5 is converted to add/remove
epsilon=4, delta=1.79862099620916e-7. Independently verified hazard epsilon is
7.9855810614 (site), 7.9862946596 (pooled). The site/pooled sigma ratios are
1.48104 (hazard) and 1.47756 (AFT): this difference cannot by itself explain
the hazard-specific gap. With independent site noises, equal averaging reduces
the direct noise component by sqrt(3); nonlinear training and local drift mean
this is not an exact final-model noise decomposition.

The implementation does NOT flatten person-period rows into sampling units.
Each patient has one feature row, K logits, K event labels and K exposure masks.
Its loss is sum_j mask_ij*BCE_ij/K; Opacus clips the joint gradient once at norm 1.
For a linear head, g_ij=mask_ij*(sigmoid(z_ij)-d_ij)*[x_i,1]/K.
With 15 bounded covariates, its norm is at most 4/sqrt(K): 1.26491 for K=10;
at zero logits it is at most 2/sqrt(K)=0.63246. Clipping can attenuate a whole
patient's contribution across periods, but /K already suppresses gradients.
No observed clipping fraction was stored in the historical evidence, so heavy
clipping is not established. Reducing K increases signal per coordinate while
leaving patient accounting unchanged; /K NLL is not comparable across K.

The head has 15*K slopes plus K learned biases: 160 parameters at K=10,
versus 16 for AFT. These are independent interval heads, not a shared
proportional-hazard slope plus a baseline. Every bias is sampled, jointly clipped,
noised and averaged with the slopes. Fixed num-examples=1 yields equal-site
FedAvg; because full sites all have 2428 subjects, weights equal patient weights,
but not period-specific exposure weights. Averaging logistic parameters does not
pool likelihood sufficient statistics. Late intervals have fewer exposed patients,
and their gradients are further divided by K. Equal-width first interval spans
182.5 days, compressing early deaths; quantile grids can redistribute signal.
The registered declarative linear contract does not expose an alternative shared
baseline parameterization; changing it would require forbidden package/runner
changes and is excluded here.

Local epochs traverse patients, not up to 10 times as many rows. At h06 each
round has 4*38=152 sequential site updates versus 4*114=456 pooled updates.
FedAvg averages three endpoint changes; it does not concatenate their optimizer
steps. Thus the pooled twin has three times the sequential optimization horizon
at the same learning rate. With /K weak gradients and 10 independent heads,
under-optimization and exposure imbalance are credible mechanisms; AFT's scalar
head has denser signal and no /K attenuation. Longer local schedules, larger
learning rates, smaller K and quantile bins are the prioritized interventions.
Local drift, clipping, and noise effects are intertwined; historical records alone
do not establish their causal shares. New development-only diagnostics will test
the under-optimization explanation without using confirmation outcomes.

The existing contract already maps age [0,100], comorbidity [0,10] and
binary [0,1] covariates to [-1,1]. Raw age scale is therefore not the missing
intervention. Repeating public-bound standardisation is algebraically identical;
we retain it for every candidate rather than label duplicate fits as a new lever.
At K=10 the last logit cannot affect left-endpoint RMST ranking, although its
parameters still receive noise. AFT's single global intercept cannot change
ranking; hazard's multiple intercepts change relative interval contributions.

## Development-only mechanism controls

These controls were declared before any live development score returned. They use only inner training/validation data. They are unnoised public linear-SGD calculations, with direct patient gradients checked against the frozen autograd loss; they are not actual federations or DP releases and never enter selection.

| Grid | Pooled, full step count | Pooled, site step count | Unnoised equal-site averaging | Same, unit-clipped |
|---|---:|---:|---:|---:|
| K10 equal-width | 0.621301 | 0.593293 | 0.593365 | 0.593365 |
| K5 event quantiles | 0.631817 | 0.600879 | 0.600647 | 0.600654 |

Initial gradient geometry on the same inner training splits (1942 patients/site):

| Contract | Parameters | Patients clipped at initialization | Norm of mean clipped gradient | Site noise RMS L2 per update |
|---|---:|---:|---:|---:|
| hazard h06 (K10) | 160 | 0.00% | 0.115697 | 0.338204 |
| lognormal | 16 | 94.32% | 0.193459 | 0.055527 |
| weibull | 16 | 95.05% | 0.190718 | 0.055527 |

This is an initial geometry diagnostic, not a final-model signal-to-noise decomposition. AFT clips far more patients initially yet has denser, stronger mean signal and much smaller total noise magnitude. The hazard problem is weak /K signal relative to fixed unit-clip noise across many coordinates, not large raw covariate scales.

For h06, reducing only the pooled sequential update count explains a 0.028008 C-index drop on these development splits; equal-site averaging differs from that step-matched pooled control by +0.000072. The h06 initial mean patient-gradient norm is 0.3070/0.3081 and maximum 0.7664/0.7647; no sampled gradient exceeds 1 throughout these unnoised controls. This supports insufficient sequential optimization as the dominant tested explanation, rather than heavy clipping or equal-site weighting itself. It does not assign an exact causal fraction of the historical noisy test gap.

At K10, first-bin event counts are 2761/2741; last-bin counts 19/16, with 221/216 exposed patients. K5 quantiles redistribute events to roughly 772–831 per interval. K5 initial clipping fractions are 23.52%/23.69%, but only 0.150%/0.142% of sampled visits exceed 1 over the clipped unnoised trajectories. Grid coarsening improves both pooled and federated controls; its effect is not simply removing clipping.

## Frozen development sweep

20 configurations × two seeds (1101/1102), epsilon 8, three sites of 1942 inner-training patients and 1458 pooled validation patients. Every selected score comes from actual isolated DSLite federation with patient DP. All 35 planned configurations and per-seed quantile boundaries are in `frozen_configurations.json`; `sweep.csv` and `development/` retain the full executed table and cell records; `sweep-full.csv` also marks every unstarted planned configuration. 15 configurations were omitted by the predeclared time cutoff.

No new wave starts at or after 02:40 UTC; every started pair and wave must finish. The optional envelope arms are included only when selection locks before 02:50 UTC. Both time rules were fixed before the sweep and operate before any confirmation outcome is available.

Selection: maximum mean federated-DP inner C; exact ties use fewer total epochs, smaller K, then ID. No pooled/control score or outer confirmation metric entered selection. Public-bound scaling was already enabled and remains fixed. Quantile edges use each seed’s inner-training events, never validation events; the resulting public grids remain unchanged at confirmation.

| Rank | ID | K/grid | Strategy | Optimizer/LR | Rounds × local epochs | Batch | Seed1101 C | Seed1102 C | Mean C |
|---:|---|---|---|---|---|---:|---:|---:|---:|
| 1 | g10 | 5/quantile | fedavgm | sgd/0.05 | 20 × 2 | 64 | 0.633339 | 0.646463 | 0.639901 |
| 2 | g08 | 5/quantile | fedavg | adam/0.02 | 10 × 1 | 64 | 0.638045 | 0.638319 | 0.638182 |
| 3 | g16 | 5/quantile | fedavg | sgd/0.05 | 10 × 4 | 32 | 0.627456 | 0.631210 | 0.629333 |
| 4 | g12 | 5/quantile | fedadam | sgd/0.05 | 20 × 2 | 64 | 0.629412 | 0.623813 | 0.626613 |
| 5 | g20 | 5/quantile | fedavg | sgd/0.05 | 20 × 4 | 64 | 0.625526 | 0.622668 | 0.624097 |
| 6 | g06 | 5/quantile | fedavg | sgd/0.1 | 10 × 4 | 64 | 0.625595 | 0.617976 | 0.621785 |
| 7 | g09 | 8/quantile | fedavg | adam/0.02 | 10 × 1 | 64 | 0.620612 | 0.610391 | 0.615502 |
| 8 | g11 | 8/quantile | fedavgm | sgd/0.05 | 20 × 2 | 64 | 0.616045 | 0.601155 | 0.608600 |
| 9 | g18 | 5/quantile | fedavg | sgd/0.1 | 20 × 1 | 64 | 0.609897 | 0.592360 | 0.601129 |
| 10 | g02 | 5/quantile | fedavg | sgd/0.05 | 10 × 4 | 64 | 0.608676 | 0.591760 | 0.600218 |
| 11 | g17 | 8/quantile | fedavg | sgd/0.05 | 10 × 4 | 32 | 0.608235 | 0.584539 | 0.596387 |
| 12 | g01 | 10/equal_width | fedavg | sgd/0.05 | 10 × 4 | 64 | 0.598836 | 0.592299 | 0.595567 |
| 13 | g07 | 8/quantile | fedavg | sgd/0.1 | 10 × 4 | 64 | 0.601796 | 0.587181 | 0.594489 |
| 14 | g13 | 8/quantile | fedadam | sgd/0.05 | 20 × 2 | 64 | 0.590103 | 0.583863 | 0.586983 |
| 15 | g03 | 8/quantile | fedavg | sgd/0.05 | 10 × 4 | 64 | 0.584995 | 0.573748 | 0.579371 |
| 16 | g19 | 8/quantile | fedavg | sgd/0.1 | 20 × 1 | 64 | 0.584362 | 0.566378 | 0.575370 |
| 17 | g05 | 10/quantile | fedavg | sgd/0.05 | 10 × 4 | 64 | 0.572957 | 0.559175 | 0.566066 |
| 18 | g04 | 12/quantile | fedavg | sgd/0.05 | 10 × 4 | 64 | 0.562238 | 0.555661 | 0.558950 |
| 19 | g15 | 8/quantile | fedyogi | sgd/0.05 | 20 × 2 | 64 | 0.531396 | 0.519159 | 0.525277 |
| 20 | g14 | 5/quantile | fedyogi | sgd/0.05 | 20 × 2 | 64 | 0.525975 | 0.509838 | 0.517907 |

For the same K5 quantile grid and plain SGD, the baseline mean is 0.600218. Doubling learning rate gives 0.621785; halving batch size gives 0.629333; doubling total epochs gives 0.624097. Doubling learning rate while halving total epochs leaves learning rate × update count unchanged and gives 0.601129. These development comparisons support the update-count diagnosis, together with the unnoised controls. Batch/epoch changes also change calibrated noise, and all DP draws are independent; these are not isolated noise-only effects.

## One confirmation pass

Selected **g10: K=5, quantile, fedavgm, sgd LR 0.05, 20 rounds × 2 local epochs, batch 64**. Selection locked at 2026-09-22T02:43:02.980652+00:00. Aggregation settings: server learning rate 1, server momentum 0.9, fixed unit site weights. Each twin matches the grid, architecture, public feature transform, fixed initialization seed 0, local optimizer reset schedule and server post-processing; pooled q and sequential step counts differ by population. Seeds define subject splits, not published DP noise seeds.

Optional envelope arms omitted by the predeclared time rule: none.

**This is the third confirmation of the hazard contract, after the v1 matrix and v2 schedule h06.** The historical holdouts are reused, not new independent validation. Development opens only the training files for each split. This is disjointness within each seed, not global disjointness across seeds: another seed’s training set can include a held-out patient. The shared configuration is selected across development replicates, so this is not independent nested validation. All intervals and the final verdict remain descriptive benchmark evidence. There was one selected configuration and one confirmation matrix, with no confirmation-driven tuning or repeat fit.

Training-ID-only overlap audit (no outer test file or outcome read): development 1101 includes 1469 subjects held out by seed 1102; development 1101 includes 1449 subjects held out by seed 1103; development 1102 includes 1469 subjects held out by seed 1101; development 1102 includes 1471 subjects held out by seed 1103. See `split_overlap.json`.

| Arm | ε | Federated-DP C | Pooled-DP C | Pooled nonprivate C | Null C |
|---|---:|---:|---:|---:|---:|
| full (3 sites; 2428,2428,2428/site) | 1 | 0.623475 ± 0.010064 | 0.610166 ± 0.008382 | 0.641773 ± 0.006815 | 0.500000 ± 0.000000 |
| full (3 sites; 2428,2428,2428/site) | 4 | 0.638433 ± 0.001760 | 0.642821 ± 0.002095 | 0.641773 ± 0.006815 | 0.500000 ± 0.000000 |
| full (3 sites; 2428,2428,2428/site) | 8 | 0.641160 ± 0.003684 | 0.642864 ± 0.007542 | 0.641773 ± 0.006815 | 0.500000 ± 0.000000 |
| heterogeneous (3 sites; 2428,2428,2428/site) | 8 | 0.639300 ± 0.005645 | 0.640756 ± 0.004947 | 0.643890 ± 0.006645 | 0.500000 ± 0.000000 |
| small600 (3 sites; 200,200,200/site) | 8 | 0.572795 ± 0.018412 | 0.609357 ± 0.014820 | 0.610005 ± 0.010790 | 0.500000 ± 0.000000 |
| two-sites (2 sites; 3642,3642/site) | 8 | 0.641777 ± 0.004734 | 0.638714 ± 0.005711 | 0.641773 ± 0.006815 | 0.500000 ± 0.000000 |

Epsilon 8 accounting geometry (seed 1101; the other seeds have the same population/schedule geometry):

| Arm / population | Patients | Poisson q | Total sequential steps | Noise multiplier | Gradient noise SD per coordinate |
|---|---:|---:|---:|---:|---:|
| full / site | 2428 | 0.02631579 | 1520 | 1.52587891 | 0.02422030 |
| full / pooled | 7284 | 0.00877193 | 4560 | 1.03027344 | 0.01635355 |
| two-sites / site | 3642 | 0.01754386 | 2280 | 1.30371094 | 0.02069382 |
| two-sites / pooled | 7284 | 0.00877193 | 4560 | 1.03027344 | 0.01635355 |
| heterogeneous / site | 2428 | 0.02631579 | 1520 | 1.52587891 | 0.02422030 |
| heterogeneous / pooled | 7284 | 0.00877193 | 4560 | 1.03027344 | 0.01635355 |
| small600 / site | 200 | 0.25000000 | 160 | 4.20898438 | 0.08417969 |
| small600 / pooled | 600 | 0.10000000 | 400 | 2.72949219 | 0.04549154 |

The two-site envelope changes per-site sampling, calibrated noise and sequential optimization steps together; it does not isolate a noise-only effect. Fixed unit weights remain equal patient weights for these equal-size partitions.

The heterogeneous arm sorts the same training patients by age before partitioning. This also reorders the pooled input, so its deterministic nonprivate minibatch path differs from the full arm. The pooled comparator is matched within each cell; the cross-arm comparison does not isolate site heterogeneity alone.

At epsilon 8 the selected three-site route changes mean C by +0.046048 relative to historical h06. Its pooled-DP minus federated gap is 0.001703, compared with the historical 0.025803. The selected pooled nonprivate fit reaches 0.641773, so the historical 0.621839 ceiling should be interpreted as schedule/grid specific. The two-site envelope changes federated C by +0.000617 relative to the selected three-site route. Configuration and runtime changes prevent treating these cross-version differences as an isolated causal effect of any single lever.

At 200 patients/site, federated C is 0.572795 versus pooled-DP 0.609357; the gap widens to 0.036562. This is below the 0.60 absolute floor and retains a small-cohort boundary for this configuration. The same frozen public grid is used; population changes alter both calibrated noise and sequential update counts.

**Three-site verdict: PASS.** Mean epsilon 8 C=0.641160; null=0.500000; required C≥0.600000 and C≥0.550000. Split-replicate 95% t interval: [0.632008, 0.650313]. The two-site arm is a patients-per-site envelope and cannot substitute for admission of the three-site route.

Held-out NLL, paired central/federated gaps, sample SDs, and Student-t 95% intervals are in `summary.json`. Only three overlapping split replicates support these intervals. NLL/K is not comparable between grids or likelihoods. Ranking performance does not establish probability calibration.

## Execution incidents and validation

The preprovisioned pod lacked dsBase because Matrix 1.4-0 blocked lme4. Updating the missing R dependencies in the task library restored dsBase 6.3.5; canonical install/freeze succeeded. No package source was changed. Five campaign checks, 30 survival-contract tests, 102 DP-safety checks, runner byte synchronization and tag source fingerprints passed.

One development attempt (g01/1102) failed at SuperNode startup, before server training submission, with no model or scores. Its unchanged failure record and redacted diagnostics are in `failures/`. The exact low-level cause is unavailable; shared spawn-lock contention is plausible, not established. Seven successful first-wave cells were reused. Only the failed pre-training attempt was recovered, once, in a new directory. Subsequent launches are spaced 30 seconds and each federation receives one CPU by OS affinity; the trusted launcher strips OMP/BLAS environment settings, so those alone did not limit the original workers. The grid, split data, privacy mechanism and selection rule stayed fixed.

A synthetic artifact-copy path error occurred after successful fitting/scoring; the existing model was exported without retraining. Unit-test import resolution and a NumPy alias in the test expectation were corrected before cohort fitting. Before confirmation, the pooled adaptive-strategy eta_l setting was corrected to the local learning rate to match the R API; development fits and scores do not use pooled aggregation, and the synthetic pilot used FedAvgM. All 12 strategy/learning-rate settings were checked against the installed R API; see `strategy_parameter_audit.json`. All 18 confirmation cells were validated and all four model artifacts per cell were independently reloaded/rescored. The 40 successful development models were also archived and independently reloaded/rescored. No confirmation fit was repeated. Details are in `artifact_verification.json` and `infrastructure_investigation.json`.

## Provenance

- Packages: dsFlower 0.5.0 and dsFlowerClient 0.5.0. 101 server and 105 client source files match tag v0.5.0; full fingerprints and installed commits are in `provenance/`.
- Runner SHA256: `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
- Pod: `x0w6ewmpinpsuk`, Ubuntu 22.04, 32 available CPUs, 64000000000-byte memory limit, CPU only; left running.
- Runtime: R 4.6.1, Python 3.11.10, Torch 2.4.1+cpu, Opacus 1.5.2, Flower 1.31.0. Full Python/R versions are archived.
- SUPPORT2 SHA256: `9da794bbd5c3a6a816e677cc17535e58c122d9ef4cbefd404489330a9f9cd2de`; UCI source and license attribution retained in each cell.
- Development seeds 1101/1102; confirmation 1101/1102/1103. All three outer train/test file hashes reproduce the historical splits exactly.
- `preregistration.json`, `frozen_configurations.json`, `selection.json`, confirmation start/completion records and per-cell hashes bind the execution order and settings.
- `diagnosis_before_training.md` preserves the exact initial summary bytes bound by the preregistration diagnosis hash.
- Historical v2 ran on CUDA A40 with package labels 0.4.5/0.4.4 and runner `ac08384…e4ac8`; current 0.5.0 source and CPU runtime are recorded separately. Cross-version changes are not a controlled device-only comparison.
- Each reported epsilon is a per-fit contract, not a privacy budget for the public development/confirmation campaign as a whole. Private-data selection would compose; data-derived private quantile grids would need their own accounted release.
- Only public evidence and whitelisted released models are archived. No node secrets or staged private manifests are exported.
