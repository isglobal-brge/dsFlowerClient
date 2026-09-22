# Hazard v3 development and one confirmation

## Frozen scope

Only SUPPORT2 UCI 880, checksum and encodings from the unchanged
`../prepare_data.py`. Both installed packages are 0.5.0; package source,
registry, runner, loss, privacy settings and existing evidence are unchanged.
The training mechanism remains epsilon 8 per development fit, delta 1e-5,
patient-level norm-1 clipping, replace-one adjacency, secure Poisson draws,
and audited full-horizon accountant. Development selection does not have a
single epsilon-8 guarantee on private data. All inputs here are public.

Historical full hazard h06 has mean C 0.595112 versus pooled-DP 0.620915.
The first summary section records the numerical/code diagnosis before training.
Public-bound standardisation is already performed by the runner; it remains
active in every candidate. Applying the same affine transform twice is not a
new intervention. Smaller K and event quantiles address weak interval signal;
optimizers, schedules and aggregation address the shorter sequential site
trajectory. The K-head linear architecture is retained, with no shared-slope
or baseline initialization changes.

## Development

Outer splits exactly reproduce SHA256(seed:id) 80/20 using seeds1101/1102/1103.
Their files are frozen at ingestion before any fit. Development opens only
outer site training CSVs and manifests, never an outer test CSV. Within each
site sort SHA256(hazard-v3-inner:seed:id), first floor(.8*2428)=1942 train,
486 validate. Each development fit has 5826 training and 1458 validation
patients. Only seeds1101/1102 are scored for selection. Seed1103's inner
training file fixes its grid before any confirmation.

`protocol.py:grid()` declares 35 ordered configurations, each with two seeds.
K in 5/8/10/12, quantile edges from that seed's inner-training observed events
(time1..1825); endpoints0/1825. Linear interpolation, and strict increasing
edges required. One h06 equal-width baseline remains. The same edges are
retained for that seed at confirmation, including all cohort-size arms.
These public-benchmark training-derived grids are public design constants;
private-cohort event quantiles would require a separately accounted mechanism
and are not authorized by this protocol.

All candidates use supplied public bounds scaled to[-1,1], no learned moments.
SGD momentum0 or Adam beta1 .9/beta2 .999/eps1e-8/amsgrad false; weight decay0,
no scheduler. FedAvg, FedAvgM(server LR1, momentum.9), FedAdam(default .1),
FedYogi(default .01), with other public package defaults. FedAvgM momentum is
explicit because its package default0 is equivalent to FedAvg.

Four synchronous federation slots, one thread per numerical library. Candidates
run in ordered waves of four complete configurations (two seeds each). A
wall-clock cutoff may omit an unstarted suffix, never remove a completed
candidate. Cutoff is fixed before launch and recorded; all started pairs must
finish. Any training failure or missing/nonfinite score aborts selection;
infrastructure defects discovered before scored fitting may be repaired with
the failed attempt retained. No outcome-driven retry. Development omits pooled
twins because selection uses only actual three-site federated-DP C-index.
Full scores, candidate order and omissions are archived.

Select highest arithmetic mean inner-validation C-index over the two seeds;
exact ties: fewer total epochs, smaller K, ascending ID. Freeze selection JSON
and all per-seed config hashes before opening any outer test file. No threshold
or pooled score affects selection. No additional candidates after selection.

## Confirmation

Exactly one pass, selected configuration, seeds1101/1102/1103. Full three-site
(2428/site) epsilon1/4/8, mandatory two-site(3642/site) epsilon8. Include
heterogeneous age-contiguous three-site and small600(200/site) epsilon8 if
remaining time permits; decide their inclusion before opening test data.
All use the same outer test patients per seed and the frozen per-seed grid.
Every cell includes pooled-DP, pooled-nonprivate and covariate-free null twins.
Twins use the exact local optimizer and reset it each round; adaptive server
aggregation uses the same installed Flower class with one pooled endpoint.
Pooled sample rate/step count necessarily differ. DP secrets are unrecorded.
Nonprivate removes clipping/noise but preserves expected-batch divisor.

Three-site admission at epsilon8 requires mean C>=.60 and >=mean null+.05.
Two-site and other stress arms are envelopes only. Report C/NLL, matched gaps,
SD and Student-t95% intervals across overlapping split replicates, with the
limited precision disclosed. NLL/K values are not comparable across grids.

This is the THIRD confirmation of the hazard contract, after v1 and v2 h06.
The holdouts have been used historically, so this is not independent validation.
No confirmation result triggers further tuning, a repeat fit, or another pass.
Missing confirmation results are unavailable; a utility FAIL remains FAIL.

Only additive tooling/evidence are committed on evidence/representative-cells.
Leave the dedicated hazard pod running. Export only public evidence and released
models, never node secrets, staged training manifests, or authentication files.
