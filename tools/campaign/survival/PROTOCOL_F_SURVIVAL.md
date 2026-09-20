# F-SURVIVAL preregistration v1

Frozen 2026-09-18 before any scored survival run. Implementation correctness
pilots may use synthetic data. This protocol is not evidence of execution.

## Data and splits

- Primary: SUPPORT2 UCI 880, release metadata last updated 2024-09-09,
  https://archive.ics.uci.edu/static/public/880/data.csv,
  SHA256 `9da794bbd5c3a6a816e677cc17535e58c122d9ef4cbefd404489330a9f9cd2de`.
  Attribution: Frank Harrell, SUPPORT investigators, 1995,
  DOI 10.3886/ICPSR02957.v2. CC BY 4.0 is explicitly listed at
  https://archive-beta.ics.uci.edu/dataset/880/support2; the current non-beta
  page delegates licensing to the original source. Preserve this distinction.
- Secondary: TCIA NSCLC-Radiomics, Lung1 clinical version 3 October 2019
  (clinical file linked by collection version 4, 2020-10-22),
  https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv,
  SHA256 `132f72b58b9660bf5e6b24b9817b335f1896360bc253e1f2034a3ffee593e6fd`.
  CC BY-NC 3.0, research/noncommercial use; attribution Aerts et al. 2014,
  DOI 10.7937/K9/TCIA.2015.PF0M9REI. No CT images needed.
- Stable public subject IDs are `id` and `PatientID`. No private fixture
  counts or tensors enter reports. All campaign counts refer to these public
  releases. Preserve every source row, including invalid targets.
- Seeds 1101, 1102, 1103 define SHA256 subject order: hash UTF-8
  `seed + ':' + id`. First floor(0.8 N) train; rest test. No stratification
  using outcomes. Lock every split and its hash before scoring any cell.
- Assign ordered training subjects cyclically to three disjoint DSLite sites.
  Small SUPPORT2 subset: first 600 training subjects, same held-out test.
  Heterogeneous stress: sort training subjects by baseline age (ID breaks
  ties), divide into three contiguous groups, use the same test set.
- No development split or model selection. No choice depends on scored
  outcomes. Failed cells and utility floors remain in the record.

## Baseline covariates and preprocessing

SUPPORT2 uses only `age`, `sex`, `num.co`, `diabetes`, `dementia`, `ca`,
and `dzgroup`. Age bounds [0,100]; comorbidity count [0,10]. Binary male,
diabetes and dementia in [0,1]; cancer indicators `yes`, `metastatic`;
fixed disease indicators ARF/MOSF w/Sepsis, CHF, COPD, Cirrhosis,
Colon Cancer, Coma, Lung Cancer, MOSF w/Malig (all [0,1]). Missing or unknown
categorical values give all-zero indicators; missing continuous values use
the public range midpoint before the runtime's fixed scaling to [-1,1].
No day-3 physiology, scores, `hospdead`, `slos`, `sfdm2`, charges, length
of stay, clinician prognosis, DNR timing or post-entry ADL fields.

LUNG1 uses age [0,100], clinical T [0,4], N [0,3], M [0,1], male [0,1],
fixed histology indicators large cell, squamous cell carcinoma,
adenocarcinoma, nos [0,1]. Missing numeric values use public midpoints.
No observed extrema, means, quantiles or fitted private encoding.

Targets are ordered time,event: SUPPORT2 `d.time`,`death` (study entry);
LUNG1 `Survival.time`,`deadstatus.event` (treatment start). Units days,
event 1=death, 0=right censoring. Minimum resolution 1 day, horizon 1825
days, AFT time scale 365 days. Finite time >1825 is administratively censored;
event exactly at 1825 remains an event. Invalid time/event/duplicate subject
uses a safe placeholder with valid=0 and remains in the subject denominator.

## Models, training and matrices

- AFT Weibull shape 1; AFT lognormal sigma 1. Scalar linear predictor.
- Hazard linear head with 16 periods; edges
  `[0,7,14,21,30,45,60,90,120,180,270,365,540,730,1095,1460,1825]`.
  Event intervals are `(left,right]`; censoring includes completed periods
  only. Subject loss is masked BCE summed over periods /16.
- Fixed public initialization seed 0, exact same declarative head and
  initialization in each twin. Replicate conditions differ by split seed.
- SGD learning rate 0.05, momentum 0, weight decay 0, no scheduler,
  batch size 128, 10 federation rounds, 2 local epochs per round.
  This schedule is fixed before scoring; it is not a tuned optimum.
- Custodian patient privacy, replace-one adjacency, epsilon {1,4,8},
  delta 1e-5, clip 1. No analyst privacy controls. Existing secure Poisson
  sampler and noise, replace-one conversion, full horizon accounting and
  sticky retries are unchanged. Each subject is sampled/clipped once.
- Full SUPPORT2 and LUNG1: all three models × three epsilons × three seeds.
  Nested SUPPORT2-600: same matrix. Heterogeneous full SUPPORT2 stress:
  three models × epsilon 8 × three seeds.
- Three actual isolated DSLite workers, no simulated FedAvg substitutes.
  Two-round three-node synthetic checks precede cohort scoring.
- Each split/model has a pooled nonprivate exact architecture/loss/scale/
  initialization/optimizer/schedule twin and pooled DP diagnostic twin.
  Pooled epochs = rounds × local epochs; site and pooled step counts differ
  with their declared N. Report this inherent trajectory difference.
- Covariate-free null: same loss and schedule with zero covariates. Constant
  risk yields C-index 0.5 when comparable pairs exist. No optional Brier in v1.

## Scores and floors

Channel B only: public held-out C-index and mean original-time AFT NLL
(including event log(time_scale) Jacobian), or mean discrete NLL /K.
Risk is -mu for AFT; negative left-endpoint restricted mean survival
sum_j (b_j-b_(j-1)) S(b_(j-1)) for hazard. Curves may cross.

Concordance rule: include pairs only when one subject has a strictly earlier
observed event than the other's time; exclude all tied observed times;
higher risk for earlier event is concordant; exact risk ties receive 0.5.
No comparable pairs => unavailable, never fabricated 0.5. Invalid held-out
records do not contribute metrics; report evaluation denominator only for
these public releases. No test-cohort censoring estimator is required.

Primary utility floor at epsilon 8 for each SUPPORT2 model: mean replicate
C-index ≥0.60 AND ≥mean null+0.05. Report each failed model as failed; no
post-hoc best-model pooling. Report per-replicate values and two-sided 95%
Student-t confidence intervals over three replicate scores and matched gaps.
These intervals describe split variation; overlapping splits are not an
independent clinical sample and three replicates give limited precision.

Four empirical envelope diagnostics (not privacy proofs):
1. Positive shortfall G=U_central-U_federated. This reverses the original
   article's delta sign. Flag G_next-G_previous > max(adjacent gap SDs).
2. Primary epsilon-8 floor above.
3. Small-N gap SD at epsilon 8 ≤ gap SD at epsilon 1, otherwise flag.
4. Per replicate flag N_train*epsilon<2000, U_central<0.95 and |G|<0.005.
   Also report a separately labelled minimum-site-N companion diagnostic.
Investigate flags with sigma/q/steps, subject denominator and artifact hash.
Do not delete replicates or relax budgets/clip/floors.

## Evidence and gates

Executed JSON schema v1 lives in dsFlowerClient/inst/extdata/campaign/survival/.
Record executed/failure status and UTC times, package commits/versions,
runner hashes, dataset release/hash/licence, protocol hash, split hashes,
source rows and N per site, K, settings, effective accountant/sigma/q/steps/
expected-batch divisor per site, dependency/device versions, model checksum,
replicate scores/CIs, envelopes, exact-twin differences, runtime/memory and
cleanup. Never store node secrets or private fixtures. Missing results are
explicitly unavailable, not placeholder scores. Scored artifacts must come
from successful enforced-DP runs. Mechanism/authority/staging/sticky/API
gates in design §6.1 items 1–7 are blocking; Claude decides promotion.

Real private-cohort experiments with different models/splits compose.
Public benchmark execution does not grant unlimited private experimentation.


# Preregistration v2 — discrete hazard only (2026-09-18)

Frozen before the v2 development sweep. AFT retains v1 and its passing
results; all v1 hazard results and failures remain historical evidence.
`PROTOCOL_F_SURVIVAL_V1.md` preserves the original protocol bytes/hash.
The existing test results motivated this new protocol; they are NOT new
untouched test data. V2 is a follow-up evaluation on reused outer holdouts,
not independent validation of a post-v1 hypothesis. No outer-test score
enters configuration selection or triggers retries or further tuning.

Reduction rule, fixed before execution: keep a linear head (hidden layers
hurt the pooled probe), fix K=10 (slightly better than K20 in inner validation),
and nominal batch=64 (the successful pooled probe setting). Retain all three
requested schedules despite the epsilon1 shorter-schedule finding, because
it does not settle epsilon8 federated behavior; retain both learning rates.
Six configurations remain from the requested 24-way grid:

| ID | Rounds | Local epochs | Total epochs | Batch | LR | K |
|---|---:|---:|---:|---:|---:|---:|
| h01 | 10 | 2 | 20 | 64 | 0.02 | 10 |
| h02 | 10 | 2 | 20 | 64 | 0.05 | 10 |
| h03 | 20 | 2 | 40 | 64 | 0.02 | 10 |
| h04 | 20 | 2 | 40 | 64 | 0.05 | 10 |
| h05 | 10 | 4 | 40 | 64 | 0.02 | 10 |
| h06 | 10 | 4 | 40 | 64 | 0.05 | 10 |

K10 edges are public equal-width days: [0,182.5,365,547.5,730,912.5,
1095,1277.5,1460,1642.5,1825]. Keep loss /K, horizon1825, fixed bounds,
SGD momentum0, weight decay0, no scheduler, initialization0, clip1,
delta1e-5, replace-one patient privacy and the audited secure mechanism.
The v1 runner/protocol use LR0.05, not the LR0.02 stated in the task context.

Development uses epsilon8 only, seeds1101/1102/1103 and actual three-site
federation for every candidate. For each of the twelve existing outer
training splits (SUPPORT2 full/small600/heterogeneous and LUNG1 full × three
seeds), sort each existing site's training subjects by SHA256 UTF-8
`hazard-v2-inner:seed:subject_id`; first floor(0.8*site_N) stay in training,
remainder become inner validation. Pool validation across sites for scoring.
Preserve the outer site's assignment, including age-contiguous stress sites.
No outer test file is opened in development. Fixed preprocessing uses no
fitted statistics. Hash all development files before training. This gives
72 development federations (6 × 4 × 3), scheduled in two slots.

Selection: highest arithmetic mean federated-DP inner-validation C-index
over the twelve equally weighted split/subset cells (four settings × three
seeds). Exact numeric ties: fewer total epochs, then smaller K, then ascending
configuration ID. All twelve finite scores are required for every candidate;
any failed/missing/nonfinite development cell aborts selection, without
score-driven retry, candidate removal or fallback. Twins/null are recorded
but never used for selection. Persist all scores, means, hashes and the
winner to `hazard_v2_selection.json` before opening any outer test file.

Refit the single winning configuration from initialization0 on each entire
outer training split. Confirmatory matrix: SUPPORT2 full9, small6009,
heterogeneous3 (epsilon8), LUNG1 full9; other groups use epsilon1/4/8,
all seeds1101/1102/1103. Each of the30 cells includes matching pooled-DP,
pooled-nonprivate and covariate-free null twins with the same selected
architecture, optimizer and total schedule. Recalibrate full-horizon noise
for each population, epsilon and schedule; never reuse v1 calibration.
All v1 metrics, epsilon8 primary floor, intervals and diagnostics remain.
Report ranking versus NLL separately; /K NLL is not comparable across grids.
Repeated fits and selection compose on private data; epsilon8 is per run,
not a privacy guarantee for this public-data sweep as a whole.

Write v2 runs/splits/logs under `runtime/hazard_v2`, development evidence
under `inst/extdata/campaign/survival_hazard_v2_development`, confirmatory
evidence under sibling `survival_hazard_v2`, and selection under
`runtime/hazard_v2/hazard_v2_selection.json`. Never replace v1 evidence.
Run the hazard-only summariser after all confirmatory attempts, retain
failures, and require reviewer investigation/promotion. No automatic retry
or new tuning after selection. Driver exits after summary (nonzero if failed).
