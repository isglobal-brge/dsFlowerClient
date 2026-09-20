# F-SURVIVAL decisions

| ID | UTC date | Decision and rationale |
|---|---|---|
| D001 | 2026-09-18 | Start from reviewer-approved main commits `4b8dcbf` / `01280c2`; no rebase needed because both branches already pointed there. Preserve feature-view consumption. |
| D002 | 2026-09-18 | Survival only, additive shared-file edits; no segmentation registration and no equality assertion fixing total catalogue at 30. Existing 28 names/semantics form the compatibility baseline. |
| D003 | 2026-09-18 | Transport public survival config as canonical base64 JSON under `survival-config-b64` because Flower run-config values are scalars. Server validates and authors the structured manifest `survival-config`. This grants no privacy controls. |
| D004 | 2026-09-18 | Initial fixed public dispersion grid {0.5, 1, 2} and bounded predictor location ±10; enforce joint Weibull exponential bound ≤60. Fixed grid implements approved small public grid without learned global parameters. |
| D005 | 2026-09-18 | Preserve M source rows and N subjects separately. Derived subject tensors are a separate local artifact; invalid or duplicate subjects contribute zero loss with an unchanged sampling denominator. |
| D006 | 2026-09-18 | Baseline full R suites plus existing DP Python suites precede new gates; record baseline failures rather than attributing them to survival. |
| D007 | 2026-09-18 | Use only baseline demographic/comorbidity/disease SUPPORT2 fields; omit day-3 physiology and derived prognostic scores to avoid changing time origin or conditioning on survival to assessment. Public fixed encodings/bounds only. |
| D008 | 2026-09-18 | Fixed survival initialization seed 0 in a scoped Torch RNG context, identical across central/federated models. Replicate split seeds vary public experimental conditions; fixed initialization avoids claiming independent sticky retries. Existing contracts retain their initialization behavior. |
| D009 | 2026-09-18 | Protocol freezes 16 hazard periods, dispersion = 1 for both AFT distributions, linear heads and SGD schedule. No development/tuning on scored outcomes; all floor failures will remain visible. |
| D010 | 2026-09-18 | UCI non-beta metadata delegates licensing, while official UCI beta page explicitly lists CC BY 4.0; preserve both provenance facts and pin downloaded release hash. TCIA clinical file is version 3 linked from version 4 collection. |
| D011 | 2026-09-18 | Public held-out concordance excludes every observed-time tie and gives exact risk ties 0.5. Invalid evaluation records may be excluded only in channel B on the public fixtures, with explicit denominator. Training never drops them. |
| D012 | 2026-09-18 | Pod R is unavailable before `R_STACK_DONE`; use isolated local R library and Python runtime rather than modifying shared pod bootstrap/system libraries. All final versions must reflect executed environment. |
| D013 | 2026-09-18 | No new private diagnostics endpoint. Campaign will derive mechanism diagnostics from its public subject manifests and exact audited runtime function, label them as recomputation, and verify PRV epsilon/delta without budget slack. |
| D014 | 2026-09-18 | Survival HPO training rejects within an explicit objective context before transport. Existing released-model public scoring remains possible; no brittle objective-closure inspection. |
| D015 | 2026-09-18 | Original-time AFT NLL includes density Jacobian. Exact twins use zeroed features for invalid subjects and the same fixed bounds, loss, initialization, optimizer, and total epochs; disclose pooled-vs-federated q/steps and no clipping/noise in nonprivate twin. |
| D016 | 2026-09-18 | Full-precision survival JSON/CSV and explicit one-element arrays prevent interval-boundary drift and scalar/vector coercion; existing contracts' serializers unchanged. |
| D017 | 2026-09-18 | Canonical survival class/label pins at 2 close an irrelevant-field sticky-noise variation; reject malformed values or values other than 2 before private access. |
| D018 | 2026-09-18 | Record exact installed build commits/hash separately from source tooling HEAD. Await install completion and restart workers before using a new build. Archive pretraining and unavailable-model failures without invented scores. |
| D019 | 2026-09-18 | Coherent lower-version test uses image Torch 2.4.1 with Opacus 1.5.2. Opacus 1.6 requires Torch >= 2.6; do not force an incompatible dependency combination. The task-specific pod environment remains separate from segmentation and the shared virtual environment. |
| D020 | 2026-09-18 | Flower startup-token failures are investigated separately from model/staging errors. No token lifetime, privacy threshold or DP default has been relaxed. |
| D021 | 2026-09-18 | Independent PRV verification is labelled separately from runtime's PRV-then-RDP calibration. Device evidence records the actual twin device and CUDA availability separately. |
| D022 | 2026-09-18 | Export failures after successful training recover from the identical released model; never rerun private training merely to repair local scoring/export. Preserve failed export record and annotate recovery. |
| D023 | 2026-09-18 | Exact central CSV ingestion uses round-trip floating conversion; no score-driven hyperparameter changes. Cohort execution waits for all three complete synthetic evidence cells. |
| D024 | 2026-09-18 | Campaign completion requires exactly 90 executed cohort cells, three synthetic cells, and 12 matched-seed groups; all flagged empirical envelopes require an artifact/accounting review, while failed preregistered floors remain failed. |
| D025 | 2026-09-18 | Schema v1 fixes days/from-baseline, requires an explicit AFT horizon, permits public time fields in [1e-6, 1e6], and derives hazard horizon from strictly increasing edges beginning at 0. Hazard rejects irrelevant AFT fields. This avoids ambiguous time origins and unused sticky pins. |
| D026 | 2026-09-18 | Explicit ordered (time, event) targets and baseline features must be disjoint from each other and patient identifier; reserve `__survival_` names. Reject target recoding/bounds and derived-artifact overrides at public preflight. |
| D027 | 2026-09-18 | Canonical subject order is first appearance; duplicate IDs or the shared missing-ID sentinel invalidate the whole subject. Source M and unique canonical N remain intact. Descriptor/parquet routes use the same assembler and Python independently reconstructs it. |
| D028 | 2026-09-18 | Local response/median returns time; hazard medians beyond horizon map to NA, with piecewise grid survival and negative left-endpoint RMST risk. Portable bundles retain authoritative metadata in existing sibling assets, avoiding a new format. |
| D029 | 2026-09-18 | Initializer uses scoped seed 0 only for new survival tasks; secure DP secrets/noise remain independent and unrecorded. No old-task randomness is altered. Differences in pooled versus site sampling geometry and FedAvg equal weights are disclosed in every exact-twin result. |
| D030 | 2026-09-18 | Validate executed evidence from observed versions, installed paired commits, runner hash, released artifact hash and strict recomputed accounting. Tooling provenance may differ from installed package revision; synthetic exporter recovery explicitly records the postprocessing script hash. |
| D031 | 2026-09-18 | Strengthen the subject-locality staging gate by changing a valid subject; an already-invalid duplicate cannot establish useful locality. This is a test correction, with production contracts frozen. |
| D032 | 2026-09-18 | Report confidence intervals for C-index and held-out NLL for all four scored branches, plus paired central-minus-federated gaps. RSS explicitly identifies native units and central/scoring process scope; federation peak memory was not measured. |
| D033 | 2026-09-18 | Limit local cohort scheduling to one cell at a time after concurrent synthetic startup showed substantial host contention. Keep all preregistered model, sample, privacy and training settings unchanged. |
| D034 | 2026-09-18 | Preserve sanitized public-campaign CLI/SuperLink/node diagnostics before cleanup when fit throws. The original error propagates unchanged; diagnostic failure cannot mask it. The existing runtime status 1 synthesizes multiple failure paths and is not proof of an actual CLI exit or a specific cause. |
| D035 | 2026-09-18 | Qualify `utils::tail` to remove newly introduced R CMD check namespace notes: server `00b09a0` and client `fd61a94`. This does not alter staging mathematics or public configuration values. Each pilot still identifies the exact frozen installed build; a source fix is not retrospectively attributed to an older execution. |
| D036 | 2026-09-18 | Add focused survival Python CI with the tested campaign core profile: Python 3.11, Torch 2.14.0 CPU, NumPy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, Flower 1.31.0, cryptography 46.0.7, Opacus 1.6.0, and SciPy 1.17.1. Server/client workflows cover runner and relevant API/helper/test edits. Manual paired review requires a full server SHA and compares the actual checked-out trees; the existing main-branch comparator is unchanged. Local tests passed; no hosted CI execution or remote push is claimed. |
| D037 | 2026-09-18 | Keep archive schema validation separate from campaign completion. Historical failed attempts remain valid records with errors and no scores; they never substitute for executed cells. The completion validator recomputes all summaries/CIs/envelopes from the exact matrix and requires a reviewed, nonempty explanation for each flagged heuristic. Unit-test records are constructed only in memory and never archived as empirical evidence. |
| D038 | 2026-09-18 | Run package checks in isolated copies with `--no-manual --no-tests --no-build-vignettes`; full regression and vignette generation are recorded separately. Preserve snapshot attribution: pre-fix tail notes are not clean checks of later source. The client native-connection note is preexisting because `src/` is unchanged from `01280c2`; do not expand this task into an unrelated C API change or launch another baseline R stack during host contention. |
| D039 | 2026-09-18 | Distinguish passing historical tests from pending edits. The R evidence-schema result of 222 assertions predates nine added topology/calibration/memory assertions and the two later synthetic failures. Those assertions remain pending until the next targeted R run; successful Python validation of the recovered cell is recorded separately. Preserve summary-reporter baseline outcomes without inventing exact expectation counts from dots. |
| D040 | 2026-09-18 | Migrate execution to the pod only after observing R_STACK_DONE and verifying actual dependencies. Record R4.6.1 rather than the expected4.5; preserve a separate survival R library and Python environment. |
| D041 | 2026-09-18 | Exact twins train on the same cuda-if-available device policy as the trusted DP loop. Reuse its deterministic setup, retain TF32 defaults and record them; evaluate all models on CPU. Hardware migration does not change the frozen model/split/privacy protocol. |
| D042 | 2026-09-18 | Verify the actual train, test, site and protocol file hashes before live execution, and independently verify train/test hashes before central parsing. Transported manifests alone do not establish which inputs were used. |
| D043 | 2026-09-18 | The pod workspace FUSE mount cannot enforce directory permissions and an in-place tmpfs mount was denied. Use an owner-only task-specific OS temporary directory through workspace runtime links for ephemeral node secrets/staging. Keep persistent project inputs/outputs in the survival workspace; do not bypass any privacy filesystem check. |

Implementation/test corrections and executed evidence are detailed in [SERVER_F_NOTES.md](SERVER_F_NOTES.md), [CLIENT_F_NOTES.md](CLIENT_F_NOTES.md), and [RUNNER_F_NOTES.md](RUNNER_F_NOTES.md). These records preserve caught failures, stale in-flight test snapshots, and dependency/runtime variances without treating them as clean passes. They form part of this decision record.

| D044 | 2026-09-18 | Defer runtime discovery optimization after successful live hazard execution. Preserve the tested installed core and record the measured startup cost; it does not justify changing unrelated capability discovery during the campaign. |

| D045 | 2026-09-18 | Reopen Gate6 when adversarial review demonstrates a new noise stream through computationally inert wire aliases. Fix survival-only seed canonicalization and test accepted equivalents before cohort execution; earlier passing tests cannot override the counterexample. |
| D046 | 2026-09-18 | Choose concurrency from actual container CPU/RAM quotas, not host totals. Runtime placement may be adjusted within task-owned storage to reduce FUSE overhead; model, privacy, split and utility-floor pins remain fixed. |

| D047 | 2026-09-18T12:45:24.561192+00:00 | Resumed: retain protocol v1 unchanged. The reviewer permits v2 but does not require it; preserving the frozen schedule avoids mixing protocols. Review and finish the interrupted sticky canonicalization tests before new scoring. |

| D048 | 2026-09-18T12:47:48.538350+00:00 | Canonicalize only the three demonstrated accepted inert survival wire aliases. Distribution, public grids, effective bounds/tensors and all effective training pins stay bound; existing 28-task seed semantics remain unchanged. Retain successful pre-fix pilots with historical build identities and verify the fixed runner separately. |

| D049 | 2026-09-18T12:50:15.817493+00:00 | Present lognormal as the introductory article example and explicitly attribute the ranking-versus-NLL caveat to reviewer exploratory probes, not release evidence. No model selection or protocol change follows from campaign outcomes. |

| D050 | 2026-09-18T12:53:28.835703+00:00 | Activate the already copied task-owned Python environment on POSIX temporary backing only after a full content/symlink comparison (9400 entries, tree SHA256 376414d74d239effb349604972fc1b10fb4151539092d3fd763de88391243034). Preserve the original FUSE environment as pyvenv-fuse-preserved and the same lexical runtime paths. System packages, package contents, training/privacy settings and source commits are unchanged; rerun runtime readiness before scoring. This reduces measured FUSE traversal overhead without modifying runtime discovery code. |

| D051 | 2026-09-18T13:02:10.156254+00:00 | A measured R metadata walk takes82.117s through the FUSE alias and0.133s through its canonical POSIX path, returning the same site-packages directory. Pause only matrix scheduling after the active cell; resolve the campaign custodian environment root and place runtime/server itself on POSIX backing at the cell boundary. This is tooling/path correction, not a runtime discovery algorithm, training schedule or privacy change. Client64b0784 changes only run_cell.R path resolution and passes R parsing. |

| D052 | 2026-09-18T13:05:01.873845+00:00 | Interrupt the first cohort startup after more than10minutes with no observed SuperNode process or released model, because independent measurement identifies severe layered-FUSE discovery overhead. Preserve the interrupted attempt as failed with no scores, clean up task-owned processes, then retry the same frozen cell in a fresh directory after canonical-root correction. This is an infrastructure retry, not score-based model selection or a privacy relaxation. |

| D053 | 2026-09-18T13:18:22.460193+00:00 | Read-only imports of trusted ClientApp and Opacus take27.598s through the FUSE interpreter prefix versus4.357s through its canonical POSIX prefix. At the next cell boundary, relocate only generated venv launcher shebangs to the verified canonical interpreter, keeping package modules/library contents and all training/privacy pins unchanged. Retain old environment, record each launcher before/after hash and separate relocation timestamp in installed-build provenance; pause scheduling rather than changing an active cell. |

| D054 | 2026-09-18T13:26:36.131369+00:00 | After a complete clean ten-round cell and resolved-path readiness/30 survival tests, cap pod campaign concurrency at2. Actual quota is7.65CPU and49,999,998,976bytes RAM, with18.54GB used and GPU memory idle at the boundary; original local concurrent failures are retained. Single-thread BLAS/OMP and isolated per-cell workers/ports/directories remain. This is an orchestration choice from observed container resources, not a model, budget or split change. Monitor new cells for infrastructure errors and stop scheduling if they recur. |

| D055 | 2026-09-18T13:38:13.528040+00:00 | Assign LUNG1 small-N cells to the authorized Mac and all SUPPORT2 cells to the pod, without splitting a cohort across hosts. The frozen protocol fixes model/split/privacy/training, not hardware; each host uses matched central/federated runtime/device policy and records actual versions. Add a dataset-only scheduling filter (`bfcf736`), leaving complete90-cell membership validation unchanged. Hold pod scheduling at the current pair boundary before restarting with SUPPORT2-only ownership; Mac remains single-cell concurrency under D033. Refresh and gate the local pair before any LUNG1 scoring. Cross-cohort platform differences must be disclosed, not attributed solely to N. |

| D056 | 2026-09-18 | Preregister hazard-only v2: fixed linear K10/equal-width horizon1825, batch64; six schedules/LRs and twelve inner splits per candidate. Deterministic mean-C selection with epoch/K/ID ties, fail closed; selected config refit across30 cells in two federation slots. V1/AFT evidence preserved; reused holdouts explicitly disclosed. Exact grid and rules follow. |


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
