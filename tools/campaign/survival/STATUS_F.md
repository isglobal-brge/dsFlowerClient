# F-SURVIVAL status

Current state at 2026-09-18T17:52:22.026126+00:00: **SUPPORT2 execution is complete (63/63); 27/27 previously executed Mac LUNG1 records were validated and copied to the pod.** Seven historical failed attempts remain recorded; none requires another training retry. The LUNG1 driver was submitted under nohup to reuse completed cells and regenerate the combined summary. No post-launch polling or completion claim; evidence/artifact assembly, envelope/floor review and pkgdown remain for the reviewer’s next handoff.

The milestone log is chronological. Statements that a run was “executing” describe that timestamp; later entries record its outcome. Historical and superseded test runs are distinguished from the clean regression and focused gates used for the current assessment.

## 2026-09-18T10:38Z — Start / baseline

- Scope: F1 → F2 → F3 → F5 survival → F6. No promotion claimed; Claude reviews increments.
- Both repositories were clean on `feature/survival`, at dsFlower `4b8dcbf` and dsFlowerClient `01280c2`. Main anchors were recorded in `DESIGN_F_CODEX.md` before implementation.
- `python3 dsFlowerClient/tools/check-runner-sync.py`: PASS, byte-identical baseline.
- Baseline full R tests started with `Rscript -e 'devtools::test("dsFlower", reporter="summary")'` and analogous client command; logs in logs/baseline-{server,client}-R.log. Python baseline selected DP safety/guards/runner suites started; counts pending completion.
- Local versions: R 4.5.2; Python 3.11; torch 2.10.0, Opacus 1.6.0, Flower 1.29.0. Required R packages available.
- Pod bootstrap had not yet printed `R_STACK_DONE`. No pod R invoked. Bootstrap log identifies R 4.6.1, a variance from requested 4.5; record actual version in evidence.
- Implementation delegated by ownership: trusted runner, server R staging, client R API/prediction. Root owns protocol, campaign, integration and reporting.
- Open questions: exact UCI release licence provenance is being verified (current landing page delegates licence to original repository). No scored cells run.

## 2026-09-18T10:45Z — F1 mechanism foundation

- Server F1 `f4ec8f9`, client mirror `ea4b4eb`: 13 focused unittest methods pass, including independent SciPy AFT references, every trainable parameter compared with single-subject autograd, one subject clip for periods, Cox and flattened-period negative controls, and public prediction semantics. Runner integration/staging/authority remain for F2/F3; no promotion claimed.
- Baseline existing DP safety: 102 passed, 0 failed. Guards/runner: 120 passed, 1 skipped, 111 subtests passed. The safety file is executable, not pytest-collectable (`sys.exit` at module scope); corrected invocation to direct script plus pytest for other suites, without code changes.
- Full baseline server and client R testthat suites completed without failures; the server reported three environment/platform skips. The original summary-reporter logs contain dots rather than structured expectation totals, so no exact baseline expectation count is claimed. Structured post-change totals are recorded separately below.
- Protocol frozen before scoring; SUPPORT2/LUNG1 exact source SHA256 and licensing URLs pinned. UCI beta explicitly lists CC BY 4.0. Download hashes verified. All 12 dataset/subset/split manifests generated before scoring via prepare_data.py. No scores computed.
- Open: pod R bootstrap still pending; integrated API/staging pilots next after agent commits.

## 2026-09-18T10:53:11.433014+00:00 — F2 / F3 integration progress

- F2 server AFT `d4e7136`, sub-resolution correction `6ba75ba`; 70 focused AFT expectations plus 437 regression passes / 2 platform skips. F2 client `4bb26a2`: 440 passing assertions. F2 runner server `1ce9dc3`, client `36447c7`: 20 focused Python tests pass, runner sync passes.
- F3 server `715c2b6`: focused survival 110 expectations, selected regression 477 passes / 2 Windows skips. R-to-Python public staging smoke passed both AFT variants and hazard.
- Gates still in progress: exact interval serialization, live federation, sticky release arrays, full regression and real evidence. Nothing promoted.
- Pod bootstrap terminated without `R_STACK_DONE` because dsBase and dependencies failed to install (missing system headers, including libuv/fontconfig); no pod R used. The dedicated local library `runtime/rlib` is the fallback.
- First local infrastructure pilot exposed root setup sequencing error: client readiness marker missing when pilot started, so package provisioner rebuilt the workspace venv. No scored result exists. Correcting readiness checks before further runs; baseline global interpreter unchanged.
- Campaign metrics unit tests: 4 passed. Independent held-out density Jacobian, risk/time tie conventions, hazard interval-end and reversed-sign envelope tests executed.

## 2026-09-18T11:14:36.884755+00:00 — F3 implementation complete / F5 execution / F6 regression

- F3 runtime and both R APIs/staging committed; the original 28 registry records unchanged against main. Final numerical serialization fixes preserve adjacent-double interval edges and one-feature vector bounds. Survival's otherwise unused class and label pins are fixed at 2; alternatives are rejected.
- Trusted runner frozen server `962281a` / client `36f5c73`. Local Torch 2.14 / Opacus 1.6 / Flower 1.31: focused survival 29, safety 102, guards 105, runner 16, validation 61, holdout 28, CV 24, and release 24 all pass; byte sync passes. The prior system Torch 2.10 profile also passed (one environment skip in guards).
- Pod isolated Python (R unused): Torch 2.4.1+cu124 / Opacus 1.5.2 / Flower 1.31 with CUDA: survival 29 pass; guards 104 pass / 1 vision skip; safety 102 pass. Logs copied to runtime/pod-verification. No shared /workspace/venv modification.
- Clean full server R regression: 1,542 passed, 3 existing skips, and 0 failures/errors/warnings. The later focused class-pin subset passed 370 assertions. Full client R regression: 372 cases, 1,933 assertions, and 0 failures/errors/skips/warnings. The separate earlier client Python helper regression passed 51 tests; this is not a claim that all 51 were rerun under the later pinned campaign profile. Public R prediction and portable RDS/JSON reload passed for Weibull, lognormal, and hazard.
- F5 initial three failed synthetic attempts archived without scores. Causes include missing campaign proxy symbol methods, root install/provision sequencing, stale installed one-feature bounds, and a separate possible Flower startup-token expiry. Root corrected lifecycle ordering and now records installed paired commits/hash via install_and_freeze.sh; no privacy guard changed.
- Frozen install at 2026-09-18T11:09:30Z: server `962281a`, client `19e8823`, runner hash `5f5754a49bef98bc4da5000c7117e88b7642c6270bf25e1ad338f465de6263d7`. At this milestone, the final synthetic Weibull run was executing across three live sites. No cohort score existed.
- Asked asynchronous clarification because user's explicit `R_STACK_DONE` instruction prevents repairing/using failed pod R bootstrap; independent local work continues. Pod R is not a completed dependency.

## 2026-09-18T11:23:01.320228+00:00 — F5 first successful live federation

- Three-site synthetic Weibull completed 2/2 rounds with 0 failures and released model SHA256 `9cb9d4176c2e7104bd834b55f3d7074cabc820ea46bd8ee82599c9b13930b6b2`. Package channel-B prediction passed.
- Export initially failed after federation because the processx timeout of 3,600,000 was interpreted as seconds and overflowed; corrected to 3,600 seconds. Archived failure, then scored the same released artifact without rerunning federation. Executed JSON and recovery provenance archived; no scores imputed. Independent full-horizon PRV gates pass.
- All three centralized synthetic exact-twin CLIs independently executed and inspected by runner auditor; finite scores, output widths 1/1/16, pooled and three-site accounting, and constant-risk null checks passed. Round-trip CSV parsing applied for exact preprocessing consistency. Artifacts runtime/central-smoke.
- Live lognormal/hazard synthetic runs executing concurrently. Pod bootstrap still lacks `R_STACK_DONE`; no pod R invoked.
- Independent audit found staging one-subject mutation test used already-invalid duplicate; strengthening with valid-subject changes without core runtime modifications.

## 2026-09-18T11:29:49.154389+00:00 — F6 independent gate audit

- Server test-only commit `d7fb0d4` closes the valid-subject staging-locality gap: 11 cases, 159 assertions, and 0 failures/errors/warnings/skips. Production runner remains unchanged.
- Client `d3dfaf8` adds the strict completion validator. Twelve Python tests pass, covering C-index/NLL confidence summaries for all four score branches, missing/duplicate/tampered-cell rejection, and honest failed floors. At this milestone, the R evidence schema passed 222 assertions against one executed cell and four failures. The completion gate correctly rejected the missing 90 cohort cells and two remaining synthetic cells.
- Root commit `9b1958c` records the corrected export timeout, exact central parser, complete 90-cell driver, expanded CIs, and recovered Weibull evidence. Four independent metric tests pass.
- `pkgdown::build_article("survival-segmentation", pkg="dsFlowerClient", new_process=TRUE, quiet=TRUE)` exited 0 and created `docs/articles/survival-segmentation.html`. Pandoc warns about its deprecated highlight-style flag (tool-generated invocation). Article will be rerendered after cohort evidence.
- Independent runner audit maps all seven blocking gates to executed tests; no additional production blocker identified. Detailed mapping and exact-twin/envelope audit in RUNNER_F_NOTES.md.
- Isolated server/client R CMD build/checks underway, no live runtime/library mutation.

| Blocking gate | Current result | Executed evidence |
|---|---|---|
| Independent mathematical losses | PASS | Python survival suite: both AFT distributions and all three public dispersions; hazard masked BCE, boundary/empty/invalid cases |
| All-parameter subject gradients | PASS | Every trainable parameter: Opacus versus separate single-subject autograd; one clip per subject; Cox and flattened-period negative controls |
| N/accounting independent of K | PASS | Focused suite plus existing DP guards, strict independent PRV and sampler/empty-step checks |
| Authority/egress | PASS | Server/client preflight, runner tamper rejection, fixed `num-examples=1` and availability-only replies |
| Staging censuses/locality | PASS | Actual R → Python staging for three variants, plus 159 focused R expectations including valid-subject mutation |
| Sticky semantics | PASS | Executed identical retries and operational rebinding; changed parameters/distribution/dispersion/grid/effective validity change arrays |
| Original 28/API compatibility | PASS | Exact registry records/formals/bodies against `01280c2`; full R/Python regression; portable RDS/JSON predictions |

These unit/API gates do not substitute for the remaining public cohort execution, evidence completeness or reviewer promotion.

## 2026-09-18T11:34:10.033665+00:00 — F5 concurrent hazard infrastructure failure

- The hazard synthetic attempt failed after 714.26 seconds with synthesized Flower runtime status 1, before a model was accepted or saved. Cleanup reported TRUE; no scores were produced. Archived as `failure-synthetic-hazard-concurrent-host.json`. The cause remained under investigation; observed host contention was not causal proof. Lognormal was still running at this milestone. No cohort cells had started.
- The isolated server R CMD check completed with 0 errors, 0 warnings, and 1 new NOTE: an unqualified `tail` call in survival configuration. An explicit `utils::tail` qualification was authorized in source, without changing the live installed runtime. Direct baseline comparison established attribution.

## 2026-09-18T11:42:23.560610+00:00 — F5 sequential recovery preparation

- The concurrent lognormal attempt failed after 1,108.12 seconds, with no released model or scores; cleanup reported TRUE. Two UNAUTHENTICATED messages and three tracebacks were observed in one task-owned node log during execution. Detailed logs were removed, so the final cause is not proven. Archived as `failure-synthetic-lognormal-concurrent-host.json`.
- A post-run process audit found no remaining task-owned Flower processes. Both failed attempts were retained without changing privacy, model, or training gates.
- `9ec468b` retains sanitized public CLI/SuperLink/node logs before the original error propagates and cleanup runs. `c3d9267` adds the survival article source and reserves a segmentation section. Server `00b09a0` qualifies `utils::tail`: the preceding full package check had 0 errors, 0 warnings, and 1 NOTE; the focused codetools check after the fix was clean.
- The paired local installation is being refreshed after all workers stopped. Subsequent sequential attempts use new directories and the same fixed configurations. Matrix gates consume archived successful synthetic records, preserving failed runtime attempts. The cohort matrix remains unstarted.

## 2026-09-18T11:50:21Z — F6 package checks, current evidence, and handoff

Package checks completed without errors or warnings. Their exact scope matters:

| Package snapshot | Executed check flags | Result and attribution |
|---|---|---|
| Server baseline, main `4b8dcbf` | `R CMD check --no-manual --no-build-vignettes --no-tests dsFlower_0.4.5.tar.gz` | Status OK: 0 errors, 0 warnings, 0 notes. Log: `runtime/checks/server-baseline/check.log`. |
| Server survival snapshot before `00b09a0` | Same flags as the baseline | 0 errors, 0 warnings, 1 NOTE for an unqualified `tail` call. Fixed by `00b09a07ebca2b7d8c058221bd4d539f0aed1a2d`; the focused codetools check after the fix was clean. Log: `runtime/checks/server-final/check.log`. This was not a full package recheck of the fixed source. |
| Client snapshot before `fd61a94` and the final reporting-only checks | `DSFLOWER_SKIP_PYTHON_SETUP=1 _R_CHECK_FORCE_SUGGESTS_=false R CMD check --no-manual --no-tests --no-build-vignettes dsFlowerClient_0.4.4.tar.gz`, with task-local `TMPDIR` | Exit 0: 0 errors, 0 warnings, 2 NOTEs. One note covers two unqualified `tail` calls, fixed by `fd61a944a5cdf60ee28e48e5dad1a336a6a2b877`. The other covers existing native `R_GetConnection` / `R_WriteConnection` calls: `git diff 01280c2 -- src` is empty, and `src/dsi_socket.c:25–26` matches main. No unrelated native-code change or extra baseline R check was made. Log: `runtime/checks-client/check.log`. |

- The client build command was `DSFLOWER_SKIP_PYTHON_SETUP=1 R CMD build source/dsFlowerClient` from `runtime/checks-client`. It passed, including vignette generation. The check also passed all four vignette R-code runs, Rd checks, documentation consistency, installation, and namespace checks. Tests and vignette rebuilding were deliberately skipped in the package check because regression and the successful build were recorded separately.
- Client tarball SHA256: `c1de3ad9a2c0ff3cb0c85366a79adc91f5506cd0fcb4120c2b498a9d4bb1bd4f`. Its check must not be described as a check of the later `fd61a94` source or the final reporting-only schema/tool changes.
- Namespace fixes preserve numerical and privacy semantics. Client `fd61a94` was committed at 11:43:41, before the refreshed client installation began at 11:44:11. The paired rebuild/readiness verification is still in progress at this milestone; the last completed `runtime/build.json` read during this audit remains the 11:09:30Z build recorded above.
- Focused CI was added in server `f727c3f` and client `8b94060`. The pinned campaign profile executed 29 canonical survival Python tests and 10 original-plus-survival prediction tests successfully. Requirements files match and runner trees are byte-identical. A manual paired review requires an exact server commit SHA; the original main-branch runner comparator remains unchanged. Hosted CI is configured, not claimed executed.
- The campaign metric/completion suite passed 12 tests after the C-index/NLL CI additions and again after the memory/topology checks. Commits `d3dfaf8` and `1f2dd1d` enforce exact matrix membership, matched seeds/splits, provenance, subject-level accounting, CIs, and reviewed explanations for flagged diagnostics. Failed utility floors remain failed results.
- The last executed R evidence-schema run passed 222 assertions against one success and four failures. **Nine subsequent R assertions for topology, calibration, and memory provenance remain pending**, and the two later failure records were not part of that 222-assertion run. The stricter Python per-cell validator did pass the recovered Weibull record after those metadata checks were added. No extra R process was started for this pending test during host contention.

Historical tests that must not be mistaken for current passes:

- The first full server post-change R run had 1,538 results: 1,534 passed, 3 skipped, and 1 failure in the exact public capability-list expectation. The expectation was extended additively for the three survival losses; no old capability was removed.
- A later in-flight server run loaded the namespace before the width-one bounds fix while sourcing the new tests afterward; its two failures are a stale snapshot. The clean `server-f6-clean-tests.*` run supersedes it: 1,545 results, 1,542 passed, 3 skips, and 0 failures/errors/warnings.
- An in-flight client full regression was stopped around the width-one bounds change and was not treated as a pass. The clean client run at `b66928f8df962f2ce7e039a50bc81b7dc2979779` has 372 cases and 1,933 assertions, with 0 failures/errors/skips/warnings. A public artifact smoke script also had an earlier trailing parse error after an in-flight edit; the unchanged clean rerun exited 0 and is the accepted result.
- The seven mechanism/API gates in the earlier table refer to the executed suites and focused follow-ups cited there. They do not claim that every later reporting edit or namespace-only fix received a fresh full regression or package check.

### Done and not done at this milestone, exact source state, and reproduction

Done: F1 mechanism foundation; F2 AFT and F3 subject-level hazard contracts; synchronized runner; server/client authority and API gates; documented regression; CI configuration; public campaign tooling and strict completion gate; one successful three-site synthetic Weibull release with public scoring and exact twins; six archived failed attempts; survival documentation with a reserved segmentation section. No remote push or thesis edit was made.

Not done: two successful synthetic integration gates (lognormal and hazard), all 90 cohort cells, the 12-group completed evidence summary, utility-floor/envelope review on those executed cohorts, the nine pending R schema assertions, and final evidence-driven documentation refresh. F5 remains incomplete. F6 has substantial completed checks and documentation but is not an overall completion or promotion claim. Claude remains the reviewer and promotion authority.

Source heads at this audit:

- dsFlower: `00b09a07ebca2b7d8c058221bd4d539f0aed1a2d`.
- dsFlowerClient: `fd61a944a5cdf60ee28e48e5dad1a336a6a2b877`.
- The completed installed build and each evidence cell retain their own exact commits and runner hash. Source HEAD must not be substituted for an older executed installation.

Reproduction commands, from the workspace root unless stated otherwise:

```sh
python3 dsFlowerClient/tools/check-runner-sync.py --server dsFlower
runtime/venv/bin/python -m unittest -v dsFlower/inst/python/tests/test_survival_contracts.py
runtime/venv/bin/python -m unittest -v dsFlowerClient/inst/python/tests/test_predict_helper.py dsFlowerClient/inst/python/tests/test_survival_predict_helper.py
runtime/venv/bin/python -m unittest discover -s dsFlowerClient/tools/campaign/survival -p 'test_*.py' -v
runtime/venv/bin/python dsFlowerClient/tools/campaign/survival/validate_completion.py dsFlowerClient/inst/extdata/campaign/survival
```

The last command must currently fail because the campaign is incomplete. The next targeted R schema command, to run only when the host is ready, is `Rscript -e 'testthat::test_file("tests/testthat/test-survival-evidence.R")'` from `dsFlowerClient`. Full R regression commands and structured results are recorded in `SERVER_F_NOTES.md` and `CLIENT_F_NOTES.md`; baseline summary dots are not converted into invented expectation counts. Campaign preparation, installation, and execution commands are in `dsFlowerClient/tools/campaign/survival/README.md` and `PROTOCOL_F_SURVIVAL.md`.

## 2026-09-18T12:03:57.744348+00:00 — Pod R readiness and isolated migration

- Pod bootstrap now contains `R_STACK_DONE`; the user’s wait condition is satisfied. R probe confirms actual R4.6.1, DSI1.8.0, DSLite1.4.1, dsBase6.3.5 and resourcer1.5.1. Missing packages installed only in `/workspace/survival/rlib`; Arrow/filelock round-trips pass. Exact report: `runtime/checks/pod-r-dependencies.json`. No segmentation/global R library changes.
- Transferred local git bundles (no remote pushes) and frozen public split/protocol archive. Pod clones retain original commit identities on feature/survival. Earlier Python-only snapshot preserved separately. Paired package installation is running with isolated runtime/library paths.
- Pod Python extras added within isolated pyvenv, retaining image Torch2.4.1+cu124, Opacus1.5.2, NumPy1.26.4 and Flower1.31. MONAI1.4 matches this supported Torch/NumPy range; GPU runtime dependencies align to CUDA12.4. Rerunning CUDA gates after dependency installation; in-flight tests are excluded.
- Mac refreeze completed at11:57:28Z: server00b09a0/clientfd61a94, unchanged runnerSHA256. Full import readiness and runner sync pass. New jobs will use the pod because it has432GiB available hostmemory and45GB free A40 memory at probe.
- Central tooling is being aligned to actual CUDA training for both nonprivate/null and DP twins, with common CPU held-out inference and exact device/determinism metadata. No architecture, optimization, sample, split, privacy or floor change.

## 2026-09-18T12:16:31.120575+00:00 — Final pod gates / secure runtime

- Final coherent CUDA environment: survival29, DPguards105 and DPsafety102 all pass; the previous optional MONAI skip now executes. Central Weibull/lognormal/hazard CLIs pass with CUDA training for nonprivate/null/DP and CPU held-out evaluation. CPU central/null scores exactly match pre-device-change outputs. Tooling commit7c7f8f8; train/test hash tampering rejects before parsing. Artifacts: `runtime/central-device-audit/pod-final/`.
- Pod client R:41cases568assertions,0fail/error/warn/skip; evidence schema259 includes the nine previously pending assertions. Structured logs: `runtime/pod-verification/client-targeted-*`.
- Pod server first run:437passes,1bootstrap error,3skips. `/workspace` fuseblk forces directories to0777 and ignores chmod; an attempted tmpfs mount in the work directory was denied. Guard remained intact.
- Created a task-owned OS temporary directory on overlayfs, ownerroot/mode0700, connected through `/workspace/survival/secure-runtime` and runtime/tmp,runs links. Persistent source, input data, package libraries, logs and evidence remain under `/workspace/survival`; only ephemeral runtime/secret storage uses POSIX temporary backing. No source/permission guard change.
- Exact server rerun there:77cases449expectations,446passes,0fail/error/warning,3expected skips. All159survival assertions and nine formerly unreachable bootstrap assertions pass. Earlier failure and new pass retained separately: `runtime/pod-verification/posix/`.
- Pod pair frozen at12:04:20Z, server00b09a0/clientfd61a94; source campaign tools fast-forwarded locally to7c7f8f8 without changing installed core. Starting sequential live hazard retry in a new directory; prior failed attempts retained.

## 2026-09-18T12:29:49.390034+00:00 — Executed hazard integration gate

- Pod hazard pilot executed three DSLite sites, two rounds, zero client failures and successful cleanup. Strict Python evidence validation passes. Public artifact SHA256 `30c2dc58b896eab9ab11a48ae67c38d816b4b9d9137b30a19c586dda56578d79`; executed record `cell-synthetic-hazard.json`. Duration 526.48 seconds. Synthetic utility is not cohort evidence.
- Started lognormal pilot in a new task-owned pod attempt directory. Cohort scoring remains gated on all three synthetic successes.
- Read-only timing identified repeated recursive venv metadata discovery (5.059 seconds per walk on FUSE). It explains part, not all, of startup latency. Deferred optimization to retain the frozen tested core; evidence `runtime/pod-verification/venv-lookup-timing.log`.

## 2026-09-18T12:33:50.572520+00:00 — Adversarial sticky review finding

- Final read-only review found accepted inert `model` and `data-kind=tabular` wire aliases could change the survival seed digest without changing computation. This is a substantive sticky-semantics gap despite prior passing focused tests. Gate6 is reopened; cohort matrix has not started. Assigned minimal survival-only seed canonicalization and regression tests, followed by byte sync, pod tests and paired refreeze. Historical pilot outputs retain exact older build identities.
- Container quotas are 7.65 effective CPU cores and approximately46.6GiB RAM, despite host reporting96CPU/503GiB. Do not use host totals to choose campaign concurrency. Investigating runtime metadata filesystem overhead without model/protocol changes.

## 2026-09-18T12:45:24.561192+00:00 — Resumed after tool-quota interruption

- Read the existing milestone/decision logs before inspecting both worktrees. Server has coherent uncommitted sticky-alias regression tests; client has an uncommitted read-only artifact auditor and generated article HTML. No production fix was left half-written. Both branches remain feature/survival.
- Continue from reopened Gate6; completed milestones are not restarted. Pod bootstrap R_STACK_DONE remains present; no campaign processes were active at inspection.

## 2026-09-18T12:47:48.538350+00:00 — Resumed Gate6 correction and recovered lognormal pilot

- Server `dc6570b` and client `cdb8c92` remove only accepted computationally inert survival wire aliases (`model`, `data-kind`, `target-bounds`) from the seed contract. Existing tasks retain their prior handling. Interrupted negative test log records 9 failing variant/alias checks before the fix; the complete local survival suite now passes 30 tests, including exact repeated release arrays for all three variants. Runner byte sync passes.
- Client `f48df68` archives the already completed pod lognormal pilot with its original installed build, preserves the read-only public artifact auditor and generated article. The record passes strict Python per-cell validation; no old execution is attributed to the new core. Previous auditor verification is 9 passing tests in `/workspace/logs/survival-artifact-audit-tests.log`.
- Pod CUDA survival/guard/safety gates and paired installation are running; no cohort scoring until successful readiness.

## 2026-09-18T12:50:15.817493+00:00 — Resumed regression and recovered package-check attribution

- Full local server regression at `dc6570b`: 271 cases, 1587 expectations, 1584 passes, 3 existing platform skips, 0 failures/errors/warnings; `Rscript runtime/resumed_full_R.R dsFlower`, structured result `logs/resumed-full-dsFlower.json`. Local campaign utilities: 12 tests pass.
- Recovered a completed but previously unlogged server package recheck: `runtime/checks/server-00b09a0/check-provenance.json` binds source `00b09a0`, tarball SHA256 `6f30953344a87e24a53ddd602fe91e3370a80fafa1352d52041bf3769d7e4b3f`, and R CMD check status OK (0 errors/warnings/notes). This predates the Python-only sticky fix and is not presented as its package check.
- CUDA post-fix survival 30 pass; DP guards 105 pass plus 102 subtests. Safety and paired refreeze are still running. Local pytest is absent from the pinned runtime; used the existing unittest runners rather than adding a dependency. Client full R and shared-path Python regression are in progress.

## 2026-09-18T12:51:38.585474+00:00 — Final source regression after resumed fix

- Full client R: 373 cases, 2540 passing expectations, zero failures/errors/warnings/skips, source `26420b6`; `Rscript runtime/resumed_full_R.R dsFlowerClient`, structured output `logs/resumed-full-dsFlowerClient.json`. The archive schema now includes all three successful pilots and six historical failures.
- Shared-path Python regression (runner, validation, holdout, CV, release, original/survival prediction): 163 tests pass. Pod DP safety: 102 pass. Gate6 alias correction now passes local and CUDA tests; no outstanding mechanism/API failure.
- Article rerender succeeds with the updated public evidence census and explicitly attributed exploratory NLL caveat; tool-generated Pandoc deprecation warning retained. Cohort matrix has not yet started; paired runtime installation is finishing.

## 2026-09-18T12:53:28.835703+00:00 — Gate6 closed and paired pod refreeze

- Pod installation completes FROZEN_INSTALL_READY and WORKSPACE_RUNTIME_READY. Installed server `dc6570b9908642a79763483e3b602d9040b55a6e`, client `f48df6802940c0d7918416375789f7593b7cf42f`, runner SHA256 `ac08384b65fe18eeb1a1bbc7c757f8103cb59990f164f32cc7b047405d8e4ac8`, built 2026-09-18T12:51:42Z. New mechanism/API gates pass; Gate6 is closed. Reviewer promotion remains separate.
- Before cohort scoring, verified byte-identical Python POSIX copy and retained original. Readiness after activation is pending. Evidence identities will use the installed pair above, not subsequent documentation commits.

## 2026-09-18T12:55:09.048073+00:00 — F5 v1 cohort execution started

- Command: `nohup sh /workspace/survival/runtime/start_resumed_matrix.sh > /workspace/logs/survival_matrix_v1.log 2>&1`, driver PID93735. Wrapper validates all three synthetic records and runner sync, uses isolated R_LIBS_USER and single-thread BLAS/OMP, then `run_matrix.py /workspace/survival --jobs 1`. First cell SUPPORT2/full/Weibull/epsilon1/seed1101 is executing. All 90 configurations retain frozen v1 settings.
- Post-placement runtime readiness passes with Torch2.4.1+cu124, Opacus1.5.2 and CUDA. Logical-path metadata walk still takes5.754s; no speedup is claimed. Original Python environment retained, byte-comparison/readiness reports under runtime.
- Other task workloads are active on the shared pod; this is recorded only as runtime context, not a demonstrated cause of any failure. No segmentation files/processes were changed.

## 2026-09-18T12:59:11.659762+00:00 — Final committed package snapshots checked

- `sh runtime/resumed_package_checks.sh` completed both builds (including vignette generation) and `R CMD check --no-manual --no-build-vignettes --no-tests`. Server `dc6570b`: Status OK, 0 errors/warnings/notes. Client `0e9bf1f`: 0 errors/warnings, one existing native API NOTE for R_GetConnection/R_WriteConnection; `git diff 01280c2 -- src` remains empty. Both source commits and tarball hashes are bound in `runtime/checks-resumed/<package>/check-provenance.json`. Full tests are the separate executed regressions above.
- Cohort1 remains in infrastructure startup; no score or utility-floor assessment exists yet. Read-only timing of logical versus canonical task environment paths is running to investigate startup overhead without altering training.

## 2026-09-18T13:02:10.156254+00:00 — Measured filesystem bottleneck; next-cell scheduling held

- Read-only measurement `runtime/measure_survival_paths.R` returns equivalent directory sets, with82.117s logical FUSE lookup versus0.133s canonical POSIX lookup. First cohort cell still has no released result.
- `kill -STOP 93735` holds only the campaign driver (not its active R/Flower children), so it cannot start the next cell until placement is corrected. Active cell is untouched. Client `64b0784` resolves the custodian environment root; R parsing and diff checks pass. Source/core runner and all model/privacy/protocol values are unchanged.

## 2026-09-18T13:07:43.742675+00:00 — Infrastructure attempt retained; canonical-root retry

- First SUPPORT2/full/Weibull/epsilon1/seed1101 attempt was interrupted after over10minutes without observed SuperNode process or released model. INT did not stop its workers promptly; TERM caused socket errors, parent cleanup stopped SuperLink, and the cell wrote failed evidence. All task-owned PIDs exited. Automatic node/symbol destruction lacked worker acknowledgements, so the archived enriched record conservatively sets cleanup_ok=false and records intervention; no score is present. Original run directory and log are retained with `-fuse-startup-aborted` suffix.
- Recovered archive: `failure-support2-full-weibull-eps1-seed1101-fuse-startup.json`. Original runtime JSON is untouched; intervention metadata is explicit. Seven failed attempts total, three synthetic successes, zero scored cohorts at restart.
- Source tooling advances to `558bdec`; installed core remains serverdc6570b/clientf48df68. Runtime/server now resolves to the task-owned POSIX server root with the same verified Python environment. WORKSPACE_RUNTIME_READY passes; full capability discovery0.482s. Record: runtime/canonical-root-readiness.json.
- Restarted sequential matrix from fresh first-cell directory with `/workspace/survival/runtime` paths and unchanged protocol and privacy semantics; actual command `nohup sh /workspace/survival/runtime/start_resumed_matrix.sh > /workspace/logs/survival_matrix_v1_canonical.log 2>&1`. No model, split, optimizer, round, epsilon/delta or clip pin changed.

## 2026-09-18T13:26:36.131369+00:00 — F5 first executed cohort and launcher placement

- Client `9e98370` archives SUPPORT2/full/Weibull/epsilon1/seed1101: ten rounds complete, cleanup true, strict per-cell validator passes, elapsed870.50s. Public held-out C-index0.62438094 (null0.5); NLL5.38080546 versus null5.09497370. Ranking improves while NLL worsens. No epsilon8 floor decision is possible from this single epsilon1 cell. 1/90 cohorts executed; all3 synthetic variants and7 failed attempts retained.
- Generated venv launchers only:37 interpreter shebangs relocated to canonical POSIX Python; package module files unchanged, old environment preserved, every launcher before/afterSHA256 embedded in runtime/build.json under separately timestamped runtime_placement. Original build.json retained separately. Core package revisions and runnerhash remain unchanged.
- Post-relocation WORKSPACE_RUNTIME_READY and30 CUDA/subject survival tests pass (16.024s). Source/algorithm unchanged; test logs in `/workspace/logs/survival_relocated_*`.
- Next scheduling uses2 concurrent cells under actual7.65CPU/50GBcontainer quotas; BLAS/OMPthreads remain1. Completed first record is reused. No scored model/protocol selection occurred.

## 2026-09-18T13:29:13.184654+00:00 — Independent first-cohort artifact audit

- Whitelisted public exports only: config/evidence/scores plus released model.pt and metadata.json. No node-secret or staged training artifacts exported.
- `runtime/venv/bin/python dsFlowerClient/tools/campaign/survival/audit_artifacts.py "$PWD" --out "$PWD/runtime/first-cohort-artifact-audit.json"`: PASS on1 cohort, incomplete coverage explicitly1/90. Checks actual release SHA256, runner identity, public model/training metadata, frozen train/test/site hashes and disjoint partition, and recomputed public C-index/NLL. Independent local CPU inference agrees with the pod record within pinned tolerances. This is not campaign completion or privacy proof.
- Matrix restarted as PID126405, two concurrent cells, log `/workspace/logs/survival_matrix_v1_relocated.log`. Current cells are remaining SUPPORT2/full/Weibull/epsilon1 seeds1102/1103.

## 2026-09-18T13:39:56.193118+00:00 — LUNG1 host partition and refreshed local gates

- Mac has16GiB RAM/8logical CPUs; use single-cell concurrency only. Refreshed installed pair: serverdc6570b/clientbfcf736, built2026-09-18T13:38:07Z, runnerac08384…e4ac8. WORKSPACE_RUNTIME_READY/FROZEN_INSTALL_READY and byte sync pass. Actual Python profile: Torch2.14.0, Opacus1.6.0, Flower1.31.0, CPU. Local DPguards105 and DPsafety102 pass; source survival30/shared-path163/fullR regressions were already executed on this same Python/code profile.
- Started `run_matrix.py "$PWD" --datasets lung1 --jobs 1` via nohup, log `logs/survival-matrix-lung1-mac.log`. Only LUNG1 is assigned to this host; every central/federated twin uses its actual CPU profile. All public model/privacy/split/schedule pins and protocolSHA remain frozen.
- Pod scheduling held after its active SUPPORT2 epsilon4 pair. Once both finish, preserve their records, update tooling, restart SUPPORT2-only to make host ownership explicit and prevent duplicate LUNG1 execution. No active cell was interrupted for this partition.

## 2026-09-18T13:42:19.713418+00:00 — Local driver launch correction

- The initial local background launch did not survive its spawning shell: PID23157 no longer existed, its log was empty, and no LUNG1 cell directory/log was created. This was a driver launch attempt, not an executed or failed training cell. Restarted the identical dataset-scoped command as a managed foreground execution session and will verify actual worker startup. Pod execution is unaffected.

## 2026-09-18T13:49:23.428100+00:00 — Host ownership established; first LUNG1 audit

- Pod active epsilon4 pair completed successfully (478.6s/467.3s), yielding5 SUPPORT2 records. They pass strict validation and are committed in clientfad5fea. Updated pod tooling tobfcf736 after preserving equal untracked evidence; restarted PID170263 with `--datasets support2 --jobs 2`, log `/workspace/logs/survival_matrix_support2_v1.log`. Existing records reused, no active cell interrupted.
- Local managed execution session7336 / PID26218 runs `--datasets lung1 --jobs 1`. First LUNG1 cell completes in166.49s, cleanup true, strict validator and direct public artifact audit pass; client58ccbb6. Public epsilon1/seed1101 C-index0.47731370 and NLL6.62119845. This below-null ranking result is retained; no LUNG1 utility floor is designated. Second LUNG1 cell also executes in160.1s and awaits grouped archival commit. Current merged archive7/90cohorts,3synthetic,7failed attempts; later live counts may advance.
- First LUNG1 audit file `runtime/first-lung1-artifact-audit.json`; verified actual model/checksum/configuration/splits and recomputed metrics, with no training or DP calls.

## 2026-09-18T13:54:15.518390+00:00 — Ten executed cohort records

- Clientd03fd92 archives10 executed cohort cells:7 SUPPORT2 and3 LUNG1. Every archived cell passes validate_cell; no additional failed attempt is recorded. This includes the complete Weibull epsilon1 seed group in each cohort, complete SUPPORT2 Weibull epsilon4 seeds, and first SUPPORT2 epsilon8 seed. No epsilon8 floor is assessed before its full three-replicate group completes.
- Drivers remain podPID170263 (SUPPORT2-only,2concurrent) and localPID26218/session7336 (LUNG1-only,1concurrent). All configs/splits/protocol hashes remain frozen.


## 2026-09-18 — Reviewer handoff: SUPPORT2 closure and LUNG1 archive reconciliation

Read STATUS_F.md, PROTOCOL_F_SURVIVAL.md and REVIEWER_NOTE_PROBES.md before acting. Protocol v1 and its hash remain unchanged. Read-only preflight validated all 63 executed SUPPORT2 cells on the pod. The seven failures predate this completed matrix; they are not seven failed matrix cells. The summariser's `incomplete 9 groups 7 failed attempts` means nine present SUPPORT2 dataset/subset/model groups; the three LUNG1 model groups (27 cells) were absent from the pod archive. No SUPPORT2 cell needs re-queuing.

All synthetic rows below use seed1101/epsilon8; the SUPPORT2 row is full/Weibull/epsilon1/seed1101. Historical failure JSON filenames are under `dsFlowerClient/inst/extdata/campaign/survival/` on both hosts and remain untouched.

| Failed attempt filename | Phase and supported cause | Disposition |
|---|---|---|
| `failure-synthetic-weibull-symbol-census.json` | Infrastructure preflight: proxy used inherited dsListSymbols against an empty dummy server; missing worker forwarding prevented unused-symbol verification. | Harness corrected; later successful synthetic Weibull already exists. |
| `failure-synthetic-weibull-install-order.json` | Infrastructure preflight: launch overlapped installation; bundled declarative validator temporarily absent. | Installation sequencing corrected; already recovered. |
| `failure-synthetic-weibull-bound-array.json` | Synthetic integration: installed R package predated one-feature bounds array fix; rounds unavailable and no release. Log also contains UNAUTHENTICATED before app input; stale bounds alone cannot explain every failure. | Refreshed runtime already recovered; retain mixed/uncertain attribution. |
| `failure-synthetic-weibull-postprocess-timeout.json` | Post-training export: processx timeout used 3,600,000 seconds, causing integer-millisecond overflow (`ms is not a length 1 integer`). | Trained model retained; same release scored after timeout correction, without training again. |
| `failure-synthetic-hazard-concurrent-host.json` | Federation/runtime: synthesized status1, no accepted/saved model. Detailed node logs were deleted by historical cleanup; exact cause is unproven. Host contention is not established as causal. | Keep uncertain failure; sequential pod synthetic hazard later succeeded. |
| `failure-synthetic-lognormal-concurrent-host.json` | Federation/runtime: status1, no accepted/saved model; historical observation records two UNAUTHENTICATED messages and three tracebacks. Detailed node logs removed; final causal attribution unavailable. | Keep uncertain failure; sequential pod synthetic lognormal later succeeded. |
| `failure-support2-full-weibull-eps1-seed1101-fuse-startup.json` | Startup/compatibility preflight stalled on FUSE, then operator interruption caused socket errors and incomplete cleanup acknowledgements (`cleanup_ok=false`). Historical timing measured 82.117s logical lookup versus 0.133s canonical lookup. | Canonical-root retry already executed successfully; original failed attempt retained. |

Log evidence: the six synthetic failures originated on the Mac, not the pod. Inspected local `logs/pilot-weibull.log`, `logs/pilot-weibull-2.log`, `logs/pilot-weibull-3.log`, `logs/final-pilot-weibull.log`, `logs/final-pilot-hazard.log`, `logs/final-pilot-lognormal.log`; the third pilot retains the authentication traceback. Inspected pod `/workspace/survival/logs/support2-full-weibull-eps1-seed1101-fuse-startup-aborted.log` (socket/cleanup failure) and `/workspace/logs/survival_matrix_support2_v1.log` (final summariser traceback). Cell logs are in `/workspace/survival/logs/`; driver logs are in `/workspace/logs/`. Missing historical detailed logs cannot be reconstructed. The archived investigation fields and prior recovery milestones provide the qualified causal record above.

The local LUNG1 driver had in fact finished all 27 cells after the last status update. All 27 pass `validate_cell`; their exact JSON bytes were copied via `runtime/lung1-handoff-v1/` to the pod archive, with a no-conflicting-destination check. Their Mac CPU/installed-build provenance remains intact; these are not pod CUDA executions. No records were overwritten with new training, and the original local untracked records remain uncommitted for the later evidence-assembly task. The pod's previous summary is backed up at `/workspace/survival/runtime/lung1-handoff-v1/pod-summary-before-handoff.json`.

The existing runtime wrapper was copied to `/workspace/survival/runtime/start_lung1_matrix_v1.sh`, changing only `--datasets support2` to `--datasets lung1`. It retains synthetic validation, runner sync, isolated R libraries, single-thread BLAS/OMP and `--jobs 2`. Effective driver command:

```sh
cd /workspace/survival
"$(readlink -f runtime/venv)/bin/python" dsFlowerClient/tools/campaign/survival/run_matrix.py /workspace/survival --jobs 2 --datasets lung1
```

Launch command submitted over SSH after successful preflight and shell syntax validation:

```sh
nohup sh /workspace/survival/runtime/start_lung1_matrix_v1.sh > /workspace/logs/survival_matrix_lung1_v1.log 2>&1 < /dev/null &
```

SSH returned exit0 and the preflight PASS for 63 SUPPORT2 + 27 LUNG1. No PID was captured and no post-launch status/log polling was performed. The driver should reuse all 27 LUNG1 records and regenerate the summary; completion is deliberately not asserted. No separate retry driver is warranted because no unresolved environment-failed cell remains.

The scipy failure was in the bare `python3` summary subprocess, outside the matrix interpreter. Both selected runtime and preserved venv import SciPy1.17.1. Existing platform freeze files already listed scipy; added an explicit `requirements-tooling.txt` (`scipy==1.17.1`) and README install instruction, and changed `run_matrix.py` to use `sys.executable` for the summary. The corrected driver and requirements file were copied to the pod without reinstalling core packages. Twelve campaign metric/completion tests pass; a mocked completed-LUNG1 driver check confirms no training subprocess and exactly one summary invocation using the driver interpreter. The initial check expected an unresolved macOS /var path; correcting the test expectation to the driver's canonical path passed without further production changes.

No pushes, thesis changes, floor/envelope judgments or pkgdown build. Reviewer resumes monitoring and later evidence/artifact assembly.

Tooling committed in dsFlowerClient on `feature/survival`: `1af99d3` (`Declare survival reporting dependency and preserve matrix interpreter`). STATUS_F.md lives outside the package Git repositories.


## 2026-09-18 — Hazard-only v2 preregistration and one-shot launch

AFT and all v1 evidence remain unchanged. Six preregistered candidates fix
linear/K10/batch64 and cross schedules10x2/20x2/10x4 with LR0.02/0.05.
Development:72 actual three-site federations, epsilon8, inner validation
within each of12 original training splits; deterministic selection only on
mean validation C, ties epochs/K/ID. Confirmation:30 hazard cells plus
matched pooled-DP/nonprivate/null twins. Two federation slots throughout.
These reuse known outer holdouts; v2 is explicitly a follow-up evaluation.
Protocol and D056 were written before any v2 launch.

Exact pod launch command (one invocation, no polling):

```sh
nohup sh /workspace/survival/dsFlowerClient/tools/campaign/survival/start_hazard_v2.sh > /workspace/logs/survival_hazard_v2_driver.log 2>&1 < /dev/null &
```

Selection: `/workspace/survival/runtime/hazard_v2/hazard_v2_selection.json`.
Per-cell logs/runs/splits: `/workspace/survival/runtime/hazard_v2/`.
Evidence siblings: `inst/extdata/campaign/survival_hazard_v2_development/`
and `inst/extdata/campaign/survival_hazard_v2/`; the latter receives summary.json.
A pre-existing runtime/hazard_v2 directory prevents accidental second launch.
No automatic retries; missing/failed development aborts before confirmation.
Reviewer monitors `/workspace/logs/survival_hazard_v2_driver.log` and relaunches
assembly after driver exit. This status records launch, not completed results.

Validation before launch:16 campaign unit tests pass, including development
outer-test poison/missing-file isolation, site preservation, selection ties,
missing/nonfinite rejection and hazard-only summary scope. R runner parses.
Mocked complete driver verifies72 development then30 confirmation calls and selection-before-holdout ordering. Pod preflight confirms existing canonical Python/R runtime and pandas/scipy;
no survival matrix/cell process was active (unrelated segmentation untouched).
