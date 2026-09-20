# F segmentation preregistration — v4 public development

Preregistered 2026-09-19 under the reviewer disposition accepting the per-site geometry diagnosis. V3 remains FAILED and the registration remains vetted=FALSE. This section supersedes only model/schedule/selection/execution provisions below; the preserved v3 protocol follows for all unchanged definitions.

Development uses BUS-BRA only, all three original seeds 20260919/20/21. For each original training site, order subjects by SHA256(`inner-v4|<seed>|<subject-id>`), subject ID breaking ties; reserve the first floor(site N/5) for inner validation and train on the remainder. Retain original site assignments and every original training subject in exactly one inner partition. The outer test subjects are excluded from development training AND evaluation. Hash and freeze all three inner manifests before the first development run. Confirmation restores the complete original site cohorts and outer test sets unchanged. There is no outcome-based resplitting or early stopping.

Twelve candidates: decoder {current (41,537 parameters), narrow (9,521 parameters), pointwise (129 parameters)} × batch {16,64} × rounds {10,20}, three local epochs per round. Narrow is Conv128→8 3×3 pad1, ReLU, nearest×2, Conv8→4 3×3 pad1, ReLU, nearest×4, Conv4→1 1×1. Pointwise is Conv128→1 1×1 then nearest×8. Frozen encoder/output/loss/selection/invalid retention are unchanged. Adam0.001 and its v3 defaults/reset each round are unchanged. Epsilon8, delta1e-5, clip1; calibrate each site's noise to its own exact full horizon and inner-training population. Real three-site Flower/DSLite federated DP ONLY is used for development scores, not pooled screening. No physical splitting.

Selection: highest arithmetic mean inner-validation subject Dice across the three seeds, then fewer parameters, then fewer total logical site updates summed across the three seeds, then decoder name and batch as stable final keys. Only candidates with all three successfully validated releases/accountant transcripts are eligible; failed attempts remain failed, no retraining. If none is eligible, stop with failure. Report inner foreground-positive Dice and all four trivial masks. An inner epsilon8 floor failure (mean Dice<0.50 or <mean strongest trivial+0.10) does NOT prevent confirmation. Selection is public development, not a private HPO service or a promoted utility claim.

Confirmation retains the 66-cell design: the selected decoder and round schedule run across BOTH original batch arms (33 cells each); the selected batch is primary, the other a prespecified sensitivity arm, never a second selection. All cohorts, seeds, epsilons, extensions and exact twins are retained. Recalibrate each mechanism at unchanged privacy. Record this interpretation explicitly in the selection JSON. Do not merge batch scores or relabel v3. Run the existing evidence/accounting/envelope assembler per arm, reporting v3 and v4 side by side with foreground-positive Dice and failures. Confirmation uses previously inspected public test sets: label it public benchmark confirmation after development, not a pristine unseen test.

One detached pod driver executes development → deterministic selection → confirmation → summariser. Every attempt is immutable, all errors persist, no automatic retries. The driver catches cell failures to finish remaining planned cells. Summary failure remains explicit. Exit the interactive session after nohup launch without polling outcomes. Protocol SHA256, split hashes, commits, tool/runner hashes and launch command are recorded before scoring.

---

# F segmentation preregistration — v3

Status: preregistered 2026-09-18 before any v3 scored cell, under the binding reviewer decision after diagnosis. All v2 cells remain archived as invalid (schedule defect). Changes after the first scored cell must create a separately named protocol and retain this file. This protocol implements the segmentation portion of approved DESIGN_F_CODEX.md against dsFlower main 4b8dcbf and dsFlowerClient main 01280c2. Promotion is the reviewer's decision.

## Cohorts and provenance

BrEaST / TCIA Breast-Lesions-USG Version 1 (updated 2024-01-08), DOI 10.7937/9WKK-Q141, is the correctness cohort. BUS-BRA Zenodo record 8231412, version 1.0, DOI 10.5281/zenodo.8231412, is the scaling and primary utility cohort. Download public releases only onto `/workspace/segmentation/data`; record the publisher MD5 where available, full downloaded SHA256, licence text SHA256, metadata snapshot, and authors' attribution in `dsFlowerClient/inst/extdata/campaign/segmentation/provenance/`. These datasets declare CC BY 4.0. Audit BUS-BRA's 1,875 images to 1,064 patients from the released identifiers and publisher description before subject-level claims. Failure to establish the patient mapping blocks this cohort; do not silently treat images as subjects. ISIC is excluded from this protocol.

No governed records are used. Counts in public benchmark provenance come from these public releases and are never added to production telemetry. The checkpoint is `ResNet18_Weights.IMAGENET1K_V1`, URL `https://download.pytorch.org/models/resnet18-f37072fd.pth`; record full SHA256 and upstream weight-use terms. Do not redistribute checkpoint tensors in the package.

Preparation clarification, recorded before scoring: BUS-BRA's released `Case` column has exactly 1..1064 with 811 two-view and 253 one-view cases; both publisher CV files keep cases together. BrEaST's RGBA masks have identical binary RGB channels: retain their silhouette and write L-mode PNG {0,255}, ignoring alpha as a class. BUS-BRA's mode-1 masks are likewise written as L-mode {0,255}. Preserve native geometry and pin every source/converted mask hash. Union BrEaST tumor and `other` masks (the publisher identifies the latter as additional suspicious lesions); generate explicit empty masks only for XLSX `Classification=normal` cases 45, 61, 209 and 213. The source archives are unchanged. Splits use floor(N/5): BrEaST 205 train/51 test and BUS-BRA 852 train/212 test, divided into three disjoint nodes.

## Fixed contract

Name `pytorch_resnet18_segmentation`, task `segmentation`, neural/image, loss `segmentation_bce_dice`. Frozen encoder profile `resnet18_layer2_128_v1`, through layer2, outside released decoder; eval/no_grad, immutable BatchNorm. Decoder: Conv2d 128→32 3×3 pad1, ReLU, nearest ×2, Conv2d 32→16 3×3 pad1, ReLU, nearest ×4, Conv2d 16→1 1×1. Input grid 128×128, output shape [1,128,128]. Use the runtime's pinned image conversion/resizing and nearest mask resize identically in all twins. No augmentation or mixed precision. Disable CUDA/cuDNN TF32 as well as autocast: all encoder/decoder arithmetic is full float32 on the pinned runtime. This numerical clarification follows the synthetic all-parameter gradient parity gate before any utility score.

One image per patient: canonical string image IDs, lexicographic minimum; duplicate declarations for that exact image union lesion masks, while conflicting image pairing totalizes the selected subject as invalid. The image-selection decision does not depend on annotation availability, foreground area, or prediction. Valid empty masks must be explicitly declared. Invalid selected images/masks/pairings remain in N with fixed safe tensors and validity zero. Pixel masks declare their source vocabulary; {0,1} and {0,255} map explicitly to {0,1}. No intensity normalization of masks.

Per-subject loss is 0.5 BCE (mean pixels) + 0.5 (1−soft Dice), smoothing 1; mean across all sampled subjects including invalid zero contributions. Alpha=1 is a separately labelled BCE ablation, never a replacement after inspecting results. The original source-row census and subject census are preserved.

## Frozen splits and schedule

Three public seeds: 20260919, 20260920, 20260921. Each changes the public subject split, site allocation, and public decoder initialization; identical sticky retries are not independent replicates. For each cohort/seed, order subjects by SHA256 of `split-v1|<seed>|<canonical-subject-id>` (subject ID breaks any hash tie). The first floor(N/5) subjects form the test set; all remaining subjects form training. This is the fixed 80/20 rule; no development set, early stopping, or test-selected hyperparameters. Write and hash every split manifest before any model score. Apply subject splitting before image selection, even when the release is one image per subject.

Primary topology: three disjoint DSLite nodes with independent PSOCK R workers and actual Flower SuperNodes. Hash-order training subjects with domain `sites-v1`, then deal round robin across the three sites. A synthetic three-node two-round cell must pass before public scored federation. For one preregistered heterogeneity stress cell (BUS-BRA epsilon 8, all three seeds), sort the TRAIN subjects by their publisher pathology class then public site hash and allocate contiguous thirds; test membership is unchanged. This publicly observed class stratification is benchmark-only and does not change the production contract.

Primary matrix: BrEaST and BUS-BRA × epsilon {1,4,8} × three seeds. Node-owned replace-one patient privacy; delta=1e-5, global clip=1. Train ten rounds, three local epochs per round, Adam learning rate 0.001, separate logical batch arms 16 and 64, betas (0.9,0.999), epsilon 1e-8, no AMSGrad or weight decay, uniform FedAvg node weighting. Fixed decoder initialization is identical in corresponding twins. Use the runtime's exact `L=ceil(N/batch_size)`, `q=1/L`, expected divisor `max(1,floor(N/L))`, and 30L total steps per site; preserve empty draws and cryptographic sampling/noise. Never truncate Poisson draws, drop subjects, or count pixels as units. First attempt uses complete logical batches; physical splitting is prohibited unless its separate BatchMemoryManager/accountant/secure-RNG tests pass.

Small-N curve: within each BUS-BRA training split, the first 192 subjects under a separately domain-separated `subset-v1` hash form a nested small cohort, with the same held-out subjects. Repeat all three epsilon values and seeds on that subset. BrEaST is an additional independent small-cohort check. BCE-only ablation uses BUS-BRA epsilon 8 and all three seeds with the same split/schedule. Do not use ablation results to replace the preregistered primary model.

## Exact twins and local scoring

For each primary split/schedule, fit pooled nonprivate and pooled DP twins with identical selected tensors, validity, encoder checkpoint/profile, decoder initialization, loss, optimizer and thirty total epochs. DP pooled accounting uses pooled training N and its own exact q/sigma/step horizon; record these differences from site accounting explicitly. A federated nonprivate diagnostic uses the same site tensors, initialization, ten-round/three-local-epoch schedule, equal node weighting and optimizer, but exists only in isolated public-data benchmark tooling. It must not introduce a runtime privacy off switch. Any twin mismatch invalidates the comparison until explained.

Only channel B evaluates public test masks through released artifacts. Probability threshold is fixed at 0.5. Score mean subject Dice and IoU on the canonical 128×128 grid, and report foreground-positive and empty-reference strata separately. Both empty = 1; exactly one empty = 0; otherwise Dice = 2 intersection/(predicted+reference), IoU = intersection/union. No smoothing in reported metrics. Zero-member strata are null, not zero. A native-resolution score is outside the primary protocol.

Preregistered trivial masks: always-empty, always-full, fixed centered disk radius 32 pixels, fixed centered square of side 64 pixels, all on the canonical grid. The strongest trivial mask is the highest mean Dice among these four fixed predictions for the same public held-out split. There is no fitted or test-adjusted shape. Report all four, then the maximum.

## Utility gates and uncertainty

BUS-BRA primary epsilon 8 must achieve across-seed mean Dice ≥0.50 AND exceed the across-seed strongest trivial mean Dice by ≥0.10. Foreground-positive Dice is mandatory. A failed floor remains a failed floor and blocks a positive practical-utility claim. Lower epsilon and BrEaST may fail; report every executed cell and failure.

Use paired replicate gaps G = U_pooled_nonprivate − U_federated_DP. With three seeds report mean, sample SD and a two-sided 95% Student t interval for the mean (df=2); do not mislabel SD as confidence bounds. Bounds may exceed [0,1] when normal approximations are poor; retain them and explain rather than clipping. If optional additional seeds are needed for uncertainty, preregister them as a separate extension and retain all original seeds.

Four DESIGN_F_CODEX §6.3 envelope checks:
1. Epsilon envelope flags adjacent increases G_next−G_previous > max(SD_previous,SD_next), using matched replicate gaps.
2. The above BUS-BRA epsilon 8 Dice and trivial-margin floor.
3. On the 192-subject subset, flag SD_gap(epsilon8) > SD_gap(epsilon1).
4. Flag each near-central replicate when training subject n × epsilon <2000, central Dice <0.95 and abs(G)<0.005. Report smallest per-node N too, as a separate companion diagnostic.

Flags are empirical diagnostics, not privacy proofs. Investigate sigma/N/q/steps/clip/subject gradients and artifact identity. Never omit a lucky or failed replicate to improve envelopes. Unexplained flags prevent a validated-utility claim.

## Evidence and execution gates

Before scoring require mechanism gates §6.1 items1–7, including every decoder parameter's per-subject gradient, batch-wide Dice negative control, training-mode BatchNorm negative control, source/unit census, authority, sticky retry semantics, fixed output geometry, invalid subject retention, and local prediction/private-validation separation. Exact failures are recorded rather than relaxed.

Each executed evidence JSON records schema/status/date, protocol SHA256, dataset release/licence/checkpoint hashes, subject mapping/split hashes, public M/N and per-site N, exact commits and runner hash, device/dependencies/determinism settings, encoder/selection/mask semantics, optimizer/schedule, epsilon/delta/clip/adjacency, per-site accountant/sigma/q/expected divisor/full horizon, independent recomputed epsilon, artifact checksum, all replicate scores/strata/intervals, twin matching, all envelope outcomes/explanations, timing/GPU memory and cleanup. Store failure statuses without placeholder scores. Privacy experiments on a real overlapping cohort would compose; public benchmark replication is not an unlimited private budget.

Use `/workspace/segmentation/`, `R_LIBS_USER=/workspace/segmentation/rlib`, and logs under `/workspace/logs/`; wait for `/workspace/logs/install_r.log` to contain `R_STACK_DONE` before invoking R. Commit tooling/evidence separately; never push or edit the thesis.

## Preregistration implementation details, still before scoring

The benchmark-only `benchmark_hooks/sitecustomize.py` seeds and captures the exact public server initialization, then observes the existing accountant after each completed node round. Activate it only with `F_SEG_PUBLIC_BENCHMARK=1`; it is outside the released runner tree. It stores no secret, feature tensor or mask. Recorded feature/target hashes are public-fixture checks for exact-twin parity. Nonprivate twins preserve the same Poisson geometry and expected-batch divisor, omitting clipping/noise only inside benchmark tooling. DP twins call the unchanged trusted `_dp_fit` loop. A persisted secret local to the public benchmark binds repeat twin runs to the same cryptographic streams; it must not be copied into evidence.

## Protocol v3 authorization and prospective execution

The reviewer authorized this training-only replacement after the public-development probe and matched diagnostic established the schedule defect. This decision derives from the probe's inner validation, not any campaign held-out test score. Diagnostic scores establish mechanism/schedule plausibility only and are not release evidence. Retain the fixed campaign splits and all v1/v2 records; do not relabel or reuse old trained artifacts.

Execute all 33 cells separately in each nominal batch arm {16,64}: 66 new cells total, including primary, small192, BCE-only and heterogeneous cells, all epsilon values and all three seeds. Each arm has separate exact twins, floors and envelopes. Ten rounds × three local epochs gives 30 total epochs; optimizer state resets each round in all four branches. Recalibrate site and pooled noise for the full 30-epoch horizon with unchanged epsilon/delta/clip and trusted accounting. No decoder change, physical splitting, augmentation or mixed precision. Epsilon1 failures remain failures. No arm selection or pooling.

The immutable feature cache was prepared under v1; its old optimizer metadata is not an active training configuration. Verify all cached encoder/mask/selection semantics and actual site tensor hashes, and derive exact-twin optimization pins from the fresh captured v3 federation configuration. Preserve cache bytes and split hashes.

Use /workspace/segmentation/v3 as the fresh execution root, with read-only reuse of prepared data/features via symlinks; retain the parent CAMPAIGN_HOLD.json to protect v2. Launch two workers total, one per arm, sharing the existing two federation slots. Existing all-seven gates and synthetic integration remain valid; focused tests verify the new schedule and full accountant horizon before launch. No automatic evidence assembly or render: reviewer monitors the launched matrix and requests assembly later.
