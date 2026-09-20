# V5 reconciliation — 2026-09-20

**Verdict (c): stale packaged evidence / wrong evidence path.** The four cited `campaign-v5/inst/extdata/campaign/segmentation/batch{16,64}/{busbra,breast}-evidence.json` files are byte-for-byte identical to **v3** evidence. They are not v5 confirmation results. The v5 floor is correct for its stated BUS-BRA outer-test arm. Both development and confirmation applied the BUSI epochs60 decoder checkpoint. No binding fix or training rerun is justified by this audit.

All remote paths below are under `/workspace/segmentation` on pod2. Reconciliation outputs live only in `reconciliation-v5-20260920/`; originals were not modified.

## Decisive existing v5 results, independently rescored

Batch16, full cohorts, ε8, seeds 20260919/20/21. Fresh CPU inference loaded each released `model.pt` with strict state-dict loading, used the existing frozen public feature tensors and outer-test masks, and reproduced saved channel-B metrics exactly. The six cells also passed `load_replicate`, including release hashes, split provenance, accountant captures, twin initialization and twin artifact hashes. No new training was run.

| Cohort | Seed | Foreground Dice | Predicted foreground pixels | Test subjects with predicted foreground |
|---|---:|---:|---:|---:|
| busbra | 20260919 | 0.661786446 | 7.5393% | 211/212 |
| busbra | 20260920 | 0.636023088 | 7.4398% | 211/212 |
| busbra | 20260921 | 0.640433075 | 7.4882% | 211/212 |
| breast | 20260919 | 0.531196144 | 5.0934% | 48/51 |
| breast | 20260920 | 0.526532103 | 5.6085% | 49/51 |
| breast | 20260921 | 0.520600290 | 5.7675% | 47/51 |

| Across-seed foreground Dice | Federated DP | Pooled DP | Federated nonprivate | Pooled nonprivate | fedDP ≥0.50 |
|---|---:|---:|---:|---:|---|
| busbra | 0.646080870 | 0.657275234 | 0.703646758 | 0.702660088 | PASS |
| breast | 0.526109512 | 0.558132576 | 0.617039937 | 0.605920287 | PASS |

The formal v5 confirmation floor is defined for BUS-BRA, not BrEaST. BrEaST’s PASS above means only that its observed mean exceeds the requested absolute 0.50 comparison; it does not create a new preregistered endpoint. Twin scores above are retained, hash-validated scores; the six fresh forward passes were for federated DP.

## Initialization: checkpoint applied in both paths

`checkpoint_sha256` hashes an NPZ file; `initial_random_tensor_sha256` hashes concatenated raw tensors. These are different hash domains and must not be compared directly. The audit verified the checkpoint file digest, each of its six float32 tensor digests, exact array equality with `public-capture/public-initial-arrays.npz`, and concatenated tensor digests. In every audited confirmation cell, the captured tensors equal the pretrained tensors and differ from the random-initialization digest. The same check passed all three selected development cells.

| Seed (same checkpoint for both cohorts) | Checkpoint NPZ SHA256 | Captured/pretrained concatenated tensor SHA256 | Random concatenated tensor SHA256 |
|---|---|---|---|
| 20260919 | `b6db994dbb3a922b7b62e60a97a6dac5ce56757015d8e3756b1788e08673d311` | `c73cab8750abef0ee9350b17dc8eed930493c52f2193019b3a574d15afcac02a` | `f5bf3f833764d036be5266fc5d6c80a0af5e33cdac2852646c5ae8150fae6771` |
| 20260920 | `b49b39a40ac00287ba5292f053aafff4c8ef042d09ac9c3a51d88ef2d94e07ed` | `d92d726853a1fc82432f0bdf0696ae99b8644ce55ea3d9ec6ad3466e5d7dc867` | `c585298347592d2ad1ff05c2fae23f5441924569918ea36013f8860f4215dcfd` |
| 20260921 | `54a83fc0751d6a3fba2f461db5a9b1e06ef7b828beed30bc0fc2fc7530452abb` | `376ee989fb7d3062b434c4145f1b6a1c155689f2529598fdd1dbaee8a366c618` | `c1e15bb99cefebfb5133c67b0e4da1fde7a0d07c920703d1b5bbb6afa2edc73f` |

These are persisted round-zero initialization captures, not a retrospective fresh random draw. The code binds their meaning to the actual federation: `sitecustomize.py` wraps `server_app._initial_arrays`, applies the checkpoint to the model, constructs a new `ArrayRecord` from those arrays, saves the capture, and returns the model and record. The runner passes that returned record into `strategy.start(initial_arrays=initial)`.

The driver path is shared: `run_v5.activate()` sets `F_SEG_V5_BINDINGS`; `run_federated.sh` invokes `bind_public_initialization.py` before R; the binding loader verifies the checkpoint digest; the Python observer applies the parameters. Development invokes this shell driver directly with `F_SEG_INNER_SPLIT`. Confirmation invokes it through `run_matrix.py`, which inherits the selected binding environment. R fits and then calls `ds.flower.predict(fit, ..., type="prob")` on the released fit. `score_public.py` binds the score to the released artifact hash and effective split hash. `assemble_evidence.load_replicate()` assigns `federated_dp=channel["metrics"]` and checks model/twin/capture provenance.

Source snapshots are in [pod-results/source](pod-results/source). Key locations: `run_v5.py:34,55,140,173,193`; `run_federated.sh:13`; `bind_public_initialization.py:9`; `sitecustomize.py:31`; `public_initialization.py:25,43`; `run_federated.R:193–221`; `assemble_evidence.py:199–205,261`.

## What the floor measured

`run_v5.confirmation_floor()` reads `v5/runs-batch{batch}/busbra-full-eps8-seed{seed}/channel-b.json` after validating each replicate. It measures the **released federated DP model**, ε8, full BUS-BRA, **852 outer-training / 212 outer-test subjects**, independently for each seed. It averages subject-macro foreground-positive Dice across three seeds and requires ≥0.50 plus ≥0.10 over the strongest trivial baseline. It does not read development scores, public BUSI training scores, or nonprivate twins.

Reproduced batch16: `(0.6617864461182665 + 0.6360230875122249 + 0.6404330752749733) / 3 = 0.6460808696351549`. The strongest-trivial across-seed score is 0.29553372019881846. Batch64’s recorded floor is 0.6352267313214685; its genuine v5 evidence summary agrees. Batch64 models were not independently forward-scored in this audit.

Development used BUS-BRA **684 inner-training / 168 inner-validation** subjects drawn from outer training, with the same narrow 9,521-parameter decoder, epochs60 checkpoint, batch16 and 20 rounds. Selected seed scores were 0.628513151, 0.619046894, 0.649094825 (mean **0.6322182899493723**). Different training/evaluation subjects explain why this is not identical to 0.646081. The apparent ≈0 discrepancy was a cross-version comparison, not development-versus-confirmation generalization failure.

## Exact stale-file provenance

Packaged protocol: `5a157dd8bb6700386eb22e2c977d8f1773d09ff56c793fbbedac68113f220edd`. Actual v5 protocol: `28a713f8ffe6f7bb7b74e7d59d13d1be265d5738b43503c60e6581a67e0c5820`. All four packaged files also match the copies shipped inside `campaign-v4/inst/extdata`.

| Packaged file suffix | SHA256 (identical to v3/evidence at same suffix) |
|---|---|
| `batch16/busbra-evidence.json` | `b9d123db7cec517c87f3bb6bcc98afc3215c6c5ff38d5b6d0c09b3aaca55233b` |
| `batch16/breast-evidence.json` | `65cb660808dc592e7c11544305acf542ef80232a01a02f48180f4853a6ddcfc8` |
| `batch64/busbra-evidence.json` | `578253a74fb4aea9b8e884b346afd3600a4cce2cebdccd7014fda71af710561d` |
| `batch64/breast-evidence.json` | `b068bfd2cbdee875f22826e53fea834ffa45dac9723446bcf4f24b2f737fefbb` |

The copied files were executed on September 18; v5’s evidence was assembled September 19. `run_v5.py` writes assembled results to `root/evidence/batch{batch}` and embeds them in `summary-v5.json`; it does not replace the packaged extdata evidence. The directory name `campaign-v5` identifies the tooling bundle, not the provenance of every inherited file inside it. This explains both the repeated pooled-DP ≈0.599 and the fedDP collapse values.

## Did the old evidence score a wrong/empty export?

The old packaged evidence’s ε8 artifact hashes point to actual **v3** released models. Independently loading and scoring all six corresponding v3 models reproduced their evidence metrics exactly. BUS-BRA genuinely predicted all background at the 0.5 threshold; its exports were loadable models, not empty files. BrEaST had one almost-all-foreground seed and two all-background seeds.

| Old v3 cohort | Seed | Foreground Dice | Foreground pixel rate | Maximum probability |
|---|---:|---:|---:|---:|
| busbra | 20260919 | 0.000000000 | 0.000000000 | 0.373638719 |
| busbra | 20260920 | 0.000000000 | 0.000000000 | 0.329381287 |
| busbra | 20260921 | 0.000000000 | 0.000000000 | 0.306888372 |
| breast | 20260919 | 0.132653396 | 0.996744792 | 0.530312896 |
| breast | 20260920 | 0.000000000 | 0.000000000 | 0.478864640 |
| breast | 20260921 | 0.000000000 | 0.000000000 | 0.441468507 |

Thus the cited BrEaST ε8 mean is 0.044217799, and the cited BUS-BRA ε8 zero is real **for v3**, not v5. No score was relabeled.

## Deliverables and exact final state

- [audit-results.json](pod-results/audit-results.json): six v5 round-zero checks, released artifact hashes, fresh predictions/metrics, full validated replicate records including unchanged privacy mechanisms, development checks, and stale-file identity.
- [old-export-audit.json](pod-results/old-export-audit.json): six independently scored v3 exports.
- [verified-existing-v5-evidence](pod-results/verified-existing-v5-evidence): byte-identical copies of the four actual `v5/evidence` files. These are existing results with corrected source selection, not newly generated runs. Deep model verification covered only the six batch16 full ε8 cells.
- [protected-sha256.json](pod-results/protected-sha256.json): audited input hashes, rechecked unchanged at audit completion.
- [audit.py](audit.py), [audit_old.py](audit_old.py): reproducible read-only audits (outputs restricted to the new reconciliation directory).

Completed on pod2. Detached audit processes 3905418 and 3905703 finished; both were launched with `< /dev/null`. No training jobs remain from this reconciliation. No privacy parameter was changed: ε8, δ=1e-5, clipping=1.0, replace-one adjacency, original accounted noise multipliers and schedules. No source binding change, thesis edit, tag, or push. No retraining is pending because case (a) did not occur. The reviewable correction is to cite the genuine v5 evidence paths, retaining the old files explicitly as v3.
