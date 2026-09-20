# Federated segmentation DP diagnosis (2026-09-19)

The frozen v3 failed floors remain failed, and segmentation remains `vetted = FALSE`.
This is a public diagnostic, not replacement utility evidence or a v3.1 promotion.
No production runner or privacy setting has changed.

## Findings established before the final traced federation completes

The original BUS-BRA/batch16/epsilon8/seed20260919 cell has 30 accountant captures.
Every captured feature/target hash matches the independent cached public tensors.
Each site contains 284 subjects, all valid. Masks, packed validity planes and
patient selection therefore do not explain this cell's collapse.

A cached three-site reproduction calls the unchanged production `_dp_fit` and
averages its released parameters, without Flower transport or artifact loading.
With the captured initialization and frozen optimizer/horizon, Dice is 0.163974
at round 1, 0.154719 at round 2 and 0 at round 10. The matched nonprivate diagnostic
uses the same initial parameters, subject split and deterministic public ChaCha
sampling streams: Dice is 0.072540 at round 1 and 0.613418 at round 2. These public
RNG seeds are diagnostic only and are not a replacement for secret release RNGs.

The nonprivate loss drops from approximately 0.78–0.84 to 0.48–0.51 in round 1,
and to 0.30–0.34 in round 2. DP losses remain approximately 0.77–0.83 over these
rounds. The actual node observer confirms Adam 0.001 and the declared loss pins.
The first actual site trace has signal norm 0.21–0.29 and noised gradient norm
about 25.3, with parameter maximum 0.228. Saturation at the 1,000,000 bound is not
occurring. The decoder has 41,537 parameters; the observed noise norm agrees
with the prescribed noise scale, rather than showing an extra noise multiplier.

| Geometry | Each federated node | Pooled DP twin |
|---|---:|---:|
| Subjects | 284 | 852 |
| Poisson q | 1/18 | 1/54 |
| Mean actual draw size | 15.7778 | 15.7778 |
| Pinned gradient divisor | 15 | 15 |
| Steps/epoch | 18 | 54 |
| Steps over 30 epochs | 540 | 1,620 |
| Noise multiplier at epsilon 8 | 1.86279296875 | 1.2109375 |

An empty node draw has probability `(17/18)^284`, approximately 8.91e-08.
Equal epsilon is not equal optimization geometry: the pooled model has three
times as many sequential Adam updates and less noise per update. Consequently,
working pooled DP does not establish a structural federated runner failure.
The unchanged local DP mechanism itself reproduces the collapse.

## Independent mechanism check

Server test `test_two_round_dp_adam_matches_independent_subject_gradient_reference`
computes each subject's BCE/Dice and backward pass independently, applies the
coordinate bound and global L2 clip explicitly, draws the same domain-separated
ChaCha noise, and updates a separate Adam optimizer. Every parameter matches the
production loop over two rounds. It exercises clipped subjects and retains an
invalid subject. It passes before any production change, on both the Mac and the
pod's different Torch/Opacus runtime. This is a diagnostic confirmation, not a
red-first reproduction of a discovered implementation bug.

The public trace observer also has byte-identical output against an uninstrumented
run on the same synthetic inputs and streams. It is installed only in the dedicated
segmentation environment, attached after the existing mandatory integrity guard,
and only activates for the explicitly configured public diagnostic path. No runner
integrity check is disabled. Released public arrays stay on the pod; reports contain
hashes and aggregate public diagnostics only.

The shared DP loop, spatial output, packed targets, image-wise reduction and
all-parameter updates are tested directly. The separate survival session and its
clones were not accessed or modified; its reported C-index does not provide a
matched control for this larger spatial decoder's DP optimization.

## Final actual federation and disposition

The unchanged three-node ten-round diagnostic completed with cleanup confirmed.
Its channel-B Dice and foreground-positive Dice are both **0.0000**, IoU 0.0000.
All 30 fresh feature/target/accountant signatures match the original v3 cell.
All six parameters change in every round. Every aggregate exactly matches the
next round's three node inputs, and the round-10 aggregate exactly matches the
saved artifact, including Flower's float32 aggregation order. The original v3
artifact also differs from initialization in all six parameters; neither result
is a round-0 artifact.

Across 1,620 node-level logical batches there are no empty draws or nonfinite
parameters. Mean loss falls from 0.816980 in round 1 to 0.595345 in round 10.
Mean clipped signal norm falls from 0.283586 to 0.131187 while noised gradient
norm stays about 25.31. The final round's maximum parameter magnitude is 0.416663,
far below the clamp. The original observer was restored byte-for-byte and the
temporary installed diagnostic helper removed.

No structural runner defect was found in the requested tested paths. This does
not prove absence of every possible defect, nor establish an optimized private
utility ceiling. It establishes that collapse reproduces with valid data,
correct pinned updates and trained-state aggregation under this protocol.
A patch claiming to repair invalid subjects, clamp saturation, empty sampling,
wrong loss reduction or stale parameters would contradict the measured evidence.
The prerequisite fixed-runner gate is therefore unmet; **v3.1 was not launched**.
The reviewer must decide whether to accept this diagnosis with failed utility or
authorize a separately preregistered public-development optimization study. No
privacy weakening, post-hoc protocol change or utility relabelling is proposed.

## Verification and reproduction

- Server full Python: **507 passed**, 5 existing environment skips, 555 subtests
  (before this increment: 506 passed). Standalone DP-safety: **102 passed**.
- Segmentation suite: **32 passed** on Mac; independent reference additionally
  passes on the pod's Torch 2.4.1/Opacus 1.5.4 runtime. The final reference fixture
  explicitly exercises clipping. Guarded observer byte-parity check passes.
- Both source runner trees remain byte-identical; production/R code is unchanged.
  Prior original-28 and full R checks remain applicable; they were not repeated
  for a Python-test and diagnostic-report-only change.
- Server reference-test commit: `bea18de`. The installed production server remains
  `d6243e6`; installed production client remains `909685b`. No push occurred.

Machine-readable public summary: [dp-path-diagnosis-v31.json](../../../inst/extdata/campaign/segmentation/dp-path-diagnosis-v31.json).
It contains hashes, per-round diagnostic summaries, geometry and public metrics;
no tensors or secrets. The original v3 evidence is unchanged.

From the shared workspace, run:

```sh
python3 -m pytest -q dsFlower/inst/python/tests/test_segmentation_contracts.py -k two_round_dp_adam
python3 checks/check-diagnostic-observer-parity.py
python3 dsFlowerClient/tools/check-runner-sync.py --server dsFlower
./pod2 '/workspace/segmentation/venv/bin/python /workspace/segmentation/verify-dp-diagnosis-v31.py'
./pod2 'python3 /workspace/segmentation/finalize-dp-v31.py'
```

Exact diagnostic scripts and launch command are retained in workspace `checks/`:
`diagnose-dp-v31.py`, `diagnose-dp-ten-v31.py`, `diagnose-nonprivate-v31.py`,
`public-dp-trace-v31.py`, `install-diagnostic-trace-v31.py`,
`launch-diagnostic-v31.sh`, `verify-dp-diagnosis-v31.py`, `finalize-dp-v31.py` and
`check-diagnostic-observer-parity.py`. Their hashes are archived alongside the
summary. Pod logs use `/workspace/logs/segmentation-*-v31.log`; detailed public
traces and released diagnostic arrays stay under
`/workspace/segmentation/diagnosis-dp-v31/`. The actual federation launcher refuses
to overwrite its completed capture directory; replaying verification is read-only
apart from regenerating derived diagnostic JSON. Do not relaunch scoring into an
existing cell or treat these diagnostics as new matrix replicas.

## Reviewer disposition — 2026-09-19

The reviewer accepted the diagnosis: the observed federated-DP collapse is the
per-site geometry boundary of this fixed decoder/schedule, with no runner defect
found. Site N284, sigma1.86279 and540 updates differ from pooled N852,
sigma1.21094 and1620 updates; pooled-DP Dice~0.60 at epsilon8 demonstrates useful
learning at larger N. V3 floors remain FAILED and vetted=FALSE. The reviewer
separately authorized the preregistered v4 public-development study of decoder
capacity and schedule, at unchanged epsilon/delta/clip and original confirmation
cohorts. V4 does not repair or relabel v3, and is not a promotion decision.
