# Synthetic integration verifier decisions

2026-09-18; segmentation-only reviewer Addendum1 follow-up. The verifier is public benchmark tooling, outside both released runner trees. It does not create an all-seven gate record, change privacy configuration, score utility, or certify promotion.

- Require successful actual federation status, cleanup, two available history rounds, three client sites, six distinct site/round accountant captures, and a digest-bound released artifact. Missing evidence fails; synthetic unit-test records exist only inside temporary test directories.
- Decode the fixed synthetic fixture independently of `load_subject_tensors` and `read_pair`: lexicographic image selection, selected-mask union, declared empties and invalid-mask retention. Every site must retain16 subjects from18 source rows, including one invalid subject and one valid empty subject. Compare both effective tensor hashes with every actual capture.
- Public fixture empty declarations use uppercase`TRUE`/`FALSE`, matching the fixed package vocabulary. The initially generated lowercase strings would invalidate every subject and were corrected in the fixture generator, without widening the package contract.
- Reconstruct q=.5 and four full-horizon steps from the fixed N16/batch8/one-local-epoch/two-round schedule. Check each actual per-round history has exactly two noise/accountant steps. Recompute replace-one epsilon and delta independently with PRV and require strict epsilon≤the requested budget and delta≤1e-5; no tolerance that relaxes privacy targets.
- Release inspection admits exactly the six finite decoder parameter tensors and no buffers. Recompute predictions from the persisted state and independently extracted held-out fixture features. Use CPU decoder evaluation to match the existing local prediction helper, while keeping the encoder on its pinned selected device; probability equality allows only1e-12 for CSV decimal serialization.
- Keep all subject tensors and test probabilities local to temporary tests or the public pod run. Archived verifier output contains only public synthetic censuses, hashes, mechanism summaries, runtime identifiers and gate evidence.

Local focused verification from the workspace root:

```sh
PYTHONPATH=dsFlowerClient/inst/flower_app:dsFlowerClient/tools/campaign/segmentation python3 -m unittest discover -s dsFlowerClient/tools/campaign/segmentation -p 'test_verify_synthetic.py' -v
```

Actual pod verification, after the real synthetic run completes:

```sh
PYTHONPATH=/workspace/segmentation/runtime /workspace/segmentation/venv/bin/python verify_synthetic.py --prepared SYNTHETIC_DIR --run SYNTHETIC_RUN_DIR --out SYNTHETIC_GATE_JSON
```

The local tests exercise verifier acceptance and rejection paths. They are not a real three-node federation result. The actual pod verifier must pass before its integration evidence is promoted into the combined blocking-gate record.

Final local result:6 tests passed in120.260 seconds on Python3.11.14/Opacus1.6.0; log`checks/resume-synthetic-verifier-tests.log` in the workspace. A first invocation from the nested tools directory selected an unprovisioned Python3.14; the final workspace-root command above selected the existing provisioned Python3.11. An intermediate development run was stopped after the final tests were expanded; its partial log is retained and is not counted as a pass.

## Actual synthetic03 failure and client preflight correction

Synthetic03 failed before training on all three actual sites: the client unconditionally supplied the multilabel field`num-labels=2`, while the segmentation server correctly rejects scalar target declarations. Cleanup succeeded. The transient worker FUSE wait was not the terminal cause. No verifier or combined passing-gate record was produced for this attempt.

Keep the server contract unchanged. Remove`num-labels` only from segmentation preparation and segmentation TOML; other neural contracts retain their existing field. Add an explicit absence assertion to the client submit test and a paired public preflight integration test using both loaded package sources. The paired test exercises the actual client submission config through the actual server normalizer, then captures the generated TOML without starting federation.

Reproduction: `DSFLOWER_SKIP_PYTHON_SETUP=true Rscript dsFlowerClient/tools/integration/segmentation-paired-preflight.R dsFlower dsFlowerClient` from the workspace. Before the correction:3 failed/2 passed expectations with the actual server rejection. After:5 passed. Logs are`checks/resume-paired-preflight-before.log` and`checks/resume-paired-preflight-after.log`. Synthetic attempts01–03 remain archived as failed with successful cleanup; gate7 remains pending the next actual execution and strict verifier.

Synthetic04 reached real Flower rounds but both were unavailable; it produced no node accountant captures. Its persisted metadata showed that the actual client output directory is a generated model directory below`run/artifact`, not that parent itself. Resolve the verifier artifact directory from the actual status`output_dir`, require it to exist beneath this run's artifact root, and test both the generated subdirectory and rejection of an unrelated directory. This is an evidence-loader correction, not a release or privacy change.
## 2026-09-18: observe the actual trusted node after integrity admission

Synthetic attempt 04 reached the real three-node/two-round path but returned an
unavailable model; it is failed evidence, and gate 7 remains incomplete. The node
environment intentionally removes inherited benchmark `PYTHONPATH` and `F_SEG_*`
variables. Preserve that isolation and the mandatory integrity bootstrap.

Install `segmentation_public_observer.py` and its one-line `.pth` only into the
dedicated `/workspace/segmentation/venv`. The installer refuses any other Python
prefix. At interpreter startup the observer imports only stdlib and registers a
deferred finder; it never imports the runner. The unchanged mandatory integrity
finder runs first and verifies the actual runner package bytes. The observer
wraps only `dsflower_runner.client_app` after the original loader executes, and
requires the active mandatory finder, its verified-package record, its package
pin, and the canonical unified ClientApp reference. The same checks run before
every observed fit. No production runner, environment allowlist, or integrity
hook is edited.

Custodian opt-in is the adjacent `segmentation-public-benchmark.json` beside the
node secret file; no secret bytes are read or recorded. Its directory must be
same-owner mode 0700 and the non-symlink regular file same-owner mode 0600. Exactly
four fields are accepted: `public_fixture_only: true`, `dataset` in
`synthetic/breast/busbra`, integer `seed`, and absolute `capture_dir` beneath
`/workspace/segmentation/` ending in `public-capture`. Missing opt-in leaves the
observer inactive. This instrumentation is exclusively for these public cohorts.

The observer passes original fit/private-factory arguments and returns unchanged,
captures the actual returned accountant history after successful fit, and refuses
capture unless one engine and the expected logical-step count were observed. It
records public fixture tensor hashes and accounting geometry, not raw tensors or
secrets. Setting Torch CPU threads to two restores the campaign's operational CPU
bound because the clean environment also omits inherited OMP/MKL variables;
deterministic algorithms and disabled TF32/cuDNN benchmarking match the existing
public campaign. Noise, sampling, clipping, secure RNG and privacy defaults are
unchanged.

Validation: seven focused tests passed, including a subprocess with the real
mandatory hook: altered package pin exits 99 before observer attachment; correct
pin permits actual history capture with unchanged package bytes and original
return/secure-RNG argument; wrong logical steps create no capture. Permission,
opt-in, absent config, unguarded/unpinned/finder-order and non-campaign installer
negative controls also pass. The thread-setting follow-up reruns the guarded
capture test on the final observer source. These tests do not establish gate 7;
the next actual federation and strict verifier must do so.
