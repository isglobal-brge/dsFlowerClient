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
