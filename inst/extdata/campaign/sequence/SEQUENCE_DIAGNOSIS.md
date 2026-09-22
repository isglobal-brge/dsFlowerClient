# Sequence import-guard diagnosis

Token: `FLOWER_CELLS_SEQUENCE_2026-09-22`. Date: 2026-09-22.
Pod: `pod-flower-sequence` (`5abpdvx54g6uz2`), NVIDIA A40.

The original three-node logs contain the default-deny message but no stack.
`tools/campaign/sequence/trace_guard.py` reproduces it in separate, fresh LSTM
and GRU interpreters under the installed 0.5.0 guard and canonical runner pin.
Its diagnostic-only `_abort` wrapper prints the stack, then calls the original
abort unchanged. Both exit 99 at model construction, before optimizer creation.
It reads no HAR data. Full traces are in
`evidence/sequence/blocked-0.5.0/diagnosis-0.5.0/{lstm,gru}.log` and `trace.json`.

Observed runtime: R 4.6.1; Python 3.11.10; torch 2.6.0+cu124;
Opacus 1.6.0; Flower 1.31.0. Both R packages were initially 0.5.0.

## Exact import chain

The stack from each guarded reproduction is:

1. `dsflower_runner.params.load_user_model` (`params.py:89`) imports
   `opacus.validators.ModuleValidator`.
2. `opacus.__init__` imports `privacy_engine.PrivacyEngine`; `privacy_engine`
   imports `accountants.create_accountant`; `accountants.accountant` imports
   `optimizers.DPOptimizer`.
3. `opacus.optimizers.__init__` eagerly imports
   `fsdpoptimizer_fast_gradient_clipping.FSDPOptimizerFastGradientClipping`,
   which imports `torch.distributed._tensor.experimental.implicit_replication`.
4. `torch.distributed._tensor.__init__` imports `tensor._shards_wrapper`, then
   `checkpoint.metadata` → `checkpoint.__init__` → `default_planner` →
   `_shard._utils` → `_shard.__init__` → `_shard.api` →
   `_shard.sharded_tensor.__init__` → `sharded_tensor.api` →
   `sharded_tensor.reshard` → `torch.distributed.nn.functional`.
5. `torch.distributed.nn.__init__` imports `api.remote_module.RemoteModule`;
   `remote_module.py:42` calls
   `instantiator.instantiate_non_scriptable_remote_module_template()`;
   `instantiator.py:154,96` writes its fixed template into a fresh temporary
   directory, adds that directory to `sys.path`, then calls
   `importlib.import_module("_remote_module_non_scriptable")`.
6. `sitecustomize._IntegrityFinder.find_spec` resolves that temporary file as
   foreign; `_verify_foreign` finds no matching package pin and exits 99.

This happens before choosing the recurrent layer, so GRU fails identically.
It is not caused by a privacy check, accountant failure or recurrent gradient.
The installed source and trace establish the chain; the
[PyTorch 2.6 generator source](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/distributed/nn/jit/instantiator.py)
also documents the temporary module construction.

## Guard and pin contract

The guard is `dsFlower/inst/python/sitecustomize.py`, installed at
`/opt/cells-sequence/Rlib/dsFlower/python/sitecustomize.py`. dsFlowerClient
has no production copy of this guard. The campaign's separate `sitecustomize`
file instruments the public ServerApp only; it does not replace the node guard.

`R/interface.R` writes `pinned_packages.json` into each handle's `staging_dir`,
alongside `manifest.json`; the child receives that directory through
`DSFLOWER_MANIFEST_DIR`. It is a per-run `{package_name: sha256}` map, not a
shipped dependency list. Existing maps must match the node's installed runner
hash. New maps are atomically renamed into place with mode 0600. Hook runs may
add their separately checked uploaded package (`R/app_store.R`), but uploaded
HookApps remain forbidden in the trusted parent.

The Python guard accepts a nonempty JSON dictionary; missing or malformed
maps fail closed when foreign code is imported. The canonical runner is
always hash-checked even inside site-packages. Hashing recursively sorts
forward-slash relative paths and digests `path + newline + bytes + NUL`,
excluding `__pycache__`, `.pyc` and `.pyo`. The ordinary guard trusts stdlib and
installed site-packages paths. Thus **torch is an installed trusted dependency,
not an entry in this application-code pin map**; its dependency versions are
recorded/enforced separately by the Python environment contract.

`.compute_harness_hash()` hashes only `flower_app/dsflower_runner`. Neither
`pinned_packages.json` nor `python/sitecustomize.py` is part of that hash.
The package includes the node guard separately. Both installed runner copies
remain byte-identical with SHA-256
`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.

## Minimal patch and verification

Branch `fix/import-guard-torch-generated-modules`, based directly on main
`408f08c539329e2711260050ab40a6567aa4d89e` (v0.5.0), commit
`c4eaaf153db4a7204878bf1d8995c16faa615110`. No native-tree patch is included.
Only dsFlower changes: `DESCRIPTION` (0.5.1), `NEWS.md`, the guard and its tests.
dsFlowerClient stays at 0.5.0; neither runner copy changes.

Avoiding the import in our recurrent constructor cannot fix this: the
necessary Opacus validator/PrivacyEngine initializes the package and eagerly
loads the FSDP optimizer before our recurrent layer is selected. Altering the
third-party package, stubbing its modules or skipping its validator would be
broader and less faithful than the explicit generated-source rule.

The rule admits exactly `_remote_module_non_scriptable`, only when torch is
inside a trusted interpreter installation, both already-loaded generator
modules resolve to their exact paths in that torch installation, the generated
origin matches that instantiator's own temporary directory, and its entire
source equals the installed fixed template with the non-scriptable arguments.
The loader executes the checked source snapshot, never a subsequently changed
file or cached bytecode. Other names, scriptable templates, shadow locations,
modified content and uploaded HookApps retain normal default-deny handling.
No privacy mechanisms, randomness, accountant, clipping, runner or registry
defaults changed.

Tests: 9 import-guard tests on the pod, including fresh guarded 128 × 9 LSTM
and GRU DP updates; 105 DP-SGD guard tests; 102 DP-safety checks (including
recurrent architecture/DP checks); 16 runner-contract tests. The server R
Python-dependency and runtime suites also passed. All 24 release-guard tests passed. Full verification logs are retained with
the evidence. The local lightweight
guard run skips its integration test because torch is absent; the pod runs it
without skips. Total verification stayed below the requested 20-minute cap.

The representative run retains the frozen subject-disjoint protocol. The
released patient path pools each subject's windows and modal label before
per-subject clipping; each site therefore trains on seven pooled sequences.
The central twin uses the same 21 pooled sequences without DP. This limitation
is reported with the window-level held-out metrics, not silently repaired.

## Campaign tooling recovery (before held-out scoring)

The first patched federation completed all 15 node rounds and cleaned up in
132.1684 seconds. The previously unexecuted central checker then stopped because
it hashed int64 target bytes whereas the runner's `task._load_target` returns
float32. Feature tensors were already byte-identical. Matching the target dtype
verified all 15 captures without changing data values or training settings.

A subsequent TRAIN-only predictor smoke check found that the central writer
omitted four shared recurrent parameter aliases. Central export now uses stock
`state_dict`, matching the unchanged released ServerApp. Replaying that central
fit before scoring produced bit-identical tensors for all six unique parameters;
only the checkpoint's shared aliases changed. The saved federated artifact also
passes the unchanged predictor's TRAIN-only smoke check. Neither interruption
read held-out data or repeated the completed federation. Logs and tensor hashes
are in `tooling-interruption/`, `tooling-recovery.json`, and
`central-replay-verification.json`. These are campaign-tooling fixes only.

## Completed representative cell

All nine LSTM replicates completed (epsilon 1/8/4, three public seeds) with 135
successful node-round captures. All model hashes, cleanup records, full-horizon
accounting and original protocol hash were verified before the sole test-scoring
marker. No local-epoch reduction or scored rerun occurred. The GRU fallback was
unnecessary. dsFlower 0.5.1 and dsFlowerClient 0.5.0 were the installed packages.

Mean federated macro AUC was 0.434854 / 0.434235 / 0.435451 at epsilon 1 / 4 / 8.
Central mean AUC was 0.435143. At epsilon 8 accuracy was 0.169551, below the
majority baseline's 0.182219; mean AUC was below 0.5. Both utility annotations
are therefore false; execution is complete and the observed results are retained.
See `evidence/sequence/README.md`, the three cell JSONs and `summary.json` for
all metrics, paired gaps and sample SDs. A metadata-only reporting correction
separates TRAIN-preparation test access from the completed held-out scoring;
it changed no metric or setting and did not rerun scoring.
