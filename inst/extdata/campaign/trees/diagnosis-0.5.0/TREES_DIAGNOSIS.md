# Native-tree canonical-container failure

On 2026-09-22, one new, unscored breast/random_forest/epsilon-1 replicate
(split seed 20260820, three sites, one round, registry defaults) reproduced
the 0.5.0 failure on pod-flower-tabular. `tools/campaign/trees/diagnose.R`
traced the validator entry solely to copy its public artifact and arguments
before temporary-directory cleanup. No validator was bypassed and no held-out
prediction was called. Training occurred and is not claimed to cost zero privacy.

## Exact failing check

`identical(bytes, .native_tree_json(value))` in
`dsFlowerClient/R/validate.R` is false. All container field-set, contract,
engine, task, aggregation, nonempty model-list, version and schema-hash
checks pass. The preceding size, SHA-256 and sanitization checks also pass,
as execution reaches this check. `version` is JSON integer `1`, parsed by R
as integer `1L`; it is not `1.0`.

The first difference is byte **553 (one-based; 552 zero-based)**:

```text
Python: ..."leaf_values":[0.5,0.5,0.5,0.9021311641755351,0.5,0.9631214808926953,...
R:      ..."leaf_values":[0.5,0.5,0.5,0.902131164175535,0.5,0.963121480892695,...
```

The Python artifact is 12,091 bytes; the R rewrite is 11,714 bytes.
Its SHA-256 is
`8c2d04a22f869ab9de1d11b802a013b4695047d76893360280bf0fcc115d74ed`.
Root keys are exactly `aggregation, contract, engine, models,
public_schema_sha256, task, version`, in sorted order. Both outputs are compact
JSON ending in `}` (byte 125), with no trailing newline. Re-encoding the
original with the Python writer reproduces it exactly. The defect is
jsonlite's lossy decimal serialization of binary64 leaves despite `digits=NA`,
not key order, whitespace, NumPy scalar encoding, or integer spelling.

## Runtime and emitter

Pod ID `2sy2g4pb3xwgqt`, hostname `b85270415cca`, R 4.6.1,
jsonlite 2.0.0, Python 3.11.16 (Clang 22.1.3), NumPy 2.4.6,
stdlib `json.__version__` 2.0.9. Native-node and client Python environments
use those Python/NumPy versions. Both installed R packages were 0.5.0 during
the reproduction. The unchanged node source is
`408f08c539329e2711260050ab40a6567aa4d89e`; the client baseline is
`50dda000a32ffcbdd039c2b74c909df451392bfb`.

The writer path in the identical bundled runners is:

1. `native_tree_client_app.py:843` calls `native_tree_engine.train_model`,
   dispatching to `random_forest_adapter` training and sanitization.
2. `forest_sanitizer.py` validates the strict prediction-only schema and emits
   Python floats using `json.dumps(ensure_ascii=True, allow_nan=False,
   sort_keys=True, separators=(",", ":"))`.
3. `native_tree_server_app.py:655` calls `native_tree_engine.build_ensemble`.
   `random_forest_adapter.py:749-785` re-sanitizes members, sorts them by digest
   and bytes, and emits the ensemble using that same JSON configuration.
4. The ServerApp creates the bound profile and atomically saves the release;
   the R client calls `.native_tree_release_metadata()` and then the failing
   validator. `R/native_tree_contract.R:135-143` uses jsonlite with
   `digits=NA, always_decimal=TRUE`, which cannot reproduce these leaf values.

Python uses shortest round-trippable binary64 spellings. The existing comment
that pure R/Python tree formats were byte-round-trippable was incorrect.
NumPy 2.4.6 is present, but the final emitter handles ordinary Python floats;
there is no evidence of a NumPy-version-specific defect.

## Correction and regression

Only dsFlowerClient changes, to version 0.5.1. A bounded, isolated stdlib Python
canonical writer reproduces ensemble bytes for the R validator; exact byte
comparison remains mandatory for pure engines. Duplicate keys and nonfinite
values fail closed. The existing SHA-256, size cap, exact fields, engine/task/
contract identities, sanitization attestation and public schema hash remain
enforced; the container version is required to be integer `1L`. Request/schema
serialization, privacy mechanisms, noise, clipping, accountant, defaults and
the runner are unchanged. dsFlower remains 0.5.0.

The regression fixture trains a one-tree, depth-one random forest on 512 fixed
public synthetic records through the actual native engine and ensemble emitter.
Its leaves are `0.0038312680927895396` and `0.9959833181690035`. R tests verify
that the old writer cannot round-trip it, the corrected writer can, and both
the release-metadata path and artifact validator accept the unchanged release.
Negative tests cover canonical spelling, whitespace, ordering, duplicate keys,
field/identity/version changes, digest, size, schema and attestation. A Python
test re-trains the fixture and checks the real output through isolated serialization.

The corrected installed validator also accepts the exact 12,091-byte breast
artifact retained from the failing run. Both installed runner hashes remain
`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
Test results, patch commit, executed cells and environment are recorded in
`evidence/trees/`; original failure records remain in `failed-0.5.0/`.
Public diagnostic artifacts are in `diagnosis-0.5.0/`. Full diagnostic run and
site secrets remain only on the pod at `/workspace/cells/trees-diagnosis-0.5.0/`.
