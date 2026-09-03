# Changelog

## dsFlowerClient 0.4.4

#### Explicit resource routing and lifecycle

- Resource initialization now routes through a validated `resource_kind`
  of `"imaging"` or `"tabular"`; the selector defaults to `"imaging"`
  for backward compatibility, and a failed imaging admission is never
  retried as another resource type. Existing tabular ResourceClient
  workflows use the explicit tabular route.
- High- and low-level lifecycle helpers remember only imaging handles
  they created and destroy that exact handle, including partial-failure
  retries.
- Client validation consistently forwards the server-pinned label
  vocabulary and sanitizes remaining application, association,
  convenience, validation, and cross-validation transport calls.
- Association setup reads the effective privacy unit from the
  initialized server handle before hashing or preparing a job, rather
  than trusting a site-wide default that may not describe an imaging
  collection.
- A tunnel socket timeout that reports zero progress now closes the
  ambiguous stream and aborts instead of retrying bytes that R may
  already have written.

## dsFlowerClient 0.4.3

#### dsImaging session integration

- `ds.flower.connect(resource=)` and `ds.flower.nodes.init(resource=)`
  now perform the explicit resource assignment -\> `imagingInitDS()` -\>
  `flowerInitDS()` chain. An already initialized `img` handle can be
  supplied with `symbol=` without reinitializing it.
- Every assignment requires an explicit acknowledgement from every node
  before the next stage starts. The high-level connection helper removes
  its partial resource/imaging/Flower symbols when initialization fails.
- Resource initialization uses a deterministic protocol-owned temporary
  symbol, and
  [`ds.flower.nodes.destroy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.destroy.md)
  removes it when given the session-owned imaging symbol. This keeps
  cleanup reachable after both automatic removal attempts fail.
- [`ds.flower.nodes.destroy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.destroy.md)
  removes Flower and session-owned imaging handles per node only after
  an exact destroy acknowledgement written to a deterministic
  protocol-owned symbol. A retry removes an orphaned ACK before
  continuing. Partial failures therefore preserve the opaque handle for
  a targeted retry instead of replacing it with `NULL`; preparation
  rollback also reports any node that did not acknowledge cleanup
  without hiding the original failure.
- Documentation and tests make the package boundary explicit: dsImaging
  owns storage and full-cohort admission; dsFlower consumes only the
  opaque object already present in the same DataSHIELD session.
- Sensitive DSI aggregate calls now suppress and restore DSI
  progress/error printing, so capability tokens, uploaded chunks, and
  tunneled model traffic are never deparsed into consoles or notebooks.
  Remote aggregate failures are reported with node-scoped generic
  diagnostics rather than raw server text.
- The low-level
  [`ds.flower.nodes.prepare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.prepare.md)
  no longer accepts the deprecated `label_set` shortcut. Public target
  levels belong in the validated run configuration and, for dsImaging
  resources, must agree with the collection’s manifest-declared public
  label vocabulary.

## dsFlowerClient 0.4.2

#### Fixes

- [`ds.flower.cross_validate()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.cross_validate.md)
  (and `ds.flower.fit(cross_validation = k)`) over a real
  SuperLink/SuperNode federation no longer fails closed with “Federated
  cross-validation failed or produced a forbidden fold artifact; no
  result was accepted.” The defect was node-side: dsFlower 0.4.1 wrote
  the staged manifest’s computed cross-validation budget split with 15
  significant digits, which the trusted runner’s release guard
  (correctly) rejected against its exact IEEE recomputation at
  `rel_tol = 1e-15`, so every fold round was reported
  `public-preflight-unavailable` and the all-rounds gate refused the
  job. Fixed in dsFlower 0.4.2 (lossless manifest serialization); deploy
  both packages at 0.4.2 together. The client-side gates are unchanged,
  and the byte-verified `dsflower_runner` bundled here is unchanged from
  0.4.1, so the sticky noise identity (runner hash) and all committed
  `ds.flower.fit` campaign evidence are preserved. Found by the
  run-at-pin utility campaign’s live-federation cross-validation harness
  (`tools/campaign/run_cv_demo.R`), which no packaged suite reached at
  0.4.1.
