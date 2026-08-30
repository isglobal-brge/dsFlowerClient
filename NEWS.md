# dsFlowerClient 0.4.2

### Fixes

* `ds.flower.cross_validate()` (and `ds.flower.fit(cross_validation = k)`)
  over a real SuperLink/SuperNode federation no longer fails closed with
  "Federated cross-validation failed or produced a forbidden fold artifact;
  no result was accepted." The defect was node-side: dsFlower 0.4.1 wrote the
  staged manifest's computed cross-validation budget split with 15
  significant digits, which the trusted runner's release guard (correctly)
  rejected against its exact IEEE recomputation at `rel_tol = 1e-15`, so
  every fold round was reported `public-preflight-unavailable` and the
  all-rounds gate refused the job. Fixed in dsFlower 0.4.2 (lossless manifest
  serialization); deploy both packages at 0.4.2 together. The client-side
  gates are unchanged, and the byte-verified `dsflower_runner` bundled here
  is unchanged from 0.4.1, so the sticky noise identity (runner hash) and all
  committed `ds.flower.fit` campaign evidence are preserved. Found by the
  run-at-pin utility campaign's live-federation cross-validation harness
  (`tools/campaign/run_cv_demo.R`), which no packaged suite reached at 0.4.1.
