# Reproduce the survival campaign

Read `PROTOCOL_F_SURVIVAL.md` first. No scored model selection is permitted.
The scripts use public fixtures only. Never interpret the benchmark's repeated
privacy budgets as permission to repeat experiments on a private cohort.

1. Use paired survival revisions of dsFlower and dsFlowerClient in one workspace,
   with R 4.5, DSI/DSLite/dsBase/resourcer and normal package dependencies.
   Prepare `runtime/venv` with the coherent versions in
   `requirements-macos-arm64.txt` (Mac ARM64 execution record). The pod execution profile is recorded in
   `requirements-pod-linux-x86_64.txt`: Python 3.11.10, Torch 2.4.1+cu124,
   Opacus 1.5.2 and Flower 1.31.0. It inherits the matching image Torch/CUDA
   stack in an isolated venv; the freeze records the environment, rather than
   promising a portable installer. The pod used R 4.6.1; the Mac used R 4.5.
   Keep separate task-owned R libraries and record the actual versions. Set
   `UV_CACHE_DIR` within this workspace. Create `runtime/server/pytorch` as a
   link to `../venv`; package installs use only `runtime/rlib`.
2. Download the two URLs in the protocol to `data/survival/support2.csv` and
   `data/survival/lung1.csv`. The preparation script enforces exact SHA256s.
3. From the workspace root:
   ```sh
   python3 dsFlowerClient/tools/campaign/survival/prepare_data.py \
     --data data/survival --out data/survival/splits \
     --protocol dsFlowerClient/tools/campaign/survival/PROTOCOL_F_SURVIVAL.md
   python3 dsFlowerClient/tools/campaign/survival/prepare_synthetic.py \
     data/survival/splits/synthetic-public-1101 \
     dsFlowerClient/tools/campaign/survival/PROTOCOL_F_SURVIVAL.md
   dsFlowerClient/tools/campaign/survival/install_and_freeze.sh
   ```
   Wait for `FROZEN_INSTALL_READY`. Do not install concurrently with a run.
4. Run each variant `weibull`, `lognormal`, `hazard` on the synthetic fixture:
   ```sh
   Rscript dsFlowerClient/tools/campaign/survival/run_cell.R \
     "$PWD" "$PWD/data/survival/splits/synthetic-public-1101" \
     weibull 8 "$PWD/runtime/runs/final-synthetic-weibull" 2 1
   ```
   The two-round synthetic run includes released-artifact local prediction,
   pooled nonprivate/DP/null twins and independent accountant verification.
   Repeat using the matching variant in the output directory name.
   Archive each successful `evidence.json` as
   `inst/extdata/campaign/survival/cell-synthetic-<variant>.json` in the client
   repository. Keep every failed attempt separately, and use a new runtime
   directory for an investigated retry so the earlier attempt is preserved.
5. Only after all mechanism/regression gates and all three synthetic cells pass:
   ```sh
   python3 dsFlowerClient/tools/campaign/survival/run_matrix.py "$PWD" --jobs 1
   ```
   It executes the frozen 90-cell matrix, retains failures without scores, and
   summarizes matched replicate intervals and all four utility diagnostics.
   Review every flagged envelope against the matched model hashes and full-horizon
   accounting, then record the investigation in `summary.json`. Run
   `python3 dsFlowerClient/tools/campaign/survival/validate_completion.py
   dsFlowerClient/inst/extdata/campaign/survival` as the final completeness gate.
   Existing executed cells are reused; failed attempts require investigation
   before a separately recorded retry. Nothing pushes to a remote.

`run_cell.R` configures patient privacy inside isolated custodian DSLite workers.
It uses the existing real federation lifecycle, not array-only FedAvg simulation.
Diagnostics are recomputed from the **public** fixture manifest and audited
runtime. `verification_accountant=PRV` identifies the independent check; runtime
calibration tries PRV then RDP. It does not claim a new private telemetry API.

The nonprivate twin removes clipping/noise but keeps architecture, initialization,
likelihood, fixed feature transformation, expected batch divisor, optimizer and
round/epoch schedule. Its pooled subject population necessarily changes sampling
q/step counts relative to a site. The pooled DP diagnostic reuses `_dp_fit` with
secure, unrecorded random secrets; it has no public deterministic noise seed.
Both differences are explicit in JSON, as are package commits and runner hashes.

Failure JSONs from development are retained with their actual phase and known
limitations. A failure or incomplete matrix must not be reported as validated
utility. Claude decides promotion.

The runtime requires owner-only POSIX permissions for node secrets. If the pod
workspace is mounted on a filesystem that ignores chmod, use a task-owned
mode-0700 OS temporary directory as backing for `runtime/tmp` and `runtime/runs`
through workspace symlinks. Persistent sources, inputs, libraries and evidence
remain in the task workspace. Preserve released artifacts before recycling the
pod; never copy node-secret files into evidence.

For a FUSE-mounted workspace, repeated Python package-directory discovery can
also be slow. After verifying a task-owned environment copy byte-for-byte,
the pod retains its original environment and uses POSIX backing for Python.
Make `runtime/server` itself point to a POSIX directory containing the `pytorch`
environment link: `run_cell.R` resolves this root before passing it to custodian
workers. Resolve the root, not only its child symlink; an observed R directory
walk took 82.1 seconds through the FUSE alias and 0.13 seconds directly.
Change runtime placement only between cells and rerun readiness checks.
