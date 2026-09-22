# Hazard v3

Read `PROTOCOL.md` and the diagnosis in the packaged `hazard-v3` directory.
This user-side harness keeps the canonical campaign files and package code
unchanged. `run_cell.R` gives the existing federation helper a local binding
of the public `ds.flower.fit` call to pass the selected strategy. No package
namespace is modified. All selection cells use actual isolated DSLite workers.

The pod reuses its existing Ubuntu22.04/R4.6.1/Python3.11.10 installation.
`provision.sh` and `install_r.R` preserve the initial bootstrap whose hashes appear in provenance.
`resume_environment.sh` repairs the previously missing R dependencies and
runs the canonical install/freeze. The CPU requirements retain the campaign's
Torch2.4.1/Opacus1.5.2/Flower1.31.0 numerical versions; the complete actual
freeze is in evidence provenance. A CPU-only wheel replaces the old CUDA wheel.

From `/workspace/hazard`, in the foreground with single-thread numerical libs:

```
runtime/venv/bin/python dsFlowerClient/tools/campaign/survival/hazard_v3/test_protocol.py
runtime/venv/bin/python dsFlowerClient/tools/campaign/survival/hazard_v3/run.py /workspace/hazard prepare
runtime/venv/bin/python dsFlowerClient/tools/campaign/survival/hazard_v3/run.py /workspace/hazard pilot
runtime/venv/bin/python -u dsFlowerClient/tools/campaign/survival/hazard_v3/run.py /workspace/hazard run --jobs 4 --sweep-cutoff 2026-09-22T02:40:00+00:00
```

Preparation downloads checksum-pinned UCI SUPPORT2 and uses the canonical
`prepare_data.encode` and exact historical split rule. It freezes outer files
at ingestion. Development subsequently opens only site training files and
inner splits; a poisoned/missing outer holdout test guards this boundary.
Grid construction uses inner training events only, including seed1103.

`run_started.json` is exclusive: never remove it to rerun. The ordered sweep
may stop before a new wave at its recorded cutoff. Selection and confirmation
locks are exclusive files. A single confirmation includes full/two-site arms
and, if the predeclared time rule permits, both stress arms. An infrastructure
postprocessing repair must reuse the existing model and retain its incident;
it does not authorize another fit. Scored failures never authorize retuning.

`report.py` checks v3 schema extensions, selected configuration identity,
accounting geometry, independently verified budgets, clean complete federation,
package/runner/model identities and paired replicate coverage. It imports the
original accountant-validator primitive and metric/CI functions. The original
v1 matrix validator is deliberately not modified to admit the new matrix.

Only whitelisted released artifacts are archived. Runtime directories contain
node secrets and must never be copied recursively into evidence. The public
archive contains cell JSONs, all development scores, config grids, hashes,
selected model files, provenance, summary and the complete sweep CSV.

The first development wave encountered one startup failure before training.
`infrastructure_investigation.json` retains the evidence and recovery scope.
The driver amendment staggers starts by30seconds and uses `taskset` to assign
each whole federation one CPU. This matters because the trusted launcher's
clean environment intentionally excludes OMP/BLAS thread variables. A direct
probe confirms PyTorch uses one thread under one-CPU affinity. This changes
only OS scheduling. The original grid, models, split files and DP mechanism
remain frozen. `--resume-after-startup-failure` reuses executed cells and
permits only the named pre-training failure in a new attempt directory.
It cannot run after selection or confirmation has begun. The initial driver
and resumed driver hashes are both retained in provenance.

`diagnose_development.py` is a separate public, unnoised linear-SGD control,
planned before receiving a live development score. It compares full pooled
steps, pooled training limited to the site step count, unnoised equal-site
averaging, and its clipped counterpart. Direct per-patient gradients are checked
against the frozen autograd loss. These controls open inner data only, make
no DP or actual-federation claim, and cannot enter configuration selection.
