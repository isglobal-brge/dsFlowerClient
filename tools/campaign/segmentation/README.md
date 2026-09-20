# Protocol v3 execution (current)

The binding reviewer authorized Adam 0.001, ten rounds × three local epochs (30 total), with round-local optimizer state and recalibrated noise at unchanged privacy settings. Both batch arms execute all 33 cells (66 fresh cells). The v2 SGD results remain invalid for the schedule defect; never use them for floors, envelopes or release evidence.

Use the fresh root `/workspace/segmentation/v3`, sharing only the immutable `prepared` and `features` trees via symlinks. Use the v3 tooling snapshot under `/workspace/segmentation/campaign-v3`. Launch one worker per arm with `run_matrix.py --root /workspace/segmentation/v3 --workers 1 --batch-size 16` and `--batch-size 64`. These share the existing two public federation slots. Logs have the `segmentation-v3-` prefix. Keep the v2 parent hold intact. The deployment launcher records exact commit and protocol hashes before any scored cell.

Twins derive active optimization pins from the captured v3 initialization, not the legacy feature-cache schedule. All 30 node-round captures must pass the unchanged tensor/source-census checks and the new three-local-epoch/30-total-epoch accounting checks. The prior synthetic smoke remains SGD with two one-epoch rounds, preserving its gate semantics.

Run the matrix under nohup and stop the session after launch; the reviewer requests evidence assembly later. No automatic assembly, floor verdict or render is scheduled. The commands below describe the earlier operational workflow; substitute the v3 execution root/tooling and 30-capture schedule for new runs. The archived protocol-v2.md is historical only.

# Segmentation public campaign

This directory contains the preregistered public benchmark tooling; it is outside the released runner. The archived `campaign-status.json` is the execution authority: a pending matrix has no scores and establishes no utility claim. The frozen protocol is `inst/extdata/campaign/segmentation/protocol.md` (workspace original `PROTOCOL_F_SEGMENTATION.md`).

Prerequisites: recorded §6.1 segmentation gates 1–7, byte-identical paired runner trees, the public pretrained checkpoint, working three-node R/DSLite stack, and Python torch/torchvision/Opacus/Flower/numpy/pandas/Pillow/scipy/requests/jsonschema. On the pod use the isolated `/workspace/segmentation/venv`, `/workspace/segmentation/rlib`, and `/workspace/segmentation/` data/work paths. The shell launcher requires a genuine `R_STACK_DONE` line in `/workspace/logs/install_r.log` or `/workspace/segmentation/r-stack-ready.log`. The latter records the scoped repair authorized by Addendum 1, after actual required-package and CUDA loading succeeds. A failed provisioning run is not a completion marker. Implementer gate verification does not constitute reviewer promotion.

Prepare unscored public fixtures:

```sh
python fetch_public_data.py --root /workspace/segmentation/data
python prepare_public_data.py --root /workspace/segmentation/data --out /workspace/segmentation/prepared
mkdir -p /workspace/segmentation/torch/hub/checkpoints
cp /workspace/segmentation/data/resnet18-f37072fd.pth /workspace/segmentation/torch/hub/checkpoints/
export TORCH_HOME=/workspace/segmentation/torch
export XDG_CACHE_HOME=/workspace/segmentation
export PYTHONPATH=/workspace/segmentation/runtime
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python feature_smoke.py --prepared /workspace/segmentation/prepared --out /workspace/segmentation/features
/workspace/segmentation/venv/bin/python install_public_observer.py
```

The preparation audit writes source mask/converted-mask SHA256, explicit empty declarations, source/patient censuses and all subject split hashes. The feature smoke verifies unchanged frozen state and subject retention on public fixtures without scoring. It writes public subject tensor caches only on the pod, never to package extdata. This is not a training/integration claim.

After all seven gate results and their evidence references exist in a JSON object with `segmentation_6_1_1` through `segmentation_6_1_7` set to `true`, run one full-size primary cell (repeat each cohort × epsilon 1,4,8 × seed 20260919,20260920,20260921). Package installs and environment paths must identify the paired commits.

```sh
export F_SEG_GATES_JSON=/workspace/segmentation/mechanism-gates.json
export F_SEG_RUNNER_PARENT=/workspace/segmentation/runtime
export DSFLOWER_VENV_ROOT=/workspace/segmentation/server-venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/segmentation/client-venv
export R_LIBS_USER=/workspace/segmentation/rlib
./run_federated.sh /workspace/segmentation/prepared/busbra /workspace/segmentation/prepared/busbra/split-20260919.json 8 20260919 /workspace/segmentation/runs/busbra-eps8-seed20260919
segmentation_artifact_dir=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' /workspace/segmentation/runs/busbra-eps8-seed20260919/federation-status.json)
python score_public.py --features /workspace/segmentation/features/busbra --split /workspace/segmentation/runs/busbra-eps8-seed20260919/effective-split.json --probabilities /workspace/segmentation/runs/busbra-eps8-seed20260919/public-probabilities.csv --artifact "$segmentation_artifact_dir/model.pt" --out /workspace/segmentation/runs/busbra-eps8-seed20260919/channel-b.json
python central_twins.py --features /workspace/segmentation/features/busbra --split /workspace/segmentation/runs/busbra-eps8-seed20260919/effective-split.json --capture /workspace/segmentation/runs/busbra-eps8-seed20260919/public-capture --gates "$F_SEG_GATES_JSON" --epsilon 8 --out /workspace/segmentation/runs/busbra-eps8-seed20260919/twins
```

The analyst-side benchmark hook is opt-in through `F_SEG_PUBLIC_BENCHMARK=1`, set by the R driver, and seeds/captures public initial arrays. Nodes keep the unchanged clean environment and mandatory integrity hook. The separate observer installer accepts only the dedicated segmentation venv; its deferred loader waits for mandatory runner hash verification. The driver writes an explicit public-cohort configuration beside each protected ephemeral node secret (0700 parent, 0600 configuration). Only then does the observer record actual accountant history/steps and effective tensor hashes. The supported `XDG_CACHE_HOME` route makes the pinned checkpoint available through the clean environment. Benchmark CPU threads are fixed at 2.

Neither observer alters sampling, clipping, noise or production telemetry. The capture directory must be empty for a new cell; stale captures fail twin validation. Do not install this instrumentation in a production environment. The central twins require all 15 actual node-round captures to match the selected public tensors and use the same captured initial arrays; pooled DP calls the trusted runtime `_dp_fit`. The nonprivate diagnostics use the identical model/loss/optimizer and Poisson geometry exclusively within benchmark tooling. Their scope is public data only.

The default CLI wires the full-size primary alpha=.5 matrix. Set `F_SEG_VARIANT=small192`, `heterogeneous`, or `bce` for the preregistered BUS-BRA extensions (the last two require epsilon8). The driver writes `effective-split.json`; use that file for the scoring and twin `--split` arguments for every variant. All campaign execution, final evidence assembly and cross-cell envelope checks remain pending actual scored cells. `segmentation_metrics.envelopes` accepts complete matched replicate results and refuses missing seeds; its positive shortfall is pooled nonprivate Dice minus federated DP Dice. Report failed floors and every failure/flag without deleting replicates. The JSON schema forbids pending/failed records from carrying fabricated scores.

Verification:

```sh
python -m unittest discover -s . -p 'test_*.py' -v
```

For executed cells archive only hashes/configurations/public summary metrics, runtime/accountant evidence and failure/cleanup status. Exclude benchmark secrets, public tensor caches and pretrained checkpoint bytes from commits. No governed data are downloaded or uploaded.

The synthetic instrumentation smoke is reproducible with a fresh directory under the workspace:

```sh
mkdir -p /workspace/segmentation/hook-smoke
F_SEG_PUBLIC_BENCHMARK=1 F_SEG_INIT_SEED=20260919 F_SEG_CAPTURE_DIR=/workspace/segmentation/hook-smoke PYTHONPATH="$PWD/benchmark_hooks:/workspace/segmentation/runtime:$PWD" python smoke_benchmark_hooks.py
```

This synthetic check does not score any dataset. It verifies that repeated captured public initialization matches and that the actual wrapped accountant observes exactly two logical Poisson steps in a four-subject/one-local-epoch fixture.

The actual three-node gate uses the same federation driver, with a deterministic generated fixture, 16 subjects per site, batch8 and two one-epoch rounds. Run it before writing gate 7 as passed:

```sh
python prepare_synthetic.py --out /workspace/segmentation/prepared/synthetic
F_SEG_SYNTHETIC=1 ./run_federated.sh /workspace/segmentation/prepared/synthetic /workspace/segmentation/prepared/synthetic/split-20260919.json 8 20260919 /workspace/segmentation/synthetic-run
python verify_synthetic.py --prepared /workspace/segmentation/prepared/synthetic --run /workspace/segmentation/synthetic-run --out /workspace/segmentation/synthetic-gate.json
```

After all seven gates pass, `run_matrix.py --root /workspace/segmentation --workers 2` executes all 33 preregistered primary, small192, BCE and heterogeneous cells, followed by their exact twins. Run it under `nohup`; each cell logs under `/workspace/logs/`. It refuses existing cell directories and retains failed attempts. Twin training uses CUDA; twin channel-B evaluation matches the public local predictor's CPU decoder arithmetic with the same cached CUDA encoder features.

Assemble the observed results without inventing missing scores:

```sh
python assemble_evidence.py --runs /workspace/segmentation/runs --provenance /path/to/inst/extdata/campaign/segmentation/provenance --runtime /workspace/segmentation/runtime-resume.json --protocol /path/to/inst/extdata/campaign/segmentation/protocol.md --out /workspace/segmentation/evidence
```

Runtime metadata records exact package commits, runner hash, dependencies, device and deterministic settings. The assembler checks every site's five distinct rounds, observed logical steps, independent full-horizon accounting, selected tensor hashes and scored artifact identity. Cohort evidence retains failed floors and envelope flags; campaign status separately lists every planned cell.

Protocol v2: run `run_matrix.py --root /workspace/segmentation --workers 2 --batch-size 16` and separately `--batch-size 64`. Each executes 33 cells into `runs-batch16` / `runs-batch64`; retain both arms (66 cells). Pass the same `--batch-size` to central_twins.py and assemble_evidence.py, and assemble each arm into a separate evidence directory. Direct federation selects the arm with `F_SEG_BATCH_SIZE=16` or `64`. All other frozen settings remain unchanged; see protocol.md amendment.

For a recorded nonzero R exit **after** successful saved prediction and cleanup, `resume_postprocessing.py --root ROOT --work CELL --batch-size 16|64` validates the existing model, split, initialization and all accountant captures, preserves the original execution status, then runs only channel-B and twins. It never retrains federation and refuses previously attempted postprocessing. Upload tool changes to a temporary filename and atomically rename; never overwrite an Rscript inode that an active interpreter may still be reading. Startup-only retries must preserve the original directory and prove absence of initialization, training and score artifacts first; `run_matrix.py --cell EXACT_PLANNED_NAME` retains the same existing-directory rejection.

After assembling both arms, place their public JSON archives under `inst/extdata/campaign/segmentation/batch16` and `batch64`, then run:

```sh
python combine_evidence.py --root /path/to/inst/extdata/campaign/segmentation
```

The combined v2 status validates all 66 planned cell identities, preserves failures, and references both arm archives by hash. It never pools scores or selects a winning arm. The pkgdown article shows successful observations from incomplete cohorts without presenting them as complete three-seed results.

The campaign pod's measured cgroup CPU quota is 7.65 cores despite 96 visible CPUs. `run_federated.sh` uses two POSIX public-federation slots on that pod; excess executor workers wait before invoking R. This bounds simultaneous federations without changing rounds, sampling, numerical thread settings or privacy controls. Scoped runtime console scripts use their verified POSIX interpreter directly to avoid repeated FUSE path resolution; their Python bodies and dependencies are unchanged.
