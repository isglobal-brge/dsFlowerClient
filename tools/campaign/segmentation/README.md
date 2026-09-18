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
export PYTHONPATH=/workspace/segmentation/runtime
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python feature_smoke.py --prepared /workspace/segmentation/prepared --out /workspace/segmentation/features
```

The preparation audit writes source mask/converted-mask SHA256, explicit empty declarations, source/patient censuses and all subject split hashes. The feature smoke verifies unchanged frozen state and subject retention on public fixtures without scoring. It writes public subject tensor caches only on the pod, never to package extdata. This is not a training/integration claim.

After all seven gate results and their evidence references exist in a JSON object with `segmentation_6_1_1` through `segmentation_6_1_7` set to `true`, run one full-size primary cell (repeat each cohort × epsilon 1,4,8 × seed 20260919,20260920,20260921). Package installs and environment paths must identify the paired commits.

```sh
export F_SEG_GATES_JSON=/workspace/segmentation/reviewed-gates.json
export F_SEG_RUNNER_PARENT=/workspace/segmentation/runtime
export DSFLOWER_VENV_ROOT=/workspace/segmentation/server-venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/segmentation/client-venv
export R_LIBS_USER=/workspace/segmentation/rlib
./run_federated.sh /workspace/segmentation/prepared/busbra /workspace/segmentation/prepared/busbra/split-20260919.json 8 20260919 /workspace/segmentation/runs/busbra-eps8-seed20260919
python score_public.py --features /workspace/segmentation/features/busbra --split /workspace/segmentation/prepared/busbra/split-20260919.json --probabilities /workspace/segmentation/runs/busbra-eps8-seed20260919/public-probabilities.csv --artifact /workspace/segmentation/runs/busbra-eps8-seed20260919/artifact/model.pt --out /workspace/segmentation/runs/busbra-eps8-seed20260919/channel-b.json
python central_twins.py --features /workspace/segmentation/features/busbra --split /workspace/segmentation/prepared/busbra/split-20260919.json --capture /workspace/segmentation/runs/busbra-eps8-seed20260919/public-capture --gates /workspace/segmentation/reviewed-gates.json --epsilon 8 --out /workspace/segmentation/runs/busbra-eps8-seed20260919/twins
```

The isolated benchmark hook is opt-in through `F_SEG_PUBLIC_BENCHMARK=1` and set by the R driver. It seeds/captures public initial arrays and records the existing node accountant history/step count plus effective tensor hashes. It does not alter sampling, clipping, noise or production telemetry. Its sidecar directory must be empty for a new cell; stale captures fail twin validation. Do not install this hook in a production environment. The central twins require all 15 actual node-round captures to match the selected public tensors and use the same captured initial arrays; pooled DP calls the trusted runtime `_dp_fit`. The nonprivate diagnostics use the identical model/loss/optimizer and Poisson geometry exclusively within benchmark tooling. Their scope is public data only.

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
