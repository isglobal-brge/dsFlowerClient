# Vision R5: diagnosis and selection under the privacy contract

Selected `sgd_m0.9_lr3_e2_b227_wd0` by the highest actual ε=8 federated-DP inner-validation AUC: **0.705486**, accuracy 0.678363, Brier 0.321637, log loss 11.076008. This exceeds the prespecified 0.60 diagnostic threshold.

This is a training-only diagnosis and schedule selection. No final cell was run. The fixed outer split remains 852 training patients and 212 test patients, unopened in R4/R5; the R4 inner split remains 681 fitting patients, three sites of 227, and 171 validation patients. All reported new evaluation uses cached training-patient features. The original R3/R4 records are retained byte-for-byte.

## Verified contract and the R4 failure

The unchanged contract is `pytorch_resnet18`: frozen ImageNet ResNet18, patient-mean 512-dimensional features and deterministic modal patient labels, two-logit affine head with its architectural FiniteClamp output in [-30, 30], five rounds, equal weight 1 for each of three sites, δ=10⁻⁶ and clipping norm 1. The backbone uses the runner's per-image preprocessing. Its pooled features are not normalized to [0, 1]: the cached maximum is approximately 8.587. Finite totalization bounds are [-10⁶, 10⁶].

For site population n and requested batch b, the runner uses s=ceil(n/b), Poisson inclusion q=1/s, expected-batch mean divisor max(1, floor(n/s)), and 5 × local_epochs × s accounted updates per site. The actual q is not b/n at ceil boundaries. Each sample gradient is coordinate-clamped to [-1, 1], then clipped to global L2 norm 1 across weight and bias; independent Gaussian noise is added to the summed gradients before division. The accountant calibrates replace-one privacy, converting to add/remove (ε/2, δ/(1+exp(ε/2))) before Opacus calibration. SGD momentum or Adam moments reset each round. The emulator retains finite-output derivatives, parameter saturation and weight decay after noising.

| Inner requested batch | Updates per local epoch | Executed q | Mean divisor |
|---:|---:|---:|---:|
| 32 | 8 | 0.125 | 28 |
| 64 | 4 | 0.25 | 56 |
| 128 | 2 | 0.5 | 113 |
| full site = 227 | 1 | 1 | 227 |

R4 pruned 25 schedules without privacy and selected 20 local epochs with Adam at 0.003/0.01, batch 32. Under the real inner contract those schedules require 800 updates per site, q=0.125 and noise multiplier 4.98046875; their recorded AUCs were 0.500940 and 0.478683. The high update count raises the accountant's required noise, and Adam's coordinate scaling changes the impact of noisy gradients. Non-private pruning therefore did not identify a useful private schedule. Those results motivated direct DP-aware ranking here. The R4 converged central logistic head reached 0.779405 ± 0.025903 inner-CV AUC, so the frozen representation carries signal.

## Emulator validation

The emulator and unchanged runner were compared over all five rounds using captured initial arrays, exact cached-feature/label hashes, the same accountant, and identical fresh independent per-site/per-round SecureNumpyRng sampling and Gaussian streams. The actual historical custodial secrets were not reconstructed. These numerical parity checks precede ranking.

The reordered outer feature cache did not match the captured inner-site tensor hashes. Inner training features were therefore re-extracted from the same raw inner collections in their exact manifest order and verified against the actual R4 captures. The outer cache remains the source for the unchanged 171 validation patients and the R3 training-only comparison. `inner-feature-parity.json` records this cache correction; no test data was involved.

| Historical schedule | Updates/site | σ | Maximum CUDA parameter error | Maximum CPU parameter error | Maximum probability error | Passed |
|---|---:|---:|---:|---:|---:|:---:|
| r3_registry_default | 45 | 1.459961 | 7.45e-09 | 7.45e-09 | 8.94e-08 | True |
| adam_lr0.003_e20_b32 | 800 | 4.980469 | 8.94e-08 | 8.94e-08 | 4.17e-07 | True |
| adam_lr0.01_e20_b32 | 800 | 4.980469 | 4.77e-07 | 3.87e-07 | 1.49e-06 | True |

The absolute pass tolerance is 10⁻⁴. Independent-noise comparisons use 12 fresh replicates per schedule:

| Comparison population/schedule | Historical AUC | Emulated mean ± sample SD | Emulated range | Historical AUC in range |
|---|---:|---:|---:|:---:|
| R4, 171 inner-validation patients; adam_m0_lr0.003_e20_b32_wd0 | 0.500940 | 0.494854 ± 0.007889 | 0.479624–0.510031 | True |
| R4, 171 inner-validation patients; adam_m0_lr0.01_e20_b32_wd0 | 0.478683 | 0.498785 ± 0.011613 | 0.480408–0.524138 | False |
| R3, 852 training patients; sgd_m0_lr0.001_e1_b32_wd0 | 0.551452 | 0.551472 ± 0.000660 | 0.550089–0.552669 | True |

The R3 default check uses its actual n=284 site mechanism (45 updates, q=1/9, divisor 31), captured initialization and training features. The previously published R3 test AUC 0.533 is context only: no test records or predictions were opened, and that test AUC was not recomputed. Historical-noise compatibility is descriptive; numerical parity is the emulator validation gate.

The additional FiniteClamp stress check verifies the output-clamp derivative in the high-learning-rate regime. Its recorded results are:

```json
{
  "passed": true,
  "cases": [
    {
      "name": "finite_clamp",
      "candidate": {
        "id": "sgd_m0_lr1_e1_b227_wd0",
        "optimizer": "sgd",
        "learning_rate": 1,
        "local_epochs": 1,
        "batch_size": 227,
        "momentum": 0,
        "weight_decay": 0,
        "batch_mode": "full_site"
      },
      "mechanism": {
        "adjacency": "replace_one",
        "noise_multiplier": 3.076171875,
        "clipping_norm": 1.0,
        "accounting_population": 227,
        "steps_per_epoch": 1,
        "sample_rate": 1.0,
        "expected_batch_size": 227,
        "total_epochs": 5,
        "total_steps": 5,
        "policy_hash": "8ac4f358373308a95e6f1564791fbbbee35a0e436d85b868adf424d244cb085d"
      },
      "saturation_fraction_at_initialization": 0.5,
      "cpu_parameter_max_abs": 2.2351741790771484e-08,
      "cpu_probability_max_abs": 0.0,
      "cuda_parameter_max_abs": 1.4901161193847656e-08,
      "cuda_probability_max_abs": 0.0,
      "passed": true
    },
    {
      "name": "sgd_momentum_weight_decay",
      "candidate": {
        "id": "sgd_m0.9_lr0.3_e2_b64_wd0.0001",
        "optimizer": "sgd",
        "learning_rate": 0.3,
        "local_epochs": 2,
        "batch_size": 64,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "batch_mode": "fixed"
      },
      "mechanism": {
        "adjacency": "replace_one",
        "noise_multiplier": 2.5048828125,
        "clipping_norm": 1.0,
        "accounting_population": 227,
        "steps_per_epoch": 4,
        "sample_rate": 0.25,
        "expected_batch_size": 56,
        "total_epochs": 10,
        "total_steps": 40,
        "policy_hash": "62da88f8ab6817e17faee26ec18cc3adec73c176d2d86c642ea44d3023b43fae"
      },
      "saturation_fraction_at_initialization": null,
      "cpu_parameter_max_abs": 2.384185791015625e-07,
      "cpu_probability_max_abs": 1.7881393432617188e-06,
      "cuda_parameter_max_abs": 2.384185791015625e-07,
      "cuda_probability_max_abs": 1.8477439880371094e-06,
      "passed": true
    },
    {
      "name": "adam_weight_decay",
      "candidate": {
        "id": "adam_m0_lr0.003_e2_b128_wd0.0001",
        "optimizer": "adam",
        "learning_rate": 0.003,
        "local_epochs": 2,
        "batch_size": 128,
        "momentum": 0,
        "weight_decay": 0.0001,
        "batch_mode": "fixed"
      },
      "mechanism": {
        "adjacency": "replace_one",
        "noise_multiplier": 3.349609375,
        "clipping_norm": 1.0,
        "accounting_population": 227,
        "steps_per_epoch": 2,
        "sample_rate": 0.5,
        "expected_batch_size": 113,
        "total_epochs": 10,
        "total_steps": 20,
        "policy_hash": "d26549e904820df9231ba495202d495e7b4343c8dc4f64508fc75323a30d7c95"
      },
      "saturation_fraction_at_initialization": null,
      "cpu_parameter_max_abs": 7.450580596923828e-09,
      "cpu_probability_max_abs": 5.21540641784668e-08,
      "cuda_parameter_max_abs": 7.450580596923828e-09,
      "cuda_probability_max_abs": 4.470348358154297e-08,
      "passed": true
    }
  ],
  "tolerance_absolute": 0.0001,
  "test_accessed": false,
  "scope": "One full local round per case, identical secure sampling/noise; unchanged CUDA _dp_fit compared with CPU and CUDA emulator."
}
```

A further check covers all three shortlisted schedules, all three search seeds and all five rounds, comparing unchanged CUDA training with CPU/CUDA emulation. It checks native prediction ranks because tiny saturated probabilities can hide ranking errors in an absolute probability tolerance. Maximum observed AUC difference: 0; maximum cross-class pair-order disagreements: 0. Full results are in `shortlist-validation.json`.

Installed Flower normalizes each unit weight to 1/3 before addition; the emulator adds then divides. A separate five-round check uses the installed Flower aggregator and unchanged CUDA training for all shortlisted schedules and search seeds. It also tests all six final-round reply orders on the same node arrays. Maximum AUC difference is 0; maximum parameter error is 3.13e-07. `aggregation-validation.json` records this float32 arithmetic check. Earlier-round arrival orders use canonical site order; historical secret streams are not replayed.

## Search and real confirmation

The complete declared grid has 736 schedules: SGD with momentum 0 or 0.9 at learning rates {0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}; Adam at {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1}; local epochs {1, 2, 4, 8}; batch {32, 64, 128, full site}; weight decay {0, 0.0001}. Scheduler is none, L1 is zero, and five rounds are fixed. Ranking uses ε=8 mean inner-validation AUC over three independent noise/sampling seeds 20260923, 20260924, 20260925; the captured initialization seed is 20260922. Ties use ascending candidate ID.

The three highest emulated means were confirmed once each through the actual dsFlower federation. All 15 node-round captures per federation verify feature/target hashes, optimizer pins, observed updates, privacy configuration and accountant geometry. Selection uses the highest real ε=8 AUC; the same ID tie-break applies.

| Actual candidate | ε | Updates/site | σ | AUC | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| adam_m0_lr0.03_e1_b227_wd0 | 8 | 5 | 3.076172 | 0.670063 | 0.678363 | 0.321311 | 2.491522 |
| sgd_m0.9_lr3_e2_b227_wd0 | 8 | 10 | 4.355469 | 0.705486 | 0.678363 | 0.321637 | 11.076008 |
| sgd_m0.9_lr3_e2_b227_wd0.0001 | 8 | 10 | 4.355469 | 0.612226 | 0.678363 | 0.321637 | 11.108963 |

The largest confirmation discrepancy is `sgd_m0.9_lr3_e2_b227_wd0.0001`: emulated mean AUC 0.709039, real AUC 0.612226. Three emulated seeds did not characterize the full observed outcome spread. The real runs use fresh custodial randomness, and one real fit per candidate cannot separate a small weight-decay effect from noise-realization variability. The actual feature, mechanism and optimizer checks passed; no scored configuration was repeated or adjusted.

The third/fourth emulated mean gap is only 0.000052247. The shortlist is the declared numerical ranking, not evidence that the third schedule is materially better than the fourth. In emulation, all three shortlisted candidates have threshold-0.5 accuracy equal to the inner majority rate (116/171 = 0.678363), Brier about 0.3214–0.3216, and log loss about 2.5–11.1. A positive PROCEED decision establishes above-chance discrimination under the contract; it does not establish calibrated probabilities or useful threshold-0.5 classification.

The best real ε=8 AUC exceeds 0.60, so the prespecified conditional ε=4/ε=1 diagnosis was not triggered.

These are selection estimates: 736 schedules were ranked on the same 171 validation patients, and the best of three real noisy fits was chosen on those patients. Their AUCs are optimistic for a future independent evaluation. One real fit per candidate does not estimate real-fit noise variance, and the three-seed emulation SD is not a patient-sampling confidence interval. The quoted ε is per federation, not a composed privacy budget for the entire public-data benchmark search and selection process.

## Selected schedule and implementation comparator definitions

Selected optimizer **sgd**, learning rate **3**, momentum **0.9**, local epochs **2**, batch **full site (227)**, weight decay **0**, five rounds. At the inner ε=8 contract: 10 updates/site, q=1, divisor 227, σ=4.355469. The noise standard deviation per mean-gradient coordinate is 0.019187, compared with 0.177874 in R4. Batch geometry, learning rate and optimizer all change across these schedules; this comparison does not isolate an individual causal effect.

The batch search dimension `full site` is a prespecified logical schedule choice: 227 on the inner sites, 284 on each original training site, and 852 for the pooled-DP twin. It retains q=1 and the same updates per local epoch. Numeric batch choices 32, 64 and 128 remain literal when moving to the full training cohort. This rule is fixed at diagnosis; no implementation or selected-schedule fit on the outer cohort has been performed. The R3 check above was diagnostic replay of its default schedule on training features only.

For this selected schedule the later full-cohort federation would use batch 284, 10 updates/site, q=1, divisor 284. Its noise multiplier must be recalibrated for that mechanism. The pooled-DP twin would use batch 852, 10 updates, q=1, divisor 852; it requires its own n=852 accountant calibration.

| Comparator | Fixed definition for the implementation |
|---|---|
| Converged central logistic head | Fit a pooled C=1 L2 logistic head on the 852 frozen patient feature vectors; unpenalized intercept, converged solver, same malignant-positive task. This is a representation/optimization comparator, not a finite-step twin. |
| Non-private federated finite-schedule twin | Three original 284-patient sites, the selected logical schedule, same initial affine head and architectural FiniteClamp, same Poisson sampling geometry, expected-batch division, equal site weights and optimizer reset each round. Remove DP per-gradient coordinate/global-norm clipping and Gaussian noise; retain the selected regularization and public numerical totalization. |
| Pooled-DP finite-schedule twin | Pool all 852 training patients; same selected optimizer, learning rate, local epochs, five optimizer-reset rounds, logical batch rule, initial head, architecture, patient preprocessing and DP gradient operations. Recompute q, divisor, updates and σ using n=852 with its own ε/δ accountant. Do not reuse the inner-site noise multiplier. |
| Trivial majority | For primary image metrics, predict the training-image majority class and training-image prevalence of malignancy as the constant probability, matching R4. For secondary patient metrics, use the 852-patient training majority and prevalence. Determine neither class nor probability from test patients. |

The implementation retains the previously declared per-image primary metrics, patient-level secondary metrics and the already fixed 852/212 split. Selection here uses patient-level AUC. Implementation training must extract features afresh; diagnosis caches remain diagnostic inputs only. These comparator definitions do not authorize test access within this diagnosis.

## Pod and record state

Runtime/package verification is recorded in `runtime-preflight.json`:

```json
{
  "blocker_kind": null,
  "elapsed_s": 44.169,
  "error": null,
  "expected_runner_sha256": "2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724",
  "expected_versions": {
    "dsFlower": "0.5.1",
    "dsFlowerClient": "0.5.0"
  },
  "filesystem_probes": [
    {
      "argv": [
        "stat",
        "--",
        "/workspace"
      ],
      "elapsed_s": 0.004,
      "returncode": 0,
      "stderr": "",
      "stdout": "  File: /workspace\n  Size: 3002352   \tBlocks: 5864       IO Block: 65536  directory\nDevice: 50h/80d\tInode: 63964822    Links: 3\nAccess: (0777/drwxrwxrwx)  Uid: (    0/    root)   Gid: (    0/    root)\nAccess: 2026-09-22 00:17:01.000000000 +0000\nModify: 2026-09-22 00:18:24.000000000 +0000\nChange: 2026-09-22 00:18:24.000000000 +0000\n Birth: -\n",
      "timed_out": false,
      "timeout_s": 10
    },
    {
      "argv": [
        "stat",
        "--",
        "/workspace/cells-vision/Rlib/dsFlower/DESCRIPTION"
      ],
      "elapsed_s": 0.006,
      "returncode": 0,
      "stderr": "",
      "stdout": "  File: /workspace/cells-vision/Rlib/dsFlower/DESCRIPTION\n  Size: 1796      \tBlocks: 4          IO Block: 65536  regular file\nDevice: 50h/80d\tInode: 197603666   Links: 1\nAccess: (0666/-rw-rw-rw-)  Uid: (    0/    root)   Gid: (    0/    root)\nAccess: 2026-09-22 01:19:28.000000000 +0000\nModify: 2026-09-22 01:19:25.000000000 +0000\nChange: 2026-09-22 01:19:50.000000000 +0000\n Birth: -\n",
      "timed_out": false,
      "timeout_s": 10
    },
    {
      "argv": [
        "stat",
        "--",
        "/workspace/cells-vision/prepared/busbra/audit.json"
      ],
      "elapsed_s": 0.006,
      "returncode": 0,
      "stderr": "",
      "stdout": "  File: /workspace/cells-vision/prepared/busbra/audit.json\n  Size: 407937    \tBlocks: 797        IO Block: 65536  regular file\nDevice: 50h/80d\tInode: 108898944   Links: 1\nAccess: (0666/-rw-rw-rw-)  Uid: (    0/    root)   Gid: (    0/    root)\nAccess: 2026-09-22 01:04:18.000000000 +0000\nModify: 2026-09-22 00:26:43.000000000 +0000\nChange: 2026-09-22 00:26:43.000000000 +0000\n Birth: -\n",
      "timed_out": false,
      "timeout_s": 10
    }
  ],
  "generated_at": "2026-09-22T06:53:41.134245+00:00",
  "gpu_probe": {
    "argv": [
      "nvidia-smi",
      "--query-gpu=name,uuid,memory.total,memory.used",
      "--format=csv"
    ],
    "elapsed_s": 0.031,
    "returncode": 0,
    "stderr": "",
    "stdout": "name, uuid, memory.total [MiB], memory.used [MiB]\nNVIDIA A40, GPU-71ab6101-1d6b-147c-5ee3-af1fc7a677ee, 46068 MiB, 0 MiB\n",
    "timed_out": false,
    "timeout_s": 10
  },
  "gpus": [
    {
      "memory.total [MiB]": "46068 MiB",
      "memory.used [MiB]": "0 MiB",
      "name": "NVIDIA A40",
      "uuid": "GPU-71ab6101-1d6b-147c-5ee3-af1fc7a677ee"
    }
  ],
  "hostname": "5dc98e434db0",
  "installed_runner_sha256": {
    "dsFlower": "2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724",
    "dsFlowerClient": "2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724"
  },
  "installed_versions": {
    "dsFlower": "0.5.1",
    "dsFlowerClient": "0.5.0"
  },
  "r_version": "R version 4.6.1 (2026-06-24)",
  "release_probe": {
    "argv": [
      "Rscript",
      "-e",
      "args <- commandArgs(TRUE)\n.libPaths(c(args[[1]], .libPaths()))\nlibrary(dsFlower)\nlibrary(dsFlowerClient)\ncat(jsonlite::toJSON(list(\n  r_version = R.version.string,\n  versions = list(dsFlower = as.character(packageVersion(\"dsFlower\")),\n                  dsFlowerClient = as.character(packageVersion(\"dsFlowerClient\"))),\n  runner_sha256 = list(dsFlower = dsFlower:::.compute_harness_hash(),\n                      dsFlowerClient = dsFlowerClient:::.compute_local_runner_hash())),\n  auto_unbox = TRUE))\n",
      "/workspace/cells-vision/Rlib"
    ],
    "elapsed_s": 2.127,
    "returncode": 0,
    "stderr": "dsFlower v0.5.1 loaded.\ndsFlowerClient 0.5.0 -- Python environment OK\n",
    "stdout": "{\"r_version\":\"R version 4.6.1 (2026-06-24)\",\"versions\":{\"dsFlower\":\"0.5.1\",\"dsFlowerClient\":\"0.5.0\"},\"runner_sha256\":{\"dsFlower\":\"2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724\",\"dsFlowerClient\":\"2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724\"}}",
    "timed_out": false,
    "timeout_s": 45
  },
  "schema": "dsflower-vision-runtime-preflight-v1",
  "scoring_started": false,
  "status": "verified",
  "torch_probe": {
    "argv": [
      "/workspace/cells-vision/venvs/pytorch-gpu/bin/python",
      "-c",
      "import json, torch, torchvision; assert torch.cuda.is_available(); print(json.dumps(dict(torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())))"
    ],
    "elapsed_s": 41.993,
    "returncode": 0,
    "stderr": "",
    "stdout": "{\"torch\": \"2.6.0+cu124\", \"torchvision\": \"0.21.0+cu124\", \"cuda\": \"12.4\", \"gpu\": \"NVIDIA A40\"}\n",
    "timed_out": false,
    "timeout_s": 45
  },
  "training_started": false,
  "workspace_mount": [
    "1584 1563 0:80 /podvolumes/5yue4r7xuzpm/0nk8si7zupfczc /workspace rw,nosuid,nodev,relatime - fuse mfs\\043eu-se-1.runpod.net:9421 rw,user_id=0,group_id=0,allow_other"
  ]
}
```

Final package versions and both installed runner hashes were reverified in `runner-final.json`. The original 45-second Torch import probe timed out without reporting an import error; its record is preserved in `runner-final-import-timeout.json`. The identical import probe passed with a 120-second allowance. No package or model was changed or rerun.

Final pod state is recorded in `pod-final.json`: pod left running = True; matching training processes = 0. name, memory.used [MiB]; NVIDIA A40, 0 MiB

The isolated vision pod is `/workspace/cells-vision` on `pod-flower-vision` (A40). The real federation drivers record successful SuperLink/SuperNode cleanup. The pod is left running; no other pod is part of this work. Package source is unchanged. R3/R4 records and failed-attempt evidence are preserved; R5 tooling and new records are separate. `preservation.json` verifies 111 pre-existing local vision files byte-for-byte, including pre-existing uncommitted R4 working edits. Those existing edits remain unchanged and outside the R5 commit.

## Complete ranked DP-aware grid

Mean ± SD is across the three emulated ε=8 noise seeds. Updates and σ are per inner site over all five rounds. A dash means no actual federation was run for that candidate. The table contains every declared candidate in the recorded rank order.

| Rank | Candidate | Optimizer | Momentum | LR | Local epochs | Batch | Weight decay | Updates/site | σ | Emulated AUC mean ± SD | Real ε8 AUC |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | adam_m0_lr0.03_e1_b227_wd0 | adam | 0 | 0.03 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.709979 ± 0.027940 | 0.670063 |
| 2 | sgd_m0.9_lr3_e2_b227_wd0 | sgd | 0.9 | 3 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.709143 ± 0.008643 | 0.705486 |
| 3 | sgd_m0.9_lr3_e2_b227_wd0.0001 | sgd | 0.9 | 3 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.709039 ± 0.008868 | 0.612226 |
| 4 | adam_m0_lr0.03_e1_b227_wd0.0001 | adam | 0 | 0.03 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.708986 ± 0.029294 | — |
| 5 | adam_m0_lr0.1_e1_b227_wd0 | adam | 0 | 0.1 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.700888 ± 0.009764 | — |
| 6 | adam_m0_lr0.1_e1_b227_wd0.0001 | adam | 0 | 0.1 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.697388 ± 0.014041 | — |
| 7 | sgd_m0.9_lr3_e1_b128_wd0 | sgd | 0.9 | 3 | 1 | 128 | 0 | 10 | 2.529297 | 0.696761 ± 0.005717 | — |
| 8 | sgd_m0.9_lr3_e1_b128_wd0.0001 | sgd | 0.9 | 3 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.696761 ± 0.005396 | — |
| 9 | adam_m0_lr0.1_e2_b227_wd0 | adam | 0 | 0.1 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.693103 ± 0.017732 | — |
| 10 | adam_m0_lr0.1_e2_b227_wd0.0001 | adam | 0 | 0.1 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.692999 ± 0.017186 | — |
| 11 | sgd_m0_lr3_e1_b128_wd0.0001 | sgd | 0 | 3 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.674765 ± 0.023450 | — |
| 12 | sgd_m0_lr3_e1_b128_wd0 | sgd | 0 | 3 | 1 | 128 | 0 | 10 | 2.529297 | 0.672466 ± 0.026435 | — |
| 13 | adam_m0_lr0.1_e1_b128_wd0 | adam | 0 | 0.1 | 1 | 128 | 0 | 10 | 2.529297 | 0.645820 ± 0.102131 | — |
| 14 | adam_m0_lr0.1_e1_b128_wd0.0001 | adam | 0 | 0.1 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.645298 ± 0.101018 | — |
| 15 | adam_m0_lr0.1_e4_b227_wd0 | adam | 0 | 0.1 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.622466 ± 0.027302 | — |
| 16 | adam_m0_lr0.1_e4_b227_wd0.0001 | adam | 0 | 0.1 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.622466 ± 0.027629 | — |
| 17 | adam_m0_lr0.003_e2_b227_wd0 | adam | 0 | 0.003 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.613427 ± 0.009718 | — |
| 18 | adam_m0_lr0.003_e2_b227_wd0.0001 | adam | 0 | 0.003 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.613427 ± 0.009718 | — |
| 19 | adam_m0_lr0.003_e1_b128_wd0 | adam | 0 | 0.003 | 1 | 128 | 0 | 10 | 2.529297 | 0.611285 ± 0.009046 | — |
| 20 | adam_m0_lr0.003_e1_b128_wd0.0001 | adam | 0 | 0.003 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.611285 ± 0.009046 | — |
| 21 | sgd_m0_lr3_e2_b64_wd0 | sgd | 0 | 3 | 2 | 64 | 0 | 40 | 2.504883 | 0.609248 ± 0.034888 | — |
| 22 | adam_m0_lr0.1_e1_b64_wd0.0001 | adam | 0 | 0.1 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.608203 ± 0.024428 | — |
| 23 | adam_m0_lr0.1_e1_b64_wd0 | adam | 0 | 0.1 | 1 | 64 | 0 | 20 | 1.967773 | 0.608098 ± 0.024685 | — |
| 24 | sgd_m0_lr3_e2_b64_wd0.0001 | sgd | 0 | 3 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.607576 ± 0.030767 | — |
| 25 | sgd_m0_lr3_e1_b64_wd0 | sgd | 0 | 3 | 1 | 64 | 0 | 20 | 1.967773 | 0.606949 ± 0.006940 | — |
| 26 | adam_m0_lr0.001_e8_b227_wd0.0001 | adam | 0 | 0.001 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.606740 ± 0.002719 | — |
| 27 | adam_m0_lr0.001_e8_b227_wd0 | adam | 0 | 0.001 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.606635 ± 0.002892 | — |
| 28 | adam_m0_lr0.001_e4_b128_wd0 | adam | 0 | 0.001 | 4 | 128 | 0 | 40 | 4.560547 | 0.606583 ± 0.003560 | — |
| 29 | adam_m0_lr0.001_e4_b128_wd0.0001 | adam | 0 | 0.001 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.606583 ± 0.003560 | — |
| 30 | adam_m0_lr0.001_e2_b64_wd0 | adam | 0 | 0.001 | 2 | 64 | 0 | 40 | 2.504883 | 0.605799 ± 0.002618 | — |
| 31 | adam_m0_lr0.001_e2_b64_wd0.0001 | adam | 0 | 0.001 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.605747 ± 0.002548 | — |
| 32 | adam_m0_lr0.003_e1_b227_wd0 | adam | 0 | 0.003 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.602874 ± 0.001727 | — |
| 33 | adam_m0_lr0.003_e1_b227_wd0.0001 | adam | 0 | 0.003 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.602821 ± 0.001808 | — |
| 34 | adam_m0_lr0.003_e1_b64_wd0.0001 | adam | 0 | 0.003 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.601724 ± 0.010123 | — |
| 35 | adam_m0_lr0.003_e1_b64_wd0 | adam | 0 | 0.003 | 1 | 64 | 0 | 20 | 1.967773 | 0.601620 ± 0.009980 | — |
| 36 | sgd_m0_lr3_e8_b227_wd0 | sgd | 0 | 3 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.601254 ± 0.021129 | — |
| 37 | adam_m0_lr0.003_e2_b128_wd0.0001 | adam | 0 | 0.003 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.600993 ± 0.010269 | — |
| 38 | adam_m0_lr0.003_e2_b128_wd0 | adam | 0 | 0.003 | 2 | 128 | 0 | 20 | 3.349609 | 0.600836 ± 0.010269 | — |
| 39 | sgd_m0.9_lr1_e1_b128_wd0 | sgd | 0.9 | 1 | 1 | 128 | 0 | 10 | 2.529297 | 0.599478 ± 0.039496 | — |
| 40 | sgd_m0.9_lr1_e1_b128_wd0.0001 | sgd | 0.9 | 1 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.599373 ± 0.039537 | — |
| 41 | sgd_m0_lr3_e1_b64_wd0.0001 | sgd | 0 | 3 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.599216 ± 0.010143 | — |
| 42 | sgd_m0_lr0.03_e1_b64_wd0 | sgd | 0 | 0.03 | 1 | 64 | 0 | 20 | 1.967773 | 0.599112 ± 0.004656 | — |
| 43 | sgd_m0_lr0.03_e1_b64_wd0.0001 | sgd | 0 | 0.03 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.599112 ± 0.004656 | — |
| 44 | adam_m0_lr0.003_e4_b227_wd0 | adam | 0 | 0.003 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.598746 ± 0.006729 | — |
| 45 | adam_m0_lr0.003_e4_b227_wd0.0001 | adam | 0 | 0.003 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.598694 ± 0.006800 | — |
| 46 | sgd_m0_lr0.03_e2_b128_wd0 | sgd | 0 | 0.03 | 2 | 128 | 0 | 20 | 3.349609 | 0.598433 ± 0.004594 | — |
| 47 | sgd_m0_lr0.03_e2_b128_wd0.0001 | sgd | 0 | 0.03 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.598433 ± 0.004594 | — |
| 48 | adam_m0_lr0.001_e1_b32_wd0 | adam | 0 | 0.001 | 1 | 32 | 0 | 40 | 1.523438 | 0.598119 ± 0.005196 | — |
| 49 | adam_m0_lr0.001_e1_b32_wd0.0001 | adam | 0 | 0.001 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.598119 ± 0.005196 | — |
| 50 | adam_m0_lr0.01_e1_b227_wd0 | adam | 0 | 0.01 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.598119 ± 0.009867 | — |
| 51 | sgd_m0_lr0.03_e4_b227_wd0 | sgd | 0 | 0.03 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.598119 ± 0.003123 | — |
| 52 | sgd_m0_lr0.03_e4_b227_wd0.0001 | sgd | 0 | 0.03 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.598119 ± 0.003123 | — |
| 53 | adam_m0_lr0.01_e1_b227_wd0.0001 | adam | 0 | 0.01 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.597492 ± 0.010241 | — |
| 54 | sgd_m0.9_lr3_e2_b128_wd0.0001 | sgd | 0.9 | 3 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.596813 ± 0.036998 | — |
| 55 | sgd_m0.9_lr3_e2_b128_wd0 | sgd | 0.9 | 3 | 2 | 128 | 0 | 20 | 3.349609 | 0.596499 ± 0.036206 | — |
| 56 | sgd_m0.9_lr0.1_e1_b227_wd0 | sgd | 0.9 | 0.1 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.596499 ± 0.002761 | — |
| 57 | sgd_m0.9_lr0.1_e1_b227_wd0.0001 | sgd | 0.9 | 0.1 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.596499 ± 0.002761 | — |
| 58 | sgd_m0_lr0.1_e1_b227_wd0 | sgd | 0 | 0.1 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.596499 ± 0.002761 | — |
| 59 | sgd_m0_lr0.1_e1_b227_wd0.0001 | sgd | 0 | 0.1 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.596499 ± 0.002761 | — |
| 60 | adam_m0_lr0.001_e2_b32_wd0 | adam | 0 | 0.001 | 2 | 32 | 0 | 80 | 1.875000 | 0.595977 ± 0.007434 | — |
| 61 | adam_m0_lr0.001_e2_b32_wd0.0001 | adam | 0 | 0.001 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.595977 ± 0.007434 | — |
| 62 | adam_m0_lr0.001_e4_b64_wd0 | adam | 0 | 0.001 | 4 | 64 | 0 | 80 | 3.320312 | 0.595716 ± 0.008152 | — |
| 63 | adam_m0_lr0.001_e4_b64_wd0.0001 | adam | 0 | 0.001 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.595716 ± 0.008152 | — |
| 64 | sgd_m0_lr0.1_e2_b227_wd0 | sgd | 0 | 0.1 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.594775 ± 0.000239 | — |
| 65 | sgd_m0_lr0.1_e2_b227_wd0.0001 | sgd | 0 | 0.1 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.594775 ± 0.000239 | — |
| 66 | sgd_m0.9_lr3_e4_b227_wd0 | sgd | 0.9 | 3 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.594148 ± 0.010748 | — |
| 67 | sgd_m0_lr0.1_e1_b128_wd0 | sgd | 0 | 0.1 | 1 | 128 | 0 | 10 | 2.529297 | 0.594044 ± 0.003466 | — |
| 68 | sgd_m0_lr0.1_e1_b128_wd0.0001 | sgd | 0 | 0.1 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.594044 ± 0.003466 | — |
| 69 | sgd_m0.9_lr0.01_e1_b64_wd0 | sgd | 0.9 | 0.01 | 1 | 64 | 0 | 20 | 1.967773 | 0.593992 ± 0.005399 | — |
| 70 | sgd_m0.9_lr0.01_e1_b64_wd0.0001 | sgd | 0.9 | 0.01 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.593939 ± 0.005360 | — |
| 71 | sgd_m0_lr0.01_e4_b64_wd0 | sgd | 0 | 0.01 | 4 | 64 | 0 | 80 | 3.320312 | 0.593783 ± 0.001870 | — |
| 72 | sgd_m0_lr0.01_e4_b64_wd0.0001 | sgd | 0 | 0.01 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.593783 ± 0.001870 | — |
| 73 | adam_m0_lr0.001_e8_b128_wd0.0001 | adam | 0 | 0.001 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.593260 ± 0.008951 | — |
| 74 | sgd_m0_lr3_e4_b128_wd0 | sgd | 0 | 3 | 4 | 128 | 0 | 40 | 4.560547 | 0.593260 ± 0.025236 | — |
| 75 | adam_m0_lr0.001_e8_b128_wd0 | adam | 0 | 0.001 | 8 | 128 | 0 | 80 | 6.308594 | 0.593208 ± 0.008875 | — |
| 76 | sgd_m0.9_lr3_e4_b227_wd0.0001 | sgd | 0.9 | 3 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.593103 ± 0.012699 | — |
| 77 | sgd_m0_lr0.01_e2_b32_wd0 | sgd | 0 | 0.01 | 2 | 32 | 0 | 80 | 1.875000 | 0.592999 ± 0.002942 | — |
| 78 | sgd_m0_lr0.01_e2_b32_wd0.0001 | sgd | 0 | 0.01 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.592999 ± 0.002942 | — |
| 79 | adam_m0_lr0.0003_e8_b64_wd0 | adam | 0 | 0.0003 | 8 | 64 | 0 | 160 | 4.531250 | 0.592947 ± 0.003585 | — |
| 80 | adam_m0_lr0.0003_e8_b64_wd0.0001 | adam | 0 | 0.0003 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.592947 ± 0.003585 | — |
| 81 | adam_m0_lr0.0003_e8_b32_wd0 | adam | 0 | 0.0003 | 8 | 32 | 0 | 320 | 3.261719 | 0.592529 ± 0.004236 | — |
| 82 | adam_m0_lr0.0003_e8_b32_wd0.0001 | adam | 0 | 0.0003 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.592529 ± 0.004236 | — |
| 83 | sgd_m0_lr0.01_e8_b128_wd0 | sgd | 0 | 0.01 | 8 | 128 | 0 | 80 | 6.308594 | 0.592268 ± 0.002284 | — |
| 84 | sgd_m0_lr0.01_e8_b128_wd0.0001 | sgd | 0 | 0.01 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.592268 ± 0.002284 | — |
| 85 | sgd_m0.9_lr0.01_e2_b128_wd0 | sgd | 0.9 | 0.01 | 2 | 128 | 0 | 20 | 3.349609 | 0.591797 ± 0.004533 | — |
| 86 | sgd_m0.9_lr0.01_e2_b128_wd0.0001 | sgd | 0.9 | 0.01 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.591797 ± 0.004533 | — |
| 87 | sgd_m0_lr3_e2_b128_wd0 | sgd | 0 | 3 | 2 | 128 | 0 | 20 | 3.349609 | 0.590805 ± 0.041656 | — |
| 88 | sgd_m0.9_lr0.01_e4_b227_wd0 | sgd | 0.9 | 0.01 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.590230 ± 0.003935 | — |
| 89 | sgd_m0.9_lr0.01_e4_b227_wd0.0001 | sgd | 0.9 | 0.01 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.590230 ± 0.003935 | — |
| 90 | adam_m0_lr0.001_e4_b227_wd0 | adam | 0 | 0.001 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.589289 ± 0.002509 | — |
| 91 | adam_m0_lr0.001_e4_b227_wd0.0001 | adam | 0 | 0.001 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.589289 ± 0.002509 | — |
| 92 | sgd_m0.9_lr0.03_e1_b128_wd0 | sgd | 0.9 | 0.03 | 1 | 128 | 0 | 10 | 2.529297 | 0.588871 ± 0.005236 | — |
| 93 | sgd_m0.9_lr0.03_e1_b128_wd0.0001 | sgd | 0.9 | 0.03 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.588871 ± 0.005236 | — |
| 94 | sgd_m0.9_lr0.03_e1_b64_wd0 | sgd | 0.9 | 0.03 | 1 | 64 | 0 | 20 | 1.967773 | 0.588715 ± 0.004261 | — |
| 95 | sgd_m0.9_lr0.03_e1_b64_wd0.0001 | sgd | 0.9 | 0.03 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.588715 ± 0.004261 | — |
| 96 | sgd_m0_lr0.03_e4_b128_wd0.0001 | sgd | 0 | 0.03 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.588297 ± 0.003610 | — |
| 97 | sgd_m0_lr0.03_e4_b128_wd0 | sgd | 0 | 0.03 | 4 | 128 | 0 | 40 | 4.560547 | 0.588245 ± 0.003686 | — |
| 98 | sgd_m0_lr0.03_e8_b227_wd0 | sgd | 0 | 0.03 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.588192 ± 0.002984 | — |
| 99 | sgd_m0_lr0.03_e8_b227_wd0.0001 | sgd | 0 | 0.03 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.588192 ± 0.002984 | — |
| 100 | sgd_m0.9_lr0.03_e4_b227_wd0 | sgd | 0.9 | 0.03 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.588036 ± 0.003381 | — |
| 101 | sgd_m0.9_lr0.03_e4_b227_wd0.0001 | sgd | 0.9 | 0.03 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.588036 ± 0.003381 | — |
| 102 | sgd_m0.9_lr0.03_e2_b128_wd0 | sgd | 0.9 | 0.03 | 2 | 128 | 0 | 20 | 3.349609 | 0.587983 ± 0.006398 | — |
| 103 | sgd_m0.9_lr0.03_e2_b128_wd0.0001 | sgd | 0.9 | 0.03 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.587983 ± 0.006398 | — |
| 104 | sgd_m0_lr0.03_e2_b64_wd0 | sgd | 0 | 0.03 | 2 | 64 | 0 | 40 | 2.504883 | 0.587670 ± 0.003847 | — |
| 105 | sgd_m0_lr0.03_e2_b64_wd0.0001 | sgd | 0 | 0.03 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.587670 ± 0.003847 | — |
| 106 | adam_m0_lr0.0003_e4_b32_wd0 | adam | 0 | 0.0003 | 4 | 32 | 0 | 160 | 2.426758 | 0.587409 ± 0.006022 | — |
| 107 | adam_m0_lr0.0003_e4_b32_wd0.0001 | adam | 0 | 0.0003 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.587409 ± 0.006022 | — |
| 108 | sgd_m0.9_lr3_e1_b64_wd0.0001 | sgd | 0.9 | 3 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.587252 ± 0.079228 | — |
| 109 | sgd_m0.9_lr0.001_e4_b64_wd0 | sgd | 0.9 | 0.001 | 4 | 64 | 0 | 80 | 3.320312 | 0.587147 ± 0.002566 | — |
| 110 | sgd_m0.9_lr0.001_e4_b64_wd0.0001 | sgd | 0.9 | 0.001 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.587147 ± 0.002566 | — |
| 111 | sgd_m0_lr0.03_e1_b32_wd0.0001 | sgd | 0 | 0.03 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.586991 ± 0.007291 | — |
| 112 | sgd_m0_lr0.03_e1_b32_wd0 | sgd | 0 | 0.03 | 1 | 32 | 0 | 40 | 1.523438 | 0.586938 ± 0.007207 | — |
| 113 | sgd_m0.9_lr0.01_e2_b64_wd0 | sgd | 0.9 | 0.01 | 2 | 64 | 0 | 40 | 2.504883 | 0.586311 ± 0.003575 | — |
| 114 | sgd_m0.9_lr0.01_e2_b64_wd0.0001 | sgd | 0.9 | 0.01 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.586311 ± 0.003575 | — |
| 115 | sgd_m0_lr3_e4_b128_wd0.0001 | sgd | 0 | 3 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.586311 ± 0.006185 | — |
| 116 | adam_m0_lr0.03_e2_b227_wd0.0001 | adam | 0 | 0.03 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.586207 ± 0.017646 | — |
| 117 | sgd_m0.9_lr0.01_e4_b128_wd0 | sgd | 0.9 | 0.01 | 4 | 128 | 0 | 40 | 4.560547 | 0.586102 ± 0.003406 | — |
| 118 | sgd_m0.9_lr0.01_e4_b128_wd0.0001 | sgd | 0.9 | 0.01 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.586102 ± 0.003406 | — |
| 119 | sgd_m0.9_lr0.01_e1_b32_wd0 | sgd | 0.9 | 0.01 | 1 | 32 | 0 | 40 | 1.523438 | 0.585632 ± 0.005800 | — |
| 120 | sgd_m0.9_lr0.01_e1_b32_wd0.0001 | sgd | 0.9 | 0.01 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.585580 ± 0.005850 | — |
| 121 | adam_m0_lr0.03_e2_b227_wd0 | adam | 0 | 0.03 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.585475 ± 0.016588 | — |
| 122 | sgd_m0.9_lr0.001_e8_b128_wd0 | sgd | 0.9 | 0.001 | 8 | 128 | 0 | 80 | 6.308594 | 0.585319 ± 0.001997 | — |
| 123 | sgd_m0.9_lr0.001_e8_b128_wd0.0001 | sgd | 0.9 | 0.001 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.585319 ± 0.001997 | — |
| 124 | sgd_m0.9_lr0.01_e8_b227_wd0 | sgd | 0.9 | 0.01 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.585319 ± 0.002519 | — |
| 125 | sgd_m0.9_lr0.01_e8_b227_wd0.0001 | sgd | 0.9 | 0.01 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.585319 ± 0.002519 | — |
| 126 | sgd_m0.9_lr3_e1_b64_wd0 | sgd | 0.9 | 3 | 1 | 64 | 0 | 20 | 1.967773 | 0.585293 ± 0.080178 | — |
| 127 | sgd_m0.9_lr0.1_e1_b128_wd0 | sgd | 0.9 | 0.1 | 1 | 128 | 0 | 10 | 2.529297 | 0.585110 ± 0.007837 | — |
| 128 | sgd_m0.9_lr0.1_e1_b128_wd0.0001 | sgd | 0.9 | 0.1 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.585110 ± 0.007837 | — |
| 129 | sgd_m0_lr3_e8_b227_wd0.0001 | sgd | 0 | 3 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.585110 ± 0.015475 | — |
| 130 | sgd_m0.9_lr0.001_e2_b32_wd0 | sgd | 0.9 | 0.001 | 2 | 32 | 0 | 80 | 1.875000 | 0.584953 ± 0.004927 | — |
| 131 | sgd_m0.9_lr0.001_e2_b32_wd0.0001 | sgd | 0.9 | 0.001 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.584953 ± 0.004927 | — |
| 132 | sgd_m0.9_lr0.03_e2_b227_wd0 | sgd | 0.9 | 0.03 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.584953 ± 0.002590 | — |
| 133 | sgd_m0.9_lr0.03_e2_b227_wd0.0001 | sgd | 0.9 | 0.03 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.584953 ± 0.002590 | — |
| 134 | adam_m0_lr0.001_e2_b128_wd0.0001 | adam | 0 | 0.001 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.584848 ± 0.003377 | — |
| 135 | adam_m0_lr0.001_e2_b128_wd0 | adam | 0 | 0.001 | 2 | 128 | 0 | 20 | 3.349609 | 0.584796 ± 0.003318 | — |
| 136 | sgd_m0.9_lr0.001_e8_b64_wd0 | sgd | 0.9 | 0.001 | 8 | 64 | 0 | 160 | 4.531250 | 0.584692 ± 0.001727 | — |
| 137 | sgd_m0.9_lr0.001_e8_b64_wd0.0001 | sgd | 0.9 | 0.001 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.584692 ± 0.001727 | — |
| 138 | sgd_m0.9_lr0.3_e1_b227_wd0 | sgd | 0.9 | 0.3 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.584639 ± 0.004889 | — |
| 139 | sgd_m0.9_lr0.3_e1_b227_wd0.0001 | sgd | 0.9 | 0.3 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.584639 ± 0.004889 | — |
| 140 | sgd_m0_lr0.3_e1_b227_wd0 | sgd | 0 | 0.3 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.584639 ± 0.004889 | — |
| 141 | sgd_m0_lr0.3_e1_b227_wd0.0001 | sgd | 0 | 0.3 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.584639 ± 0.004889 | — |
| 142 | adam_m0_lr0.1_e2_b128_wd0.0001 | adam | 0 | 0.1 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.584378 ± 0.046830 | — |
| 143 | adam_m0_lr0.1_e2_b128_wd0 | adam | 0 | 0.1 | 2 | 128 | 0 | 20 | 3.349609 | 0.584222 ± 0.046935 | — |
| 144 | sgd_m0_lr3_e2_b128_wd0.0001 | sgd | 0 | 3 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.584117 ± 0.024953 | — |
| 145 | sgd_m0.9_lr0.1_e2_b227_wd0 | sgd | 0.9 | 0.1 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.584065 ± 0.001783 | — |
| 146 | sgd_m0.9_lr0.1_e2_b227_wd0.0001 | sgd | 0.9 | 0.1 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.584065 ± 0.001783 | — |
| 147 | sgd_m0.9_lr0.001_e4_b32_wd0 | sgd | 0.9 | 0.001 | 4 | 32 | 0 | 160 | 2.426758 | 0.583386 ± 0.002609 | — |
| 148 | sgd_m0.9_lr0.001_e4_b32_wd0.0001 | sgd | 0.9 | 0.001 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.583386 ± 0.002609 | — |
| 149 | sgd_m0_lr0.01_e2_b64_wd0 | sgd | 0 | 0.01 | 2 | 64 | 0 | 40 | 2.504883 | 0.582497 ± 0.004091 | — |
| 150 | sgd_m0_lr0.01_e2_b64_wd0.0001 | sgd | 0 | 0.01 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.582497 ± 0.004091 | — |
| 151 | sgd_m0.9_lr1_e1_b227_wd0.0001 | sgd | 0.9 | 1 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.581923 ± 0.034712 | — |
| 152 | sgd_m0_lr1_e1_b227_wd0.0001 | sgd | 0 | 1 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.581923 ± 0.034712 | — |
| 153 | sgd_m0.9_lr1_e1_b227_wd0 | sgd | 0.9 | 1 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.581714 ± 0.034691 | — |
| 154 | sgd_m0_lr1_e1_b227_wd0 | sgd | 0 | 1 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.581714 ± 0.034691 | — |
| 155 | sgd_m0_lr3_e4_b227_wd0 | sgd | 0 | 3 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.581296 ± 0.006505 | — |
| 156 | sgd_m0_lr3_e4_b227_wd0.0001 | sgd | 0 | 3 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.580773 ± 0.007480 | — |
| 157 | sgd_m0_lr0.01_e1_b32_wd0 | sgd | 0 | 0.01 | 1 | 32 | 0 | 40 | 1.523438 | 0.580355 ± 0.007339 | — |
| 158 | sgd_m0_lr0.01_e1_b32_wd0.0001 | sgd | 0 | 0.01 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.580355 ± 0.007339 | — |
| 159 | adam_m0_lr0.003_e1_b32_wd0.0001 | adam | 0 | 0.003 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.579781 ± 0.010985 | — |
| 160 | adam_m0_lr0.003_e1_b32_wd0 | adam | 0 | 0.003 | 1 | 32 | 0 | 40 | 1.523438 | 0.579728 ± 0.010905 | — |
| 161 | sgd_m0_lr0.01_e4_b128_wd0 | sgd | 0 | 0.01 | 4 | 128 | 0 | 40 | 4.560547 | 0.579101 ± 0.002410 | — |
| 162 | sgd_m0_lr0.01_e4_b128_wd0.0001 | sgd | 0 | 0.01 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.579101 ± 0.002410 | — |
| 163 | sgd_m0_lr0.01_e8_b227_wd0 | sgd | 0 | 0.01 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.578213 ± 0.002309 | — |
| 164 | sgd_m0_lr0.01_e8_b227_wd0.0001 | sgd | 0 | 0.01 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.578213 ± 0.002309 | — |
| 165 | adam_m0_lr0.001_e1_b64_wd0 | adam | 0 | 0.001 | 1 | 64 | 0 | 20 | 1.967773 | 0.576489 ± 0.002177 | — |
| 166 | adam_m0_lr0.001_e1_b64_wd0.0001 | adam | 0 | 0.001 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.576489 ± 0.002177 | — |
| 167 | adam_m0_lr0.003_e2_b64_wd0 | adam | 0 | 0.003 | 2 | 64 | 0 | 40 | 2.504883 | 0.575758 ± 0.010459 | — |
| 168 | adam_m0_lr0.003_e2_b64_wd0.0001 | adam | 0 | 0.003 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.575758 ± 0.010493 | — |
| 169 | adam_m0_lr0.003_e4_b128_wd0 | adam | 0 | 0.003 | 4 | 128 | 0 | 40 | 4.560547 | 0.574138 ± 0.009835 | — |
| 170 | adam_m0_lr0.003_e4_b128_wd0.0001 | adam | 0 | 0.003 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.574138 ± 0.009835 | — |
| 171 | adam_m0_lr0.001_e4_b32_wd0.0001 | adam | 0 | 0.001 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.570324 ± 0.009183 | — |
| 172 | adam_m0_lr0.001_e4_b32_wd0 | adam | 0 | 0.001 | 4 | 32 | 0 | 160 | 2.426758 | 0.570219 ± 0.009272 | — |
| 173 | sgd_m0_lr0.1_e1_b64_wd0 | sgd | 0 | 0.1 | 1 | 64 | 0 | 20 | 1.967773 | 0.569801 ± 0.004942 | — |
| 174 | sgd_m0_lr0.1_e1_b64_wd0.0001 | sgd | 0 | 0.1 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.569801 ± 0.004942 | — |
| 175 | adam_m0_lr0.003_e8_b227_wd0.0001 | adam | 0 | 0.003 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.568704 ± 0.007513 | — |
| 176 | adam_m0_lr0.001_e8_b64_wd0 | adam | 0 | 0.001 | 8 | 64 | 0 | 160 | 4.531250 | 0.568652 ± 0.006891 | — |
| 177 | adam_m0_lr0.001_e8_b64_wd0.0001 | adam | 0 | 0.001 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.568652 ± 0.006891 | — |
| 178 | adam_m0_lr0.003_e8_b227_wd0 | adam | 0 | 0.003 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.568548 ± 0.007784 | — |
| 179 | sgd_m0_lr0.1_e2_b128_wd0 | sgd | 0 | 0.1 | 2 | 128 | 0 | 20 | 3.349609 | 0.568130 ± 0.003923 | — |
| 180 | sgd_m0_lr0.1_e2_b128_wd0.0001 | sgd | 0 | 0.1 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.568130 ± 0.003923 | — |
| 181 | sgd_m0_lr0.1_e4_b227_wd0.0001 | sgd | 0 | 0.1 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.567137 ± 0.002680 | — |
| 182 | sgd_m0_lr0.1_e4_b227_wd0 | sgd | 0 | 0.1 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.567085 ± 0.002733 | — |
| 183 | sgd_m0.9_lr0.3_e1_b128_wd0 | sgd | 0.9 | 0.3 | 1 | 128 | 0 | 10 | 2.529297 | 0.564995 ± 0.007166 | — |
| 184 | sgd_m0.9_lr0.3_e1_b128_wd0.0001 | sgd | 0.9 | 0.3 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.564995 ± 0.007166 | — |
| 185 | sgd_m0_lr3_e2_b227_wd0 | sgd | 0 | 3 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.564890 ± 0.090835 | — |
| 186 | sgd_m0_lr0.01_e8_b64_wd0 | sgd | 0 | 0.01 | 8 | 64 | 0 | 160 | 4.531250 | 0.564263 ± 0.002852 | — |
| 187 | sgd_m0_lr0.01_e8_b64_wd0.0001 | sgd | 0 | 0.01 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.564211 ± 0.002805 | — |
| 188 | sgd_m0.9_lr1_e2_b64_wd0 | sgd | 0.9 | 1 | 2 | 64 | 0 | 40 | 2.504883 | 0.564159 ± 0.009690 | — |
| 189 | sgd_m0.9_lr3_e1_b227_wd0 | sgd | 0.9 | 3 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.564159 ± 0.092760 | — |
| 190 | sgd_m0_lr3_e1_b227_wd0 | sgd | 0 | 3 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.564159 ± 0.092760 | — |
| 191 | sgd_m0.9_lr1_e2_b64_wd0.0001 | sgd | 0.9 | 1 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.564002 ± 0.010072 | — |
| 192 | adam_m0_lr0.03_e1_b128_wd0.0001 | adam | 0 | 0.03 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.563636 ± 0.016194 | — |
| 193 | adam_m0_lr0.01_e1_b128_wd0 | adam | 0 | 0.01 | 1 | 128 | 0 | 10 | 2.529297 | 0.563532 ± 0.008196 | — |
| 194 | adam_m0_lr0.03_e1_b128_wd0 | adam | 0 | 0.03 | 1 | 128 | 0 | 10 | 2.529297 | 0.563532 ± 0.016315 | — |
| 195 | adam_m0_lr0.01_e1_b128_wd0.0001 | adam | 0 | 0.01 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.563480 ± 0.008284 | — |
| 196 | sgd_m0_lr0.01_e4_b32_wd0 | sgd | 0 | 0.01 | 4 | 32 | 0 | 160 | 2.426758 | 0.563480 ± 0.003673 | — |
| 197 | sgd_m0_lr0.01_e4_b32_wd0.0001 | sgd | 0 | 0.01 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.563480 ± 0.003673 | — |
| 198 | adam_m0_lr0.0003_e8_b128_wd0 | adam | 0 | 0.0003 | 8 | 128 | 0 | 80 | 6.308594 | 0.562539 ± 0.001636 | — |
| 199 | adam_m0_lr0.0003_e8_b128_wd0.0001 | adam | 0 | 0.0003 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.562539 ± 0.001636 | — |
| 200 | sgd_m0.9_lr0.3_e2_b227_wd0 | sgd | 0.9 | 0.3 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.562278 ± 0.001803 | — |
| 201 | sgd_m0.9_lr0.3_e2_b227_wd0.0001 | sgd | 0.9 | 0.3 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.562278 ± 0.001803 | — |
| 202 | sgd_m0.9_lr3_e1_b227_wd0.0001 | sgd | 0.9 | 3 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.562173 ± 0.092566 | — |
| 203 | sgd_m0_lr3_e1_b227_wd0.0001 | sgd | 0 | 3 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.562173 ± 0.092566 | — |
| 204 | adam_m0_lr0.01_e2_b227_wd0.0001 | adam | 0 | 0.01 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.560972 ± 0.006464 | — |
| 205 | adam_m0_lr0.01_e2_b227_wd0 | adam | 0 | 0.01 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.560711 ± 0.006039 | — |
| 206 | adam_m0_lr0.0003_e4_b64_wd0 | adam | 0 | 0.0003 | 4 | 64 | 0 | 80 | 3.320312 | 0.559875 ± 0.001357 | — |
| 207 | adam_m0_lr0.0003_e4_b64_wd0.0001 | adam | 0 | 0.0003 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.559875 ± 0.001357 | — |
| 208 | sgd_m0_lr3_e1_b32_wd0 | sgd | 0 | 3 | 1 | 32 | 0 | 40 | 1.523438 | 0.559143 ± 0.053288 | — |
| 209 | sgd_m0_lr3_e1_b32_wd0.0001 | sgd | 0 | 3 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.558777 ± 0.050513 | — |
| 210 | sgd_m0.9_lr1_e1_b32_wd0 | sgd | 0.9 | 1 | 1 | 32 | 0 | 40 | 1.523438 | 0.557027 ± 0.039766 | — |
| 211 | adam_m0_lr0.1_e1_b32_wd0 | adam | 0 | 0.1 | 1 | 32 | 0 | 40 | 1.523438 | 0.557001 ± 0.006724 | — |
| 212 | adam_m0_lr0.1_e1_b32_wd0.0001 | adam | 0 | 0.1 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.556949 ± 0.007139 | — |
| 213 | sgd_m0.9_lr1_e1_b32_wd0.0001 | sgd | 0.9 | 1 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.556818 ± 0.039814 | — |
| 214 | sgd_m0_lr0.001_e8_b32_wd0 | sgd | 0 | 0.001 | 8 | 32 | 0 | 320 | 3.261719 | 0.556479 ± 0.001067 | — |
| 215 | sgd_m0_lr0.001_e8_b32_wd0.0001 | sgd | 0 | 0.001 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.556479 ± 0.001067 | — |
| 216 | sgd_m0_lr0.03_e1_b128_wd0 | sgd | 0 | 0.03 | 1 | 128 | 0 | 10 | 2.529297 | 0.556426 ± 0.001636 | — |
| 217 | sgd_m0_lr0.03_e1_b128_wd0.0001 | sgd | 0 | 0.03 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.556426 ± 0.001636 | — |
| 218 | sgd_m0.9_lr1_e4_b128_wd0 | sgd | 0.9 | 1 | 4 | 128 | 0 | 40 | 4.560547 | 0.556374 ± 0.002908 | — |
| 219 | sgd_m0.9_lr1_e4_b128_wd0.0001 | sgd | 0.9 | 1 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.556270 ± 0.003147 | — |
| 220 | sgd_m0.9_lr0.1_e1_b64_wd0 | sgd | 0.9 | 0.1 | 1 | 64 | 0 | 20 | 1.967773 | 0.555016 ± 0.009630 | — |
| 221 | sgd_m0.9_lr0.1_e1_b64_wd0.0001 | sgd | 0.9 | 0.1 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.554963 ± 0.009720 | — |
| 222 | adam_m0_lr0.1_e4_b128_wd0.0001 | adam | 0 | 0.1 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.554441 ± 0.010753 | — |
| 223 | sgd_m0_lr0.3_e1_b128_wd0 | sgd | 0 | 0.3 | 1 | 128 | 0 | 10 | 2.529297 | 0.554389 ± 0.004120 | — |
| 224 | sgd_m0_lr0.3_e1_b128_wd0.0001 | sgd | 0 | 0.3 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.554389 ± 0.004120 | — |
| 225 | adam_m0_lr0.1_e4_b128_wd0 | adam | 0 | 0.1 | 4 | 128 | 0 | 40 | 4.560547 | 0.553866 ± 0.010664 | — |
| 226 | sgd_m0_lr3_e2_b227_wd0.0001 | sgd | 0 | 3 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.553396 ± 0.100359 | — |
| 227 | sgd_m0_lr0.03_e2_b227_wd0 | sgd | 0 | 0.03 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.552978 ± 0.000953 | — |
| 228 | sgd_m0_lr0.03_e2_b227_wd0.0001 | sgd | 0 | 0.03 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.552978 ± 0.000953 | — |
| 229 | sgd_m0.9_lr1_e8_b227_wd0 | sgd | 0.9 | 1 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.552821 ± 0.006149 | — |
| 230 | sgd_m0.9_lr1_e8_b227_wd0.0001 | sgd | 0.9 | 1 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.552456 ± 0.006258 | — |
| 231 | sgd_m0_lr0.3_e2_b227_wd0 | sgd | 0 | 0.3 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.552142 ± 0.004299 | — |
| 232 | sgd_m0_lr0.3_e2_b227_wd0.0001 | sgd | 0 | 0.3 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.552142 ± 0.004299 | — |
| 233 | adam_m0_lr0.1_e2_b64_wd0 | adam | 0 | 0.1 | 2 | 64 | 0 | 40 | 2.504883 | 0.552090 ± 0.016120 | — |
| 234 | adam_m0_lr0.0003_e2_b32_wd0 | adam | 0 | 0.0003 | 2 | 32 | 0 | 80 | 1.875000 | 0.551985 ± 0.004028 | — |
| 235 | adam_m0_lr0.0003_e2_b32_wd0.0001 | adam | 0 | 0.0003 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.551933 ± 0.003945 | — |
| 236 | adam_m0_lr0.1_e2_b64_wd0.0001 | adam | 0 | 0.1 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.551933 ± 0.016115 | — |
| 237 | sgd_m0.9_lr0.1_e2_b128_wd0 | sgd | 0.9 | 0.1 | 2 | 128 | 0 | 20 | 3.349609 | 0.550940 ± 0.011367 | — |
| 238 | sgd_m0.9_lr0.1_e2_b128_wd0.0001 | sgd | 0.9 | 0.1 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.550940 ± 0.011367 | — |
| 239 | sgd_m0.9_lr0.3_e1_b64_wd0.0001 | sgd | 0.9 | 0.3 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.550470 ± 0.031850 | — |
| 240 | sgd_m0.9_lr0.3_e1_b64_wd0 | sgd | 0.9 | 0.3 | 1 | 64 | 0 | 20 | 1.967773 | 0.550313 ± 0.031721 | — |
| 241 | adam_m0_lr0.01_e1_b64_wd0 | adam | 0 | 0.01 | 1 | 64 | 0 | 20 | 1.967773 | 0.550104 ± 0.011739 | — |
| 242 | adam_m0_lr0.01_e1_b64_wd0.0001 | adam | 0 | 0.01 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.550052 ± 0.011460 | — |
| 243 | adam_m0_lr0.001_e2_b227_wd0 | adam | 0 | 0.001 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.549791 ± 0.002770 | — |
| 244 | adam_m0_lr0.001_e2_b227_wd0.0001 | adam | 0 | 0.001 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.549791 ± 0.002770 | — |
| 245 | sgd_m0_lr0.03_e4_b64_wd0 | sgd | 0 | 0.03 | 4 | 64 | 0 | 80 | 3.320312 | 0.549007 ± 0.005185 | — |
| 246 | sgd_m0_lr0.03_e4_b64_wd0.0001 | sgd | 0 | 0.03 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.549007 ± 0.005185 | — |
| 247 | sgd_m0.9_lr0.1_e4_b227_wd0 | sgd | 0.9 | 0.1 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.548903 ± 0.009484 | — |
| 248 | sgd_m0.9_lr0.1_e4_b227_wd0.0001 | sgd | 0.9 | 0.1 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.548903 ± 0.009484 | — |
| 249 | sgd_m0_lr0.03_e8_b128_wd0.0001 | sgd | 0 | 0.03 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.548380 ± 0.005270 | — |
| 250 | sgd_m0_lr0.03_e8_b128_wd0 | sgd | 0 | 0.03 | 8 | 128 | 0 | 80 | 6.308594 | 0.548276 ± 0.005322 | — |
| 251 | sgd_m0.9_lr1_e2_b227_wd0 | sgd | 0.9 | 1 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.548171 ± 0.033646 | — |
| 252 | sgd_m0.9_lr1_e2_b227_wd0.0001 | sgd | 0.9 | 1 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.547962 ± 0.033709 | — |
| 253 | adam_m0_lr0.001_e8_b32_wd0.0001 | adam | 0 | 0.001 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.547753 ± 0.004832 | — |
| 254 | adam_m0_lr0.001_e8_b32_wd0 | adam | 0 | 0.001 | 8 | 32 | 0 | 320 | 3.261719 | 0.547701 ± 0.004770 | — |
| 255 | sgd_m0_lr0.03_e2_b32_wd0 | sgd | 0 | 0.03 | 2 | 32 | 0 | 80 | 1.875000 | 0.547544 ± 0.006130 | — |
| 256 | sgd_m0_lr0.03_e2_b32_wd0.0001 | sgd | 0 | 0.03 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.547544 ± 0.006130 | — |
| 257 | adam_m0_lr0.1_e8_b227_wd0.0001 | adam | 0 | 0.1 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.546813 ± 0.005511 | — |
| 258 | adam_m0_lr0.1_e8_b227_wd0 | adam | 0 | 0.1 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.546656 ± 0.005435 | — |
| 259 | sgd_m0.9_lr0.03_e4_b128_wd0 | sgd | 0.9 | 0.03 | 4 | 128 | 0 | 40 | 4.560547 | 0.546029 ± 0.006112 | — |
| 260 | sgd_m0.9_lr0.03_e4_b128_wd0.0001 | sgd | 0.9 | 0.03 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.546029 ± 0.006112 | — |
| 261 | sgd_m0.9_lr0.03_e2_b64_wd0.0001 | sgd | 0.9 | 0.03 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.544984 ± 0.007074 | — |
| 262 | sgd_m0.9_lr0.03_e2_b64_wd0 | sgd | 0.9 | 0.03 | 2 | 64 | 0 | 40 | 2.504883 | 0.544932 ± 0.007164 | — |
| 263 | adam_m0_lr0.003_e2_b32_wd0 | adam | 0 | 0.003 | 2 | 32 | 0 | 80 | 1.875000 | 0.544566 ± 0.008488 | — |
| 264 | adam_m0_lr0.003_e2_b32_wd0.0001 | adam | 0 | 0.003 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.544514 ± 0.008481 | — |
| 265 | sgd_m0.9_lr0.001_e8_b32_wd0 | sgd | 0.9 | 0.001 | 8 | 32 | 0 | 320 | 3.261719 | 0.543887 ± 0.001659 | — |
| 266 | sgd_m0.9_lr0.001_e8_b32_wd0.0001 | sgd | 0.9 | 0.001 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.543887 ± 0.001659 | — |
| 267 | sgd_m0.9_lr0.03_e8_b227_wd0 | sgd | 0.9 | 0.03 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.543469 ± 0.006463 | — |
| 268 | sgd_m0.9_lr0.03_e8_b227_wd0.0001 | sgd | 0.9 | 0.03 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.543469 ± 0.006463 | — |
| 269 | sgd_m0.9_lr0.03_e1_b32_wd0.0001 | sgd | 0.9 | 0.03 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.542894 ± 0.012011 | — |
| 270 | sgd_m0.9_lr0.03_e1_b32_wd0 | sgd | 0.9 | 0.03 | 1 | 32 | 0 | 40 | 1.523438 | 0.542842 ± 0.012000 | — |
| 271 | adam_m0_lr0.001_e1_b128_wd0 | adam | 0 | 0.001 | 1 | 128 | 0 | 10 | 2.529297 | 0.541641 ± 0.003766 | — |
| 272 | adam_m0_lr0.001_e1_b128_wd0.0001 | adam | 0 | 0.001 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.541641 ± 0.003766 | — |
| 273 | adam_m0_lr0.01_e2_b128_wd0 | adam | 0 | 0.01 | 2 | 128 | 0 | 20 | 3.349609 | 0.540596 ± 0.013304 | — |
| 274 | adam_m0_lr0.01_e2_b128_wd0.0001 | adam | 0 | 0.01 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.540596 ± 0.013304 | — |
| 275 | adam_m0_lr0.003_e4_b64_wd0 | adam | 0 | 0.003 | 4 | 64 | 0 | 80 | 3.320312 | 0.540491 ± 0.007894 | — |
| 276 | adam_m0_lr0.003_e4_b64_wd0.0001 | adam | 0 | 0.003 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.540491 ± 0.007894 | — |
| 277 | adam_m0_lr0.0001_e8_b32_wd0 | adam | 0 | 0.0001 | 8 | 32 | 0 | 320 | 3.261719 | 0.538819 ± 0.000863 | — |
| 278 | adam_m0_lr0.0001_e8_b32_wd0.0001 | adam | 0 | 0.0001 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.538819 ± 0.000863 | — |
| 279 | sgd_m0.9_lr0.3_e2_b128_wd0.0001 | sgd | 0.9 | 0.3 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.538819 ± 0.033684 | — |
| 280 | sgd_m0.9_lr0.3_e2_b128_wd0 | sgd | 0.9 | 0.3 | 2 | 128 | 0 | 20 | 3.349609 | 0.538767 ± 0.033596 | — |
| 281 | sgd_m0.9_lr3_e1_b32_wd0 | sgd | 0.9 | 3 | 1 | 32 | 0 | 40 | 1.523438 | 0.538114 ± 0.066015 | — |
| 282 | adam_m0_lr0.003_e8_b128_wd0 | adam | 0 | 0.003 | 8 | 128 | 0 | 80 | 6.308594 | 0.537722 ± 0.006808 | — |
| 283 | adam_m0_lr0.003_e8_b128_wd0.0001 | adam | 0 | 0.003 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.537722 ± 0.006808 | — |
| 284 | sgd_m0.9_lr1_e1_b64_wd0.0001 | sgd | 0.9 | 1 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.536729 ± 0.008779 | — |
| 285 | sgd_m0.9_lr1_e1_b64_wd0 | sgd | 0.9 | 1 | 1 | 64 | 0 | 20 | 1.967773 | 0.536625 ± 0.008610 | — |
| 286 | adam_m0_lr0.1_e4_b64_wd0 | adam | 0 | 0.1 | 4 | 64 | 0 | 80 | 3.320312 | 0.536573 ± 0.010517 | — |
| 287 | adam_m0_lr0.1_e4_b64_wd0.0001 | adam | 0 | 0.1 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.536520 ± 0.010672 | — |
| 288 | adam_m0_lr0.1_e2_b32_wd0 | adam | 0 | 0.1 | 2 | 32 | 0 | 80 | 1.875000 | 0.536155 ± 0.006956 | — |
| 289 | adam_m0_lr0.1_e2_b32_wd0.0001 | adam | 0 | 0.1 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.535998 ± 0.006391 | — |
| 290 | sgd_m0_lr0.1_e2_b64_wd0 | sgd | 0 | 0.1 | 2 | 64 | 0 | 40 | 2.504883 | 0.535946 ± 0.006813 | — |
| 291 | sgd_m0_lr0.1_e2_b64_wd0.0001 | sgd | 0 | 0.1 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.535841 ± 0.006755 | — |
| 292 | sgd_m0_lr0.1_e4_b128_wd0 | sgd | 0 | 0.1 | 4 | 128 | 0 | 40 | 4.560547 | 0.535632 ± 0.005678 | — |
| 293 | sgd_m0_lr0.1_e4_b128_wd0.0001 | sgd | 0 | 0.1 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.535632 ± 0.005678 | — |
| 294 | sgd_m0_lr0.1_e1_b32_wd0 | sgd | 0 | 0.1 | 1 | 32 | 0 | 40 | 1.523438 | 0.535371 ± 0.009876 | — |
| 295 | sgd_m0_lr0.1_e1_b32_wd0.0001 | sgd | 0 | 0.1 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.535371 ± 0.009876 | — |
| 296 | sgd_m0_lr0.1_e8_b227_wd0 | sgd | 0 | 0.1 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.534587 ± 0.005990 | — |
| 297 | sgd_m0_lr0.1_e8_b227_wd0.0001 | sgd | 0 | 0.1 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.534587 ± 0.005990 | — |
| 298 | adam_m0_lr0.01_e4_b227_wd0.0001 | adam | 0 | 0.01 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.533699 ± 0.012737 | — |
| 299 | adam_m0_lr0.0003_e8_b227_wd0 | adam | 0 | 0.0003 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.533647 ± 0.001783 | — |
| 300 | adam_m0_lr0.0003_e8_b227_wd0.0001 | adam | 0 | 0.0003 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.533647 ± 0.001783 | — |
| 301 | adam_m0_lr0.01_e4_b227_wd0 | adam | 0 | 0.01 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.533647 ± 0.012767 | — |
| 302 | adam_m0_lr0.1_e8_b128_wd0.0001 | adam | 0 | 0.1 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.533386 ± 0.007788 | — |
| 303 | adam_m0_lr0.1_e8_b128_wd0 | adam | 0 | 0.1 | 8 | 128 | 0 | 80 | 6.308594 | 0.533333 ± 0.008175 | — |
| 304 | adam_m0_lr0.0003_e4_b128_wd0 | adam | 0 | 0.0003 | 4 | 128 | 0 | 40 | 4.560547 | 0.531870 ± 0.002196 | — |
| 305 | adam_m0_lr0.0003_e4_b128_wd0.0001 | adam | 0 | 0.0003 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.531870 ± 0.002196 | — |
| 306 | sgd_m0_lr0.01_e8_b32_wd0 | sgd | 0 | 0.01 | 8 | 32 | 0 | 320 | 3.261719 | 0.530355 ± 0.003717 | — |
| 307 | sgd_m0_lr0.01_e8_b32_wd0.0001 | sgd | 0 | 0.01 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.530355 ± 0.003717 | — |
| 308 | sgd_m0.9_lr0.01_e4_b64_wd0 | sgd | 0.9 | 0.01 | 4 | 64 | 0 | 80 | 3.320312 | 0.529781 ± 0.004209 | — |
| 309 | sgd_m0.9_lr0.01_e4_b64_wd0.0001 | sgd | 0.9 | 0.01 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.529781 ± 0.004209 | — |
| 310 | sgd_m0.9_lr0.01_e8_b128_wd0.0001 | sgd | 0.9 | 0.01 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.528579 ± 0.005522 | — |
| 311 | sgd_m0.9_lr0.01_e8_b128_wd0 | sgd | 0.9 | 0.01 | 8 | 128 | 0 | 80 | 6.308594 | 0.528527 ± 0.005610 | — |
| 312 | sgd_m0_lr1_e1_b64_wd0 | sgd | 0 | 1 | 1 | 64 | 0 | 20 | 1.967773 | 0.528474 ± 0.024934 | — |
| 313 | sgd_m0_lr1_e1_b64_wd0.0001 | sgd | 0 | 1 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.528474 ± 0.024934 | — |
| 314 | sgd_m0.9_lr0.01_e2_b32_wd0 | sgd | 0.9 | 0.01 | 2 | 32 | 0 | 80 | 1.875000 | 0.527900 ± 0.005439 | — |
| 315 | sgd_m0.9_lr0.01_e2_b32_wd0.0001 | sgd | 0.9 | 0.01 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.527900 ± 0.005439 | — |
| 316 | sgd_m0_lr0.01_e1_b64_wd0 | sgd | 0 | 0.01 | 1 | 64 | 0 | 20 | 1.967773 | 0.527586 ± 0.001659 | — |
| 317 | sgd_m0_lr0.01_e1_b64_wd0.0001 | sgd | 0 | 0.01 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.527586 ± 0.001659 | — |
| 318 | sgd_m0.9_lr1_e2_b128_wd0 | sgd | 0.9 | 1 | 2 | 128 | 0 | 20 | 3.349609 | 0.527534 ± 0.006505 | — |
| 319 | sgd_m0.9_lr1_e2_b128_wd0.0001 | sgd | 0.9 | 1 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.527482 ± 0.006509 | — |
| 320 | adam_m0_lr0.0003_e2_b64_wd0 | adam | 0 | 0.0003 | 2 | 64 | 0 | 40 | 2.504883 | 0.527116 ± 0.002261 | — |
| 321 | adam_m0_lr0.0003_e2_b64_wd0.0001 | adam | 0 | 0.0003 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.527116 ± 0.002261 | — |
| 322 | sgd_m0_lr0.01_e2_b128_wd0 | sgd | 0 | 0.01 | 2 | 128 | 0 | 20 | 3.349609 | 0.525340 ± 0.001481 | — |
| 323 | sgd_m0_lr0.01_e2_b128_wd0.0001 | sgd | 0 | 0.01 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.525340 ± 0.001481 | — |
| 324 | sgd_m0_lr0.3_e1_b64_wd0 | sgd | 0 | 0.3 | 1 | 64 | 0 | 20 | 1.967773 | 0.525287 ± 0.012672 | — |
| 325 | sgd_m0_lr0.3_e1_b64_wd0.0001 | sgd | 0 | 0.3 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.525287 ± 0.012672 | — |
| 326 | sgd_m0_lr0.01_e4_b227_wd0 | sgd | 0 | 0.01 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.524869 ± 0.001044 | — |
| 327 | sgd_m0_lr0.01_e4_b227_wd0.0001 | sgd | 0 | 0.01 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.524869 ± 0.001044 | — |
| 328 | sgd_m0.9_lr0.3_e4_b227_wd0 | sgd | 0.9 | 0.3 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.524713 ± 0.020786 | — |
| 329 | sgd_m0.9_lr0.3_e4_b227_wd0.0001 | sgd | 0.9 | 0.3 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.524660 ± 0.020823 | — |
| 330 | sgd_m0.9_lr0.3_e2_b32_wd0 | sgd | 0.9 | 0.3 | 2 | 32 | 0 | 80 | 1.875000 | 0.524608 ± 0.001437 | — |
| 331 | sgd_m0.9_lr0.3_e2_b32_wd0.0001 | sgd | 0.9 | 0.3 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.524190 ± 0.001646 | — |
| 332 | sgd_m0_lr0.3_e2_b128_wd0.0001 | sgd | 0 | 0.3 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.523250 ± 0.013157 | — |
| 333 | sgd_m0_lr0.3_e2_b128_wd0 | sgd | 0 | 0.3 | 2 | 128 | 0 | 20 | 3.349609 | 0.523197 ± 0.013068 | — |
| 334 | adam_m0_lr0.01_e1_b32_wd0.0001 | adam | 0 | 0.01 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.521839 ± 0.015583 | — |
| 335 | adam_m0_lr0.01_e1_b32_wd0 | adam | 0 | 0.01 | 1 | 32 | 0 | 40 | 1.523438 | 0.521735 ± 0.015495 | — |
| 336 | sgd_m0_lr1_e1_b128_wd0.0001 | sgd | 0 | 1 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.521055 ± 0.023456 | — |
| 337 | sgd_m0_lr1_e1_b128_wd0 | sgd | 0 | 1 | 1 | 128 | 0 | 10 | 2.529297 | 0.521003 ± 0.023308 | — |
| 338 | sgd_m0_lr1_e1_b32_wd0.0001 | sgd | 0 | 1 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.520533 ± 0.014031 | — |
| 339 | sgd_m0_lr1_e1_b32_wd0 | sgd | 0 | 1 | 1 | 32 | 0 | 40 | 1.523438 | 0.520428 ± 0.014135 | — |
| 340 | sgd_m0_lr0.3_e4_b227_wd0.0001 | sgd | 0 | 0.3 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.520115 ± 0.007553 | — |
| 341 | sgd_m0_lr0.3_e4_b227_wd0 | sgd | 0 | 0.3 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.520063 ± 0.007613 | — |
| 342 | sgd_m0.9_lr0.3_e1_b32_wd0 | sgd | 0.9 | 0.3 | 1 | 32 | 0 | 40 | 1.523438 | 0.519383 ± 0.008519 | — |
| 343 | sgd_m0.9_lr0.3_e1_b32_wd0.0001 | sgd | 0.9 | 0.3 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.519279 ± 0.008569 | — |
| 344 | sgd_m0_lr1_e2_b227_wd0.0001 | sgd | 0 | 1 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.519122 ± 0.010726 | — |
| 345 | sgd_m0_lr1_e2_b227_wd0 | sgd | 0 | 1 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.518966 ± 0.010588 | — |
| 346 | adam_m0_lr0.01_e4_b128_wd0 | adam | 0 | 0.01 | 4 | 128 | 0 | 40 | 4.560547 | 0.518757 ± 0.007504 | — |
| 347 | adam_m0_lr0.01_e4_b128_wd0.0001 | adam | 0 | 0.01 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.518704 ± 0.007592 | — |
| 348 | sgd_m0_lr1_e2_b64_wd0 | sgd | 0 | 1 | 2 | 64 | 0 | 40 | 2.504883 | 0.518234 ± 0.010146 | — |
| 349 | sgd_m0_lr1_e2_b64_wd0.0001 | sgd | 0 | 1 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.518130 ± 0.010195 | — |
| 350 | adam_m0_lr0.01_e2_b64_wd0 | adam | 0 | 0.01 | 2 | 64 | 0 | 40 | 2.504883 | 0.517868 ± 0.013290 | — |
| 351 | adam_m0_lr0.01_e2_b64_wd0.0001 | adam | 0 | 0.01 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.517868 ± 0.013290 | — |
| 352 | sgd_m0.9_lr10_e2_b227_wd0 | sgd | 0.9 | 10 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.517764 ± 0.024178 | — |
| 353 | adam_m0_lr0.0003_e1_b32_wd0 | adam | 0 | 0.0003 | 1 | 32 | 0 | 40 | 1.523438 | 0.517555 ± 0.003881 | — |
| 354 | adam_m0_lr0.0003_e1_b32_wd0.0001 | adam | 0 | 0.0003 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.517555 ± 0.003881 | — |
| 355 | sgd_m0.9_lr0.3_e2_b64_wd0 | sgd | 0.9 | 0.3 | 2 | 64 | 0 | 40 | 2.504883 | 0.517503 ± 0.007706 | — |
| 356 | sgd_m0.9_lr0.3_e2_b64_wd0.0001 | sgd | 0.9 | 0.3 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.517450 ± 0.007937 | — |
| 357 | sgd_m0_lr1_e2_b128_wd0 | sgd | 0 | 1 | 2 | 128 | 0 | 20 | 3.349609 | 0.516980 ± 0.018736 | — |
| 358 | sgd_m0_lr1_e2_b128_wd0.0001 | sgd | 0 | 1 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.516928 ± 0.018802 | — |
| 359 | sgd_m0.9_lr0.3_e4_b64_wd0 | sgd | 0.9 | 0.3 | 4 | 64 | 0 | 80 | 3.320312 | 0.516823 ± 0.001960 | — |
| 360 | adam_m0_lr0.1_e4_b32_wd0.0001 | adam | 0 | 0.1 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.516510 ± 0.005435 | — |
| 361 | sgd_m0.9_lr0.3_e4_b64_wd0.0001 | sgd | 0.9 | 0.3 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.516458 ± 0.002073 | — |
| 362 | adam_m0_lr0.03_e1_b64_wd0 | adam | 0 | 0.03 | 1 | 64 | 0 | 20 | 1.967773 | 0.516144 ± 0.011042 | — |
| 363 | adam_m0_lr0.0001_e8_b64_wd0 | adam | 0 | 0.0001 | 8 | 64 | 0 | 160 | 4.531250 | 0.516040 ± 0.000593 | — |
| 364 | adam_m0_lr0.0001_e8_b64_wd0.0001 | adam | 0 | 0.0001 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.516040 ± 0.000593 | — |
| 365 | sgd_m0_lr1_e4_b227_wd0 | sgd | 0 | 1 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.515778 ± 0.013540 | — |
| 366 | sgd_m0_lr1_e4_b227_wd0.0001 | sgd | 0 | 1 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.515778 ± 0.013540 | — |
| 367 | adam_m0_lr0.03_e1_b64_wd0.0001 | adam | 0 | 0.03 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.515517 ± 0.010606 | — |
| 368 | adam_m0_lr0.001_e1_b227_wd0 | adam | 0 | 0.001 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.514943 ± 0.000741 | — |
| 369 | adam_m0_lr0.001_e1_b227_wd0.0001 | adam | 0 | 0.001 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.514890 ± 0.000718 | — |
| 370 | adam_m0_lr0.003_e8_b64_wd0 | adam | 0 | 0.003 | 8 | 64 | 0 | 160 | 4.531250 | 0.514786 ± 0.007743 | — |
| 371 | adam_m0_lr0.003_e8_b64_wd0.0001 | adam | 0 | 0.003 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.514786 ± 0.007743 | — |
| 372 | sgd_m0.9_lr0.3_e4_b128_wd0 | sgd | 0.9 | 0.3 | 4 | 128 | 0 | 40 | 4.560547 | 0.514734 ± 0.008948 | — |
| 373 | sgd_m0.9_lr0.3_e4_b128_wd0.0001 | sgd | 0.9 | 0.3 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.514629 ± 0.009006 | — |
| 374 | sgd_m0.9_lr0.3_e8_b128_wd0 | sgd | 0.9 | 0.3 | 8 | 128 | 0 | 80 | 6.308594 | 0.514394 ± 0.003554 | — |
| 375 | sgd_m0.9_lr0.3_e8_b128_wd0.0001 | sgd | 0.9 | 0.3 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.514394 ± 0.003498 | — |
| 376 | adam_m0_lr0.003_e4_b32_wd0 | adam | 0 | 0.003 | 4 | 32 | 0 | 160 | 2.426758 | 0.513950 ± 0.009170 | — |
| 377 | adam_m0_lr0.003_e4_b32_wd0.0001 | adam | 0 | 0.003 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.513950 ± 0.009170 | — |
| 378 | adam_m0_lr0.01_e8_b227_wd0.0001 | adam | 0 | 0.01 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.513427 ± 0.007894 | — |
| 379 | adam_m0_lr0.01_e8_b227_wd0 | adam | 0 | 0.01 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.513218 ± 0.008128 | — |
| 380 | sgd_m0.9_lr1_e4_b227_wd0 | sgd | 0.9 | 1 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.513062 ± 0.005870 | — |
| 381 | sgd_m0.9_lr1_e4_b227_wd0.0001 | sgd | 0.9 | 1 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.513062 ± 0.005928 | — |
| 382 | adam_m0_lr0.0001_e4_b32_wd0 | adam | 0 | 0.0001 | 4 | 32 | 0 | 160 | 2.426758 | 0.512591 ± 0.001783 | — |
| 383 | adam_m0_lr0.0001_e4_b32_wd0.0001 | adam | 0 | 0.0001 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.512591 ± 0.001783 | — |
| 384 | adam_m0_lr0.03_e2_b128_wd0 | adam | 0 | 0.03 | 2 | 128 | 0 | 20 | 3.349609 | 0.512539 ± 0.004215 | — |
| 385 | sgd_m0.9_lr10_e1_b227_wd0.0001 | sgd | 0.9 | 10 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.512382 ± 0.011773 | — |
| 386 | sgd_m0_lr10_e1_b227_wd0.0001 | sgd | 0 | 10 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.512382 ± 0.011773 | — |
| 387 | adam_m0_lr0.03_e2_b128_wd0.0001 | adam | 0 | 0.03 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.512278 ± 0.004799 | — |
| 388 | adam_m0_lr0.03_e2_b64_wd0 | adam | 0 | 0.03 | 2 | 64 | 0 | 40 | 2.504883 | 0.512226 ± 0.013306 | — |
| 389 | adam_m0_lr0.03_e2_b64_wd0.0001 | adam | 0 | 0.03 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.512226 ± 0.013304 | — |
| 390 | sgd_m0_lr1_e8_b64_wd0 | sgd | 0 | 1 | 8 | 64 | 0 | 160 | 4.531250 | 0.512121 ± 0.012565 | — |
| 391 | sgd_m0.9_lr3_e4_b128_wd0.0001 | sgd | 0.9 | 3 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.512069 ± 0.020095 | — |
| 392 | sgd_m0_lr1_e8_b64_wd0.0001 | sgd | 0 | 1 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.511964 ± 0.012915 | — |
| 393 | adam_m0_lr0.03_e1_b32_wd0 | adam | 0 | 0.03 | 1 | 32 | 0 | 40 | 1.523438 | 0.511494 ± 0.016544 | — |
| 394 | sgd_m0.9_lr0.1_e4_b32_wd0 | sgd | 0.9 | 0.1 | 4 | 32 | 0 | 160 | 2.426758 | 0.511494 ± 0.012851 | — |
| 395 | adam_m0_lr0.03_e1_b32_wd0.0001 | adam | 0 | 0.03 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.511390 ± 0.016740 | — |
| 396 | sgd_m0.9_lr0.1_e4_b32_wd0.0001 | sgd | 0.9 | 0.1 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.511390 ± 0.013077 | — |
| 397 | sgd_m0.9_lr0.1_e8_b64_wd0 | sgd | 0.9 | 0.1 | 8 | 64 | 0 | 160 | 4.531250 | 0.511233 ± 0.013375 | — |
| 398 | sgd_m0.9_lr0.1_e8_b64_wd0.0001 | sgd | 0.9 | 0.1 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.511076 ± 0.013375 | — |
| 399 | adam_m0_lr0.03_e4_b128_wd0.0001 | adam | 0 | 0.03 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.510658 ± 0.010727 | — |
| 400 | sgd_m0.9_lr0.3_e8_b227_wd0 | sgd | 0.9 | 0.3 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.510606 ± 0.005777 | — |
| 401 | sgd_m0.9_lr0.3_e8_b227_wd0.0001 | sgd | 0.9 | 0.3 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.510606 ± 0.005661 | — |
| 402 | adam_m0_lr0.03_e4_b128_wd0 | adam | 0 | 0.03 | 4 | 128 | 0 | 40 | 4.560547 | 0.510293 ± 0.010771 | — |
| 403 | sgd_m0.9_lr10_e1_b32_wd0 | sgd | 0.9 | 10 | 1 | 32 | 0 | 40 | 1.523438 | 0.510214 ± 0.017692 | — |
| 404 | sgd_m0_lr0.03_e8_b64_wd0 | sgd | 0 | 0.03 | 8 | 64 | 0 | 160 | 4.531250 | 0.510084 ± 0.005617 | — |
| 405 | sgd_m0_lr0.03_e8_b64_wd0.0001 | sgd | 0 | 0.03 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.510084 ± 0.005617 | — |
| 406 | sgd_m0_lr1_e4_b128_wd0 | sgd | 0 | 1 | 4 | 128 | 0 | 40 | 4.560547 | 0.510084 ± 0.008193 | — |
| 407 | sgd_m0_lr1_e4_b128_wd0.0001 | sgd | 0 | 1 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.509979 ± 0.008023 | — |
| 408 | sgd_m0_lr0.001_e8_b64_wd0 | sgd | 0 | 0.001 | 8 | 64 | 0 | 160 | 4.531250 | 0.509770 ± 0.000789 | — |
| 409 | sgd_m0_lr0.001_e8_b64_wd0.0001 | sgd | 0 | 0.001 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.509770 ± 0.000789 | — |
| 410 | sgd_m0.9_lr10_e1_b227_wd0 | sgd | 0.9 | 10 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.509378 ± 0.012396 | — |
| 411 | sgd_m0_lr10_e1_b227_wd0 | sgd | 0 | 10 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.509378 ± 0.012396 | — |
| 412 | sgd_m0_lr1_e8_b227_wd0 | sgd | 0 | 1 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.509248 ± 0.004642 | — |
| 413 | sgd_m0_lr1_e8_b227_wd0.0001 | sgd | 0 | 1 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.509195 ± 0.004629 | — |
| 414 | sgd_m0_lr0.001_e4_b32_wd0 | sgd | 0 | 0.001 | 4 | 32 | 0 | 160 | 2.426758 | 0.509143 ± 0.000181 | — |
| 415 | sgd_m0_lr0.001_e4_b32_wd0.0001 | sgd | 0 | 0.001 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.509143 ± 0.000181 | — |
| 416 | adam_m0_lr0.1_e4_b32_wd0 | adam | 0 | 0.1 | 4 | 32 | 0 | 160 | 2.426758 | 0.509039 ± 0.017419 | — |
| 417 | sgd_m0_lr0.03_e4_b32_wd0 | sgd | 0 | 0.03 | 4 | 32 | 0 | 160 | 2.426758 | 0.508673 ± 0.006175 | — |
| 418 | sgd_m0_lr0.03_e4_b32_wd0.0001 | sgd | 0 | 0.03 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.508673 ± 0.006175 | — |
| 419 | sgd_m0_lr1_e4_b32_wd0 | sgd | 0 | 1 | 4 | 32 | 0 | 160 | 2.426758 | 0.508673 ± 0.012086 | — |
| 420 | sgd_m0.9_lr10_e2_b128_wd0.0001 | sgd | 0.9 | 10 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.508621 ± 0.014931 | — |
| 421 | sgd_m0_lr1_e4_b32_wd0.0001 | sgd | 0 | 1 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.508516 ± 0.012029 | — |
| 422 | adam_m0_lr0.03_e4_b227_wd0 | adam | 0 | 0.03 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.508307 ± 0.009406 | — |
| 423 | adam_m0_lr0.03_e4_b227_wd0.0001 | adam | 0 | 0.03 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.508307 ± 0.009406 | — |
| 424 | sgd_m0.9_lr0.1_e2_b64_wd0 | sgd | 0.9 | 0.1 | 2 | 64 | 0 | 40 | 2.504883 | 0.508203 ± 0.008430 | — |
| 425 | sgd_m0.9_lr0.1_e2_b64_wd0.0001 | sgd | 0.9 | 0.1 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.508098 ± 0.008450 | — |
| 426 | adam_m0_lr0.03_e8_b227_wd0.0001 | adam | 0 | 0.03 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.507994 ± 0.007074 | — |
| 427 | adam_m0_lr0.03_e8_b227_wd0 | adam | 0 | 0.03 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.507680 ± 0.007390 | — |
| 428 | sgd_m0.9_lr0.1_e4_b128_wd0.0001 | sgd | 0.9 | 0.1 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.507576 ± 0.007962 | — |
| 429 | sgd_m0.9_lr0.1_e4_b128_wd0 | sgd | 0.9 | 0.1 | 4 | 128 | 0 | 40 | 4.560547 | 0.507471 ± 0.007953 | — |
| 430 | sgd_m0.9_lr0.03_e1_b227_wd0 | sgd | 0.9 | 0.03 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.506740 ± 0.001495 | — |
| 431 | sgd_m0.9_lr0.03_e1_b227_wd0.0001 | sgd | 0.9 | 0.03 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.506740 ± 0.001495 | — |
| 432 | sgd_m0_lr0.03_e1_b227_wd0 | sgd | 0 | 0.03 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.506740 ± 0.001495 | — |
| 433 | sgd_m0_lr0.03_e1_b227_wd0.0001 | sgd | 0 | 0.03 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.506740 ± 0.001495 | — |
| 434 | sgd_m0.9_lr0.1_e1_b32_wd0.0001 | sgd | 0.9 | 0.1 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.506740 ± 0.009623 | — |
| 435 | sgd_m0.9_lr0.001_e2_b64_wd0 | sgd | 0.9 | 0.001 | 2 | 64 | 0 | 40 | 2.504883 | 0.506688 ± 0.001857 | — |
| 436 | sgd_m0.9_lr0.001_e2_b64_wd0.0001 | sgd | 0.9 | 0.001 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.506688 ± 0.001857 | — |
| 437 | sgd_m0.9_lr0.1_e1_b32_wd0 | sgd | 0.9 | 0.1 | 1 | 32 | 0 | 40 | 1.523438 | 0.506688 ± 0.009550 | — |
| 438 | sgd_m0.9_lr0.001_e1_b32_wd0 | sgd | 0.9 | 0.001 | 1 | 32 | 0 | 40 | 1.523438 | 0.506635 ± 0.003847 | — |
| 439 | sgd_m0.9_lr0.001_e1_b32_wd0.0001 | sgd | 0.9 | 0.001 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.506635 ± 0.003847 | — |
| 440 | sgd_m0.9_lr0.01_e1_b128_wd0 | sgd | 0.9 | 0.01 | 1 | 128 | 0 | 10 | 2.529297 | 0.506635 ± 0.001609 | — |
| 441 | sgd_m0.9_lr0.01_e1_b128_wd0.0001 | sgd | 0.9 | 0.01 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.506635 ± 0.001609 | — |
| 442 | adam_m0_lr0.0003_e4_b227_wd0.0001 | adam | 0 | 0.0003 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.506583 ± 0.001808 | — |
| 443 | adam_m0_lr0.0003_e4_b227_wd0 | adam | 0 | 0.0003 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.506531 ± 0.001719 | — |
| 444 | sgd_m0_lr0.3_e1_b32_wd0.0001 | sgd | 0 | 0.3 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.506374 ± 0.007056 | — |
| 445 | sgd_m0_lr0.3_e1_b32_wd0 | sgd | 0 | 0.3 | 1 | 32 | 0 | 40 | 1.523438 | 0.506322 ± 0.006979 | — |
| 446 | sgd_m0.9_lr0.1_e8_b227_wd0 | sgd | 0.9 | 0.1 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.506113 ± 0.006816 | — |
| 447 | sgd_m0.9_lr0.1_e8_b227_wd0.0001 | sgd | 0.9 | 0.1 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.506113 ± 0.006816 | — |
| 448 | sgd_m0.9_lr0.001_e4_b128_wd0 | sgd | 0.9 | 0.001 | 4 | 128 | 0 | 40 | 4.560547 | 0.505643 ± 0.001808 | — |
| 449 | sgd_m0.9_lr0.001_e4_b128_wd0.0001 | sgd | 0.9 | 0.001 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.505643 ± 0.001808 | — |
| 450 | sgd_m0_lr0.3_e2_b64_wd0 | sgd | 0 | 0.3 | 2 | 64 | 0 | 40 | 2.504883 | 0.505590 ± 0.005973 | — |
| 451 | sgd_m0_lr0.3_e2_b64_wd0.0001 | sgd | 0 | 0.3 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.505538 ± 0.006055 | — |
| 452 | sgd_m0.9_lr0.01_e2_b227_wd0 | sgd | 0.9 | 0.01 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.505381 ± 0.001044 | — |
| 453 | sgd_m0.9_lr0.01_e2_b227_wd0.0001 | sgd | 0.9 | 0.01 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.505381 ± 0.001044 | — |
| 454 | sgd_m0.9_lr0.001_e8_b227_wd0 | sgd | 0.9 | 0.001 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.505277 ± 0.001134 | — |
| 455 | sgd_m0.9_lr0.001_e8_b227_wd0.0001 | sgd | 0.9 | 0.001 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.505277 ± 0.001134 | — |
| 456 | adam_m0_lr0.1_e8_b64_wd0.0001 | adam | 0 | 0.1 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.504859 ± 0.010877 | — |
| 457 | adam_m0_lr0.1_e8_b64_wd0 | adam | 0 | 0.1 | 8 | 64 | 0 | 160 | 4.531250 | 0.504754 ± 0.012402 | — |
| 458 | adam_m0_lr0.0003_e2_b128_wd0 | adam | 0 | 0.0003 | 2 | 128 | 0 | 20 | 3.349609 | 0.504389 ± 0.002590 | — |
| 459 | adam_m0_lr0.0003_e2_b128_wd0.0001 | adam | 0 | 0.0003 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.504389 ± 0.002590 | — |
| 460 | sgd_m0.9_lr0.3_e8_b32_wd0 | sgd | 0.9 | 0.3 | 8 | 32 | 0 | 320 | 3.261719 | 0.504310 ± 0.007466 | — |
| 461 | adam_m0_lr0.03_e8_b32_wd0.0001 | adam | 0 | 0.03 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.504284 ± 0.011834 | — |
| 462 | sgd_m0_lr10_e2_b64_wd0.0001 | sgd | 0 | 10 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.504284 ± 0.007420 | — |
| 463 | adam_m0_lr0.03_e8_b32_wd0 | adam | 0 | 0.03 | 8 | 32 | 0 | 320 | 3.261719 | 0.504232 ± 0.011824 | — |
| 464 | sgd_m0_lr10_e1_b128_wd0.0001 | sgd | 0 | 10 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.504023 ± 0.006968 | — |
| 465 | sgd_m0.9_lr1_e4_b64_wd0.0001 | sgd | 0.9 | 1 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.503814 ± 0.006879 | — |
| 466 | sgd_m0_lr0.3_e4_b128_wd0.0001 | sgd | 0 | 0.3 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.503657 ± 0.005251 | — |
| 467 | sgd_m0_lr0.3_e4_b128_wd0 | sgd | 0 | 0.3 | 4 | 128 | 0 | 40 | 4.560547 | 0.503553 ± 0.005267 | — |
| 468 | sgd_m0.9_lr3_e8_b227_wd0.0001 | sgd | 0.9 | 3 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.503448 ± 0.008387 | — |
| 469 | sgd_m0.9_lr3_e8_b227_wd0 | sgd | 0.9 | 3 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.503422 ± 0.008050 | — |
| 470 | sgd_m0.9_lr1_e4_b64_wd0 | sgd | 0.9 | 1 | 4 | 64 | 0 | 80 | 3.320312 | 0.503396 ± 0.004242 | — |
| 471 | sgd_m0.9_lr3_e2_b64_wd0.0001 | sgd | 0.9 | 3 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.503083 ± 0.003663 | — |
| 472 | sgd_m0_lr10_e2_b64_wd0 | sgd | 0 | 10 | 2 | 64 | 0 | 40 | 2.504883 | 0.502874 ± 0.004977 | — |
| 473 | sgd_m0_lr10_e4_b128_wd0.0001 | sgd | 0 | 10 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.502874 ± 0.004977 | — |
| 474 | adam_m0_lr0.03_e8_b64_wd0 | adam | 0 | 0.03 | 8 | 64 | 0 | 160 | 4.531250 | 0.502508 ± 0.018116 | — |
| 475 | adam_m0_lr0.03_e8_b64_wd0.0001 | adam | 0 | 0.03 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.502456 ± 0.018036 | — |
| 476 | adam_m0_lr0.03_e4_b64_wd0.0001 | adam | 0 | 0.03 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.502299 ± 0.012735 | — |
| 477 | adam_m0_lr0.03_e4_b64_wd0 | adam | 0 | 0.03 | 4 | 64 | 0 | 80 | 3.320312 | 0.502247 ± 0.012728 | — |
| 478 | sgd_m0_lr0.3_e8_b227_wd0.0001 | sgd | 0 | 0.3 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.501881 ± 0.005806 | — |
| 479 | sgd_m0_lr0.3_e8_b227_wd0 | sgd | 0 | 0.3 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.501829 ± 0.005725 | — |
| 480 | sgd_m0.9_lr3_e4_b128_wd0 | sgd | 0.9 | 3 | 4 | 128 | 0 | 40 | 4.560547 | 0.501672 ± 0.017302 | — |
| 481 | sgd_m0.9_lr0.3_e8_b32_wd0.0001 | sgd | 0.9 | 0.3 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.501437 ± 0.002489 | — |
| 482 | sgd_m0.9_lr3_e1_b32_wd0.0001 | sgd | 0.9 | 3 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.501437 ± 0.002489 | — |
| 483 | sgd_m0_lr10_e1_b32_wd0.0001 | sgd | 0 | 10 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.501437 ± 0.002489 | — |
| 484 | sgd_m0_lr10_e1_b64_wd0 | sgd | 0 | 10 | 1 | 64 | 0 | 20 | 1.967773 | 0.501437 ± 0.002489 | — |
| 485 | sgd_m0.9_lr1_e2_b32_wd0 | sgd | 0.9 | 1 | 2 | 32 | 0 | 80 | 1.875000 | 0.501306 ± 0.006757 | — |
| 486 | sgd_m0_lr10_e2_b128_wd0 | sgd | 0 | 10 | 2 | 128 | 0 | 20 | 3.349609 | 0.501306 ± 0.002262 | — |
| 487 | sgd_m0_lr10_e2_b128_wd0.0001 | sgd | 0 | 10 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.501306 ± 0.002262 | — |
| 488 | sgd_m0_lr10_e4_b128_wd0 | sgd | 0 | 10 | 4 | 128 | 0 | 40 | 4.560547 | 0.501306 ± 0.006757 | — |
| 489 | sgd_m0.9_lr0.03_e8_b32_wd0.0001 | sgd | 0.9 | 0.03 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.500836 ± 0.009573 | — |
| 490 | sgd_m0.9_lr0.03_e8_b32_wd0 | sgd | 0.9 | 0.03 | 8 | 32 | 0 | 320 | 3.261719 | 0.500731 ± 0.009489 | — |
| 491 | sgd_m0_lr0.3_e8_b32_wd0 | sgd | 0 | 0.3 | 8 | 32 | 0 | 320 | 3.261719 | 0.500549 ± 0.011512 | — |
| 492 | sgd_m0_lr0.3_e8_b32_wd0.0001 | sgd | 0 | 0.3 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.500549 ± 0.011641 | — |
| 493 | sgd_m0.9_lr10_e1_b32_wd0.0001 | sgd | 0.9 | 10 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.500000 ± 0.000000 | — |
| 494 | sgd_m0.9_lr10_e1_b64_wd0 | sgd | 0.9 | 10 | 1 | 64 | 0 | 20 | 1.967773 | 0.500000 ± 0.000000 | — |
| 495 | sgd_m0.9_lr10_e1_b64_wd0.0001 | sgd | 0.9 | 10 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.500000 ± 0.000000 | — |
| 496 | sgd_m0.9_lr10_e2_b128_wd0 | sgd | 0.9 | 10 | 2 | 128 | 0 | 20 | 3.349609 | 0.500000 ± 0.000000 | — |
| 497 | sgd_m0.9_lr10_e2_b64_wd0 | sgd | 0.9 | 10 | 2 | 64 | 0 | 40 | 2.504883 | 0.500000 ± 0.000000 | — |
| 498 | sgd_m0.9_lr10_e2_b64_wd0.0001 | sgd | 0.9 | 10 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.500000 ± 0.000000 | — |
| 499 | sgd_m0.9_lr10_e4_b128_wd0 | sgd | 0.9 | 10 | 4 | 128 | 0 | 40 | 4.560547 | 0.500000 ± 0.000000 | — |
| 500 | sgd_m0.9_lr10_e4_b128_wd0.0001 | sgd | 0.9 | 10 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.500000 ± 0.000000 | — |
| 501 | sgd_m0.9_lr10_e4_b227_wd0 | sgd | 0.9 | 10 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.500000 ± 0.000000 | — |
| 502 | sgd_m0.9_lr10_e4_b227_wd0.0001 | sgd | 0.9 | 10 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.500000 ± 0.000000 | — |
| 503 | sgd_m0.9_lr10_e4_b32_wd0 | sgd | 0.9 | 10 | 4 | 32 | 0 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 504 | sgd_m0.9_lr10_e4_b32_wd0.0001 | sgd | 0.9 | 10 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 505 | sgd_m0.9_lr10_e4_b64_wd0 | sgd | 0.9 | 10 | 4 | 64 | 0 | 80 | 3.320312 | 0.500000 ± 0.000000 | — |
| 506 | sgd_m0.9_lr10_e4_b64_wd0.0001 | sgd | 0.9 | 10 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.500000 ± 0.000000 | — |
| 507 | sgd_m0.9_lr10_e8_b227_wd0 | sgd | 0.9 | 10 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.500000 ± 0.000000 | — |
| 508 | sgd_m0.9_lr10_e8_b227_wd0.0001 | sgd | 0.9 | 10 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.500000 ± 0.000000 | — |
| 509 | sgd_m0.9_lr10_e8_b32_wd0 | sgd | 0.9 | 10 | 8 | 32 | 0 | 320 | 3.261719 | 0.500000 ± 0.017241 | — |
| 510 | sgd_m0.9_lr1_e4_b32_wd0 | sgd | 0.9 | 1 | 4 | 32 | 0 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 511 | sgd_m0.9_lr1_e4_b32_wd0.0001 | sgd | 0.9 | 1 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 512 | sgd_m0.9_lr1_e8_b32_wd0 | sgd | 0.9 | 1 | 8 | 32 | 0 | 320 | 3.261719 | 0.500000 ± 0.000000 | — |
| 513 | sgd_m0.9_lr1_e8_b32_wd0.0001 | sgd | 0.9 | 1 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.500000 ± 0.000000 | — |
| 514 | sgd_m0.9_lr1_e8_b64_wd0 | sgd | 0.9 | 1 | 8 | 64 | 0 | 160 | 4.531250 | 0.500000 ± 0.000000 | — |
| 515 | sgd_m0.9_lr1_e8_b64_wd0.0001 | sgd | 0.9 | 1 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.500000 ± 0.000000 | — |
| 516 | sgd_m0.9_lr3_e2_b32_wd0 | sgd | 0.9 | 3 | 2 | 32 | 0 | 80 | 1.875000 | 0.500000 ± 0.000000 | — |
| 517 | sgd_m0.9_lr3_e2_b32_wd0.0001 | sgd | 0.9 | 3 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.500000 ± 0.000000 | — |
| 518 | sgd_m0.9_lr3_e4_b32_wd0 | sgd | 0.9 | 3 | 4 | 32 | 0 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 519 | sgd_m0.9_lr3_e4_b32_wd0.0001 | sgd | 0.9 | 3 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 520 | sgd_m0.9_lr3_e4_b64_wd0 | sgd | 0.9 | 3 | 4 | 64 | 0 | 80 | 3.320312 | 0.500000 ± 0.000000 | — |
| 521 | sgd_m0.9_lr3_e4_b64_wd0.0001 | sgd | 0.9 | 3 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.500000 ± 0.000000 | — |
| 522 | sgd_m0.9_lr3_e8_b128_wd0 | sgd | 0.9 | 3 | 8 | 128 | 0 | 80 | 6.308594 | 0.500000 ± 0.000000 | — |
| 523 | sgd_m0.9_lr3_e8_b128_wd0.0001 | sgd | 0.9 | 3 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.500000 ± 0.000000 | — |
| 524 | sgd_m0.9_lr3_e8_b32_wd0.0001 | sgd | 0.9 | 3 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.500000 ± 0.000000 | — |
| 525 | sgd_m0.9_lr3_e8_b64_wd0 | sgd | 0.9 | 3 | 8 | 64 | 0 | 160 | 4.531250 | 0.500000 ± 0.000000 | — |
| 526 | sgd_m0.9_lr3_e8_b64_wd0.0001 | sgd | 0.9 | 3 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.500000 ± 0.000000 | — |
| 527 | sgd_m0_lr10_e1_b32_wd0 | sgd | 0 | 10 | 1 | 32 | 0 | 40 | 1.523438 | 0.500000 ± 0.000000 | — |
| 528 | sgd_m0_lr10_e2_b227_wd0.0001 | sgd | 0 | 10 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.500000 ± 0.000000 | — |
| 529 | sgd_m0_lr10_e2_b32_wd0 | sgd | 0 | 10 | 2 | 32 | 0 | 80 | 1.875000 | 0.500000 ± 0.000000 | — |
| 530 | sgd_m0_lr10_e2_b32_wd0.0001 | sgd | 0 | 10 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.500000 ± 0.000000 | — |
| 531 | sgd_m0_lr10_e4_b32_wd0 | sgd | 0 | 10 | 4 | 32 | 0 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 532 | sgd_m0_lr10_e4_b32_wd0.0001 | sgd | 0 | 10 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.500000 ± 0.000000 | — |
| 533 | sgd_m0_lr10_e4_b64_wd0 | sgd | 0 | 10 | 4 | 64 | 0 | 80 | 3.320312 | 0.500000 ± 0.000000 | — |
| 534 | sgd_m0_lr10_e4_b64_wd0.0001 | sgd | 0 | 10 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.500000 ± 0.000000 | — |
| 535 | sgd_m0_lr10_e8_b32_wd0 | sgd | 0 | 10 | 8 | 32 | 0 | 320 | 3.261719 | 0.500000 ± 0.000000 | — |
| 536 | sgd_m0_lr10_e8_b32_wd0.0001 | sgd | 0 | 10 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.500000 ± 0.000000 | — |
| 537 | sgd_m0_lr10_e8_b64_wd0 | sgd | 0 | 10 | 8 | 64 | 0 | 160 | 4.531250 | 0.499896 ± 0.000181 | — |
| 538 | sgd_m0_lr10_e8_b64_wd0.0001 | sgd | 0 | 10 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.499896 ± 0.000181 | — |
| 539 | adam_m0_lr0.0003_e1_b64_wd0 | adam | 0 | 0.0003 | 1 | 64 | 0 | 20 | 1.967773 | 0.499843 ± 0.001766 | — |
| 540 | adam_m0_lr0.0003_e1_b64_wd0.0001 | adam | 0 | 0.0003 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.499843 ± 0.001766 | — |
| 541 | sgd_m0_lr1_e4_b64_wd0 | sgd | 0 | 1 | 4 | 64 | 0 | 80 | 3.320312 | 0.499843 ± 0.012382 | — |
| 542 | sgd_m0_lr1_e4_b64_wd0.0001 | sgd | 0 | 1 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.499791 ± 0.012343 | — |
| 543 | sgd_m0.9_lr10_e2_b32_wd0 | sgd | 0.9 | 10 | 2 | 32 | 0 | 80 | 1.875000 | 0.499739 ± 0.000452 | — |
| 544 | sgd_m0_lr10_e1_b64_wd0.0001 | sgd | 0 | 10 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.499660 ± 0.004829 | — |
| 545 | sgd_m0.9_lr10_e8_b128_wd0 | sgd | 0.9 | 10 | 8 | 128 | 0 | 80 | 6.308594 | 0.499634 ± 0.000633 | — |
| 546 | sgd_m0.9_lr10_e8_b128_wd0.0001 | sgd | 0.9 | 10 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.499634 ± 0.000633 | — |
| 547 | sgd_m0_lr1_e2_b32_wd0 | sgd | 0 | 1 | 2 | 32 | 0 | 80 | 1.875000 | 0.499634 ± 0.013026 | — |
| 548 | adam_m0_lr0.03_e4_b32_wd0 | adam | 0 | 0.03 | 4 | 32 | 0 | 160 | 2.426758 | 0.499582 ± 0.015924 | — |
| 549 | sgd_m0_lr10_e8_b227_wd0 | sgd | 0 | 10 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.499582 ± 0.000394 | — |
| 550 | sgd_m0_lr1_e2_b32_wd0.0001 | sgd | 0 | 1 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.499530 ± 0.013348 | — |
| 551 | sgd_m0.9_lr10_e8_b64_wd0 | sgd | 0.9 | 10 | 8 | 64 | 0 | 160 | 4.531250 | 0.499451 ± 0.000950 | — |
| 552 | adam_m0_lr0.03_e4_b32_wd0.0001 | adam | 0 | 0.03 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.499425 ± 0.015988 | — |
| 553 | adam_m0_lr0.03_e2_b32_wd0 | adam | 0 | 0.03 | 2 | 32 | 0 | 80 | 1.875000 | 0.499112 ± 0.007460 | — |
| 554 | adam_m0_lr0.03_e2_b32_wd0.0001 | adam | 0 | 0.03 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.499112 ± 0.007460 | — |
| 555 | adam_m0_lr0.003_e8_b32_wd0.0001 | adam | 0 | 0.003 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.499060 ± 0.004710 | — |
| 556 | adam_m0_lr0.03_e8_b128_wd0 | adam | 0 | 0.03 | 8 | 128 | 0 | 80 | 6.308594 | 0.499060 ± 0.013680 | — |
| 557 | adam_m0_lr0.03_e8_b128_wd0.0001 | adam | 0 | 0.03 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.499060 ± 0.013680 | — |
| 558 | adam_m0_lr0.003_e8_b32_wd0 | adam | 0 | 0.003 | 8 | 32 | 0 | 320 | 3.261719 | 0.498955 ± 0.004621 | — |
| 559 | adam_m0_lr0.01_e4_b64_wd0 | adam | 0 | 0.01 | 4 | 64 | 0 | 80 | 3.320312 | 0.498955 ± 0.012083 | — |
| 560 | sgd_m0_lr3_e4_b64_wd0.0001 | sgd | 0 | 3 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.498851 ± 0.036181 | — |
| 561 | adam_m0_lr0.01_e4_b64_wd0.0001 | adam | 0 | 0.01 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.498798 ± 0.012103 | — |
| 562 | sgd_m0.9_lr3_e8_b32_wd0 | sgd | 0.9 | 3 | 8 | 32 | 0 | 320 | 3.261719 | 0.498563 ± 0.002489 | — |
| 563 | adam_m0_lr0.01_e2_b32_wd0 | adam | 0 | 0.01 | 2 | 32 | 0 | 80 | 1.875000 | 0.498328 ± 0.010797 | — |
| 564 | sgd_m0.9_lr1_e2_b32_wd0.0001 | sgd | 0.9 | 1 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.498302 ± 0.002941 | — |
| 565 | sgd_m0_lr1_e8_b128_wd0 | sgd | 0 | 1 | 8 | 128 | 0 | 80 | 6.308594 | 0.498276 ± 0.014078 | — |
| 566 | sgd_m0_lr1_e8_b128_wd0.0001 | sgd | 0 | 1 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.498276 ± 0.014118 | — |
| 567 | adam_m0_lr0.01_e2_b32_wd0.0001 | adam | 0 | 0.01 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.498224 ± 0.010697 | — |
| 568 | adam_m0_lr0.01_e8_b128_wd0.0001 | adam | 0 | 0.01 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.497283 ± 0.011641 | — |
| 569 | adam_m0_lr0.01_e8_b128_wd0 | adam | 0 | 0.01 | 8 | 128 | 0 | 80 | 6.308594 | 0.497179 ± 0.011691 | — |
| 570 | adam_m0_lr0.0001_e8_b128_wd0 | adam | 0 | 0.0001 | 8 | 128 | 0 | 80 | 6.308594 | 0.497022 ± 0.002194 | — |
| 571 | adam_m0_lr0.0001_e8_b128_wd0.0001 | adam | 0 | 0.0001 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.497022 ± 0.002194 | — |
| 572 | sgd_m0.9_lr0.03_e4_b64_wd0 | sgd | 0.9 | 0.03 | 4 | 64 | 0 | 80 | 3.320312 | 0.496813 ± 0.008747 | — |
| 573 | sgd_m0.9_lr0.03_e4_b64_wd0.0001 | sgd | 0.9 | 0.03 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.496813 ± 0.008747 | — |
| 574 | adam_m0_lr0.0001_e4_b64_wd0 | adam | 0 | 0.0001 | 4 | 64 | 0 | 80 | 3.320312 | 0.496499 ± 0.001985 | — |
| 575 | adam_m0_lr0.0001_e4_b64_wd0.0001 | adam | 0 | 0.0001 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.496499 ± 0.001985 | — |
| 576 | sgd_m0_lr0.1_e4_b64_wd0 | sgd | 0 | 0.1 | 4 | 64 | 0 | 80 | 3.320312 | 0.496343 ± 0.010402 | — |
| 577 | sgd_m0_lr0.1_e4_b64_wd0.0001 | sgd | 0 | 0.1 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.496343 ± 0.010304 | — |
| 578 | sgd_m0.9_lr0.1_e2_b32_wd0.0001 | sgd | 0.9 | 0.1 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.496186 ± 0.005882 | — |
| 579 | sgd_m0.9_lr3_e2_b64_wd0 | sgd | 0.9 | 3 | 2 | 64 | 0 | 40 | 2.504883 | 0.496160 ± 0.012481 | — |
| 580 | sgd_m0.9_lr0.1_e2_b32_wd0 | sgd | 0.9 | 0.1 | 2 | 32 | 0 | 80 | 1.875000 | 0.496082 ± 0.005973 | — |
| 581 | adam_m0_lr0.1_e8_b32_wd0.0001 | adam | 0 | 0.1 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.495899 ± 0.048762 | — |
| 582 | adam_m0_lr0.1_e8_b32_wd0 | adam | 0 | 0.1 | 8 | 32 | 0 | 320 | 3.261719 | 0.495768 ± 0.048906 | — |
| 583 | sgd_m0.9_lr0.03_e2_b32_wd0 | sgd | 0.9 | 0.03 | 2 | 32 | 0 | 80 | 1.875000 | 0.495402 ± 0.007478 | — |
| 584 | sgd_m0.9_lr0.03_e2_b32_wd0.0001 | sgd | 0.9 | 0.03 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.495402 ± 0.007478 | — |
| 585 | sgd_m0.9_lr0.1_e4_b64_wd0 | sgd | 0.9 | 0.1 | 4 | 64 | 0 | 80 | 3.320312 | 0.495246 ± 0.010332 | — |
| 586 | sgd_m0.9_lr10_e1_b128_wd0 | sgd | 0.9 | 10 | 1 | 128 | 0 | 10 | 2.529297 | 0.495246 ± 0.008235 | — |
| 587 | sgd_m0.9_lr0.1_e4_b64_wd0.0001 | sgd | 0.9 | 0.1 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.495141 ± 0.010263 | — |
| 588 | sgd_m0_lr3_e8_b32_wd0.0001 | sgd | 0 | 3 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.495141 ± 0.008416 | — |
| 589 | sgd_m0.9_lr10_e1_b128_wd0.0001 | sgd | 0.9 | 10 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.494880 ± 0.008868 | — |
| 590 | sgd_m0_lr0.1_e8_b128_wd0 | sgd | 0 | 0.1 | 8 | 128 | 0 | 80 | 6.308594 | 0.494723 ± 0.009540 | — |
| 591 | sgd_m0_lr0.1_e8_b128_wd0.0001 | sgd | 0 | 0.1 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.494723 ± 0.009540 | — |
| 592 | sgd_m0.9_lr10_e8_b32_wd0.0001 | sgd | 0.9 | 10 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.494645 ± 0.005947 | — |
| 593 | sgd_m0.9_lr0.03_e8_b128_wd0 | sgd | 0.9 | 0.03 | 8 | 128 | 0 | 80 | 6.308594 | 0.494566 ± 0.009072 | — |
| 594 | sgd_m0.9_lr0.03_e8_b128_wd0.0001 | sgd | 0.9 | 0.03 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.494566 ± 0.009072 | — |
| 595 | sgd_m0_lr1_e8_b32_wd0.0001 | sgd | 0 | 1 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.494148 ± 0.026281 | — |
| 596 | sgd_m0_lr3_e8_b32_wd0 | sgd | 0 | 3 | 8 | 32 | 0 | 320 | 3.261719 | 0.493835 ± 0.010678 | — |
| 597 | sgd_m0_lr0.1_e2_b32_wd0 | sgd | 0 | 0.1 | 2 | 32 | 0 | 80 | 1.875000 | 0.493417 ± 0.010370 | — |
| 598 | sgd_m0_lr0.1_e2_b32_wd0.0001 | sgd | 0 | 0.1 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.493417 ± 0.010370 | — |
| 599 | adam_m0_lr0.0001_e2_b32_wd0 | adam | 0 | 0.0001 | 2 | 32 | 0 | 80 | 1.875000 | 0.493312 ± 0.001538 | — |
| 600 | adam_m0_lr0.0001_e2_b32_wd0.0001 | adam | 0 | 0.0001 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.493312 ± 0.001538 | — |
| 601 | sgd_m0_lr0.01_e1_b128_wd0 | sgd | 0 | 0.01 | 1 | 128 | 0 | 10 | 2.529297 | 0.493208 ± 0.001601 | — |
| 602 | sgd_m0_lr0.01_e1_b128_wd0.0001 | sgd | 0 | 0.01 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.493208 ± 0.001601 | — |
| 603 | sgd_m0_lr10_e4_b227_wd0 | sgd | 0 | 10 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.492921 ± 0.012262 | — |
| 604 | sgd_m0.9_lr1_e8_b128_wd0.0001 | sgd | 0.9 | 1 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.492816 ± 0.039795 | — |
| 605 | sgd_m0_lr10_e2_b227_wd0 | sgd | 0 | 10 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.492555 ± 0.012895 | — |
| 606 | sgd_m0_lr3_e4_b32_wd0.0001 | sgd | 0 | 3 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.492241 ± 0.013438 | — |
| 607 | sgd_m0_lr10_e8_b128_wd0 | sgd | 0 | 10 | 8 | 128 | 0 | 80 | 6.308594 | 0.492111 ± 0.013665 | — |
| 608 | sgd_m0_lr0.01_e2_b227_wd0 | sgd | 0 | 0.01 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.492006 ± 0.001183 | — |
| 609 | sgd_m0_lr0.01_e2_b227_wd0.0001 | sgd | 0 | 0.01 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.492006 ± 0.001183 | — |
| 610 | sgd_m0_lr10_e8_b128_wd0.0001 | sgd | 0 | 10 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.491954 ± 0.013936 | — |
| 611 | sgd_m0_lr10_e4_b227_wd0.0001 | sgd | 0 | 10 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.491902 ± 0.014027 | — |
| 612 | sgd_m0.9_lr0.01_e8_b64_wd0 | sgd | 0.9 | 0.01 | 8 | 64 | 0 | 160 | 4.531250 | 0.491379 ± 0.008923 | — |
| 613 | sgd_m0.9_lr0.01_e8_b64_wd0.0001 | sgd | 0.9 | 0.01 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.491379 ± 0.008923 | — |
| 614 | sgd_m0.9_lr10_e2_b32_wd0.0001 | sgd | 0.9 | 10 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.491223 ± 0.015203 | — |
| 615 | sgd_m0.9_lr0.1_e8_b128_wd0.0001 | sgd | 0.9 | 0.1 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.491066 ± 0.009561 | — |
| 616 | sgd_m0.9_lr0.1_e8_b128_wd0 | sgd | 0.9 | 0.1 | 8 | 128 | 0 | 80 | 6.308594 | 0.491014 ± 0.009633 | — |
| 617 | sgd_m0_lr3_e4_b64_wd0 | sgd | 0 | 3 | 4 | 64 | 0 | 80 | 3.320312 | 0.490439 ± 0.045856 | — |
| 618 | sgd_m0_lr1_e8_b32_wd0 | sgd | 0 | 1 | 8 | 32 | 0 | 320 | 3.261719 | 0.489864 ± 0.026259 | — |
| 619 | sgd_m0.9_lr0.01_e4_b32_wd0 | sgd | 0.9 | 0.01 | 4 | 32 | 0 | 160 | 2.426758 | 0.489812 ± 0.008777 | — |
| 620 | sgd_m0.9_lr0.01_e4_b32_wd0.0001 | sgd | 0.9 | 0.01 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.489812 ± 0.008777 | — |
| 621 | adam_m0_lr0.0003_e2_b227_wd0 | adam | 0 | 0.0003 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.489655 ± 0.001437 | — |
| 622 | adam_m0_lr0.0003_e2_b227_wd0.0001 | adam | 0 | 0.0003 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.489655 ± 0.001437 | — |
| 623 | sgd_m0_lr0.03_e8_b32_wd0 | sgd | 0 | 0.03 | 8 | 32 | 0 | 320 | 3.261719 | 0.489498 ± 0.003147 | — |
| 624 | sgd_m0_lr0.03_e8_b32_wd0.0001 | sgd | 0 | 0.03 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.489498 ± 0.003147 | — |
| 625 | sgd_m0_lr3_e4_b32_wd0 | sgd | 0 | 3 | 4 | 32 | 0 | 160 | 2.426758 | 0.489342 ± 0.018461 | — |
| 626 | sgd_m0_lr3_e8_b64_wd0 | sgd | 0 | 3 | 8 | 64 | 0 | 160 | 4.531250 | 0.489054 ± 0.034969 | — |
| 627 | sgd_m0.9_lr0.03_e8_b64_wd0 | sgd | 0.9 | 0.03 | 8 | 64 | 0 | 160 | 4.531250 | 0.488715 ± 0.011564 | — |
| 628 | adam_m0_lr0.01_e8_b64_wd0.0001 | adam | 0 | 0.01 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.488715 ± 0.012317 | — |
| 629 | adam_m0_lr0.01_e8_b64_wd0 | adam | 0 | 0.01 | 8 | 64 | 0 | 160 | 4.531250 | 0.488610 ± 0.012227 | — |
| 630 | sgd_m0.9_lr0.03_e8_b64_wd0.0001 | sgd | 0.9 | 0.03 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.488610 ± 0.011520 | — |
| 631 | sgd_m0.9_lr10_e2_b227_wd0.0001 | sgd | 0.9 | 10 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.488245 ± 0.028159 | — |
| 632 | sgd_m0.9_lr0.3_e4_b32_wd0.0001 | sgd | 0.9 | 0.3 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.488009 ± 0.039006 | — |
| 633 | adam_m0_lr0.01_e8_b32_wd0 | adam | 0 | 0.01 | 8 | 32 | 0 | 320 | 3.261719 | 0.487461 ± 0.008073 | — |
| 634 | sgd_m0_lr0.001_e4_b64_wd0 | sgd | 0 | 0.001 | 4 | 64 | 0 | 80 | 3.320312 | 0.487461 ± 0.000565 | — |
| 635 | sgd_m0_lr0.001_e4_b64_wd0.0001 | sgd | 0 | 0.001 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.487461 ± 0.000565 | — |
| 636 | adam_m0_lr0.01_e8_b32_wd0.0001 | adam | 0 | 0.01 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.487409 ± 0.008072 | — |
| 637 | adam_m0_lr0.0003_e1_b128_wd0.0001 | adam | 0 | 0.0003 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.487304 ± 0.000953 | — |
| 638 | sgd_m0.9_lr1_e8_b128_wd0 | sgd | 0.9 | 1 | 8 | 128 | 0 | 80 | 6.308594 | 0.487252 ± 0.050093 | — |
| 639 | adam_m0_lr0.0003_e1_b128_wd0 | adam | 0 | 0.0003 | 1 | 128 | 0 | 10 | 2.529297 | 0.487200 ± 0.001044 | — |
| 640 | sgd_m0_lr10_e8_b227_wd0.0001 | sgd | 0 | 10 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.487043 ± 0.032122 | — |
| 641 | adam_m0_lr0.0001_e8_b227_wd0.0001 | adam | 0 | 0.0001 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.487017 ± 0.001556 | — |
| 642 | adam_m0_lr0.0001_e8_b227_wd0 | adam | 0 | 0.0001 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.486991 ± 0.001591 | — |
| 643 | sgd_m0_lr0.001_e2_b32_wd0 | sgd | 0 | 0.001 | 2 | 32 | 0 | 80 | 1.875000 | 0.486991 ± 0.001512 | — |
| 644 | sgd_m0_lr0.001_e2_b32_wd0.0001 | sgd | 0 | 0.001 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.486991 ± 0.001512 | — |
| 645 | sgd_m0_lr0.001_e8_b128_wd0 | sgd | 0 | 0.001 | 8 | 128 | 0 | 80 | 6.308594 | 0.486729 ± 0.001044 | — |
| 646 | sgd_m0_lr0.001_e8_b128_wd0.0001 | sgd | 0 | 0.001 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.486729 ± 0.001044 | — |
| 647 | sgd_m0.9_lr0.03_e4_b32_wd0.0001 | sgd | 0.9 | 0.03 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.486468 ± 0.011221 | — |
| 648 | sgd_m0.9_lr0.03_e4_b32_wd0 | sgd | 0.9 | 0.03 | 4 | 32 | 0 | 160 | 2.426758 | 0.486416 ± 0.011049 | — |
| 649 | adam_m0_lr0.0001_e4_b128_wd0 | adam | 0 | 0.0001 | 4 | 128 | 0 | 40 | 4.560547 | 0.486364 ± 0.001659 | — |
| 650 | adam_m0_lr0.0001_e4_b128_wd0.0001 | adam | 0 | 0.0001 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.486364 ± 0.001659 | — |
| 651 | adam_m0_lr0.01_e4_b32_wd0.0001 | adam | 0 | 0.01 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.486102 ± 0.012717 | — |
| 652 | sgd_m0_lr0.3_e8_b64_wd0 | sgd | 0 | 0.3 | 8 | 64 | 0 | 160 | 4.531250 | 0.486102 ± 0.010656 | — |
| 653 | adam_m0_lr0.01_e4_b32_wd0 | adam | 0 | 0.01 | 4 | 32 | 0 | 160 | 2.426758 | 0.486050 ± 0.012665 | — |
| 654 | sgd_m0_lr0.3_e8_b64_wd0.0001 | sgd | 0 | 0.3 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.485998 ± 0.010837 | — |
| 655 | sgd_m0.9_lr10_e8_b64_wd0.0001 | sgd | 0.9 | 10 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.484979 ± 0.026017 | — |
| 656 | sgd_m0.9_lr0.01_e8_b32_wd0.0001 | sgd | 0.9 | 0.01 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.484953 ± 0.005161 | — |
| 657 | sgd_m0.9_lr0.01_e8_b32_wd0 | sgd | 0.9 | 0.01 | 8 | 32 | 0 | 320 | 3.261719 | 0.484848 ± 0.004980 | — |
| 658 | adam_m0_lr0.0001_e2_b64_wd0 | adam | 0 | 0.0001 | 2 | 64 | 0 | 40 | 2.504883 | 0.484483 ± 0.002056 | — |
| 659 | adam_m0_lr0.0001_e2_b64_wd0.0001 | adam | 0 | 0.0001 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.484483 ± 0.002056 | — |
| 660 | sgd_m0_lr0.3_e4_b32_wd0 | sgd | 0 | 0.3 | 4 | 32 | 0 | 160 | 2.426758 | 0.484065 ± 0.012271 | — |
| 661 | sgd_m0_lr0.3_e4_b32_wd0.0001 | sgd | 0 | 0.3 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.483908 ± 0.012176 | — |
| 662 | sgd_m0_lr0.3_e4_b64_wd0 | sgd | 0 | 0.3 | 4 | 64 | 0 | 80 | 3.320312 | 0.482654 ± 0.014188 | — |
| 663 | sgd_m0_lr0.3_e4_b64_wd0.0001 | sgd | 0 | 0.3 | 4 | 64 | 0.0001 | 80 | 3.320312 | 0.482550 ± 0.014089 | — |
| 664 | sgd_m0_lr3_e8_b64_wd0.0001 | sgd | 0 | 3 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.481975 ± 0.031220 | — |
| 665 | sgd_m0_lr0.1_e8_b64_wd0.0001 | sgd | 0 | 0.1 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.481975 ± 0.008975 | — |
| 666 | sgd_m0_lr0.1_e8_b64_wd0 | sgd | 0 | 0.1 | 8 | 64 | 0 | 160 | 4.531250 | 0.481923 ± 0.008946 | — |
| 667 | sgd_m0_lr0.1_e8_b32_wd0 | sgd | 0 | 0.1 | 8 | 32 | 0 | 320 | 3.261719 | 0.481818 ± 0.005806 | — |
| 668 | sgd_m0_lr0.1_e8_b32_wd0.0001 | sgd | 0 | 0.1 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.481818 ± 0.005892 | — |
| 669 | sgd_m0_lr0.3_e2_b32_wd0 | sgd | 0 | 0.3 | 2 | 32 | 0 | 80 | 1.875000 | 0.481348 ± 0.015077 | — |
| 670 | sgd_m0_lr0.3_e2_b32_wd0.0001 | sgd | 0 | 0.3 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.481348 ± 0.015077 | — |
| 671 | sgd_m0_lr0.1_e4_b32_wd0 | sgd | 0 | 0.1 | 4 | 32 | 0 | 160 | 2.426758 | 0.480930 ± 0.008620 | — |
| 672 | sgd_m0_lr0.1_e4_b32_wd0.0001 | sgd | 0 | 0.1 | 4 | 32 | 0.0001 | 160 | 2.426758 | 0.480878 ± 0.008561 | — |
| 673 | adam_m0_lr0.0001_e1_b32_wd0 | adam | 0 | 0.0001 | 1 | 32 | 0 | 40 | 1.523438 | 0.480721 ± 0.000565 | — |
| 674 | adam_m0_lr0.0001_e1_b32_wd0.0001 | adam | 0 | 0.0001 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.480721 ± 0.000565 | — |
| 675 | sgd_m0_lr0.3_e8_b128_wd0 | sgd | 0 | 0.3 | 8 | 128 | 0 | 80 | 6.308594 | 0.480617 ± 0.013181 | — |
| 676 | sgd_m0_lr0.3_e8_b128_wd0.0001 | sgd | 0 | 0.3 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.480617 ± 0.013181 | — |
| 677 | sgd_m0.9_lr0.3_e8_b64_wd0.0001 | sgd | 0.9 | 0.3 | 8 | 64 | 0.0001 | 160 | 4.531250 | 0.480120 ± 0.047846 | — |
| 678 | sgd_m0.9_lr0.1_e8_b32_wd0.0001 | sgd | 0.9 | 0.1 | 8 | 32 | 0.0001 | 320 | 3.261719 | 0.479781 ± 0.033076 | — |
| 679 | sgd_m0.9_lr0.1_e8_b32_wd0 | sgd | 0.9 | 0.1 | 8 | 32 | 0 | 320 | 3.261719 | 0.479389 ± 0.034758 | — |
| 680 | adam_m0_lr0.0003_e1_b227_wd0 | adam | 0 | 0.0003 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.478945 ± 0.000789 | — |
| 681 | adam_m0_lr0.0003_e1_b227_wd0.0001 | adam | 0 | 0.0003 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.478945 ± 0.000789 | — |
| 682 | sgd_m0.9_lr0.3_e8_b64_wd0 | sgd | 0.9 | 0.3 | 8 | 64 | 0 | 160 | 4.531250 | 0.478710 ± 0.039463 | — |
| 683 | adam_m0_lr0.0001_e4_b227_wd0 | adam | 0 | 0.0001 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.478004 ± 0.001456 | — |
| 684 | adam_m0_lr0.0001_e4_b227_wd0.0001 | adam | 0 | 0.0001 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.478004 ± 0.001456 | — |
| 685 | adam_m0_lr0.0001_e2_b128_wd0 | adam | 0 | 0.0001 | 2 | 128 | 0 | 20 | 3.349609 | 0.476959 ± 0.000415 | — |
| 686 | adam_m0_lr0.0001_e2_b128_wd0.0001 | adam | 0 | 0.0001 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.476959 ± 0.000415 | — |
| 687 | sgd_m0_lr3_e8_b128_wd0.0001 | sgd | 0 | 3 | 8 | 128 | 0.0001 | 80 | 6.308594 | 0.476045 ± 0.047205 | — |
| 688 | sgd_m0.9_lr0.01_e1_b227_wd0 | sgd | 0.9 | 0.01 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.475810 ± 0.000394 | — |
| 689 | sgd_m0.9_lr0.01_e1_b227_wd0.0001 | sgd | 0.9 | 0.01 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.475810 ± 0.000394 | — |
| 690 | sgd_m0_lr0.01_e1_b227_wd0 | sgd | 0 | 0.01 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.475810 ± 0.000394 | — |
| 691 | sgd_m0_lr0.01_e1_b227_wd0.0001 | sgd | 0 | 0.01 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.475810 ± 0.000394 | — |
| 692 | sgd_m0.9_lr0.001_e1_b64_wd0 | sgd | 0.9 | 0.001 | 1 | 64 | 0 | 20 | 1.967773 | 0.475653 ± 0.000593 | — |
| 693 | sgd_m0.9_lr0.001_e1_b64_wd0.0001 | sgd | 0.9 | 0.001 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.475653 ± 0.000593 | — |
| 694 | sgd_m0.9_lr0.001_e2_b128_wd0 | sgd | 0.9 | 0.001 | 2 | 128 | 0 | 20 | 3.349609 | 0.475549 ± 0.000415 | — |
| 695 | sgd_m0.9_lr0.001_e2_b128_wd0.0001 | sgd | 0.9 | 0.001 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.475549 ± 0.000415 | — |
| 696 | sgd_m0.9_lr0.001_e4_b227_wd0 | sgd | 0.9 | 0.001 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.475444 ± 0.000239 | — |
| 697 | sgd_m0.9_lr0.001_e4_b227_wd0.0001 | sgd | 0.9 | 0.001 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.475444 ± 0.000239 | — |
| 698 | adam_m0_lr0.0001_e1_b64_wd0 | adam | 0 | 0.0001 | 1 | 64 | 0 | 20 | 1.967773 | 0.475392 ± 0.000953 | — |
| 699 | adam_m0_lr0.0001_e1_b64_wd0.0001 | adam | 0 | 0.0001 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.475392 ± 0.000953 | — |
| 700 | sgd_m0_lr3_e2_b32_wd0 | sgd | 0 | 3 | 2 | 32 | 0 | 80 | 1.875000 | 0.475366 ± 0.032155 | — |
| 701 | sgd_m0_lr0.001_e2_b64_wd0 | sgd | 0 | 0.001 | 2 | 64 | 0 | 40 | 2.504883 | 0.473981 ± 0.000683 | — |
| 702 | sgd_m0_lr0.001_e2_b64_wd0.0001 | sgd | 0 | 0.001 | 2 | 64 | 0.0001 | 40 | 2.504883 | 0.473981 ± 0.000683 | — |
| 703 | sgd_m0_lr0.001_e1_b32_wd0 | sgd | 0 | 0.001 | 1 | 32 | 0 | 40 | 1.523438 | 0.473615 ± 0.000789 | — |
| 704 | sgd_m0_lr0.001_e1_b32_wd0.0001 | sgd | 0 | 0.001 | 1 | 32 | 0.0001 | 40 | 1.523438 | 0.473615 ± 0.000789 | — |
| 705 | sgd_m0_lr0.001_e4_b128_wd0 | sgd | 0 | 0.001 | 4 | 128 | 0 | 40 | 4.560547 | 0.473406 ± 0.000326 | — |
| 706 | sgd_m0_lr0.001_e4_b128_wd0.0001 | sgd | 0 | 0.001 | 4 | 128 | 0.0001 | 40 | 4.560547 | 0.473406 ± 0.000326 | — |
| 707 | sgd_m0_lr0.001_e8_b227_wd0 | sgd | 0 | 0.001 | 8 | full site (227) | 0 | 40 | 8.710938 | 0.473354 ± 0.000313 | — |
| 708 | sgd_m0_lr0.001_e8_b227_wd0.0001 | sgd | 0 | 0.001 | 8 | full site (227) | 0.0001 | 40 | 8.710938 | 0.473354 ± 0.000313 | — |
| 709 | sgd_m0_lr3_e8_b128_wd0 | sgd | 0 | 3 | 8 | 128 | 0 | 80 | 6.308594 | 0.472884 ± 0.044158 | — |
| 710 | adam_m0_lr0.0001_e2_b227_wd0 | adam | 0 | 0.0001 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.471682 ± 0.000394 | — |
| 711 | adam_m0_lr0.0001_e2_b227_wd0.0001 | adam | 0 | 0.0001 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.471682 ± 0.000394 | — |
| 712 | sgd_m0_lr10_e1_b128_wd0 | sgd | 0 | 10 | 1 | 128 | 0 | 10 | 2.529297 | 0.471630 ± 0.062479 | — |
| 713 | adam_m0_lr0.0001_e1_b128_wd0 | adam | 0 | 0.0001 | 1 | 128 | 0 | 10 | 2.529297 | 0.470794 ± 0.000090 | — |
| 714 | adam_m0_lr0.0001_e1_b128_wd0.0001 | adam | 0 | 0.0001 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.470794 ± 0.000090 | — |
| 715 | sgd_m0.9_lr0.3_e4_b32_wd0 | sgd | 0.9 | 0.3 | 4 | 32 | 0 | 160 | 2.426758 | 0.468548 ± 0.053527 | — |
| 716 | sgd_m0_lr0.001_e1_b64_wd0 | sgd | 0 | 0.001 | 1 | 64 | 0 | 20 | 1.967773 | 0.468286 ± 0.000452 | — |
| 717 | sgd_m0_lr0.001_e1_b64_wd0.0001 | sgd | 0 | 0.001 | 1 | 64 | 0.0001 | 20 | 1.967773 | 0.468286 ± 0.000452 | — |
| 718 | adam_m0_lr0.0001_e1_b227_wd0 | adam | 0 | 0.0001 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.468077 ± 0.000181 | — |
| 719 | adam_m0_lr0.0001_e1_b227_wd0.0001 | adam | 0 | 0.0001 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.468077 ± 0.000181 | — |
| 720 | sgd_m0_lr0.001_e2_b128_wd0 | sgd | 0 | 0.001 | 2 | 128 | 0 | 20 | 3.349609 | 0.467868 ± 0.000683 | — |
| 721 | sgd_m0_lr0.001_e2_b128_wd0.0001 | sgd | 0 | 0.001 | 2 | 128 | 0.0001 | 20 | 3.349609 | 0.467868 ± 0.000683 | — |
| 722 | sgd_m0_lr0.001_e4_b227_wd0 | sgd | 0 | 0.001 | 4 | full site (227) | 0 | 20 | 6.152344 | 0.467764 ± 0.000653 | — |
| 723 | sgd_m0_lr0.001_e4_b227_wd0.0001 | sgd | 0 | 0.001 | 4 | full site (227) | 0.0001 | 20 | 6.152344 | 0.467764 ± 0.000653 | — |
| 724 | sgd_m0.9_lr0.001_e1_b128_wd0 | sgd | 0.9 | 0.001 | 1 | 128 | 0 | 10 | 2.529297 | 0.467241 ± 0.000313 | — |
| 725 | sgd_m0.9_lr0.001_e1_b128_wd0.0001 | sgd | 0.9 | 0.001 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.467241 ± 0.000313 | — |
| 726 | sgd_m0.9_lr0.001_e2_b227_wd0 | sgd | 0.9 | 0.001 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.467032 ± 0.000239 | — |
| 727 | sgd_m0.9_lr0.001_e2_b227_wd0.0001 | sgd | 0.9 | 0.001 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.467032 ± 0.000239 | — |
| 728 | sgd_m0_lr3_e2_b32_wd0.0001 | sgd | 0 | 3 | 2 | 32 | 0.0001 | 80 | 1.875000 | 0.466614 ± 0.028324 | — |
| 729 | sgd_m0_lr0.001_e1_b128_wd0 | sgd | 0 | 0.001 | 1 | 128 | 0 | 10 | 2.529297 | 0.465778 ± 0.000239 | — |
| 730 | sgd_m0_lr0.001_e1_b128_wd0.0001 | sgd | 0 | 0.001 | 1 | 128 | 0.0001 | 10 | 2.529297 | 0.465778 ± 0.000239 | — |
| 731 | sgd_m0_lr0.001_e2_b227_wd0 | sgd | 0 | 0.001 | 2 | full site (227) | 0 | 10 | 4.355469 | 0.465674 ± 0.000313 | — |
| 732 | sgd_m0_lr0.001_e2_b227_wd0.0001 | sgd | 0 | 0.001 | 2 | full site (227) | 0.0001 | 10 | 4.355469 | 0.465674 ± 0.000313 | — |
| 733 | sgd_m0.9_lr0.001_e1_b227_wd0 | sgd | 0.9 | 0.001 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.463689 ± 0.000239 | — |
| 734 | sgd_m0.9_lr0.001_e1_b227_wd0.0001 | sgd | 0.9 | 0.001 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.463689 ± 0.000239 | — |
| 735 | sgd_m0_lr0.001_e1_b227_wd0 | sgd | 0 | 0.001 | 1 | full site (227) | 0 | 5 | 3.076172 | 0.463689 ± 0.000239 | — |
| 736 | sgd_m0_lr0.001_e1_b227_wd0.0001 | sgd | 0 | 0.001 | 1 | full site (227) | 0.0001 | 5 | 3.076172 | 0.463689 ± 0.000239 | — |

PROCEED: yes
