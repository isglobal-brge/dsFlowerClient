# Regression R4 diagnosis — 2026-09-22

The default finite, clipped optimization schedule explains the above-trivial regression error. The failure is reproduced without Gaussian noise. Five rounds of one local epoch leave substantial initialization dependence; increasing local optimization within the unchanged contract removes most of the gap. No released-code target-shape, gradient-divisor, learning-rate-scaling or prediction-transform defect was found. Eight schedules are ready for a later real-contract confirmation. No new evaluation cell was declared or run, and no sealed test CSV was opened or created.

## Inputs and scope

The original CDC cohort seed is **20260819**; the original split seeds are **20260820, 20260821, 20260822**. The cohort has 45,000 respondents, with 36,000 training rows in three original sites of 12,000. The original R functions in `tools/campaign/campaign_lib.R` reconstructed the same cohort and split identities. All **12** original training/site CSV checksums and the original prepared-cohort checksum passed. Only source routing and index identities involved the held-out partition; no held-out outcomes were summarized or scored.

The UCI source is [CDC Diabetes Health Indicators, data.csv](https://archive.ics.uci.edu/static/public/891/data.csv), DOI [10.24432/C53919](https://doi.org/10.24432/C53919). Its SHA-256 is `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`. The previously downloaded UCI file was reused after verifying that checksum; a slow fresh pod download was abandoned before preparation. The original prepared logistic cohort SHA-256 is `5f521899386021fd6670d166f4f5ef9a4dcaa80e11d8df10942e60bd18724d3b`.

For each outer split, take a fixed 2,400-row inner holdout from each site's training rows using NumPy `default_rng(20260922 + site).permutation(12000)`, sites numbered 1–3. Inner training therefore has **28,800 rows, 9,600/site**, and inner validation has **7,200 rows**. Training/validation position hashes are retained. Five diagnostic initialization seeds, **101, 202, 303, 404, 505**, were fixed before running the emulation; sampling is paired between noise/no-noise comparisons, with a separate noise stream. These public diagnostic random streams do not constitute private releases. The three outer training partitions overlap, and the same initializations are paired across splits: reported SDs are descriptive, not independent-replicate uncertainty estimates.

The inner geometry executes **300 steps/epoch and 1,500 steps/site** at the default schedule, with q=1/300 and expected batch size 32. Separately, the original **12,000/site, q=1/375, 1,875-step** geometry is emulated and scored on those training rows only. This distinction preserves a true inner holdout while checking the exact original horizon. Features use the released public bounds transform; training targets use `(clip(BMI,12,98)-55)/43`. All reconstructed training targets are already within [12,98]. Emulator metrics use the float32 public target and BMI factor 43; the real-run scores use raw BMI. Their baseline RMSE differences are below 0.000001 BMI.

## H1 — finite schedule and noise

Across the 15 split/initialization combinations, the original-geometry **no-noise training RMSE is 6.869296 ± 0.391565**, range **6.399756–7.572126**. The same original training partitions have mean OLS RMSE **6.161522** and trivial RMSE **6.591979**. Adding epsilon-8 noise gives **6.859617 ± 0.399392**. The paired mean change is **−0.009679**, with individual changes **−0.032384 to +0.013594**. This reproduces the scale and dispersion of the historical epsilon-8 test result, 6.866393 ± 0.423674, without needing Gaussian noise. It is not a reconstruction of those exact historical models: their weights and initialization seeds were not archived.

Inner-validation results, averaging five initializations per split:

| Original split seed | OLS | Trivial inner-training mean | Default, no noise | Default, epsilon-8 noise |
|---|---:|---:|---:|---:|
| 20260820 | 6.061388 | 6.543757 | 7.013355 ± 0.457735 | 7.008841 ± 0.463515 |
| 20260821 | 5.980302 | 6.438805 | 6.907443 ± 0.458352 | 6.896257 ± 0.462252 |
| 20260822 | 6.212247 | 6.642629 | 7.110908 ± 0.448419 | 7.101715 ± 0.453330 |
| Mean across splits | **6.084646** | **6.541731** | **7.010569** | **7.002271** |

Across all 15 inner runs, the no-noise SD is **0.429809**, versus **0.434411** with noise. The paired mean noise effect is **−0.008297 BMI**, range **−0.042099 to +0.016099**. The RMS difference between paired final weight coordinates is **0.002009 public units** (original geometry: **0.001943**). The small negative average is a realized paired result, not a claim that noise generally improves utility.

The runner converts replace-one `(epsilon,delta)` to add/remove `(epsilon/2, delta/(1+exp(epsilon/2)))` before calling Opacus. Explicit **PRV** calibration succeeds; there is no RDP fallback in these results. At the original q=1/375, T=1,875, delta=1e-6:

| Replace-one epsilon | Gaussian multiplier | SD per coordinate of averaged gradient | SD of one SGD update at LR .01 |
|---|---:|---:|---:|
| 1 | 1.259765625 | 0.039367676 | 0.000393677 |
| 4 | 0.772705078125 | 0.024147034 | 0.000241470 |
| 8 | **0.660400390625** | **0.020637512** | **0.000206375** |

At epsilon 8, the PRV add/remove epsilon upper bound is 3.997414, giving a replace-one epsilon upper bound 7.994829 at delta no greater than 1e-6. Ignoring gradient contraction, `LR*sigma/32*sqrt(1875)` is **0.008936 public units per coordinate per site**, or **0.384261 BMI**. Averaging three independent site noise paths gives **0.005159 public units**, or **0.221853 BMI**. These are free-random-walk reference scales, not predicted RMSE changes; the actual paired emulation is the relevant noise-effect measurement. Inner epsilon-8 sigma is 0.6781005859375 because its population and horizon differ.

The mechanism behind the optimization failure is visible in controls on split 20260820. These are explanatory non-private counterfactuals, not changes proposed for the privacy contract:

| Control | Inner-validation RMSE, mean ± SD over five initializations/sampling streams |
|---|---:|
| Released default semantics | 7.013355 ± 0.457735 |
| Remove coordinate clamp, retain norm-1 clipping | 7.007230 ± 0.455224 |
| Remove both clipping stages, retain the finite schedule | 6.320858 ± 0.109536 |
| Zero initialization, retain released clipping and default schedule | 6.762672 ± 0.024068 |
| Retain random initialization and both clips, increase to 20 local epochs | **6.143807 ± 0.010981** |

The coordinate clamp is real but has little effect on this observed failure. Global norm clipping materially slows the default optimization, and random initialization explains much of its dispersion. For initialization 101, default inner RMSE after rounds 1–5 is **10.8003, 9.0266, 8.3624, 8.0067, 7.7515**: training is still improving. At round 5 in the first split, **34.0–38.3%** of example gradients exceed norm 1 after coordinate saturation, while **0.9–2.4%** have at least one raw coordinate above magnitude 1. Clipping also changes the estimating equation, so a stationary point can differ from OLS. The long control remains 0.082419 BMI above that split's OLS; this residual may include clipping bias and remaining finite optimization, since convergence was not proved. The default schedule's gap is much larger.

The historical epsilon-1/4/8 SDs, **0.058812 / 0.501409 / 0.423674**, came from three separately initialized trainings per budget. They are not paired noise comparisons. Their non-monotonicity and differing dispersion are consistent with uncontrolled initialization plus a short clipped schedule; these few runs do not identify an epsilon-dependent variance law.

## H2 — released step semantics and initialization

Audited runner files are byte-identical to the released v0.5.0 node code. Paths below are relative to either package's mirrored `inst/flower_app/dsflower_runner/` unless otherwise stated.

- `client_app.py:658–678` applies `(clip(x,lo,hi)-centre)/half_range`; `:744` invokes it. `task.py:105–127` clips the target without affine rescaling. `_prep_target` at `client_app.py:367–388` produces [N,1], matching the linear output. No broadcasting defect exists.
- For augmented feature z=[x,1], the individual MSE gradient is **g=2*(prediction-target)*z**. `dp_harness.py:65–88` clamps every per-example gradient coordinate to **[−1,1]** for C=1. `:400–404` runs this before Opacus globally clips the combined weight-and-bias gradient to norm 1, using factor `min(1,1/(norm+1e-6))`.
- Installed Opacus 1.6.0 `grad_sample/grad_sample_module.py:402–404` multiplies the mean-loss backprop by the **actual** batch length, recovering individual-example gradients. `optimizers/optimizer.py:468` sums clipped examples; `:554–559` orders clipping, noise and scaling; `:502–504` divides once by **expected_batch_size × accumulated_iterations**. Here accumulated_iterations=1. Consequently the SGD step is **−LR × (sum of clipped individual gradients + Gaussian noise)/32**. There is no second actual-batch divisor or hidden LR factor. This matches the [Opacus optimizer source documentation](https://opacus.ai/api/_modules/opacus/optimizers/optimizer.html).
- `dp_harness.py:263–269` pins steps=ceil(n/B), q=1/steps and expected divisor=max(1,floor(n/steps)); `:92–114` performs independent Poisson membership draws. `:387` explicitly sets the divisor. The `poisson_sampling=False` argument only prevents Opacus replacing the already-Poisson trusted sampler.
- **Numerical check:** analytic emulator updates match actual Opacus updates with **maximum parameter difference 0.0** for actual batch lengths **7, 32 and 53**, each with expected divisor 32, nonzero residuals and both clipping stages. This independently excludes the proposed sum/mean or target-shape error.
- `client_app.py:391–410` constructs stock optimizers with the pinned LR. `:474` constructs a fresh optimizer each node-round, so momentum/adaptive moments reset between rounds. Schedulers use global epoch at `:413–441`. `client_app.py:109–123` reports fixed aggregation weight 1; `server_app.py:505–511` uses it, giving equal-weight FedAvg over three sites.
- **Initialization is random and uncontrolled by the recorded split seeds.** `server_app.py:86–128,332–342` constructs and distributes the model without seeding the regression path; the nearby fixed seed is specific to survival. `model_spec.py:256` uses stock `nn.Linear(20,1)`, whose weights and bias are uniform on [−1/sqrt(20),1/sqrt(20)]. The node seed at `client_app.py:316–321` does not determine the global start: `:323` overwrites every parameter with the server arrays. The R split seeds select data partitions; they do not seed the server PyTorch process.

For residual magnitude O(1), coordinate gradients can exceed 1 and are saturated; the global norm clip further reduces the signal. Even around the fitted BMI distribution, global clipping is frequent. With augmented feature norm roughly four, norm clipping begins near public residual magnitude 1/8, or about 5.4 BMI. The initial motion-budget argument therefore established reachability, not convergence of the useful feature directions. The observed initialization sensitivity and long-schedule control establish the missing convergence problem numerically.

No package code was changed. Random initialization is a reproducibility limitation here, not evidence of a sum/mean implementation defect. Neither initialization nor clipping needs to be changed to obtain the recommended schedules' improvement.

## H3 — one real federated release and prediction check

The original cell records retain model SHA-256 values but no weights; no archived regression `.pt` or equivalent weights were found in committed or laptop evidence. Therefore exactly **one** new default-schedule federated-DP training was run on the first inner split: epsilon=8, delta=1e-6, row privacy, C=1, three sites of 9,600, five rounds, SGD .01, batch 32, one local epoch. The node-authored staged configurations confirm those settings and public bounds. It completed in **101.659 seconds**, with three clients, five rounds, zero failures and successful federation-process cleanup.

| Scoring rows | Released model RMSE | Trivial training mean | OLS |
|---|---:|---:|---:|
| Inner training, 28,800 | **6.915142** | 6.619928 | 6.197861 |
| Inner validation, 7,200 | **6.854918** | 6.543758 | 6.061388 |

Training R² is −0.091178; inner-validation R² is −0.097518. Mean residuals are −0.041735 and −0.109220 BMI. Thus the model is already worse than trivial on its own training rows; a prediction-time location offset is not the explanation.

The original model artifact is retained with SHA-256 **`33b719bdde582f49c904694e49f13acd49bef2c95bccf03eb6a3618c82933e06`**. Independently loading only its linear weight and bias, applying the public bounds transform and `55+43*prediction`, gives RMSE **6.915142 / 6.854918**. The maximum discrepancy from `ds.flower.predict` is **2.3842e-7 public units**, or **1.0252e-5 BMI**, on both partitions, consistent with float32 arithmetic. The source agrees: `R/predict.R:536–557` restores feature order/bounds, `inst/python/predict_helper.py:306–334` applies the same transform, and its MSE response at `:86–89` is the raw linear output. The campaign applies the target inverse once.

## Recommended parameter surface and candidates

The accepted surface (`R/model_registry.R:315–321`; node `task.py:783–838`) is LR **(0,10]**, integer local epochs **[1,1000]**, integer batch size **[1,65536]**, and weight decay **[0,1000]**. Optimizers are **SGD, Adam, AdamW, RMSprop**. Momentum, Adam betas/epsilon/AMSGrad and none/step/exponential/cosine schedulers are also supported. Weight decay applies to bias as well as weights. The linear contract retains no hidden layers.

A useful focused search is **SGD LR .01–.10 or Adam/AdamW LR .001–.003; 1, 3 or 5 local epochs; batch 32 or 64; decay 0 or .01**. SGD and Adam cover the principal optimization change; RMSprop is accepted but not needed for this shortlist. Keep scheduler none, L1=0 and other optimizer defaults. The 20-epoch control explains convergence but adds no observed utility advantage over the cheaper candidates. **Five rounds, equal-weight FedAvg, row privacy, replace-one adjacency, delta=1e-6, C=1, all public bounds and target transform remain fixed.** Every changed batch/epoch schedule must recalibrate noise over its complete horizon at the requested epsilon; a baseline multiplier must never be reused.

Ranked by mean non-private inner-validation RMSE across all three splits and five paired initializations, with the released clipping semantics throughout:

| Rank | Optimizer | LR | Local epochs | Batch | Decay | Inner RMSE ± descriptive SD | Mean by split 20260820 / 21 / 22 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | SGD | .03 | 5 | 64 | 0 | **6.143758 ± .102092** | 6.145370 / 6.022938 / 6.262966 |
| 2 | SGD | .01 | 5 | 32 | 0 | **6.147912 ± .095318** | 6.141905 / 6.039123 / 6.262709 |
| 3 | SGD | .03 | 3 | 32 | 0 | **6.159746 ± .089510** | 6.142609 / 6.069868 / 6.266760 |
| 4 | AdamW | .001 | 3 | 32 | .01 | **6.161926 ± .091808** | 6.163677 / 6.058429 / 6.263672 |
| 5 | Adam | .001 | 3 | 32 | 0 | **6.162560 ± .091824** | 6.163910 / 6.059321 / 6.264449 |
| 6 | SGD | .05 | 3 | 32 | 0 | **6.174942 ± .095185** | 6.147479 / 6.091835 / 6.285512 |
| 7 | Adam | .003 | 1 | 32 | 0 | **6.187221 ± .117038** | 6.188442 / 6.067843 / 6.305379 |
| 8 | SGD | .10 | 1 | 32 | 0 | **6.201167 ± .120243** | 6.188479 / 6.111779 / 6.303242 |

Start confirmation with ranks **1 and 2**; their 0.004155 BMI difference is too small to establish a meaningful ordering before real-DP confirmation. Rank 2 changes only local epochs and is a particularly clean comparison. These are candidate recommendations, not a newly declared cell, and no candidate was run through real federated DP. At the original n=12,000, batch 64 implies **188 steps/epoch, q=1/188 and expected divisor 63**, whereas the inner n=9,600 geometry divides exactly by 64. Implementation must use the runner's actual geometry, including that ceil/floor boundary.

Accountant-only checks at n=12,000/site give epsilon-8 multipliers **0.660400 / 0.707703 / 0.741577** for batch 32 with 1/3/5 local epochs (1,875/5,625/9,375 steps), and **0.847168** for batch 64 with five local epochs (4,700 steps). These calibrations are retained for all eight candidates; they are not utility confirmations.

## Pod state, tooling and retained evidence

Only `pod-flower-regression` was accessed. It remains running, with **32 CPUs, 64,000,000,000-byte cgroup memory limit, no NVIDIA device**. The interrupted state already had R 4.6.1, uv and R dependencies but no Python 3.11, extracted package sources or venvs; the wheel transfer was incomplete. Provisioning was continued using the track's dependency/lock tooling and existing verified source archives. All 52 archive chunks and individual wheel checksums passed. The original provisioning start was **06:08:09 UTC**, and completion was **06:25:20 UTC** (17m11 total, including prior work; approximately 11 minutes during this diagnosis).

- **dsFlower v0.5.0**, commit `408f08c539329e2711260050ab40a6567aa4d89e`; **dsFlowerClient v0.5.0**, commit `50dda000a32ffcbdd039c2b74c909df451392bfb`. These are the explicitly allowed 0.5.0 tags and match the measured track's releases.
- Both installed canonical runner hashes: **`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`**. No package code changes.
- **R 4.6.1; Python 3.11.15; PyTorch 2.14.0+cpu; Opacus 1.6.0; Flower 1.31.0; NumPy 2.4.6**. R DSI 1.8.0, DSLite 1.4.1, arrow 25.0.1; Python pyarrow 23.0.1. Arrow roundtrip and all venv health checks passed.
- R library `/workspace/cells/Rlib`; node venvs `/workspace/cells/venvs/{native-tree,pytorch}`; client venv `/workspace/cells/client/venv`. Complete package versions, archive/lock hashes and before/after state are in `tools/campaign/regression/r4/provisioning.json`.
- Training inputs `/workspace/cells/data_cache/cdcbmi_seed{20260820,20260821,20260822}`. No `test.csv` exists under that cache. Branch diagnosis tooling is staged at `/workspace/cells/r4-client`; installed tag source trees remain `/workspace/cells/{dsFlower,dsFlowerClient}`. Results and the retained real model are under `/workspace/cells/r4/`.

The emulator is **`tools/campaign/regression/r4/emulate.py`** on branch **`evidence/representative-cells`** of dsFlowerClient. `git pull --rebase origin evidence/representative-cells` was performed before changes. The same directory contains training preparation, inner staging, the guarded one-fit R driver, independent prediction verification, accounting and numeric summaries. `results/` retains baseline/candidate/control replicates, the real fit's public diagnostics and released `model.pt`. A README gives reproduction commands. No raw training CSV, prediction-row CSV, node credential or private R fit object is committed. The original scored cell records remain unchanged.

The diagnosis, emulator, candidate results and provisioning evidence are published together with an ordinary diagnosis commit; the delivery gives its commit SHA. Work stops after this diagnosis and recommendation. There is no code defect or provisioning blocker requiring an implementation repair before schedule confirmation.

PROCEED: yes
