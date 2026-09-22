# Sequence diagnostic source audit

These references are to the unchanged `dsFlowerClient` runner used by R4. This is a training-only mechanism diagnosis; it does not define an evaluation cell. Runtime fidelity and accounting checks are recorded separately by `verify.py` and the emulator.

## Model and initialisation

- `R/model_registry.R:814–819` constructs reshape `[128,9]`, recurrent hidden layer, then a six-logit linear head. The raw window is flattened in C order, so reshape preserves token-major ordering.
- `inst/flower_app/dsflower_runner/model_spec.py:375–411` constructs Opacus `DPLSTM(..., batch_first=True)` and returns `out[:, -1, :]`. This is the released architecture; replacing it with an unverified `torch.nn.LSTM` would not establish fidelity.
- `server_app.py:86–118,332–342` builds the initial model using the same declarative builder. `tools/campaign/sequence/benchmark_hooks/sitecustomize.py:30–41` places construction inside a CPU RNG fork with `torch.manual_seed(20260922)` and records every initial tensor. It changes public initialisation only.
- `params.py:12–27,30–68` serialises the actual trainable tensors in stock `named_parameters` order and requires exact shapes and dtypes on loading. A diagnostic must compare its initial tensor hashes against the real contract capture.
- `dp_harness.py:373–384` calls `PrivacyEngine.make_private` with the normal default gradient-sampling mode; it does not request `force_functorch`. Runtime inspection distinguishes registered DPLSTM/RNNLinear gradient samplers from fallback functorch handling.

## Bounds and prediction

- `client_app.py:658–676` performs the public clip-and-affine transform. With symmetric channel bounds it is `clip(x,-b,b)/b`, where `b=[1,1,1,1,1,1,2,2,2]` repeated for 128 timesteps.
- `client_app.py:743–745` applies those bounds once during training, then applies a finiteness gate (no second public scaling). Row privacy has no subject pooling; pooling only occurs when group IDs are supplied at `750–752`.
- `inst/python/predict_helper.py:306–332` performs the same public transform; `125–144` reconstructs the declarative model, strictly loads weights and predicts float32 rows. Cross-entropy prediction uses softmax in output-column order (`95–98`). This predicts each original window independently, without subject pooling.
- `R/predict.R:171–196` associates zero-based class index `j` with target level `j+1` in the R vector and labels probability columns with those levels. R4 pins `target_levels=as.character(0:5)`, matching archive activities 1–6 shifted to 0–5 and scoring probability columns 0–5.

## Local training and aggregation

- `client_app.py:391–409,474` builds a new optimizer per node-round. Adam uses the pinned betas, epsilon and weight decay. With scheduler `none`, learning rate remains fixed (`413–425`). Optimizer moments do not cross round boundaries.
- `client_app.py:486–495` uses a DataLoader only to declare batch/horizon geometry; the harness replaces it with secure Poisson sampling. Loss is mean-reduced (`500–516`).
- `dp_harness.py:93–113` independently includes each row at rate `q=1/ceil(N/B)` on each logical step. At `264–284`, steps per epoch are `ceil(N/B)`, total steps are `ceil(N/B)*local_epochs*rounds`, and the expected batch divisor is `max(1,floor(N/ceil(N/B)))`. `387` explicitly installs that divisor on the DP optimizer.
- Therefore the original full sites `2553/2397/2402` with batch 256 all have **200** steps over 4 epochs × 5 rounds, not approximately 190. Inner-split geometry is recomputed from each retained site population and is recorded by the node.
- `client_app.py:109–123` gives every successful node the constant aggregation weight `num-examples=1`. `server_app.py:503–511` uses that key for FedAvg, so the three sites have equal weights independent of row counts.
- `client_app.py:765–794` always calibrates and executes the neural DP path. The diagnostic non-private/clipped-noiseless twins therefore require a separate emulator; no released package files are changed.

## Clipping, noise and accountant

- `dp_harness.py:65–88` maps non-finite per-example gradients to finite values and clamps every gradient coordinate to `[-C,C]`. The wrapper at `402–404` invokes this before Opacus's global per-example L2 clip. For this contract both coordinate bound and norm bound are 1.
- `dp_harness.py:412–426` adds Gaussian noise to the sum of clipped gradients, with coordinate standard deviation `noise_multiplier*C`, before the expected-batch normalization. The secure noise and sampling RNGs come from separate node-owned release streams (`client_app.py:494–495`). Public diagnostic seeding does not replace them.
- `dp_harness.py:161–183` converts the replace-one target into an add/remove target: `epsilon0=epsilon/2`, `delta0=delta/(1+exp(epsilon0))`. At epsilon 8 and delta 1e-6 this means epsilon0 4 and delta0 approximately 1.7986e-8.
- `dp_harness.py:186–216` calibrates to all five rounds' logical steps with PRV (RDP fallback only if needed); `245–290` describes the exact effective mechanism. At fixed `q`, more local epochs increase total steps and the required multiplier. Changing batch size changes both `q` and the divisor, so sigma alone does not rank mean-gradient noise sensitivity.
- The existing public node observer captures each round's actual accountant history, its multiplier, observed step count, geometry, input tensor hashes, privacy policy and pins (`benchmark_hooks/sequence_public_observer.py:123–160`). R4 retains this observer unchanged behind the mandatory runner-integrity finder.

## Contract search surface

- `R/model_registry.R:831–856,1009–1022` exposes learning rate, batch size, local epochs, optimizer, optimizer settings, scheduler, and LSTM hidden size. Hidden size is a real contract parameter.
- Allowed optimizers are SGD, Adam, AdamW and RMSprop (`371–386`). Learning rate lies in `(0,10]`, batch size in `[1,65536]`, local epochs in `[1,1000]` (`315–317`); these broad validation limits are not recommended search ranges.
- Keep rounds 5, clipping norm 1, delta 1e-6, epsilon 8, six classes, `[128,9]` layout and public channel bounds fixed. Prefer a small empirical range inside the contract limits. Adam and AdamW are identical when weight decay is zero, so both need not be separate candidates in that case.

## Interpretation limits

The additive diagnostic contrasts are conditional on the inner subject split, seed, chosen order of ablations and exact batching. They do not uniquely allocate the previously observed three-seed held-out gap of -0.168. To isolate clipping, pair the unclipped and clipped federation with identical Poisson masks and normalization; otherwise their difference also includes a sampling change. A finite central twin must state which inner-site step count it matches and preserve round-boundary optimizer resets. Independent cryptographic DP randomness means a one-seed real-contract versus noiseless contrast is an estimate with training variability, not a paired-noise exact causal effect.
