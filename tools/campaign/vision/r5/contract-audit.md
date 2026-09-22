# R5 vision contract audit

This audit covers the unchanged runner source. The runtime preflight must verify
that the pod imports the same files. No held-out test records or predictions are
needed for any check below.

## Mechanism

- The registry uses a two-logit affine head, `cross_entropy`, and 512 ResNet18
  features: `R/model_registry.R:1066`, `params.py:load_user_model`, and
  `model_spec.py:output_width`. The binary head therefore contains 1026 trainable
  coordinates. The unused convenience `vision.build_head` is not this contract.
- The declarative model inserts a parameter-free finite clamp after the final
  linear layer: cross-entropy logits are clipped to [-30, 30]
  (`model_spec.py:54-81`, `482-483`). Analytical training must apply its
  derivative (zero outside that interval) before gradient clipping, and
  prediction must use the clipped logits. This matters for the large learning
  rates included in the search.
- The image transform is per-image finite min/max normalization to [0, 1], gray
  channel replication, and bilinear resize to 224 with `align_corners=False`;
  there is no ImageNet channel normalization (`vision.py:290-339`). The encoder
  uses explicit `ResNet18_Weights.IMAGENET1K_V1`, an identity final layer,
  evaluation mode and frozen parameters (`vision.py:419-443`). The resulting
  features are not constrained to [0, 1].
- Features are totalized to finite float32 values in [-1e6, 1e6] before patient
  pooling and again after pooling (`client_app.py:350-361`, `740-752`). Patient
  features are arithmetic means accumulated in float64. Labels use a
  deterministic mode with the lowest-label tie break. Patient order is first
  appearance in the staged manifest (`client_app.py:584-641`).
- For a site of `n` patients and requested batch `b`, the executed and accounted
  mechanism uses `s = ceil(n/b)` steps per epoch, Poisson inclusion probability
  `q = 1/s`, and mean-loss divisor `max(1, floor(n/s))`. It does **not** use
  `b/n` or divisor `b` at ceil boundaries (`dp_harness.py:245-284`, `337-387`).
  For the inner sites:

  | Requested batch | Steps/epoch | Actual/accounted q | Mean divisor |
  |---:|---:|---:|---:|
  | 32 | 8 | 0.125 | 28 |
  | 64 | 4 | 0.25 | 56 |
  | 128 | 2 | 0.5 | 113 |
  | 227 (full site) | 1 | 1 | 227 |

- Accounted updates per site are `s * local_epochs * 5`. The target adjacency is
  replace-one. Calibration first converts `(epsilon, delta)` to add/remove
  `(epsilon/2, delta/(1 + exp(epsilon/2)))`, then asks Opacus PRV for the exact
  update count, falling back to RDP only if PRV fails
  (`dp_harness.py:161-226`). R4's recorded mechanism already uses q=0.125,
  divisor 28 and sigma=4.98046875 for 800 updates; the shorthand `32/227` is
  not its executed sampling probability.
- Each sample's weight and bias gradients are totalized and coordinate-clamped
  to [-C, C] before Opacus computes one global L2 clip across both tensors
  (`dp_harness.py:65-91`, `400-404`). With C=1, the analytical implementation
  must retain both operations. Independent Gaussian noise with standard
  deviation `sigma*C` is added to each coordinate of the clipped gradient sum,
  then divided by the expected-batch divisor. Noise is generated separately
  for the weight tensor and bias tensor in parameter order
  (`dp_harness.py:406-431`).
- Sampling and noise use independent ChaCha20 streams derived from a 32-byte
  master (`client_app.py:487-495`). The noise generator uses Box-Muller and sums
  four draws divided by two. The sampler performs integer rejection sampling
  for exact reciprocal inclusion probability (`seeding.py:422-515`). NumPy
  PCG draws are distributional emulation only; exact numerical parity requires
  `seeding.np_rng(seeding.sub_seed(master, "sample"/"noise"))`.
- A fresh optimizer is constructed for each site each round
  (`client_app.py:474`). SGD momentum and Adam moments are consequently reset
  every round. Weight decay is applied by the optimizer to the already noised
  gradient and public weights. Parameters are saturated to [-1e6, 1e6] after
  each step (`client_app.py:467-540`).
- Every client reply carries `num-examples=1` (`client_app.py:110-113`). FedAvg
  uses that key with all sites required (`server_app.py:503-511`), so the
  federation is the arithmetic mean of three site models.

## Exposed schedule surface and initialization

SGD momentum and weight decay are exposed for vision through the common neural
registry. Learning rate is in (0, 10], momentum is in [0, 1), weight decay is in
[0, 1000], batch size is in [1, 65536], and local epochs are in [1, 1000]
(`R/model_registry.R:315-332`, `832-848`, `1066-1080`). SGD defaults to momentum
0 and no Nesterov; Adam defaults to betas (0.9, 0.999), epsilon 1e-8 and no
AMSGrad (`R/model_registry.R:392-404`, `client_app.py:391-410`). R5 keeps the
scheduler `none`, L1 penalty 0, and five rounds.

The ordinary vision ServerApp does not force seed 0. The public campaign's
existing observer seeds initial model construction with `F_VISION_INIT_SEED`
and captures the exact arrays (`benchmark_hooks/sitecustomize.py:26-47`). R4
uses seed 20260922; R3 uses seed 20260919. Parity validation loads the captured
initial arrays rather than assuming either zero initialization or seed 0.

Node DP randomness is a semantic HMAC of a custodial key, the effective
configuration/privacy mechanism, round, incoming public arrays, effective
private arrays and runtime fingerprint (`seeding.py:376-388`,
`client_app.py:754-777`). The benchmark observer captures accountant metadata
and hashes but not the master key. The benchmark seed is therefore not a
replayable seed for the historic private model.

## Validation without test access

1. Compare the analytical implementation with unchanged `client_app._dp_fit`
   using the same initial arrays, cached training-only tensors, shared fresh
   master, secure sampler/noise streams, schedule and accountant result. Check
   parameter differences and probabilities over five federated rounds, with
   tolerances for float32 gradient reduction order.
2. On R4's unchanged 681/171 training-only inner split, report the emulated
   distributions for the two historic private schedules beside their recorded
   AUCs 0.5009404388714733 and 0.4786833855799373. Distinguish statistical
   compatibility from exact historical-noise replay.
3. For the R3 default schedule, verify its captured mechanism (n=284, b=32,
   s=9, q=1/9, divisor=31, 45 updates), train-feature hashes and public initial
   arrays. Evaluate the released R3 model and emulator on training-only cached
   patient features. The published test AUC may be quoted as an existing
   aggregate, but must not be recomputed or used for selection. It cannot be
   numerically reproduced without opening the test set and must not be labeled
   as such.

All paths above are relative to `inst/flower_app/dsflower_runner/` unless they
explicitly begin with `R/` or `benchmark_hooks/` (the latter is relative to
`tools/campaign/vision/`).
