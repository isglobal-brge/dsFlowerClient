# Discrete-time hazard v3

## Diagnosis recorded before new training

Packaged SUPPORT2 full epsilon-8 means (seeds 1101/1102/1103):

| Contract | Federated-DP | Pooled-DP | Pooled non-private | Pooled-DP minus federated | Non-private minus pooled-DP |
|---|---:|---:|---:|---:|---:|
| Hazard v2 h06 | 0.5951121874 | 0.6209146899 | 0.6218393185 | 0.0258025026 | 0.0009246285 |
| Lognormal AFT | 0.6339464325 | 0.6386643651 | 0.6486589680 | 0.0047179325 | 0.0099946029 |
| Weibull AFT | 0.6224505587 | 0.6274792577 | 0.6386557207 | 0.0050286990 | 0.0111764630 |

This is an observed gap decomposition, not identification of separate causal effects.
Hazard's evaluated non-private ceiling is 0.0268196495 below lognormal and
0.0168164023 below Weibull; it is a ceiling for this fit/schedule, not a proven
limit of the entire discrete-hazard model family.

| Epsilon 8 mechanism | Site hazard | Pooled hazard | Site AFT | Pooled AFT |
|---|---:|---:|---:|---:|
| Patients | 2428 | 7284 | 2428 | 7284 |
| Nominal batch / expected divisor | 64 / 63 | 64 / 63 | 128 / 127 | 128 / 127 |
| Poisson probability | 1/38 | 1/114 | 1/19 | 1/57 |
| Total epochs | 40 | 40 | 20 | 20 |
| Total steps | 1520 | 4560 | 380 | 1140 |
| Noise multiplier | 1.52587890625 | 1.03027343750 | 1.56738281250 | 1.060791015625 |
| Per-coordinate gradient noise SD | 0.02422030 | 0.01635355 | 0.01234160 | 0.00835269 |

All use the unchanged audited PRV calibration with RDP fallback and independent
PRV verification. Replace-one epsilon=8, delta=1e-5 is converted to add/remove
epsilon=4, delta=1.79862099620916e-7. Independently verified hazard epsilon is
7.9855810614 (site), 7.9862946596 (pooled). The site/pooled sigma ratios are
1.48104 (hazard) and 1.47756 (AFT): this difference cannot by itself explain
the hazard-specific gap. With independent site noises, equal averaging reduces
the direct noise component by sqrt(3); nonlinear training and local drift mean
this is not an exact final-model noise decomposition.

The implementation does NOT flatten person-period rows into sampling units.
Each patient has one feature row, K logits, K event labels and K exposure masks.
Its loss is sum_j mask_ij*BCE_ij/K; Opacus clips the joint gradient once at norm 1.
For a linear head, g_ij=mask_ij*(sigmoid(z_ij)-d_ij)*[x_i,1]/K.
With 15 bounded covariates, its norm is at most 4/sqrt(K): 1.26491 for K=10;
at zero logits it is at most 2/sqrt(K)=0.63246. Clipping can attenuate a whole
patient's contribution across periods, but /K already suppresses gradients.
No observed clipping fraction was stored in the historical evidence, so heavy
clipping is not established. Reducing K increases signal per coordinate while
leaving patient accounting unchanged; /K NLL is not comparable across K.

The head has 15*K slopes plus K learned biases: 160 parameters at K=10,
versus 16 for AFT. These are independent interval heads, not a shared
proportional-hazard slope plus a baseline. Every bias is sampled, jointly clipped,
noised and averaged with the slopes. Fixed num-examples=1 yields equal-site
FedAvg; because full sites all have 2428 subjects, weights equal patient weights,
but not period-specific exposure weights. Averaging logistic parameters does not
pool likelihood sufficient statistics. Late intervals have fewer exposed patients,
and their gradients are further divided by K. Equal-width first interval spans
182.5 days, compressing early deaths; quantile grids can redistribute signal.
The registered declarative linear contract does not expose an alternative shared
baseline parameterization; changing it would require forbidden package/runner
changes and is excluded here.

Local epochs traverse patients, not up to 10 times as many rows. At h06 each
round has 4*38=152 sequential site updates versus 4*114=456 pooled updates.
FedAvg averages three endpoint changes; it does not concatenate their optimizer
steps. Thus the pooled twin has three times the sequential optimization horizon
at the same learning rate. With /K weak gradients and 10 independent heads,
under-optimization and exposure imbalance are credible mechanisms; AFT's scalar
head has denser signal and no /K attenuation. Longer local schedules, larger
learning rates, smaller K and quantile bins are the prioritized interventions.
Local drift, clipping, and noise effects are intertwined; historical records alone
do not establish their causal shares. New development-only diagnostics will test
the under-optimization explanation without using confirmation outcomes.

The existing contract already maps age [0,100], comorbidity [0,10] and
binary [0,1] covariates to [-1,1]. Raw age scale is therefore not the missing
intervention. Repeating public-bound standardisation is algebraically identical;
we retain it for every candidate rather than label duplicate fits as a new lever.
At K=10 the last logit cannot affect left-endpoint RMST ranking, although its
parameters still receive noise. AFT's single global intercept cannot change
ranking; hazard's multiple intercepts change relative interval contributions.

## Development and confirmation

Pending execution. Selection will use only mean inner-validation federated-DP
C-index over two split seeds. Confirmation is one selected configuration, three
seeds and epsilon 1/4/8. These reuse historical outer holdouts; they are not a new
independent validation cohort. No v3 confirmation score can affect selection.
