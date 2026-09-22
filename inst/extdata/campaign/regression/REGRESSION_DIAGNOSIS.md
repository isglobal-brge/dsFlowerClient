# Regression scale diagnosis — 2026-09-22

This diagnosis uses the existing scored JSON records and unchanged release source only. No model was trained, loaded for prediction, or rescored, and no prepared held-out CSV was opened for the diagnosis. The two historical cells and their corrected central reference remain immutable.

## Saved numerical evidence

| Cell | epsilon | Central RMSE | DP RMSE | Trivial RMSE | DP R2 | Paired gap mean ± sample SD | Excess-error scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parkinson | 1 | 11.309447 | 29.503164 | 10.771196 | -6.879554 | 18.193717 ± 3.082866 | 27.498066 |
| Parkinson | 4 | 11.309447 | 29.456259 | 10.771196 | -6.883872 | 18.146812 ± 4.167444 | 27.467350 |
| Parkinson | 8 | 11.309447 | 29.549561 | 10.771196 | -6.919012 | 18.240114 ± 3.516164 | 27.545764 |
| CDC BMI raw | 1 | 6.195724 | 10.371563 | 6.580037 | -1.484565 | 4.175839 ± 0.070229 | 8.017388 |
| CDC BMI raw | 4 | 6.195724 | 10.398831 | 6.580037 | -1.497708 | 4.203107 ± 0.110651 | 8.052925 |
| CDC BMI raw | 8 | 6.195724 | 10.414030 | 6.580037 | -1.504846 | 4.218306 ± 0.120771 | 8.072923 |

The epsilon range of mean DP RMSE is 0.093302 (0.316%) for Parkinson and 0.042467 (0.409%) for BMI. There is no meaningful monotonic utility improvement across this grid; “epsilon-independent” describes this approximate empirical plateau, not mathematical equality or a causal proof that noise has no effect. Parkinson's central values use `parkinsons_corrected_central_twin.json`, not the superseded central fields of the original pilot files.

The final column is `sqrt(mean_seed(DP_RMSE^2 - trivial_RMSE^2))`: approximately 27.5 UPDRS or 8.0 BMI units of excess error. Thus the proposed quadrature signature is numerically supported: `sqrt(6.6^2+8^2)=10.371114`; `sqrt(10.8^2+27.5^2)=29.544712`. Saved training means are 28.902606/29.444770/29.616542 UPDRS and 28.408389/28.387944/28.408750 BMI.

An exact signed-bias decomposition is not identifiable from these saved records: prediction/residual means, residual variance and prediction vectors were not saved. For training mean m and d=prediction-m, the exact identity is `MSE_DP=MSE_trivial+E[d^2]-2E[(y-m)d]`. The excess-error scale must not be relabelled as measured bias. Likewise, the raw BMI plateau does not establish predictions literally near zero: an 8-unit offset below mean 28 would be near 20. Parkinson's very small update budget is consistent with predictions near initialization. The new cell records residual mean and variance at its single scoring pass so `MSE=mean_residual^2+residual_variance` can be checked directly.

## Bounds and unchanged runner

Both previous cells passed target bounds, not merely documented them: every site's retained staged configuration contains `target-bounds`, [0,176] for Parkinson and [12,98] for BMI. The original harness passes these to `ds.flower.fit`. Every site also received the declared feature bounds.

The released neural runner consumes target bounds for clipping and missing-value replacement, but does **not** affine-rescale the continuous target. Features are clipped and transformed as `(x-(lower+upper)/2)/((upper-lower)/2)` into [-1,1]. Its MSE regression fits the clipped target in the units supplied. The local predictor repeats the feature transform and returns the linear response directly; it performs no inverse target transformation. Thus target bounds are not ignored entirely, but they provide no target normalization. No package code is changed; the new target transformation and inverse prediction transformation belong to the campaign harness.

The registry and node-authored staged configurations agree: learning rate 0.01, one local epoch, batch size 32, SGD, no scheduler, no L1/L2 penalty, one linear layer with an intercept. Initialization is standard PyTorch `nn.Linear`, including a random bias, not an explicitly zeroed intercept. The public transformation places the **coordinate origin** at BMI 55; it does not change initialization or guarantee an initial prediction of exactly 55.

## Clipped motion and schedule decision, made before execution

For MSE with augmented features z=(x,1), a single-example gradient is `2(prediction-y)z`. Unit norm clipping caps its intercept coordinate by 1 (and in the saturated regime by `1/||z||`). With plain SGD the signal contribution to one bias update is bounded by `0.01 * actual_batch_size / expected_batch_size`. Poisson batching uses expected batch size, so 0.01 is the nominal/expected per-step bound, not an absolute bound for every realized batch. Gaussian DP noise is unbounded; all motion-budget statements here refer to the clipped data signal, not the noise-inclusive update.

For CDC BMI each site has 12,000 rows: ceil(12000/32)=375 steps per local epoch, or 1,875 steps over five rounds. The nominal cumulative signal budget is at most 18.75 intercept units; FedAvg averages sites rather than adding their budgets. That is smaller than the raw target mean about 28.4, and feature coordinates share the clipping norm. This supports an optimization-scale limit, but is not a proof about the full model's achievable prediction mean, which also depends on feature weights.

After the public target transform `y'=(clip(BMI,12,98)-55)/43`, all training targets lie in [-1,1] and the saved raw training means correspond to approximately -0.619. The 18.75 motion budget easily exceeds this scale; there is no bounded-motion justification for extending the schedule. Keep all registry defaults, including **one local epoch**. Reachability alone does not establish convergence, and no training pilot or test-based schedule selection is performed.

Parkinson patient-mode training averages transformed features and outcomes within subjects, leaving 12/11/11 units per site, one batch per round, and five total steps. Its nominal signal budget is only 0.05 units against a target mean around 29. It remains a boundary measurement for this protocol with a dozen privacy units per site, outside the observed useful regime. The corrected central reference fits all training recordings, whereas DP fits subject means; the gap also includes that fitting-unit difference.

## Third cell and interpretation

The third and final declared cell reuses the 45,000-person CDC cohort and exact prepared splits/seeds. Transform only training targets in the client harness using public constants 55 and 43; retain the runner's public feature transform and pass target bounds [-1,1]. Map predictions back with `55+43*prediction` before scoring raw held-out BMI. No empirical mean or scale enters the model transform. Central OLS uses the same rows and public-unit variables; trivial predicts the raw training mean.

This is the track's public-unit **privacy-cost measurement**, operationally the federated-DP minus central-OLS gap after removing the raw-unit mismatch. It is not a noise-only causal estimate: finite optimization, gradient clipping and federation may still contribute. The previous raw-unit gap is optimization-limited and must not be presented as a privacy cost. No success threshold changes settings: RMSE below trivial and R2>0 at epsilon 8 are annotations only.

Provenance: [UCI CDC Diabetes Health Indicators, ID 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), DOI [10.24432/C53919](https://doi.org/10.24432/C53919), thesis citation key `uci_cdc_diabetes_health_indicators`. The [CDC 2015 BRFSS codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/CODEBOOK15_LLCP.pdf), printed page 115, documents BMI coding; it does not establish [12,98] as universal physical endpoints. Those are the already declared public admissible bounds reused here, not cohort-estimated extrema. Original source/cohort/split checksums are in `cdcbmi_provenance.json` and remain unchanged.

## Audited source locations

Paths are relative to the unchanged release source trees on the pod (the mirrored client runner has identical source):

- `dsFlower/inst/flower_app/dsflower_runner/task.py:95-127`: target imputation and clipping, no rescaling.
- `dsFlower/inst/flower_app/dsflower_runner/client_app.py:658-678` and `:744`: bounded affine feature transform; `:388`: continuous MSE target tensor; `:391-400`: SGD; `:504-525`: local update loop.
- `dsFlower/inst/flower_app/dsflower_runner/dp_harness.py:264-269` and `:387`: step/sample geometry and expected batch divisor.
- `dsFlower/inst/flower_app/dsflower_runner/model_spec.py:244-256`, `server_app.py:123-128` and `:341`: ordinary linear initialization, distributed as server weights.
- `dsFlowerClient/inst/python/predict_helper.py:82-86` and `:306-332`: raw regression output and public feature transform, no target inversion.
- `dsFlowerClient/R/model_registry.R:843-887`: default SGD schedule and MSE linear contract.
- `dsFlowerClient/tools/campaign/regression/run_cell.R:94-95`, `run_cdcbmi.R:104-108`, `campaign_lib.R:267-268`: original target/feature bounds passed to fit.

Both installed packages remain version 0.5.0. Canonical runner SHA-256: `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
