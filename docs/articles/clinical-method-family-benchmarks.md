# Clinical method-family benchmarks

This vignette records the non-binary clinical method-family validation
used for the dsFlower thesis evidence. The benchmark uses SUPPORT2, the
Study to Understand Prognoses and Preferences for Outcomes and Risks of
Treatments, from the UCI Machine Learning Repository and executes
PyTorch templates under the `clinical_default` profile, so server-side
row policies and Secure Aggregation are active. A second evidence file
repeats the DP-compatible non-survival families under
`high_sensitivity_dp`, where SecAgg+ is combined with Opacus DP-SGD.

The benchmark complements the binary clinical benchmarks: classification
and XGBoost are covered there, while this vignette covers continuous
regression, count regression, multiclass classification, multilabel
classification and time-to-event survival.

``` r

evidence_path <- system.file(
  "extdata", "dsflower_method_family_results.json",
  package = "dsFlowerClient"
)
evidence <- jsonlite::fromJSON(evidence_path)
dp_evidence_path <- system.file(
  "extdata", "dsflower_method_family_dp_results.json",
  package = "dsFlowerClient"
)
dp_evidence <- jsonlite::fromJSON(dp_evidence_path)

cat("Dataset:", evidence$dataset_label, "\n")
#> Dataset: SUPPORT2 Study
cat("Privacy profile:", evidence$privacy_profile, "\n")
#> Privacy profile: clinical_default
cat("Secure Aggregation supported:", evidence$secagg_supported, "\n")
#> Secure Aggregation supported: TRUE
cat("Rows:", evidence$n_total, "\n")
#> Rows: 3000
print(as.data.frame(evidence$site_train_n))
#>   opal1 opal2 opal3
#> 1  1000  1000  1000
```

The rows are split evenly across three DataSHIELD servers. The event
count shown below is used by the survival route disclosure guard.

``` r

site_summary <- data.frame(
  server = names(evidence$site_train_n),
  train_n = as.integer(unlist(evidence$site_train_n, use.names = FALSE)),
  event_n = as.integer(unlist(evidence$site_event_n, use.names = FALSE))
)
site_summary
#>   server train_n event_n
#> 1  opal1    1000     259
#> 2  opal2    1000     259
#> 3  opal3    1000     259
```

Each row in the result table compares the centralized PyTorch loss
against the loss obtained by evaluating the federated model artefact
saved by dsFlower after the SecAgg run completed. The route is accepted
only when all clients complete without failures and the federated loss
remains within the declared validation envelope for that method family.

``` r

results <- evidence$results
results[, c(
  "family", "method", "centralized_metric", "centralized_loss",
  "federated_loss", "delta_loss", "federated_n_failures",
  "validation_status"
)]
#>                          family                    method
#> 1 continuous outcome regression pytorch_linear_regression
#> 2              count regression           pytorch_poisson
#> 3     multiclass classification        pytorch_multiclass
#> 4     multilabel classification        pytorch_multilabel
#> 5        time-to-event survival             pytorch_coxph
#>       centralized_metric centralized_loss federated_loss   delta_loss
#> 1                    mse       0.57412314      0.5671876 -0.006935537
#> 2            poisson_nll       0.30471310      0.4711799  0.166466832
#> 3          cross_entropy       0.05172214      0.1897127  0.137990564
#> 4         multilabel_bce       0.39282042      0.4596463  0.066825897
#> 5 cox_partial_likelihood       6.26183891      6.3498344  0.087995529
#>   federated_n_failures validation_status
#> 1                    0              pass
#> 2                    0              pass
#> 3                    0              pass
#> 4                    0              pass
#> 5                    0              pass
```

``` r

dp_results <- dp_evidence$results
dp_results[, c(
  "family", "method", "centralized_metric", "centralized_loss",
  "federated_loss", "delta_loss", "federated_n_failures",
  "validation_status"
)]
#>                          family                    method centralized_metric
#> 1 continuous outcome regression pytorch_linear_regression                mse
#> 2              count regression           pytorch_poisson        poisson_nll
#> 3     multiclass classification        pytorch_multiclass      cross_entropy
#> 4     multilabel classification        pytorch_multilabel     multilabel_bce
#>   centralized_loss federated_loss delta_loss federated_n_failures
#> 1       0.57412314     0.60775775 0.03363460                    0
#> 2       0.30471310     0.52731127 0.22259817                    0
#> 3       0.02058272     0.05970215 0.03911944                    0
#> 4       0.39282042     0.49365425 0.10083383                    0
#>   validation_status
#> 1              pass
#> 2              pass
#> 3              pass
#> 4              pass
```

The DP-SGD table is defined over the SUPPORT2 routes whose losses
decompose by sample. CoxPH is validated under Secure Aggregation in the
table above; its partial-likelihood objective couples samples through
risk sets, so its validated privacy route is SecAgg rather than Opacus
DP-SGD. The four non-survival families shown here are the
DP-SGD-compatible SUPPORT2 routes; all four complete under the stricter
profile and remain within their declared utility envelopes. Multiclass
classification uses six federated rounds in this profile because the
stricter DP-SGD optimiser required a slightly longer training budget
than the non-DP family run.

``` r

compatibility <- data.frame(
  family = c(
    "Continuous regression",
    "Count regression",
    "Multiclass classification",
    "Multilabel classification",
    "Survival"
  ),
  clinical_default = c("pass", "pass", "pass", "pass", "pass"),
  high_sensitivity_dp = c("pass", "pass", "pass", "pass", "SecAgg route"),
  reason = c(
    "MSE is decomposable by row.",
    "Poisson NLL is decomposable by row.",
    "Cross entropy is decomposable by row.",
    "Binary cross entropy is decomposable by row.",
    "Cox partial likelihood is validated under SecAgg."
  ),
  stringsAsFactors = FALSE
)
knitr::kable(compatibility)
```

| family | clinical_default | high_sensitivity_dp | reason |
|:---|:---|:---|:---|
| Continuous regression | pass | pass | MSE is decomposable by row. |
| Count regression | pass | pass | Poisson NLL is decomposable by row. |
| Multiclass classification | pass | pass | Cross entropy is decomposable by row. |
| Multilabel classification | pass | pass | Binary cross entropy is decomposable by row. |
| Survival | pass | SecAgg route | Cox partial likelihood is validated under SecAgg. |

``` r

if (requireNamespace("ggplot2", quietly = TRUE)) {
  plot_data <- results
  plot_data$label <- sub("^pytorch_", "", plot_data$method)
  plot_data$abs_delta <- abs(plot_data$delta_loss)
  ggplot2::ggplot(plot_data, ggplot2::aes(
    x = stats::reorder(label, abs_delta),
    y = abs_delta
  )) +
    ggplot2::geom_col(width = 0.68, fill = "#4D6A7F") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      x = NULL,
      y = "Absolute federated-centralized loss delta",
      title = "SUPPORT2 method-family validation under clinical_default"
    ) +
    ggplot2::theme_minimal(base_size = 12)
}
```

![Point-range chart comparing centralized and federated SUPPORT2 losses
across method
families.](clinical-method-family-benchmarks_files/figure-html/loss-plot-1.png)

``` r

if (requireNamespace("ggplot2", quietly = TRUE)) {
  plot_data <- dp_results
  plot_data$label <- sub("^pytorch_", "", plot_data$method)
  plot_data$abs_delta <- abs(plot_data$delta_loss)
  ggplot2::ggplot(plot_data, ggplot2::aes(
    x = stats::reorder(label, abs_delta),
    y = abs_delta
  )) +
    ggplot2::geom_col(width = 0.68, fill = "#6F5B3E") +
    ggplot2::coord_flip() +
    ggplot2::labs(
      x = NULL,
      y = "Absolute federated-centralized loss delta",
      title = "SUPPORT2 DP-SGD subset under high_sensitivity_dp"
    ) +
    ggplot2::theme_minimal(base_size = 12)
}
```

![Bar chart comparing DP-SGD federated-centralized SUPPORT2 losses for
DP-compatible method
families.](clinical-method-family-benchmarks_files/figure-html/dp-loss-plot-1.png)

The canonical run can be regenerated from a local three-Opal
demonstration deployment with:

``` r

Sys.setenv(
  DSFLOWER_SUPPORT2_LIMIT = "3000",
  DSFLOWER_FAMILY_PRIVACY_PROFILE = "clinical_default"
)
source(system.file(
  "demos", "benchmark_method_families.R",
  package = "dsFlowerClient"
))
```

``` r

Sys.setenv(
  DSFLOWER_SUPPORT2_LIMIT = "3000",
  DSFLOWER_FAMILY_PRIVACY_PROFILE = "high_sensitivity_dp",
  DSFLOWER_FAMILY_MODELS =
    "pytorch_linear_regression,pytorch_poisson,pytorch_multiclass,pytorch_multilabel",
  DSFLOWER_FAMILY_MULTICLASS_ROUNDS = "6",
  DSFLOWER_FAMILY_DP_EPSILON = "8",
  DSFLOWER_FAMILY_DP_DELTA = "1e-5",
  DSFLOWER_FAMILY_DP_CLIP = "1",
  DSFLOWER_FAMILY_EVIDENCE_FILE =
    "inst/extdata/dsflower_method_family_dp_results.json"
)
source(system.file(
  "demos", "benchmark_method_families.R",
  package = "dsFlowerClient"
))
```
