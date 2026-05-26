# LUNG1 dsImaging Radiomics To dsFlower Evidence

This article records the real LUNG1 radiomics handoff run used as thesis
evidence. The rendered article uses the recorded output of the live May
2026 workflow so that package checks do not require active Opal servers.
The live workflow is implemented in
`inst/demos/lung1_radiomics_to_flower.R`.

The validation path is:

1.  `dsImaging` resolves the LUNG1 imaging resource and radiomics
    assets.
2.  `ds.imaging.radiomics.load_features()` loads each site’s derived
    feature table into a server-side DataSHIELD symbol.
3.  `dsFlowerClient` trains `sklearn_logreg` with FedAvg over the
    server-side tables.
4.  Only aggregate training history and model parameters are returned.

``` r

cat("Source script: inst/demos/lung1_radiomics_to_flower.R\n")
#> Source script: inst/demos/lung1_radiomics_to_flower.R
cat("Recorded run_id:", evidence$run_id, "\n")
#> Recorded run_id: 17493579680844882070
```

## Reproduction Command

``` r

Sys.setenv(
  DSFLOWER_OPAL_URLS = "https://localhost:8443,https://localhost:8444,https://localhost:8445",
  OPAL_USER = "administrator",
  OPAL_PASSWORD = "admin123",
  LUNG1_WORKDIR = "/tmp/dsimaging_lung1_full",
  LUNG1_OPAL_PROJECT = "dsdemo",
  LUNG1_OPAL_RESOURCE = "lung1_full_study",
  DSFLOWER_RADFLOWER_ROUNDS = "2",
  DSFLOWER_RADFLOWER_PRIVACY = "trusted_internal"
)
source(system.file("demos", "lung1_radiomics_to_flower.R",
                   package = "dsFlowerClient"))
```

## Server-Side Cohort

``` r

site_dim_names <- grep("opal", names(evidence$dimensions), value = TRUE)
site_dims <- do.call(rbind, lapply(site_dim_names, function(name) {
  dims <- unlist(evidence$dimensions[[name]])
  data.frame(
    site = sub("dimensions of rad in ", "", name, fixed = TRUE),
    rows = dims[[1]],
    columns = dims[[2]]
  )
}))
combined_dims <- unlist(evidence$dimensions[["dimensions of rad in combined studies"]])
knitr::kable(site_dims)
```

| site  | rows | columns |
|:------|-----:|--------:|
| opal1 |  142 |      20 |
| opal2 |  143 |      20 |
| opal3 |  137 |      20 |

``` r


data.frame(
  resource = evidence$resource,
  target = evidence$target,
  combined_rows = combined_dims[[1]],
  combined_columns = combined_dims[[2]],
  stringsAsFactors = FALSE
)
#>                  resource       target combined_rows combined_columns
#> 1 dsdemo.lung1_full_study os_2yr_alive           422               20
```

## Features

``` r

feature_roles <- c(rep("radiomic", 4), "metadata", "metadata")
feature_table <- data.frame(
  feature = unlist(evidence$features),
  role = feature_roles,
  stringsAsFactors = FALSE
)
knitr::kable(feature_table)
```

| feature                                  | role     |
|:-----------------------------------------|:---------|
| original_firstorder_Energy               | radiomic |
| original_shape_Compactness1              | radiomic |
| original_glrlm_RunLengthNonUniformity    | radiomic |
| wavelet.HLH_glrlm_RunLengthNonUniformity | radiomic |
| age                                      | metadata |
| gender_male                              | metadata |

## Federated Training

``` r

training <- data.frame(
  model = "sklearn_logreg",
  strategy = "fedavg",
  privacy_profile = evidence$privacy,
  rounds = evidence$rounds,
  max_iter = evidence$max_iter,
  stringsAsFactors = FALSE
)
knitr::kable(training)
```

| model          | strategy | privacy_profile  | rounds | max_iter |
|:---------------|:---------|:-----------------|-------:|---------:|
| sklearn_logreg | fedavg   | trusted_internal |      2 |      100 |

``` r

history <- do.call(rbind, lapply(evidence$history, function(x) {
  data.frame(
    round = x$round,
    loss = x$loss,
    n_clients = x$n_clients,
    n_failures = x$n_failures
  )
}))
knitr::kable(history, digits = 4)
```

| round |   loss | n_clients | n_failures |
|------:|-------:|----------:|-----------:|
|     1 | 0.6503 |         3 |          0 |
|     2 | 0.6474 |         3 |          0 |

``` r


cat("[dsFlower] run_id:", evidence$run_id, "\n")
#> [dsFlower] run_id: 17493579680844882070
cat("[dsFlower] model_id:", evidence$model_id, "\n")
#> [dsFlower] model_id: sklearn_logreg_FedAvg_2r_20260510_161938
cat("[dsFlower] output_dir:", evidence$flower_output_dir, "\n")
#> [dsFlower] output_dir: ./dsflower_output/sklearn_logreg_FedAvg_2r_20260510_161938
cat("[acceptance] final loss:", sprintf("%.4f", tail(history$loss, 1L)), "\n")
#> [acceptance] final loss: 0.6474
cat("[acceptance] total client failures:",
    sum(as.integer(history$n_failures)), "\n")
#> [acceptance] total client failures: 0
cat("[acceptance] PASS:",
    evidence$status == 0 && sum(as.integer(history$n_failures)) == 0, "\n")
#> [acceptance] PASS: TRUE
```

``` r

if (requireNamespace("ggplot2", quietly = TRUE)) {
  ggplot2::ggplot(history, ggplot2::aes(x = round, y = loss)) +
    ggplot2::geom_line(color = "#2C6F7D", linewidth = 0.8) +
    ggplot2::geom_point(color = "#2C6F7D", size = 2.8) +
    ggplot2::scale_x_continuous(breaks = history$round) +
    ggplot2::coord_cartesian(ylim = c(min(history$loss) - 0.01,
                                      max(history$loss) + 0.01)) +
    ggplot2::labs(x = "Federated round", y = "Loss",
                  title = "LUNG1 radiomics-derived federated loss") +
    ggplot2::theme_minimal(base_size = 11)
} else {
  plot(history$round, history$loss, type = "b",
       xlab = "Federated round", ylab = "Loss",
       main = "LUNG1 radiomics-derived federated loss")
}
```

![Line chart showing two rounds of federated logistic-regression loss on
the LUNG1 radiomics-derived
cohort.](lung1-radiomics-to-flower_files/figure-html/loss-plot-1.png)

## Evidence Scope

``` r

scope <- data.frame(
  item = c("Validated claim", "Explicit non-claim", "Global model shape"),
  value = c(
    "End-to-end functional validation that dsFlower can train on dsImaging-derived radiomic feature tables without moving images, masks or row-level feature data to the researcher.",
    "This two-round logistic-regression run is not a clinical performance claim.",
    "1 x 6 coefficients plus intercept"
  ),
  stringsAsFactors = FALSE
)
knitr::kable(scope)
```

| item | value |
|:---|:---|
| Validated claim | End-to-end functional validation that dsFlower can train on dsImaging-derived radiomic feature tables without moving images, masks or row-level feature data to the researcher. |
| Explicit non-claim | This two-round logistic-regression run is not a clinical performance claim. |
| Global model shape | 1 x 6 coefficients plus intercept |
