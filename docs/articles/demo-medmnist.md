# MedMNIST Federated Benchmark

This vignette demonstrates that image-origin data can be turned into a
compact federated learning task without sending images to the client.
The demo uses BreastMNIST from MedMNIST, average-pools each 28x28 image
to a 7x7 feature map, and trains a binary logistic model through the
same DataSHIELD/Flower path used by the tabular demos.

The script bootstraps the Python client environment with `medmnist` when
needed.

``` sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export OPAL_USER="administrator"
export OPAL_PASSWORD="admin123"
Rscript inst/demos/benchmark_medmnist.R
```

## What The Script Does

This is an image-origin benchmark, but the federated model receives a
compact feature table rather than raw pixels. The script downloads
BreastMNIST through Python, balances a small subset, average-pools every
`28 x 28` image to a `7 x 7` feature map, and writes 49 numeric columns
named `px_00` to `px_48`.

``` r

dataset <- load_breastmnist()
features <- grep("^px_", names(dataset), value = TRUE)

run_dsflower_benchmark(
  data = dataset,
  features = features,
  demo_id = "medmnist_breastmnist",
  dataset_label = "MedMNIST BreastMNIST tabularized pixel benchmark",
  data_mode = "medmnist::BreastMNIST pooled 7x7 pixels",
  default_rounds = 2L
)
```

After tabularization, the DataSHIELD path is identical to the
clinical-tabular demos: each site receives a table, the client logs into
all Opals, and Flower trains a federated logistic model without sending
the pooled image-derived table back to the researcher.

``` r

conns <- DSI::datashield.login(
  logins = builder$build(),
  assign = TRUE,
  symbol = "D"
)

fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = features,
  model = "sklearn_logreg",
  model_params = list(max_iter = 100L),
  strategy = "fedavg",
  privacy = "trusted_internal",
  rounds = 2L
)
```

``` r

demo_file <- system.file("demos", "benchmark_medmnist.R", package = "dsFlowerClient")
if (!nzchar(demo_file)) demo_file <- file.path("..", "inst", "demos", "benchmark_medmnist.R")
source(demo_file)
```

Validation run on 2026-05-10:

    #>               metric    value
    #> 1               rows 510.0000
    #> 2              train 383.0000
    #> 3               test 127.0000
    #> 4              sites   3.0000
    #> 5           features  49.0000
    #> 6        central_auc   0.7654
    #> 7      federated_auc   0.7985
    #> 8   central_accuracy   0.6929
    #> 9 federated_accuracy   0.7165

This is not intended to replace a deep-learning imaging benchmark. It is
a small, reproducible proof that image-derived feature matrices can
enter the federated training path cleanly.
