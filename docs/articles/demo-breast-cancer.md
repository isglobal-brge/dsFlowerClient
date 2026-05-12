# Breast Cancer Federated Benchmark

This vignette demonstrates a federated logistic-regression benchmark on
the Wisconsin Breast Cancer dataset from `mlbench`. The script creates a
local central baseline, partitions the training rows across the
configured Opal servers, runs Flower through DataSHIELD, and writes a
JSON/RDS audit record.

The default Opal configuration is:

``` sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export OPAL_USER="administrator"
export OPAL_PASSWORD="admin123"
Rscript inst/demos/benchmark_breast_cancer.R
```

## What The Script Does

The benchmark script is intentionally transparent. It loads the public
dataset, creates a binary outcome, partitions the training rows across
the configured sites, uploads one Opal table per site, logs into
DataSHIELD, and launches a federated logistic regression with
[`ds.flower.fit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.fit.md).

``` r

data("BreastCancer", package = "mlbench")
raw <- BreastCancer[stats::complete.cases(BreastCancer), , drop = FALSE]

feature_cols <- setdiff(names(raw), c("Id", "Class"))
features <- paste0("bc_", seq_along(feature_cols))

dataset <- data.frame(
  id = paste0("bcw_", seq_len(nrow(raw))),
  setNames(lapply(raw[feature_cols], function(x) as.numeric(as.character(x))), features),
  outcome = as.integer(raw$Class == "malignant"),
  check.names = FALSE
)

run_dsflower_benchmark(
  data = dataset,
  features = features,
  demo_id = "breast_cancer_wisconsin",
  dataset_label = "Breast Cancer Wisconsin diagnostic benchmark",
  data_mode = "mlbench::BreastCancer",
  default_rounds = 2L
)
```

The helper expands to the same explicit DataSHIELD flow used in
production demos:

``` r

builder <- DSI::newDSLoginBuilder()
builder$append(
  server = "opal1",
  url = "https://localhost:8443",
  user = Sys.getenv("OPAL_USER", "administrator"),
  password = Sys.getenv("OPAL_PASSWORD", "admin123"),
  table = "dsflower_demo.demo_breast_cancer_wisconsin_site1",
  driver = "OpalDriver"
)
# opal2 and opal3 are appended in the same way.

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

The demo can also be launched while rendering this vignette:

``` r

demo_file <- system.file("demos", "benchmark_breast_cancer.R", package = "dsFlowerClient")
if (!nzchar(demo_file)) demo_file <- file.path("..", "inst", "demos", "benchmark_breast_cancer.R")
source(demo_file)
```

Validation run on 2026-05-10:

    #>               metric    value
    #> 1               rows 683.0000
    #> 2              train 513.0000
    #> 3               test 170.0000
    #> 4              sites   3.0000
    #> 5        central_auc   0.9878
    #> 6      federated_auc   0.9908
    #> 7   central_accuracy   0.9471
    #> 8 federated_accuracy   0.9647

The expected output directory is `dsflower_output/demo_benchmarks/`,
with `summary.json` and `benchmark.rds` for each run. The Flower run
must finish with zero client failures and no active SuperNodes left
behind.
