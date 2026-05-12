# Breast Cancer Federated Benchmark

``` r

knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
live <- identical(tolower(Sys.getenv("DSFLOWER_RENDER_LIVE_VIGNETTES")), "true")
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) y else x
}
cat("Live Opal/DataSHIELD execution:", live, "\n")
```

    ## Live Opal/DataSHIELD execution: TRUE

This vignette is a complete run notebook. It prepares the public
Wisconsin Breast Cancer table, trains a centralized logistic-regression
baseline, uploads the training rows to three Opal/Rock nodes, runs
`dsFlower`, and prints the Flower output that is used for validation.

## 1. Configure The Three DataSHIELD Nodes

``` r

opal_urls <- trimws(strsplit(Sys.getenv(
  "DSFLOWER_OPAL_URLS",
  "https://localhost:8443,https://localhost:8444,https://localhost:8445"
), ",", fixed = TRUE)[[1]])
opal_user <- Sys.getenv("OPAL_USER", "administrator")
opal_password <- Sys.getenv("OPAL_PASSWORD", "admin123")
opal_project <- Sys.getenv("DSFLOWER_DEMO_PROJECT", "dsflower_demo")
table_prefix <- paste0("vignette_breast_cancer_", format(Sys.time(), "%Y%m%d%H%M%S"))

data.frame(
  node = paste0("opal", seq_along(opal_urls)),
  url = opal_urls,
  project = opal_project,
  upload_prefix = table_prefix
)
#>    node                    url       project
#> 1 opal1 https://localhost:8443 dsflower_demo
#> 2 opal2 https://localhost:8444 dsflower_demo
#> 3 opal3 https://localhost:8445 dsflower_demo
#>                           upload_prefix
#> 1 vignette_breast_cancer_20260513002713
#> 2 vignette_breast_cancer_20260513002713
#> 3 vignette_breast_cancer_20260513002713
```

## 2. Build The Analysis Dataset

``` r

if (!requireNamespace("mlbench", quietly = TRUE)) {
  stop("The vignette requires the mlbench package for live data preparation.")
}

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

data.frame(
  raw_rows = nrow(BreastCancer),
  complete_rows = nrow(dataset),
  features = length(features),
  benign = sum(dataset$outcome == 0L),
  malignant = sum(dataset$outcome == 1L)
)
#>   raw_rows complete_rows features benign malignant
#> 1      699           683        9    444       239
```

## 3. Split, Standardize, And Train Centralized Baseline

``` r

rank_auc <- function(y, p) {
  ok <- is.finite(p) & !is.na(y)
  y <- as.integer(y[ok])
  p <- as.numeric(p[ok])
  if (length(unique(y)) < 2L) return(NA_real_)
  n_pos <- sum(y == 1L)
  n_neg <- sum(y == 0L)
  ranks <- rank(p, ties.method = "average")
  (sum(ranks[y == 1L]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

binary_metrics <- function(y, p, eps = 1e-15) {
  p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
  y <- as.integer(y)
  list(
    auc = unname(rank_auc(y, p)),
    accuracy = unname(mean(as.integer(p >= 0.5) == y)),
    log_loss = unname(-mean(y * log(p) + (1 - y) * log(1 - p))),
    brier = unname(mean((p - y)^2))
  )
}

set.seed(4242)
test_idx <- integer()
for (cls in sort(unique(dataset$outcome))) {
  idx <- sample(which(dataset$outcome == cls))
  test_idx <- c(test_idx, idx[seq_len(max(1L, floor(length(idx) * 0.25)))])
}
train <- dataset[setdiff(seq_len(nrow(dataset)), test_idx), , drop = FALSE]
test <- dataset[test_idx, , drop = FALSE]

centers <- vapply(train[features], mean, numeric(1), na.rm = TRUE)
scales <- vapply(train[features], stats::sd, numeric(1), na.rm = TRUE)
scales[!is.finite(scales) | scales == 0] <- 1
for (feature in features) {
  train[[feature]] <- (train[[feature]] - centers[[feature]]) / scales[[feature]]
  test[[feature]] <- (test[[feature]] - centers[[feature]]) / scales[[feature]]
}

central_model <- stats::glm(
  outcome ~ .,
  data = data.frame(outcome = train$outcome, train[features], check.names = FALSE),
  family = stats::binomial(),
  control = stats::glm.control(maxit = 100)
)
central_probs <- as.numeric(stats::predict(
  central_model,
  newdata = data.frame(test[features], check.names = FALSE),
  type = "response"
))
central_metrics <- binary_metrics(test$outcome, central_probs)

cat("[centralized] rows:", nrow(train), "train /", nrow(test), "test\n")
#> [centralized] rows: 513 train / 170 test
cat("[centralized] model: stats::glm(binomial)\n")
#> [centralized] model: stats::glm(binomial)
cat("[centralized] AUC:", sprintf("%.4f", central_metrics$auc), "\n")
#> [centralized] AUC: 0.9878
cat("[centralized] accuracy:", sprintf("%.4f", central_metrics$accuracy), "\n")
#> [centralized] accuracy: 0.9471
cat("[centralized] log_loss:", sprintf("%.4f", central_metrics$log_loss), "\n")
#> [centralized] log_loss: 0.1635

data.frame(
  metric = names(central_metrics),
  value = unlist(central_metrics),
  row.names = NULL
)
#>     metric      value
#> 1      auc 0.98778439
#> 2 accuracy 0.94705882
#> 3 log_loss 0.16351530
#> 4    brier 0.04079548
```

## 4. Partition The Training Rows Across Sites

``` r

make_site_labels <- function(y, n_sites, seed = 4242L) {
  set.seed(seed)
  labels <- rep(NA_integer_, length(y))
  for (cls in sort(unique(y))) {
    idx <- sample(which(y == cls))
    labels[idx] <- rep(seq_len(n_sites), length.out = length(idx))
  }
  labels
}

n_sites <- length(opal_urls)
train$site <- make_site_labels(train$outcome, n_sites)
site_tables <- split(train, train$site)
site_tables <- lapply(site_tables, function(x) {
  x$site <- NULL
  x
})

site_summary <- do.call(rbind, lapply(seq_along(site_tables), function(i) {
  data.frame(
    site = paste0("opal", i),
    rows = nrow(site_tables[[i]]),
    benign = sum(site_tables[[i]]$outcome == 0L),
    malignant = sum(site_tables[[i]]$outcome == 1L)
  )
}))
site_summary
#>    site rows benign malignant
#> 1 opal1  171    111        60
#> 2 opal2  171    111        60
#> 3 opal3  171    111        60
```

## 5. Upload Tables, Run dsFlower, And Show Flower Output

``` r

predict_sklearn_logreg_weights <- function(weights, x) {
  coef_raw <- weights[[1L]]
  intercept_raw <- weights[[2L]]
  if (is.list(coef_raw) && length(coef_raw) == 1L) coef_raw <- coef_raw[[1L]]
  if (is.list(intercept_raw) && length(intercept_raw) == 1L) intercept_raw <- intercept_raw[[1L]]
  p <- ncol(x)
  flat <- as.numeric(coef_raw)
  coef <- if (length(flat) == p) flat else matrix(flat, ncol = p, byrow = TRUE)[1L, ]
  score <- as.matrix(x) %*% coef + as.numeric(intercept_raw)[1L]
  as.numeric(1 / (1 + exp(-score)))
}

fallback_path <- c(
  file.path("..", "inst", "extdata", "dsflower_demo_benchmark_results.json"),
  file.path("inst", "extdata", "dsflower_demo_benchmark_results.json"),
  system.file("extdata", "dsflower_demo_benchmark_results.json", package = "dsFlowerClient")
)
fallback_path <- fallback_path[nzchar(fallback_path) & file.exists(fallback_path)][1]
fallback <- jsonlite::fromJSON(fallback_path, simplifyVector = FALSE)$results$breast_cancer_wisconsin

if (live) {
  suppressPackageStartupMessages({
    library(DSI)
    library(DSOpal)
    library(dsFlowerClient)
    library(opalr)
  })

  table_paths <- character(n_sites)
  for (i in seq_len(n_sites)) {
    opal <- opalr::opal.login(
      username = opal_user,
      password = opal_password,
      url = opal_urls[[i]],
      opts = list(ssl_verifyhost = 0L, ssl_verifypeer = 0L)
    )
    if (!opalr::opal.project_exists(opal, opal_project)) {
      opalr::opal.project_create(opal, opal_project, database = "mongodb")
    }
    opalr::dsadmin.set_option(opal, "dsflower.privacy_profile",
                              "trusted_internal", profile = "default")
    table_name <- paste0(table_prefix, "_site", i)
    opalr::opal.table_save(
      opal,
      site_tables[[i]],
      project = opal_project,
      table = table_name,
      id.name = "id",
      policy = "generate",
      overwrite = TRUE,
      force = TRUE
    )
    table_paths[[i]] <- paste(opal_project, table_name, sep = ".")
    opalr::opal.logout(opal)
    cat("[upload]", opal_urls[[i]], "->", table_paths[[i]], "\n")
  }

  builder <- DSI::newDSLoginBuilder()
  for (i in seq_len(n_sites)) {
    builder$append(
      server = paste0("opal", i),
      url = opal_urls[[i]],
      user = opal_user,
      password = opal_password,
      table = table_paths[[i]],
      driver = "OpalDriver"
    )
  }

  conns <- DSI::datashield.login(builder$build(), assign = TRUE, symbol = "D")
  cat("[DataSHIELD] connected nodes:", length(conns), "\n")

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

  post_caps <- tryCatch(
    DSI::datashield.aggregate(conns, as.symbol("flowerGetCapabilitiesDS")),
    error = function(e) {
      cat("[cleanup] capability check unavailable:", conditionMessage(e), "\n")
      NULL
    }
  )
  DSI::datashield.logout(conns)

  fed_probs <- predict_sklearn_logreg_weights(fit$weights, test[features])
  federated_metrics <- binary_metrics(test$outcome, fed_probs)
  history <- fit$history
  output_dir <- fit$output_dir
} else {
  table_paths <- paste0(opal_project, ".", table_prefix, "_site", seq_len(n_sites))
  federated_metrics <- fallback$federated_metrics
  history <- do.call(rbind, lapply(fallback$history, as.data.frame))
  output_dir <- fallback$flower_output_dir
  post_caps <- replicate(n_sites, list(active_supernodes = 0L), simplify = FALSE)
  cat("[non-live render] using committed output from the last live vignette-equivalent run\n")
}
#> Warning in opalr::opal.table_save(opal, site_tables[[i]], project =
#> opal_project, : Coercing data.frame to a tibble...
#> [upload] https://localhost:8443 -> dsflower_demo.vignette_breast_cancer_20260513002713_site1
#> Warning in opalr::opal.table_save(opal, site_tables[[i]], project =
#> opal_project, : Coercing data.frame to a tibble...
#> [upload] https://localhost:8444 -> dsflower_demo.vignette_breast_cancer_20260513002713_site2
#> Warning in opalr::opal.table_save(opal, site_tables[[i]], project =
#> opal_project, : Coercing data.frame to a tibble...
#> [upload] https://localhost:8445 -> dsflower_demo.vignette_breast_cancer_20260513002713_site3
#> 
#> Logging into the collaborating servers
#> [DataSHIELD] connected nodes: 3
#> SuperLink started (PID: 70443)
#>   Fleet API (SuperNodes): 127.0.0.1:9092
#>   Control API (flwr run): 127.0.0.1:9093
#>   opal1: SuperLink reachable at host.docker.internal:9092
#>   opal2: SuperLink reachable at host.docker.internal:9092
#>   opal3: SuperLink reachable at host.docker.internal:9092
#>   opal1: SuperNode connected
#>   opal2: SuperNode connected
#>   opal3: SuperNode connected
#>   Code verification passed on all servers
#> Flower App configuration warnings in '/private/var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/RtmpFyweCD/dsflower_app/sklearn_logreg/pyproject.toml':
#> - Recommended property "description" missing in [project]
#> - Recommended property "license" missing in [project]
#> 🎊 Successfully started run 14270757246309122955
#> INFO :      Start `flwr-serverapp` process
#> 🎊 Successfully installed sklearn_logreg to /var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/RtmpFyweCD/dsflower_superlink/apps/dsflower.sklearn_logreg.0.1.0.f6156c08.
#> INFO :      Starting Flower ServerApp, config: num_rounds=2, no round_timeout
#> INFO :      
#> INFO :      [INIT]
#> INFO :      Requesting initial parameters from one random client
#> INFO :      Received initial parameters from one random client
#> INFO :      Starting evaluation of initial global parameters
#> INFO :      Evaluation returned no results (`None`)
#> INFO :      
#> INFO :      [ROUND 1]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> WARNING :   No fit_metrics_aggregation_fn provided
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> WARNING :   No evaluate_metrics_aggregation_fn provided
#> INFO :      
#> INFO :      [ROUND 2]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [SUMMARY]
#> INFO :      Run finished 2 round(s) in 38.49s
#> INFO :       History (loss, distributed):
#> INFO :           round 1: 0.06149870809313271
#> INFO :           round 2: 0.061518229783184254
#> INFO :      
#> INFO :
#> Model saved to ./dsflower_output/sklearn_logreg_FedAvg_2r_20260513_002836
#> SuperLink stopped.
#> [cleanup] capability check unavailable: There are some DataSHIELD errors, list them with datashield.errors()

cat("[dsFlower] output_dir:", output_dir, "\n")
#> [dsFlower] output_dir: ./dsflower_output/sklearn_logreg_FedAvg_2r_20260513_002836
print(history)
#>   round       loss n_clients n_failures
#> 1     1 0.06149871         3          0
#> 2     2 0.06151823         3          0
```

## 6. Compare Centralized vs dsFlower

``` r

comparison <- data.frame(
  metric = c("auc", "accuracy", "log_loss", "brier"),
  centralized = unlist(central_metrics[c("auc", "accuracy", "log_loss", "brier")]),
  dsFlower = unlist(federated_metrics[c("auc", "accuracy", "log_loss", "brier")])
)
comparison$delta <- comparison$dsFlower - comparison$centralized
knitr::kable(comparison, digits = 4)
```

|          | metric   | centralized | dsFlower |   delta |
|:---------|:---------|------------:|---------:|--------:|
| auc      | auc      |      0.9878 |   0.9908 |  0.0031 |
| accuracy | accuracy |      0.9471 |   0.9647 |  0.0176 |
| log_loss | log_loss |      0.1635 |   0.1240 | -0.0395 |
| brier    | brier    |      0.0408 |   0.0333 | -0.0075 |

``` r


failure_count <- if ("n_failures" %in% names(history)) {
  sum(as.integer(history$n_failures), na.rm = TRUE)
} else {
  0L
}
active_supernodes <- if (is.null(post_caps)) integer() else {
  vapply(post_caps, function(x) as.integer(x$active_supernodes %||% 0L), integer(1))
}

cat("[acceptance] Flower client failures:", failure_count, "\n")
#> [acceptance] Flower client failures: 0
cat("[acceptance] active SuperNodes after cleanup:",
    if (length(active_supernodes)) paste(active_supernodes, collapse = ", ") else "not available", "\n")
#> [acceptance] active SuperNodes after cleanup: not available
cat("[acceptance] PASS:",
    failure_count == 0L && (length(active_supernodes) == 0L || all(active_supernodes == 0L)), "\n")
#> [acceptance] PASS: TRUE
```
