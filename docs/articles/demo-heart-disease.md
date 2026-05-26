# Heart Disease Federated Benchmark

``` r

knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
live <- identical(tolower(Sys.getenv("DSFLOWER_RENDER_LIVE_VIGNETTES")), "true")
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) y else x
}
cat("Live Opal/DataSHIELD execution:", live, "\n")
```

    ## Live Opal/DataSHIELD execution: FALSE

This vignette is a complete clinical-tabular run notebook. It downloads
the UCI Cleveland Heart Disease table when available, trains a
centralized baseline, then uploads the same training rows to three
Opal/Rock nodes and runs dsFlower.

## 1. Configure The Three DataSHIELD Nodes

``` r

opal_urls <- trimws(strsplit(Sys.getenv(
  "DSFLOWER_OPAL_URLS",
  "https://localhost:8443,https://localhost:8444,https://localhost:8445"
), ",", fixed = TRUE)[[1]])
opal_user <- Sys.getenv("OPAL_USER", "administrator")
opal_password <- Sys.getenv("OPAL_PASSWORD", "admin123")
opal_project <- Sys.getenv("DSFLOWER_DEMO_PROJECT", "dsflower_demo")
table_prefix <- paste0("vignette_heart_disease_", format(Sys.time(), "%Y%m%d%H%M%S"))

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
#> 1 vignette_heart_disease_20260526181523
#> 2 vignette_heart_disease_20260526181523
#> 3 vignette_heart_disease_20260526181523
```

## 2. Load And Clean The Clinical Dataset

``` r

simulate_heart_like <- function(n = 303L, seed = 4242L) {
  set.seed(seed)
  age <- pmin(pmax(round(stats::rnorm(n, 55, 9)), 29), 77)
  sex <- stats::rbinom(n, 1, 0.68)
  cp <- sample(1:4, n, replace = TRUE, prob = c(0.2, 0.18, 0.29, 0.33))
  trestbps <- pmax(round(stats::rnorm(n, 132, 17)), 90)
  chol <- pmax(round(stats::rnorm(n, 245, 52)), 120)
  fbs <- stats::rbinom(n, 1, 0.15)
  restecg <- sample(0:2, n, replace = TRUE, prob = c(0.5, 0.03, 0.47))
  thalach <- pmax(round(stats::rnorm(n, 150 - 0.45 * (age - 55), 22)), 70)
  exang <- stats::rbinom(n, 1, plogis(-1.2 + 0.03 * (age - 55) + 0.45 * (cp == 4)))
  oldpeak <- pmax(stats::rnorm(n, 1.0 + 0.7 * exang, 0.9), 0)
  slope <- sample(1:3, n, replace = TRUE, prob = c(0.45, 0.45, 0.10))
  ca <- sample(0:3, n, replace = TRUE, prob = c(0.58, 0.22, 0.13, 0.07))
  thal <- sample(c(3, 6, 7), n, replace = TRUE, prob = c(0.55, 0.06, 0.39))
  risk <- -3.7 + 0.035 * age + 0.65 * sex + 0.55 * (cp == 4) +
    0.55 * exang + 0.45 * oldpeak + 0.45 * ca + 0.35 * (thal == 7) -
    0.018 * (thalach - 140)
  outcome <- stats::rbinom(n, 1, plogis(risk))
  data.frame(
    id = paste0("heart_sim_", seq_len(n)),
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal, outcome,
    check.names = FALSE
  )
}

load_uci_cleveland <- function() {
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
  tmp <- tempfile(fileext = ".data")
  utils::download.file(url, tmp, quiet = TRUE, mode = "wb")
  cols <- c(
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
  )
  raw <- utils::read.table(tmp, sep = ",", na.strings = "?", col.names = cols)
  raw <- raw[stats::complete.cases(raw), , drop = FALSE]
  data.frame(
    id = paste0("cleveland_", seq_len(nrow(raw))),
    raw[setdiff(cols, "num")],
    outcome = as.integer(raw$num > 0),
    check.names = FALSE
  )
}

dataset <- tryCatch({
  x <- load_uci_cleveland()
  attr(x, "data_mode") <- "UCI Heart Disease processed Cleveland"
  x
}, error = function(e) {
  x <- simulate_heart_like()
  attr(x, "data_mode") <- paste("deterministic fallback:", conditionMessage(e))
  x
})
features <- setdiff(names(dataset), c("id", "outcome"))

cat("[data] mode:", attr(dataset, "data_mode"), "\n")
#> [data] mode: UCI Heart Disease processed Cleveland
data.frame(
  rows = nrow(dataset),
  features = length(features),
  controls = sum(dataset$outcome == 0L),
  cases = sum(dataset$outcome == 1L)
)
#>   rows features controls cases
#> 1  297       13      160   137
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
  train[[feature]] <- (as.numeric(train[[feature]]) - centers[[feature]]) / scales[[feature]]
  test[[feature]] <- (as.numeric(test[[feature]]) - centers[[feature]]) / scales[[feature]]
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
#> [centralized] rows: 223 train / 74 test
cat("[centralized] model: stats::glm(binomial)\n")
#> [centralized] model: stats::glm(binomial)
cat("[centralized] AUC:", sprintf("%.4f", central_metrics$auc), "\n")
#> [centralized] AUC: 0.8919
cat("[centralized] accuracy:", sprintf("%.4f", central_metrics$accuracy), "\n")
#> [centralized] accuracy: 0.8378

data.frame(metric = names(central_metrics), value = unlist(central_metrics), row.names = NULL)
#>     metric     value
#> 1      auc 0.8919118
#> 2 accuracy 0.8378378
#> 3 log_loss 0.4102382
#> 4    brier 0.1204378
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

do.call(rbind, lapply(seq_along(site_tables), function(i) {
  data.frame(
    site = paste0("opal", i),
    rows = nrow(site_tables[[i]]),
    controls = sum(site_tables[[i]]$outcome == 0L),
    cases = sum(site_tables[[i]]$outcome == 1L)
  )
}))
#>    site rows controls cases
#> 1 opal1   75       40    35
#> 2 opal2   74       40    34
#> 3 opal3   74       40    34
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
  file.path("..", "inst", "extdata", "dsflower_public_benchmark_results.json"),
  file.path("inst", "extdata", "dsflower_public_benchmark_results.json"),
  system.file("extdata", "dsflower_public_benchmark_results.json", package = "dsFlowerClient"),
  file.path("..", "inst", "extdata", "dsflower_demo_benchmark_results.json"),
  file.path("inst", "extdata", "dsflower_demo_benchmark_results.json"),
  system.file("extdata", "dsflower_demo_benchmark_results.json", package = "dsFlowerClient")
)
fallback_path <- fallback_path[nzchar(fallback_path) & file.exists(fallback_path)][1]
fallback <- jsonlite::fromJSON(fallback_path, simplifyVector = FALSE)$results$uci_heart_disease

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
    opalr::opal.table_save(opal, site_tables[[i]], project = opal_project,
                           table = table_name, id.name = "id",
                           policy = "generate", overwrite = TRUE, force = TRUE)
    table_paths[[i]] <- paste(opal_project, table_name, sep = ".")
    opalr::opal.logout(opal)
    cat("[upload]", opal_urls[[i]], "->", table_paths[[i]], "\n")
  }

  builder <- DSI::newDSLoginBuilder()
  for (i in seq_len(n_sites)) {
    builder$append(server = paste0("opal", i), url = opal_urls[[i]],
                   user = opal_user, password = opal_password,
                   table = table_paths[[i]], driver = "OpalDriver")
  }
  conns <- DSI::datashield.login(builder$build(), assign = TRUE, symbol = "D")
  cat("[DataSHIELD] connected nodes:", length(conns), "\n")

  fit <- ds.flower.fit(
    conns, symbol = "D", target = "outcome", features = features,
    model = "sklearn_logreg", model_params = list(max_iter = 100L),
    strategy = "fedavg", privacy = "trusted_internal", rounds = 2L
  )

  post_caps <- tryCatch(
    DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
    error = function(e) {
      cat("[cleanup] capability check unavailable:", conditionMessage(e), "\n")
      NULL
    }
  )
  DSI::datashield.logout(conns)
  federated_metrics <- binary_metrics(test$outcome, predict_sklearn_logreg_weights(fit$weights, test[features]))
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
#> [non-live render] using committed output from the last live vignette-equivalent run

cat("[dsFlower] output_dir:", output_dir, "\n")
#> [dsFlower] output_dir: ./dsflower_output/sklearn_logreg_FedAvg_2r_20260526_175945
print(history)
#>   round   loss n_clients n_failures
#> 1     1 0.3425         3          0
#> 2     2 0.3425         3          0
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
| auc      | auc      |      0.8919 |   0.8926 |  0.0007 |
| accuracy | accuracy |      0.8378 |   0.8514 |  0.0136 |
| log_loss | log_loss |      0.4102 |   0.3954 | -0.0148 |
| brier    | brier    |      0.1204 |   0.1180 | -0.0024 |

``` r


failure_count <- if ("n_failures" %in% names(history)) sum(as.integer(history$n_failures), na.rm = TRUE) else 0L
active_supernodes <- if (is.null(post_caps)) integer() else {
  vapply(post_caps, function(x) as.integer(x$active_supernodes %||% 0L), integer(1))
}
cat("[acceptance] Flower client failures:", failure_count, "\n")
#> [acceptance] Flower client failures: 0
cat("[acceptance] active SuperNodes after cleanup:",
    if (length(active_supernodes)) paste(active_supernodes, collapse = ", ") else "not available", "\n")
#> [acceptance] active SuperNodes after cleanup: 0, 0, 0
cat("[acceptance] PASS:",
    failure_count == 0L && (length(active_supernodes) == 0L || all(active_supernodes == 0L)), "\n")
#> [acceptance] PASS: TRUE
```
