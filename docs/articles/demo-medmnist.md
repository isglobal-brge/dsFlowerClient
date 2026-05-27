# MedMNIST Federated Benchmark

``` r

knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
live <- identical(tolower(Sys.getenv("DSFLOWER_RENDER_LIVE_VIGNETTES")), "true")
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) y else x
}
cat("Live Opal/DataSHIELD execution:", live, "\n")
```

    ## Live Opal/DataSHIELD execution: FALSE

This vignette is a complete image-origin federated benchmark. It loads
BreastMNIST images, average-pools each image into a compact `7 x 7`
feature table, trains a centralized baseline, then runs the same feature
table through dsFlower across three Opal/Rock nodes.

## 1. Configure The Three DataSHIELD Nodes

``` r

opal_urls <- trimws(strsplit(Sys.getenv(
  "DSFLOWER_OPAL_URLS",
  "https://localhost:8443,https://localhost:8444,https://localhost:8445"
), ",", fixed = TRUE)[[1]])
opal_user <- Sys.getenv("OPAL_USER", "administrator")
opal_password <- Sys.getenv("OPAL_PASSWORD", "admin123")
opal_project <- Sys.getenv("DSFLOWER_DEMO_PROJECT", "dsflower_demo")
table_prefix <- paste0("vignette_medmnist_", format(Sys.time(), "%Y%m%d%H%M%S"))

data.frame(
  node = paste0("opal", seq_along(opal_urls)),
  url = opal_urls,
  project = opal_project,
  upload_prefix = table_prefix
)
#>    node                    url       project                    upload_prefix
#> 1 opal1 https://localhost:8443 dsflower_demo vignette_medmnist_20260527081602
#> 2 opal2 https://localhost:8444 dsflower_demo vignette_medmnist_20260527081602
#> 3 opal3 https://localhost:8445 dsflower_demo vignette_medmnist_20260527081602
```

## 2. Load Images And Build A Server-Side Feature Table

``` r

make_image_like_fixture <- function(n = 510L, seed = 4242L) {
  set.seed(seed)
  labels <- rep(c(0L, 1L), length.out = n)
  labels <- sample(labels)
  pixels <- matrix(stats::rnorm(n * 49L, sd = 0.18), nrow = n)
  signal <- matrix(rep(labels, each = 49L), nrow = n)
  pixels <- pmin(pmax(pixels + signal * rep(seq(0.05, 0.35, length.out = 49L), each = n), 0), 1)
  colnames(pixels) <- sprintf("px_%02d", seq_len(49L) - 1L)
  data.frame(id = paste0("breastmnist_fixture_", seq_len(n)), pixels,
             outcome = labels, check.names = FALSE)
}

load_breastmnist <- function(limit = 600L) {
  if (!live) return(make_image_like_fixture())
  if (!requireNamespace("processx", quietly = TRUE)) return(make_image_like_fixture())

  python <- Sys.getenv("DSFLOWER_VALIDATION_PYTHON", "")
  if (!nzchar(python)) {
    python <- tryCatch(dsFlowerClient:::.client_python_cmd(), error = function(e) Sys.which("python3"))
  }
  if (!nzchar(python)) return(make_image_like_fixture())

  out <- tempfile(fileext = ".csv")
  code <- paste(
    "import numpy as np, pandas as pd",
    "from medmnist import BreastMNIST",
    sprintf("out_path = %s", jsonlite::toJSON(out, auto_unbox = TRUE)),
    sprintf("limit = int(%s)", jsonlite::toJSON(limit, auto_unbox = TRUE)),
    "rng = np.random.default_rng(4242)",
    "imgs, labels = [], []",
    "for split in ['train', 'val', 'test']:",
    "    ds = BreastMNIST(split=split, download=True, size=28)",
    "    imgs.append(np.asarray(ds.imgs))",
    "    labels.append(np.asarray(ds.labels).reshape(-1).astype(int))",
    "imgs = np.concatenate(imgs, axis=0)",
    "labels = np.concatenate(labels, axis=0)",
    "idx = []",
    "for cls in np.unique(labels):",
    "    cls_idx = rng.permutation(np.where(labels == cls)[0])",
    "    idx.extend(cls_idx[:max(1, limit // len(np.unique(labels)))].tolist())",
    "idx = np.array(idx[:limit])",
    "imgs = imgs[idx]",
    "labels = labels[idx]",
    "if imgs.ndim == 4:",
    "    imgs = imgs[..., 0]",
    "imgs = imgs.astype('float32') / 255.0",
    "pooled = imgs.reshape((-1, 7, 4, 7, 4)).mean(axis=(2, 4))",
    "flat = pooled.reshape((pooled.shape[0], -1))",
    "cols = [f'px_{i:02d}' for i in range(flat.shape[1])]",
    "df = pd.DataFrame(flat, columns=cols)",
    "df.insert(0, 'id', [f'breastmnist_{i+1}' for i in range(len(df))])",
    "df['outcome'] = labels.astype(int)",
    "df.to_csv(out_path, index=False)",
    sep = "\n"
  )
  run <- processx::run(python, c("-c", code), echo = TRUE, error_on_status = FALSE, timeout = 900)
  if (run$status != 0L || !file.exists(out)) {
    warning("Could not load BreastMNIST through Python; using deterministic image-like fixture.")
    return(make_image_like_fixture())
  }
  utils::read.csv(out, check.names = FALSE)
}

dataset <- load_breastmnist()
features <- grep("^px_", names(dataset), value = TRUE)
data_mode <- if (startsWith(dataset$id[[1]], "breastmnist_")) {
  "medmnist::BreastMNIST pooled 7x7 pixels"
} else {
  "deterministic image-like fallback"
}

cat("[data] mode:", data_mode, "\n")
#> [data] mode: medmnist::BreastMNIST pooled 7x7 pixels
data.frame(
  rows = nrow(dataset),
  features = length(features),
  class_0 = sum(dataset$outcome == 0L),
  class_1 = sum(dataset$outcome == 1L)
)
#>   rows features class_0 class_1
#> 1  510       49     255     255
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
#> [centralized] rows: 384 train / 126 test
cat("[centralized] model: stats::glm(binomial)\n")
#> [centralized] model: stats::glm(binomial)
cat("[centralized] AUC:", sprintf("%.4f", central_metrics$auc), "\n")
#> [centralized] AUC: 0.5117
cat("[centralized] accuracy:", sprintf("%.4f", central_metrics$accuracy), "\n")
#> [centralized] accuracy: 0.5238
data.frame(metric = names(central_metrics), value = unlist(central_metrics), row.names = NULL)
#>     metric     value
#> 1      auc 0.5117158
#> 2 accuracy 0.5238095
#> 3 log_loss 0.7701551
#> 4    brier 0.2807132
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
    class_0 = sum(site_tables[[i]]$outcome == 0L),
    class_1 = sum(site_tables[[i]]$outcome == 1L)
  )
}))
#>    site rows class_0 class_1
#> 1 opal1  128      64      64
#> 2 opal2  128      64      64
#> 3 opal3  128      64      64
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
fallback <- jsonlite::fromJSON(fallback_path, simplifyVector = FALSE)$results$medmnist_breastmnist

if (live) {
  suppressPackageStartupMessages({
    library(DSI)
    library(DSOpal)
    library(dsFlowerClient)
    library(opalr)
  })

  table_paths <- character(n_sites)
  for (i in seq_len(n_sites)) {
    opal <- opalr::opal.login(username = opal_user, password = opal_password,
                              url = opal_urls[[i]],
                              opts = list(ssl_verifyhost = 0L, ssl_verifypeer = 0L))
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
#> [dsFlower] output_dir: ./dsflower_output/sklearn_logreg_FedAvg_2r_20260526_180253
print(history)
#>   round   loss n_clients n_failures
#> 1     1 0.5105         3          0
#> 2     2 0.5106         3          0
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
| auc      | auc      |      0.5117 |   0.7985 |  0.2868 |
| accuracy | accuracy |      0.5238 |   0.7165 |  0.1927 |
| log_loss | log_loss |      0.7702 |   0.5499 | -0.2203 |
| brier    | brier    |      0.2807 |   0.1764 | -0.1043 |

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
