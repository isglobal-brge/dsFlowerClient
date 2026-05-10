required <- c("DSI", "DSOpal", "dsFlowerClient", "jsonlite", "opalr")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Missing required demo packages: ", paste(missing, collapse = ", "), call. = FALSE)
}

suppressPackageStartupMessages({
  library(DSI)
  library(DSOpal)
  library(dsFlowerClient)
  library(jsonlite)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) y else x
}

demo_env <- function(name, default = NULL) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) default else value
}

demo_config <- function(demo_id, default_rounds = 2L) {
  urls <- strsplit(demo_env(
    "DSFLOWER_OPAL_URLS",
    demo_env("DSFLOWER_DEMO_URLS", "https://localhost:8443,https://localhost:8444,https://localhost:8445")
  ), ",", fixed = TRUE)[[1]]
  urls <- trimws(urls)

  users <- strsplit(demo_env("DSFLOWER_OPAL_USERS", demo_env("OPAL_USER", "administrator")), ",", fixed = TRUE)[[1]]
  passwords <- strsplit(demo_env("DSFLOWER_OPAL_PASSWORDS", demo_env("OPAL_PASSWORD", "admin123")), ",", fixed = TRUE)[[1]]

  if (length(users) == 1L) users <- rep(users, length(urls))
  if (length(passwords) == 1L) passwords <- rep(passwords, length(urls))
  if (length(users) != length(urls) || length(passwords) != length(urls)) {
    stop("DSFLOWER_OPAL_USERS and DSFLOWER_OPAL_PASSWORDS must have one value or one value per Opal URL.")
  }

  list(
    demo_id = demo_id,
    urls = urls,
    users = users,
    passwords = passwords,
    project = demo_env("DSFLOWER_DEMO_PROJECT", "dsflower_demo"),
    table_prefix = demo_env("DSFLOWER_DEMO_TABLE_PREFIX", demo_env("DSFLOWER_DEMO_PREFIX", paste0("demo_", demo_id))),
    output_root = demo_env("DSFLOWER_DEMO_OUTPUT_ROOT", file.path(getwd(), "dsflower_output", "demo_benchmarks")),
    rounds = as.integer(demo_env("DSFLOWER_DEMO_ROUNDS", default_rounds)),
    max_iter = as.integer(demo_env("DSFLOWER_DEMO_MAX_ITER", 100L)),
    seed = as.integer(demo_env("DSFLOWER_DEMO_SEED", 4242L)),
    profile = demo_env("DSFLOWER_DEMO_PRIVACY_PROFILE", demo_env("DSFLOWER_DEMO_PRIVACY", "trusted_internal"))
  )
}

opal_login <- function(cfg, index) {
  opalr::opal.login(
    username = cfg$users[[index]],
    password = cfg$passwords[[index]],
    url = cfg$urls[[index]],
    opts = list(ssl_verifyhost = 0L, ssl_verifypeer = 0L)
  )
}

ensure_project <- function(opal, project) {
  if (!opalr::opal.project_exists(opal, project)) {
    opalr::opal.project_create(opal, project, database = "mongodb")
  } else {
    datasource_type <- tryCatch(
      opalr::opal.project(opal, project)$datasource$type,
      error = function(e) NA_character_
    )
    if (identical(datasource_type, "null")) {
      opalr::opal.project_delete(opal, project, archive = FALSE, silent = TRUE)
      opalr::opal.project_create(opal, project, database = "mongodb")
    }
  }
  invisible(TRUE)
}

set_privacy_profile <- function(opal, profile) {
  opalr::dsadmin.set_option(opal, "dsflower.privacy_profile", profile, profile = "default")
  if (identical(profile, "sandbox_open")) {
    opalr::dsadmin.set_option(opal, "dsflower.allow_sandbox", TRUE, profile = "default")
  }
  invisible(TRUE)
}

upload_site_table <- function(opal, project, table, data) {
  if (opalr::opal.table_exists(opal, project, table)) {
    opalr::opal.table_delete(opal, project, table)
  }
  suppressWarnings(opalr::opal.table_save(
    opal,
    tibble = data,
    project = project,
    table = table,
    id.name = "id",
    policy = "generate",
    overwrite = TRUE,
    force = TRUE
  ))
  paste(project, table, sep = ".")
}

make_stratified_site_labels <- function(y, n_sites, seed = 4242L) {
  set.seed(seed)
  labels <- rep(NA_integer_, length(y))
  for (cls in sort(unique(y))) {
    idx <- which(y == cls)
    idx <- sample(idx, length(idx))
    labels[idx] <- rep(seq_len(n_sites), length.out = length(idx))
  }
  labels
}

stratified_train_test <- function(data, target, test_frac = 0.25, seed = 4242L) {
  set.seed(seed)
  y <- data[[target]]
  test_idx <- integer()
  for (cls in sort(unique(y))) {
    idx <- which(y == cls)
    idx <- sample(idx, length(idx))
    n_test <- max(1L, floor(length(idx) * test_frac))
    test_idx <- c(test_idx, idx[seq_len(n_test)])
  }
  train_idx <- setdiff(seq_len(nrow(data)), test_idx)
  list(train = data[train_idx, , drop = FALSE], test = data[test_idx, , drop = FALSE])
}

standardize_train_test <- function(train, test, features) {
  means <- vapply(train[features], function(x) mean(as.numeric(x), na.rm = TRUE), numeric(1))
  sds <- vapply(train[features], function(x) stats::sd(as.numeric(x), na.rm = TRUE), numeric(1))
  sds[!is.finite(sds) | sds == 0] <- 1

  for (feature in features) {
    train[[feature]] <- (as.numeric(train[[feature]]) - means[[feature]]) / sds[[feature]]
    test[[feature]] <- (as.numeric(test[[feature]]) - means[[feature]]) / sds[[feature]]
  }

  list(train = train, test = test, center = means, scale = sds)
}

rank_auc <- function(y, p) {
  ok <- is.finite(p) & !is.na(y)
  y <- as.integer(y[ok])
  p <- as.numeric(p[ok])
  if (length(unique(y)) < 2L) return(NA_real_)
  n_pos <- sum(y == 1L)
  n_neg <- sum(y == 0L)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  ranks <- rank(p, ties.method = "average")
  (sum(ranks[y == 1L]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

log_loss <- function(y, p, eps = 1e-15) {
  p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
  y <- as.integer(y)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

binary_metrics <- function(y, p) {
  y <- as.integer(y)
  p <- as.numeric(p)
  list(
    auc = unname(rank_auc(y, p)),
    accuracy = unname(mean(as.integer(p >= 0.5) == y)),
    log_loss = unname(log_loss(y, p)),
    brier = unname(mean((p - y)^2))
  )
}

fit_local_glm <- function(train, test, features, target) {
  local_train <- data.frame(outcome = as.integer(train[[target]]), train[features], check.names = FALSE)
  local_test <- data.frame(test[features], check.names = FALSE)
  model <- suppressWarnings(stats::glm(
    outcome ~ .,
    data = local_train,
    family = stats::binomial(),
    control = stats::glm.control(maxit = 100)
  ))
  probs <- as.numeric(stats::predict(model, newdata = local_test, type = "response"))
  list(model = model, probabilities = probs, metrics = binary_metrics(test[[target]], probs))
}

extract_weight_item <- function(weights, index) {
  if (is.null(weights) || length(weights) < index) return(NULL)
  item <- weights[[index]]
  if (is.list(item) && length(item) == 1L) item <- item[[1L]]
  item
}

predict_sklearn_logreg_weights <- function(weights, x) {
  if (is.null(weights) || length(weights) < 2L) {
    stop("Federated run did not return sklearn logistic-regression weights.", call. = FALSE)
  }

  coef_raw <- extract_weight_item(weights, 1L)
  intercept_raw <- extract_weight_item(weights, 2L)
  p <- ncol(x)

  dims <- dim(coef_raw)
  if (!is.null(dims) && length(dims) >= 2L) {
    coef_mat <- matrix(as.numeric(coef_raw), nrow = dims[[1L]], ncol = prod(dims[-1L]) / dims[[1L]])
    if (ncol(coef_mat) != p && nrow(coef_mat) == p) coef_mat <- t(coef_mat)
    coef <- coef_mat[nrow(coef_mat), ]
  } else {
    flat <- as.numeric(coef_raw)
    if (length(flat) == p) {
      coef <- flat
    } else if (length(flat) %% p == 0L) {
      coef <- matrix(flat, ncol = p, byrow = TRUE)[length(flat) / p, ]
    } else {
      stop("Cannot map returned coefficient tensor to ", p, " feature columns.", call. = FALSE)
    }
  }

  intercept <- as.numeric(intercept_raw)[1L] %||% 0
  score <- as.matrix(x) %*% as.numeric(coef) + intercept
  as.numeric(1 / (1 + exp(-score)))
}

safe_history <- function(run) {
  hist <- run$history %||% data.frame()
  if (is.data.frame(hist)) hist else as.data.frame(hist)
}

validate_run <- function(run, post_capabilities = NULL) {
  if (!is.null(run$status) && !identical(as.integer(run$status), 0L)) {
    stop("Federated run finished with non-zero status: ", run$status, call. = FALSE)
  }

  hist <- safe_history(run)
  if ("n_failures" %in% names(hist) && any(as.integer(hist$n_failures) > 0L, na.rm = TRUE)) {
    stop("Federated run reported client failures.", call. = FALSE)
  }

  if (!is.null(post_capabilities)) {
    active <- vapply(post_capabilities, function(x) as.integer(x$active_supernodes %||% 0L), integer(1))
    if (any(active > 0L, na.rm = TRUE)) {
      stop("A Flower SuperNode remained active after the run: ", paste(active, collapse = ", "), call. = FALSE)
    }
  }

  invisible(TRUE)
}

prepare_for_benchmark <- function(data, features, target = "outcome") {
  stopifnot(is.data.frame(data))
  required_cols <- c("id", features, target)
  missing <- setdiff(required_cols, names(data))
  if (length(missing)) stop("Dataset is missing required columns: ", paste(missing, collapse = ", "))

  data <- data[complete.cases(data[, required_cols, drop = FALSE]), required_cols, drop = FALSE]
  data$id <- as.character(data$id)
  data[[target]] <- as.integer(data[[target]])
  if (!all(data[[target]] %in% c(0L, 1L))) stop("Target must be binary encoded as 0/1.")

  for (feature in features) data[[feature]] <- as.numeric(data[[feature]])
  data
}

run_dsflower_benchmark <- function(data,
                                   features,
                                   demo_id,
                                   dataset_label,
                                   data_mode,
                                   target = "outcome",
                                   default_rounds = 2L,
                                   test_frac = 0.25) {
  cfg <- demo_config(demo_id, default_rounds = default_rounds)
  data <- prepare_for_benchmark(data, features, target = target)
  if (nrow(data) < 120L) stop("The demo requires at least 120 complete rows after preprocessing.")

  split <- stratified_train_test(data, target = target, test_frac = test_frac, seed = cfg$seed)
  scaled <- standardize_train_test(split$train, split$test, features)
  train <- scaled$train
  test <- scaled$test

  n_sites <- length(cfg$urls)
  train$site <- make_stratified_site_labels(train[[target]], n_sites = n_sites, seed = cfg$seed)
  if (any(table(train$site) < 50L)) {
    stop("At least one site has fewer than 50 training rows; DataSHIELD filters would block the demo.")
  }

  central <- fit_local_glm(train, test, features, target)
  site_tables <- split(train, train$site)

  table_paths <- character(n_sites)
  for (i in seq_len(n_sites)) {
    opal <- opal_login(cfg, i)
    on.exit(try(opalr::opal.logout(opal), silent = TRUE), add = TRUE)
    ensure_project(opal, cfg$project)
    set_privacy_profile(opal, cfg$profile)
    table_name <- paste0(cfg$table_prefix, "_site", i)
    table_paths[[i]] <- upload_site_table(opal, cfg$project, table_name, site_tables[[as.character(i)]])
    opalr::opal.logout(opal)
  }

  builder <- DSI::newDSLoginBuilder()
  for (i in seq_len(n_sites)) {
    builder$append(
      server = paste0("opal", i),
      url = cfg$urls[[i]],
      user = cfg$users[[i]],
      password = cfg$passwords[[i]],
      table = table_paths[[i]],
      driver = "OpalDriver"
    )
  }

  message("Connecting to ", n_sites, " Opal nodes for ", dataset_label, "...")
  conns <- DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
  on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

  run <- ds.flower.fit(
    conns,
    symbol = "D",
    target = target,
    features = features,
    model = "sklearn_logreg",
    model_params = list(max_iter = cfg$max_iter),
    strategy = "fedavg",
    privacy = "trusted_internal",
    rounds = cfg$rounds
  )
  fed_probs <- predict_sklearn_logreg_weights(run$weights, test[features])
  fed_metrics <- binary_metrics(test[[target]], fed_probs)

  post_caps <- tryCatch(
    DSI::datashield.aggregate(conns, as.symbol("flowerGetCapabilitiesDS")),
    error = function(e) NULL
  )
  validate_run(run, post_caps)

  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_dir <- file.path(cfg$output_root, paste0(demo_id, "_", timestamp))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  summary <- list(
    demo_id = demo_id,
    dataset_label = dataset_label,
    data_mode = data_mode,
    n_total = nrow(data),
    n_train = nrow(train),
    n_test = nrow(test),
    n_sites = n_sites,
    rounds = cfg$rounds,
    features = features,
    central_metrics = central$metrics,
    federated_metrics = fed_metrics,
    metric_delta = list(
      auc = fed_metrics$auc - central$metrics$auc,
      accuracy = fed_metrics$accuracy - central$metrics$accuracy,
      log_loss = fed_metrics$log_loss - central$metrics$log_loss
    ),
    flower_output_dir = run$output_dir %||% NA_character_,
    history = safe_history(run)
  )

  jsonlite::write_json(summary, file.path(out_dir, "summary.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
  saveRDS(list(summary = summary, run = run, central_model = central$model), file.path(out_dir, "benchmark.rds"))

  cat("\n", dataset_label, "\n", sep = "")
  cat("Data mode: ", data_mode, "\n", sep = "")
  cat("Rows: ", nrow(data), " | Train: ", nrow(train), " | Test: ", nrow(test), " | Sites: ", n_sites, "\n", sep = "")
  cat("Central AUC: ", sprintf("%.4f", central$metrics$auc), " | FL AUC: ", sprintf("%.4f", fed_metrics$auc), "\n", sep = "")
  cat("Central accuracy: ", sprintf("%.4f", central$metrics$accuracy), " | FL accuracy: ", sprintf("%.4f", fed_metrics$accuracy), "\n", sep = "")
  cat("Output: ", normalizePath(out_dir), "\n", sep = "")
  cat("PASS\n")

  invisible(summary)
}
