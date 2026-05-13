#!/usr/bin/env Rscript

file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = FALSE)
} else {
  tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
}
if (is.null(script_file) || !nzchar(script_file)) {
  script_file <- "inst/demos/validate_methods.R"
}
demo_lib <- file.path(dirname(script_file), "lib", "benchmark_utils.R")
if (!file.exists(demo_lib)) {
  demo_lib <- file.path("inst", "demos", "lib", "benchmark_utils.R")
}
source(demo_lib)

required <- c("jsonlite", "opalr", "DSI", "DSOpal")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Missing required validation packages: ", paste(missing, collapse = ", "), call. = FALSE)
}

as_scalar <- function(x, default = NA_real_) {
  if (is.null(x) || length(x) == 0L) return(default)
  x <- suppressWarnings(as.numeric(x))
  if (length(x) == 0L || is.na(x[[1L]])) default else x[[1L]]
}

find_validation_python <- function() {
  candidates <- c(
    demo_env("DSFLOWER_VALIDATION_PYTHON"),
    "/opt/homebrew/opt/python@3.11/libexec/bin/python",
    Sys.which("python3"),
    Sys.which("python")
  )
  candidates <- unique(candidates[nzchar(candidates)])
  for (candidate in candidates) {
    if (file.exists(candidate) || nzchar(Sys.which(candidate))) return(candidate)
  }
  stop("No Python interpreter found. Set DSFLOWER_VALIDATION_PYTHON.", call. = FALSE)
}

validation_python_script <- function() {
  pkg_path <- system.file("python", "method_validation_baselines.py", package = "dsFlowerClient")
  if (nzchar(pkg_path)) return(pkg_path)
  local_path <- file.path("inst", "python", "method_validation_baselines.py")
  if (file.exists(local_path)) return(local_path)
  stop("Cannot find method_validation_baselines.py.", call. = FALSE)
}

make_method_validation_data <- function(n_per_site = 90L, n_sites = 3L, seed = 4242L) {
  set.seed(seed)
  n <- as.integer(n_per_site) * as.integer(n_sites)
  p <- 12L
  x <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  x <- scale(x)
  colnames(x) <- paste0("x", seq_len(p))

  lin_bin <- 0.9 * x[, 1] - 0.7 * x[, 2] + 0.45 * x[, 3] + stats::rnorm(n, sd = 0.4)
  binary <- as.integer(lin_bin > stats::median(lin_bin))

  lin_reg <- 1.4 * x[, 1] - 1.1 * x[, 4] + 0.5 * x[, 7] + stats::rnorm(n, sd = 0.45)

  score_multi <- x[, 2] - 0.5 * x[, 5] + 0.35 * x[, 8] + stats::rnorm(n, sd = 0.5)
  class3 <- as.integer(cut(score_multi,
                           breaks = stats::quantile(score_multi, probs = c(0, 1 / 3, 2 / 3, 1)),
                           include.lowest = TRUE, labels = FALSE)) - 1L

  label_a <- as.integer(x[, 1] + x[, 6] + stats::rnorm(n, sd = 0.5) > 0)
  label_b <- as.integer(-x[, 2] + x[, 7] + stats::rnorm(n, sd = 0.5) > 0)
  label_c <- as.integer(x[, 3] - x[, 8] + stats::rnorm(n, sd = 0.5) > 0)

  lambda <- exp(pmin(pmax(0.2 + 0.35 * x[, 1] - 0.25 * x[, 9], -2), 2))
  count <- stats::rpois(n, lambda = lambda)

  base_time <- exp(2.0 - 0.25 * x[, 1] + 0.18 * x[, 5] + stats::rnorm(n, sd = 0.35))
  censor_time <- exp(2.1 + stats::rnorm(n, sd = 0.45))
  time <- pmax(pmin(base_time, censor_time), 0.05)
  event <- as.integer(base_time <= censor_time)

  cause_score <- 0.6 * x[, 10] - 0.45 * x[, 11] + stats::rnorm(n, sd = 0.6)
  event_type <- ifelse(event == 0L, 0L, ifelse(cause_score > 0, 1L, 2L))

  data <- data.frame(
    id = sprintf("mv_%04d", seq_len(n)),
    as.data.frame(x, check.names = FALSE),
    outcome = binary,
    y = as.numeric(lin_reg),
    class3 = as.integer(class3),
    label_a = label_a,
    label_b = label_b,
    label_c = label_c,
    count = as.integer(count),
    time = as.numeric(time),
    event = as.integer(event),
    event_type = as.integer(event_type),
    stringsAsFactors = FALSE
  )

  data$site <- make_stratified_site_labels(data$outcome, n_sites = n_sites, seed = seed)
  data
}

method_specs <- function() {
  common_fast <- list(batch_size = 32L, local_epochs = 1L)
  list(
    list(method = "sklearn_logreg", model = "sklearn_logreg", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = list(max_iter = 100L), rounds = 1L,
         notes = "Federated sklearn logistic regression."),
    list(method = "sklearn_ridge", model = "sklearn_ridge", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = list(alpha = 1.0), rounds = 1L,
         notes = "Federated ridge classifier."),
    list(method = "sklearn_sgd", model = "sklearn_sgd", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = list(loss = "log_loss", alpha = 0.001,
                             lr_schedule = "constant", eta0 = 0.01,
                             max_iter = 1000L), rounds = 1L,
         notes = "Federated SGD classifier with logistic loss."),
    list(method = "sklearn_svm", model = "sklearn_svm", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = list(alpha = 0.0001, lr_schedule = "optimal"), rounds = 1L,
         notes = "Linear SVM via SGD hinge loss."),
    list(method = "sklearn_elastic_net", model = "sklearn_elastic_net", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = list(l1_ratio = 0.15, alpha = 0.001,
                             loss = "log_loss", lr_schedule = "constant",
                             eta0 = 0.01, max_iter = 1000L), rounds = 1L,
         notes = "Elastic-net regularized SGD classifier."),
    list(method = "pytorch_logreg", model = "pytorch_logreg", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(learning_rate = 0.03)), rounds = 1L,
         notes = "PyTorch binary logistic regression."),
    list(method = "pytorch_mlp", model = "pytorch_mlp", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(hidden_layers = "16,8", learning_rate = 0.01)), rounds = 1L,
         notes = "Small tabular MLP."),
    list(method = "pytorch_linear_regression", model = "pytorch_linear_regression", task = "regression",
         target = "y", features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(learning_rate = 0.03)), rounds = 1L,
         notes = "PyTorch linear regression with MSE."),
    list(method = "pytorch_multiclass", model = "pytorch_multiclass", task = "classification",
         target = "class3", features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(hidden_layers = "", n_classes = 3L, learning_rate = 0.03)), rounds = 1L,
         notes = "Three-class PyTorch classifier."),
    list(method = "pytorch_multilabel", model = "pytorch_multilabel", task = "classification",
         target = c("label_a", "label_b", "label_c"), features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(n_labels = 3L, hidden_layers = "16", learning_rate = 0.02)), rounds = 1L,
         notes = "Multi-label classifier with three binary endpoints."),
    list(method = "pytorch_poisson", model = "pytorch_poisson", task = "regression",
         target = "count", features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(hidden_layers = "", learning_rate = 0.02)), rounds = 1L,
         notes = "Poisson regression for count outcomes."),
    list(method = "pytorch_coxph", model = "pytorch_coxph", task = "survival",
         target = c("time", "event"), features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(learning_rate = 0.01)), rounds = 1L,
         notes = "Cox proportional hazards survival model."),
    list(method = "pytorch_lognormal_aft", model = "pytorch_lognormal_aft", task = "survival",
         target = c("time", "event"), features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(learning_rate = 0.01)), rounds = 1L,
         notes = "Log-normal accelerated failure time model."),
    list(method = "pytorch_cause_specific_cox", model = "pytorch_cause_specific_cox", task = "survival",
         target = c("time", "event_type"), features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(n_causes = 2L, learning_rate = 0.01)), rounds = 1L,
         notes = "Cause-specific Cox competing-risks model."),
    list(method = "pytorch_lstm", model = "pytorch_lstm", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(hidden_size = 8L, num_layers = 1L, learning_rate = 0.005)), rounds = 1L,
         notes = "LSTM over ordered tabular features interpreted as a short sequence."),
    list(method = "pytorch_tcn", model = "pytorch_tcn", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = modifyList(common_fast, list(n_channels = 1L, kernel_size = 3L, n_layers = 1L, learning_rate = 0.005)), rounds = 1L,
         notes = "Temporal convolutional network over ordered tabular features."),
    list(method = "xgboost", model = "xgboost", task = "classification",
         target = "outcome", features = paste0("x", 1:12),
         model_params = list(n_trees = 3L, max_depth = 2L, eta = 0.3,
                             reg_lambda = 1.0, n_bins = 16L,
                             objective = "binary:logistic"), rounds = 1L,
         notes = "Histogram XGBoost; SecAgg is enforced by secure privacy profiles.")
  )
}

selected_specs <- function(specs) {
  selection <- demo_env("DSFLOWER_VALIDATE_METHODS", "all")
  if (!nzchar(selection) || identical(tolower(selection), "all")) return(specs)
  wanted <- trimws(strsplit(selection, ",", fixed = TRUE)[[1]])
  specs[vapply(specs, function(x) x$method %in% wanted, logical(1))]
}

write_site_tables <- function(cfg, data) {
  n_sites <- length(cfg$urls)
  table_paths <- character(n_sites)
  for (i in seq_len(n_sites)) {
    opal <- opal_login(cfg, i)
    on.exit(try(opalr::opal.logout(opal), silent = TRUE), add = TRUE)
    ensure_project(opal, cfg$project)
    set_privacy_profile(opal, cfg$profile)
    site_data <- data[data$site == i, , drop = FALSE]
    site_data$site <- NULL
    table_name <- paste0(cfg$table_prefix, "_site", i)
    table_paths[[i]] <- upload_site_table(opal, cfg$project, table_name, site_data)
    opalr::opal.logout(opal)
  }
  table_paths
}

connect_validation_tables <- function(cfg, table_paths) {
  builder <- DSI::newDSLoginBuilder()
  for (i in seq_along(cfg$urls)) {
    builder$append(
      server = paste0("opal", i),
      url = cfg$urls[[i]],
      user = cfg$users[[i]],
      password = cfg$passwords[[i]],
      table = table_paths[[i]],
      driver = "OpalDriver"
    )
  }
  DSI::datashield.login(logins = builder$build(), assign = TRUE, symbol = "D")
}

run_central_baseline <- function(spec, data_csv, out_dir, python, baseline_script, seed) {
  out_json <- file.path(out_dir, paste0(spec$method, "_centralized.json"))
  params_json <- file.path(out_dir, paste0(spec$method, "_params.json"))
  jsonlite::write_json(spec$model_params, params_json,
                       auto_unbox = TRUE, pretty = TRUE, null = "null")
  args <- c(
    baseline_script,
    "--method", spec$method,
    "--data", data_csv,
    "--features", paste(spec$features, collapse = ","),
    "--target", paste(spec$target, collapse = ","),
    "--model-params-file", params_json,
    "--rounds", as.character(spec$rounds),
    "--seed", as.character(seed),
    "--output", out_json
  )
  res <- system2(python, args = args, stdout = TRUE, stderr = TRUE)
  status <- attr(res, "status") %||% 0L
  if (!identical(as.integer(status), 0L)) {
    return(list(status = "failed", error = paste(res, collapse = "\n")))
  }
  parsed <- jsonlite::fromJSON(out_json, simplifyVector = FALSE)
  parsed$status <- parsed$status %||% "ok"
  parsed
}

history_metric <- function(hist, name) {
  if (is.null(hist) || !NROW(hist)) return(NA_real_)
  if (name %in% names(hist)) {
    vals <- suppressWarnings(as.numeric(hist[[name]]))
    vals <- vals[is.finite(vals)]
    if (length(vals)) return(tail(vals, 1L))
  }
  if (all(c("metric", "value") %in% names(hist))) {
    idx <- tolower(hist$metric) == tolower(name)
    vals <- suppressWarnings(as.numeric(hist$value[idx]))
    vals <- vals[is.finite(vals)]
    if (length(vals)) return(tail(vals, 1L))
  }
  NA_real_
}

acceptance_params <- function(spec) {
  acc <- spec$acceptance %||% list()
  list(
    max_loss_ratio = as_scalar(acc$max_loss_ratio, 2.5),
    max_loss_margin = as_scalar(acc$max_loss_margin, 0.25)
  )
}

acceptable_federated_loss <- function(spec, central_loss, fed_loss,
                                      n_failures) {
  params <- acceptance_params(spec)
  if (!is.finite(central_loss) || !is.finite(fed_loss)) return(FALSE)
  failures <- suppressWarnings(as.numeric(n_failures))
  no_failures <- length(failures) == 0L || is.na(failures[[1L]]) ||
    failures[[1L]] <= 0
  no_failures &&
    fed_loss <= central_loss * params$max_loss_ratio + params$max_loss_margin
}

evaluate_xgboost_artifact <- function(output_dir, data_csv, features, target) {
  model_path <- file.path(output_dir, "global_model.json")
  if (!file.exists(model_path)) return(NULL)
  model <- jsonlite::fromJSON(model_path, simplifyVector = FALSE)
  if (!identical(model$model_type, "xgboost")) return(NULL)

  data <- utils::read.csv(data_csv, stringsAsFactors = FALSE)
  X <- as.matrix(data[, features, drop = FALSE])
  y <- as.numeric(data[[target[[1L]]]])
  raw <- numeric(nrow(X))
  learning_rate <- as_scalar(model$learning_rate, 0.3)

  for (tree in model$trees) {
    splits <- tree$splits %||% list()
    leaves <- tree$leaves %||% list()
    for (i in seq_len(nrow(X))) {
      node <- "0"
      repeat {
        split <- splits[[node]]
        if (is.null(split)) break
        feature_idx <- as.integer(split$feature) + 1L
        threshold <- as.numeric(split$threshold)
        node <- if (X[i, feature_idx] <= threshold) {
          as.character(split$left)
        } else {
          as.character(split$right)
        }
      }
      raw[[i]] <- raw[[i]] + learning_rate * as_scalar(leaves[[node]], 0)
    }
  }

  if (identical(model$objective %||% "binary:logistic", "binary:logistic")) {
    prob <- 1 / (1 + exp(-raw))
    prob <- pmin(pmax(prob, 1e-7), 1 - 1e-7)
    loss <- -mean(y * log(prob) + (1 - y) * log(1 - prob))
    accuracy <- mean(as.integer(prob > 0.5) == as.integer(y))
    return(list(loss = loss, accuracy = accuracy))
  }

  loss <- mean((raw - y)^2)
  list(loss = loss, accuracy = NA_real_)
}

run_federated_method <- function(spec, conns, privacy, secagg_supported,
                                 data_csv) {
  if (isTRUE(spec$requires_secagg) && !isTRUE(secagg_supported)) {
    return(list(
      status = "blocked_by_design",
      loss = NA_real_,
      accuracy = NA_real_,
      n_failures = NA_integer_,
      run_id = NA_character_,
      error = "Template requires server-side Secure Aggregation, but the current validation runtime reports no SecAgg support."
    ))
  }

  run <- tryCatch({
    ds.flower.fit(
      conns,
      symbol = "D",
      target = spec$target,
      features = spec$features,
      model = spec$model,
      model_params = spec$model_params,
      strategy = "fedavg",
      privacy = privacy,
      rounds = spec$rounds,
      task = spec$task,
      verbose = FALSE
    )
  }, error = function(e) e)

  if (inherits(run, "error")) {
    return(list(
      status = "failed",
      loss = NA_real_,
      accuracy = NA_real_,
      n_failures = NA_integer_,
      run_id = NA_character_,
      error = conditionMessage(run)
    ))
  }

  post_caps <- tryCatch(
    DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
    error = function(e) NULL
  )
  validation_error <- tryCatch({
    validate_run(run, post_caps)
    NULL
  }, error = function(e) conditionMessage(e))

  hist <- safe_history(run)
  loss <- history_metric(hist, "loss")
  if (!is.finite(loss)) loss <- history_metric(hist, "mse")
  accuracy <- history_metric(hist, "accuracy")
  n_failures <- history_metric(hist, "n_failures")
  if (!is.finite(loss) && identical(spec$model, "xgboost") &&
      !is.null(run$output_dir) && nzchar(run$output_dir)) {
    artifact_eval <- evaluate_xgboost_artifact(
      run$output_dir, data_csv, spec$features, spec$target
    )
    if (!is.null(artifact_eval)) {
      loss <- artifact_eval$loss
      accuracy <- artifact_eval$accuracy
      failures <- suppressWarnings(as.numeric(n_failures))
      if (length(failures) == 0L || !is.finite(failures[[1L]])) {
        n_failures <- 0L
      }
    }
  }

  list(
    status = if (is.null(validation_error)) "ok" else "failed",
    loss = loss,
    accuracy = accuracy,
    n_failures = n_failures,
    run_id = run$run_id %||% NA_character_,
    output_dir = run$output_dir %||% NA_character_,
    error = validation_error %||% NA_character_
  )
}

record_for_spec <- function(spec, central, federated) {
  central_status <- central$status %||% "unknown"
  central_loss <- as_scalar(central$loss)
  central_accuracy <- as_scalar(central$accuracy)
  fed_status <- federated$status %||% "unknown"
  fed_loss <- as_scalar(federated$loss)
  fed_accuracy <- as_scalar(federated$accuracy)
  fed_n_failures <- as_scalar(federated$n_failures)
  acceptance <- acceptance_params(spec)
  acceptable_loss <- acceptable_federated_loss(
    spec, central_loss, fed_loss, fed_n_failures
  )
  validation_status <- if (identical(central_status, "ok") &&
                           identical(fed_status, "ok") &&
                           isTRUE(acceptable_loss)) {
    "pass"
  } else if (identical(fed_status, "blocked_by_design")) {
    "blocked_by_design"
  } else {
    "failed"
  }

  data.frame(
    method = spec$method,
    task = spec$task,
    target = paste(spec$target, collapse = ","),
    n_features = length(spec$features),
    rounds = spec$rounds,
    centralized_status = central_status,
    centralized_metric = central$metric_name %||% NA_character_,
    centralized_loss = central_loss,
    centralized_accuracy = central_accuracy,
    federated_status = fed_status,
    federated_loss = fed_loss,
    federated_accuracy = fed_accuracy,
    federated_n_failures = fed_n_failures,
    federated_run_id = federated$run_id %||% NA_character_,
    delta_loss = if (is.finite(central_loss) && is.finite(fed_loss)) fed_loss - central_loss else NA_real_,
    acceptance_max_loss_ratio = acceptance$max_loss_ratio,
    acceptance_max_loss_margin = acceptance$max_loss_margin,
    acceptable_loss = acceptable_loss,
    validation_status = validation_status,
    notes = spec$notes %||% "",
    error = paste(na.omit(c(central$error %||% NA_character_, federated$error %||% NA_character_)), collapse = "\n"),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  cfg <- demo_config("method_validation", default_rounds = 1L)
  cfg$profile <- demo_env("DSFLOWER_METHOD_VALIDATION_PRIVACY",
                          demo_env("DSFLOWER_DEMO_PRIVACY_PROFILE", "sandbox_open"))
  cfg$table_prefix <- demo_env("DSFLOWER_METHOD_VALIDATION_TABLE_PREFIX",
                               paste0("method_validation_", format(Sys.time(), "%Y%m%d%H%M%S")))
  cfg$output_root <- demo_env("DSFLOWER_METHOD_VALIDATION_OUTPUT_ROOT",
                              file.path(getwd(), "dsflower_output", "method_validation"))

  n_per_site <- as.integer(demo_env("DSFLOWER_METHOD_VALIDATION_N_PER_SITE", 90L))
  specs <- selected_specs(method_specs())
  if (!length(specs)) stop("No method specs selected.", call. = FALSE)

  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_dir <- file.path(cfg$output_root, timestamp)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  data <- make_method_validation_data(n_per_site = n_per_site,
                                      n_sites = length(cfg$urls),
                                      seed = cfg$seed)
  data_csv <- file.path(out_dir, "pooled_validation_data.csv")
  utils::write.csv(data[, setdiff(names(data), "site"), drop = FALSE],
                   data_csv, row.names = FALSE)

  message("Uploading method-validation table to ", length(cfg$urls), " Opal nodes...")
  table_paths <- write_site_tables(cfg, data)
  conns <- connect_validation_tables(cfg, table_paths)
  on.exit(try(DSI::datashield.logout(conns), silent = TRUE), add = TRUE)

  caps <- tryCatch(DSI::datashield.aggregate(conns, expr = call("flowerGetCapabilitiesDS")),
                   error = function(e) NULL)
  secagg_supported <- !is.null(caps) && all(vapply(caps, function(x) {
    isTRUE(x$secure_aggregation_supported)
  }, logical(1)))

  python <- find_validation_python()
  baseline_script <- validation_python_script()

  records <- list()
  for (spec in specs) {
    message("Validating ", spec$method, "...")
    central <- run_central_baseline(spec, data_csv, out_dir, python,
                                    baseline_script, cfg$seed)
    federated <- run_federated_method(
      spec, conns, cfg$profile, secagg_supported, data_csv
    )
    records[[spec$method]] <- record_for_spec(spec, central, federated)
    utils::write.csv(do.call(rbind, records),
                     file.path(out_dir, "method_validation_results.csv"),
                     row.names = FALSE)
  }

  results <- do.call(rbind, records)
  rownames(results) <- NULL

  summary <- list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    demo_id = "method_validation",
    privacy_profile = cfg$profile,
    n_sites = length(cfg$urls),
    n_per_site = n_per_site,
    n_total = nrow(data),
    opal_urls = cfg$urls,
    table_paths = table_paths,
    secagg_supported = secagg_supported,
    python = python,
    results = results
  )

  result_json <- file.path(out_dir, "method_validation_results.json")
  jsonlite::write_json(summary, result_json,
                       auto_unbox = TRUE, pretty = TRUE, na = "null",
                       digits = NA)

  extdata_dir <- file.path(getwd(), "inst", "extdata")
  if (dir.exists("inst") || basename(getwd()) == "dsFlowerClient") {
    dir.create(extdata_dir, recursive = TRUE, showWarnings = FALSE)
    file.copy(result_json,
              file.path(extdata_dir, "dsflower_method_validation_results.json"),
              overwrite = TRUE)
  }

  print(results[, c("method", "centralized_loss", "federated_loss",
                    "delta_loss", "acceptable_loss", "validation_status")])
  message("Wrote method validation results to ", result_json)
  if (any(results$validation_status != "pass")) {
    stop("One or more method validation methods failed.", call. = FALSE)
  }
  invisible(summary)
}

if (identical(environment(), globalenv())) {
  main()
}
