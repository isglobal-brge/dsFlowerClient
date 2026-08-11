.native_binary_holdout_metrics <- function() {
  list(
    n = 6, accuracy = 1, sensitivity = 1, specificity = 1,
    precision = 1, negative_predictive_value = 1, f1 = 1,
    balanced_accuracy = 1, roc_auc = 1, pr_auc = 1,
    brier = 0.015625, expected_calibration_error = 0.125,
    roc = list(
      fpr = c(0, 0, 0, 0, 1, 1),
      tpr = c(0, 1, 1, 1, 1, 1)),
    precision_recall = list(
      recall = c(0, 1, 1, 1, 1),
      precision = c(1, 1, 1, 1, 0.5)),
    calibration = list(
      predicted = c(0.125, 0.375, 0.625, 0.875),
      observed = c(0, 0, 0, 1),
      weight = c(3, 0, 0, 3)),
    decision_curve = list(
      threshold = c(0.125, 0.375, 0.625, 0.875),
      net_benefit = c(3 / 7, 0.5, 0.5, 0.5),
      treat_all = c(3 / 7, 0.2, -1 / 3, -3),
      treat_none = c(0, 0, 0, 0)))
}

.native_regression_holdout_metrics <- function() {
  list(
    n = 3, mae = 2 / 3, mse = 4 / 3,
    rmse = sqrt(4 / 3), r_squared = -1)
}

test_that("holdout fraction has one canonical seed-free contract", {
  spec <- dsFlowerClient:::.normalize_holdout(0.2)
  expect_identical(spec$test_millionths, 200000L)
  expect_identical(dsFlowerClient:::.normalize_holdout(NULL), NULL)
  expect_error(dsFlowerClient:::.normalize_holdout(0), "strictly between")
  expect_error(dsFlowerClient:::.normalize_holdout(1), "strictly between")
  expect_error(dsFlowerClient:::.normalize_holdout(c(0.2, 0.3)), "one")
  expect_error(dsFlowerClient:::.normalize_holdout(0.2000004), "six decimal")

  contract <- dsFlowerClient:::.holdout_contract(spec, "patient")
  expect_identical(contract$method, "holdout")
  expect_identical(contract$privacy_unit, "patient")
  expect_identical(
    contract$sha256,
    "00b0a490eb3d92fec7ce532e452523a32cbf73d19953372194faffc21eb4c75b")
  expect_false(any(grepl("seed|salt|nonce", names(contract), ignore.case = TRUE)))
})

test_that("fit forwards holdout for exact tabular model tracks", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )
  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y", features = "x",
    holdout = 0.2)
  expect_equal(seen$holdout, 0.2)

  expect_no_error(dsFlowerClient:::.assert_holdout_supported(
    list(track = "native_tree"), "tabular"))
  expect_error(dsFlowerClient:::.assert_holdout_supported(
    list(track = "egress"), "tabular"), "neural and native-tree")
  expect_error(
    dsFlowerClient:::.assert_holdout_supported(list(track = "neural"), "image"),
    "tabular"
  )
})

test_that("pooled holdout output is strict and contains no node transcript", {
  root <- withr::local_tempdir()
  jsonlite::write_json(list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "holdout",
    task = "binary",
    n_nodes = 2L,
    metrics = list(accuracy = 0.75, roc_auc = 0.8)
  ), file.path(root, "holdout.json"), auto_unbox = TRUE)

  value <- dsFlowerClient:::.read_holdout_result(root)
  expect_identical(value$method, "holdout")
  expect_false(any(c("per_node", "predictions", "folds") %in% names(value)))

  jsonlite::write_json(list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "holdout", task = "binary", n_nodes = 2L,
    metrics = list(accuracy = 0.75), per_node = list(site = 0.75)
  ), file.path(root, "holdout.json"), auto_unbox = TRUE)
  expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")

  jsonlite::write_json(list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "holdout", task = "binary", n_nodes = 2L,
    metrics = list(accuracy = 1.1)
  ), file.path(root, "holdout.json"), auto_unbox = TRUE)
  expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")

  jsonlite::write_json(list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "holdout", task = "binary", n_nodes = 2L,
    metrics = list(per_node = list(site = 0.75))
  ), file.path(root, "holdout.json"), auto_unbox = TRUE)
  expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")
})

test_that("native-tree holdout requires exact recursive-safe provenance", {
  root <- withr::local_tempdir()
  payload <- list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "holdout", task = "binary", n_nodes = 2L,
    metrics = .native_binary_holdout_metrics(),
    provenance = list(
      resampling_contract_sha256 = strrep("a", 64L),
      native_tree_request_sha256 = strrep("b", 64L),
      public_schema_sha256 = strrep("c", 64L),
      artifact_sha256 = strrep("d", 64L)))
  write_payload <- function(value) jsonlite::write_json(
    value, file.path(root, "holdout.json"), auto_unbox = TRUE, null = "null")

  write_payload(payload)
  value <- dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE)
  expect_identical(value$provenance$artifact_sha256, strrep("d", 64L))
  expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")

  bad <- payload
  bad$provenance$artifact_sha256 <- "not-a-sha"
  write_payload(bad)
  expect_error(
    dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE),
    "pooled-only")
  bad <- payload
  bad$provenance$artifact_sha256 <- NULL
  write_payload(bad)
  expect_error(
    dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE),
    "pooled-only")
  bad <- payload
  bad$provenance$extra_sha256 <- strrep("e", 64L)
  write_payload(bad)
  expect_error(
    dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE),
    "pooled-only")

  for (field in c(
      "transcript", "per_node", "per_site", "per_client", "predictions",
      "models", "fold", "folds", "per_fold", "fold_metrics")) {
    bad <- payload
    bad$metrics$diagnostics <- list(value = 1)
    bad$metrics$diagnostics[[field]] <- list(value = 2)
    write_payload(bad)
    expect_error(
      dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE),
      "pooled-only", info = field)
  }
})

test_that("native-tree holdout requires exact validation metric schemas", {
  root <- withr::local_tempdir()
  provenance <- list(
    resampling_contract_sha256 = strrep("a", 64L),
    native_tree_request_sha256 = strrep("b", 64L),
    public_schema_sha256 = strrep("c", 64L),
    artifact_sha256 = strrep("d", 64L))
  payload <- function(task, metrics) list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "holdout", task = task, n_nodes = 2L,
    metrics = metrics, provenance = provenance)
  write_payload <- function(value) jsonlite::write_json(
    value, file.path(root, "holdout.json"), auto_unbox = TRUE, null = "null")
  expect_rejected <- function(value, info) {
    write_payload(value)
    expect_error(
      dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE),
      "pooled-only", info = info)
  }

  binary <- .native_binary_holdout_metrics()
  write_payload(payload("binary", binary))
  parsed <- dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE)
  expect_identical(names(parsed$metrics), names(binary))
  expect_length(parsed$metrics$roc$fpr, 6L)
  expect_length(parsed$metrics$precision_recall$recall, 5L)
  expect_length(parsed$metrics$calibration$predicted, 4L)

  nullable_binary <- binary
  nullable_binary["roc_auc"] <- list(NULL)
  write_payload(payload("binary", nullable_binary))
  expect_null(dsFlowerClient:::.read_holdout_result(
    root, native_tree = TRUE)$metrics$roc_auc)

  binary_adversarial <- list()
  binary_adversarial$extra_metric <- binary
  binary_adversarial$extra_metric$diagnostics <- list(reason = "private")
  binary_adversarial$missing_metric <- binary
  binary_adversarial$missing_metric$brier <- NULL
  binary_adversarial$list_scalar <- binary
  binary_adversarial$list_scalar$accuracy <- list(1)
  binary_adversarial$nested_extra <- binary
  binary_adversarial$nested_extra$roc$diagnostics <- 0
  binary_adversarial$wrong_curve_length <- binary
  binary_adversarial$wrong_curve_length$roc$fpr <- c(0, 0, 0, 1, 1)
  binary_adversarial$named_curve <- binary
  binary_adversarial$named_curve$calibration$predicted <- structure(
    as.list(binary$calibration$predicted),
    names = paste0("bin", seq_along(binary$calibration$predicted)))
  binary_adversarial$nested_list_value <- binary
  binary_adversarial$nested_list_value$decision_curve$threshold[[1L]] <- list(0.125)
  binary_adversarial$too_few_bins <- binary
  for (field in names(binary_adversarial$too_few_bins$calibration)) {
    binary_adversarial$too_few_bins$calibration[[field]] <-
      binary_adversarial$too_few_bins$calibration[[field]][1:3]
  }
  for (field in names(binary_adversarial$too_few_bins$decision_curve)) {
    binary_adversarial$too_few_bins$decision_curve[[field]] <-
      binary_adversarial$too_few_bins$decision_curve[[field]][1:3]
  }
  binary_adversarial$too_few_bins$roc <- lapply(
    binary_adversarial$too_few_bins$roc, function(x) x[1:5])
  binary_adversarial$too_few_bins$precision_recall <- lapply(
    binary_adversarial$too_few_bins$precision_recall, function(x) x[1:4])
  for (case in names(binary_adversarial)) {
    expect_rejected(
      payload("binary", binary_adversarial[[case]]), case)
  }
  expect_rejected(payload("binary", unname(binary)), "unnamed metrics")

  regression <- .native_regression_holdout_metrics()
  write_payload(payload("regression", regression))
  parsed <- dsFlowerClient:::.read_holdout_result(root, native_tree = TRUE)
  expect_identical(names(parsed$metrics), names(regression))
  nullable_regression <- regression
  nullable_regression["r_squared"] <- list(NULL)
  write_payload(payload("regression", nullable_regression))
  expect_null(dsFlowerClient:::.read_holdout_result(
    root, native_tree = TRUE)$metrics$r_squared)

  regression_extra <- regression
  regression_extra$diagnostics <- list(residuals = c(0.1, 0.2))
  expect_rejected(payload("regression", regression_extra), "regression extra")
  regression_missing <- regression
  regression_missing$rmse <- NULL
  expect_rejected(
    payload("regression", regression_missing), "regression missing")
  regression_list <- regression
  regression_list$mae <- list(2 / 3)
  expect_rejected(payload("regression", regression_list), "regression list")
  regression_null <- regression
  regression_null["mae"] <- list(NULL)
  expect_rejected(payload("regression", regression_null), "regression null")
  expect_rejected(
    payload("regression", unname(regression)), "unnamed regression")
  expect_rejected(payload("count", regression), "native count")
  expect_rejected(payload("multiclass", binary), "native multiclass")
})

test_that("atomic native holdout commits the pinned ensemble, not model.pt", {
  root <- withr::local_tempdir()
  bytes <- charToRaw("bounded-native-ensemble")
  artifact <- list(
    file = "model.xgboost-ensemble.json",
    format = "dsflower-xgboost-ensemble-json-v1",
    size_bytes = as.integer(length(bytes)),
    sha256 = digest::digest(bytes, algo = "sha256", serialize = FALSE))
  writeBin(bytes, file.path(root, artifact$file))
  writeLines("{}", file.path(
    root, "model.xgboost-ensemble.profile.json"))
  writeLines("{}", file.path(root, "holdout.json"))
  jsonlite::write_json(
    list(list(round = 1L, available = TRUE)),
    file.path(root, "history.json"), auto_unbox = TRUE)
  history <- data.frame(round = 1L, available = TRUE)
  release <- list(artifact = artifact)

  expect_true(dsFlowerClient:::.atomic_holdout_commit_complete(
    history, 1L, root, native_tree = TRUE, native_release = release))

  bad_histories <- list(
    extra_list_field = list(list(
      round = 1L, available = TRUE, diagnostics = list(0.11, 0.22))),
    unnamed_entry = list(list(1L, TRUE)),
    extra_entry = list(
      list(round = 1L, available = TRUE),
      list(round = 1L, available = TRUE)))
  for (case in names(bad_histories)) {
    jsonlite::write_json(
      bad_histories[[case]], file.path(root, "history.json"),
      auto_unbox = TRUE)
    expect_false(dsFlowerClient:::.atomic_holdout_commit_complete(
      history, 1L, root, native_tree = TRUE, native_release = release),
      info = case)
  }
  jsonlite::write_json(
    list(list(round = 1L, available = TRUE)),
    file.path(root, "history.json"), auto_unbox = TRUE)
  tampered <- bytes
  tampered[[1L]] <- as.raw(bitwXor(as.integer(tampered[[1L]]), 1L))
  expect_length(tampered, length(bytes))
  writeBin(tampered, file.path(root, artifact$file))
  expect_false(dsFlowerClient:::.atomic_native_tree_model_exists(
    root, artifact))
  expect_false(dsFlowerClient:::.atomic_holdout_commit_complete(
    history, 1L, root, native_tree = TRUE, native_release = release))

  writeBin(bytes, file.path(root, artifact$file))
  file.create(file.path(root, "model.pt"))
  expect_false(dsFlowerClient:::.atomic_holdout_commit_complete(
    history, 1L, root, native_tree = TRUE, native_release = release))
})

test_that("native run accepts only holdout provenance bound to its artifact", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(
    process = list(is_alive = function() TRUE),
    flwr_home = withr::local_tempdir())

  holdout_contract <- dsFlowerClient:::.holdout_contract(
    dsFlowerClient:::.normalize_holdout(0.2), "row")
  request_sha <- strrep("b", 64L)
  schema_sha <- strrep("c", 64L)
  artifact_bytes <- charToRaw("bounded-native-ensemble")
  artifact <- list(
    file = "model.xgboost-ensemble.json",
    format = "dsflower-xgboost-ensemble-json-v1",
    size_bytes = as.integer(length(artifact_bytes)),
    sha256 = digest::digest(
      artifact_bytes, algo = "sha256", serialize = FALSE))
  recipe <- structure(list(
    model = list(
      name = "xgboost", framework = "xgboost", track = "native_tree",
      engine = "xgboost", task = "binary"),
    strategy = list(name = "mean_prediction", params = list()),
    num_rounds = 1L, data_kind = "tabular", features = "x",
    feature_lower = 0, feature_upper = 1,
    target_levels = c("control", "case"), target_bounds = NULL,
    native_tree_request_b64 = "public-request",
    native_tree_request_sha256 = request_sha,
    public_schema_sha256 = schema_sha,
    holdout_contract = holdout_contract,
    cross_validation_contract = NULL), class = "dsflower_recipe")
  payload <- list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "holdout", task = "binary", n_nodes = 2L,
    metrics = .native_binary_holdout_metrics(),
    provenance = list(
      resampling_contract_sha256 = holdout_contract$sha256,
      native_tree_request_sha256 = request_sha,
      public_schema_sha256 = schema_sha,
      artifact_sha256 = artifact$sha256))
  output_root <- withr::local_tempdir()
  extra_result <- FALSE
  copy_ok <- TRUE

  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., results_dir) {
      jsonlite::write_json(
        data.frame(round = 1L, available = TRUE),
        file.path(results_dir, "history.json"), auto_unbox = TRUE)
      writeBin(artifact_bytes, file.path(results_dir, artifact$file))
      writeLines("{}", file.path(
        results_dir, "model.xgboost-ensemble.profile.json"))
      jsonlite::write_json(
        payload, file.path(results_dir, "holdout.json"), auto_unbox = TRUE)
      if (extra_result) {
        writeLines("{}", file.path(results_dir, "per_node.json"))
      }
      list(status = 0L, stdout = "run_id=native-holdout", stderr = "")
    },
    .native_tree_release_metadata = function(...) list(
      artifact = artifact,
      sanitization = dsFlowerClient:::.native_tree_sanitization_attestation(
        "xgboost")),
    .copy_native_holdout_result_files = function(
        results_dir, output_dir, files) {
      if (!copy_ok) return(FALSE)
      copied <- base::file.copy(
        file.path(results_dir, files), file.path(output_dir, files),
        overwrite = FALSE)
      length(copied) == length(files) && all(!is.na(copied) & copied)
    },
    .package = "dsFlowerClient")

  run <- ds.flower.run.start(
    recipe, conns = list(site1 = TRUE, site2 = TRUE),
    app_dir = withr::local_tempdir(), results_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "complete", silent = TRUE)
  expect_equal(run$holdout$accuracy, 1)
  expect_true(file.exists(file.path(
    run$output_dir, "model.xgboost-ensemble.json")))
  expect_true(file.exists(file.path(
    run$output_dir, "model.xgboost-ensemble.profile.json")))
  expect_false(file.exists(file.path(run$output_dir, "model.pt")))

  nested <- ds.flower.run.start(
    recipe, conns = list(site1 = TRUE, site2 = TRUE),
    app_dir = withr::local_tempdir(), results_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "nested/complete", silent = TRUE)
  expect_identical(
    normalizePath(nested$output_dir, winslash = "/", mustWork = TRUE),
    normalizePath(file.path(output_root, "nested", "complete"),
                  winslash = "/", mustWork = TRUE))

  extra_result <- TRUE
  expect_error(ds.flower.run.start(
    recipe, conns = list(site1 = TRUE, site2 = TRUE),
    app_dir = withr::local_tempdir(), results_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "extra-result", silent = TRUE),
    "commit marker")
  expect_false(dir.exists(file.path(output_root, "extra-result")))
  extra_result <- FALSE

  occupied <- file.path(output_root, "occupied")
  dir.create(occupied)
  writeLines("keep", file.path(occupied, "sentinel"))
  expect_error(ds.flower.run.start(
    recipe, conns = list(site1 = TRUE, site2 = TRUE),
    app_dir = withr::local_tempdir(), results_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "occupied", silent = TRUE),
    "must not already exist")
  expect_true(file.exists(file.path(occupied, "sentinel")))

  copy_ok <- FALSE
  expect_error(ds.flower.run.start(
    recipe, conns = list(site1 = TRUE, site2 = TRUE),
    app_dir = withr::local_tempdir(), results_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "copy-failure", silent = TRUE),
    "exact native-tree holdout result set")
  expect_false(dir.exists(file.path(output_root, "copy-failure")))
  copy_ok <- TRUE

  original <- payload
  for (field in c(
      "resampling_contract_sha256", "native_tree_request_sha256",
      "public_schema_sha256", "artifact_sha256", "n_nodes", "task")) {
    payload <- original
    if (identical(field, "n_nodes")) {
      payload$n_nodes <- 3L
    } else if (identical(field, "task")) {
      payload$task <- "regression"
      payload$metrics <- .native_regression_holdout_metrics()
    } else {
      payload$provenance[[field]] <- strrep("f", 64L)
    }
    expect_error(ds.flower.run.start(
      recipe, conns = list(site1 = TRUE, site2 = TRUE),
      app_dir = withr::local_tempdir(), results_dir = withr::local_tempdir(),
      output_dir = output_root, output_name = paste0("bad-", field),
      silent = TRUE), "public provenance", info = field)
  }
})

test_that("run accepts model and metrics together and rejects a missing release", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(
    process = list(is_alive = function() TRUE),
    flwr_home = withr::local_tempdir())
  recipe <- ds.flower.recipe(
    model = ds.flower.model.pytorch_logreg(), num_rounds = 1L,
    features = "x")
  recipe$model$track <- "neural"
  recipe$holdout_contract <- dsFlowerClient:::.holdout_contract(
    dsFlowerClient:::.normalize_holdout(0.2), "row")
  release <- list(metrics = list(accuracy = 0.75))
  training_history <- data.frame(
    round = 1L, n_failures = 0L, available = TRUE)
  output_root <- withr::local_tempdir()

  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .ensure_client_framework = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(...) list(
      status = 0L, stdout = "run_id=atomic-holdout", stderr = ""),
    .read_model_weights = function(...) list(coef = 1),
    .read_training_history = function(...) training_history,
    .read_holdout_result = function(...) release,
    .atomic_neural_model_exists = function(...) TRUE,
    .package = "dsFlowerClient")

  run <- ds.flower.run.start(
    recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "complete", silent = TRUE)
  expect_equal(run$holdout$accuracy, 0.75)
  saved <- readRDS(run$saved_path)
  expect_equal(saved$holdout$accuracy, 0.75)
  expect_match(paste(capture.output(print(run)), collapse = "\n"),
               "pooled DP metrics")

  training_history <- NULL
  expect_error(
    ds.flower.run.start(
      recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
      output_dir = output_root, output_name = "uncommitted", silent = TRUE),
    "commit marker")
  expect_false(dir.exists(file.path(output_root, "uncommitted")))

  training_history <- data.frame(
    round = 1L, n_failures = 0L, available = TRUE)
  release <- NULL
  expect_error(
    ds.flower.run.start(
      recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
      output_dir = output_root, output_name = "missing", silent = TRUE),
    "no pooled test metric release")
  expect_false(dir.exists(file.path(output_root, "missing")))
})
