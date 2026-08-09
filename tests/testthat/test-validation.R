validation_model_fixture <- function(track = "neural", loss = "bce_logits") {
  path <- withr::local_tempdir(.local_envir = parent.frame())
  spec <- list(
    kind = "sequential", layers = list(list(op = "linear", out = "@out")))
  meta <- list(
    track = track, model_spec = spec, loss_name = loss,
    data_kind = "tabular",
    model_params = list(n_classes = 2L, num_labels = 2L),
    features = c("age", "marker"),
    feature_lower = c(0, -5), feature_upper = c(120, 5),
    target_levels = if (identical(loss, "bce_logits")) c("control", "case") else NULL,
    target_bounds = if (loss %in% c("mse", "quantile"))
      list(lower = 0, upper = 100) else NULL)
  jsonlite::write_json(meta, file.path(path, "metadata.json"),
                       auto_unbox = TRUE, null = "null")
  writeBin(charToRaw("public-model"), file.path(path, "model.pt"))
  path
}

test_that("validation contract resolves a declarative neural artifact", {
  path <- validation_model_fixture()
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
  expect_identical(contract$track, "neural")
  expect_identical(contract$task, "binary")
  expect_identical(contract$bins, 24L)
  expect_identical(contract$features, c("age", "marker"))
  expect_equal(contract$feature_bounds$lower, c(0, -5))
  expect_identical(contract$target_levels, c("control", "case"))
})

test_that("validation treats quantile output as bounded regression", {
  path <- validation_model_fixture(loss = "quantile")
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
  expect_identical(contract$task, "regression")
  expect_identical(contract$loss_name, "quantile")
  expect_equal(contract$target_bounds, list(lower = 0, upper = 100))
})

test_that("validation rejects unsupported artifacts and public config early", {
  path <- validation_model_fixture()
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 3L),
    "bins must be one integer")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, "32"),
    "bins must be one integer")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, TRUE),
    "bins must be one integer")
  expect_error(
    ds.flower.validate(
      conns = list(), model = path, target = "outcome",
      torch_backend = "rocm"),
    "torch_backend")
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$track <- "egress"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "declarative neural artifacts")

  meta$track <- "trees"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "declarative neural artifacts")

  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$model_params$n_classes <- 3L
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "not a valid multiclass artifact")
})

test_that("validation rejects image artifacts before DSI", {
  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$data_kind <- "image"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")

  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "tabular artifacts only")
})

test_that("validation requires an explicit data kind contract", {
  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$data_kind <- NULL
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")

  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "invalid data_kind contract")
})

test_that("validation rejects corrupt public artifacts before DSI or staging", {
  path <- validation_model_fixture()
  reached_cli <- FALSE
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    .package = "dsFlowerClient")
  expect_error(
    ds.flower.validate(
      conns = list(site = TRUE), model = path, target = "outcome",
      symbol = "D"),
    "Saved model artifact preflight failed")
  expect_false(reached_cli)
})

test_that("validation metadata supports up to 1024 classes and labels", {
  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$loss_name <- "cross_entropy"
  meta$model_params$n_classes <- 1024L
  meta$model_params$num_labels <- 1024L
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 32L)
  expect_identical(contract$n_classes, 1024L)
  expect_identical(contract$n_labels, 1024L)

  meta$model_params$n_classes <- 1025L
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "2, 1024")
})

test_that("validation refuses mixed row/patient estimands", {
  expect_error(
    dsFlowerClient:::.validation_common_privacy_unit(list(
      row_site = list(privacy_unit = "row"),
      patient_site = list(privacy_unit = "patient"))),
    "same row/patient privacy unit")
})

test_that("validation contract SHA-256 has a cross-package canonical wire", {
  config <- list(
    "dp-track" = "validation", "validation-model-track" = "neural",
    "validation-task" = "multiclass", "validation-bins" = 24L,
    "task-type" = "classification", "loss-name" = "cross_entropy",
    "model-spec-b64" = "e30=", "num-server-rounds" = 1L,
    "num-features" = 2L, "num-classes" = 3L, "num-labels" = 2L,
    "feature-bounds" = list(lower = c(0, -5), upper = c(100, 5)),
    "target-levels" = c("a", "b", "c"))
  expect_identical(
    dsFlowerClient:::.validation_contract_sha256(
      config, c("age", "marker"), "outcome", "patient"),
    "ecf31e300ff0087e26799f6a4fbe1894e8f8357e0fd35667936653318b040e3f")
  changed <- config
  changed[["feature-bounds"]]$upper[[1L]] <- 101
  expect_false(identical(
    dsFlowerClient:::.validation_contract_sha256(
      changed, c("age", "marker"), "outcome", "patient"),
    dsFlowerClient:::.validation_contract_sha256(
      config, c("age", "marker"), "outcome", "patient")))
})

test_that("ds.flower.validate stages one validation release and returns pooled metrics", {
  path <- validation_model_fixture()
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(flwr_home = withr::local_tempdir())
  prepared <- NULL
  app_config <- NULL
  release_available <- TRUE

  local_mocked_bindings(
    .validate_validation_artifact_preflight = function(...) TRUE,
    .require_flwr_cli = function() TRUE,
    .validate_dsi_transport_security = function(...) TRUE,
    ds.flower.connect = function(conns, ...) structure(
      list(conns = conns, symbol = "validation_handle"),
      class = "dsflower_connection"),
    .assert_runner_compatibility = function(...) list(
      site_a = list(privacy_unit = "row"),
      site_b = list(privacy_unit = "row")),
    ds.flower.nodes.prepare = function(..., run_config) {
      prepared <<- run_config
      invisible(NULL)
    },
    .build_submission_app = function(sub, config_lines, ...) {
      app_config <<- config_lines
      withr::local_tempdir()
    },
    .ensure_client_framework = function(...) TRUE,
    ds.flower.link.up = function(...) TRUE,
    ds.flower.nodes.ensure = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., results_dir) {
      payload <- list(
        pooled_only = TRUE,
        privacy = "node-dp-pooled-postprocessing",
        task = "binary", n_nodes = 2L, available = release_available)
      if (release_available) {
        payload$metrics <- list(accuracy = 0.81, roc_auc = 0.87)
      }
      jsonlite::write_json(payload,
        file.path(results_dir, "validation.json"), auto_unbox = TRUE)
      list(status = 0L, stdout = "run_id=validation", stderr = "")
    },
    ds.flower.link.down = function(...) TRUE,
    ds.flower.nodes.cleanup = function(...) TRUE,
    ds.flower.disconnect = function(...) TRUE,
    .package = "dsFlowerClient")

  result <- ds.flower.validate(
    list(site_a = TRUE, site_b = TRUE), model = path,
    target = "outcome", symbol = "D", bins = 16L, silent = TRUE)
  expect_s3_class(result, "dsflower_validation")
  expect_true(result$pooled_only)
  expect_true(result$available)
  expect_equal(result$metrics$accuracy, 0.81)
  expect_identical(prepared[["dp-track"]], "validation")
  expect_identical(prepared[["num-server-rounds"]], 1L)
  expect_identical(prepared[["validation-bins"]], 16L)
  expect_match(prepared[["validation-contract-sha256"]], "^[0-9a-f]{64}$")
  expect_identical(prepared[["feature-bounds"]]$upper, c(120, 5))
  expect_true(any(grepl('dp-track = "validation"', app_config, fixed = TRUE)))
  expect_true(any(grepl('loss-name = "bce_logits"', app_config, fixed = TRUE)))
  expect_true(any(grepl(
    paste0('validation-contract-sha256 = "',
           prepared[["validation-contract-sha256"]], '"'),
    app_config, fixed = TRUE)))
  expect_true(any(grepl("num-server-rounds = 1", app_config, fixed = TRUE)))
  expect_false("per_node" %in% names(result))

  release_available <- FALSE
  unavailable <- ds.flower.validate(
    list(site_a = TRUE, site_b = TRUE), model = path,
    target = "outcome", symbol = "D", bins = 16L, silent = TRUE)
  expect_s3_class(unavailable, "dsflower_validation")
  expect_false(unavailable$available)
  expect_null(unavailable$metrics)
  expect_output(print(unavailable), "no complete private metric release")
})

test_that("pooled validation reader refuses a per-node result", {
  path <- withr::local_tempdir()
  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 1L, available = TRUE,
    metrics = list(accuracy = 0.5),
    per_node = list(site = list(accuracy = 1))),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_private_validation_result(path),
    "pooled-only privacy contract")

  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 1L, available = TRUE,
    metrics = list(accuracy = 0.5),
    statistics = list(histogram = c(1, 2))),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_private_validation_result(path),
    "pooled-only privacy contract")
})

test_that("unavailable validation is explicit and never contains fake metrics", {
  path <- withr::local_tempdir()
  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 2L, available = FALSE),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  result <- dsFlowerClient:::.read_private_validation_result(path)
  expect_false(result$available)
  expect_null(result$metrics)

  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 2L, available = FALSE,
    metrics = list(accuracy = 0)),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_private_validation_result(path),
    "pooled-only privacy contract")
})
