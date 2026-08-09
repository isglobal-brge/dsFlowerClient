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

xgboost_validation_model_fixture <- function(task = "binary") {
  path <- withr::local_tempdir(.local_envir = parent.frame())
  artifact_name <- "model.xgboost-ensemble.json"
  model <- ds.flower.model.xgboost(task = task)
  request <- dsFlowerClient:::.build_xgboost_request(
    model$params, c("age", "marker"),
    list(lower = c(0, -5), upper = c(120, 5)),
    list(c(18, 40, 65), c(-1, 0, 1)),
    target_name = "outcome",
    target_levels = if (identical(task, "binary")) c("control", "case") else NULL,
    target_bounds = if (identical(task, "regression")) {
      list(lower = 0, upper = 100)
    } else NULL)
  public_schema_sha256 <- request$value$public_schema$sha256
  container <- list(
    contract = "dsflower-xgboost-ensemble-v1", version = 1L,
    engine = "xgboost", task = task, aggregation = "mean_prediction",
    public_schema_sha256 = public_schema_sha256,
    models = list(list(learner = list(attributes = list()),
                       version = list(3L, 4L, 0L))))
  artifact_path <- file.path(path, artifact_name)
  jsonlite::write_json(container, artifact_path, auto_unbox = TRUE,
                       null = "null")
  bytes <- readBin(artifact_path, "raw", n = file.info(artifact_path)$size)
  artifact_hash <- digest::digest(bytes, algo = "sha256", serialize = FALSE)
  sanitization <- list(
    profile = "dsflower-xgboost-ensemble-v1",
    privacy_basis = "direct-dp-training-postprocessing",
    contains_raw_records = FALSE,
    contains_unnoised_statistics = FALSE,
    contains_feature_names = FALSE,
    contains_target_name = FALSE,
    contains_training_history = FALSE,
    contains_backend_logs = FALSE,
    contains_paths = FALSE,
    contains_executable_payload = FALSE)
  meta <- list(
    track = "native_tree", engine = "xgboost", task = task,
    data_kind = "tabular", features = c("age", "marker"),
    feature_lower = c(0, -5), feature_upper = c(120, 5),
    target_levels = if (identical(task, "binary")) c("control", "case") else NULL,
    target_bounds = if (identical(task, "regression"))
      list(lower = 0, upper = 100) else NULL,
    native_tree_request_b64 = request$b64,
    native_tree_request_sha256 = request$sha256,
    public_schema_sha256 = public_schema_sha256,
    artifact = list(
      file = artifact_name,
      format = "dsflower-xgboost-ensemble-json-v1",
      size_bytes = as.integer(length(bytes)), sha256 = artifact_hash),
    sanitization = sanitization)
  jsonlite::write_json(meta, file.path(path, "metadata.json"),
                       auto_unbox = TRUE, null = "null")
  profile <- list(
    artifact = list(
      format = "dsflower-xgboost-ensemble-json-v1",
      sha256 = artifact_hash, size_bytes = as.integer(length(bytes))),
    contract = "dsflower-xgboost-prediction-profile-v1",
    native_tree_request_b64 = request$b64,
    native_tree_request_sha256 = request$sha256,
    public_schema_sha256 = public_schema_sha256,
    task = task,
    version = 1L)
  writeBin(
    dsFlowerClient:::.native_tree_json(profile),
    file.path(path, "model.xgboost-ensemble.profile.json"))
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

test_that("validation resolves only a sanitized XGBoost ensemble contract", {
  path <- xgboost_validation_model_fixture()
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
  expect_identical(contract$track, "native_tree")
  expect_identical(contract$engine, "xgboost")
  expect_identical(contract$task, "binary")
  expect_identical(contract$data_kind, "tabular")
  expect_identical(
    contract$artifact_format, "dsflower-xgboost-ensemble-json-v1")
  expect_match(contract$artifact_sha256, "^[0-9a-f]{64}$")
  expect_match(contract$native_tree_request_sha256, "^[0-9a-f]{64}$")
  expect_identical(
    rawToChar(jsonlite::base64_dec(contract$native_tree_request_b64)),
    dsFlowerClient:::.validate_native_tree_request_wire(
      contract$native_tree_request_b64,
      contract$native_tree_request_sha256)$json)
  expect_match(contract$public_schema_sha256, "^[0-9a-f]{64}$")
  expect_identical(contract$loss_name, "bce_logits")

  regression <- dsFlowerClient:::.resolve_validation_contract(
    xgboost_validation_model_fixture("regression"), 32L)
  expect_identical(regression$task, "regression")
  expect_equal(regression$target_bounds, list(lower = 0, upper = 100))
  expect_identical(regression$loss_name, "mse")
})

test_that("prediction uses the exact XGBoost contract without provisioning torch", {
  path <- xgboost_validation_model_fixture()
  expect_true(file.exists(file.path(
    path, "model.xgboost-ensemble.profile.json")))
  resolved <- dsFlowerClient:::.resolve_model_for_predict(path)
  expect_identical(resolved$framework, "xgboost")
  expect_identical(basename(resolved$model_file),
                   "model.xgboost-ensemble.json")
  reached_framework <- FALSE
  local_mocked_bindings(
    .ensure_client_framework = function(...) {
      reached_framework <<- TRUE
      stop("must not provision torch")
    },
    .run_xgboost_local_predict = function(native_contract, frame, type) {
      expect_identical(native_contract$native_tree_request_sha256,
                       resolved$native_contract$native_tree_request_sha256)
      expect_identical(names(frame), c("age", "marker"))
      expect_identical(type, "response")
      c(0.49, 0.5)
    },
    .package = "dsFlowerClient")
  expect_identical(
    ds.flower.predict(
      path, data.frame(marker = c(0, 1), age = c(40, 65))),
    c("control", "case"))
  expect_false(reached_framework)
})

test_that("XGBoost local prediction validates columns, task and output", {
  expect_identical(
    names(dsFlowerClient:::.xgboost_prediction_frame(
      data.frame(marker = 0, age = 40), c("age", "marker"))),
    c("age", "marker"))
  expect_error(
    dsFlowerClient:::.xgboost_prediction_frame(
      data.frame(age = 40), c("age", "marker")),
    "missing: marker")
  expect_error(
    dsFlowerClient:::.xgboost_prediction_frame(
      data.frame(age = 40, marker = "x"), c("age", "marker")),
    "numeric or logical")

  path <- xgboost_validation_model_fixture("regression")
  expect_error(
    ds.flower.predict(
      path, data.frame(age = 40, marker = 0), type = "prob"),
    "unavailable for XGBoost regression")

  output <- tempfile(fileext = ".json")
  on.exit(unlink(output), add = TRUE)
  writeBin(charToRaw(paste0(
    '{"contract":"dsflower-xgboost-local-prediction-v1",',
    '"predictions":[0.25,0.75],"task":"binary",',
    '"type":"prob","version":1}')), output)
  expect_equal(
    dsFlowerClient:::.read_xgboost_prediction_output(
      output, 2L, "binary", "prob"),
    c(0.25, 0.75))
})

test_that("native release requires the exact bounded canonical profile sidecar", {
  expect_identical(
    dsFlowerClient:::.XGBOOST_PREDICTION_PROFILE_MAX_BYTES, 128 * 1024L)
  path <- xgboost_validation_model_fixture()
  meta <- jsonlite::fromJSON(
    file.path(path, "metadata.json"), simplifyVector = TRUE)
  recipe <- list(
    model = list(task = "binary"),
    native_tree_request_b64 = meta$native_tree_request_b64,
    native_tree_request_sha256 = meta$native_tree_request_sha256,
    public_schema_sha256 = meta$public_schema_sha256)
  expect_no_error(dsFlowerClient:::.native_xgboost_release_metadata(
    recipe, path))
  profile_path <- file.path(path, "model.xgboost-ensemble.profile.json")
  writeBin(charToRaw("opaque-public-profile"),
           profile_path)
  expect_error(
    dsFlowerClient:::.native_xgboost_release_metadata(recipe, path),
    "exact contract")
  writeBin(raw(128 * 1024L), profile_path)
  expect_error(
    dsFlowerClient:::.native_xgboost_release_metadata(recipe, path),
    "exact contract")
  writeBin(raw(128 * 1024L + 1L), profile_path)
  expect_error(
    dsFlowerClient:::.native_xgboost_release_metadata(recipe, path),
    "outside its byte bound")
  unlink(profile_path)
  expect_error(
    dsFlowerClient:::.native_xgboost_release_metadata(recipe, path),
    "missing or outside its byte bound")
})

test_that("portable save revalidates and copies only the exact XGBoost sidecar", {
  path <- xgboost_validation_model_fixture()
  writeBin(charToRaw("must-not-copy"), file.path(
    path, "model.xgboost-ensemble.profile.json.bak"))
  run <- structure(list(
    model_id = "native", weights = NULL, available = TRUE,
    history = data.frame(round = 1L), model = "xgboost",
    strategy = "mean_prediction", num_rounds = 1L, run_id = "run-native",
    output_dir = path), class = "dsflower_run")
  destination <- file.path(withr::local_tempdir(), "native.rds")
  expect_identical(
    ds.flower.save_model(run, destination),
    normalizePath(destination, winslash = "/", mustWork = TRUE))
  assets <- paste0(destination, ".assets")
  expect_true(file.exists(file.path(
    assets, "model.xgboost-ensemble.json")))
  expect_true(file.exists(file.path(
    assets, "model.xgboost-ensemble.profile.json")))
  expect_false(file.exists(file.path(
    assets, "model.xgboost-ensemble.profile.json.bak")))

  path <- xgboost_validation_model_fixture()
  profile_path <- file.path(path, "model.xgboost-ensemble.profile.json")
  bytes <- readBin(profile_path, "raw", n = file.info(profile_path)$size)
  writeBin(c(bytes, charToRaw("\n")), profile_path)
  run$output_dir <- path
  expect_error(
    ds.flower.save_model(
      run, file.path(withr::local_tempdir(), "tampered.rds")),
    "not canonically encoded")
})

test_that("XGBoost validation fails closed before capability, DSI, or private IO", {
  expect_identical(
    dsFlowerClient:::.XGBOOST_ENSEMBLE_MAX_BYTES, 64 * 1024^2)
  path <- xgboost_validation_model_fixture()
  reached_cli <- FALSE
  reached_dsi <- FALSE
  local_mocked_bindings(
    .validate_dsi_transport_security = function(...) {
      reached_dsi <<- TRUE
      stop("transport must not be reached")
    },
    .assert_native_xgboost_capability = function(...) {
      reached_dsi <<- TRUE
      stop("capability DSI must not be reached")
    },
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    ds.flower.connect = function(...) {
      reached_dsi <<- TRUE
      stop("private connection must not be reached")
    },
    .package = "dsFlowerClient")
  expect_error(
    ds.flower.validate(
      conns = list(site = TRUE), model = path, target = "outcome",
      symbol = "D"),
    "recognized but fail-closed")
  expect_false(reached_cli)
  expect_false(reached_dsi)

  path <- xgboost_validation_model_fixture()
  profile_path <- file.path(path, "model.xgboost-ensemble.profile.json")
  profile_bytes <- readBin(
    profile_path, what = "raw", n = file.info(profile_path)$size)
  writeBin(c(profile_bytes, charToRaw("\n")), profile_path)
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "not canonically encoded")
  expect_error(
    dsFlowerClient:::.resolve_model_for_predict(path),
    "not canonically encoded")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$artifact$sha256 <- strrep("0", 64L)
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "SHA-256 mismatch")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$artifact$size_bytes <- 64 * 1024^2 + 1
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "size is outside")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$sanitization$contains_backend_logs <- TRUE
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "sanitization attestation")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$public_schema_sha256 <- strrep("c", 64L)
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "request differs|public schema SHA-256 mismatch")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$target_bounds <- list(lower = 0, upper = 1)
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "two public target levels and no regression target bounds")

  path <- xgboost_validation_model_fixture("regression")
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$task <- "multiclass"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "XGBoost binary or regression only")
  expect_false(reached_cli)
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
