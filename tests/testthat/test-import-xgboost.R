external_xgboost_import_fixture <- function(
    task = "binary", parent = withr::local_tempdir(
      .local_envir = parent.frame()), output = NULL) {
  artifact <- testthat::test_path(
    "fixtures", paste0("xgboost-3.4-", task, ".json"))
  if (is.null(output)) {
    output <- file.path(parent, paste0(task, "-bundle"))
  }
  model <- ds.flower.model.xgboost(
    task = task, n_estimators = 1L, max_depth = 1L,
    learning_rate = 0.25)
  path <- ds.flower.import_xgboost(
    artifact = artifact, model = model,
    features = c("age", "marker"),
    feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
    feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
    target = "outcome",
    target_levels = if (identical(task, "binary")) {
      c("control", "case")
    } else {
      NULL
    },
    target_bounds = if (identical(task, "regression")) {
      list(lower = -10, upper = 10)
    } else {
      NULL
    },
    output_dir = output)
  list(path = path, artifact = artifact, output = output)
}

write_external_xgboost_metadata <- function(path, mutate) {
  metadata_path <- file.path(path, "metadata.json")
  value <- jsonlite::fromJSON(metadata_path, simplifyVector = FALSE)
  value <- mutate(value)
  writeBin(dsFlowerClient:::.native_tree_json(value), metadata_path)
}

test_that("external XGBoost JSON imports, predicts, and preserves provenance", {
  for (task in c("binary", "regression")) {
    withr::local_tempdir()
    imported <- external_xgboost_import_fixture(
      task, parent = withr::local_tempdir(.local_envir = environment()))
    expect_identical(
      sort(list.files(imported$path, all.files = FALSE)),
      c("metadata.json", "model.xgboost-ensemble.json",
        "model.xgboost-ensemble.profile.json"))
    contract <- dsFlowerClient:::.resolve_validation_contract(
      imported$path, 32L)
    expect_identical(contract$engine, "xgboost")
    expect_identical(contract$task, task)
    expect_identical(
      contract$model_training_privacy, "external-unverified")
    expect_identical(
      contract$sanitization$privacy_basis, "external-unverified")
    expect_identical(
      contract$sanitization$contains_unnoised_statistics, TRUE)
    expect_no_error(
      dsFlowerClient:::.validate_validation_artifact_preflight(contract))

    newdata <- data.frame(
      marker = c(0, 0), age = c(0, 20), check.names = FALSE)
    if (identical(task, "binary")) {
      expect_equal(
        ds.flower.predict(imported$path, newdata, type = "prob"),
        c(1 / (1 + exp(0.1)), 1 / (1 + exp(-0.1))),
        tolerance = 1e-7)
      expect_identical(
        ds.flower.predict(imported$path, newdata, type = "response"),
        c("control", "case"))
    } else {
      expect_equal(
        ds.flower.predict(imported$path, newdata, type = "response"),
        c(-0.1, 0.1), tolerance = 1e-7)
    }
  }
})

test_that("external import preserves canonical Unicode public schema", {
  parent <- withr::local_tempdir()
  path <- ds.flower.import_xgboost(
    artifact = testthat::test_path(
      "fixtures", "xgboost-3.4-binary.json"),
    model = ds.flower.model.xgboost(
      task = "binary", n_estimators = 1L, max_depth = 1L,
      learning_rate = 0.25),
    features = c("âge", "marcador😀"),
    feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
    feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
    target = "résultat", target_levels = c("contrôle", "cas😀"),
    output_dir = file.path(parent, "unicode-bundle"))
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 32L)
  expect_identical(contract$features, c("âge", "marcador😀"))
  expect_identical(unname(contract$target_levels), c("contrôle", "cas😀"))
  expect_no_error(
    dsFlowerClient:::.validate_validation_artifact_preflight(contract))
})

test_that("local prediction validates the bundle before touching newdata", {
  imported <- external_xgboost_import_fixture()
  contract <- dsFlowerClient:::.resolve_validation_contract(
    imported$path, 32L)
  touched <- FALSE
  local_mocked_bindings(
    .validate_validation_artifact_preflight = function(...) {
      stop("injected model preflight failure")
    },
    .native_tree_prediction_frame = function(...) {
      touched <<- TRUE
      stop("newdata was touched")
    },
    .package = "dsFlowerClient")
  expect_error(
    dsFlowerClient:::.predict_native_tree_local(
      list(native_contract = contract), data.frame(secret = 1), "prob"),
    "injected model preflight failure")
  expect_false(touched)
})

test_that("private validation reports external model provenance separately", {
  imported <- external_xgboost_import_fixture()
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(flwr_home = withr::local_tempdir())

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
    ds.flower.nodes.prepare = function(...) invisible(NULL),
    .build_submission_app = function(...) withr::local_tempdir(),
    ds.flower.link.up = function(...) TRUE,
    ds.flower.nodes.ensure = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., results_dir) {
      jsonlite::write_json(list(
        pooled_only = TRUE,
        privacy = "node-dp-pooled-postprocessing",
        task = "binary", n_nodes = 2L, available = TRUE,
        metrics = list(accuracy = 0.75, roc_auc = 0.8)),
        file.path(results_dir, "validation.json"), auto_unbox = TRUE)
      list(status = 0L, stdout = "run_id=external-validation", stderr = "")
    },
    ds.flower.link.down = function(...) TRUE,
    ds.flower.nodes.cleanup = function(...) TRUE,
    ds.flower.disconnect = function(...) TRUE,
    .package = "dsFlowerClient")

  result <- ds.flower.validate(
    list(site_a = TRUE, site_b = TRUE), model = imported$path,
    target = "outcome", symbol = "D", bins = 16L, silent = TRUE)
  expect_identical(result$privacy, "node-dp-pooled-postprocessing")
  expect_identical(result$model_training_privacy, "external-unverified")
  expect_output(
    print(result), "external training privacy unverified", fixed = TRUE)
})

test_that("external import metadata cannot claim direct DP training", {
  imported <- external_xgboost_import_fixture()
  metadata <- jsonlite::fromJSON(
    file.path(imported$path, "metadata.json"), simplifyVector = FALSE)
  expect_identical(
    sort(names(metadata)),
    sort(c(
      "artifact", "contract", "data_kind", "engine", "feature_lower",
      "feature_upper", "features", "native_tree_request_b64",
      "native_tree_request_sha256", "prediction_profile",
      "public_schema_sha256", "sanitization", "target_bounds",
      "target_levels", "task", "track", "version")))
  expect_identical(
    metadata$contract,
    dsFlowerClient:::.EXTERNAL_XGBOOST_IMPORT_CONTRACT)
  expect_identical(metadata$version, 1L)

  write_external_xgboost_metadata(imported$path, function(value) {
    value$sanitization$privacy_basis <-
      "direct-dp-training-postprocessing"
    value
  })
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(imported$path, 32L),
    "sanitization attestation")
})

test_that("external ensemble cannot be relabeled as an internal DP release", {
  imported <- external_xgboost_import_fixture()
  write_external_xgboost_metadata(imported$path, function(value) {
    value$contract <- NULL
    value$prediction_profile <- NULL
    value$version <- NULL
    value$sanitization$profile <-
      dsFlowerClient:::.native_tree_release_spec("xgboost")$artifact_contract
    value$sanitization$privacy_basis <-
      "direct-dp-training-postprocessing"
    value$sanitization$contains_unnoised_statistics <- FALSE
    value
  })
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(imported$path, 32L),
    "canonical container contract")
})

test_that("external metadata scalar types and spellings are exact", {
  mutations <- list(
    version_decimal = function(value) {
      value$version <- 1
      value
    },
    uppercase_engine = function(value) {
      value$engine <- "XGBOOST"
      value
    },
    numeric_feature = function(value) {
      value$features[[1L]] <- 1L
      value
    })
  for (name in names(mutations)) {
    imported <- external_xgboost_import_fixture(
      parent = withr::local_tempdir())
    write_external_xgboost_metadata(imported$path, mutations[[name]])
    expect_error(
      dsFlowerClient:::.resolve_validation_contract(imported$path, 32L),
      "unsupported contract version",
      info = name)
  }
})

test_that("external metadata requires recursively sorted object keys", {
  imported <- external_xgboost_import_fixture()
  metadata_path <- file.path(imported$path, "metadata.json")
  value <- jsonlite::fromJSON(metadata_path, simplifyVector = FALSE)
  value <- rev(value)
  writeBin(dsFlowerClient:::.native_tree_json(value), metadata_path)
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(imported$path, 32L),
    "not canonical ASCII JSON")
})

test_that("validation bounds metadata before parsing it", {
  model_dir <- withr::local_tempdir()
  writeBin(charToRaw(strrep("x", 129L)), file.path(model_dir, "metadata.json"))
  local_mocked_bindings(
    .VALIDATION_METADATA_MAX_BYTES = 128L,
    .package = "dsFlowerClient")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(model_dir, 32L),
    "metadata.json contract")
})

test_that("external contract and profile bindings fail closed", {
  imported <- external_xgboost_import_fixture()
  write_external_xgboost_metadata(imported$path, function(value) {
    value$contract <- "unsupported-import-contract"
    value
  })
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(imported$path, 32L),
    "unsupported or missing fields")

  imported <- external_xgboost_import_fixture(
    parent = withr::local_tempdir())
  write_external_xgboost_metadata(imported$path, function(value) {
    value$prediction_profile$sha256 <- strrep("0", 64L)
    value
  })
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(imported$path, 32L),
    "profile binding")

  imported <- external_xgboost_import_fixture(
    parent = withr::local_tempdir())
  metadata_path <- file.path(imported$path, "metadata.json")
  writeBin(c(readBin(
    metadata_path, what = "raw", n = file.info(metadata_path)$size),
    charToRaw("\n")), metadata_path)
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(imported$path, 32L),
    "not canonical ASCII JSON")
})

test_that("rejected artifacts and preflight failures never publish a bundle", {
  parent <- withr::local_tempdir()
  artifact <- file.path(parent, "not-a-model.json")
  writeBin(charToRaw("\x80\x04pickle"), artifact)
  destination <- file.path(parent, "rejected")
  expect_error(
    ds.flower.import_xgboost(
      artifact = artifact,
      model = ds.flower.model.xgboost(
        n_estimators = 1L, max_depth = 1L),
      features = c("age", "marker"),
      feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
      feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
      target = "outcome", target_levels = c("control", "case"),
      output_dir = destination),
    "was rejected")
  expect_false(file.exists(destination))
  expect_false(dir.exists(destination))

  parent <- withr::local_tempdir()
  destination <- file.path(parent, "preflight-failure")
  calls <- 0L
  local_mocked_bindings(
    .validate_validation_artifact_preflight = function(...) {
      calls <<- calls + 1L
      if (calls == 2L) stop("injected final preflight failure")
      invisible(TRUE)
    },
    .package = "dsFlowerClient")
  expect_error(
    external_xgboost_import_fixture(
      parent = parent, output = destination),
    "injected final preflight failure")
  expect_identical(calls, 2L)
  expect_false(file.exists(destination))
  expect_false(dir.exists(destination))
})

test_that("external import rejects the wrong profile and existing output", {
  expect_error(
    ds.flower.import_xgboost(
      artifact = testthat::test_path(
        "fixtures", "xgboost-3.4-binary.json"),
      model = ds.flower.model.random_forest(),
      features = c("age", "marker"),
      feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
      feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
      target = "outcome", target_levels = c("control", "case"),
      output_dir = file.path(withr::local_tempdir(), "bundle")),
    "model.xgboost")

  parent <- withr::local_tempdir()
  destination <- file.path(parent, "existing")
  dir.create(destination)
  expect_error(
    ds.flower.import_xgboost(
      artifact = testthat::test_path(
        "fixtures", "xgboost-3.4-binary.json"),
      model = ds.flower.model.xgboost(
        n_estimators = 1L, max_depth = 1L),
      features = c("age", "marker"),
      feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
      feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
      target = "outcome", target_levels = c("control", "case"),
      output_dir = destination),
    "must not already exist")
})
