# Tests for R/run.R — flwr CLI Integration

test_that(".parse_run_id extracts run ID", {
  stdout <- "Starting run: run_id=abc-123-def\nDone."
  expect_equal(dsFlowerClient:::.parse_run_id(stdout), "abc-123-def")
})

test_that(".parse_run_id extracts UUID", {
  stdout <- "Run 12345678-abcd-ef01-2345-6789abcdef01 started"
  result <- dsFlowerClient:::.parse_run_id(stdout)
  expect_equal(result, "12345678-abcd-ef01-2345-6789abcdef01")
})

test_that(".parse_run_id returns NULL for empty input", {
  expect_null(dsFlowerClient:::.parse_run_id(NULL))
  expect_null(dsFlowerClient:::.parse_run_id(""))
})

test_that(".parse_run_id returns NULL for no match", {
  expect_null(dsFlowerClient:::.parse_run_id("No run id here"))
})

test_that("run.start rejects forged unsupported recipes before side effects", {
  touched <- FALSE
  base <- ds.flower.recipe(model = "pytorch_logreg")
  local_mocked_bindings(
    .require_flwr_cli = function() touched <<- TRUE,
    .package = "dsFlowerClient"
  )

  evaluation <- base
  evaluation$evaluation_only <- TRUE
  expect_error(ds.flower.run.start(evaluation), "ds.flower.validate")

  labels <- base
  labels$label_set <- "labels"
  expect_error(ds.flower.run.start(labels), "label-set")

  segmentation <- base
  segmentation$masks <- "mask"
  expect_error(ds.flower.run.start(segmentation), "segmentation")

  expect_false(touched)
})

test_that(".flower_runtime_status detects ServerApp failures masked by CLI status", {
  stdout <- paste(
    "INFO : Requesting initial parameters",
    "ERROR : ServerApp raised an exception",
    "ERROR : Exit Code: 201",
    sep = "\n"
  )
  expect_equal(
    dsFlowerClient:::.flower_runtime_status(
      cli_status = 0L,
      stdout = stdout,
      weights = NULL,
      history = NULL
    ),
    201L
  )
})

test_that(".flower_runtime_status rejects partial federation despite artifacts", {
  stdout <- paste(
    "An exception was raised when attempting to load ClientApp",
    "aggregate_train: Received 1 results and 1 failures",
    sep = "\n"
  )
  expect_equal(
    dsFlowerClient:::.flower_runtime_status(
      cli_status = 0L,
      stdout = stdout,
      weights = list(coef = 1),
      history = data.frame(round = 1L, n_failures = 0L)
    ),
    1L
  )
})

test_that(".flower_runtime_status requires training artifacts for successful runs", {
  expect_equal(
    dsFlowerClient:::.flower_runtime_status(
      cli_status = 0L,
      stdout = "Successfully started run 123",
      weights = NULL,
      history = NULL
    ),
    1L
  )
  expect_equal(
    dsFlowerClient:::.flower_runtime_status(
      cli_status = 0L,
      stdout = "Successfully started run 123",
      weights = list(coef = 1),
      history = NULL
    ),
    0L
  )
})

test_that("run.start never persists a model from a failed federation", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(
    process = list(is_alive = function() TRUE),
    flwr_home = withr::local_tempdir()
  )
  recipe <- ds.flower.recipe(
    model = ds.flower.model.pytorch_logreg(),
    num_rounds = 1L,
    features = "x"
  )
  output_root <- withr::local_tempdir()

  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .ensure_client_framework = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(...) list(
      status = 0L,
      stdout = paste(
        "An exception was raised when attempting to load ClientApp",
        "aggregate_train: Received 1 results and 1 failures",
        sep = "\n"
      ),
      stderr = ""
    ),
    .read_model_weights = function(...) list(coef = 1),
    .read_training_history = function(...) data.frame(
      round = 1L, n_failures = 0L
    ),
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.run.start(
      recipe,
      conns = list(site1 = TRUE, site2 = TRUE),
      app_dir = withr::local_tempdir(),
      output_dir = output_root,
      output_name = "partial-model",
      silent = TRUE
    ),
    "Federated training failed"
  )
  expect_false(dir.exists(file.path(output_root, "partial-model")))
})

test_that("run.start persists the exact public model reconstruction contract", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(
    process = list(is_alive = function() TRUE),
    flwr_home = withr::local_tempdir())
  recipe <- ds.flower.recipe(
    model = ds.flower.model.pytorch_logreg(), num_rounds = 2L,
    features = c("a", "b"))
  recipe$model$track <- "neural"
  recipe$model_spec <- list(kind = "sequential", layers = list(
    list(op = "linear", out = "@out")))
  recipe$model_params <- list(n_classes = 2L, learning_rate = 0.1)
  recipe$loss_name <- "bce_logits"
  recipe$data_kind <- "tabular"
  output_root <- withr::local_tempdir()

  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .ensure_client_framework = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(...) list(
      status = 0L, stdout = "run_id=contract", stderr = ""),
    .read_model_weights = function(...) list(coef = c(0, 0), intercept = 0),
    .read_training_history = function(...) data.frame(
      round = 1:2, n_failures = c(0L, 0L)),
    .package = "dsFlowerClient")

  run <- ds.flower.run.start(
    recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "contract-model", silent = TRUE)
  meta <- jsonlite::fromJSON(
    file.path(run$output_dir, "metadata.json"), simplifyVector = FALSE)
  expect_identical(meta$model_spec$kind, "sequential")
  expect_identical(meta$model_spec$layers[[1L]]$out, "@out")
  expect_identical(meta$loss_name, "bce_logits")
  expect_identical(meta$data_kind, "tabular")
  expect_equal(meta$model_params$learning_rate, 0.1)
  expect_equal(meta$num_rounds, 2L)
})

test_that("load_model retains the relocated bundle directory for prediction", {
  model_dir <- withr::local_tempdir()
  saveRDS(list(model_id = "m", template = "pytorch_logreg"),
          file.path(model_dir, "m.rds"))

  loaded <- ds.flower.load_model(model_dir)
  expect_identical(
    loaded$source_dir,
    normalizePath(model_dir, winslash = "/", mustWork = TRUE))
  expect_identical(
    ds.flower.load_model(file.path(model_dir, "m.rds"))$source_dir,
    loaded$source_dir)
})

test_that("save_model creates portable RDS and JSON native-artifact bundles", {
  source_dir <- file.path(withr::local_tempdir(), "original")
  dir.create(source_dir)
  file.create(file.path(source_dir, "model.pt"))
  jsonlite::write_json(
    list(model = "pytorch_logreg", template = "pytorch_logreg",
         data_kind = "tabular", features = c("x1", "x2")),
    file.path(source_dir, "metadata.json"), auto_unbox = TRUE)
  jsonlite::write_json(
    data.frame(round = 1L, available = TRUE),
    file.path(source_dir, "history.json"), auto_unbox = TRUE)
  run <- structure(list(
    model_id = "portable", weights = NULL, available = TRUE,
    history = data.frame(round = 1L), model = "pytorch_logreg",
    strategy = "FedAvg", num_rounds = 1L, run_id = "run-1",
    output_dir = source_dir
  ), class = "dsflower_run")
  destination_dir <- file.path(withr::local_tempdir(), "nested")
  destinations <- file.path(destination_dir, c("saved.rds", "saved.json"))
  for (destination in destinations) {
    expect_identical(
      ds.flower.save_model(run, destination),
      normalizePath(destination, winslash = "/", mustWork = TRUE))
    assets <- paste0(destination, ".assets")
    expect_true(file.exists(file.path(assets, "model.pt")))
    expect_true(file.exists(file.path(assets, "metadata.json")))
  }

  unlink(source_dir, recursive = TRUE)
  for (destination in destinations) {
    assets <- paste0(destination, ".assets")
    loaded <- ds.flower.load_model(destination)
    expect_identical(
      loaded$source_dir,
      normalizePath(assets, winslash = "/", mustWork = TRUE))
    resolved <- dsFlowerClient:::.resolve_model_for_predict(loaded)
    expect_identical(
      resolved$model_file,
      normalizePath(file.path(assets, "model.pt"),
                    winslash = "/", mustWork = TRUE))
  }
})

test_that("load_model rejects missing or unsafe companion bundle paths", {
  model_dir <- withr::local_tempdir()
  missing <- file.path(model_dir, "missing.rds")
  saveRDS(list(artifact_bundle = "missing.rds.assets"), missing)
  expect_error(ds.flower.load_model(missing), "bundle is missing")

  unsafe <- file.path(model_dir, "unsafe.rds")
  saveRDS(list(artifact_bundle = "../outside"), unsafe)
  expect_error(ds.flower.load_model(unsafe), "invalid artifact bundle")
})

test_that("run.start rejects stale caller-supplied result artifacts", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(
    process = list(is_alive = function() TRUE),
    flwr_home = withr::local_tempdir()
  )
  results_dir <- withr::local_tempdir()
  writeLines("stale", file.path(results_dir, "history.json"))
  recipe <- ds.flower.recipe(model = ds.flower.model.pytorch_logreg())
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.run.start(
      recipe, conns = list(site = TRUE), app_dir = ".",
      results_dir = results_dir, silent = TRUE
    ),
    "must be empty"
  )
})

test_that(".training_artifacts_complete waits for final round", {
  results_dir <- withr::local_tempdir()
  jsonlite::write_json(
    data.frame(round = 1L, loss = 0.5, n_clients = 3L, n_failures = 0L),
    file.path(results_dir, "history.json"),
    auto_unbox = TRUE
  )

  expect_false(dsFlowerClient:::.training_artifacts_complete(
    results_dir, num_rounds = 2L
  ))
  file.create(file.path(results_dir, "model.pt"))
  expect_true(dsFlowerClient:::.training_artifacts_complete(
    results_dir, num_rounds = 1L
  ))
})

test_that("privacy-tail history completes without inventing a model artifact", {
  results_dir <- withr::local_tempdir()
  jsonlite::write_json(
    data.frame(round = 1L, n_failures = 0L, available = FALSE),
    file.path(results_dir, "history.json"), auto_unbox = TRUE)

  expect_true(dsFlowerClient:::.training_artifacts_complete(
    results_dir, num_rounds = 1L, expect_artifacts = TRUE))
  expect_false(dsFlowerClient:::.model_artifact_exists(results_dir))
})

test_that("run.start reports a privacy tail as unavailable, not trained", {
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
  output_root <- withr::local_tempdir()

  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .ensure_client_framework = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(...) list(
      status = 0L, stdout = "run_id=privacy-tail", stderr = ""),
    .read_model_weights = function(...) NULL,
    .read_training_history = function(...) data.frame(
      round = 1L, n_failures = 0L, available = FALSE),
    .package = "dsFlowerClient")

  run <- ds.flower.run.start(
    recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "privacy-tail", silent = TRUE)
  expect_identical(run$status, 0L)
  expect_false(run$available)
  expect_length(run$available_rounds, 0L)
  expect_null(run$weights)
  meta <- jsonlite::fromJSON(file.path(run$output_dir, "metadata.json"))
  expect_identical(meta$status, "unavailable")
  expect_false(meta$available)
  expect_false(any(file.exists(file.path(
    run$output_dir, c("model.pt", "model.npz", "global_model.json")))))
})

test_that(".training_artifacts_complete accepts skipped JSON marker", {
  results_dir <- withr::local_tempdir()
  jsonlite::write_json(
    data.frame(round = 1L, loss = 0.5, n_clients = 3L, n_failures = 0L),
    file.path(results_dir, "history.json"),
    auto_unbox = TRUE
  )
  jsonlite::write_json(
    list(reason = "weights_exceed_json_limit"),
    file.path(results_dir, "global_model.skipped.json"),
    auto_unbox = TRUE
  )
  expect_true(dsFlowerClient:::.training_artifacts_complete(
    results_dir, num_rounds = 1L
  ))
})

test_that("retired tree artifacts are not accepted as model releases", {
  results_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(model_type = "xgboost", n_trees = 1L, trees = list()),
    file.path(results_dir, "global_model.json"),
    auto_unbox = TRUE
  )
  expect_null(dsFlowerClient:::.read_model_weights(results_dir))

  unlink(file.path(results_dir, "global_model.json"))
  file.create(file.path(results_dir, "booster.json"))
  expect_null(dsFlowerClient:::.read_model_weights(results_dir))
  expect_false(dsFlowerClient:::.model_artifact_exists(results_dir))
})

test_that(".read_model_weights reconstructs portable neural arrays", {
  results_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(
      "0" = list(c(1.25, -2.5)),
      "1" = list(0.75),
      "__shapes__" = list(c(1L, 2L), 1L),
      "__round__" = 2L
    ),
    file.path(results_dir, "global_model.json"),
    auto_unbox = TRUE
  )
  weights <- dsFlowerClient:::.read_model_weights(results_dir)
  expect_named(weights, c("coef", "intercept"))
  expect_equal(weights$coef, matrix(c(1.25, -2.5), nrow = 1L))
  expect_equal(as.numeric(weights$intercept), 0.75)
  expect_equal(attr(weights, "round"), 2L)
})

test_that(".read_model_weights gives arbitrary modules stable parameter names", {
  results_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(
      "0" = list(1), "1" = list(2), "2" = list(3),
      "__shapes__" = list(1L, 1L, 1L), "__round__" = 1L
    ),
    file.path(results_dir, "global_model.json"),
    auto_unbox = TRUE
  )
  weights <- dsFlowerClient:::.read_model_weights(results_dir)
  expect_named(weights, paste0("parameter_", 0:2))
})

test_that(".flwr_run_timeout_secs accepts environment override", {
  withr::local_envvar(DSFLOWER_RUN_TIMEOUT_SECS = "7")
  expect_equal(dsFlowerClient:::.flwr_run_timeout_secs(), 7)
})

test_that(".require_flwr_cli accepts provisioned client environment", {
  expect_true(isTRUE(dsFlowerClient:::.require_flwr_cli()))
})
