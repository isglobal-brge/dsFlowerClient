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

test_that("fit forwards holdout and unsupported tracks fail explicitly", {
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

  expect_error(
    dsFlowerClient:::.assert_holdout_supported(
      list(track = "native_tree"), "tabular"),
    "neural"
  )
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
    metrics = list(per_node = list(site = 0.75))
  ), file.path(root, "holdout.json"), auto_unbox = TRUE)
  expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")
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
  output_root <- withr::local_tempdir()

  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .ensure_client_framework = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(...) list(
      status = 0L, stdout = "run_id=atomic-holdout", stderr = ""),
    .read_model_weights = function(...) list(coef = 1),
    .read_training_history = function(...) data.frame(
      round = 1L, n_failures = 0L),
    .read_holdout_result = function(...) release,
    .package = "dsFlowerClient")

  run <- ds.flower.run.start(
    recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = "complete", silent = TRUE)
  expect_equal(run$holdout$accuracy, 0.75)
  saved <- readRDS(run$saved_path)
  expect_equal(saved$holdout$accuracy, 0.75)
  expect_match(paste(capture.output(print(run)), collapse = "\n"),
               "pooled DP metrics")

  release <- NULL
  expect_error(
    ds.flower.run.start(
      recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
      output_dir = output_root, output_name = "missing", silent = TRUE),
    "no pooled test metric release")
  expect_false(dir.exists(file.path(output_root, "missing")))
})
