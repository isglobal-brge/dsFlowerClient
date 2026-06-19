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

test_that(".read_model_weights returns native XGBoost model JSON", {
  results_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(model_type = "xgboost", n_trees = 1L, trees = list()),
    file.path(results_dir, "global_model.json"),
    auto_unbox = TRUE
  )
  weights <- dsFlowerClient:::.read_model_weights(results_dir)
  expect_equal(weights$model_type, "xgboost")
  expect_equal(weights$n_trees, 1L)
})

test_that(".flwr_run_timeout_secs accepts environment override", {
  withr::local_envvar(DSFLOWER_RUN_TIMEOUT_SECS = "7")
  expect_equal(dsFlowerClient:::.flwr_run_timeout_secs(), 7)
})

test_that("Secure Aggregation is decided by node count, not the recipe", {
  caps3 <- replicate(3, list(secure_aggregation_supported = TRUE), simplify = FALSE)
  names(caps3) <- c("a", "b", "c")
  caps2 <- caps3[1:2]
  # >=3 supporting nodes -> SecAgg (distributed DP); <3 -> local DP only.
  expect_true(dsFlowerClient:::.resolve_secagg(caps3, verbose = FALSE))
  expect_false(dsFlowerClient:::.resolve_secagg(caps2, verbose = FALSE))
})

test_that(".require_flwr_cli accepts provisioned client environment", {
  expect_true(isTRUE(dsFlowerClient:::.require_flwr_cli()))
})
