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

test_that(".require_flwr_cli accepts provisioned client environment", {
  expect_true(isTRUE(dsFlowerClient:::.require_flwr_cli()))
})
