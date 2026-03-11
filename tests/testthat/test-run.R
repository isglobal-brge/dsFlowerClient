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

test_that(".require_flwr_cli errors when flwr not found", {
  withr::with_path("", action = "replace", {
    expect_error(
      dsFlowerClient:::.require_flwr_cli(),
      "flwr.*not found"
    )
  })
})
