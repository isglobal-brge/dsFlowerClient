# Tests for R/superlink.R — SuperLink Lifecycle

test_that("superlink status returns not running when nothing started", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  env$.superlink <- NULL

  status <- ds.flower.superlink.status()
  expect_type(status, "list")
  expect_false(status$running)
  expect_null(status$pid)
  expect_null(status$fleet_address)
  expect_null(status$control_address)
})

test_that("superlink stop is safe when nothing running", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  env$.superlink <- NULL
  expect_message(ds.flower.superlink.stop(), "No SuperLink")
})

test_that(".detect_local_ip returns a valid IP address", {
  # This test runs on the actual machine; skip in CI if no network
  ip <- tryCatch(
    dsFlowerClient:::.detect_local_ip(),
    error = function(e) NULL
  )
  skip_if(is.null(ip), "Could not detect local IP (no network?)")
  expect_type(ip, "character")
  expect_true(grepl("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", ip))
})
