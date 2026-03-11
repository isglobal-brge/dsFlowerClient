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
