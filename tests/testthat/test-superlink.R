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

test_that("superlink status reports ports correctly", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  mock_proc <- list(is_alive = function() TRUE)
  old <- env$.superlink
  env$.superlink <- list(
    process = mock_proc, pid = 123,
    fleet_address = "127.0.0.1:9092",
    control_address = "127.0.0.1:9093",
    fleet_port = 9092L, control_port = 9093L,
    serverappio_port = 9091L,
    flwr_home = tempdir(), log_path = tempfile(),
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  status <- ds.flower.superlink.status()
  expect_true(status$running)
  expect_equal(status$ports$fleet, 9092L)
  expect_equal(status$ports$control, 9093L)
})
