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
    ca_cert_pem = "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----",
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  status <- ds.flower.superlink.status()
  expect_true(status$running)
  expect_equal(status$ports$fleet, 9092L)
  expect_equal(status$ports$control, 9093L)
})

# --- TLS certificate generation tests ---

test_that(".run_openssl errors on bad command", {
  openssl_path <- Sys.which("openssl")
  skip_if(!nzchar(openssl_path), "openssl not available")
  expect_error(
    dsFlowerClient:::.run_openssl(openssl_path, c("req", "-in", "/nonexistent_dsflower_file")),
    "openssl command failed"
  )
})

test_that(".generate_tls_certs creates certificate files", {
  openssl_path <- Sys.which("openssl")
  skip_if(!nzchar(openssl_path), "openssl not available")

  cert_dir <- file.path(tempdir(), "dsflower_tls_test")
  on.exit(unlink(cert_dir, recursive = TRUE))

  result <- dsFlowerClient:::.generate_tls_certs(cert_dir)

  expect_true(file.exists(result$ca_cert_path))
  expect_true(file.exists(result$ca_key_path))
  expect_true(file.exists(result$srv_cert_path))
  expect_true(file.exists(result$srv_key_path))
  expect_true(nchar(result$ca_cert_pem) > 0)
  expect_true(grepl("BEGIN CERTIFICATE", result$ca_cert_pem))

  # CA key should have restricted permissions
  info <- file.info(result$ca_key_path)
  mode_str <- as.character(as.octmode(info$mode))
  expect_equal(mode_str, "600")
})

test_that(".generate_tls_certs errors when openssl is missing", {
  # Temporarily modify PATH so openssl can't be found
  withr::local_envvar(PATH = "/nonexistent_path_only")
  expect_error(
    dsFlowerClient:::.generate_tls_certs(tempdir()),
    "openssl CLI not found"
  )
})

test_that("superlink status includes TLS fields", {
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
    federation_id = "fl-test123",
    ca_cert_pem = "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----",
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  status <- ds.flower.superlink.status()
  expect_true(grepl("BEGIN CERTIFICATE", status$ca_cert_pem))
})

test_that("superlink status returns NULL ca_cert_pem when not running", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  env$.superlink <- NULL
  status <- ds.flower.superlink.status()
  expect_null(status$ca_cert_pem)
})
