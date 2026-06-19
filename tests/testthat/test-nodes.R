# Tests for R/nodes.R — Node Orchestration

test_that(".detect_local_ip returns a valid IPv4 address", {
  ip <- tryCatch(
    dsFlowerClient:::.detect_local_ip(),
    error = function(e) NULL
  )
  skip_if(is.null(ip), "Could not detect local IP (no network?)")
  expect_type(ip, "character")
  expect_true(grepl("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", ip))
})

# --- Federation verification ---

test_that(".verify_federation passes when all IDs match", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = "fl-abc123")
  )
  expect_silent(dsFlowerClient:::.verify_federation(results, "fl-abc123"))
})

test_that(".verify_federation warns on mismatch", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = "fl-DIFFERENT")
  )
  expect_warning(
    dsFlowerClient:::.verify_federation(results, "fl-abc123"),
    "Federation ID mismatch"
  )
})

test_that(".verify_federation warns on missing IDs (mixed versions)", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = NULL)
  )
  expect_warning(
    dsFlowerClient:::.verify_federation(results, "fl-abc123"),
    "did not report a federation_id"
  )
})

# --- CA cert passthrough ---

test_that("nodes.ensure passes ca_cert_pem when TLS is enabled", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old <- env$.superlink

  mock_proc <- list(is_alive = function() TRUE)
  ca_pem <- "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----"
  env$.superlink <- list(
    process = mock_proc, pid = 999,
    fleet_address = "127.0.0.1:9092",
    control_address = "127.0.0.1:9093",
    fleet_port = 9092L, control_port = 9093L,
    serverappio_port = 9091L,
    flwr_home = tempdir(), log_path = tempfile(),
    federation_id = "fl-test",
    ca_cert_pem = ca_pem,
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  captured_expr <- NULL
  local_mocked_bindings(
    .ds_safe_aggregate = function(conns, expr) {
      list(opal1 = list(prepared = TRUE, node_ensured = TRUE,
                         federation_id = "fl-test"))
    },
    .wait_supernodes_ready = function(conns, symbol, timeout) {
      list(opal1 = list(supernode_running = TRUE, federation_id = "fl-test"))
    }
  )

  # Mock DSI::datashield.assign.expr to capture the call
  local_mocked_bindings(
    datashield.assign.expr = function(conns, symbol, expr) {
      captured_expr <<- expr
    },
    .package = "DSI"
  )

  suppressMessages(
    ds.flower.nodes.ensure(conns = list(opal1 = NULL), symbol = "flower",
                            superlink_address = "127.0.0.1:9092")
  )

  # The 4th argument should be the B64-encoded ca_cert_pem
  expect_true(length(captured_expr) >= 5)  # fn + 3 args + ca_cert
  fourth_arg <- captured_expr[[5]]
  expect_true(startsWith(fourth_arg, "B64:"))
})

test_that(".verify_federation is silent when expected ID is NULL", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = "fl-DIFFERENT")
  )
  # When researcher didn't start SuperLink via our API, federation_id is NULL
  # -> skip verification entirely
  expect_silent(dsFlowerClient:::.verify_federation(results, NULL))
})
