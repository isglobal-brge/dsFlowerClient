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
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      captured_expr <<- expr
      for (node in names(conns)) success(node)
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

test_that("nodes.prepare preserves multilabel target vectors in its DSI argument", {
  captured <- NULL
  local_mocked_bindings(
    .ds_safe_aggregate = function(...) list(),
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      captured <<- expr
      for (node in names(conns)) success(node)
    },
    .package = "DSI"
  )

  ds.flower.nodes.prepare(
    conns = list(site = NULL), symbol = "flower",
    target_column = c("label_a", "label_b"), feature_columns = "x",
    run_config = list(`loss-name` = "multilabel_bce", `num-labels` = 2L)
  )

  encoded <- captured[[3L]]
  payload <- chartr("-_", "+/", substring(encoded, 5L, nchar(encoded)))
  payload <- paste0(payload, strrep("=", (4L - nchar(payload) %% 4L) %% 4L))
  decoded <- jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(payload)))
  expect_identical(decoded, c("label_a", "label_b"))
})

test_that("a normalized per-node ASSIGN error is never treated as success", {
  conns <- list(site1 = TRUE, site2 = TRUE)
  local_mocked_bindings(
    datashield.assign.expr = function(conns, symbol, expr, success, error, ...) {
      success("site1")
      error("site2", "hidden node error")
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    dsFlowerClient:::.dsi_assign_expr_exact(
      conns, "flower", call("flowerPrepareRunDS"), "Run preparation"),
    "no ACK on: site2"
  )
})

test_that("safe aggregate records a named NULL as a node error", {
  conns <- list(site1 = TRUE, site2 = TRUE)
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      node <- names(conns)
      stats::setNames(list(if (node == "site1") list(ok = TRUE) else NULL), node)
    },
    .package = "DSI"
  )

  result <- dsFlowerClient:::.ds_safe_aggregate(conns, call("flowerStatusDS"))
  expect_named(result, "site1")
  expect_named(attr(result, "ds_errors"), "site2")
})

test_that("DSI results must be associated with the exact requested nodes", {
  conns <- list(site1 = TRUE, site2 = TRUE)
  expect_null(dsFlowerClient:::.dsi_exact_node_results(list(), conns))
  expect_null(dsFlowerClient:::.dsi_exact_node_results(
    list(site1 = list(ok = TRUE), wrong = list(ok = TRUE)), conns))
  mapped <- dsFlowerClient:::.dsi_exact_node_results(
    list(site2 = NULL, site1 = list(ok = TRUE)), conns)
  expect_named(mapped, c("site1", "site2"))
  expect_null(mapped$site2)
})

test_that("nodes.ensure sends torch_backend in the current server ABI slot", {
  captured <- NULL
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old <- env$.superlink
  env$.superlink <- list(
    process = list(is_alive = function() TRUE), pid = 999,
    fleet_address = "127.0.0.1:9092", control_address = "127.0.0.1:9093",
    fleet_port = 9092L, control_port = 9093L, serverappio_port = 9091L,
    flwr_home = tempdir(), log_path = tempfile(), federation_id = "fl-test",
    ca_cert_pem = NULL, started_at = Sys.time())
  on.exit(env$.superlink <- old)

  local_mocked_bindings(
    .wait_supernodes_ready = function(...) {
      list(site = list(supernode_running = TRUE, federation_id = "fl-test"))
    },
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      captured <<- expr
      for (node in names(conns)) success(node)
    },
    .package = "DSI"
  )

  suppressMessages(ds.flower.nodes.ensure(
    list(site = NULL), superlink_address = "127.0.0.1:9092",
    torch_backend = "cpu"))

  expect_identical(as.character(captured[[1L]]), "flowerEnsureSuperNodeDS")
  expect_length(captured, 6L)
  expect_identical(captured[[6L]], "cpu")
})

test_that("describe queries capabilities without a handle argument", {
  captured <- NULL
  flower <- structure(list(
    conns = list(site = NULL), symbol = "private_handle",
    data = "D", data_kind = "tabular"), class = "dsflower_connection")
  local_mocked_bindings(
    ds.flower.labels = function(...) list(),
    ds.flower.masks = function(...) data.frame(),
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      captured <<- expr
      list(site = list(runner_abi = 3L))
    },
    .package = "DSI"
  )

  result <- ds.flower.describe(flower)

  expect_s3_class(result, "dsflower_description")
  expect_identical(as.character(captured[[1L]]), "flowerGetCapabilitiesDS")
  expect_length(captured, 1L)
})
