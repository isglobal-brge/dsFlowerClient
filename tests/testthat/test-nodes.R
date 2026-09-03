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

test_that("nodes.prepare no longer exposes an external label-set argument", {
  expect_false("label_set" %in% names(formals(ds.flower.nodes.prepare)))
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

test_that("nodes.prepare rolls every node back after partial preparation", {
  methods <- character()
  local_mocked_bindings(
    datashield.assign.expr = function(conns, symbol, expr, success, error, ...) {
      method <- as.character(expr[[1L]])
      methods <<- c(methods, paste(names(conns), method, sep = ":"))
      if (identical(method, "flowerPrepareRunDS")) {
        success("site1")
        error("site2", "private preparation failure")
      } else {
        success(names(conns)[[1L]])
      }
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.prepare(
      list(site1 = NULL, site2 = NULL), symbol = "flower",
      target_column = "diagnosis"),
    "site2")
  expect_setequal(
    methods[grepl("flowerCleanupRunDS$", methods)],
    c("site1:flowerCleanupRunDS", "site2:flowerCleanupRunDS"))
})

test_that("nodes.prepare reports rollback failures after the original error", {
  cleanup_attempts <- c(site1 = 0L, site2 = 0L)
  local_mocked_bindings(
    datashield.assign.expr = function(conns, symbol, expr, success, error, ...) {
      method <- as.character(expr[[1L]])
      if (identical(method, "flowerPrepareRunDS")) {
        success("site1")
        error("site2", "private preparation failure")
      } else {
        host <- names(conns)[[1L]]
        cleanup_attempts[[host]] <<- cleanup_attempts[[host]] + 1L
        if (identical(host, "site1")) success(host)
      }
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.prepare(
      list(site1 = NULL, site2 = NULL), symbol = "flower",
      target_column = "diagnosis"),
    "Run preparation.*site2.*Rollback failed.*site2"
  )
  expect_identical(cleanup_attempts, c(site1 = 1L, site2 = 3L))
})

test_that("nodes.destroy retains no-ACK targets and retries only those nodes", {
  state <- list(
    site1 = c("flower", "flower_img"),
    site2 = c("flower", "flower_img")
  )
  fail_site2_flower <- TRUE
  assignments <- character()
  removed <- character()
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.expr = function(conns, symbol, expr, success, error, ...) {
      host <- names(conns)[[1L]]
      method <- as.character(expr[[1L]])
      assignments <<- c(assignments, paste(host, method, sep = ":"))
      if (identical(host, "site2") && identical(method, "flowerDestroyDS") &&
          isTRUE(fail_site2_flower)) {
        error(host, "temporary destroy failure")
      } else {
        success(host)
      }
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      host <- names(conns)[[1L]]
      state[[host]] <<- setdiff(state[[host]], symbol)
      removed <<- c(removed, paste(host, symbol, sep = ":"))
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.destroy(
      list(site1 = NULL, site2 = NULL), symbol = "flower",
      imaging_symbol = "flower_img"),
    "site2:flower\\[destroy\\].*retained for retry"
  )
  expect_setequal(
    removed,
    c("site1:flower", "site1:flower_img"))
  expect_identical(state$site2, c("flower", "flower_img"))

  fail_site2_flower <- FALSE
  result <- ds.flower.nodes.destroy(
    list(site1 = NULL, site2 = NULL), symbol = "flower",
    imaging_symbol = "flower_img")

  expect_s3_class(result, "dsflower_result")
  expect_identical(result$per_site$site1$flower$state, "absent")
  expect_identical(result$per_site$site2$flower$state, "destroyed")
  expect_length(state$site1, 0L)
  expect_length(state$site2, 0L)
  expect_identical(
    assignments,
    c(
      "site1:flowerDestroyDS", "site1:imagingDestroyDS",
      "site2:flowerDestroyDS", "site2:flowerDestroyDS",
      "site2:imagingDestroyDS"
    )
  )
})

test_that("nodes.destroy never removes a handle without a destroy ACK", {
  removed <- character()
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) list(site = "flower"),
    datashield.assign.expr = function(conns, symbol, expr, success, error, ...) {
      error("site", "destroy rejected")
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      removed <<- c(removed, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  failure <- tryCatch(
    ds.flower.nodes.destroy(list(site = NULL), symbol = "flower"),
    error = identity)
  expect_s3_class(failure, "error")
  expect_match(conditionMessage(failure), "retained for retry")
  expect_match(
    conditionMessage(failure),
    'ds\\.flower\\.nodes\\.destroy\\(conns = conns, symbol = "flower"\\)')
  expect_length(removed, 0L)
})

test_that("nodes.destroy preserves a retryable handle when removal fails", {
  state <- list(site = "flower")
  output_symbols <- character()
  fail_first_handle_removal <- TRUE
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      expect_identical(as.character(expr[[1L]]), "flowerDestroyDS")
      expect_false(identical(symbol, "flower"))
      output_symbols <<- c(output_symbols, symbol)
      state$site <<- unique(c(state$site, symbol))
      success("site")
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      if (identical(symbol, "flower") &&
          isTRUE(fail_first_handle_removal)) {
        fail_first_handle_removal <<- FALSE
        stop("temporary workspace removal failure")
      }
      state$site <<- setdiff(state$site, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.destroy(list(site = NULL), symbol = "flower"),
    "site:flower\\[remove\\].*retained for retry"
  )
  expect_identical(state$site, "flower")

  result <- ds.flower.nodes.destroy(
    list(site = NULL), symbol = "flower")

  expect_identical(result$per_site$site$flower$state, "destroyed")
  expect_length(state$site, 0L)
  expect_length(output_symbols, 2L)
  expect_true(all(grepl("^dsf_ack_[0-9a-f]{32}$", output_symbols)))
  expect_identical(output_symbols[[1L]], output_symbols[[2L]])
})

test_that("nodes.destroy retries deterministic orphan ACK cleanup", {
  state <- list(site = "flower")
  assignments <- 0L
  fail_ack_removal <- TRUE
  ack_symbol <- dsFlowerClient:::.dsi_destroy_ack_symbol(
    "flower", "flowerDestroyDS")
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      expect_identical(symbol, ack_symbol)
      state$site <<- unique(c(state$site, symbol))
      assignments <<- assignments + 1L
      success("site")
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      if (identical(symbol, ack_symbol) && isTRUE(fail_ack_removal)) {
        stop("simulated ACK removal failure")
      }
      state$site <<- setdiff(state$site, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.destroy(list(site = NULL), symbol = "flower"),
    "site:flower\\[ack-cleanup\\].*retained for retry")
  expect_identical(state$site, ack_symbol)
  expect_identical(assignments, 1L)

  fail_ack_removal <- FALSE
  result <- ds.flower.nodes.destroy(list(site = NULL), symbol = "flower")
  expect_identical(result$per_site$site$flower$state, "absent")
  expect_identical(state$site, character())
  expect_identical(assignments, 1L)
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

test_that("private aggregate transport suppresses DSI expression progress", {
  secret <- "B64:PRIVATE_TUNNEL_PAYLOAD_ABC123"
  observed <- NULL
  withr::local_options(
    datashield.progress = TRUE,
    datashield.errors.print = TRUE,
    progress_enabled = TRUE
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      observed <<- c(
        progress = getOption("datashield.progress"),
        errors = getOption("datashield.errors.print"),
        progress_enabled = getOption("progress_enabled"))
      if (isTRUE(getOption("datashield.progress"))) cat(deparse(expr))
      list(site = list(ok = TRUE))
    },
    .package = "DSI"
  )

  output <- capture.output(result <- dsFlowerClient:::.dsi_private_aggregate(
    list(site = NULL), call("flowerTunnelExchangeDS", secret)))
  expect_identical(observed,
    c(progress = FALSE, errors = FALSE, progress_enabled = FALSE))
  expect_identical(result, list(site = list(ok = TRUE)))
  expect_false(grepl(secret, paste(output, collapse = ""), fixed = TRUE))
  expect_true(getOption("datashield.progress"))
  expect_true(getOption("datashield.errors.print"))
  expect_true(getOption("progress_enabled"))
})

test_that("safe aggregate never returns raw remote error details", {
  conns <- list(site1 = TRUE, site2 = TRUE)
  private_details <- paste(
    "/srv/private/patient-42/scan.nii.gz",
    "s3://private-bucket/patient-42/scan.nii.gz")
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      node <- names(conns)
      if (identical(node, "site2")) stop(private_details)
      stats::setNames(list(list(ok = TRUE)), node)
    },
    .package = "DSI"
  )

  result <- dsFlowerClient:::.ds_safe_aggregate(conns, call("flowerStatusDS"))
  errors <- attr(result, "ds_errors")
  expect_named(result, "site1")
  expect_identical(errors, list(site2 = "remote aggregate call failed"))
  expect_false(grepl("patient-42|private-bucket|/srv/private",
                     paste(errors, collapse = " ")))
})

test_that("nodes.cleanup returns only generic aggregate diagnostics", {
  private_details <- paste(
    "/srv/private/patient-42/staging",
    "s3://private-bucket/patient-42/staging")
  local_mocked_bindings(
    .dsi_assign_expr_exact = function(...) invisible(TRUE),
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(...) stop(private_details),
    .package = "DSI"
  )

  result <- ds.flower.nodes.cleanup(list(site = NULL), symbol = "flower")
  expect_identical(result$per_site$site$status_error,
                   "remote aggregate call failed")
  expect_false(grepl("patient-42|private-bucket|/srv/private",
                     paste(result$per_site$site, collapse = " ")))
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
