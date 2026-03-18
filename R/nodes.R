# Module: Node Orchestration via DSI
# Calls server-side DataSHIELD methods to manage nodes.

#' Initialize Flower handles on all servers
#'
#' Creates a Flower handle on each server from a symbol already assigned
#' in the DataSHIELD session (data.frame, matrix, or any object loaded
#' via \code{datashield.assign.table}, \code{datashield.assign.resource},
#' or DataSHIELD operations).
#'
#' Accepts a single string (same symbol on all servers) or a named list
#' (one entry per server):
#'
#' \preformatted{
#' # Same symbol on all servers
#' ds.flower.nodes.init(conns, data = "D")
#'
#' # Different symbol per server
#' ds.flower.nodes.init(conns, data = list(
#'   hospital_a = "D_filtered",
#'   hospital_b = "D_merged",
#'   hospital_c = "D"
#' ))
#' }
#'
#' @param conns DSI connections object.
#' @param data Character or named list; symbol name(s) of data already
#'   assigned in the DataSHIELD session.
#' @param symbol Character; symbol name for the Flower handle (default
#'   \code{"flower"}).
#' @return A \code{dsflower_result} with per-site init results.
#' @export
ds.flower.nodes.init <- function(conns, data, symbol = "flower") {
  srv_names <- names(conns)
  data_symbols <- if (is.list(data)) data else {
    stats::setNames(rep(data, length(srv_names)), srv_names)
  }

  for (srv in srv_names) {
    sym <- data_symbols[[srv]]
    if (is.null(sym)) {
      stop("No data symbol for server '", srv, "'.", call. = FALSE)
    }
    DSI::datashield.assign.expr(
      conns[srv], symbol = symbol,
      expr = call("flowerInitDS", sym)
    )
  }

  # Store connections for later use (run.start, templates)
  .dsflower_client_env$.conns <- conns

  code <- .build_code("ds.flower.nodes.init", data = data, symbol = symbol)
  results <- .ds_safe_aggregate(conns, expr = call("flowerPingDS"))

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
}

#' Prepare a training run on all servers
#'
#' Calls \code{flowerPrepareRunDS} on each server to stage data.
#'
#' @param conns DSI connections object.
#' @param symbol Character; symbol name of the handle.
#' @param target_column Character; name of the target column.
#' @param feature_columns Character vector or NULL; feature column names.
#' @param run_config Named list; additional run configuration.
#' @return A \code{dsflower_result} with per-site status.
#' @export
ds.flower.nodes.prepare <- function(conns, symbol = "flower",
                                     target_column, feature_columns = NULL,
                                     run_config = list(), privacy = NULL,
                                     template_name = NULL) {
  # Inject privacy settings into run_config if a privacy spec is provided
  if (!is.null(privacy) && inherits(privacy, "dsflower_privacy")) {
    run_config[["privacy-mode"]] <- privacy$mode
    for (nm in names(privacy$params)) {
      run_config[[paste0("privacy-", nm)]] <- privacy$params[[nm]]
    }
  }

  # Pass template name for profile compatibility checks on the server
  if (!is.null(template_name)) {
    run_config[["template_name"]] <- template_name
  }

  feat_enc <- .ds_encode(feature_columns)
  config_enc <- .ds_encode(run_config)

  DSI::datashield.assign.expr(
    conns,
    symbol = symbol,
    expr = call("flowerPrepareRunDS", symbol, target_column,
                feat_enc, config_enc)
  )

  code <- .build_code("ds.flower.nodes.prepare",
    symbol = symbol,
    target_column = target_column,
    feature_columns = feature_columns
  )

  results <- .ds_safe_aggregate(conns, expr = call("flowerStatusDS", symbol))

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
}

#' Ensure SuperNodes are running on all servers
#'
#' Calls \code{flowerEnsureSuperNodeDS} on each server. If
#' \code{superlink_address} is \code{NULL}, auto-detects the correct address
#' per node by querying each Opal's environment (Docker vs bare metal).
#'
#' @param conns DSI connections object.
#' @param symbol Character; symbol name of the handle.
#' @param superlink_address Character, named list, or NULL.
#'   \itemize{
#'     \item \code{NULL} (default): auto-detect per node.
#'     \item Single string: broadcast to all nodes.
#'     \item Named list: per-node addresses (names must match connection names).
#'   }
#' @return A \code{dsflower_result} with per-site status.
#' @export
ds.flower.nodes.ensure <- function(conns, symbol = "flower",
                                    superlink_address = NULL) {
  # Auto-detect if not provided
  if (is.null(superlink_address)) {
    superlink_address <- .auto_resolve_superlink(conns, symbol)
  }

  # Get federation_id and ca_cert_pem from the local SuperLink (if we started it)
  sl_status <- ds.flower.superlink.status()
  fed_id <- sl_status$federation_id

  # B64-encode ca_cert_pem for DSI transport (if TLS is enabled)
  ca_cert_b64 <- NULL
  if (!is.null(sl_status$ca_cert_pem)) {
    ca_cert_b64 <- .ds_encode(list(pem = sl_status$ca_cert_pem))
  }

  if (is.character(superlink_address) && length(superlink_address) == 1L) {
    # Single address for all nodes
    DSI::datashield.assign.expr(
      conns,
      symbol = symbol,
      expr = call("flowerEnsureSuperNodeDS", symbol,
                  superlink_address, fed_id, ca_cert_b64)
    )
  } else if (is.list(superlink_address)) {
    # Per-node addresses
    for (srv in names(superlink_address)) {
      DSI::datashield.assign.expr(
        conns[srv],
        symbol = symbol,
        expr = call("flowerEnsureSuperNodeDS", symbol,
                    superlink_address[[srv]], fed_id, ca_cert_b64)
      )
    }
  }

  code <- .build_code("ds.flower.nodes.ensure",
    symbol = symbol,
    superlink_address = superlink_address
  )

  # Wait for all SuperNodes to be running
  results <- .wait_supernodes_ready(conns, symbol, timeout = 30)

  # Verify all nodes joined the same federation
  .verify_federation(results, fed_id)

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
}

#' Wait for all SuperNodes to be running
#'
#' Polls each server until \code{supernode_running = TRUE} or timeout.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol name.
#' @param timeout Numeric; seconds to wait.
#' @return Named list of per-node status results.
#' @keywords internal
.wait_supernodes_ready <- function(conns, symbol, timeout = 30) {
  deadline <- Sys.time() + timeout
  srv_names <- names(conns)
  ready <- stats::setNames(rep(FALSE, length(srv_names)), srv_names)

  while (Sys.time() < deadline) {
    pending <- srv_names[!ready]
    if (length(pending) == 0) break

    statuses <- .ds_safe_aggregate(
      conns[pending],
      expr = call("flowerStatusDS", symbol)
    )

    for (srv in pending) {
      if (isTRUE(statuses[[srv]]$supernode_running)) {
        ready[[srv]] <- TRUE
        message("  ", srv, ": SuperNode connected")
      }
    }

    if (all(ready)) break
    Sys.sleep(2)
  }

  if (!all(ready)) {
    failed <- srv_names[!ready]
    warning("SuperNodes not ready on: ", paste(failed, collapse = ", "),
            " (timed out after ", timeout, "s).", call. = FALSE)
  }

  # Final status from all nodes
  .ds_safe_aggregate(conns, expr = call("flowerStatusDS", symbol))
}

#' Verify all nodes joined the same federation
#'
#' Compares the \code{federation_id} reported by each node against the
#' expected value from the local SuperLink. Warns if any mismatch is found.
#'
#' @param results Named list of per-node status results.
#' @param expected_fed_id Character or NULL; expected federation ID.
#' @return Invisible NULL. Emits warnings on mismatches.
#' @keywords internal
.verify_federation <- function(results, expected_fed_id) {
  if (is.null(expected_fed_id)) return(invisible(NULL))

  reported_ids <- vapply(results, function(st) {
    st$federation_id %||% NA_character_
  }, character(1))

  mismatched <- names(reported_ids)[
    !is.na(reported_ids) & reported_ids != expected_fed_id
  ]
  missing <- names(reported_ids)[is.na(reported_ids)]

  if (length(mismatched) > 0) {
    warning(
      "Federation ID mismatch! These nodes may be connected to a different ",
      "SuperLink: ", paste(mismatched, collapse = ", "), ". ",
      "Expected '", expected_fed_id, "' but got: ",
      paste(unique(reported_ids[mismatched]), collapse = ", "), ".",
      call. = FALSE
    )
  }

  if (length(missing) > 0 && length(missing) < length(reported_ids)) {
    warning(
      "Some nodes did not report a federation_id: ",
      paste(missing, collapse = ", "), ". ",
      "They may be running an older version of dsFlower.",
      call. = FALSE
    )
  }

  invisible(NULL)
}

#' Clean up training run on all servers
#'
#' Calls \code{flowerCleanupRunDS} on each server.
#'
#' @param conns DSI connections object.
#' @param symbol Character; symbol name of the handle.
#' @return A \code{dsflower_result} with cleanup confirmation.
#' @export
ds.flower.nodes.cleanup <- function(conns, symbol = "flower") {
  DSI::datashield.assign.expr(
    conns,
    symbol = symbol,
    expr = call("flowerCleanupRunDS", symbol)
  )

  code <- .build_code("ds.flower.nodes.cleanup", symbol = symbol)

  results <- .ds_safe_aggregate(conns, expr = call("flowerStatusDS", symbol))

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
}
