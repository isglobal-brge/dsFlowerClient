# Module: Node Orchestration via DSI
# Calls server-side DataSHIELD methods to manage nodes.

#' List available label sets for an imaging dataset
#'
#' Queries the server for label sets defined in the dataset's manifest.
#'
#' @param flower A \code{dsflower_connection} from \code{ds.flower.connect()},
#'   or NULL to use the last connection.
#' @return Per-server list of data.frames with columns: name, type, columns, description.
#' @export
ds.flower.labels <- function(flower) {
  if (missing(flower) || is.null(flower))
    stop("'flower' connection handle required. Use: ds.flower.labels(flower)",
         call. = FALSE)
  if (!inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection from ds.flower.connect().",
         call. = FALSE)
  DSI::datashield.aggregate(flower$conns,
    expr = call("flowerImageLabelsDS", flower$symbol))
}

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
#'
#' # From an Opal resource (e.g. imaging+dataset://)
#' ds.flower.nodes.init(conns, resource = "chest_xray")
#' }
#'
#' @param conns DSI connections object.
#' @param data Character or named list; symbol name(s) of data already
#'   assigned in the DataSHIELD session. Mutually exclusive with
#'   \code{resource}.
#' @param resource Character or NULL; name of an Opal resource to assign
#'   and resolve before init. When provided, the resource is assigned,
#'   resolved to a ResourceClient, and passed to \code{flowerInitDS}.
#' @param symbol Character; symbol name for the Flower handle (default
#'   \code{"flower"}).
#' @return A \code{dsflower_result} with per-site init results.
#' @export
ds.flower.nodes.init <- function(conns, data = NULL, resource = NULL,
                                  symbol = "flower") {
  # Resource path: assign resource, resolve to client, then init
  if (!is.null(resource)) {
    if (!is.null(data)) {
      stop("Specify either 'data' or 'resource', not both.", call. = FALSE)
    }
    res_symbol <- paste0(symbol, "_res")

    # Assign the resource on each server
    .dsi_assign_resource_exact(
      conns, res_symbol, resource, "Resource assignment")
    # Resolve to ResourceClient
    .dsi_assign_expr_exact(
      conns, res_symbol, call("as.resource.client", as.name(res_symbol)),
      "Resource resolution")
    data <- res_symbol
  }

  if (is.null(data)) {
    stop("Either 'data' or 'resource' must be provided.", call. = FALSE)
  }

  srv_names <- names(conns)
  data_symbols <- if (is.list(data)) data else {
    stats::setNames(rep(data, length(srv_names)), srv_names)
  }

  for (srv in srv_names) {
    sym <- data_symbols[[srv]]
    if (is.null(sym)) {
      stop("No data symbol for server '", srv, "'.", call. = FALSE)
    }
    .dsi_assign_expr_exact(
      conns[srv], symbol, call("flowerInitDS", sym),
      "Flower handle initialization")
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
#' @param template_name Deprecated compatibility argument. Named executable
#'   templates have been retired; non-NULL values fail before contacting a node.
#' @param label_set Optional imaging label-set name for imaging-backed runs.
#' @return A \code{dsflower_result} with per-site status.
#' @export
ds.flower.nodes.prepare <- function(conns, symbol = "flower",
                                     target_column, feature_columns = NULL,
                                     run_config = list(),
                                     template_name = NULL,
                                     label_set = NULL) {
  # DP policy/mechanism pins and the next lifetime epsilon/delta allocation are
  # set entirely by the node at prepare, before private staging. Ensure later
  # confirms that same reservation idempotently. The client never injects
  # privacy parameters.

  if (!is.null(template_name)) {
    stop("'template_name' is retired. Submit a declarative model specification ",
         "through ds.flower.submit() instead.", call. = FALSE)
  }

  if (!is.null(label_set)) {
    run_config[["label_set"]] <- label_set
  }

  feat_enc <- .ds_encode(feature_columns)
  config_enc <- .ds_encode(run_config)
  target_enc <- .ds_encode(target_column)   # multilabel target vectors must remain one
                                            # typed argument across DSI expression parsing

  .dsi_assign_expr_exact(
    conns, symbol,
    call("flowerPrepareRunDS", symbol, target_enc, feat_enc, config_enc),
    "Run preparation")

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
#' @param template_name Deprecated compatibility argument. Named executable
#'   templates have been retired; non-NULL values fail before contacting a node.
#' @param torch_backend Character or NULL; requested node-side torch backend.
#' @return A \code{dsflower_result} with per-site status.
#' @export
ds.flower.nodes.ensure <- function(conns, symbol = "flower",
                                    superlink_address = NULL,
                                    template_name = NULL,
                                    torch_backend = NULL) {
  if (!is.null(template_name)) {
    stop("'template_name' is retired. The hash-pinned declarative runner selects ",
         "its runtime from the prepared DP track.", call. = FALSE)
  }
  # All transport is the DSI tunnel: each SuperNode dials its own node-local
  # loopback forwarder, so the address here is only a placeholder. There is no
  # remote coordinator / LAN auto-resolution path in v2.
  if (is.null(superlink_address)) {
    if (is.null(.dsflower_client_env$.tunnel)) {
      stop("No DSI tunnel is active. Call ds.flower.link.up(conns) first ",
           "(the SuperNode<->SuperLink transport is the DataSHIELD tunnel).",
           call. = FALSE)
    }
    sl <- ds.flower.superlink.status()
    superlink_address <- sl$fleet_address %||% "127.0.0.1:9092"
  }

  # Get federation_id and ca_cert_pem from the SuperLink (local or remote)
  sl_status <- ds.flower.superlink.status()
  fed_id <- sl_status$federation_id

  # B64-encode ca_cert_pem for DSI transport (if TLS is enabled)
  ca_cert_b64 <- NULL
  if (!is.null(sl_status$ca_cert_pem)) {
    ca_cert_b64 <- .ds_encode(list(pem = sl_status$ca_cert_pem))
  }

  if (is.character(superlink_address) && length(superlink_address) == 1L) {
    # Single address for all nodes
    .dsi_assign_expr_exact(
      conns, symbol,
      call("flowerEnsureSuperNodeDS", symbol, superlink_address, fed_id,
           ca_cert_b64, template_name, torch_backend),
      "SuperNode startup")
  } else if (is.list(superlink_address)) {
    # Per-node addresses
    for (srv in names(superlink_address)) {
      .dsi_assign_expr_exact(
        conns[srv], symbol,
        call("flowerEnsureSuperNodeDS", symbol, superlink_address[[srv]],
             fed_id, ca_cert_b64, template_name, torch_backend),
        "SuperNode startup")
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
        .dsf_msg("  ", srv, ": SuperNode connected")
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
  code <- .build_code("ds.flower.nodes.cleanup", symbol = symbol)
  results <- list()

  for (srv in names(conns)) {
    cleaned <- FALSE
    error <- NULL

    for (attempt in seq_len(3L)) {
      ok <- tryCatch({
        .dsi_assign_expr_exact(
          conns[srv], symbol, call("flowerCleanupRunDS", symbol),
          "Run cleanup")
        TRUE
      }, error = function(e) {
        error <<- conditionMessage(e)
        FALSE
      })

      if (isTRUE(ok)) {
        cleaned <- TRUE
        break
      }
      Sys.sleep(1)
    }

    status <- tryCatch(
      DSI::datashield.aggregate(conns[srv], expr = call("flowerStatusDS", symbol))[[srv]],
      error = function(e) list(status_error = conditionMessage(e))
    )
    status$cleanup_ok <- cleaned
    if (!isTRUE(cleaned)) status$cleanup_error <- error %||% "unknown cleanup error"
    results[[srv]] <- status
  }

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
}
