# Module: Node Orchestration via DSI
# Calls server-side DataSHIELD methods to manage nodes.

#' Initialize Flower handles on all servers
#'
#' Creates a Flower handle on each server from either a DataSHIELD resource
#' or a symbol already loaded in the R session (e.g. a data.frame from
#' previous DataSHIELD operations). Exactly one of \code{resource} or
#' \code{table} must be provided.
#'
#' @param conns DSI connections object.
#' @param resource Character or NULL; name of the resource in the Opal project
#'   (e.g. \code{"project.resource_name"}).
#' @param table Character or NULL; symbol name of a data.frame already assigned
#'   in the DataSHIELD session (e.g. \code{"D"} after a
#'   \code{datashield.assign.table} call).
#' @param symbol Character; symbol name for the handle (default "flower").
#' @return A \code{dsflower_result} with per-site init results.
#' @export
ds.flower.nodes.init <- function(conns, resource = NULL, table = NULL,
                                  symbol = "flower") {
  if (is.null(resource) && is.null(table)) {
    stop("Provide either 'resource' or 'table'.", call. = FALSE)
  }
  if (!is.null(resource) && !is.null(table)) {
    stop("Provide only one of 'resource' or 'table', not both.", call. = FALSE)
  }

  if (!is.null(resource)) {
    # Resource path: assign the resource, then init from it
    res_symbol <- .generate_symbol("res")
    DSI::datashield.assign.resource(conns, symbol = res_symbol,
                                     resource = resource)
    DSI::datashield.assign.expr(
      conns, symbol = symbol,
      expr = call("flowerInitDS", res_symbol)
    )
    code <- .build_code("ds.flower.nodes.init",
      resource = resource, symbol = symbol)
  } else {
    # Table path: the symbol already exists in the session
    DSI::datashield.assign.expr(
      conns, symbol = symbol,
      expr = call("flowerInitDS", table)
    )
    code <- .build_code("ds.flower.nodes.init",
      table = table, symbol = symbol)
  }

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
                                     run_config = list()) {
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

  results <- .ds_safe_aggregate(conns, expr = call("flowerStatusDS", symbol))

  # Verify all nodes joined the same federation
  .verify_federation(results, fed_id)

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
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
