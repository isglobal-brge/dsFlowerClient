# Module: Node Orchestration via DSI
# Calls server-side DataSHIELD methods to manage nodes.

#' Initialize Flower handles on all servers
#'
#' Assigns the resource and calls \code{flowerInitDS} on each server.
#'
#' @param conns DSI connections object.
#' @param resource Character; name of the resource in the project.
#' @param symbol Character; symbol name for the handle (default "flower").
#' @return A \code{dsflower_result} with per-site init results.
#' @export
ds.flower.nodes.init <- function(conns, resource, symbol = "flower") {
  # Assign resources
  res_symbol <- .generate_symbol("res")
  DSI::datashield.assign.resource(conns, symbol = res_symbol,
                                   resource = resource)

  # Call flowerInitDS on each server
  DSI::datashield.assign.expr(
    conns,
    symbol = symbol,
    expr = call("flowerInitDS", res_symbol)
  )

  code <- .build_code("ds.flower.nodes.init",
    resource = resource,
    symbol = symbol
  )

  # Ping to verify
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

  if (is.character(superlink_address) && length(superlink_address) == 1L) {
    # Single address for all nodes
    DSI::datashield.assign.expr(
      conns,
      symbol = symbol,
      expr = call("flowerEnsureSuperNodeDS", symbol, superlink_address)
    )
  } else if (is.list(superlink_address)) {
    # Per-node addresses
    for (srv in names(superlink_address)) {
      DSI::datashield.assign.expr(
        conns[srv],
        symbol = symbol,
        expr = call("flowerEnsureSuperNodeDS", symbol,
                    superlink_address[[srv]])
      )
    }
  }

  code <- .build_code("ds.flower.nodes.ensure",
    symbol = symbol,
    superlink_address = superlink_address
  )

  results <- .ds_safe_aggregate(conns, expr = call("flowerStatusDS", symbol))

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
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
