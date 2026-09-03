# Module: Connection
# High-level entry point that absorbs resource/imaging/flower init chain.

#' Connect to a data source for federated learning
#'
#' Single entry point that handles the full init chain. Resource type is an
#' explicit public choice: imaging resources are admitted through dsImaging,
#' while tabular resources are resolved with \code{as.resource.client()}.
#'
#' Uses unique capability-named symbols per connection to avoid collisions
#' when multiple connections are active. The names remain visible to the DSI
#' symbol API so exact per-node teardown can distinguish absent from retained
#' handles.
#'
#' @param conns DSI connections object.
#' @param data Character; auto-detected data source. Use explicit params
#'   if ambiguous.
#' @param resource Character; explicit Opal resource name (e.g.
#'   "RSRC.brain_mri"). For an assigned object, use \code{symbol}.
#' @param resource_kind Character; exactly \code{"imaging"} or
#'   \code{"tabular"}. This is never inferred by retrying a failed admission.
#' @param symbol Character; explicit DS symbol already assigned (e.g. "D"),
#'   including an imaging handle created by
#'   \code{dsImagingClient::ds.imaging.init()}.
#' @return A \code{dsflower_connection} object.
#' @export
ds.flower.connect <- function(conns, data = NULL, resource = NULL,
                               symbol = NULL, resource_kind = "imaging") {
  resource_kind <- .resource_kind(resource_kind)
  # Exactly one of data/resource/symbol must be provided
  n_args <- sum(!is.null(data), !is.null(resource), !is.null(symbol))
  if (n_args != 1L)
    stop("Provide exactly one of: data, resource, or symbol.", call. = FALSE)

  # If data is provided, resolve deterministically
  if (!is.null(data)) {
    resolved <- .resolve_data_source(data, conns)
    if (resolved$kind == "resource") resource <- data
    else if (resolved$kind == "symbol") symbol <- data
    else stop("Cannot resolve '", data, "'. Use resource= or symbol= explicitly.",
              call. = FALSE)
  }

  # DSI implementations commonly enumerate symbols with `ls()` and therefore
  # omit dot-prefixed names. Use an OS-entropy name that remains observable so
  # exact teardown can identify successful nodes on a later retry.
  fl_sym <- .new_capability_token("dsf")
  data_kind <- if (!is.null(resource)) "resource" else "symbol"
  res_sym <- if (identical(data_kind, "resource")) {
    .dsi_init_resource_symbol(fl_sym)
  } else {
    NULL
  }
  img_sym <- if (identical(data_kind, "resource") &&
                 identical(resource_kind, "imaging")) {
    paste0(fl_sym, "_img")
  } else {
    NULL
  }
  .dsi_require_symbols_absent(
    conns, c(fl_sym, if (!is.null(res_sym)) c(res_sym, img_sym)))
  connected <- FALSE
  on.exit({
    if (!connected) {
      cleanup_failures <- character()
      rollback <- tryCatch(
        .dsi_destroy_session_exact(conns, fl_sym, img_sym),
        error = function(e) list(
          failures = paste0("rollback-state[", conditionMessage(e), "]")))
      cleanup_failures <- c(cleanup_failures, rollback$failures)
      if (!is.null(res_sym)) {
        resource_cleanup <- tryCatch(
          .dsi_remove_workspace_symbol_exact(conns, res_sym),
          error = function(e) list(
            failures = paste0("resource-state[", conditionMessage(e), "]")))
        cleanup_failures <- c(cleanup_failures, resource_cleanup$failures)
      }
      if (length(cleanup_failures)) {
        tryCatch(warning(
          "Connection initialization failed and rollback was incomplete on: ",
          paste(unique(cleanup_failures), collapse = ", "),
          ". Retry retained handle ",
          "targets with ds.flower.nodes.destroy(conns, symbol = ",
          deparse(fl_sym), ", imaging_symbol = ", deparse(img_sym), ").",
          call. = FALSE), error = function(e) NULL)
      }
    }
  }, add = TRUE)

  if (data_kind == "resource") {
    resource_map <- stats::setNames(rep(resource, length(conns)), names(conns))

    .dsi_assign_resource_exact(
      conns, res_sym, as.list(resource_map), "Resource assignment")
    if (identical(resource_kind, "imaging")) {
      .dsi_assign_expr_exact(
        conns, img_sym, call("imagingInitDS", res_sym),
        "dsImaging privacy admission")
      flower_data_symbol <- img_sym
    } else {
      .dsi_assign_expr_exact(
        conns, res_sym, call("as.resource.client", as.name(res_sym)),
        "Tabular resource resolution")
      flower_data_symbol <- res_sym
    }
    .dsi_assign_expr_exact(
      conns, fl_sym, call("flowerInitDS", flower_data_symbol),
      "Flower handle initialization")
    resource_cleanup <- .dsi_remove_workspace_symbol_exact(conns, res_sym)
    if (length(resource_cleanup$failures)) {
      cleanup_label <- if (identical(resource_kind, "imaging")) {
        "Temporary imaging resource cleanup failed on: "
      } else {
        "Temporary tabular resource cleanup failed on: "
      }
      stop(cleanup_label,
        paste(resource_cleanup$failures, collapse = ", "), ".", call. = FALSE)
    }
  } else {
    .dsi_assign_expr_exact(
      conns, fl_sym, call("flowerInitDS", symbol),
      "Flower handle initialization")
  }

  # Gather metadata
  labels <- tryCatch({
    res <- .dsi_private_aggregate(conns,
      expr = call("flowerImageLabelsDS", fl_sym))
    res[[1]]
  }, error = function(e) {
    data.frame(name = character(0), type = character(0),
               columns = character(0), stringsAsFactors = FALSE)
  })

  conn <- list(
    conns     = conns,
    symbol    = fl_sym,
    data      = resource %||% symbol,
    data_kind = data_kind,
    resource_kind = if (identical(data_kind, "resource")) resource_kind else NULL,
    imaging_symbol = img_sym,
    labels    = labels,
    prepare_hash = NULL
  )
  class(conn) <- "dsflower_connection"

  .dsflower_client_env$.conns <- conns

  connected <- TRUE
  conn
}

#' Disconnect and clean up a flower connection
#'
#' Removes server-side symbols created by \code{ds.flower.connect()}.
#'
#' @param flower A \code{dsflower_connection} object.
#' @return Invisible TRUE.
#' @export
ds.flower.disconnect <- function(flower) {
  if (missing(flower) || !inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection.", call. = FALSE)
  imaging_symbol <- flower$imaging_symbol %||%
    if (identical(flower$data_kind, "resource") &&
        is.null(flower$resource_kind)) paste0(flower$symbol, "_img") else NULL
  ds.flower.nodes.destroy(
    flower$conns, symbol = flower$symbol,
    imaging_symbol = imaging_symbol)
  invisible(TRUE)
}

# Preserve the original error from high-level workflows while still making a
# partial, retryable session-destruction failure visible to the caller.
.dsflower_disconnect_on_exit <- function(flower) {
  disconnected <- tryCatch({
    ds.flower.disconnect(flower)
    TRUE
  }, error = identity)
  if (inherits(disconnected, "error")) {
    tryCatch(
      message(
        "Automatic dsFlower session cleanup was incomplete. ",
        conditionMessage(disconnected)),
      error = function(e) NULL)
    return(invisible(FALSE))
  }
  invisible(TRUE)
}

#' @export
print.dsflower_connection <- function(x, ...) {
  cat("dsFlower Connection\n")
  cat("  Data:    ", x$data, "(", x$data_kind, ")\n")
  cat("  Servers: ", paste(names(x$conns), collapse = ", "), "\n")
  if (NROW(x$labels) > 0) {
    cat("  Labels:\n")
    for (i in seq_len(nrow(x$labels))) {
      cat("    ", x$labels$name[i], " (", x$labels$type[i], "): ",
          x$labels$columns[i], "\n")
    }
  }
  invisible(x)
}

#' Resolve a data source deterministically
#'
#' Checks if the data string is a resource name or an existing symbol.
#' No heuristics -- checks the actual server state.
#'
#' @keywords internal
.resolve_data_source <- function(data, conns) {
  is_symbol <- FALSE
  is_resource <- FALSE

  # Check if it's an existing symbol on all servers
  syms <- tryCatch(DSI::datashield.symbols(conns), error = function(e) list())
  if (length(syms) > 0) {
    is_symbol <- all(vapply(syms, function(s) data %in% s, logical(1)))
  }

  # Check if it looks like a resource name (PROJECT.NAME format)
  is_resource <- grepl("^[A-Za-z][A-Za-z0-9_]*\\.[A-Za-z][A-Za-z0-9_]*$", data)

  # Ambiguity check: if both match, fail
  if (is_symbol && is_resource) {
    stop("Ambiguous data source '", data, "': exists as both a server symbol ",
         "and matches resource name format. Use resource= or symbol= explicitly.",
         call. = FALSE)
  }

  if (is_symbol) return(list(kind = "symbol"))
  if (is_resource) return(list(kind = "resource"))
  list(kind = "unknown")
}
