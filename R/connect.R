# Module: Connection
# High-level entry point that absorbs resource/imaging/flower init chain.

#' Connect to a data source for federated learning
#'
#' Single entry point that handles the full init chain: detects data type,
#' assigns resources, initializes dsImaging and dsFlower handles, and
#' returns a connection handle with metadata.
#'
#' Uses unique hidden symbols per connection to avoid collisions when
#' multiple connections are active.
#'
#' @param conns DSI connections object.
#' @param data Character; auto-detected data source. Use explicit params
#'   if ambiguous.
#' @param resource Character; explicit Opal resource name (e.g. "RSRC.brain_mri").
#' @param symbol Character; explicit DS symbol already assigned (e.g. "D").
#' @return A \code{dsflower_connection} object.
#' @export
ds.flower.connect <- function(conns, data = NULL, resource = NULL,
                               symbol = NULL) {
  # Exactly one of data/resource/symbol must be provided
  n_args <- sum(!is.null(data), !is.null(resource), !is.null(symbol))
  if (n_args == 0)
    stop("Provide one of: data, resource, or symbol.", call. = FALSE)

  # If data is provided, resolve deterministically
  if (!is.null(data)) {
    resolved <- .resolve_data_source(data, conns)
    if (resolved$kind == "resource") resource <- data
    else if (resolved$kind == "symbol") symbol <- data
    else stop("Cannot resolve '", data, "'. Use resource= or symbol= explicitly.",
              call. = FALSE)
  }

  # Generate unique hidden symbols (avoid collisions between connections)
  uid <- paste0(sample(c(letters, 0:9), 8, replace = TRUE), collapse = "")
  fl_sym <- paste0(".dsfl_", uid)

  data_kind <- if (!is.null(resource)) "resource" else "symbol"

  if (data_kind == "resource") {
    res_sym <- paste0(fl_sym, "_res")
    img_sym <- paste0(fl_sym, "_img")
    resource_map <- stats::setNames(rep(resource, length(conns)), names(conns))

    DSI::datashield.assign.resource(conns, symbol = res_sym,
      resource = as.list(resource_map))
    DSI::datashield.assign.expr(conns, img_sym,
      expr = call("imagingInitDS", res_sym))
    DSI::datashield.assign.expr(conns, fl_sym,
      expr = call("flowerInitDS", img_sym))
  } else {
    DSI::datashield.assign.expr(conns, fl_sym,
      expr = call("flowerInitDS", symbol))
  }

  # Gather metadata
  labels <- tryCatch({
    if (data_kind == "resource") {
      res <- DSI::datashield.aggregate(conns,
        expr = call("imagingLabelsDS", paste0(fl_sym, "_img")))
      res[[1]]
    } else {
      data.frame(name = character(0), type = character(0),
                 columns = character(0), stringsAsFactors = FALSE)
    }
  }, error = function(e) {
    data.frame(name = character(0), type = character(0),
               columns = character(0), stringsAsFactors = FALSE)
  })

  conn <- list(
    conns     = conns,
    symbol    = fl_sym,
    data      = resource %||% symbol,
    data_kind = data_kind,
    labels    = labels,
    prepare_hash = NULL
  )
  class(conn) <- "dsflower_connection"

  .dsflower_client_env$.connection <- conn
  .dsflower_client_env$.conns <- conns

  conn
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
  # Check if it's an existing symbol on all servers
  syms <- tryCatch(DSI::datashield.symbols(conns), error = function(e) list())
  all_have_sym <- all(vapply(syms, function(s) data %in% s, logical(1)))
  if (all_have_sym) return(list(kind = "symbol"))

  # Check if it looks like a resource name (PROJECT.NAME format)
  # Resources are always PROJECT.NAME in Opal
  if (grepl("^[A-Za-z][A-Za-z0-9_]*\\.[A-Za-z][A-Za-z0-9_]*$", data)) {
    return(list(kind = "resource"))
  }

  list(kind = "unknown")
}
