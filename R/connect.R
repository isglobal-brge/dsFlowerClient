# Module: Connection
# High-level entry point that absorbs resource/imaging/flower init chain.

#' Connect to a data source for federated learning
#'
#' Single entry point that handles the full init chain: detects data type
#' (table symbol, Opal resource, or imaging dataset), assigns resources,
#' initializes dsImaging and dsFlower handles, and returns a connection
#' handle with metadata about available labels, masks, and features.
#'
#' @param conns DSI connections object.
#' @param data Character; one of:
#'   \itemize{
#'     \item A DataSHIELD symbol already assigned (e.g. \code{"D"})
#'     \item An Opal resource name (e.g. \code{"IMAGING.brain_mri"})
#'   }
#' @param symbol Character; server-side handle name (default auto-generated).
#' @return A \code{dsflower_connection} object.
#' @export
ds.flower.connect <- function(conns, data, symbol = "flower") {
  data_kind <- .detect_data_kind(data, conns)

  if (data_kind == "resource") {
    # Opal resource: assign -> imagingInit -> flowerInit
    img_sym <- paste0(symbol, "_img_res")
    resource_map <- stats::setNames(rep(data, length(conns)), names(conns))

    DSI::datashield.assign.resource(conns, symbol = img_sym,
      resource = as.list(resource_map))

    img_handle_sym <- paste0(symbol, "_img")
    DSI::datashield.assign.expr(conns, img_handle_sym,
      expr = call("imagingInitDS", img_sym))

    DSI::datashield.assign.expr(conns, symbol,
      expr = call("flowerInitDS", img_handle_sym))

  } else {
    # Table symbol: direct flowerInit
    DSI::datashield.assign.expr(conns, symbol,
      expr = call("flowerInitDS", data))
  }

  # Gather metadata
  labels <- tryCatch({
    if (data_kind == "resource") {
      res <- DSI::datashield.aggregate(conns,
        expr = call("imagingLabelsDS", paste0(symbol, "_img")))
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
    symbol    = symbol,
    data      = data,
    data_kind = data_kind,
    labels    = labels
  )
  class(conn) <- "dsflower_connection"

  # Store in session for ds.flower.run() to find
  .dsflower_client_env$.connection <- conn
  .dsflower_client_env$.conns <- conns

  conn
}

#' @export
print.dsflower_connection <- function(x, ...) {
  cat("dsFlower Connection\n")
  cat("  Data:    ", x$data, "(", x$data_kind, ")\n")
  cat("  Symbol:  ", x$symbol, "\n")
  cat("  Servers: ", paste(names(x$conns), collapse = ", "), "\n")
  if (nrow(x$labels) > 0) {
    cat("  Labels:\n")
    for (i in seq_len(nrow(x$labels))) {
      cat("    ", x$labels$name[i], " (", x$labels$type[i], "): ",
          x$labels$columns[i], "\n")
    }
  }
  invisible(x)
}

#' Detect whether data is a table symbol or resource name
#' @keywords internal
.detect_data_kind <- function(data, conns) {
  if (grepl("\\.", data) && !grepl("^/", data)) {
    return("resource")
  }
  "symbol"
}
