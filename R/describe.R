# Module: Dataset Description
# Compact summary of connected data for the researcher.

#' Describe the connected dataset
#'
#' Returns a compact summary of public protocol capabilities and node-owned
#' structural imaging declarations. It does not report cohort-derived counts or
#' schema discovered from private data.
#'
#' @param flower A \code{dsflower_connection} from \code{ds.flower.connect()}.
#' @return A list with dataset summary fields, printed nicely.
#' @export
ds.flower.describe <- function(flower) {
  if (missing(flower) || !inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection.", call. = FALSE)

  # Get data-independent public protocol/runtime capabilities.
  caps <- tryCatch(
    DSI::datashield.aggregate(flower$conns,
      expr = call("flowerGetCapabilitiesDS", flower$symbol)),
    error = function(e) list()
  )

  # Get labels
  labels <- tryCatch(ds.flower.labels(flower), error = function(e) list())

  # Get masks
  masks <- tryCatch(ds.flower.masks(flower), error = function(e) data.frame())

  desc <- list(
    data      = flower$data,
    data_kind = flower$data_kind,
    servers   = names(flower$conns),
    caps      = caps,
    labels    = labels,
    masks     = masks
  )
  class(desc) <- "dsflower_description"
  desc
}

#' @export
print.dsflower_description <- function(x, ...) {
  cat("dsFlower Dataset Summary\n")
  cat("  Data:    ", x$data, "(", x$data_kind, ")\n")
  cat("  Servers: ", paste(x$servers, collapse = ", "), "\n")

  # Per-server info
  for (srv in names(x$caps)) {
    c <- x$caps[[srv]]
    cat("\n  [", srv, "]\n")
    cat("    Privacy:     node-side central DP (always enforced before egress)\n")
  }

  # Labels
  if (length(x$labels) > 0) {
    first_labels <- x$labels[[1]]
    if (NROW(first_labels) > 0) {
      cat("\n  Labels:\n")
      for (i in seq_len(nrow(first_labels))) {
        cat("    ", first_labels$name[i], " (", first_labels$type[i], "): ",
            first_labels$columns[i], "\n")
      }
    }
  }

  # Masks
  if (NROW(x$masks) > 0) {
    cat("\n  Masks:\n")
    for (i in seq_len(nrow(x$masks))) {
      cat("    ", x$masks$alias[i], " (", x$masks$provider[i], "): ",
          x$masks$status[i], "\n")
    }
  }

  invisible(x)
}

#' List available feature assets
#'
#' Queries the server for feature-table assets declared in the node-owned
#' imaging manifest. Storage locations and data-derived catalog state are not
#' returned.
#'
#' @param flower A \code{dsflower_connection} from \code{ds.flower.connect()}.
#' @return A data.frame with feature asset info, or empty.
#' @export
ds.flower.features <- function(flower) {
  if (missing(flower) || !inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection.", call. = FALSE)

  # Query public manifest assets and filter locally to feature tables.
  tryCatch({
    res <- DSI::datashield.aggregate(flower$conns,
      expr = call("flowerImageAssetsDS", flower$symbol))
    assets <- res[[1]]
    assets[tolower(assets$kind) == "feature_table", , drop = FALSE]
  }, error = function(e) {
    data.frame(alias = character(0), kind = character(0),
               provider = character(0), stringsAsFactors = FALSE)
  })
}
