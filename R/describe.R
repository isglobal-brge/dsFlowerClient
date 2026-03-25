# Module: Dataset Description
# Compact summary of connected data for the researcher.

#' Describe the connected dataset
#'
#' Returns a compact summary: modality, sample count (bucketed per profile),
#' available labels, masks, and feature assets.
#'
#' @param flower A \code{dsflower_connection} from \code{ds.flower.connect()}.
#' @return A list with dataset summary fields, printed nicely.
#' @export
ds.flower.describe <- function(flower) {
  if (missing(flower) || !inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection.", call. = FALSE)

  # Get capabilities (includes sample count, profile, templates)
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
    cat("    Profile:    ", c$privacy_profile %||% "unknown", "\n")
    if (!is.null(c$data_n_rows))
      cat("    Samples:    ", c$data_n_rows, "\n")
    if (!is.null(c$data_source))
      cat("    Source:     ", c$data_source, "\n")
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
#' Queries the server for radiomics or other derived feature assets
#' available for the connected dataset.
#'
#' @param flower A \code{dsflower_connection} from \code{ds.flower.connect()}.
#' @return A data.frame with feature asset info, or empty.
#' @export
ds.flower.features <- function(flower) {
  if (missing(flower) || !inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection.", call. = FALSE)

  # Query imaging assets filtered to feature_table kind
  tryCatch({
    img_sym <- paste0(flower$symbol, "_img")
    res <- DSI::datashield.aggregate(flower$conns,
      expr = call("imagingAssetsDS", img_sym, "feature_table"))
    res[[1]]
  }, error = function(e) {
    data.frame(alias = character(0), kind = character(0),
               provider = character(0), stringsAsFactors = FALSE)
  })
}
