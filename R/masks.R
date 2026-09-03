# Module: Mask Discovery
# Query available segmentation masks for an imaging dataset.

#' List available segmentation masks
#'
#' Queries the server for mask assets declared in the node-owned imaging
#' manifest. This structural view intentionally excludes completion counts,
#' storage locations, and data-derived catalog state.
#'
#' @param flower A \code{dsflower_connection}, or NULL for last connection.
#' @return A data.frame with public mask aliases, providers, and the constant
#'   status \code{"declared"}, or empty if none.
#' @export
ds.flower.masks <- function(flower) {
  if (missing(flower) || is.null(flower))
    stop("'flower' connection handle required. Use: ds.flower.masks(flower)",
         call. = FALSE)
  if (!inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection from ds.flower.connect().",
         call. = FALSE)

  # Query only public structural mask declarations through dsFlower.
  tryCatch({
    res <- .dsi_private_aggregate(flower$conns,
      expr = call("flowerImageMasksDS", flower$symbol))
    res[[1]]
  }, error = function(e) {
    data.frame(alias = character(0), provider = character(0),
               status = character(0),
               stringsAsFactors = FALSE)
  })
}
