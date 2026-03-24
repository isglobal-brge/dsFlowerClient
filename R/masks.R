# Module: Mask Discovery
# Query available segmentation masks for an imaging dataset.

#' List available segmentation masks
#'
#' Queries the server for validated mask assets from dsRadiomics.
#' Only shows ACTIVE, valid, non-partial masks by default.
#'
#' @param flower A \code{dsflower_connection}, or NULL for last connection.
#' @return A data.frame with mask assets, or empty if none.
#' @export
ds.flower.masks <- function(flower) {
  if (missing(flower) || is.null(flower))
    stop("'flower' connection handle required. Use: ds.flower.masks(flower)",
         call. = FALSE)
  if (!inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection from ds.flower.connect().",
         call. = FALSE)

  # Query server for mask assets via dsImaging
  tryCatch({
    img_sym <- paste0(flower$symbol, "_img")
    res <- DSI::datashield.aggregate(flower$conns,
      expr = call("imagingMasksDS", img_sym))
    res[[1]]
  }, error = function(e) {
    data.frame(alias = character(0), provider = character(0),
               status = character(0), n_valid = integer(0),
               stringsAsFactors = FALSE)
  })
}
