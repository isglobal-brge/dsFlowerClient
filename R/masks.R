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
ds.flower.masks <- function(flower = NULL) {
  if (is.null(flower)) flower <- .dsflower_client_env$.connection
  if (is.null(flower)) stop("No connection. Call ds.flower.connect() first.",
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
