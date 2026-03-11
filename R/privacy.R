# Module: Privacy Specs
# Privacy enhancement specifications for federated learning.

#' Create a research-mode privacy spec (no enhancements)
#'
#' @return A \code{dsflower_privacy} S3 object with mode = "research".
#' @export
ds.flower.privacy.research <- function() {
  obj <- list(
    mode   = "research",
    params = list()
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a differential privacy spec
#'
#' @param epsilon Numeric; privacy budget.
#' @param delta Numeric; probability of privacy leakage.
#' @param clipping_norm Numeric; gradient clipping norm.
#' @return A \code{dsflower_privacy} S3 object with mode = "dp".
#' @export
ds.flower.privacy.dp <- function(epsilon = 1.0, delta = 1e-5,
                                  clipping_norm = 1.0) {
  obj <- list(
    mode   = "dp",
    params = list(
      epsilon       = epsilon,
      delta         = delta,
      clipping_norm = clipping_norm
    )
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Print a dsflower_privacy
#' @param x A dsflower_privacy object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns x.
#' @export
print.dsflower_privacy <- function(x, ...) {
  cat("dsflower_privacy:", x$mode, "\n")
  if (length(x$params) > 0) {
    for (nm in names(x$params)) {
      cat("  ", nm, "=", .format_r_value(x$params[[nm]]), "\n")
    }
  }
  invisible(x)
}
