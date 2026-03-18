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

#' Create a secure-mode privacy spec (metric suppression + secure aggregation)
#'
#' Requests the "secure" trust profile: per-node metrics are forbidden,
#' exact sample counts are bucketed, and secure aggregation is required.
#' The server enforces these as a floor -- the actual enforcement depends
#' on the server's configured trust profile.
#'
#' @return A \code{dsflower_privacy} S3 object with mode = "secure".
#' @export
ds.flower.privacy.secure <- function() {
  obj <- list(
    mode   = "secure",
    params = list()
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a differential privacy spec
#'
#' @param epsilon Numeric; privacy budget (default 1.0).
#' @param delta Numeric; probability of privacy leakage (default 1e-5).
#' @param clipping_norm Numeric; gradient clipping norm (default 1.0).
#' @return A \code{dsflower_privacy} S3 object with mode = "dp".
#' @export
ds.flower.privacy.dp <- function(epsilon = 1.0, delta = 1e-5,
                                  clipping_norm = 1.0) {
  if (epsilon <= 0) stop("epsilon must be positive.", call. = FALSE)
  if (delta <= 0 || delta >= 1) stop("delta must be in (0, 1).", call. = FALSE)
  if (clipping_norm <= 0) stop("clipping_norm must be positive.", call. = FALSE)
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

#' Query remaining privacy budget on all servers
#'
#' Calls \code{flowerPrivacyBudgetDS} on each server to retrieve the
#' remaining (epsilon, delta) budget for the dataset.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol name (default "flower").
#' @return Named list with per-server budget information.
#' @export
ds.flower.privacy.budget <- function(conns, symbol = "flower") {
  DSI::datashield.aggregate(
    conns, expr = call("flowerPrivacyBudgetDS", symbol)
  )
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
