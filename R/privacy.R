# Module: Differential Privacy budget
#
# dsFlower ALWAYS enforces formal differential privacy. There are no privacy
# "profiles": disclosure thresholds come from standard DataSHIELD options on the
# server (nfilter.*), and Secure Aggregation is applied automatically when >=3
# nodes are available (distributed DP), falling back to local DP for <3 nodes.
# This module only carries the (epsilon, delta, clipping) budget.

#' Differential privacy budget for a federated run
#'
#' dsFlower always trains under formal (epsilon, delta)-differential privacy.
#' This sets the budget; the mechanism (Opacus DP-SGD, DP-GBDT, ...) is chosen
#' per model on the server, and Secure Aggregation is layered automatically when
#' enough nodes are available.
#'
#' @param epsilon Numeric > 0; privacy budget (default 3.0). Larger = less noise
#'   / weaker privacy. There is no way to disable DP; use a large epsilon if you
#'   knowingly want minimal noise.
#' @param delta Numeric in (0, 1); probability of guarantee failure (default 1e-5).
#' @param clipping_norm Numeric > 0; per-sample gradient/update clipping norm
#'   (default 1.0).
#' @return A \code{dsflower_privacy} S3 object.
#' @export
ds.flower.privacy <- function(epsilon = 3.0, delta = 1e-5, clipping_norm = 1.0) {
  if (!is.numeric(epsilon) || length(epsilon) != 1 || epsilon <= 0) {
    stop("epsilon must be a single positive number.", call. = FALSE)
  }
  if (!is.numeric(delta) || length(delta) != 1 || delta <= 0 || delta >= 1) {
    stop("delta must be a single number in (0, 1).", call. = FALSE)
  }
  if (!is.numeric(clipping_norm) || length(clipping_norm) != 1 || clipping_norm <= 0) {
    stop("clipping_norm must be a single positive number.", call. = FALSE)
  }
  obj <- list(
    epsilon       = epsilon,
    delta         = delta,
    clipping_norm = clipping_norm
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Query remaining differential-privacy budget on all servers
#'
#' Calls \code{flowerPrivacyBudgetDS} on each server to retrieve the remaining
#' (epsilon, delta) budget for the dataset.
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
  cat("dsflower_privacy (formal DP, always enforced)\n")
  cat("  epsilon       =", x$epsilon, "\n")
  cat("  delta         =", x$delta, "\n")
  cat("  clipping_norm =", x$clipping_norm, "\n")
  cat("  (local DP; no Secure Aggregation)\n")
  invisible(x)
}
