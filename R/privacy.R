# Module: Privacy Specs
# Privacy profile specifications for federated learning.
# 7 profiles + 1 modifier (evaluation_only).

#' Create a sandbox_open privacy spec
#'
#' Minimal restrictions. Requires explicit admin opt-in on the server
#' (\code{dsflower.allow_sandbox = TRUE}).
#'
#' @return A \code{dsflower_privacy} S3 object with mode = "sandbox_open".
#' @export
ds.flower.privacy.sandbox_open <- function() {
  obj <- list(
    mode   = "sandbox_open",
    params = list()
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a trusted_internal privacy spec
#'
#' For trusted internal collaborations. Per-node metrics allowed, no SecAgg.
#'
#' @return A \code{dsflower_privacy} S3 object with mode = "trusted_internal".
#' @export
ds.flower.privacy.trusted_internal <- function() {
  obj <- list(
    mode   = "trusted_internal",
    params = list()
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a consortium_internal privacy spec
#'
#' For consortium collaborations. Per-node metrics suppressed, fixed sampling.
#'
#' @return A \code{dsflower_privacy} S3 object with mode = "consortium_internal".
#' @export
ds.flower.privacy.consortium_internal <- function() {
  obj <- list(
    mode   = "consortium_internal",
    params = list()
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a clinical_default privacy spec (recommended)
#'
#' The recommended default for clinical federated learning. Requires SecAgg,
#' suppresses per-node metrics, enforces fixed client sampling.
#'
#' @return A \code{dsflower_privacy} S3 object with mode = "clinical_default".
#' @export
ds.flower.privacy.clinical_default <- function() {
  obj <- list(
    mode   = "clinical_default",
    params = list()
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a clinical_hardened privacy spec
#'
#' Stricter than clinical_default: higher minimum rows, requires 3+ clients.
#'
#' @return A \code{dsflower_privacy} S3 object with mode = "clinical_hardened".
#' @export
ds.flower.privacy.clinical_hardened <- function() {
  obj <- list(
    mode   = "clinical_hardened",
    params = list()
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a clinical_update_noise privacy spec
#'
#' Update-level differential privacy hardening: clips weight updates and
#' adds calibrated Gaussian noise before aggregation. SecAgg enforced.
#'
#' NOTE: This is NOT patient-level DP-SGD. It protects against an honest-but-
#' curious aggregator seeing individual updates, but does not provide formal
#' per-example privacy guarantees. For formal DP, use
#' \code{ds.flower.privacy.high_sensitivity_dp()} which uses Opacus DP-SGD.
#'
#' @param epsilon Numeric; privacy budget (default 1.0).
#' @param delta Numeric; probability of privacy leakage (default 1e-5).
#' @param clipping_norm Numeric; update clipping norm (default 1.0).
#' @return A \code{dsflower_privacy} S3 object with mode = "clinical_update_noise".
#' @export
ds.flower.privacy.clinical_update_noise <- function(epsilon = 1.0, delta = 1e-5,
                                                     clipping_norm = 1.0) {
  if (epsilon <= 0) stop("epsilon must be positive.", call. = FALSE)
  if (delta <= 0 || delta >= 1) stop("delta must be in (0, 1).", call. = FALSE)
  if (clipping_norm <= 0) stop("clipping_norm must be positive.", call. = FALSE)
  obj <- list(
    mode   = "clinical_update_noise",
    params = list(
      epsilon       = epsilon,
      delta         = delta,
      clipping_norm = clipping_norm
    )
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Create a high_sensitivity_dp privacy spec
#'
#' The most restrictive profile. Requires patient-level DP-SGD, SecAgg,
#' and 3+ clients.
#'
#' @param epsilon Numeric; privacy budget (default 1.0).
#' @param delta Numeric; probability of privacy leakage (default 1e-5).
#' @param clipping_norm Numeric; gradient clipping norm (default 1.0).
#' @return A \code{dsflower_privacy} S3 object with mode = "high_sensitivity_dp".
#' @export
ds.flower.privacy.high_sensitivity_dp <- function(epsilon = 1.0, delta = 1e-5,
                                                    clipping_norm = 1.0) {
  if (epsilon <= 0) stop("epsilon must be positive.", call. = FALSE)
  if (delta <= 0 || delta >= 1) stop("delta must be in (0, 1).", call. = FALSE)
  if (clipping_norm <= 0) stop("clipping_norm must be positive.", call. = FALSE)
  obj <- list(
    mode   = "high_sensitivity_dp",
    params = list(
      epsilon       = epsilon,
      delta         = delta,
      clipping_norm = clipping_norm
    )
  )
  class(obj) <- "dsflower_privacy"
  obj
}

#' Apply evaluation_only modifier to a privacy spec
#'
#' Forces \code{model_release = "blocked"} and
#' \code{allow_per_node_metrics = FALSE} on the server.
#'
#' @param base_privacy A \code{dsflower_privacy} S3 object.
#' @return A modified \code{dsflower_privacy} S3 object with evaluation_only = TRUE.
#' @export
ds.flower.privacy.evaluation_only <- function(base_privacy) {
  if (!inherits(base_privacy, "dsflower_privacy")) {
    stop("base_privacy must be a dsflower_privacy object.", call. = FALSE)
  }
  base_privacy$params$evaluation_only <- TRUE
  base_privacy
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
  if (isTRUE(x$params$evaluation_only)) {
    cat("  [evaluation_only]\n")
  }
  dp_params <- setdiff(names(x$params), "evaluation_only")
  if (length(dp_params) > 0) {
    for (nm in dp_params) {
      cat("  ", nm, "=", .format_r_value(x$params[[nm]]), "\n")
    }
  }
  invisible(x)
}
