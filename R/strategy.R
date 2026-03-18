# Module: Strategy Specs
# Federated aggregation strategy specifications.

#' Create a FedAvg strategy spec
#'
#' Federated Averaging: each node trains, sends updated weights, and the
#' server computes a weighted average. The number of participating clients
#' is determined automatically from the number of connected servers.
#'
#' @param fraction_fit Numeric; fraction of clients used for training (0-1).
#' @param fraction_evaluate Numeric; fraction of clients used for evaluation (0-1).
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedavg <- function(fraction_fit = 1.0,
                                       fraction_evaluate = 1.0) {
  obj <- list(
    name   = "FedAvg",
    params = list(
      fraction_fit      = fraction_fit,
      fraction_evaluate = fraction_evaluate
    )
  )
  class(obj) <- "dsflower_strategy"
  obj
}

#' Create a FedProx strategy spec
#'
#' FedProx adds a proximal term to keep local models closer to the global
#' model. Helps with heterogeneous (non-IID) data.
#'
#' @param proximal_mu Numeric; proximal term weight.
#' @param fraction_fit Numeric; fraction of clients used for training (0-1).
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedprox <- function(proximal_mu = 0.1,
                                        fraction_fit = 1.0) {
  obj <- list(
    name   = "FedProx",
    params = list(
      proximal_mu  = proximal_mu,
      fraction_fit = fraction_fit
    )
  )
  class(obj) <- "dsflower_strategy"
  obj
}

#' Print a dsflower_strategy
#' @param x A dsflower_strategy object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns x.
#' @export
print.dsflower_strategy <- function(x, ...) {
  cat("dsflower_strategy:", x$name, "\n")
  for (nm in names(x$params)) {
    cat("  ", nm, "=", .format_r_value(x$params[[nm]]), "\n")
  }
  invisible(x)
}
