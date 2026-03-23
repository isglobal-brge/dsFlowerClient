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

#' Create a FedAdam strategy spec
#'
#' FedAdam uses adaptive learning rates on the server side via Adam
#' optimizer for more stable convergence in heterogeneous settings.
#'
#' @param server_learning_rate Numeric; server-side learning rate (eta).
#' @param tau Numeric; controls adaptivity (higher = more stable).
#' @param fraction_fit Numeric; fraction of clients used for training (0-1).
#' @param fraction_evaluate Numeric; fraction of clients used for evaluation (0-1).
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedadam <- function(server_learning_rate = 0.01,
                                        tau = 1e-3,
                                        fraction_fit = 1.0,
                                        fraction_evaluate = 1.0) {
  obj <- list(
    name   = "FedAdam",
    params = list(
      eta               = server_learning_rate,
      tau               = tau,
      fraction_fit      = fraction_fit,
      fraction_evaluate = fraction_evaluate
    )
  )
  class(obj) <- "dsflower_strategy"
  obj
}

#' Create a FedAdagrad strategy spec
#'
#' FedAdagrad uses adaptive learning rates on the server side via Adagrad
#' optimizer. Suited for sparse gradients and non-IID distributions.
#'
#' @param server_learning_rate Numeric; server-side learning rate (eta).
#' @param tau Numeric; controls adaptivity (higher = more stable).
#' @param fraction_fit Numeric; fraction of clients used for training (0-1).
#' @param fraction_evaluate Numeric; fraction of clients used for evaluation (0-1).
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedadagrad <- function(server_learning_rate = 0.01,
                                           tau = 1e-3,
                                           fraction_fit = 1.0,
                                           fraction_evaluate = 1.0) {
  obj <- list(
    name   = "FedAdagrad",
    params = list(
      eta               = server_learning_rate,
      tau               = tau,
      fraction_fit      = fraction_fit,
      fraction_evaluate = fraction_evaluate
    )
  )
  class(obj) <- "dsflower_strategy"
  obj
}

#' Create a FedBN strategy spec
#'
#' Federated Batch Normalization: keeps BatchNorm layers local (not
#' aggregated) to handle feature shift between sites. Essential for
#' medical imaging across different scanners/protocols.
#'
#' Built on FedAvg but the server excludes BatchNorm parameters from
#' aggregation. Each client retains its own BN statistics.
#'
#' @param fraction_fit Numeric; fraction of clients used for training (0-1).
#' @param fraction_evaluate Numeric; fraction of clients used for evaluation (0-1).
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedbn <- function(fraction_fit = 1.0,
                                      fraction_evaluate = 1.0) {
  obj <- list(
    name   = "FedBN",
    params = list(
      fraction_fit      = fraction_fit,
      fraction_evaluate = fraction_evaluate
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
