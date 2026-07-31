# Module: Strategy Specs
# Federated aggregation strategy specifications.

.strategy_scalar <- function(value, name, lower = 0, upper = Inf,
                             lower_open = FALSE, upper_open = FALSE) {
  value <- suppressWarnings(as.numeric(value))
  bad_lower <- if (lower_open) value <= lower else value < lower
  bad_upper <- if (upper_open) value >= upper else value > upper
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      bad_lower || bad_upper) {
    brackets <- paste0(if (lower_open) "(" else "[", lower, ", ", upper,
                       if (upper_open) ")" else "]")
    stop("'", name, "' must be one finite value in ", brackets, ".",
         call. = FALSE)
  }
  value
}

.new_strategy <- function(name, params = list()) {
  structure(list(name = name, params = params), class = "dsflower_strategy")
}

#' Create a FedAvg strategy spec
#'
#' Every node participates in every training round and private evaluation is
#' disabled. Node updates carry a constant aggregation weight, so no exact
#' cohort size is disclosed to the researcher-side SuperLink.
#'
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedavg <- function() {
  .new_strategy("FedAvg")
}

#' Create a FedAdam strategy spec
#'
#' These hyperparameters affect only researcher-side post-processing of updates
#' that have already been privatized by each node.
#'
#' @param server_learning_rate Positive server learning rate (Flower \code{eta}).
#' @param beta_1 First-moment coefficient in \code{[0,1)}.
#' @param beta_2 Second-moment coefficient in \code{[0,1)}.
#' @param tau Positive adaptivity regularizer.
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedadam <- function(server_learning_rate = 0.1,
                                        beta_1 = 0.9,
                                        beta_2 = 0.99,
                                        tau = 1e-3) {
  .new_strategy("FedAdam", list(
    eta = .strategy_scalar(server_learning_rate, "server_learning_rate",
                           lower = 0, lower_open = TRUE),
    beta_1 = .strategy_scalar(beta_1, "beta_1", upper = 1,
                              upper_open = TRUE),
    beta_2 = .strategy_scalar(beta_2, "beta_2", upper = 1,
                              upper_open = TRUE),
    tau = .strategy_scalar(tau, "tau", lower = 0, lower_open = TRUE)
  ))
}

#' Create a FedAdagrad strategy spec
#'
#' @param server_learning_rate Positive server learning rate (Flower \code{eta}).
#' @param tau Positive adaptivity regularizer.
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedadagrad <- function(server_learning_rate = 0.1,
                                           tau = 1e-3) {
  .new_strategy("FedAdagrad", list(
    eta = .strategy_scalar(server_learning_rate, "server_learning_rate",
                           lower = 0, lower_open = TRUE),
    tau = .strategy_scalar(tau, "tau", lower = 0, lower_open = TRUE)
  ))
}

#' Create a FedYogi strategy spec
#'
#' @inheritParams ds.flower.strategy.fedadam
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedyogi <- function(server_learning_rate = 0.01,
                                        beta_1 = 0.9,
                                        beta_2 = 0.99,
                                        tau = 1e-3) {
  .new_strategy("FedYogi", list(
    eta = .strategy_scalar(server_learning_rate, "server_learning_rate",
                           lower = 0, lower_open = TRUE),
    beta_1 = .strategy_scalar(beta_1, "beta_1", upper = 1,
                              upper_open = TRUE),
    beta_2 = .strategy_scalar(beta_2, "beta_2", upper = 1,
                              upper_open = TRUE),
    tau = .strategy_scalar(tau, "tau", lower = 0, lower_open = TRUE)
  ))
}

#' Create a FedAvgM strategy spec
#'
#' @param server_learning_rate Positive server learning rate.
#' @param server_momentum Server momentum in \code{[0,1)}.
#' @return A \code{dsflower_strategy} S3 object.
#' @export
ds.flower.strategy.fedavgm <- function(server_learning_rate = 1.0,
                                       server_momentum = 0.0) {
  .new_strategy("FedAvgM", list(
    server_learning_rate = .strategy_scalar(
      server_learning_rate, "server_learning_rate", lower = 0,
      lower_open = TRUE),
    server_momentum = .strategy_scalar(
      server_momentum, "server_momentum", upper = 1, upper_open = TRUE)
  ))
}

.strategy_config_lines <- function(strategy, client_learning_rate = NULL) {
  if (!inherits(strategy, "dsflower_strategy")) {
    strategy <- ds.flower.strategy(strategy)
  }
  key <- .dsflower_choice_key(strategy$name)
  allowed <- c("fedavg", "fedadam", "fedadagrad", "fedyogi", "fedavgm")
  if (!key %in% allowed) {
    stop("Strategy '", strategy$name, "' is not supported by the enforced-DP runtime.",
         call. = FALSE)
  }
  keys <- c(
    eta = "strategy-eta",
    beta_1 = "strategy-beta-1", beta_2 = "strategy-beta-2",
    tau = "strategy-tau", server_learning_rate = "strategy-server-learning-rate",
    server_momentum = "strategy-server-momentum"
  )
  unknown <- setdiff(names(strategy$params), names(keys))
  if (length(unknown)) {
    stop("Unknown parameters for strategy '", strategy$name, "': ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  lines <- c(.toml_kv("strategy", key),
    vapply(names(strategy$params), function(name) {
      .toml_kv(keys[[name]], strategy$params[[name]])
    }, character(1)))
  if (key %in% c("fedadam", "fedadagrad", "fedyogi")) {
    eta_l <- .strategy_scalar(client_learning_rate, "model learning_rate",
                              lower = 0, lower_open = TRUE)
    lines <- c(lines, .toml_kv("strategy-eta-l", eta_l))
  }
  lines
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
