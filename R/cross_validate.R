# Module: Federated cross-validation

#' Cross-validate a neural model across federated data
#'
#' Runs \code{folds} complete, clean-initialized federated trainings. Each node
#' assigns whole privacy units to folds with its custodial HMAC, trains every
#' fold only on the complement, and keeps held-out sufficient statistics in
#' namespaced Flower runtime memory. Only one pooled differentially-private OOF metric vector is
#' released after all folds finish; no fold model, prediction, site metric, or
#' fold metric is saved.
#'
#' The node-owned per-job privacy contract reserves 80 percent for training and
#' divides it evenly across folds, with the remaining 20 percent used once for
#' the OOF release. Larger \code{folds} therefore gives each training less
#' privacy budget and can reduce model utility; three is the practical default.
#'
#' @inheritParams ds.flower.fit
#' @param folds Integer in \code{[2, 10]}; number of actual federated folds.
#' @return A \code{dsflower_cv} object containing only pooled OOF metrics and
#'   public job metadata.
#' @export
ds.flower.cross_validate <- function(
    conns,
    data = NULL,
    resource = NULL,
    symbol = NULL,
    target,
    features = NULL,
    model = "pytorch_logreg",
    model_params = list(),
    torch_backend = "auto",
    strategy = "fedavg",
    strategy_params = list(),
    rounds = 5L,
    task = NULL,
    folds = 3L,
    output_dir = NULL,
    output_name = NULL,
    silent = FALSE,
    verbose = FALSE,
    feature_bounds = NULL,
    target_levels = NULL,
    target_bounds = NULL,
    allow_insecure_http = getOption(
      "dsflower.dsi_allow_insecure_http", character())) {
  folds <- .normalize_cross_validation(folds)$folds
  ds.flower.fit(
    conns = conns, data = data, resource = resource, symbol = symbol,
    target = target, features = features, model = model,
    model_params = model_params, torch_backend = torch_backend,
    strategy = strategy, strategy_params = strategy_params, rounds = rounds,
    task = task, output_dir = output_dir, output_name = output_name,
    silent = silent, verbose = verbose, feature_bounds = feature_bounds,
    target_levels = target_levels, target_bounds = target_bounds,
    allow_insecure_http = allow_insecure_http, data_kind = "tabular",
    cross_validation = folds)
}

#' Print a federated cross-validation result
#' @param x A \code{dsflower_cv} object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns \code{x}.
#' @export
print.dsflower_cv <- function(x, ...) {
  cat("dsflower_cv\n")
  cat("  Model:   ", x$model, "\n")
  cat("  Folds:   ", x$folds, "\n")
  cat("  Sites:   ", x$n_nodes, "\n")
  cat("  Task:    ", x$task, "\n")
  cat("  Privacy: one pooled node-DP OOF release; no fold transcript\n")
  cat("  Metrics: ", paste(names(x$metrics), collapse = ", "), "\n")
  invisible(x)
}
