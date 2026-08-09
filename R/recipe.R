# Module: Composable Recipe
# Combines task, model, and strategy specs into a recipe. Privacy is node-owned.

#' Create a Flower federated learning recipe
#'
#' A recipe combines the analyst-controlled specification objects needed for a
#' federated learning experiment. Privacy policy is not part of the recipe; it
#' is selected and enforced by each data node. Task can be inferred from the
#' model when not specified.
#'
#' @param model A \code{dsflower_model} object or character model name
#'   accepted by \code{ds.flower.model()}.
#' @param strategy A \code{dsflower_strategy} object or character strategy
#'   name accepted by \code{ds.flower.strategy()}.
#' @param task A \code{dsflower_task} object, character task name, or NULL to
#'   infer from model.
#' @param num_rounds Integer; number of federated training rounds.
#' @param target Character; target column name(s). Multiple targets are supported
#'   only by the multilabel enforced-DP model.
#' @param features Character vector; feature column names, or NULL for auto.
#' @return A \code{dsflower_recipe} S3 object.
#' @export
ds.flower.recipe <- function(model,
                              strategy = ds.flower.strategy.fedavg(),
                              task = NULL,
                              num_rounds = 5L,
                              target = NULL,
                              features = NULL) {
  if (!is.numeric(num_rounds) || is.logical(num_rounds) ||
      length(num_rounds) != 1L || is.na(num_rounds) ||
      !is.finite(num_rounds) || num_rounds != floor(num_rounds) ||
      num_rounds < 1 || num_rounds > 500) {
    stop("'num_rounds' must be one integer in [1, 500].", call. = FALSE)
  }
  model <- ds.flower.model(model)
  strategy <- if (inherits(strategy, "dsflower_strategy")) {
    .canonicalize_strategy(strategy)
  } else {
    ds.flower.strategy(strategy)
  }
  inferred_type <- if (model$loss %in% c("poisson_nll", "negbin_nll")) {
    "count"
  } else if (model$loss %in% c("mse", "huber", "quantile", "gamma_nll")) {
    "regression"
  } else {
    "classification"
  }
  if (is.null(task)) {
    task <- ds.flower.task(inferred_type)
  } else {
    if (!inherits(task, "dsflower_task")) task <- ds.flower.task(task)
    .assert_supported_task(task)
    if (!identical(task$type, inferred_type)) {
      stop("Task '", task$type, "' is incompatible with model '",
           model$name, "' (expected ", inferred_type, ").", call. = FALSE)
    }
  }
  .assert_supported_task(task)

  obj <- list(
    task       = task,
    model      = model,
    strategy   = strategy,
    num_rounds = as.integer(num_rounds),
    target     = target %||% "target",
    features   = features
  )
  class(obj) <- "dsflower_recipe"
  obj
}

#' Print a dsflower_recipe
#' @param x A dsflower_recipe object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns x.
#' @export
print.dsflower_recipe <- function(x, ...) {
  cat("dsflower_recipe\n")
  cat("  Task:     ", x$task$type, "\n")
  cat("  Model:    ", x$model$name, "(", x$model$framework, ")\n")
  cat("  Strategy: ", x$strategy$name, "\n")
  cat("  Privacy:   differential privacy, decided + enforced by the data node\n")
  cat("  Rounds:   ", x$num_rounds, "\n")
  if (!is.null(x$target))
    cat("  Target:   ", paste(x$target, collapse = ", "), "\n")
  if (!is.null(x$features))
    cat("  Features: ", paste(x$features, collapse = ", "), "\n")
  invisible(x)
}
