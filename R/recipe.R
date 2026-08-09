# Module: Composable Recipe
# Combines task, model, and strategy specs into a recipe. Privacy is node-owned.

#' Create a Flower federated learning recipe
#'
#' A recipe combines the analyst-controlled specification objects needed for a
#' federated learning experiment. Privacy policy is not part of the recipe; it
#' is selected and enforced by each data node. Template is always inferred from
#' the model, and task can be inferred when not specified.
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
#' @param target_column Alias for \code{target} (backward compat).
#' @param label_set Reserved compatibility argument. Non-NULL values fail early;
#'   label-set staging is not implemented by the enforced-DP runtime.
#' @param features Character vector; feature column names, or NULL for auto.
#' @param feature_columns Alias for \code{features} (backward compat).
#' @param masks Reserved compatibility argument. Non-NULL values fail early
#'   because segmentation is not implemented by the enforced-DP runtime.
#' @param evaluation_only Reserved compatibility argument. TRUE fails early;
#'   evaluation-only execution is not implemented. Use
#'   \code{ds.flower.validate()} for private model validation.
#' @return A \code{dsflower_recipe} S3 object.
#' @export
ds.flower.recipe <- function(model,
                              strategy = ds.flower.strategy.fedavg(),
                              task = NULL,
                              num_rounds = 5L,
                              target = NULL,
                              target_column = NULL,
                              label_set = NULL,
                              features = NULL,
                              feature_columns = NULL,
                              masks = NULL,
                              evaluation_only = FALSE) {
  if (!is.null(masks)) {
    stop("'masks' requires segmentation, which is not supported by the ",
         "enforced-DP runtime in this release.", call. = FALSE)
  }
  if (!is.null(label_set)) {
    stop("'label_set' is not supported by the enforced-DP runtime.",
         call. = FALSE)
  }
  if (!is.logical(evaluation_only) || length(evaluation_only) != 1L ||
      is.na(evaluation_only)) {
    stop("'evaluation_only' must be TRUE or FALSE.", call. = FALSE)
  }
  if (evaluation_only) {
    stop("'evaluation_only = TRUE' is not implemented; use ",
         "ds.flower.validate() for disclosure-safe validation.",
         call. = FALSE)
  }
  if (!is.numeric(num_rounds) || is.logical(num_rounds) ||
      length(num_rounds) != 1L || is.na(num_rounds) ||
      !is.finite(num_rounds) || num_rounds != floor(num_rounds) ||
      num_rounds < 1 || num_rounds > 500) {
    stop("'num_rounds' must be one integer in [1, 500].", call. = FALSE)
  }
  if (!is.null(target) && !is.null(target_column) &&
      !identical(target, target_column)) {
    stop("'target' and 'target_column' cannot disagree.", call. = FALSE)
  }
  if (!is.null(features) && !is.null(feature_columns) &&
      !identical(features, feature_columns)) {
    stop("'features' and 'feature_columns' cannot disagree.", call. = FALSE)
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

  # Resolve target (new param wins over backward-compat)
  resolved_target <- target %||% target_column %||% "target"
  resolved_features <- features %||% feature_columns

  obj <- list(
    task            = task,
    model           = model,
    strategy        = strategy,
    num_rounds      = as.integer(num_rounds),
    target_column   = resolved_target,
    target          = resolved_target,
    feature_columns = resolved_features,
    features        = resolved_features,
    label_set       = NULL,
    masks           = masks,
    evaluation_only = FALSE
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
  cat("  Template: ", x$model$template, "\n")
  cat("  Strategy: ", x$strategy$name, "\n")
  cat("  Privacy:   differential privacy, decided + enforced by the data node\n")
  cat("  Rounds:   ", x$num_rounds, "\n")
  if (!is.null(x$target))
    cat("  Target:   ", paste(x$target, collapse = ", "), "\n")
  if (!is.null(x$label_set))
    cat("  Labels:   ", x$label_set, "\n")
  if (!is.null(x$masks))
    cat("  Masks:    ", x$masks, "\n")
  if (!is.null(x$features))
    cat("  Features: ", paste(x$features, collapse = ", "), "\n")
  invisible(x)
}
