# Module: Composable Recipe
# Combines task, model, strategy, and privacy specs into a recipe.

#' Create a Flower federated learning recipe
#'
#' A recipe combines all specification objects needed for a federated
#' learning experiment: task type, model architecture, aggregation
#' strategy, privacy settings, and data configuration.
#'
#' @param task A \code{dsflower_task} object.
#' @param model A \code{dsflower_model} object.
#' @param strategy A \code{dsflower_strategy} object.
#' @param privacy A \code{dsflower_privacy} object (default: clinical_default).
#' @param num_rounds Integer; number of federated training rounds.
#' @param target_column Character; name of the target column.
#' @param feature_columns Character vector; names of feature columns, or NULL.
#' @return A \code{dsflower_recipe} S3 object.
#' @export
ds.flower.recipe <- function(task, model, strategy,
                              privacy = ds.flower.privacy.clinical_default(),
                              num_rounds = 5L,
                              target_column = "target",
                              feature_columns = NULL) {
  if (!inherits(task, "dsflower_task")) {
    stop("'task' must be a dsflower_task object.", call. = FALSE)
  }
  if (!inherits(model, "dsflower_model")) {
    stop("'model' must be a dsflower_model object.", call. = FALSE)
  }
  if (!inherits(strategy, "dsflower_strategy")) {
    stop("'strategy' must be a dsflower_strategy object.", call. = FALSE)
  }
  if (!inherits(privacy, "dsflower_privacy")) {
    stop("'privacy' must be a dsflower_privacy object.", call. = FALSE)
  }

  obj <- list(
    task             = task,
    model            = model,
    strategy         = strategy,
    privacy          = privacy,
    num_rounds       = as.integer(num_rounds),
    target_column    = target_column,
    feature_columns  = feature_columns
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
  cat("  Privacy:  ", x$privacy$mode, "\n")
  cat("  Rounds:   ", x$num_rounds, "\n")
  cat("  Target:   ", x$target_column, "\n")
  if (!is.null(x$feature_columns)) {
    cat("  Features: ", paste(x$feature_columns, collapse = ", "), "\n")
  }
  invisible(x)
}
