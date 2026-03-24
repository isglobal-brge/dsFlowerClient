# Module: Composable Recipe
# Combines task, model, strategy, and privacy specs into a recipe.

# Task inference map: model framework -> default task type
.MODEL_DEFAULT_TASK <- list(
  pytorch_resnet18 = "classification",
  pytorch_densenet121 = "classification",
  pytorch_unet2d = "segmentation",
  pytorch_coxph = "survival",
  pytorch_lognormal_aft = "survival",
  pytorch_cause_specific_cox = "survival",
  pytorch_poisson = "regression",
  pytorch_linear_regression = "regression"
)

#' Create a Flower federated learning recipe
#'
#' A recipe combines all specification objects needed for a federated
#' learning experiment. Template is always inferred from the model.
#' Task can be inferred from the model if not specified.
#'
#' @param model A \code{dsflower_model} object (required).
#' @param strategy A \code{dsflower_strategy} object (default: FedAvg).
#' @param privacy A \code{dsflower_privacy} object (default: clinical_default).
#' @param task A \code{dsflower_task} object, or NULL to infer from model.
#' @param num_rounds Integer; number of federated training rounds.
#' @param target Character; target column name(s). For survival: c("time", "event").
#' @param target_column Alias for \code{target} (backward compat).
#' @param label_set Character; name of the label set to use (imaging datasets).
#' @param features Character vector; feature column names, or NULL for auto.
#' @param feature_columns Alias for \code{features} (backward compat).
#' @param masks Character; mask asset alias for segmentation, or NULL.
#' @param evaluation_only Logical; if TRUE, blocks model release.
#' @return A \code{dsflower_recipe} S3 object.
#' @export
ds.flower.recipe <- function(model,
                              strategy = ds.flower.strategy.fedavg(),
                              privacy = ds.flower.privacy.clinical_default(),
                              task = NULL,
                              num_rounds = 5L,
                              target = NULL,
                              target_column = NULL,
                              label_set = NULL,
                              features = NULL,
                              feature_columns = NULL,
                              masks = NULL,
                              evaluation_only = FALSE) {
  if (!inherits(model, "dsflower_model")) {
    stop("'model' must be a dsflower_model object.", call. = FALSE)
  }
  if (!inherits(strategy, "dsflower_strategy")) {
    stop("'strategy' must be a dsflower_strategy object.", call. = FALSE)
  }
  if (!inherits(privacy, "dsflower_privacy")) {
    stop("'privacy' must be a dsflower_privacy object.", call. = FALSE)
  }

  # Infer task from model if not provided
  if (is.null(task)) {
    default_type <- .MODEL_DEFAULT_TASK[[model$template]]
    if (!is.null(default_type)) {
      task <- switch(default_type,
        classification = ds.flower.task.classification(),
        regression     = ds.flower.task.regression(),
        survival       = ds.flower.task.survival(),
        segmentation   = ds.flower.task.segmentation(),
        ds.flower.task.classification()
      )
    } else {
      task <- ds.flower.task.classification()
    }
  }
  if (!inherits(task, "dsflower_task")) {
    stop("'task' must be a dsflower_task object.", call. = FALSE)
  }

  # Resolve target (new param wins over backward-compat)
  resolved_target <- target %||% target_column %||% "target"
  resolved_features <- features %||% feature_columns

  if (evaluation_only) {
    privacy$params$evaluation_only <- TRUE
  }

  obj <- list(
    task            = task,
    model           = model,
    strategy        = strategy,
    privacy         = privacy,
    num_rounds      = as.integer(num_rounds),
    target_column   = resolved_target,
    target          = resolved_target,
    feature_columns = resolved_features,
    features        = resolved_features,
    label_set       = label_set,
    masks           = masks
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
  cat("  Privacy:  ", x$privacy$mode, "\n")
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
