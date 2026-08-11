# Module: Private validation metric selection
# Keeps metric discovery and HPO score extraction as local post-processing.

# The largest public 1024-label/512-bin compact result is below 139 MiB even
# when every numeric leaf uses the longest finite JSON float representation.
.PRIVATE_METRIC_RESULT_MAX_BYTES <- 160 * 1024^2

.DSFLOWER_SCORE_METRICS <- list(
  binary = c(
    accuracy = "maximize",
    sensitivity = "maximize",
    specificity = "maximize",
    precision = "maximize",
    negative_predictive_value = "maximize",
    f1 = "maximize",
    balanced_accuracy = "maximize",
    roc_auc = "maximize",
    pr_auc = "maximize",
    brier = "minimize",
    expected_calibration_error = "minimize"
  ),
  multiclass = c(
    accuracy = "maximize",
    balanced_accuracy = "maximize",
    macro_precision = "maximize",
    macro_recall = "maximize",
    macro_f1 = "maximize",
    macro_roc_auc = "maximize"
  ),
  ordinal = c(
    accuracy = "maximize",
    balanced_accuracy = "maximize",
    macro_precision = "maximize",
    macro_recall = "maximize",
    macro_f1 = "maximize",
    macro_roc_auc = "maximize",
    ordinal_mae = "minimize"
  ),
  multilabel = c(
    macro_roc_auc = "maximize",
    macro_f1 = "maximize"
  ),
  regression = c(
    mae = "minimize",
    mse = "minimize",
    rmse = "minimize",
    r_squared = "maximize"
  ),
  count = c(
    mae = "minimize",
    mse = "minimize",
    rmse = "minimize",
    r_squared = "maximize",
    mean_poisson_deviance_normalized = "minimize"
  )
)

.score_metric_registry <- function(task) {
  if (!is.character(task) || length(task) != 1L || is.na(task) ||
      !nzchar(task) || !task %in% names(.DSFLOWER_SCORE_METRICS)) {
    stop("'task' must be one of: ",
         paste(names(.DSFLOWER_SCORE_METRICS), collapse = ", "), ".",
         call. = FALSE)
  }
  .DSFLOWER_SCORE_METRICS[[task]]
}

.private_metrics_valid <- function(metrics, task) {
  exact_object <- function(value, fields) {
    is.list(value) && !is.null(names(value)) &&
      !anyNA(names(value)) && all(nzchar(names(value))) &&
      !anyDuplicated(names(value)) && length(value) == length(fields) &&
      setequal(names(value), fields)
  }
  finite_scalar <- function(
      value, lower = -Inf, upper = Inf, nullable = FALSE) {
    if (is.null(value)) return(isTRUE(nullable))
    is.numeric(value) && !is.logical(value) && length(value) == 1L &&
      is.null(dim(value)) && is.null(names(value)) && !is.na(value) &&
      is.finite(value) && value >= lower && value <= upper
  }
  array_values <- function(value) {
    if (is.numeric(value) && !is.logical(value) && is.null(dim(value)) &&
        is.null(names(value))) {
      return(as.numeric(value))
    }
    if (!is.list(value) || !is.null(names(value)) ||
        !all(vapply(value, finite_scalar, logical(1)))) {
      return(NULL)
    }
    as.numeric(unlist(value, recursive = FALSE, use.names = FALSE))
  }
  finite_array <- function(
      value, expected_length, lower = -Inf, upper = Inf) {
    values <- array_values(value)
    !is.null(values) && length(values) == expected_length &&
      all(is.finite(values)) && all(values >= lower) && all(values <= upper)
  }
  confusion_size <- function(value) {
    if (is.matrix(value)) {
      dimensions <- dim(value)
      if (!is.numeric(value) || is.logical(value) ||
          !is.null(dimnames(value)) || length(dimensions) != 2L ||
          dimensions[[1L]] != dimensions[[2L]] ||
          dimensions[[1L]] < 2L || dimensions[[1L]] > 1024L ||
          anyNA(value) || any(!is.finite(value)) || any(value < 0)) {
        return(NA_integer_)
      }
      return(as.integer(dimensions[[1L]]))
    }
    if (!is.list(value) || !is.null(names(value)) ||
        length(value) < 2L || length(value) > 1024L) {
      return(NA_integer_)
    }
    classes <- length(value)
    if (!all(vapply(value, finite_array, logical(1),
                    expected_length = classes, lower = 0))) {
      return(NA_integer_)
    }
    as.integer(classes)
  }
  binary_bins <- function(value) {
    scalar_fields <- c(
      "n", "accuracy", "sensitivity", "specificity", "precision",
      "negative_predictive_value", "f1", "balanced_accuracy", "roc_auc",
      "pr_auc", "brier", "expected_calibration_error")
    if (!exact_object(value, c(
        scalar_fields, "roc", "precision_recall", "calibration",
        "decision_curve")) ||
        !exact_object(value$roc, c("fpr", "tpr")) ||
        !exact_object(value$precision_recall, c("recall", "precision")) ||
        !exact_object(value$calibration,
                      c("predicted", "observed", "weight")) ||
        !exact_object(value$decision_curve, c(
          "threshold", "net_benefit", "treat_all", "treat_none"))) {
      return(NA_integer_)
    }
    predicted <- array_values(value$calibration$predicted)
    bins <- length(predicted)
    if (is.null(predicted) || bins < 4L || bins > 512L ||
        !finite_scalar(value$n, lower = 0) ||
        !finite_scalar(value$accuracy, lower = 0, upper = 1) ||
        !all(vapply(scalar_fields[-(1:2)], function(field) {
          finite_scalar(value[[field]], lower = 0, upper = 1,
                        nullable = TRUE)
        }, logical(1))) ||
        !finite_array(value$roc$fpr, bins + 2L, 0, 1) ||
        !finite_array(value$roc$tpr, bins + 2L, 0, 1) ||
        !finite_array(value$precision_recall$recall, bins + 1L, 0, 1) ||
        !finite_array(value$precision_recall$precision, bins + 1L, 0, 1) ||
        !finite_array(value$calibration$predicted, bins, 0, 1) ||
        !finite_array(value$calibration$observed, bins, 0, 1) ||
        !finite_array(value$calibration$weight, bins, 0) ||
        !finite_array(value$decision_curve$threshold, bins, 0, 1) ||
        !finite_array(value$decision_curve$net_benefit, bins) ||
        !finite_array(value$decision_curve$treat_all, bins) ||
        !finite_array(value$decision_curve$treat_none, bins)) {
      return(NA_integer_)
    }
    as.integer(bins)
  }

  if (!is.character(task) || length(task) != 1L || is.na(task)) return(FALSE)
  if (identical(task, "binary")) return(!is.na(binary_bins(metrics)))

  if (task %in% c("multiclass", "ordinal")) {
    fields <- c(
      "n", "accuracy", "balanced_accuracy", "macro_precision",
      "macro_recall", "macro_f1", "macro_roc_auc", "confusion_matrix",
      if (identical(task, "ordinal")) "ordinal_mae")
    if (!exact_object(metrics, fields)) return(FALSE)
    classes <- confusion_size(metrics$confusion_matrix)
    valid <- !is.na(classes) && finite_scalar(metrics$n, lower = 0) &&
      finite_scalar(metrics$accuracy, 0, 1, nullable = TRUE) &&
      all(vapply(c(
        "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"
      ), function(field) finite_scalar(metrics[[field]], 0, 1), logical(1))) &&
      finite_scalar(metrics$macro_roc_auc, 0, 1, nullable = TRUE)
    if (identical(task, "ordinal")) {
      valid <- valid && finite_scalar(
        metrics$ordinal_mae, 0, classes - 1, nullable = TRUE)
    }
    return(isTRUE(valid))
  }

  if (identical(task, "multilabel")) {
    if (!exact_object(metrics, c("labels", "macro_roc_auc", "macro_f1")) ||
        !is.list(metrics$labels) || !is.null(names(metrics$labels)) ||
        length(metrics$labels) < 2L || length(metrics$labels) > 1024L) {
      return(FALSE)
    }
    bins <- vapply(metrics$labels, binary_bins, integer(1))
    return(all(!is.na(bins)) && length(unique(bins)) == 1L &&
      finite_scalar(metrics$macro_roc_auc, 0, 1, nullable = TRUE) &&
      finite_scalar(metrics$macro_f1, 0, 1, nullable = TRUE))
  }

  if (task %in% c("regression", "count")) {
    fields <- c(
      "n", "mae", "mse", "rmse", "r_squared",
      if (identical(task, "count")) "mean_poisson_deviance_normalized")
    valid <- exact_object(metrics, fields) &&
      finite_scalar(metrics$n, lower = 0) &&
      finite_scalar(metrics$mae, lower = 0) &&
      finite_scalar(metrics$mse, lower = 0) &&
      finite_scalar(metrics$rmse, lower = 0) &&
      finite_scalar(metrics$r_squared, upper = 1, nullable = TRUE)
    if (identical(task, "count")) {
      valid <- valid && finite_scalar(
        metrics$mean_poisson_deviance_normalized, 0, 1, nullable = TRUE)
    }
    return(isTRUE(valid))
  }
  FALSE
}

.score_metric_name <- function(metric) {
  if (!is.character(metric) || length(metric) != 1L || is.na(metric) ||
      !nzchar(metric)) {
    stop("'metric' must be one non-empty metric name.", call. = FALSE)
  }
  metric
}

.private_metric_payload <- function(result) {
  if (inherits(result, "dsflower_validation")) {
    if (!is.logical(result$available) || length(result$available) != 1L ||
        is.na(result$available)) {
      stop("Validation result has a malformed availability flag.",
           call. = FALSE)
    }
    if ((isTRUE(result$available) && !is.list(result$metrics)) ||
        (!isTRUE(result$available) && !is.null(result$metrics))) {
      stop("Validation result has a malformed metric release.",
           call. = FALSE)
    }
    payload <- list(
      task = result$task,
      metrics = result$metrics,
      released = isTRUE(result$available)
    )
  } else if (inherits(result, "dsflower_cv")) {
    if (!is.list(result$metrics)) {
      stop("Cross-validation result has a malformed metric release.",
           call. = FALSE)
    }
    payload <- list(task = result$task, metrics = result$metrics,
                    released = TRUE)
  } else if (inherits(result, "dsflower_run")) {
    if (!is.list(result$holdout) || is.null(result$holdout_task)) {
      stop("The run has no pooled holdout metric release.", call. = FALSE)
    }
    if (!identical(result$available, TRUE)) {
      stop("The run has a malformed holdout metric release.", call. = FALSE)
    }
    payload <- list(task = result$holdout_task, metrics = result$holdout,
                    released = TRUE)
  } else {
    stop("'result' must be a dsflower_validation, dsflower_cv, or a ",
         "dsflower_run with an atomic holdout release.", call. = FALSE)
  }

  .score_metric_registry(payload$task)
  if (!is.null(payload$metrics) &&
      !.private_metrics_valid(payload$metrics, payload$task)) {
    stop("Private metric release is malformed.", call. = FALSE)
  }
  payload
}

#' Inspect and select private validation metrics
#'
#' These helpers operate only on an already released pooled metric object.
#' They do not contact data nodes, spend privacy budget, or create another
#' release. The catalog contains the scalar metrics suitable for model
#' selection for the result's task. Diagnostic mass (`n`) and structured
#' outputs such as curves, calibration bins, label details, and confusion
#' matrices remain available in `result$metrics`, but cannot be selected as an
#' HPO score.
#'
#' A metric can be unavailable when the released sufficient statistics do not
#' support a finite denominator. `ds.flower.score()` fails explicitly in that
#' case instead of inventing or coercing a value.
#'
#' @param result A `dsflower_validation`, `dsflower_cv`, or `dsflower_run`
#'   containing an atomic holdout release.
#' @return `ds.flower.metrics()` returns a data frame with `task`, `metric`,
#'   `direction`, `available`, and `value`. `ds.flower.score()` returns one
#'   finite numeric scalar. `ds.flower.metric_direction()` returns
#'   `"minimize"` or `"maximize"`.
#' @export
ds.flower.metrics <- function(result) {
  payload <- .private_metric_payload(result)
  directions <- .score_metric_registry(payload$task)
  values <- rep(NA_real_, length(directions))
  available <- rep(FALSE, length(directions))

  if (!is.null(payload$metrics)) {
    for (index in seq_along(directions)) {
      name <- names(directions)[[index]]
      value <- payload$metrics[[name]]
      if (is.null(value)) next
      if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
          is.na(value) || !is.finite(value)) {
        stop("Private metric release is malformed at '", name, "'.",
             call. = FALSE)
      }
      values[[index]] <- as.numeric(value)
      available[[index]] <- isTRUE(payload$released)
    }
  }

  data.frame(
    task = rep(payload$task, length(directions)),
    metric = names(directions),
    direction = unname(directions),
    available = available,
    value = values,
    stringsAsFactors = FALSE
  )
}

#' @rdname ds.flower.metrics
#' @param metric One scalar metric name from `ds.flower.metrics(result)`.
#' @export
ds.flower.score <- function(result, metric) {
  metric <- .score_metric_name(metric)
  catalog <- ds.flower.metrics(result)
  index <- match(metric, catalog$metric)
  if (is.na(index)) {
    stop("Metric '", metric, "' is not a scoreable scalar for task '",
         unique(catalog$task), "'. Choose one of: ",
         paste(catalog$metric, collapse = ", "), ".", call. = FALSE)
  }
  if (!isTRUE(catalog$available[[index]])) {
    stop("Metric '", metric, "' is unavailable in this pooled DP release.",
         call. = FALSE)
  }
  catalog$value[[index]]
}

#' @rdname ds.flower.metrics
#' @param task Validation task: `"binary"`, `"multiclass"`, `"ordinal"`,
#'   `"multilabel"`, `"regression"`, or `"count"`.
#' @export
ds.flower.metric_direction <- function(metric, task) {
  metric <- .score_metric_name(metric)
  directions <- .score_metric_registry(task)
  direction <- directions[metric]
  if (is.na(direction)) {
    stop("Metric '", metric, "' is not a scoreable scalar for task '", task,
         "'. Choose one of: ", paste(names(directions), collapse = ", "), ".",
         call. = FALSE)
  }
  unname(direction)
}
