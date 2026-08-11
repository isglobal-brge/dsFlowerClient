# Module: Private validation metric selection
# Keeps metric discovery and HPO score extraction as local post-processing.

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
      (!is.list(payload$metrics) || !length(payload$metrics) ||
       is.null(names(payload$metrics)) || anyNA(names(payload$metrics)) ||
       any(!nzchar(names(payload$metrics))) ||
       anyDuplicated(names(payload$metrics)))) {
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
