# Module: Task Specs
# Task specification objects for federated learning.

#' Create a classification task specification
#'
#' @return A \code{dsflower_task} S3 object with type = "classification".
#' @export
ds.flower.task.classification <- function() {
  obj <- list(type = "classification")
  class(obj) <- "dsflower_task"
  obj
}

#' Create a regression task specification
#'
#' @return A \code{dsflower_task} S3 object with type = "regression".
#' @export
ds.flower.task.regression <- function() {
  obj <- list(type = "regression")
  class(obj) <- "dsflower_task"
  obj
}

#' Create a count-outcome task specification
#'
#' @return A \code{dsflower_task} S3 object with type = "count".
#' @export
ds.flower.task.count <- function() {
  obj <- list(type = "count")
  class(obj) <- "dsflower_task"
  obj
}

.assert_supported_task <- function(task) {
  if (!inherits(task, "dsflower_task") ||
      !task$type %in% c("classification", "regression", "count")) {
    type <- if (inherits(task, "dsflower_task")) task$type else "invalid"
    stop("Task '", type, "' is not supported by the enforced-DP runtime. ",
         "Supported tasks: classification, regression, count.", call. = FALSE)
  }
  invisible(task)
}

#' Print a dsflower_task
#' @param x A dsflower_task object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns x.
#' @export
print.dsflower_task <- function(x, ...) {
  cat("dsflower_task:", x$type, "\n")
  invisible(x)
}
