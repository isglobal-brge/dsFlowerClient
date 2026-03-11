# Module: Result Objects
# dsflower_result construction, printing, and utility functions.

#' Create a dsflower_result object
#'
#' @param per_site Named list mapping server names to their raw results.
#' @param pooled NULL (default) or a single aggregated result.
#' @param meta Named list of metadata.
#' @return A \code{dsflower_result} object.
#' @keywords internal
dsflower_result <- function(per_site, pooled = NULL, meta = list()) {
  obj <- list(
    per_site = per_site,
    pooled   = pooled,
    meta     = list(
      call_code      = meta$call_code %||% "",
      timestamp      = Sys.time(),
      servers        = names(per_site),
      scope          = meta$scope %||% "per_site",
      pooling_policy = meta$pooling_policy %||% "strict",
      warnings       = meta$warnings %||% character(0)
    )
  )
  class(obj) <- c("dsflower_result", "list")
  obj
}

#' Print a dsflower_result
#'
#' @param x A dsflower_result object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns x.
#' @export
print.dsflower_result <- function(x, ...) {
  cat("dsflower_result\n")
  cat("  Servers:", paste(x$meta$servers, collapse = ", ") %||% "(none)", "\n")
  cat("  Scope:  ", x$meta$scope, "\n")
  if (!is.null(x$pooled)) {
    cat("  Pooled:  yes\n")
  } else {
    cat("  Pooled:  no\n")
  }
  if (length(x$meta$warnings) > 0) {
    cat("  Warnings:", length(x$meta$warnings), "\n")
    for (w in x$meta$warnings) cat("    - ", w, "\n")
  }
  if (nchar(x$meta$call_code) > 0) {
    cat("  Code:   ", substr(x$meta$call_code, 1, 80),
        if (nchar(x$meta$call_code) > 80) "..." else "", "\n")
  }
  invisible(x)
}

#' Access dsflower_result elements
#'
#' @param x A dsflower_result object.
#' @param name Character; the element name to access.
#' @return The requested element.
#' @export
`$.dsflower_result` <- function(x, name) {
  if (name %in% c("per_site", "pooled", "meta")) return(.subset2(x, name))
  ps <- .subset2(x, "per_site")
  if (name %in% names(ps)) return(ps[[name]])
  .subset2(x, name)
}

#' Convert dsflower_result to data.frame
#'
#' @param x A dsflower_result object.
#' @param ... Additional arguments (ignored).
#' @return A data frame.
#' @export
as.data.frame.dsflower_result <- function(x, ...) {
  if (!is.null(x$pooled) && is.data.frame(x$pooled)) {
    return(x$pooled)
  }
  ps <- x$per_site
  if (length(ps) > 0) {
    first <- ps[[1]]
    if (is.data.frame(first)) return(first)
  }
  data.frame()
}

#' Get the R code that produced a result
#'
#' @param x A dsflower_result object.
#' @return Character string containing the reproducible R code.
#' @export
ds.flower.code <- function(x) {
  if (!inherits(x, "dsflower_result")) {
    stop("ds.flower.code() requires a dsflower_result object", call. = FALSE)
  }
  x$meta$call_code
}

#' Copy reproducible R code to clipboard
#'
#' @param x A dsflower_result object.
#' @return Invisibly returns the code string.
#' @export
ds.flower.copy_code <- function(x) {
  code <- ds.flower.code(x)
  tryCatch({
    if (requireNamespace("clipr", quietly = TRUE)) {
      clipr::write_clip(code)
      message("Code copied to clipboard.")
    } else {
      message("Install the 'clipr' package for clipboard support.")
      message("Code:\n", code)
    }
  }, error = function(e) {
    message("Could not copy to clipboard: ", conditionMessage(e))
    message("Code:\n", code)
  })
  invisible(code)
}
