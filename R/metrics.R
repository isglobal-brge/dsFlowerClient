# Module: Result comparison helpers.

#' Compare metrics across multiple training runs
#'
#' @param ... Named \code{dsflower_result} objects to compare.
#' @return A data.frame with columns: run, round, metric, value.
#' @export
ds.flower.compare <- function(...) {
  runs <- list(...)
  if (length(runs) == 0) {
    stop("At least one dsflower_result required.", call. = FALSE)
  }

  run_names <- names(runs)
  if (is.null(run_names)) {
    run_names <- paste0("run_", seq_along(runs))
  }

  dfs <- list()
  for (i in seq_along(runs)) {
    result <- runs[[i]]
    if (!inherits(result, "dsflower_result")) {
      warning("Argument ", i, " is not a dsflower_result, skipping.")
      next
    }

    df <- if (!is.null(result$pooled) && is.data.frame(result$pooled)) {
      result$pooled
    } else if (length(result$per_site) > 0) {
      first <- result$per_site[[1]]
      if (is.data.frame(first)) first else next
    } else {
      next
    }

    if (nrow(df) > 0 && all(c("round", "metric", "value") %in% names(df))) {
      df$run <- run_names[i]
      dfs[[length(dfs) + 1]] <- df[, c("run", "round", "metric", "value"),
                                    drop = FALSE]
    }
  }

  if (length(dfs) == 0) {
    return(data.frame(
      run = character(0), round = integer(0),
      metric = character(0), value = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  result <- do.call(rbind, dfs)
  rownames(result) <- NULL
  result
}
