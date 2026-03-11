# Module: Metrics Collection
# Retrieve, pool, and compare training metrics across servers.

#' Get training metrics from all servers
#'
#' @param symbol Character; Flower session symbol (default "flower").
#' @param since_round Integer; return only metrics from this round onward.
#' @param pool Logical; if TRUE, compute pooled metrics across servers.
#' @param conns DSI connections (required).
#' @return A \code{dsflower_result} object with training metrics.
#' @export
ds.flower.metrics <- function(symbol = "flower",
                              since_round = 0L,
                              pool = TRUE,
                              conns) {
  results <- .ds_safe_aggregate(
    conns,
    expr = call("flowerMetricsDS", symbol, as.integer(since_round))
  )

  pooled <- NULL
  if (pool && length(results) > 1) {
    pooled <- .pool_metrics(results)
  } else if (pool && length(results) == 1) {
    pooled <- results[[1]]
  }

  code <- .build_code("ds.flower.metrics",
    symbol = symbol,
    since_round = since_round
  )

  dsflower_result(
    per_site = results,
    pooled = pooled,
    meta = list(
      call_code = code,
      scope = if (pool) "pooled" else "per_site"
    )
  )
}

#' Get log output from all servers
#'
#' @param symbol Character; Flower session symbol (default "flower").
#' @param last_n Integer; number of log lines to return per server.
#' @param conns DSI connections (required).
#' @return A \code{dsflower_result} object with log lines per server.
#' @export
ds.flower.log <- function(symbol = "flower",
                          last_n = 50L,
                          conns) {
  results <- .ds_safe_aggregate(
    conns,
    expr = call("flowerLogDS", symbol, as.integer(last_n))
  )

  code <- .build_code("ds.flower.log",
    symbol = symbol,
    last_n = last_n
  )

  dsflower_result(
    per_site = results,
    meta = list(call_code = code, scope = "per_site")
  )
}

#' Pool metrics across servers
#'
#' @param results Named list of per-server metric data.frames.
#' @return A pooled data.frame with mean values.
#' @keywords internal
.pool_metrics <- function(results) {
  dfs <- list()
  for (srv in names(results)) {
    df <- results[[srv]]
    if (is.data.frame(df) && nrow(df) > 0 &&
        all(c("round", "metric", "value") %in% names(df))) {
      df$server <- srv
      dfs[[length(dfs) + 1]] <- df
    }
  }

  if (length(dfs) == 0) {
    return(data.frame(
      round = integer(0), metric = character(0),
      value = numeric(0), n_servers = integer(0),
      stringsAsFactors = FALSE
    ))
  }

  combined <- do.call(rbind, dfs)

  groups <- split(combined, list(combined$round, combined$metric))
  pooled_rows <- lapply(groups, function(grp) {
    if (nrow(grp) == 0) return(NULL)
    data.frame(
      round = grp$round[1],
      metric = grp$metric[1],
      value = mean(grp$value, na.rm = TRUE),
      n_servers = nrow(grp),
      stringsAsFactors = FALSE
    )
  })

  pooled <- do.call(rbind, Filter(Negate(is.null), pooled_rows))
  if (is.null(pooled)) {
    return(data.frame(
      round = integer(0), metric = character(0),
      value = numeric(0), n_servers = integer(0),
      stringsAsFactors = FALSE
    ))
  }

  rownames(pooled) <- NULL
  pooled[order(pooled$round, pooled$metric), , drop = FALSE]
}

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
