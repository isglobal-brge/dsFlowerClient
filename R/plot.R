# Module: Plotting
# Training curve visualization using plotly.

#' Plot training curves
#'
#' @param result A \code{dsflower_result} object or comparison data.frame.
#' @param metric Character; which metric to plot (default "loss").
#' @param per_server Logical; show per-server curves (default FALSE).
#' @param title Character; plot title (default auto-generated).
#' @return A plotly object.
#' @export
ds.flower.plot <- function(result,
                           metric = "loss",
                           per_server = FALSE,
                           title = NULL) {
  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("The plotly package is required for plotting.", call. = FALSE)
  }

  if (is.data.frame(result) && "run" %in% names(result)) {
    return(.plot_comparison(result, metric, title))
  }

  if (!inherits(result, "dsflower_result")) {
    stop("result must be a dsflower_result or comparison data.frame",
         call. = FALSE)
  }

  if (per_server) {
    return(.plot_server_comparison(result, metric, title))
  }

  .plot_training_curves(result, metric, title)
}

#' @keywords internal
.plot_training_curves <- function(result, metric = "loss", title = NULL) {
  df <- if (!is.null(result$pooled) && is.data.frame(result$pooled)) {
    result$pooled
  } else if (length(result$per_site) > 0) {
    first <- result$per_site[[1]]
    if (is.data.frame(first)) first else data.frame()
  } else {
    data.frame()
  }

  if (nrow(df) == 0 || !"metric" %in% names(df)) {
    return(plotly::plot_ly() |>
      plotly::layout(title = title %||% "No metrics available"))
  }

  df <- df[tolower(df$metric) == tolower(metric), , drop = FALSE]
  if (nrow(df) == 0) {
    return(plotly::plot_ly() |>
      plotly::layout(title = paste("No data for metric:", metric)))
  }

  df <- df[order(df$round), ]
  title <- title %||% paste("Training:", metric)

  plotly::plot_ly(df, x = ~round, y = ~value, type = "scatter",
                  mode = "lines+markers",
                  line = list(width = 2),
                  marker = list(size = 6)) |>
    plotly::layout(
      title = title,
      xaxis = list(title = "Round"),
      yaxis = list(title = metric),
      template = "plotly_white"
    )
}

#' @keywords internal
.plot_server_comparison <- function(result, metric = "loss", title = NULL) {
  title <- title %||% paste("Per-Server:", metric)
  p <- plotly::plot_ly()

  for (srv in names(result$per_site)) {
    df <- result$per_site[[srv]]
    if (!is.data.frame(df) || nrow(df) == 0) next
    if (!"metric" %in% names(df)) next

    srv_df <- df[tolower(df$metric) == tolower(metric), , drop = FALSE]
    if (nrow(srv_df) == 0) next
    srv_df <- srv_df[order(srv_df$round), ]

    p <- p |> plotly::add_trace(
      data = srv_df, x = ~round, y = ~value,
      type = "scatter", mode = "lines+markers",
      name = srv,
      line = list(width = 2),
      marker = list(size = 5)
    )
  }

  p |> plotly::layout(
    title = title,
    xaxis = list(title = "Round"),
    yaxis = list(title = metric),
    template = "plotly_white"
  )
}

#' @keywords internal
.plot_comparison <- function(df, metric = "loss", title = NULL) {
  title <- title %||% paste("Run Comparison:", metric)

  df <- df[tolower(df$metric) == tolower(metric), , drop = FALSE]
  if (nrow(df) == 0) {
    return(plotly::plot_ly() |>
      plotly::layout(title = paste("No data for metric:", metric)))
  }

  p <- plotly::plot_ly()

  for (run in unique(df$run)) {
    run_df <- df[df$run == run, , drop = FALSE]
    run_df <- run_df[order(run_df$round), ]

    p <- p |> plotly::add_trace(
      data = run_df, x = ~round, y = ~value,
      type = "scatter", mode = "lines+markers",
      name = run,
      line = list(width = 2),
      marker = list(size = 5)
    )
  }

  p |> plotly::layout(
    title = title,
    xaxis = list(title = "Round"),
    yaxis = list(title = metric),
    template = "plotly_white"
  )
}
