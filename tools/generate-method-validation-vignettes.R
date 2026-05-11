#!/usr/bin/env Rscript

result_path <- file.path("inst", "extdata", "dsflower_method_validation_results.json")
if (!file.exists(result_path)) {
  stop("Run inst/demos/validate_methods.R before generating validation vignettes.", call. = FALSE)
}

validation <- jsonlite::fromJSON(result_path)
results <- validation$results
vision_result_path <- file.path("inst", "extdata", "dsflower_vision_validation_results.json")
vision_validation <- if (file.exists(vision_result_path)) {
  jsonlite::fromJSON(vision_result_path)
} else {
  NULL
}
vision_results <- if (!is.null(vision_validation)) vision_validation$results else NULL

slug <- function(method) gsub("_", "-", method, fixed = TRUE)

write_lines <- function(path, lines) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, path, useBytes = TRUE)
}

method_vignette <- function(method) {
  title <- paste0("Validation: ", method)
  c(
    "---",
    paste0("title: \"", title, "\""),
    "output: rmarkdown::html_vignette",
    "vignette: >",
    paste0("  %\\VignetteIndexEntry{", title, "}"),
    "  %\\VignetteEngine{knitr::rmarkdown}",
    "  %\\VignetteEncoding{UTF-8}",
    "---",
    "",
    "```{r, include=FALSE}",
    "knitr::opts_chunk$set(collapse = TRUE, comment = '#>')",
    paste0("method_id <- \"", method, "\""),
    "paths <- c(",
    "  file.path('..', 'inst', 'extdata', 'dsflower_method_validation_results.json'),",
    "  file.path('inst', 'extdata', 'dsflower_method_validation_results.json'),",
    "  system.file('extdata', 'dsflower_method_validation_results.json', package = 'dsFlowerClient')",
    ")",
    "result_path <- paths[nzchar(paths) & file.exists(paths)][1]",
    "validation <- jsonlite::fromJSON(result_path)",
    "results <- validation$results",
    "row <- results[results$method == method_id, , drop = FALSE]",
    "fmt <- function(x) {",
    "  if (length(x) == 0 || is.null(x) || is.na(x)) return('NA')",
    "  if (is.numeric(x)) return(format(round(x, 4), nsmall = 4, trim = TRUE))",
    "  as.character(x)",
    "}",
    "```",
    "",
    "This vignette records a live validation of the method against a centralized baseline.",
    "The federated run used three Opal/Rock nodes and the same pooled synthetic",
    "validation cohort was used for the local baseline. The purpose is method-path",
    "validation: data staging, template verification, Flower execution, result",
    "persistence, and cleanup. It is not a clinical performance benchmark.",
    "",
    "```{r}",
    "overview <- data.frame(",
    "  Field = c(",
    "    'method', 'task', 'target', 'n_features', 'rounds',",
    "    'centralized_metric', 'centralized_loss', 'centralized_accuracy',",
    "    'federated_status', 'federated_loss', 'federated_n_failures',",
    "    'delta_loss', 'acceptance_max_loss_ratio',",
    "    'acceptance_max_loss_margin', 'acceptable_loss',",
    "    'validation_status'",
    "  ),",
    "  Value = c(",
    "    row$method, row$task, row$target, row$n_features, row$rounds,",
    "    row$centralized_metric, fmt(row$centralized_loss), fmt(row$centralized_accuracy),",
    "    row$federated_status, fmt(row$federated_loss), fmt(row$federated_n_failures),",
    "    fmt(row$delta_loss), fmt(row$acceptance_max_loss_ratio),",
    "    fmt(row$acceptance_max_loss_margin), row$acceptable_loss,",
    "    row$validation_status",
    "  )",
    ")",
    "knitr::kable(overview)",
    "```",
    "",
    paste0("```{r, fig.width=5.5, fig.height=3.2, fig.alt=\"Bar chart comparing centralized and federated validation loss for ", method, ".\"}"),
    "loss_df <- data.frame(",
    "  mode = c('Centralized', 'Federated'),",
    "  loss = c(row$centralized_loss, row$federated_loss)",
    ")",
    "loss_df <- loss_df[is.finite(loss_df$loss), , drop = FALSE]",
    "if (nrow(loss_df) > 0 && requireNamespace('ggplot2', quietly = TRUE)) {",
    "  ggplot2::ggplot(loss_df, ggplot2::aes(x = mode, y = loss, fill = mode)) +",
    "    ggplot2::geom_col(width = 0.65, show.legend = FALSE) +",
    "    ggplot2::labs(x = NULL, y = row$centralized_metric[[1]], title = method_id) +",
    "    ggplot2::theme_minimal(base_size = 11)",
    "} else {",
    "  plot.new()",
    "  text(0.5, 0.5, 'No federated loss available for this method')",
    "}",
    "```",
    "",
    "To reproduce only this method against configured Opal servers:",
    "",
    "```sh",
    paste0("DSFLOWER_VALIDATE_METHODS=", method, " Rscript inst/demos/validate_methods.R"),
    "```",
    "",
    "```{r}",
    "if (nzchar(row$error)) cat(row$error)",
    "```"
  )
}

article_link_lines <- function(methods, vision_methods = character()) {
  lines <- c("Per-method vignettes:", "")
  lines <- c(lines, paste0("- [", methods, "](validation-", slug(methods), ".html)"))
  if (length(vision_methods)) {
    lines <- c(lines, "", "Vision method vignettes:", "")
    lines <- c(lines, paste0(
      "- [", vision_methods, "](validation-vision-", slug(vision_methods), ".html)"
    ))
  }
  lines
}

overview_vignette <- function(methods, vision_methods = character()) {
  c(
    "---",
    "title: \"All Method Validation Overview\"",
    "output: rmarkdown::html_vignette",
    "vignette: >",
    "  %\\VignetteIndexEntry{All Method Validation Overview}",
    "  %\\VignetteEngine{knitr::rmarkdown}",
    "  %\\VignetteEncoding{UTF-8}",
    "---",
    "",
    "```{r, include=FALSE}",
    "knitr::opts_chunk$set(collapse = TRUE, comment = '#>')",
    "method_paths <- c(",
    "  file.path('..', 'inst', 'extdata', 'dsflower_method_validation_results.json'),",
    "  file.path('inst', 'extdata', 'dsflower_method_validation_results.json'),",
    "  system.file('extdata', 'dsflower_method_validation_results.json', package = 'dsFlowerClient')",
    ")",
    "method_result_path <- method_paths[nzchar(method_paths) & file.exists(method_paths)][1]",
    "method_validation <- jsonlite::fromJSON(method_result_path)",
    "method_results <- method_validation$results",
    "vision_paths <- c(",
    "  file.path('..', 'inst', 'extdata', 'dsflower_vision_validation_results.json'),",
    "  file.path('inst', 'extdata', 'dsflower_vision_validation_results.json'),",
    "  system.file('extdata', 'dsflower_vision_validation_results.json', package = 'dsFlowerClient')",
    ")",
    "vision_result_path <- vision_paths[nzchar(vision_paths) & file.exists(vision_paths)][1]",
    "vision_validation <- if (!is.na(vision_result_path)) jsonlite::fromJSON(vision_result_path) else NULL",
    "vision_results <- if (!is.null(vision_validation)) vision_validation$results else NULL",
    "method_display <- data.frame(",
    "  suite = 'tabular_sequence_survival',",
    "  method = method_results$method,",
    "  task = method_results$task,",
    "  centralized_metric = method_results$centralized_metric,",
    "  centralized_loss = method_results$centralized_loss,",
    "  federated_status = method_results$federated_status,",
    "  federated_loss = method_results$federated_loss,",
    "  delta_loss = method_results$delta_loss,",
    "  acceptable_loss = method_results$acceptable_loss,",
    "  validation_status = method_results$validation_status,",
    "  stringsAsFactors = FALSE",
    ")",
    "vision_display <- if (!is.null(vision_results)) data.frame(",
    "  suite = 'vision',",
    "  method = vision_results$method,",
    "  task = vision_results$task,",
    "  centralized_metric = vision_results$centralized_metric,",
    "  centralized_loss = vision_results$centralized_loss,",
    "  federated_status = vision_results$federated_status,",
    "  federated_loss = vision_results$federated_loss,",
    "  delta_loss = vision_results$delta_loss,",
    "  acceptable_loss = vision_results$acceptable_loss,",
    "  validation_status = vision_results$validation_status,",
    "  stringsAsFactors = FALSE",
    ") else method_display[0, ]",
    "results <- rbind(method_display, vision_display)",
    "```",
    "",
    "This macro-vignette summarizes the live dsFlower validation suite across",
    "tabular, sequence, survival, XGBoost, image-classification, and segmentation",
    "templates. Each method has its own vignette, and each federated run is",
    "compared with a centralized baseline trained on the corresponding pooled",
    "synthetic cohort.",
    "",
    "The suite is deliberately a functional and numerical sanity check, not a",
    "privacy claim and not a clinical benchmark. It uses `sandbox_open` so that",
    "per-node diagnostics can be observed during development. Production studies",
    "should use the stricter DataSHIELD privacy profiles. XGBoost is validated",
    "in this trusted demo mode; profiles that require Secure Aggregation still",
    "enforce SecAgg before execution.",
    "",
    "```{r}",
    "summary_table <- data.frame(",
    "  suite = c('tabular_sequence_survival', 'vision'),",
    "  generated_at = c(method_validation$generated_at, if (!is.null(vision_validation)) vision_validation$generated_at else NA),",
    "  privacy_profile = c(method_validation$privacy_profile, if (!is.null(vision_validation)) vision_validation$privacy_profile else NA),",
    "  n_sites = c(method_validation$n_sites, if (!is.null(vision_validation)) vision_validation$n_sites else NA),",
    "  n_total = c(method_validation$n_total, if (!is.null(vision_validation)) vision_validation$n_total else NA),",
    "  secagg_supported = c(method_validation$secagg_supported, NA),",
    "  stringsAsFactors = FALSE",
    ")",
    "knitr::kable(summary_table)",
    "```",
    "",
    "```{r}",
    "knitr::kable(results)",
    "```",
    "",
    "```{r, fig.width=8, fig.height=5, fig.alt=\"Horizontal bar chart of federated minus centralized loss by method.\"}",
    "plot_df <- results[is.finite(results$delta_loss), , drop = FALSE]",
    "if (nrow(plot_df) > 0 && requireNamespace('ggplot2', quietly = TRUE)) {",
    "  plot_df$label <- paste(plot_df$suite, plot_df$method, sep = ': ')",
    "  ggplot2::ggplot(plot_df, ggplot2::aes(x = stats::reorder(label, delta_loss), y = delta_loss, fill = task)) +",
    "    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, color = 'grey40') +",
    "    ggplot2::geom_col(width = 0.7) +",
    "    ggplot2::coord_flip() +",
    "    ggplot2::labs(x = NULL, y = 'Federated loss - centralized loss') +",
    "    ggplot2::theme_minimal(base_size = 11)",
    "}",
    "```",
    "",
    article_link_lines(methods, vision_methods),
    "",
    "The vision-specific overview remains available at",
    "[Vision Method Validation Overview](validation-vision-overview.html).",
    "",
    "To reproduce the complete suite against configured Opal servers:",
    "",
    "```sh",
    "Rscript inst/demos/validate_methods.R",
    "Rscript inst/demos/validate_vision_methods.R",
    "```"
  )
}

for (method in results$method) {
  path <- file.path("vignettes", paste0("validation-", slug(method), ".Rmd"))
  write_lines(path, method_vignette(method))
}

write_lines(file.path("vignettes", "validation-overview.Rmd"),
            overview_vignette(
              results$method,
              if (!is.null(vision_results)) vision_results$method else character()
            ))

message("Generated ", length(results$method) + 1L, " validation vignettes.")
