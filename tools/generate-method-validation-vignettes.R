#!/usr/bin/env Rscript

result_path <- file.path("inst", "extdata", "dsflower_method_validation_results.json")
if (!file.exists(result_path)) {
  stop("Run inst/demos/validate_methods.R before generating validation vignettes.", call. = FALSE)
}

validation <- jsonlite::fromJSON(result_path)
results <- validation$results

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
    "    'delta_loss', 'validation_status'",
    "  ),",
    "  Value = c(",
    "    row$method, row$task, row$target, row$n_features, row$rounds,",
    "    row$centralized_metric, fmt(row$centralized_loss), fmt(row$centralized_accuracy),",
    "    row$federated_status, fmt(row$federated_loss), fmt(row$federated_n_failures),",
    "    fmt(row$delta_loss), row$validation_status",
    "  )",
    ")",
    "knitr::kable(overview)",
    "```",
    "",
    "```{r, fig.width=5.5, fig.height=3.2}",
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

overview_vignette <- function(methods) {
  c(
    "---",
    "title: \"Method Validation Overview\"",
    "output: rmarkdown::html_vignette",
    "vignette: >",
    "  %\\VignetteIndexEntry{Method Validation Overview}",
    "  %\\VignetteEngine{knitr::rmarkdown}",
    "  %\\VignetteEncoding{UTF-8}",
    "---",
    "",
    "```{r, include=FALSE}",
    "knitr::opts_chunk$set(collapse = TRUE, comment = '#>')",
    "paths <- c(",
    "  file.path('..', 'inst', 'extdata', 'dsflower_method_validation_results.json'),",
    "  file.path('inst', 'extdata', 'dsflower_method_validation_results.json'),",
    "  system.file('extdata', 'dsflower_method_validation_results.json', package = 'dsFlowerClient')",
    ")",
    "result_path <- paths[nzchar(paths) & file.exists(paths)][1]",
    "validation <- jsonlite::fromJSON(result_path)",
    "results <- validation$results",
    "```",
    "",
    "This macro-vignette summarizes the live dsFlower method-validation run.",
    "The experiment created one synthetic cohort, split it across three Opal/Rock",
    "servers, trained each supported federated template, and compared the final",
    "federated loss with a centralized baseline trained on the pooled cohort.",
    "",
    "The suite is deliberately a functional and numerical sanity check, not a",
    "privacy claim and not a clinical benchmark. It uses `sandbox_open` so that",
    "per-node diagnostics can be observed during development. Production studies",
    "should use the stricter DataSHIELD privacy profiles.",
    "",
    "```{r}",
    "summary_table <- data.frame(",
    "  Field = c('generated_at', 'privacy_profile', 'n_sites', 'n_per_site', 'n_total', 'secagg_supported'),",
    "  Value = c(validation$generated_at, validation$privacy_profile, validation$n_sites,",
    "            validation$n_per_site, validation$n_total, validation$secagg_supported)",
    ")",
    "knitr::kable(summary_table)",
    "```",
    "",
    "```{r}",
    "display <- results[, c(",
    "  'method', 'task', 'centralized_metric', 'centralized_loss',",
    "  'federated_status', 'federated_loss', 'delta_loss', 'validation_status'",
    ")]",
    "knitr::kable(display)",
    "```",
    "",
    "```{r, fig.width=8, fig.height=5}",
    "plot_df <- results[is.finite(results$delta_loss), , drop = FALSE]",
    "if (nrow(plot_df) > 0 && requireNamespace('ggplot2', quietly = TRUE)) {",
    "  ggplot2::ggplot(plot_df, ggplot2::aes(x = stats::reorder(method, delta_loss), y = delta_loss, fill = task)) +",
    "    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, color = 'grey40') +",
    "    ggplot2::geom_col(width = 0.7) +",
    "    ggplot2::coord_flip() +",
    "    ggplot2::labs(x = NULL, y = 'Federated loss - centralized loss') +",
    "    ggplot2::theme_minimal(base_size = 11)",
    "}",
    "```",
    "",
    "The image-classification validation with images sourced from dsImaging is",
    "documented separately in `direct-image-resnet`. XGBoost is included here",
    "as a security guardrail check: centralized XGBoost trains locally, but the",
    "federated template is blocked unless server-side Secure Aggregation is",
    "available.",
    "",
    "To reproduce the complete suite against configured Opal servers:",
    "",
    "```sh",
    "Rscript inst/demos/validate_methods.R",
    "```"
  )
}

for (method in results$method) {
  path <- file.path("vignettes", paste0("validation-", slug(method), ".Rmd"))
  write_lines(path, method_vignette(method))
}

write_lines(file.path("vignettes", "validation-overview.Rmd"),
            overview_vignette(results$method))

message("Generated ", length(results$method) + 1L, " validation vignettes.")
