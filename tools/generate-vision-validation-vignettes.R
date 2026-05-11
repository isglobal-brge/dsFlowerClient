#!/usr/bin/env Rscript

result_path <- file.path("inst", "extdata", "dsflower_vision_validation_results.json")
if (!file.exists(result_path)) {
  stop("Run inst/demos/validate_vision_methods.R before generating vision vignettes.",
       call. = FALSE)
}

validation <- jsonlite::fromJSON(result_path)
results <- validation$results

slug <- function(method) gsub("_", "-", method, fixed = TRUE)

write_lines <- function(path, lines) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, path, useBytes = TRUE)
}

method_vignette <- function(method) {
  title <- paste0("Vision Validation: ", method)
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
    "  file.path('..', 'inst', 'extdata', 'dsflower_vision_validation_results.json'),",
    "  file.path('inst', 'extdata', 'dsflower_vision_validation_results.json'),",
    "  system.file('extdata', 'dsflower_vision_validation_results.json', package = 'dsFlowerClient')",
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
    "This vignette records a live validation of a dsFlower vision template.",
    "Synthetic PNG images were split across three Opal/Rock nodes, trained",
    "federatively through DataSHIELD, and compared with a centralized PyTorch",
    "baseline trained on the pooled synthetic cohort. This is a path validation",
    "for image staging, template execution, result persistence, and cleanup;",
    "it is not a clinical imaging benchmark.",
    "",
    "```{r}",
    "overview <- data.frame(",
    "  Field = c(",
    "    'method', 'task', 'target', 'n_total', 'image_size', 'rounds',",
    "    'centralized_metric', 'centralized_loss', 'centralized_accuracy',",
    "    'centralized_dice', 'federated_status', 'federated_loss',",
    "    'federated_n_failures', 'delta_loss',",
    "    'acceptance_max_loss_ratio', 'acceptance_max_loss_margin',",
    "    'acceptable_loss', 'validation_status'",
    "  ),",
    "  Value = c(",
    "    row$method, row$task, row$target, row$n_total, validation$image_size, row$rounds,",
    "    row$centralized_metric, fmt(row$centralized_loss), fmt(row$centralized_accuracy),",
    "    fmt(row$centralized_dice), row$federated_status, fmt(row$federated_loss),",
    "    fmt(row$federated_n_failures), fmt(row$delta_loss),",
    "    fmt(row$acceptance_max_loss_ratio), fmt(row$acceptance_max_loss_margin),",
    "    row$acceptable_loss, row$validation_status",
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
    paste0("DSFLOWER_VALIDATE_VISION_METHODS=", method, " Rscript inst/demos/validate_vision_methods.R"),
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
    "title: \"Vision Method Validation Overview\"",
    "output: rmarkdown::html_vignette",
    "vignette: >",
    "  %\\VignetteIndexEntry{Vision Method Validation Overview}",
    "  %\\VignetteEngine{knitr::rmarkdown}",
    "  %\\VignetteEncoding{UTF-8}",
    "---",
    "",
    "```{r, include=FALSE}",
    "knitr::opts_chunk$set(collapse = TRUE, comment = '#>')",
    "paths <- c(",
    "  file.path('..', 'inst', 'extdata', 'dsflower_vision_validation_results.json'),",
    "  file.path('inst', 'extdata', 'dsflower_vision_validation_results.json'),",
    "  system.file('extdata', 'dsflower_vision_validation_results.json', package = 'dsFlowerClient')",
    ")",
    "result_path <- paths[nzchar(paths) & file.exists(paths)][1]",
    "validation <- jsonlite::fromJSON(result_path)",
    "results <- validation$results",
    "```",
    "",
    "This macro-vignette summarizes the live dsFlower vision-template validation.",
    "The experiment uses synthetic image and mask files staged on the Rock",
    "servers, while metadata remains in Opal tables. Each federated run is",
    "compared with a centralized PyTorch baseline trained over the pooled files.",
    "",
    "```{r}",
    "summary_table <- data.frame(",
    "  Field = c('generated_at', 'privacy_profile', 'n_sites', 'n_per_site', 'n_total', 'image_size'),",
    "  Value = c(validation$generated_at, validation$privacy_profile, validation$n_sites,",
    "            validation$n_per_site, validation$n_total, validation$image_size)",
    ")",
    "knitr::kable(summary_table)",
    "```",
    "",
    "```{r}",
    "display <- results[, c(",
    "  'method', 'task', 'centralized_metric', 'centralized_loss',",
    "  'federated_status', 'federated_loss', 'delta_loss',",
    "  'acceptable_loss', 'validation_status'",
    ")]",
    "knitr::kable(display)",
    "```",
    "",
    "```{r, fig.width=7, fig.height=3.5, fig.alt=\"Horizontal bar chart of vision-template federated minus centralized loss.\"}",
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
    "To reproduce the complete vision suite against configured Opal servers:",
    "",
    "```sh",
    "Rscript inst/demos/validate_vision_methods.R",
    "```"
  )
}

for (method in results$method) {
  path <- file.path("vignettes", paste0("validation-vision-", slug(method), ".Rmd"))
  write_lines(path, method_vignette(method))
}

write_lines(file.path("vignettes", "validation-vision-overview.Rmd"),
            overview_vignette(results$method))

message("Generated ", length(results$method) + 1L, " vision validation vignettes.")
