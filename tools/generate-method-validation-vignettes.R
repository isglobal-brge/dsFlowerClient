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
  row_meta <- results[results$method == method, , drop = FALSE]
  target_parts <- trimws(strsplit(row_meta$target, ",", fixed = TRUE)[[1]])
  target_lit <- if (length(target_parts) == 1L) {
    paste0("'", target_parts, "'")
  } else {
    paste0("c(", paste(paste0("'", target_parts, "'"), collapse = ", "), ")")
  }
  features_lit <- paste0("paste0('x', 1:", row_meta$n_features, ")")
  rounds_lit <- as.integer(row_meta$rounds)
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
    "This vignette records the catalogue coverage check for this authorised",
    "template. The federated run used three Opal/Rock nodes and the same pooled",
    "synthetic validation cohort was used for the local baseline. The purpose is",
    "method-path validation: data staging, template verification, Flower",
    "execution, result persistence, and cleanup. It complements the biomedical",
    "reports by checking the full template surface under deterministic",
    "conditions.",
    "",
    "It is laid out in two parts: first the commands you would run to",
    "reproduce the validation, then the recorded results of that run loaded",
    "from the committed evidence artifact. No live Opal servers are required to",
    "render this page; the command chunks are shown for reference and are not",
    "executed.",
    "",
    "## Running This Validation",
    "",
    "To check this template we compare a **centralized** baseline against the",
    "**federated** dsFlower run. The validation cohort is one synthetic dataset:",
    "the centralized baseline is fit on the pooled copy, and the very same rows",
    "are partitioned across three local Opal nodes that are already set up and",
    "serving the table `dsflower_demo.validation_cohort`. The federated fit never",
    "sees the pooled data, only the three node-resident partitions.",
    "",
    "**1. Connect to the DataSHIELD nodes.** Open a single session across the",
    "three Opal nodes that hold the partitioned validation cohort.",
    "",
    "```{r connect, eval=FALSE}",
    "library(DSI)",
    "library(DSOpal)",
    "library(dsFlowerClient)",
    "",
    "builder <- DSI::newDSLoginBuilder()",
    "builder$append(server = 'node1', url = 'https://opal-node-1.example.org',",
    "               user = 'researcher', password = '********',",
    "               table = 'dsflower_demo.validation_cohort')",
    "builder$append(server = 'node2', url = 'https://opal-node-2.example.org',",
    "               user = 'researcher', password = '********',",
    "               table = 'dsflower_demo.validation_cohort')",
    "builder$append(server = 'node3', url = 'https://opal-node-3.example.org',",
    "               user = 'researcher', password = '********',",
    "               table = 'dsflower_demo.validation_cohort')",
    "conns <- DSI::datashield.login(builder$build(), assign = TRUE, symbol = 'D')",
    "```",
    "",
    "**2. Run the federated fit.** Train this template through dsFlower with",
    "FedAvg over the server-side table, then log out. Only aggregate training",
    "history and model parameters are returned to the researcher.",
    "",
    "```{r federated-fit, eval=FALSE}",
    "fit <- ds.flower.fit(",
    "  conns,",
    "  symbol   = 'D',",
    paste0("  target   = ", target_lit, ","),
    paste0("  features = ", features_lit, ","),
    paste0("  model    = '", method, "',"),
    "  strategy = 'fedavg',",
    "  privacy  = 'sandbox_open',",
    paste0("  rounds   = ", rounds_lit),
    ")",
    "",
    "DSI::datashield.logout(conns)",
    "```",
    "",
    "## Recorded Results",
    "",
    "The values below come from the committed evidence artifact for the run",
    "described above; no live servers are contacted when the article renders.",
    "",
    "This run trained the `r row$method` template on the `r row$task` task",
    "(target `r row$target`, `r row$n_features` synthetic features) for",
    "`r row$rounds` federated round(s) under the `r validation$privacy_profile`",
    "profile, across `r validation$n_sites` Opal/Rock nodes holding the same",
    "pooled cohort used for the centralized baseline.",
    "",
    "The centralized baseline reached a `r row$centralized_metric` of",
    "`r fmt(row$centralized_loss)`; the federated run reached",
    "`r fmt(row$federated_loss)` with `r row$federated_n_failures` client",
    "failures, leaving this template **`r row$validation_status`**. The",
    "acceptance comparison is shown below, followed by the centralized vs.",
    "federated loss.",
    "",
    "```{r acceptance-output}",
    "max_allowed_loss <- row$centralized_loss * row$acceptance_max_loss_ratio +",
    "  row$acceptance_max_loss_margin",
    "acceptance_output <- data.frame(",
    "  centralized_loss = row$centralized_loss,",
    "  federated_loss = row$federated_loss,",
    "  max_allowed_loss = max_allowed_loss,",
    "  federated_n_failures = row$federated_n_failures,",
    "  acceptable_loss = row$acceptable_loss,",
    "  validation_status = row$validation_status",
    ")",
    "knitr::kable(acceptance_output, digits = 4)",
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
    "title: \"Catalogue Method Validation Overview\"",
    "output: rmarkdown::html_vignette",
    "vignette: >",
    "  %\\VignetteIndexEntry{Catalogue Method Validation Overview}",
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
    "This macro-vignette summarizes the catalogue coverage suite across",
    "tabular, sequence, survival, XGBoost, image-classification, and segmentation",
    "templates. Each method has its own vignette, and each federated run is",
    "compared with a centralized baseline trained on the corresponding pooled",
    "synthetic cohort.",
    "",
    "The suite is deliberately a functional and numerical sanity check, not a",
    "privacy claim and not a clinical benchmark. It uses `sandbox_open` so that",
    "per-node diagnostics can be observed during development. Production studies",
    "should use the stricter DataSHIELD privacy profiles. Current presentation",
    "evidence is reported in the public clinical, SUPPORT2, LUNG1 and PathMNIST",
    "vignettes. XGBoost is validated here in trusted demo mode; profiles that",
    "require Secure Aggregation still enforce SecAgg before execution.",
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
    "The full suite is produced by repeating the inline Opal/DataSHIELD/Flower",
    "workflow shown in each per-method vignette for every method listed above.",
    "The persisted JSON artifacts are the audit trail consumed by this",
    "pkgdown site."
  )
}

for (method in results$method) {
  path <- file.path("vignettes", paste0("validation-", slug(method), ".Rmd"))
  write_lines(path, method_vignette(method))
}

if (tolower(Sys.getenv("DSFLOWER_REGENERATE_CATALOGUE_OVERVIEW", "false")) %in%
    c("true", "1", "yes")) {
  write_lines(file.path("vignettes", "validation-overview.Rmd"),
              overview_vignette(
                results$method,
                if (!is.null(vision_results)) vision_results$method else character()
              ))
  message("Generated ", length(results$method) + 1L, " validation vignettes.")
} else {
  message("Generated ", length(results$method), " per-method validation vignettes. ",
          "Kept vignettes/validation-overview.Rmd unchanged because it is the ",
          "current evidence overview.")
}
