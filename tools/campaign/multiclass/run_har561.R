#!/usr/bin/env Rscript
# Train the frozen matrix, then open the official test tables exactly once.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
root <- normalizePath(args[1], mustWork = TRUE)
out <- args[2]
script_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(script_dir, "har561_campaign.R"))
source(file.path(script_dir, "har561.R"))
protocol <- jsonlite::fromJSON(file.path(script_dir, "har561_protocol.json"), simplifyVector = FALSE)
venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
stopifnot(nzchar(venv_root))
cache <- file.path(root, "data_cache")
stopifnot(identical(digest::digest(file = file.path(cache, "har561_official.zip"),
                                  algo = "sha256"), HAR561_SHA256))
run_dir <- file.path(root, "runs", "har561_multiclass")
stopifnot(!dir.exists(run_dir))
dir.create(run_dir, recursive = TRUE)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
utc <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
writeLines(utc(), file.path(run_dir, "STARTED"))
seeds <- 20260819L + 1:3
epsilons <- c(1, 8, 4)
model_params <- list(n_classes = 6L)  # Dataset output size; optimization defaults unchanged.
resolved <- dsFlowerClient:::.dsflower_get_model("pytorch_multiclass")$defaults
resolved$n_classes <- 6L
resolved <- dsFlowerClient:::.dsflower_complete_training_params(
  resolved, names(dsFlowerClient:::.dsflower_get_model("pytorch_multiclass")$parameter_types))
env_info <- campaign_env_info(venv_root)
stopifnot(env_info$versions$dsflower == "0.5.0", env_info$versions$dsflowerclient == "0.5.0")
env_info$host$pod_id <- "6aq9cxaigfwlby"
env_info$host$pod_name <- "pod-flower-multiclass"
env_info$host$vcpus <- 32L
env_info$versions$runner_sha256 <- dsFlowerClient:::.compute_local_runner_hash()
stopifnot(identical(env_info$versions$runner_sha256, dsFlower:::.compute_harness_hash()))
env_info$versions$dsflower_source_commit <- "408f08c539329e2711260050ab40a6567aa4d89e"
env_info$versions$dsflowerclient_release_commit <- "50dda000a32ffcbdd039c2b74c909df451392bfb"
env_info$versions$nnet <- as.character(packageVersion("nnet"))
script_paths <- file.path(script_dir, c("run_har561.R", "har561_campaign.R", "har561.R",
                                      "har561_protocol.json"))
script_hashes <- setNames(lapply(script_paths, function(p) digest::digest(file = p, algo = "sha256")),
                         basename(script_paths))
train <- har561_load(cache, "train")
features <- HAR561_FEATURES
bounds <- list(lower = rep(-1, 561L), upper = rep(1, 561L))
sites <- lapply(seeds, function(seed) har561_sites(train, seed))
central_models <- list()
central_seconds <- numeric(3)
for (r in seq_along(seeds)) {
  cat(sprintf("Central training replicate %d, seed %d; test remains unopened\n", r, seeds[r]))
  start <- Sys.time()
  central_models[[r]] <- har561_central_fit(train, seeds[r])
  central_seconds[r] <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  saveRDS(central_models[[r]], file.path(run_dir, sprintf("central_rep%d.rds", r)))
  cat(sprintf("Central replicate %d converged in %.1fs\n", r, central_seconds[r]))
}
trained <- list()
for (epsilon in epsilons) {
  cell_id <- sprintf("har561_pytorch_multiclass_eps%g", epsilon)
  stopifnot(!file.exists(file.path(out, paste0(cell_id, ".json"))))
  cell_dir <- file.path(run_dir, cell_id)
  stopifnot(!dir.exists(cell_dir))
  dir.create(cell_dir)
  writeLines(utc(), file.path(cell_dir, "STARTED"))
  trained[[as.character(epsilon)]] <- list()
  for (r in seq_along(seeds)) {
    cat(sprintf("HAR561 epsilon %g replicate %d training; test remains unopened\n", epsilon, r))
    fed <- campaign_run_federated(
      site_data = sites[[r]]$data, test = stop("Test access during training is forbidden"),
      features = features, feature_bounds = bounds, epsilon = epsilon, delta = 1e-6,
      rounds = 5L, model_params = model_params,
      work_dir = file.path(cell_dir, sprintf("rep%d", r)), venv_root = venv_root,
      contract = "pytorch_multiclass", score_function = function(fit, test, features) NULL)
    stopifnot(fed$n_clients == 3L, fed$n_failures == 0L, fed$n_rounds_run == 5L,
              fed$cleanup_ok, identical(fed$metadata$target_levels, HAR561_LEVELS),
              identical(fed$metadata$available_rounds, 1:5))
    trained[[as.character(epsilon)]][[r]] <- fed
    saveRDS(fed, file.path(cell_dir, sprintf("trained_rep%d.rds", r)))
    cat(sprintf("HAR561 epsilon %g replicate %d complete in %.1fs; test remains unopened\n",
                epsilon, r, fed$elapsed_s))
  }
}
stopifnot(all(vapply(seq_along(script_paths), function(i) {
  identical(digest::digest(file = script_paths[i], algo = "sha256"), script_hashes[[i]])
}, logical(1))))
training_completed <- utc()
scoring_marker <- file.path(run_dir, "SCORING_STARTED")
stopifnot(!file.exists(scoring_marker))
scoring_started <- utc()
writeLines(scoring_started, scoring_marker)
cat("All nine fits complete; opening official test tables for the single scoring run\n")
test <- har561_load(cache, "test")
stopifnot(!length(intersect(train$subject, test$subject)),
          identical(sort(unique(c(train$subject, test$subject))), 1:30))
central_scores <- lapply(central_models, function(fit) {
  p <- predict(fit, har561_transform(test), type = "probs")
  c(har561_metrics(test$target, p[, HAR561_LEVELS, drop = FALSE]),
    list(convergence = fit$convergence))
})
prevalence <- as.numeric(table(factor(train$target, levels = HAR561_LEVELS))) / nrow(train)
trivial <- c(har561_metrics(test$target, matrix(rep(prevalence, each = nrow(test)), ncol = 6L)),
  list(majority_class = which.max(prevalence), training_prevalence = prevalence,
       probability_policy = "training window class frequencies; argmax is training majority"))
for (epsilon in epsilons) {
  per_replicate <- list()
  for (r in seq_along(seeds)) {
    fed <- trained[[as.character(epsilon)]][[r]]
    p <- as.matrix(ds.flower.predict(fed$fit, test[, features, drop = FALSE], type = "prob"))
    if (!is.null(colnames(p)) && all(HAR561_LEVELS %in% colnames(p))) {
      p <- p[, HAR561_LEVELS, drop = FALSE]
    }
    metrics <- har561_metrics(test$target, p)
    central <- central_scores[[r]]
    rep <- list(seed = seeds[r],
      split = list(n_train = nrow(train), n_test = nrow(test),
        n_per_site = unname(vapply(sites[[r]]$data, nrow, integer(1))),
        train_subjects = sort(unique(train$subject)), test_subjects = sort(unique(test$subject)),
        site_subjects = unname(sites[[r]]$subjects), training_effective_units = rep(7L, 3L),
        class_counts_train = har561_class_sizes(train), class_counts_test = har561_class_sizes(test),
        class_counts_sites = unname(lapply(sites[[r]]$data, har561_class_sizes)),
        train_index_sha256 = har561_index_hash(train), test_index_sha256 = har561_index_hash(test),
        site_index_sha256 = unname(lapply(sites[[r]]$data, har561_index_hash))),
      central = central, federated_dp = metrics, trivial = trivial,
      gap_macro_auc = metrics$macro_auc - central$macro_auc,
      feature_bounds = bounds, node_privacy = fed$node_privacy,
      released_metadata = fed$metadata,
      history = list(n_clients = fed$n_clients, n_failures = fed$n_failures,
                     n_rounds = fed$n_rounds_run, rounds = fed$history),
      model_sha256 = fed$model_sha256,
      diagnostic = list(accuracy_above_majority = metrics$acc > trivial$acc,
        macro_auc_above_half = metrics$macro_auc > 0.5,
        passed = metrics$acc > trivial$acc && metrics$macro_auc > 0.5),
      wall_clock = list(central_fit_s = central_seconds[r], federated_s = fed$elapsed_s,
                       total_s = central_seconds[r] + fed$elapsed_s,
                       central_reused_across_epsilon = TRUE),
      cleanup_ok = fed$cleanup_ok)
    per_replicate[[r]] <- rep
    cat(sprintf("SCORED epsilon %g rep %d central %.6f fed %.6f accuracy %.6f majority %.6f\n",
                epsilon, r, central$macro_auc, metrics$macro_auc, metrics$acc, trivial$acc))
  }
  msd <- function(values) list(mean = mean(values), sd = sd(values))
  branch_summary <- function(branch) setNames(lapply(c("macro_auc", "acc", "logloss"),
    function(metric) msd(vapply(per_replicate, function(x) x[[branch]][[metric]], numeric(1)))),
    c("macro_auc", "acc", "logloss"))
  summary <- list(central_mean_sd = branch_summary("central"),
    federated_mean_sd = branch_summary("federated_dp"), trivial_mean_sd = branch_summary("trivial"),
    gap_mean_sd = list(macro_auc = msd(vapply(per_replicate, `[[`, numeric(1), "gap_macro_auc"))),
    diagnostic = list(applicable_epsilon = 8,
      mean_accuracy_above_majority = mean(vapply(per_replicate, function(x) x$federated_dp$acc - x$trivial$acc, numeric(1))) > 0,
      mean_macro_auc_above_half = mean(vapply(per_replicate, function(x) x$federated_dp$macro_auc, numeric(1))) > 0.5,
      all_replicates_pass = all(vapply(per_replicate, function(x) x$diagnostic$passed, logical(1)))))
  cell_id <- sprintf("har561_pytorch_multiclass_eps%g", epsilon)
  artifact <- list(schema = "dsflower-campaign-v2", cell_id = cell_id, generated_at = utc(),
    host = env_info$host, versions = env_info$versions, tooling_sha256 = script_hashes,
    contract = "pytorch_multiclass", dataset = c(protocol$dataset,
      list(n_total = nrow(train) + nrow(test), features = features,
        download_sha256 = HAR561_SHA256, inner_archive_sha256 = HAR561_INNER_SHA256,
        n_train = nrow(train), n_test = nrow(test),
        n_per_site = per_replicate[[1]]$split$n_per_site)),
    split = protocol$split, protocol = protocol,
    privacy = list(epsilon = epsilon, delta = 1e-6, clipping_norm = 1,
      unit = "patient", patient_column = "subject", adjacency = "replace_one",
      guarantee_scope = "per-training",
      composition_note = "Epsilon is per node training; the nine fits are not a single composed release guarantee.",
      bounds_policy = protocol$bounds_policy, released_patient_semantics = protocol$released_patient_semantics,
      randomness = "Published seeds control subject-to-site partition and central initialization; federated initialization and node cryptographic DP noise retain release behavior."),
    scoring = list(training_completed_at_utc = training_completed, started_at_utc = scoring_started,
      test_table_loads = 1L, central_fits = 3L, federated_fits = 9L,
      central_scores_reused_across_epsilon = TRUE, changed_settings_after_scoring = FALSE),
    rounds = 5L, model_params_requested = model_params, model_params_resolved = resolved,
    per_replicate = per_replicate, summary = summary)
  out_path <- file.path(out, paste0(cell_id, ".json"))
  stopifnot(!file.exists(out_path))
  jsonlite::write_json(artifact, out_path, pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")
  cat("WROTE", out_path, "\n")
}
writeLines(utc(), file.path(run_dir, "SCORING_COMPLETED"))
