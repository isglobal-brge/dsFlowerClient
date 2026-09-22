#!/usr/bin/env Rscript
# Train the frozen matrix, then open the sealed test bundle exactly once.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
root <- normalizePath(args[1], mustWork = TRUE)
out <- args[2]
script_dir <- dirname(sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(script_dir, "cdcgenhlth_campaign.R"))
source(file.path(script_dir, "cdcgenhlth.R"))
protocol <- jsonlite::fromJSON(file.path(script_dir, "cdcgenhlth_protocol.json"), simplifyVector = FALSE)
venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
stopifnot(nzchar(venv_root))
cache <- file.path(root, "data_cache")
stopifnot(identical(digest::digest(file = file.path(cache, "cdc_diabetes_health_indicators.csv"),
                                  algo = "sha256"), CDCGENHLTH_SHA256))
run_dir <- file.path(root, "runs", "cdcgenhlth_multiclass")
stopifnot(dir.exists(run_dir), file.exists(file.path(run_dir, "PREPARATION_COMPLETED")),
          !file.exists(file.path(run_dir, "TRAINING_STARTED")),
          !file.exists(file.path(run_dir, "SCORING_STARTED")))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
utc <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
writeLines(utc(), file.path(run_dir, "TRAINING_STARTED"))
training_started <- utc()
seeds <- 20260819L + 1:3
epsilons <- c(1, 4, 8)
model_params <- list(n_classes = 5L)  # Dataset output size; optimization defaults unchanged.
resolved <- dsFlowerClient:::.dsflower_get_model("pytorch_multiclass")$defaults
resolved$n_classes <- 5L
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
script_paths <- file.path(script_dir, c("run_cdcgenhlth.R", "cdcgenhlth_campaign.R", "cdcgenhlth.R",
                                      "cdcgenhlth_protocol.json", "prepare_cdcgenhlth.R", "check_cdcgenhlth.R",
                                      "run_cdcgenhlth.sh"))
script_hashes <- setNames(lapply(script_paths, function(p) digest::digest(file = p, algo = "sha256")),
                         basename(script_paths))
splits <- readRDS(file.path(run_dir, "training.rds"))
stopifnot(length(splits) == 3L)
preparation <- jsonlite::fromJSON(file.path(run_dir, "preparation.json"), simplifyVector = FALSE)
features <- CDCGENHLTH_FEATURES
bounds <- cdcgenhlth_bounds()
training_hash <- digest::digest(file = file.path(run_dir, "training.rds"), algo = "sha256")
stopifnot(identical(training_hash, preparation$training_sha256),
          identical(preparation$legacy_prepared_sha256, protocol$legacy_logistic_prepared_sha256),
          identical(preparation$raw_sha256, CDCGENHLTH_SHA256))
for (split in splits) {
  stopifnot(identical(names(split$train), c(features, "target")),
            all(complete.cases(split$train)),
            all(is.finite(as.matrix(split$train))), all(split$train$target %in% 1:5))
}
central_models <- list()
central_seconds <- numeric(3)
for (r in seq_along(seeds)) {
  cat(sprintf("Central training replicate %d, seed %d; test remains unopened\n", r, seeds[r]))
  start <- Sys.time()
  central_models[[r]] <- cdcgenhlth_central_fit(splits[[r]]$train, seeds[r])
  central_seconds[r] <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  saveRDS(central_models[[r]], file.path(run_dir, sprintf("central_rep%d.rds", r)))
  cat(sprintf("Central replicate %d converged in %.1fs\n", r, central_seconds[r]))
}
trained <- list()
for (epsilon in epsilons) {
  cell_id <- sprintf("cdcgenhlth_pytorch_multiclass_eps%g", epsilon)
  stopifnot(!file.exists(file.path(out, paste0(cell_id, ".json"))))
  cell_dir <- file.path(run_dir, cell_id)
  stopifnot(!dir.exists(cell_dir))
  dir.create(cell_dir)
  writeLines(utc(), file.path(cell_dir, "STARTED"))
  trained[[as.character(epsilon)]] <- list()
  for (r in seq_along(seeds)) {
    cat(sprintf("CDCGENHLTH epsilon %g replicate %d training; test remains unopened\n", epsilon, r))
    fed <- campaign_run_federated(
      site_data = splits[[r]]$sites, test = stop("Test access during training is forbidden"),
      features = features, feature_bounds = bounds, epsilon = epsilon, delta = 1e-6,
      rounds = 5L, model_params = model_params,
      work_dir = file.path(cell_dir, sprintf("rep%d", r)), venv_root = venv_root,
      contract = "pytorch_multiclass", score_function = function(fit, test, features) NULL)
    stopifnot(fed$n_clients == 3L, fed$n_failures == 0L, fed$n_rounds_run == 5L,
              fed$cleanup_ok, identical(fed$metadata$target_levels, CDCGENHLTH_LEVELS),
              identical(fed$metadata$available_rounds, 1:5))
    trained[[as.character(epsilon)]][[r]] <- fed
    saveRDS(fed, file.path(cell_dir, sprintf("trained_rep%d.rds", r)))
    cat(sprintf("CDCGENHLTH epsilon %g replicate %d complete in %.1fs; test remains unopened\n",
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
cat("All nine fits complete; opening sealed test bundle for the single scoring phase\n")
stopifnot(identical(digest::digest(file = file.path(run_dir, "training.rds"), algo = "sha256"),
                    training_hash))
stopifnot(identical(digest::digest(file = file.path(run_dir, "sealed_test.rds"), algo = "sha256"),
                    preparation$sealed_test_sha256))
tests <- readRDS(file.path(run_dir, "sealed_test.rds"))
stopifnot(length(tests) == 3L)
central_scores <- trivial_scores <- list()
for (r in seq_along(seeds)) {
  train <- splits[[r]]$train
  test <- tests[[r]]
  stopifnot(nrow(train) + nrow(test) == 45000L,
            !length(intersect(rownames(train), rownames(test))),
            length(unique(c(rownames(train), rownames(test)))) == 45000L,
            identical(sort(unlist(lapply(splits[[r]]$sites, rownames))), sort(rownames(train))))
  p <- predict(central_models[[r]], cdcgenhlth_transform(test), type = "probs")
  central_scores[[r]] <- c(cdcgenhlth_metrics(test$target, p[, CDCGENHLTH_LEVELS, drop = FALSE]),
    list(convergence = central_models[[r]]$convergence))
  prevalence <- as.numeric(table(factor(train$target, levels = CDCGENHLTH_LEVELS))) / nrow(train)
  trivial_scores[[r]] <- c(cdcgenhlth_metrics(test$target,
    matrix(rep(prevalence, each = nrow(test)), ncol = 5L)),
    list(majority_class = which.max(prevalence), training_prevalence = prevalence,
         probability_policy = "training class frequencies; argmax is training majority"))
}
for (epsilon in epsilons) {
  per_replicate <- list()
  for (r in seq_along(seeds)) {
    train <- splits[[r]]$train
    test <- tests[[r]]
    trivial <- trivial_scores[[r]]
    fed <- trained[[as.character(epsilon)]][[r]]
    p <- as.matrix(ds.flower.predict(fed$fit, test[, features, drop = FALSE], type = "prob"))
    if (!is.null(colnames(p)) && all(CDCGENHLTH_LEVELS %in% colnames(p))) {
      p <- p[, CDCGENHLTH_LEVELS, drop = FALSE]
    }
    metrics <- cdcgenhlth_metrics(test$target, p)
    central <- central_scores[[r]]
    rep <- list(seed = seeds[r],
      split = list(n_train = nrow(train), n_test = nrow(test),
        n_per_site = unname(vapply(splits[[r]]$sites, nrow, integer(1))),
        training_effective_units = unname(vapply(splits[[r]]$sites, nrow, integer(1))),
        class_counts_train = cdcgenhlth_class_sizes(train), class_counts_test = cdcgenhlth_class_sizes(test),
        class_counts_sites = unname(lapply(splits[[r]]$sites, cdcgenhlth_class_sizes)),
        train_index_sha256 = cdcgenhlth_index_hash(train), test_index_sha256 = cdcgenhlth_index_hash(test),
        site_index_sha256 = unname(lapply(splits[[r]]$sites, cdcgenhlth_index_hash))),
      central = central, federated_dp = metrics, trivial = trivial,
      gap_macro_auc = metrics$macro_auc - central$macro_auc,
      feature_bounds = bounds, node_privacy = fed$node_privacy,
      released_metadata = fed$metadata,
      history = list(n_clients = fed$n_clients, n_failures = fed$n_failures,
                     n_rounds = fed$n_rounds_run, rounds = fed$history),
      model_sha256 = fed$model_sha256,
      diagnostic = list(accuracy_above_majority = metrics$acc > trivial$acc,
        macro_auc_above_half = metrics$macro_auc > 0.5),
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
      all_replicates_above_majority = all(vapply(per_replicate, function(x) x$diagnostic$accuracy_above_majority, logical(1))),
      all_replicates_above_half = all(vapply(per_replicate, function(x) x$diagnostic$macro_auc_above_half, logical(1)))))
  cell_id <- sprintf("cdcgenhlth_pytorch_multiclass_eps%g", epsilon)
  artifact <- list(schema = "dsflower-campaign-v2", cell_id = cell_id, generated_at = utc(),
    host = env_info$host, versions = env_info$versions, tooling_sha256 = script_hashes,
    contract = "pytorch_multiclass", dataset = c(protocol$dataset,
      list(n_total = nrow(train) + nrow(test),
        download_sha256 = CDCGENHLTH_SHA256,
        legacy_logistic_prepared_sha256 = protocol$legacy_logistic_prepared_sha256,
        n_train = nrow(train), n_test = nrow(test),
        n_per_site = per_replicate[[1]]$split$n_per_site)),
    split = protocol$split, protocol = protocol, preparation = preparation,
    privacy = list(epsilon = epsilon, delta = 1e-6, clipping_norm = 1,
      unit = "row", patient_column = NULL, adjacency = "replace_one",
      guarantee_scope = "per-training",
      composition_note = "Epsilon is per node training; the nine fits are not a single composed release guarantee.",
      bounds_policy = protocol$bounds_policy,
      randomness = "Published seeds control cohort partitioning and central initialization; federated initialization and node cryptographic DP noise retain release behavior."),
    scoring = list(training_started_at_utc = training_started,
      training_completed_at_utc = training_completed, started_at_utc = scoring_started,
      test_bundle_loads = 1L, held_out_partitions = 3L, central_fits = 3L, federated_fits = 9L,
      central_scores_reused_across_epsilon = TRUE, changed_settings_after_scoring = FALSE),
    rounds = 5L, model_params_requested = model_params, model_params_resolved = resolved,
    per_replicate = per_replicate, summary = summary)
  out_path <- file.path(out, paste0(cell_id, ".json"))
  stopifnot(!file.exists(out_path))
  jsonlite::write_json(artifact, out_path, pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null")
  cat("WROTE", out_path, "\n")
}
writeLines(utc(), file.path(run_dir, "SCORING_COMPLETED"))
