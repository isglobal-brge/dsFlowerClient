#!/usr/bin/env Rscript
# One predeclared public-unit CDC BMI budget, three immutable scored replicates.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1])
epsilon <- as.numeric(args[2])
out <- args[3]
stopifnot(epsilon %in% c(1, 4, 8))
script_dir <- dirname(normalizePath(sub("--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[1])))
source(file.path(script_dir, "..", "campaign_lib.R"))
regression_dir <- normalizePath(file.path(script_dir, ".."))
selection_path <- file.path(script_dir, "selection.json")
selection <- jsonlite::fromJSON(selection_path)
params <- selection$selected_params
stopifnot(!selection$sealed_test_opened, selection$epsilon == 8)
protocol_path <- file.path(regression_dir, "cdcbmi_public_units_protocol.json")
p <- jsonlite::fromJSON(protocol_path)
contract <- p$contract
cell_id <- sprintf("cdcbmi_r4_%s_eps%g", contract, epsilon)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
output <- file.path(out, paste0(cell_id, ".json"))
lock <- file.path(root, "runs", paste0(cell_id, ".started"))
dir.create(dirname(lock), recursive = TRUE, showWarnings = FALSE)
if (file.exists(output) || file.exists(lock)) stop("Cell already started; refusing a rerun.")
stopifnot(packageVersion("dsFlower") == "0.5.0",
          packageVersion("dsFlowerClient") == "0.5.0")
venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
env <- campaign_env_info(venv_root)
env$host$pod_id <- "n4emgxhiqzy5i4"
env$host$pod_name <- "pod-flower-regression"
env$versions$runner_sha256 <- dsFlowerClient:::.compute_local_runner_hash()
env$versions$release_tag <- "v0.5.0"
env$versions$release_commits <- list(
  dsFlower = "408f08c539329e2711260050ab40a6567aa4d89e",
  dsFlowerClient = "50dda000a32ffcbdd039c2b74c909df451392bfb")
stopifnot(identical(env$versions$runner_sha256,
                    dsFlower:::.compute_app_pkg_hash("dsflower_runner")))
defaults <- dsFlowerClient:::.dsflower_get_model(contract)$defaults
stopifnot(defaults$learning_rate == 0.01, defaults$local_epochs == 1L,
          defaults$batch_size == 32L, defaults$weight_decay == 0,
          defaults$l1_penalty == 0, defaults$optimizer == "sgd",
          defaults$scheduler == "none", length(defaults$hidden_layers) == 0L)
cache <- file.path(root, "data_cache")
provenance <- jsonlite::fromJSON(file.path(cache, "cdcbmi_provenance.json"),
                               simplifyVector = FALSE)
protocol_sha <- digest::digest(file = protocol_path, algo = "sha256")
stopifnot(identical(digest::digest(file = file.path(regression_dir, "cdcbmi_protocol.json"),
                                  algo = "sha256"), provenance$protocol_sha256))
declaration_commit <- trimws(readLines(file.path(regression_dir, "r4/declaration_commit.txt")))
stopifnot(length(declaration_commit) == 1L, nchar(declaration_commit) == 40L,
          p$public_target_transform$center == 55, p$public_target_transform$scale == 43,
          p$target_bounds$lower == -1, p$target_bounds$upper == 1)
features <- p$features
bounds <- p$feature_bounds
stopifnot(length(features) == 20L, !any(c("BMI", "Diabetes_binary", "ID") %in% features),
          p$target == "BMI", p$sites == 3L, p$rounds == 5L,
          identical(as.integer(p$seeds), 20260820:20260822))
writeLines(format(Sys.time(), tz = "UTC", usetz = TRUE), lock)
regression_metrics <- function(y, prediction) {
  stopifnot(length(y) == length(prediction), all(is.finite(prediction)))
  list(rmse = sqrt(mean((y - prediction)^2)), mae = mean(abs(y - prediction)),
       r2 = 1 - sum((y - prediction)^2) / sum((y - mean(y))^2))
}
per_replicate <- list()
for (r in seq_along(p$seeds)) {
  started <- Sys.time()
  seed <- p$seeds[r]
  prepared <- file.path(cache, paste0("cdcbmi_seed", seed))
  for (name in c("train.csv", "site1.csv", "site2.csv", "site3.csv", "split.json")) {
    relative <- paste0("cdcbmi_seed", seed, "/", name)
    stopifnot(identical(digest::digest(file = file.path(cache, relative), algo = "sha256"),
                        provenance$checksums[[relative]]))
  }
  split <- jsonlite::fromJSON(file.path(prepared, "split.json"), simplifyVector = FALSE)
  train <- read.csv(file.path(prepared, "train.csv"), check.names = FALSE)
  sites <- lapply(seq_len(p$sites), function(s) {
    read.csv(file.path(prepared, paste0("site", s, ".csv")), check.names = FALSE)
  })
  stopifnot(nrow(train) == 36000L, all(vapply(sites, nrow, integer(1)) == 12000L),
            identical(names(train), c(features, p$target)))
  work <- file.path(root, "runs", cell_id, paste0("rep", r))
  dir.create(work, recursive = TRUE, mode = "0700", showWarnings = FALSE)
  cat(sprintf("START %s replicate %d seed %d\n", cell_id, r, seed))
  comparator_work <- file.path(root, "r4", paste0("comparators_seed", seed))
  dir.create(comparator_work, recursive = TRUE, showWarnings = FALSE)
  if (epsilon == 1) {
    processx::run(file.path(venv_root, "pytorch", "bin", "python"),
      c(file.path(script_dir, "comparators.py"), "fit", root, as.character(seed), comparator_work, selection_path),
      error_on_status = TRUE)
  } else stopifnot(file.exists(file.path(comparator_work, "baselines.json")))
  pooled_fit <- NULL
  score <- function(fit, test, features) {
    guard <- file.path(work, "scoring.started")
    if (file.exists(guard)) stop("Scoring already started; refusing repeat evaluation.")
    writeLines(format(Sys.time(), tz = "UTC", usetz = TRUE), guard)
    # No held-out frame is passed to training. Its first analysis is here.
    relative <- paste0("cdcbmi_seed", seed, "/test.csv")
    if (!file.exists(file.path(cache, relative))) {
      source_rows <- as.integer(unlist(split$test$source_rows))
      source_data <- read.csv(file.path(cache, "cdc_diabetes_health_indicators.csv"))
      write.csv(source_data[source_rows, c(features, p$target), drop = FALSE],
                file.path(cache, relative), row.names = FALSE)
      rm(source_data)
    }
    stopifnot(identical(digest::digest(file = file.path(cache, relative), algo = "sha256"),
                        provenance$checksums[[relative]]))
    test <- read.csv(file.path(prepared, "test.csv"), check.names = FALSE)
    stopifnot(nrow(test) == 9000L, identical(names(test), names(train)))
    prediction <- as.numeric(ds.flower.predict(fit, test[, features, drop = FALSE],
                                              type = "response"))
    prediction <- p$public_target_transform$center + p$public_target_transform$scale * prediction
    saveRDS(fit, file.path(work, "fit.rds"))
    pooled_prediction <- 55 + 43 * as.numeric(ds.flower.predict(pooled_fit,
      test[, features, drop = FALSE], type = "response"))
    jsonlite::write_json(regression_metrics(test[[p$target]], pooled_prediction),
      file.path(work, "pooled_score.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
    result <- regression_metrics(test[[p$target]], prediction)
    residual <- prediction - test[[p$target]]
    result$decomposition <- list(mean_residual = mean(residual),
      residual_variance = mean((residual - mean(residual))^2),
      mean_prediction = mean(prediction), target_mean = mean(test[[p$target]]))
    jsonlite::write_json(result, file.path(work, "federated_score.json"),
                         pretty = TRUE, auto_unbox = TRUE, digits = NA)
    if (epsilon == 1) processx::run(file.path(venv_root, "pytorch", "bin", "python"),
      c(file.path(script_dir, "comparators.py"), "score", root, as.character(seed), comparator_work, selection_path),
      error_on_status = TRUE)
    result
  }
  # Only training-site targets change; cached source/split CSVs stay immutable.
  sites <- lapply(sites, function(site) {
    clipped <- pmax(p$original_target_bounds$lower,
                    pmin(p$original_target_bounds$upper, site[[p$target]]))
    site[[p$target]] <- (clipped - p$public_target_transform$center) / p$public_target_transform$scale
    site
  })
  pooled <- campaign_run_federated(
    site_data = list(do.call(rbind, sites)), test = NULL, features = features,
    feature_bounds = bounds, target_bounds = p$target_bounds, target = p$target,
    patient_column = NULL, epsilon = epsilon, delta = p$privacy$delta, rounds = p$rounds,
    model_params = params, work_dir = file.path(work, "pooled"), venv_root = venv_root,
    contract = contract, score_function = function(fit, test, features) {
      pooled_fit <<- fit
      saveRDS(fit, file.path(work, "pooled_fit.rds"))
      list(scoring_deferred = TRUE)
    })
  stopifnot(pooled$n_clients == 1L, pooled$n_rounds_run == 5L,
            pooled$n_failures == 0L, pooled$cleanup_ok)
  fed <- campaign_run_federated(
    site_data = sites, test = NULL, features = features, feature_bounds = bounds,
    target_bounds = p$target_bounds, target = p$target, patient_column = NULL,
    epsilon = epsilon, delta = p$privacy$delta, rounds = p$rounds,
    model_params = params, work_dir = work, venv_root = venv_root,
    contract = contract, score_function = score)
  baseline <- jsonlite::fromJSON(file.path(comparator_work, "baselines.json"))
  stopifnot(fed$n_clients == 3L, fed$n_rounds_run == 5L, fed$n_failures == 0L,
            fed$cleanup_ok, baseline$central_fit$n_training_rows == nrow(train),
            all(vapply(fed$node_privacy, function(node) {
              identical(node$policy$dp_unit, "row") && node$clipping_norm == 1 &&
                length(node$staged_configurations) > 0L
            }, logical(1))))
  per_replicate[[r]] <- list(seed = seed,
    split = list(n_train = nrow(train), n_test = 9000L,
      n_per_site = vapply(sites, nrow, integer(1)),
      membership_file = paste0("cdcbmi_split_seed", seed, ".json"),
      membership_sha256 = digest::digest(file = file.path(prepared, "split.json"), algo = "sha256")),
    central = baseline$central, trivial = baseline$trivial,
    nonprivate_federated = baseline$nonprivate_federated,
    nonprivate_fit = baseline$nonprivate_fit,
    pooled_dp = jsonlite::fromJSON(file.path(work, "pooled_score.json")),
    pooled_training = pooled,
    central_fit = baseline$central_fit, training_mean = baseline$training_mean,
    baseline_provenance = baseline$provenance, federated_dp = fed$metrics,
    gap_rmse = fed$metrics$rmse - baseline$central$rmse,
    diagnostics = list(rmse_below_trivial = fed$metrics$rmse < baseline$trivial$rmse,
                       r2_positive = fed$metrics$r2 > 0, role = "annotation_only"),
    node_privacy = fed$node_privacy,
    history = list(n_clients = fed$n_clients, n_failures = fed$n_failures,
                   n_rounds = fed$n_rounds_run, cleanup_ok = fed$cleanup_ok),
    model_sha256 = fed$model_sha256, federation_elapsed_s = fed$elapsed_s,
    elapsed_s = as.numeric(difftime(Sys.time(), started, units = "secs")))
  jsonlite::write_json(per_replicate[[r]], file.path(work, "replicate.json"),
                       pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat(sprintf("DONE rep %d: central RMSE %.6f | fed RMSE %.6f R2 %.6f | trivial %.6f | %.1fs\n",
    r, baseline$central$rmse, fed$metrics$rmse, fed$metrics$r2, baseline$trivial$rmse,
    per_replicate[[r]]$elapsed_s))
}
msd <- function(values) list(mean = mean(values), sd = sd(values))
branch_summary <- function(branch) setNames(lapply(c("rmse", "mae", "r2"), function(metric) {
  msd(vapply(per_replicate, function(rep) rep[[branch]][[metric]], numeric(1)))
}), c("rmse", "mae", "r2"))
summary <- list(central_mean_sd = branch_summary("central"),
  federated_mean_sd = branch_summary("federated_dp"),
  trivial_mean_sd = branch_summary("trivial"),
  nonprivate_federated_mean_sd = branch_summary("nonprivate_federated"),
  pooled_dp_mean_sd = branch_summary("pooled_dp"),
  gap_rmse_mean_sd = msd(vapply(per_replicate, function(rep) rep$gap_rmse, numeric(1))))
summary$diagnostics <- list(
  rmse_below_trivial = summary$federated_mean_sd$rmse$mean < summary$trivial_mean_sd$rmse$mean,
  r2_positive = summary$federated_mean_sd$r2$mean > 0, role = "annotation_only")
tooling <- c("run_corrected.R", "comparators.py", "emulate.py", "selection.json", "declaration_commit.txt")
effective_params <- modifyList(defaults, params)
artifact <- list(schema = "dsflower-campaign-v2", cell_id = cell_id,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  predeclaration_commit = declaration_commit, host = env$host, versions = env$versions,
  runtime = jsonlite::fromJSON(file.path(script_dir, "provisioning.json")),
  contract = contract, dataset = provenance, protocol_sha256 = protocol_sha, protocol = p,
  split = p$split, seeds = p$seeds,
  privacy = list(epsilon = epsilon, delta = p$privacy$delta, unit = "row", clipping_norm = 1),
  feature_bounds = setNames(lapply(seq_along(features), function(i) {
    list(lower = bounds$lower[i], upper = bounds$upper[i])
  }), features), target_bounds = p$target_bounds,
  rounds = p$rounds, model_params = effective_params, model_params_overrides = params, selection = selection,
  per_replicate = per_replicate, summary = summary,
  tooling_sha256 = setNames(lapply(tooling, function(name) {
    digest::digest(file = file.path(script_dir, name), algo = "sha256")
  }), tooling))
jsonlite::write_json(artifact, output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("WROTE", output, "\n")
