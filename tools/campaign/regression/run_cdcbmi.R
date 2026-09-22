#!/usr/bin/env Rscript
# One predeclared CDC BMI budget, three immutable scored replicates.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1])
epsilon <- as.numeric(args[2])
out <- args[3]
stopifnot(epsilon %in% c(1, 4, 8))
script_dir <- dirname(normalizePath(sub("--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[1])))
source(file.path(script_dir, "campaign_lib.R"))
protocol_path <- file.path(script_dir, "cdcbmi_protocol.json")
p <- jsonlite::fromJSON(protocol_path)
contract <- p$contract
cell_id <- sprintf("cdcbmi_%s_eps%g", contract, epsilon)
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
stopifnot(identical(protocol_sha, provenance$protocol_sha256))
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
  score <- function(fit, test, features) {
    # Runtime-only fallback decision precedes any held-out read or score.
    elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    if (elapsed > 900) {
      jsonlite::write_json(list(status = "unscored_runtime_limit", elapsed_s = elapsed),
                           file.path(work, "runtime_limit.json"), auto_unbox = TRUE)
      stop("cdc45k training exceeded 15 minutes before scoring; retain this attempt.")
    }
    guard <- file.path(work, "scoring.started")
    if (file.exists(guard)) stop("Scoring already started; refusing repeat evaluation.")
    writeLines(format(Sys.time(), tz = "UTC", usetz = TRUE), guard)
    # No held-out frame is passed to training. Its first analysis is here.
    relative <- paste0("cdcbmi_seed", seed, "/test.csv")
    stopifnot(identical(digest::digest(file = file.path(cache, relative), algo = "sha256"),
                        provenance$checksums[[relative]]))
    test <- read.csv(file.path(prepared, "test.csv"), check.names = FALSE)
    stopifnot(nrow(test) == 9000L, identical(names(test), names(train)))
    prediction <- as.numeric(ds.flower.predict(fit, test[, features, drop = FALSE],
                                              type = "response"))
    result <- regression_metrics(test[[p$target]], prediction)
    jsonlite::write_json(result, file.path(work, "federated_score.json"),
                         pretty = TRUE, auto_unbox = TRUE, digits = NA)
    processx::run(file.path(venv_root, "pytorch", "bin", "python"),
      c(file.path(script_dir, "central_rows.py"),
        "--train", file.path(prepared, "train.csv"),
        "--test", file.path(prepared, "test.csv"), "--protocol", protocol_path,
        "--output", file.path(work, "baselines.json")), error_on_status = TRUE)
    result
  }
  fed <- campaign_run_federated(
    site_data = sites, test = NULL, features = features, feature_bounds = bounds,
    target_bounds = p$target_bounds, target = p$target, patient_column = NULL,
    epsilon = epsilon, delta = p$privacy$delta, rounds = p$rounds,
    model_params = list(), work_dir = work, venv_root = venv_root,
    contract = contract, score_function = score)
  baseline <- jsonlite::fromJSON(file.path(work, "baselines.json"))
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
  gap_rmse_mean_sd = msd(vapply(per_replicate, function(rep) rep$gap_rmse, numeric(1))))
summary$diagnostics <- list(
  rmse_below_trivial = summary$federated_mean_sd$rmse$mean < summary$trivial_mean_sd$rmse$mean,
  r2_positive = summary$federated_mean_sd$r2$mean > 0, role = "annotation_only")
tooling <- c("run_cdcbmi.R", "run_cdcbmi.sh", "cdcbmi_protocol.json", "prepare_cdcbmi.py",
             "central_rows.py", "central_train.py", "campaign_lib.R")
artifact <- list(schema = "dsflower-campaign-v2", cell_id = cell_id,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  predeclaration_commit = "9796615", host = env$host, versions = env$versions,
  contract = contract, dataset = provenance, protocol_sha256 = protocol_sha, protocol = p,
  split = p$split, seeds = p$seeds,
  privacy = list(epsilon = epsilon, delta = p$privacy$delta, unit = "row", clipping_norm = 1),
  feature_bounds = setNames(lapply(seq_along(features), function(i) {
    list(lower = bounds$lower[i], upper = bounds$upper[i])
  }), features), target_bounds = p$target_bounds,
  rounds = p$rounds, model_params = defaults, model_params_overrides = list(),
  per_replicate = per_replicate, summary = summary,
  tooling_sha256 = setNames(lapply(tooling, function(name) {
    digest::digest(file = file.path(script_dir, name), algo = "sha256")
  }), tooling))
jsonlite::write_json(artifact, output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("WROTE", output, "\n")
