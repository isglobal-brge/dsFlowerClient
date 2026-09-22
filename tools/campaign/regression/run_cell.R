#!/usr/bin/env Rscript
# One immutable three-replicate regression cell using the campaign federation.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1])
epsilon <- as.numeric(args[2])
out <- args[3]
stopifnot(epsilon %in% c(1, 4, 8))
script_dir <- dirname(normalizePath(sub("--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[1])))
source(file.path(script_dir, "campaign_lib.R"))
protocol_path <- file.path(script_dir, "protocol.json")
p <- jsonlite::fromJSON(protocol_path, simplifyVector = TRUE)
stopifnot(packageVersion("dsFlower") == "0.5.0",
          packageVersion("dsFlowerClient") == "0.5.0")
contract <- p$contract
cell_id <- sprintf("pilot_parkinsons_%s_eps%g", contract, epsilon)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
output <- file.path(out, paste0(cell_id, ".json"))
lock <- file.path(root, "runs", paste0(cell_id, ".started"))
dir.create(dirname(lock), recursive = TRUE, showWarnings = FALSE)
if (file.exists(output) || file.exists(lock)) stop("Cell already started; refusing a rerun.")
writeLines(format(Sys.time(), tz = "UTC", usetz = TRUE), lock)

features <- p$features
bounds <- p$feature_bounds
venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
env <- campaign_env_info(venv_root)
env$host$pod_id <- "n4emgxhiqzy5i4"
env$host$pod_name <- "pod-flower-regression"
env$versions$runner_sha256 <- dsFlowerClient:::.compute_local_runner_hash()
env$versions$release_tag <- "v0.5.0"
defaults <- dsFlowerClient:::.dsflower_get_model(contract)$defaults
stopifnot(defaults$learning_rate == 0.01, defaults$local_epochs == 1L,
          defaults$batch_size == 32L, defaults$weight_decay == 0)
provenance <- jsonlite::fromJSON(file.path(root, "data_cache", "regression_provenance.json"))
protocol_sha <- digest::digest(file = protocol_path, algo = "sha256")
stopifnot(identical(protocol_sha, provenance$protocol_sha256))
data_path <- file.path(root, "data_cache", "parkinsons_updrs.data")
stopifnot(identical(digest::digest(file = data_path, algo = "sha256"),
                    provenance$checksums$parkinsons_updrs.data))
# Loading to route rows is not an exploratory analysis of held-out values.
# No outcome-dependent split, bound estimation or hyperparameter selection.
df <- read.csv(data_path, check.names = FALSE)
df <- df[, c(p$patient_column, features, p$target)]
subjects <- sort(unique(df[[p$patient_column]]))
stopifnot(nrow(df) == 5875L, length(subjects) == 42L)

regression_metrics <- function(y, prediction) {
  stopifnot(length(y) == length(prediction), all(is.finite(prediction)))
  list(rmse = sqrt(mean((y - prediction)^2)),
       mae = mean(abs(y - prediction)),
       r2 = 1 - sum((y - prediction)^2) / sum((y - mean(y))^2))
}
per_replicate <- list()
for (r in seq_along(p$seeds)) {
  started <- Sys.time()
  seed <- p$seeds[r]
  set.seed(seed)
  test_ids <- sample(subjects, p$split$n_test_subjects)
  train_ids <- sample(setdiff(subjects, test_ids))
  site_ids <- lapply(seq_len(p$sites), function(s) {
    train_ids[(seq_along(train_ids) - 1L) %% p$sites == s - 1L]
  })
  stopifnot(!anyDuplicated(unlist(site_ids)),
            !length(intersect(unlist(site_ids), test_ids)))
  train <- df[df[[p$patient_column]] %in% train_ids, , drop = FALSE]
  test <- df[df[[p$patient_column]] %in% test_ids, , drop = FALSE]
  sites <- lapply(site_ids, function(ids) df[df[[p$patient_column]] %in% ids, , drop = FALSE])
  work <- file.path(root, "runs", cell_id, paste0("rep", r))
  dir.create(work, recursive = TRUE, mode = "0700", showWarnings = FALSE)
  cat(sprintf("START %s replicate %d seed %d\n", cell_id, r, seed))
  # The first use of held-out values for inference/metrics occurs in this
  # single scoring callback after training has completed with frozen settings.
  score <- function(fit, test, features) {
    prediction <- as.numeric(ds.flower.predict(fit, test[, features, drop = FALSE],
                                              type = "response"))
    result <- regression_metrics(test[[p$target]], prediction)
    write.csv(train, file.path(work, "train.csv"), row.names = FALSE)
    write.csv(test, file.path(work, "test.csv"), row.names = FALSE)
    processx::run(file.path(venv_root, "pytorch", "bin", "python"),
      c(file.path(script_dir, "central_train.py"),
        "--train", file.path(work, "train.csv"),
        "--test", file.path(work, "test.csv"), "--protocol", protocol_path,
        "--output", file.path(work, "baselines.json")), error_on_status = TRUE)
    result
  }
  fed <- campaign_run_federated(
    site_data = sites, test = test, features = features, feature_bounds = bounds,
    target_bounds = p$target_bounds, target = p$target, patient_column = p$patient_column,
    epsilon = epsilon, delta = p$privacy$delta, rounds = p$rounds,
    model_params = list(), work_dir = work, venv_root = venv_root,
    contract = contract, score_function = score)
  baseline <- jsonlite::fromJSON(file.path(work, "baselines.json"))
  stopifnot(fed$n_clients == 3L, fed$n_rounds_run == 5L, fed$n_failures == 0L,
            fed$cleanup_ok,
            all(vapply(fed$node_privacy, function(node) {
              identical(node$policy$dp_unit, "patient") &&
                identical(node$policy$patient_column, "subject#") &&
                node$clipping_norm == 1 && length(node$staged_configurations) > 0L
            }, logical(1))))
  per_replicate[[r]] <- list(
    seed = seed,
    split = list(n_train = nrow(train), n_test = nrow(test),
      n_per_site = vapply(sites, nrow, integer(1)),
      n_subjects_per_site = lengths(site_ids),
      train_subjects = sort(train_ids), test_subjects = sort(test_ids),
      site_subjects = lapply(site_ids, sort)),
    central = baseline$central, trivial = baseline$trivial,
    central_fit = baseline$central_fit, training_mean = baseline$training_mean,
    federated_dp = fed$metrics,
    gap_rmse = fed$metrics$rmse - baseline$central$rmse,
    diagnostic_pass = fed$metrics$rmse < baseline$trivial$rmse && fed$metrics$r2 > 0,
    node_privacy = fed$node_privacy,
    history = list(n_clients = fed$n_clients, n_failures = fed$n_failures,
                   n_rounds = fed$n_rounds_run, cleanup_ok = fed$cleanup_ok),
    model_sha256 = fed$model_sha256, federation_elapsed_s = fed$elapsed_s,
    elapsed_s = as.numeric(difftime(Sys.time(), started, units = "secs")))
  jsonlite::write_json(per_replicate[[r]], file.path(work, "replicate.json"),
                       pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat(sprintf("DONE rep %d: central RMSE %.6f | fed RMSE %.6f R2 %.6f | trivial %.6f\n",
    r, baseline$central$rmse, fed$metrics$rmse, fed$metrics$r2, baseline$trivial$rmse))
}
msd <- function(values) list(mean = mean(values), sd = sd(values))
branch_summary <- function(branch) setNames(lapply(c("rmse", "mae", "r2"), function(metric) {
  msd(vapply(per_replicate, function(rep) rep[[branch]][[metric]], numeric(1)))
}), c("rmse", "mae", "r2"))
summary <- list(central_mean_sd = branch_summary("central"),
  federated_mean_sd = branch_summary("federated_dp"),
  trivial_mean_sd = branch_summary("trivial"),
  gap_rmse_mean_sd = msd(vapply(per_replicate, function(rep) rep$gap_rmse, numeric(1))))
summary$diagnostic_pass <- summary$federated_mean_sd$rmse$mean <
  summary$trivial_mean_sd$rmse$mean && summary$federated_mean_sd$r2$mean > 0
artifact <- list(schema = "dsflower-campaign-v2", cell_id = cell_id,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  host = env$host, versions = env$versions, contract = contract,
  dataset = provenance, protocol_sha256 = protocol_sha, protocol = p,
  split = p$split, seeds = p$seeds,
  privacy = list(epsilon = epsilon, delta = p$privacy$delta,
                 unit = "patient", patient_column = "subject#", clipping_norm = 1),
  feature_bounds = setNames(lapply(seq_along(features), function(i) {
    list(lower = bounds$lower[i], upper = bounds$upper[i])
  }), features), target_bounds = p$target_bounds,
  rounds = p$rounds, model_params = defaults, model_params_overrides = list(),
  per_replicate = per_replicate, summary = summary,
  provisioning = list(started_at = readLines(file.path(root, "logs", "provision-start.txt")),
                      finished_at = readLines(file.path(root, "logs", "provision-end.txt"))))
jsonlite::write_json(artifact, output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("WROTE", output, "\n")
