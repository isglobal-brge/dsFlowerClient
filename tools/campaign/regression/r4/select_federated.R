#!/usr/bin/env Rscript
# Training-only real-DP schedule confirmation; immutable fit guards.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1])
seed <- as.integer(args[2])
candidate <- args[3]
stopifnot(seed %in% 20260820:20260822, candidate %in% c("sgd_lr003_e5_b64", "sgd_lr001_e5_b32"))
params <- list(optimizer = "sgd", learning_rate = if (candidate == "sgd_lr003_e5_b64") .03 else .01, local_epochs = 5L, batch_size = if (candidate == "sgd_lr003_e5_b64") 64L else 32L)
script_dir <- dirname(normalizePath(sub("--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[1])))
source(file.path(script_dir, "..", "campaign_lib.R"))
p <- jsonlite::fromJSON(file.path(script_dir, "..", "cdcbmi_public_units_protocol.json"))
input_dir <- file.path(root, "r4", paste0("selection_inner_seed", seed))
work <- file.path(root, "r4", paste0("selection_", candidate, "_", seed))
dir.create(work, recursive = TRUE, showWarnings = FALSE, mode = "0700")
guard <- file.path(work, "training.started")
if (file.exists(guard)) stop("Diagnostic training already started; refusing a rerun.")
features <- p$features
raw_sites <- lapply(1:3, function(site) {
  read.csv(file.path(input_dir, paste0("site", site, ".csv")), check.names = FALSE)
})
validation <- read.csv(file.path(input_dir, "validation.csv"), check.names = FALSE)
training <- do.call(rbind, raw_sites)
stopifnot(all(vapply(raw_sites, nrow, integer(1)) == 9600L),
          nrow(training) == 28800L, nrow(validation) == 7200L,
          identical(names(training), c(features, "BMI")),
          identical(names(validation), names(training)),
          all(is.finite(as.matrix(training))),
          all(is.finite(as.matrix(validation))))
defaults <- dsFlowerClient:::.dsflower_get_model(p$contract)$defaults
stopifnot(defaults$learning_rate == 0.01, defaults$batch_size == 32L,
          defaults$local_epochs == 1L, defaults$optimizer == "sgd",
          defaults$scheduler == "none", defaults$weight_decay == 0,
          defaults$l1_penalty == 0, length(defaults$hidden_layers) == 0L)
venv_root <- Sys.getenv("DSFLOWER_VENV_ROOT")
environment <- campaign_env_info(venv_root)
environment$versions$runner_sha256 <- dsFlowerClient:::.compute_local_runner_hash()
stopifnot(identical(environment$versions$runner_sha256,
                    dsFlower:::.compute_app_pkg_hash("dsflower_runner")))

metrics <- function(y, prediction) {
  stopifnot(length(y) == length(prediction), all(is.finite(prediction)))
  residual <- prediction - y
  list(rmse = sqrt(mean(residual^2)), mae = mean(abs(residual)),
       r2 = 1 - sum(residual^2) / sum((y - mean(y))^2),
       mean_residual = mean(residual), mean_prediction = mean(prediction))
}
transform_features <- function(frame) {
  values <- as.matrix(frame[, features, drop = FALSE])
  lower <- p$feature_bounds$lower
  upper <- p$feature_bounds$upper
  for (j in seq_along(features)) {
    values[, j] <- (pmin(upper[j], pmax(lower[j], values[, j])) -
                    (lower[j] + upper[j]) / 2) / ((upper[j] - lower[j]) / 2)
  }
  cbind(intercept = 1, values)
}
x_train <- transform_features(training)
x_validation <- transform_features(validation)
ols <- stats::lm.fit(x_train, training$BMI)
stopifnot(ols$rank == ncol(x_train), all(is.finite(ols$coefficients)))
mean_bmi <- mean(training$BMI)
baselines <- list(
  training = list(trivial = metrics(training$BMI, rep(mean_bmi, nrow(training))),
                  ols = metrics(training$BMI, drop(x_train %*% ols$coefficients))),
  validation = list(trivial = metrics(validation$BMI, rep(mean_bmi, nrow(validation))),
                    ols = metrics(validation$BMI, drop(x_validation %*% ols$coefficients))))
sites <- lapply(raw_sites, function(site) {
  site$BMI <- (pmin(98, pmax(12, site$BMI)) - 55) / 43
  site
})
score <- function(fit, test, features) {
  saveRDS(fit, file.path(work, "fit.rds"))
  result <- lapply(list(training = training, validation = validation), function(frame) {
    public <- as.numeric(ds.flower.predict(
      fit, frame[, features, drop = FALSE], type = "response"))
    bmi <- 55 + 43 * public
    list(metrics = metrics(frame$BMI, bmi),
         predictions = data.frame(BMI = frame$BMI, public_prediction = public,
                                  BMI_prediction = bmi))
  })
  for (name in names(result)) {
    write.csv(result[[name]]$predictions,
              file.path(work, paste0(name, "_predictions.csv")), row.names = FALSE)
  }
  result <- lapply(result, `[[`, "metrics")
  jsonlite::write_json(result, file.path(work, "released_model_scores.json"),
                       pretty = TRUE, auto_unbox = TRUE, digits = NA)
  result
}
writeLines(format(Sys.time(), tz = "UTC", usetz = TRUE), guard)
fed <- campaign_run_federated(
  site_data = sites, test = NULL, features = features,
  feature_bounds = p$feature_bounds, target_bounds = p$target_bounds,
  target = "BMI", patient_column = NULL, epsilon = 8, delta = 1e-6,
  rounds = 5L, model_params = params, work_dir = work,
  venv_root = venv_root, contract = p$contract, score_function = score)
stopifnot(fed$n_clients == 3L, fed$n_rounds_run == 5L, fed$n_failures == 0L,
          fed$cleanup_ok, all(vapply(fed$node_privacy, function(node) {
            identical(node$policy$dp_unit, "row") && node$clipping_norm == 1
          }, logical(1))))
result <- list(purpose = "training-only schedule selection", seed = seed, candidate = candidate, model_params = params,
               split = jsonlite::fromJSON(file.path(input_dir, "staging.json")),
               sealed_test_opened = FALSE, environment = environment,
               n_inner_training = nrow(training), n_inner_validation = nrow(validation),
               baselines = baselines, federated = fed)
jsonlite::write_json(result, file.path(work, "selection_replicate.json"),
                     pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("Training RMSE %.6f; inner-validation RMSE %.6f; trivial %.6f; OLS %.6f\n",
            fed$metrics$training$rmse, fed$metrics$validation$rmse,
            baselines$validation$trivial$rmse, baselines$validation$ols$rmse))
