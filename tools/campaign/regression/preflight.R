#!/usr/bin/env Rscript
root <- normalizePath(commandArgs(TRUE)[1])
script_dir <- file.path(root, "dsFlowerClient", "tools", "campaign", "regression")
source(file.path(script_dir, "campaign_lib.R"))
p <- jsonlite::fromJSON(file.path(script_dir, "protocol.json"))
runner <- dsFlowerClient:::.compute_local_runner_hash()
stopifnot(packageVersion("dsFlower") == "0.5.0",
          packageVersion("dsFlowerClient") == "0.5.0",
          identical(runner, dsFlower:::.compute_app_pkg_hash("dsflower_runner")),
          identical(runner, dsFlowerClient:::.compute_local_runner_hash(
            file.path(root, "dsFlower", "inst", "flower_app", "dsflower_runner"))))
defaults <- dsFlowerClient:::.dsflower_get_model(p$contract)$defaults
stopifnot(defaults$learning_rate == 0.01, defaults$local_epochs == 1L,
          defaults$batch_size == 32L, defaults$weight_decay == 0)
set.seed(731)
sites <- lapply(1:3, function(site) {
  frame <- as.data.frame(setNames(lapply(seq_along(p$features), function(i) {
    runif(24, p$feature_bounds$lower[i], p$feature_bounds$upper[i])
  }), p$features), check.names = FALSE)
  frame[[p$patient_column]] <- rep(site * 100 + 1:12, each = 2)
  frame[[p$target]] <- seq(10, 20, length.out = 24)
  frame
})
fed <- campaign_run_federated(
  site_data = sites, test = NULL, features = p$features,
  feature_bounds = p$feature_bounds, target_bounds = p$target_bounds,
  target = p$target, patient_column = p$patient_column,
  epsilon = 1, delta = 1e-6, rounds = 5L, model_params = list(),
  work_dir = file.path(root, "runs", "synthetic-preflight"),
  venv_root = Sys.getenv("DSFLOWER_VENV_ROOT"), contract = p$contract,
  score_function = function(fit, test, features) {
    prediction <- as.numeric(ds.flower.predict(fit, sites[[1]][, features, drop = FALSE],
                                              type = "response"))
    stopifnot(length(prediction) == 24L, all(is.finite(prediction)))
    list(scored = FALSE, synthetic_prediction_count = length(prediction))
  })
stopifnot(fed$n_clients == 3L, fed$n_rounds_run == 5L, fed$n_failures == 0L,
          fed$cleanup_ok, all(vapply(fed$node_privacy, function(node) {
            node$policy$dp_unit == "patient" && node$clipping_norm == 1 &&
              length(node$staged_configurations) > 0L
          }, logical(1))))
installed <- installed.packages()
record <- list(schema = "dsflower-regression-runtime-v1",
  runner_sha256 = runner, environment = campaign_env_info(Sys.getenv("DSFLOWER_VENV_ROOT")),
  python_packages = setNames(lapply(c("venvs/native-tree", "venvs/pytorch", "client/venv"),
    function(venv) strsplit(processx::run("uv", c("pip", "freeze", "--python",
      file.path(root, venv, "bin", "python")), error_on_status = TRUE)$stdout,
      "\n", fixed = TRUE)[[1]]), c("native_tree", "pytorch", "client")),
  r_packages = as.list(setNames(installed[, "Version"], installed[, "Package"])),
  synthetic_preflight = fed[c("metrics", "node_privacy", "cleanup_ok", "n_clients",
                              "n_failures", "n_rounds_run", "model_sha256", "elapsed_s")])
out <- file.path(root, "dsFlowerClient", "inst", "extdata", "campaign", "regression")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(record, file.path(out, "runtime.json"),
                     pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("PREFLIGHT PASSED: unchanged runner, patient privacy, three sites, five rounds; no cohort scoring.\n")
