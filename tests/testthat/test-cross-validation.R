test_that("client CV contract is canonical, bounded, and seed-free", {
  expect_identical(dsFlowerClient:::.normalize_cross_validation(NULL), NULL)
  expect_identical(
    dsFlowerClient:::.normalize_cross_validation(3L)$folds, 3L)
  expect_error(dsFlowerClient:::.normalize_cross_validation(1L), "[2, 10]")
  expect_error(dsFlowerClient:::.normalize_cross_validation(11L), "[2, 10]")
  expect_error(dsFlowerClient:::.normalize_cross_validation(3.5), "[2, 10]")

  contract <- dsFlowerClient:::.cross_validation_contract(
    dsFlowerClient:::.normalize_cross_validation(3L), "row")
  expect_identical(contract$method, "cross_validation")
  expect_identical(contract$folds, 3L)
  expect_identical(
    contract$sha256,
    "b0ef0610df2006b87a348929a3f903a4e90d87c860d829063a897b2a40a21390")
  expect_false(any(grepl(
    "seed|salt|nonce", names(contract), ignore.case = TRUE)))
})

cv_job_config <- function(contract) {
  list(
    "task-type" = "classification", "dp-track" = "neural",
    "num-server-rounds" = 2L, "num-features" = 2L,
    "cv-version" = contract$version, "cv-method" = contract$method,
    "cv-assignment" = contract$assignment, "cv-folds" = contract$folds,
    "cv-privacy-unit" = contract$privacy_unit,
    "cv-unit-canonicalization" = contract$unit_canonicalization,
    "cv-contract-sha256" = contract$sha256,
    "cv-validation-bins" = 32L, "cv-n-nodes" = 2L,
    "cv-job-sha256" = strrep("f", 64L),
    "strategy" = "fedadam", "strategy-eta" = 0.1,
    "strategy-eta-l" = 0.01, "strategy-beta-1" = 0.9,
    "strategy-beta-2" = 0.99, "strategy-tau" = 0.001,
    "model-spec-b64" = "e30=", "loss-name" = "bce_logits",
    "num-classes" = 2L, "num-labels" = 2L,
    "local-epochs" = 1L, "batch-size" = 32L,
    "learning-rate" = 0.01, "weight-decay" = 0, "l1-penalty" = 0,
    "optimizer-name" = "sgd", "optimizer-momentum" = 0,
    "optimizer-nesterov" = FALSE, "scheduler-name" = "none",
    "feature-bounds" = list(lower = c(0, -1), upper = c(1, 1)),
    "target-levels" = list(type = "character", values = c("no", "yes")))
}

test_that("CV job provenance has a mirrored golden and binds every public group", {
  config <- cv_job_config(dsFlowerClient:::.cross_validation_contract(
    dsFlowerClient:::.normalize_cross_validation(3L), "row"))
  hash <- function(value = config, runner_abi = 3L,
                   runner_sha = strrep("a", 64L),
                   policy_sha = strrep("b", 64L), clipping = 1) {
    dsFlowerClient:::.cv_job_sha256(
      value, c("x1", "x2"), "y", runner_abi, runner_sha,
      policy_sha, clipping)
  }
  baseline <- hash()
  expect_identical(
    baseline,
    "5a5dc6cac8b3407895a656fa834862d2dd7615203b0fca5fcdaddf8990ce739e")

  mutations <- list(
    cv_contract = c(config[-match("cv-contract-sha256", names(config))],
                    list("cv-contract-sha256" = strrep("c", 64L))),
    cv_bins = within(config, `cv-validation-bins` <- 64L),
    rounds = within(config, `num-server-rounds` <- 3L),
    nodes = within(config, `cv-n-nodes` <- 3L),
    strategy = within(config, `strategy-eta` <- 0.2),
    eta_l = within(config, `strategy-eta-l` <- 0.02),
    schema = within(config, `feature-bounds`$upper[[1L]] <- 2),
    task = within(config, `task-type` <- "count"),
    model_spec = within(config, `model-spec-b64` <- "W10="),
    loss = within(config, `loss-name` <- "hinge"),
    classes = within(config, `num-classes` <- 3L),
    labels = within(config, `num-labels` <- 3L),
    epochs = within(config, `local-epochs` <- 2L),
    batch = within(config, `batch-size` <- 16L),
    training = within(config, `learning-rate` <- 0.02))
  expect_true(all(vapply(mutations, function(value) {
    !identical(hash(value), baseline)
  }, logical(1))))
  expect_false(identical(hash(runner_abi = 4L), baseline))
  expect_false(identical(hash(runner_sha = strrep("d", 64L)), baseline))
  expect_false(identical(hash(policy_sha = strrep("e", 64L)), baseline))
  expect_false(identical(hash(clipping = 2), baseline))

  irrelevant <- config
  irrelevant$run_token <- "run_private"
  irrelevant$`results-dir` <- "/tmp/not-part-of-provenance"
  irrelevant$timestamp <- "2099-01-01"
  irrelevant$seed <- 123L
  irrelevant$n_samples <- 999L
  expect_identical(hash(irrelevant), baseline)
})

test_that("CV requires one common public runner and node privacy contract", {
  capability <- list(
    runner_abi = 3L, runner_sha256 = strrep("a", 64L),
    privacy_policy_sha256 = strrep("b", 64L),
    privacy_clipping_norm = 1, privacy_unit = "row")
  common <- dsFlowerClient:::.cross_validation_common_capabilities(list(
    site_a = capability, site_b = capability))
  expect_identical(common$runner_abi, 3L)
  expect_identical(common$privacy_clipping_norm, 1)

  mixed <- list(site_a = capability, site_b = capability)
  mixed$site_b$privacy_policy_sha256 <- strrep("c", 64L)
  expect_error(
    dsFlowerClient:::.cross_validation_common_capabilities(mixed),
    "same runner, privacy policy")
})

test_that("cross_validate defaults to three real folds and forwards one job", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.fit = function(...) {
      seen <<- list(...)
      structure(list(metrics = list(accuracy = 0.5)), class = "dsflower_cv")
    },
    .package = "dsFlowerClient"
  )
  result <- ds.flower.cross_validate(
    conns = list(site = TRUE), symbol = "D", target = "y", features = "x")
  expect_s3_class(result, "dsflower_cv")
  expect_identical(seen$cross_validation, 3L)
  expect_identical(seen$data_kind, "tabular")
  expect_false("holdout" %in% names(seen))
})

test_that("CV rejects unsupported engines and holdout combinations before IO", {
  expect_error(
    dsFlowerClient:::.assert_cross_validation_supported(
      list(track = "native_tree"), "tabular"),
    "neural")
  expect_error(
    dsFlowerClient:::.assert_cross_validation_supported(
      list(track = "neural"), "image"),
    "tabular")

  expect_error(
    ds.flower.submit(
      conns = list(site = TRUE), model = "pytorch_logreg",
      target = "y", features = "x", holdout = 0.2,
      cross_validation = 3L),
    "cannot be combined")
})

test_that("pooled CV output is finite, provenance-pinned, and transcript-free", {
  root <- withr::local_tempdir()
  contract <- dsFlowerClient:::.cross_validation_contract(
    dsFlowerClient:::.normalize_cross_validation(3L), "row")
  payload <- list(
    pooled_only = TRUE,
    privacy = "node-dp-pooled-postprocessing",
    method = "cross_validation",
    task = "binary",
    n_nodes = 2L,
    folds = 3L,
    cv_contract_sha256 = contract$sha256,
    cv_job_sha256 = strrep("a", 64L),
    metrics = list(accuracy = 0.75, roc_auc = 0.8))
  jsonlite::write_json(
    payload, file.path(root, "cv.json"), auto_unbox = TRUE)
  value <- dsFlowerClient:::.read_cross_validation_result(root)
  expect_identical(value$cv_contract_sha256, contract$sha256)
  expect_true(is.finite(value$metrics$accuracy))
  expect_false(any(c(
    "per_node", "per_site", "predictions", "models") %in% names(value)))

  payload$metrics$accuracy <- Inf
  jsonlite::write_json(
    payload, file.path(root, "cv.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_cross_validation_result(root), "pooled-only")

  payload$metrics$accuracy <- 1.1
  jsonlite::write_json(
    payload, file.path(root, "cv.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_cross_validation_result(root), "pooled-only")

  payload$metrics$accuracy <- 0.75
  payload$metrics$per_fold <- list(0.7, 0.8, 0.75)
  jsonlite::write_json(
    payload, file.path(root, "cv.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_cross_validation_result(root), "pooled-only")
})

test_that("run accepts exactly cv.json and verifies the submitted contract", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(
    process = list(is_alive = function() TRUE),
    flwr_home = withr::local_tempdir())
  recipe <- ds.flower.recipe(
    model = ds.flower.model.pytorch_logreg(), num_rounds = 2L,
    features = "x")
  recipe$model$track <- "neural"
  recipe$cross_validation_contract <-
    dsFlowerClient:::.cross_validation_contract(
      dsFlowerClient:::.normalize_cross_validation(3L), "row")
  recipe$cross_validation_n_nodes <- 2L
  provenance <- cv_job_config(recipe$cross_validation_contract)
  job_hash <- function(config = provenance,
                       policy = strrep("b", 64L)) {
    dsFlowerClient:::.cv_job_sha256(
      config, c("x1", "x2"), "y", 3L, strrep("a", 64L), policy, 1)
  }
  recipe$cross_validation_job_sha256 <- job_hash()
  release <- list(
    task = "binary", folds = 3L, n_nodes = 2L,
    cv_contract_sha256 = recipe$cross_validation_contract$sha256,
    cv_job_sha256 = recipe$cross_validation_job_sha256,
    metrics = list(accuracy = 0.75))

  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .ensure_client_framework = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(...) list(
      status = 0L, stdout = "run_id=cv-job", stderr = ""),
    .read_model_weights = function(...) NULL,
    .read_training_history = function(...) NULL,
    .read_holdout_result = function(...) NULL,
    .read_cross_validation_result = function(...) release,
    .package = "dsFlowerClient")

  result <- ds.flower.run.start(
    recipe, conns = list(a = TRUE, b = TRUE),
    app_dir = withr::local_tempdir(), silent = TRUE)
  expect_s3_class(result, "dsflower_cv")
  expect_identical(result$folds, 3L)
  expect_equal(result$metrics$accuracy, 0.75)
  expect_null(result$saved_path)
  expect_match(paste(capture.output(print(result)), collapse = "\n"),
               "no fold transcript")

  release$cv_contract_sha256 <- paste(rep("f", 64L), collapse = "")
  expect_error(
    ds.flower.run.start(
      recipe, conns = list(a = TRUE, b = TRUE),
      app_dir = withr::local_tempdir(), silent = TRUE),
    "does not match")

  # Same K/task/node count is insufficient: results from another model,
  # horizon, strategy, schema, or node policy each have a different job pin.
  release$cv_contract_sha256 <- recipe$cross_validation_contract$sha256
  swapped_hashes <- c(
    model = job_hash(within(provenance, `model-spec-b64` <- "W10=")),
    rounds = job_hash(within(provenance, `num-server-rounds` <- 3L)),
    strategy = job_hash(within(provenance, `strategy-eta` <- 0.2)),
    schema = job_hash(within(
      provenance, `feature-bounds`$upper[[1L]] <- 2)),
    policy = job_hash(policy = strrep("c", 64L)))
  expect_false(any(swapped_hashes == recipe$cross_validation_job_sha256))
  for (swapped_hash in swapped_hashes) {
    release$cv_job_sha256 <- swapped_hash
    expect_error(
      ds.flower.run.start(
        recipe, conns = list(a = TRUE, b = TRUE),
        app_dir = withr::local_tempdir(), silent = TRUE),
      "does not match")
  }
})
