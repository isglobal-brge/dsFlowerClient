.association_capability_fixture <- function(hash, available = TRUE,
                                             privacy_unit = "row") {
  list(
    runner_abi = 3L,
    runner_sha256 = hash,
    privacy_unit = privacy_unit,
    association = list(
      contract = "dsflower-binary-association-3x3/v1",
      result_contract = "dsflower-binary-association-result/v1",
      mechanism = "binary-association-joint-gaussian/v1",
      execution_profile = "dsflower-binary-association-execution/v1",
      shape = c(3L, 3L),
      order = "exposure-major/outcome-minor",
      pooled_only = TRUE,
      privacy_units = c("row", "patient"),
      availability_semantics = "fresh-executable-node-probe",
      probed = TRUE,
      available = available))
}

.association_result_fixture <- function(
    contract_sha, job_sha, available = TRUE, n_nodes = 2L,
    unit_semantics = "row-one-hot/v1") {
  value <- list(
    available = available,
    association_contract = "dsflower-binary-association-3x3/v1",
    association_contract_sha256 = contract_sha,
    association_job_sha256 = job_sha,
    contract = "dsflower-binary-association-result/v1",
    n_nodes = n_nodes,
    pooled_only = TRUE,
    privacy = list(
      adjacency = "replace-one",
      mechanism = "binary-association-joint-gaussian/v1",
      scope = "per-job-node-dp",
      sticky = TRUE),
    schema = 1L,
    unit_semantics = unit_semantics)
  if (isTRUE(available)) {
    value$measures <- list(
      odds_ratio = 2,
      prevalence_difference = 0.2,
      prevalence_exposed = 0.6,
      prevalence_ratio = 1.5,
      prevalence_unexposed = 0.4)
    value$noise_sd_pooled <- 3.5
    value$table_dp <- list(
      c(60, 40, 5), c(40, 60, 4), c(3, 2, 1))
  }
  value
}

.write_association_result_fixture <- function(path, value,
                                               commit_available = NULL) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(
    value, file.path(path, "association.json"), auto_unbox = TRUE,
    null = "null", digits = NA)
  if (is.null(commit_available)) commit_available <- value$available
  jsonlite::write_json(
    list(list(round = 1L, available = commit_available)),
    file.path(path, "history.json"), auto_unbox = TRUE,
    null = "null", digits = NA)
}

test_that("association request and job hashes mirror the server fixtures", {
  expect_identical(
    dsFlowerClient:::.association_contract_sha256(
      "outcome", "exposure", c("no", "yes"), c(0, 1), "row"),
    "d95556c2379b4ab24e65772f883e3d4da9f9b06a3d8769dd2ccaead0e36ae858")
  expect_identical(
    dsFlowerClient:::.association_job_sha256(
      paste(rep("a", 64L), collapse = ""), 3L,
      paste(rep("b", 64L), collapse = ""), 2L),
    "ff0dcf02fc1b52f546733a6c2272834bebc7cab04d86e3436fc235c06bcc8268")
  expect_identical(
    dsFlowerClient:::.association_level_spec(
      factor(c("control", "case")), "levels"),
    list(type = "string", values = c("control", "case")))
  expect_error(
    dsFlowerClient:::.association_level_spec(c(TRUE, TRUE), "levels"),
    "must be distinct")
  expect_error(
    dsFlowerClient:::.association_job_sha256(
      paste(rep("a", 64L), collapse = ""), 4L,
      paste(rep("b", 64L), collapse = ""), 2L),
    "runner_abi=3")
})

test_that("association capability uses only the targeted dependency-light probe", {
  hash <- paste(rep("a", 64L), collapse = "")
  conns <- list(site1 = list(), site2 = list())
  captured <- NULL
  local_mocked_bindings(
    .compute_local_runner_hash = function(...) hash,
    .package = "dsFlowerClient")
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      captured <<- expr
      list(
        site1 = .association_capability_fixture(hash),
        site2 = .association_capability_fixture(hash))
    },
    .package = "DSI")

  capability <- dsFlowerClient:::.assert_association_capability(conns)
  expect_identical(capability$privacy_unit, "row")
  expect_identical(capability$n_nodes, 2L)
  expect_identical(capability$runner_sha256, hash)
  expect_identical(as.character(captured[[1L]]), "flowerGetCapabilitiesDS")
  expect_identical(as.character(captured[[2L]]), "none")
  expect_identical(as.character(captured[[3L]]), "runtime")
  expect_length(captured, 3L)

  local_mocked_bindings(
    datashield.aggregate = function(...) list(
      site1 = .association_capability_fixture(hash),
      site2 = .association_capability_fixture(hash, available = FALSE)),
    .package = "DSI")
  expect_error(
    dsFlowerClient:::.assert_association_capability(conns),
    "unavailable on: site2")
})

test_that("association FAB uses exact dedicated refs without neural dependencies", {
  app <- dsFlowerClient:::.build_submission_app(
    list(pkg_dir = NULL, track = "association"),
    c('dp-track = "association"'), tempdir(), vision = FALSE)
  toml <- paste(readLines(file.path(app, "pyproject.toml")), collapse = "\n")
  expect_match(
    toml, 'serverapp = "dsflower_runner.association_server_app:app"',
    fixed = TRUE)
  expect_match(
    toml, 'clientapp = "dsflower_runner.association_client_app:app"',
    fixed = TRUE)
  expect_match(toml, 'flwr==1.31.0', fixed = TRUE)
  expect_match(toml, 'numpy==2.4.6', fixed = TRUE)
  expect_false(grepl("torch", toml, fixed = TRUE))
  expect_false(grepl("opacus", toml, fixed = TRUE))
})

test_that("association result reader accepts only an atomic pooled commit", {
  contract_sha <- paste(rep("a", 64L), collapse = "")
  job_sha <- paste(rep("b", 64L), collapse = "")
  path <- withr::local_tempdir()
  value <- .association_result_fixture(contract_sha, job_sha)
  .write_association_result_fixture(path, value)
  result <- dsFlowerClient:::.read_association_result(
    path, contract_sha, job_sha, 2L, "row")
  expect_true(result$available)
  expect_identical(dim(result$table_dp), c(3L, 3L))
  expect_equal(result$measures$prevalence_ratio, 1.5)

  unavailable_path <- withr::local_tempdir()
  unavailable <- .association_result_fixture(
    contract_sha, job_sha, available = FALSE)
  .write_association_result_fixture(unavailable_path, unavailable)
  result <- dsFlowerClient:::.read_association_result(
    unavailable_path, contract_sha, job_sha, 2L, "row")
  expect_false(result$available)
  expect_false(any(c("table_dp", "measures", "noise_sd_pooled") %in%
                     names(result)))
})

test_that("association result reader rejects swaps, partial output, and domains", {
  contract_sha <- paste(rep("a", 64L), collapse = "")
  job_sha <- paste(rep("b", 64L), collapse = "")
  value <- .association_result_fixture(contract_sha, job_sha)

  path <- withr::local_tempdir()
  extra <- value
  extra$per_node <- list()
  .write_association_result_fixture(path, extra)
  expect_error(
    dsFlowerClient:::.read_association_result(
      path, contract_sha, job_sha, 2L, "row"),
    "pooled-only contract")

  path <- withr::local_tempdir()
  bad_domain <- value
  bad_domain$measures$prevalence_exposed <- 1.1
  .write_association_result_fixture(path, bad_domain)
  expect_error(
    dsFlowerClient:::.read_association_result(
      path, contract_sha, job_sha, 2L, "row"),
    "descriptive measures")

  path <- withr::local_tempdir()
  logical_schema <- value
  logical_schema$schema <- TRUE
  .write_association_result_fixture(path, logical_schema)
  expect_error(
    dsFlowerClient:::.read_association_result(
      path, contract_sha, job_sha, 2L, "row"),
    "pooled-only contract")

  path <- withr::local_tempdir()
  logical_nodes <- value
  logical_nodes$n_nodes <- TRUE
  .write_association_result_fixture(path, logical_nodes)
  expect_error(
    dsFlowerClient:::.read_association_result(
      path, contract_sha, job_sha, 2L, "row"),
    "pooled-only contract")

  path <- withr::local_tempdir()
  .write_association_result_fixture(path, value, commit_available = FALSE)
  expect_error(
    dsFlowerClient:::.read_association_result(
      path, contract_sha, job_sha, 2L, "row"),
    "pooled-only contract")

  path <- withr::local_tempdir()
  .write_association_result_fixture(path, value)
  expect_error(
    dsFlowerClient:::.read_association_result(
      path, paste(rep("c", 64L), collapse = ""), job_sha, 2L, "row"),
    "pooled-only contract")
})

test_that("association public API exposes no privacy or modelling knobs", {
  args <- names(formals(ds.flower.associate))
  expect_false(any(c(
    "epsilon", "delta", "seed", "budget", "strata", "covariates",
    "correction", "time", "model") %in% args))
  expect_true(all(c(
    "conns", "outcome", "exposure", "outcome_levels",
    "exposure_levels") %in% args))
})

test_that("association API sends only the frozen prepare and FAB pins", {
  hash <- paste(rep("a", 64L), collapse = "")
  captured_prepare <- NULL
  captured_app <- NULL
  captured_ensure <- "unset"
  captured_watch <- NULL
  client_env <- get(".dsflower_client_env",
                    envir = asNamespace("dsFlowerClient"))
  old_superlink <- client_env$.superlink
  withr::defer({
    client_env$.superlink <- old_superlink
  })
  client_env$.superlink <- list(flwr_home = tempdir())

  local_mocked_bindings(
    .validate_dsi_transport_security = function(...) TRUE,
    .assert_association_capability = function(...) list(
      privacy_unit = "row", runner_abi = 3L, runner_sha256 = hash,
      n_nodes = 2L, capabilities = list()),
    .require_flwr_cli = function(...) TRUE,
    ds.flower.connect = function(conns, data, resource, symbol) {
      structure(list(conns = conns, symbol = "flower"),
                class = "dsflower_connection")
    },
    ds.flower.disconnect = function(...) invisible(TRUE),
    ds.flower.nodes.prepare = function(
        conns, symbol, target_column, feature_columns, run_config) {
      captured_prepare <<- list(
        target = target_column, feature = feature_columns,
        config = run_config)
      invisible(TRUE)
    },
    .build_submission_app = function(sub, config_lines, ...) {
      captured_app <<- list(sub = sub, config = config_lines)
      tempdir()
    },
    ds.flower.link.up = function(...) invisible(TRUE),
    ds.flower.link.down = function(...) invisible(TRUE),
    ds.flower.nodes.ensure = function(conns, symbol, torch_backend) {
      captured_ensure <<- torch_backend
      invisible(TRUE)
    },
    ds.flower.nodes.cleanup = function(...) invisible(TRUE),
    .client_flwr_cmd = function(...) "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., expect_artifacts) {
      captured_watch <<- expect_artifacts
      list(status = 0L, stdout = "", stderr = "")
    },
    .read_association_result = function(...) list(
      available = TRUE, table_dp = matrix(1, 3, 3),
      measures = list(
        odds_ratio = 1, prevalence_difference = 0,
        prevalence_exposed = 0.5, prevalence_ratio = 1,
        prevalence_unexposed = 0.5),
      noise_sd_pooled = 1, n_nodes = 2L,
      privacy = list(scope = "per-job-node-dp")),
    .package = "dsFlowerClient")

  result <- ds.flower.associate(
    list(site1 = list(), site2 = list()),
    outcome = "outcome", exposure = "exposure",
    outcome_levels = c("no", "yes"), exposure_levels = c(0, 1),
    symbol = "D")
  expect_s3_class(result, "dsflower_association")
  expect_identical(captured_prepare$target, "outcome")
  expect_identical(captured_prepare$feature, "exposure")
  expect_setequal(names(captured_prepare$config), c(
    "dp-track", "data_type", "num-server-rounds",
    "association-outcome-levels", "association-exposure-levels",
    "association-contract-sha256", "association-n-nodes",
    "association-job-sha256"))
  expect_false(any(c("task-type", "loss-name", "epsilon", "delta") %in%
                     names(captured_prepare$config)))
  expect_identical(captured_app$sub$track, "association")
  expect_true(any(grepl("association-contract-sha256", captured_app$config,
                        fixed = TRUE)))
  expect_true(any(grepl("association-job-sha256", captured_app$config,
                        fixed = TRUE)))
  expect_null(captured_ensure)
  expect_false(captured_watch)
})
