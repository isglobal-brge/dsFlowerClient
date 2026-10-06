test_that("AFT constructors resolve distributions into exact wire contracts", {
  for (distribution in c("weibull", "lognormal")) {
    model <- ds.flower.model.pytorch_aft(
      horizon = 3650, distribution = distribution, time_scale = 365)
    generic <- ds.flower.model("pytorch_aft", horizon = 3650,
                               distribution = distribution, time_scale = 365)
    expect_identical(model, generic)
    sub <- dsFlowerClient:::.emit_submission(model)
    expect_identical(sub$loss, paste0("aft_", distribution, "_nll"))
    recipe <- ds.flower.recipe(model, target = c("time", "event"))
    expect_identical(recipe$task$type, "survival")
    wire <- dsFlowerClient:::.neural_training_config(sub$params, sub$loss)
    config <- jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
      wire[["survival-config-b64"]])))
    expect_identical(config$distribution, distribution)
    expect_equal(config$horizon, 3650)
    expect_equal(config$time_scale, 365)
    expect_equal(config$schema_version, 1)
    expect_false(any(c("epsilon", "delta", "clip", "patient_column") %in%
                       ds.flower.model_parameters("pytorch_aft")$name))
  }
  expect_true(all(c("pytorch_aft", "pytorch_logreg", "pytorch_lstm",
                    "xgboost", "random_forest") %in% ds.flower.list_models()$name))
})

test_that("survival public domain and role failures happen before transport", {
  expect_error(ds.flower.model("pytorch_aft"), "requires parameter")
  for (dispersion in c(0.1, 3, Inf)) {
    expect_error(ds.flower.model.pytorch_aft(10, dispersion = dispersion),
                 "dispersion")
  }
  expect_error(ds.flower.model.pytorch_aft(10, distribution = "cox"), "distribution")
  expect_error(ds.flower.model.pytorch_aft(10, t_min = 11), "t_min")
  expect_error(ds.flower.model.pytorch_aft(10, time_unit = "months"), "days")
  expect_error(ds.flower.model.pytorch_aft(1e6, time_scale = 1e-6,
                                          dispersion = 2), "envelope")
  expect_error(ds.flower.model.pytorch_aft(10, epsilon = 8), "Unknown parameter")
  model <- ds.flower.model.pytorch_aft(10)
  for (target in list("time", c("t", "e", "z"), c("t", "t"))) {
    expect_error(ds.flower.recipe(model, target = target), "target")
  }
  expect_error(ds.flower.recipe(model, target = c("t", "e"),
                               task = "regression"), "incompatible")
  expect_error(ds.flower.submit(list(), model, target = c("t", "e"),
                               features = c("x", "t")), "distinct")
  expect_error(ds.flower.submit(list(), model, target = c("t", "e"),
                               features = "x", target_levels = c(0, 1)),
               "target_levels")
  sub <- dsFlowerClient:::.emit_submission(model)
  expect_true(dsFlowerClient:::.assert_holdout_supported(sub, "tabular"))
  expect_true(dsFlowerClient:::.assert_cross_validation_supported(sub, "tabular"))
})

test_that("survival artifact metadata supports local inference and private metric contracts", {
  model <- ds.flower.model.pytorch_aft(10, distribution = "lognormal")
  sub <- dsFlowerClient:::.emit_submission(model)
  config <- dsFlowerClient:::.survival_config(sub$params, sub$loss)
  model_dir <- tempfile()
  dir.create(model_dir)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)
  jsonlite::write_json(list(track = "neural", data_kind = "tabular",
    features = "x", model_spec = sub$spec, model_params = sub$params,
    loss_name = sub$loss, survival_config = config),
    file.path(model_dir, "metadata.json"), auto_unbox = TRUE)
  contract <- dsFlowerClient:::.read_meta_model_contract(model_dir)
  expect_identical(contract$loss_name, "aft_lognormal_nll")
  expect_equal(contract$survival_config, config)
  writeBin(charToRaw("public artifact"), file.path(model_dir, "model.pt"))
  validation <- dsFlowerClient:::.resolve_validation_contract(model_dir, 32)
  expect_identical(validation$task, "survival")
  expect_equal(validation$survival_config, config)
})

test_that("fit accepts survival task and preserves ordered target roles", {
  seen <- NULL
  local_mocked_bindings(ds.flower.submit = function(...) {
    seen <<- list(...)
    structure(list(), class = "dsflower_run")
  }, .package = "dsFlowerClient")
  ds.flower.fit(list(site = TRUE), model = "pytorch_aft",
    model_params = list(horizon = 20, distribution = "lognormal"),
    task = "survival", features = "x", target = c("time", "event"))
  expect_identical(seen$model$loss, "aft_lognormal_nll")
  expect_identical(seen$target, c("time", "event"))
})

test_that("public survival horizon shorthand matches explicit prepared fit and CV contracts", {
  seen <- NULL
  conns <- list(site = structure(list(), class = "DSLiteConnection"))
  capability <- list(privacy_unit = "patient", runner_abi = 3L,
    runner_sha256 = strrep("a", 64L), privacy_policy_sha256 = strrep("b", 64L),
    privacy_clipping_norm = 1)
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .validate_declarative_model_preflight = function(...) TRUE,
    ds.flower.connect = function(...) {
      structure(list(conns = conns, symbol = "flower"), class = "dsflower_connection")
    },
    .assert_runner_compatibility = function(...) list(site = capability),
    .segmentation_prepare_nodes = function(conns, symbol, target, features, config, local) {
      seen <<- list(target = target, features = features, config = config)
      stop("captured validated survival preparation", call. = FALSE)
    },
    ds.flower.link.down = function(...) invisible(TRUE),
    ds.flower.nodes.cleanup = function(...) invisible(TRUE),
    .dsflower_disconnect_on_exit = function(...) invisible(TRUE),
    .package = "dsFlowerClient")
  capture <- function(model, params, mode, horizons = c(5, 10, 20)) {
    args <- list(conns = conns, symbol = "D", features = "x",
      target = c("time", "event"), task = ds.flower.task.survival(),
      model = model, model_params = params, survival_horizons = horizons,
      rounds = 1L, feature_bounds = list(lower = -1, upper = 1), silent = TRUE)
    if (identical(mode, "holdout")) args$holdout <- 0.2
    if (identical(mode, "cv")) args$folds <- 3L
    seen <<- NULL
    expect_error(do.call(if (identical(mode, "cv")) ds.flower.cross_validate else
      ds.flower.fit, args), "captured validated survival preparation")
    expect_type(seen$config, "list")
    seen
  }
  for (model in c("pytorch_aft", "pytorch_discrete_hazard")) {
    explicit <- if (model == "pytorch_aft") list(horizon = 20) else
      list(edges = c(0, 5, 10, 20))
    for (mode in c("ordinary", "holdout", "cv")) {
      inferred <- capture(model, list(), mode)
      declared <- capture(model, explicit, mode)
      expect_identical(inferred, declared)
      expect_identical(inferred$target, c("time", "event"))
      config <- inferred$config
      survival <- jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
        config[["survival-config-b64"]])))
      expect_equal(survival$horizon, 20)
      if (model == "pytorch_discrete_hazard") {
        expect_equal(survival$edges, c(0, 5, 10, 20))
        expect_identical(length(survival$edges) - 1L, 3L)
      }
      fields <- grep("^validation-", names(config), value = TRUE)
      if (mode == "ordinary") {
        expect_length(fields, 0L)
        expect_false(any(grepl("^(holdout|cv)-", names(config))))
      } else {
        expect_identical(jsonlite::fromJSON(config[["validation-survival-horizons"]]),
                         c(5L, 10L, 20L))
        expect_identical(config[["validation-survival-nll-bound"]], 20)
      }
    }
  }
  # Explicit domains, distributions and concrete-model overrides retain precedence.
  aft <- capture("pytorch_aft", list(horizon = 30, distribution = "lognormal"), "ordinary")
  aft_config <- jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
    aft$config[["survival-config-b64"]])))
  expect_equal(aft_config$horizon, 30)
  expect_identical(aft$config[["loss-name"]], "aft_lognormal_nll")
  hazard <- capture("pytorch_discrete_hazard", list(edges = c(0, 10, 30)), "ordinary")
  expect_equal(jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
    hazard$config[["survival-config-b64"]])))$edges, c(0, 10, 30))
  concrete <- capture(ds.flower.model.pytorch_aft(40), list(horizon = 50), "ordinary")
  expect_equal(jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
    concrete$config[["survival-config-b64"]])))$horizon, 50)
  one_interval <- capture("pytorch_discrete_hazard", list(), "ordinary", horizons = 20)
  expect_equal(jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
    one_interval$config[["survival-config-b64"]])))$edges, c(0, 20))
  fractional <- capture("pytorch_aft", list(t_min = 0.1), "ordinary", horizons = c(0.5, 1))
  expect_equal(jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
    fractional$config[["survival-config-b64"]])))$t_min, 0.1)
  expect_identical(capture("pytorch-aft", list(), "ordinary"),
                   capture("pytorch_aft", list(), "ordinary"))
})

test_that("survival shorthand cannot repair invalid public domains before transport", {
  transport <- 0L
  local_mocked_bindings(.require_flwr_cli = function() {
    transport <<- transport + 1L
    stop("unexpected transport")
  }, ds.flower.connect = function(...) {
    transport <<- transport + 1L
    stop("unexpected transport")
  }, .package = "dsFlowerClient")
  fit <- function(model, horizons = c(5, 10, 20), params = list()) {
    ds.flower.fit(list(site = TRUE), model = model, model_params = params,
      target = c("time", "event"), features = "x", survival_horizons = horizons)
  }
  for (model in c("pytorch_aft", "pytorch_discrete_hazard")) {
    for (bad in list(numeric(), TRUE, "20", c(5, NA), c(5, Inf), c(5, 5),
                     c(10, 5), c(0, 5), c(-1, 5), 1:65, c(5, 1e6 + 1))) {
      expect_error(fit(model, bad), "survival_horizons")
    }
    expect_error(fit(model, NULL), "requires parameter")
    expect_error(fit(model, c(0.5, 1)), "survival_horizons")
    parameter <- if (model == "pytorch_aft") "horizon" else "edges"
    expect_error(fit(model, params = stats::setNames(list(NULL), parameter)),
                 "cannot be NULL")
    expect_error(fit(model, params = stats::setNames(list(-1), parameter)),
                 "Invalid parameter|edges|horizon")
  }
  expect_error(ds.flower.model("pytorch_aft"), "requires parameter")
  expect_error(ds.flower.model("pytorch_discrete_hazard"), "requires parameter")
  expect_error(fit("pytorch_aft", params = list(horizon = 10)), "survival_horizons")
  expect_error(fit(ds.flower.model.pytorch_aft(10)), "survival_horizons")
  expect_error(fit("pytorch_logreg"), "Survival metric options require a survival model")
  expect_identical(transport, 0L)
})

test_that("hazard public grids preserve subjects and exact target semantics", {
  for (edges in list(c(0, 10), c(0, 5, 10), 0:64)) {
    model <- ds.flower.model.pytorch_discrete_hazard(edges)
    expect_identical(model, ds.flower.model("pytorch_discrete_hazard", edges = edges))
    sub <- dsFlowerClient:::.emit_submission(model)
    expect_identical(sub$loss, "discrete_hazard_nll")
    config <- dsFlowerClient:::.survival_config(sub$params, sub$loss)
    expect_identical(config$edges, edges)
    expect_equal(config$horizon, tail(edges, 1))
    expect_false(any(c("distribution", "dispersion", "time_scale") %in% names(config)))
    expect_identical(ds.flower.recipe(model, target = c("t", "e"))$task$type,
                     "survival")
    expect_true(dsFlowerClient:::.assert_holdout_supported(sub, "tabular"))
    expect_true(dsFlowerClient:::.assert_cross_validation_supported(sub, "tabular"))
    expect_identical(dsFlowerClient:::.validate_submission_target(sub, c("t", "e")),
                     c("t", "e"))
  }
  for (edges in list(0, c(1, 2), c(0, 1, 1), c(0, 3, 2), 0:65,
                    c(0, Inf), c(0, 1e6 + 1))) {
    expect_error(ds.flower.model.pytorch_discrete_hazard(edges),
                 "edges|horizon")
  }
  expect_error(ds.flower.model.pytorch_discrete_hazard(c(0, 10), t_min = 11), "t_min")
  expect_error(ds.flower.model.pytorch_discrete_hazard(c(0, 10), time_scale = 1),
               "Unknown parameter")
})

test_that("survival wire pins preserve fractional public time boundaries", {
  edges <- c(0, 1.000000123456789, 3.123456789012345)
  model <- ds.flower.model.pytorch_discrete_hazard(edges)
  sub <- dsFlowerClient:::.emit_submission(model)
  config <- dsFlowerClient:::.neural_training_config(sub$params, sub$loss)
  decoded <- jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
    config[["survival-config-b64"]])))
  expect_identical(decoded$edges, edges)
  expect_identical(decoded$horizon, tail(edges, 1L))
})

test_that("HPO objective context rejects survival before node contact and restores nesting", {
  evaluate <- dsFlowerClient:::.hpo_evaluate_objective
  context <- dsFlowerClient:::.DSFLOWER_HPO_CONTEXT
  expect_false(context$active)
  model <- ds.flower.model.pytorch_aft(10)
  train <- function(params) ds.flower.submit(list(), model,
    target = c("time", "event"), features = "x")
  expect_error(evaluate(train, list()), "Survival training inside HPO")
  expect_false(context$active)
  value <- evaluate(function(params) {
    expect_true(context$active)
    expect_identical(evaluate(function(params) 7, list()), 7)
    expect_true(context$active)
    expect_error(evaluate(function(params) stop("nested"), list()), "nested")
    expect_true(context$active)
    3
  }, list())
  expect_identical(value, 3)
  expect_false(context$active)
})

test_that("the public HPO API rejects a survival fit in its first trial", {
  python <- tryCatch(dsFlowerClient:::.local_hpo_python_cmd(), error = function(e) "")
  skip_if(!nzchar(python), "Optuna 4.8.0 is required")
  local_mocked_bindings(.local_hpo_python_cmd = function() python,
                       .package = "dsFlowerClient")
  expect_error(ds.flower.hpo(function(params) {
    ds.flower.fit(list(), model = "pytorch_aft",
      model_params = list(horizon = 10), target = c("t", "e"), features = "x")
  }, list(x = ds.flower.hpo.float(0, 1)), n_trials = 1),
  "Survival training inside HPO")
  expect_false(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
})

test_that("portable survival bundles retain exact public grids after reload", {
  directory <- withr::local_tempdir()
  destination <- withr::local_tempdir()
  edges <- c(0, 1.000000123456789, 3.123456789012345)
  sub <- dsFlowerClient:::.emit_submission(ds.flower.model.pytorch_discrete_hazard(edges))
  config <- dsFlowerClient:::.survival_config(sub$params, sub$loss)
  metadata <- list(model = "pytorch_discrete_hazard", data_kind = "tabular",
    model_spec = sub$spec, model_params = sub$params, loss_name = sub$loss,
    survival_config = config, features = c("x", "z"))
  jsonlite::write_json(metadata, file.path(directory, "metadata.json"),
                      auto_unbox = TRUE, digits = I(17))
  saveRDS(metadata, file.path(directory, "model.rds"))
  writeBin(charToRaw("public test artifact"), file.path(directory, "model.pt"))
  run <- structure(list(output_dir = directory, available = TRUE,
                        model_file = file.path(directory, "model.rds")),
                   class = "dsflower_run")
  for (ext in c("json", "rds")) {
    path <- file.path(destination, paste0("survival.", ext))
    ds.flower.save_model(run, path)
    loaded <- ds.flower.load_model(path)
    contract <- dsFlowerClient:::.read_meta_model_contract(loaded$source_dir)
    expect_identical(contract$survival_config$edges, edges)
    expect_identical(dsFlowerClient:::.resolve_model_for_predict(loaded)$contract$survival_config$edges,
                     edges)
    expect_identical(contract$loss_name, "discrete_hazard_nll")
  }
})

test_that("single-feature survival bounds remain exact vectors on the wire", {
  lower <- -1.000000123456789
  upper <- 1.123456789012345
  decoded <- jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(
    dsFlowerClient:::.survival_bounds_b64(list(lower = lower, upper = upper)))),
    simplifyVector = FALSE)
  expect_identical(decoded$lower, list(lower))
  expect_identical(decoded$upper, list(upper))
})
