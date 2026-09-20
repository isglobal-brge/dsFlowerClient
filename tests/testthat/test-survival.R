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
  expect_error(ds.flower.submit(list(), model, target = c("t", "e"),
                               features = "x", holdout = 0.2), "Survival private")
  expect_error(ds.flower.fit(list(), model = model, target = c("t", "e"),
                            features = "x", cross_validation = 3), "Survival private")
})

test_that("survival artifact metadata supports local inference and rejects private metrics", {
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
  expect_error(dsFlowerClient:::.resolve_validation_contract(model_dir, 32),
               "Survival private")
  # A local HPO objective cannot bypass the nested private-evaluation preflight.
  objective <- function(params) dsFlowerClient:::.resolve_validation_contract(model_dir, 32)
  expect_error(objective(list()), "private HPO")
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
    expect_error(dsFlowerClient:::.assert_holdout_supported(sub, "tabular"),
                 "Survival private")
    expect_error(dsFlowerClient:::.assert_cross_validation_supported(sub, "tabular"),
                 "Survival private")
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
