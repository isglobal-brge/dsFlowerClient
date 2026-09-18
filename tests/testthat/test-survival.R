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
