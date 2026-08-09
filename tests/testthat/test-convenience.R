# Tests for R/convenience.R -- string-friendly user API
# The fit happy path is covered end-to-end by the live federated validation.

test_that("generic strategy constructor resolves aliases", {
  strategy <- ds.flower.strategy("yogi", server_learning_rate = 0.25)
  expect_s3_class(strategy, "dsflower_strategy")
  expect_equal(strategy$name, "FedYogi")
  expect_equal(strategy$params$eta, 0.25)

  expect_equal(ds.flower.strategy("fed-average")$name, "FedAvg")
  expect_equal(ds.flower.strategy("avgm")$name, "FedAvgM")
  expect_error(ds.flower.strategy("fedprox"), "not supported")
  expect_error(ds.flower.strategy("fedbn"), "not supported")
  expect_error(ds.flower.strategy("not_a_strategy"), "Unknown strategy")
})

test_that("generic task constructor resolves aliases", {
  expect_equal(ds.flower.task("class")$type, "classification")
  expect_error(ds.flower.task("survival"), "not supported")
  expect_error(ds.flower.task("segmentation"), "not supported")
  expect_error(ds.flower.task("not_a_task"), "Unknown task")
})

test_that("fit exposes only its executable argument contract", {
  expect_identical(names(formals(ds.flower.fit)), c(
    "conns", "data", "resource", "symbol", "target", "features", "model",
    "model_params", "torch_backend", "strategy", "strategy_params", "rounds",
    "task", "output_dir", "output_name", "silent", "verbose",
    "feature_bounds", "feature_cuts", "target_levels", "target_bounds",
    "allow_insecure_http", "data_kind"))
})

test_that("fit resolves strategy_params into the executable strategy", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y", features = "x",
    strategy = "fedadam",
    allow_insecure_http = "site",
    strategy_params = list(server_learning_rate = 0.2,
                           beta_1 = 0.8, beta_2 = 0.95, tau = 1e-4)
  )

  expect_s3_class(seen$strategy, "dsflower_strategy")
  expect_identical(seen$strategy$name, "FedAdam")
  expect_equal(seen$strategy$params,
               list(eta = 0.2, beta_1 = 0.8,
                    beta_2 = 0.95, tau = 1e-4))
  expect_identical(seen$allow_insecure_http, "site")
})

test_that("fit forwards registry parameters not present in model defaults", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y",
    features = paste0("x", 1:24), model = "pytorch_tcn",
    model_params = list(input_shape = c(2L, 12L), channels = 16L)
  )

  expect_identical(seen$model$params$input_shape, c(2L, 12L))
  expect_identical(seen$model$params$channels, 16L)
})

test_that("fit canonicalizes model aliases before submission", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y", features = "x",
    model = "quantile", model_params = list(quantile = 0.9)
  )

  expect_identical(seen$model$name, "pytorch_quantile")
  expect_equal(seen$model$params$quantile, 0.9)
})

test_that("fit rejects unknown model parameters before submission", {
  submitted <- FALSE
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      submitted <<- TRUE
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.fit(
      conns = list(site = TRUE), symbol = "D", target = "y", features = "x",
      model_params = list(max_iter = 100L)
    ),
    "Unknown parameter.*max_iter"
  )
  expect_false(submitted)
})

test_that("fit derives or requires the registered input kind", {
  registry <- get(".dsflower_models", envir = asNamespace("dsFlowerClient"))
  on.exit(rm(list = c("extension_image_test", "extension_dual_test"),
             envir = registry), add = TRUE)
  generator <- function(params) {
    list(kind = "sequential", layers = list(
      list(op = "linear", "in" = "@in", out = "@out")))
  }
  ds.flower.register_model(
    "extension_image_test", "neural", generator, loss = "bce_logits",
    parameter_types = character(), data_kinds = "image", overwrite = TRUE)
  ds.flower.register_model(
    "extension_dual_test", "neural", generator, loss = "bce_logits",
    parameter_types = character(), data_kinds = c("tabular", "image"),
    overwrite = TRUE)

  seen <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y",
    model = "extension_image_test")
  expect_identical(seen$data_kind, "image")

  expect_error(
    ds.flower.fit(
      conns = list(site = TRUE), symbol = "D", target = "y", features = "x",
      model = "extension_dual_test"),
    "set 'data_kind' explicitly")
  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y",
    model = "extension_dual_test", data_kind = "image")
  expect_identical(seen$data_kind, "image")
  expect_error(
    ds.flower.fit(
      conns = list(site = TRUE), symbol = "D", target = "y", features = "x",
      model = "extension_image_test", data_kind = "tabular"),
    "supported by model")
})

test_that("fit validates required arguments before connecting", {
  expect_error(ds.flower.fit(conns = list()), "target")
  expect_error(
    ds.flower.fit(conns = list(), target = "y", rounds = 1.5),
    "positive integer"
  )
  expect_error(
    ds.flower.fit(conns = list(), symbol = "D", data = "D", target = "y"),
    "only one"
  )
})
