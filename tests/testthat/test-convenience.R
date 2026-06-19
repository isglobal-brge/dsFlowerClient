# Tests for R/convenience.R -- string-friendly user API

test_that("generic model constructor resolves aliases", {
  model <- ds.flower.model("logreg", max_iter = 123L)
  expect_s3_class(model, "dsflower_model")
  expect_equal(model$name, "sklearn_logreg")
  expect_equal(model$params$max_iter, 123L)

  expect_equal(ds.flower.model("elastic-net")$name, "sklearn_elastic_net")
  expect_error(ds.flower.model("not_a_model"), "Unknown model")
  expect_error(ds.flower.model(list()), "dsflower_model")
})

test_that("generic strategy constructor resolves aliases", {
  strategy <- ds.flower.strategy("fedprox", proximal_mu = 0.25)
  expect_s3_class(strategy, "dsflower_strategy")
  expect_equal(strategy$name, "FedProx")
  expect_equal(strategy$params$proximal_mu, 0.25)

  expect_equal(ds.flower.strategy("fed-average")$name, "FedAvg")
  expect_error(ds.flower.strategy("not_a_strategy"), "Unknown strategy")
})

test_that("generic task and privacy constructors resolve aliases", {
  expect_equal(ds.flower.task("class")$type, "classification")
  expect_equal(ds.flower.task("survival")$type, "survival")
  expect_error(ds.flower.task("not_a_task"), "Unknown task")

  p <- ds.flower.privacy(epsilon = 2.0)
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$epsilon, 2.0)
})

test_that("recipe accepts string-friendly specs", {
  recipe <- ds.flower.recipe(
    model = "logreg",
    strategy = "fedavg",
    privacy = ds.flower.privacy(),
    target = "outcome",
    features = c("x1", "x2"),
    num_rounds = 3L
  )

  expect_s3_class(recipe, "dsflower_recipe")
  expect_equal(recipe$model$name, "sklearn_logreg")
  expect_equal(recipe$strategy$name, "FedAvg")
  expect_s3_class(recipe$privacy, "dsflower_privacy")
  expect_equal(recipe$num_rounds, 3L)
})

test_that("fit builds and runs a recipe through the happy path", {
  captured <- new.env(parent = emptyenv())

  testthat::local_mocked_bindings(
    ds.flower.connect = function(conns, data = NULL, resource = NULL,
                                  symbol = NULL) {
      captured$conns <- conns
      captured$symbol <- symbol
      structure(list(conns = conns, symbol = ".mock"), class = "dsflower_connection")
    },
    ds.flower.run = function(flower, recipe, ...) {
      captured$flower <- flower
      captured$recipe <- recipe
      captured$run_args <- list(...)
      structure(list(status = 0L, recipe = recipe), class = "dsflower_run")
    },
    ds.flower.disconnect = function(flower) {
      captured$disconnected <- TRUE
      invisible(TRUE)
    },
    .package = "dsFlowerClient"
  )

  run <- ds.flower.fit(
    conns = list(site1 = TRUE),
    symbol = "D",
    target = "outcome",
    features = c("x1", "x2"),
    model = "logreg",
    model_params = list(max_iter = 12L),
    strategy = "fedprox",
    strategy_params = list(proximal_mu = 0.2),
    privacy = ds.flower.privacy(),
    rounds = 2L,
    verbose = FALSE
  )

  expect_s3_class(run, "dsflower_run")
  expect_equal(captured$symbol, "D")
  expect_true(isTRUE(captured$disconnected))
  expect_equal(captured$recipe$model$name, "sklearn_logreg")
  expect_equal(captured$recipe$model$params$max_iter, 12L)
  expect_equal(captured$recipe$strategy$name, "FedProx")
  expect_equal(captured$recipe$strategy$params$proximal_mu, 0.2)
  expect_s3_class(captured$recipe$privacy, "dsflower_privacy")
  expect_equal(captured$recipe$num_rounds, 2L)
})

test_that("fit validates required arguments before connecting", {
  expect_error(ds.flower.fit(conns = list()), "target")
  expect_error(
    ds.flower.fit(conns = list(), symbol = "D", data = "D", target = "y"),
    "only one"
  )
})
