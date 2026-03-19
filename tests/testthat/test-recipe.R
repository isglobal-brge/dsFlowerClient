# Tests for R/recipe.R — Composable Recipe

test_that("recipe creates correct structure", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.sklearn_logreg(),
    strategy = ds.flower.strategy.fedavg(),
    num_rounds = 10L,
    target_column = "target",
    feature_columns = c("f1", "f2")
  )

  expect_s3_class(recipe, "dsflower_recipe")
  expect_equal(recipe$task$type, "classification")
  expect_equal(recipe$model$name, "sklearn_logreg")
  expect_equal(recipe$strategy$name, "FedAvg")
  expect_equal(recipe$privacy$mode, "clinical_default")
  expect_equal(recipe$num_rounds, 10L)
  expect_equal(recipe$target_column, "target")
  expect_equal(recipe$feature_columns, c("f1", "f2"))
})

test_that("recipe defaults to clinical_default privacy", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.regression(),
    model = ds.flower.model.sklearn_ridge(),
    strategy = ds.flower.strategy.fedavg()
  )
  expect_equal(recipe$privacy$mode, "clinical_default")
})

test_that("recipe with clinical_dp privacy", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.pytorch_mlp(),
    strategy = ds.flower.strategy.fedprox(),
    privacy = ds.flower.privacy.clinical_dp(epsilon = 0.5)
  )
  expect_equal(recipe$privacy$mode, "clinical_dp")
  expect_equal(recipe$privacy$params$epsilon, 0.5)
})

test_that("recipe validates task type", {
  expect_error(
    ds.flower.recipe(
      task = list(type = "classification"),
      model = ds.flower.model.sklearn_logreg(),
      strategy = ds.flower.strategy.fedavg()
    ),
    "dsflower_task"
  )
})

test_that("recipe validates model type", {
  expect_error(
    ds.flower.recipe(
      task = ds.flower.task.classification(),
      model = list(name = "fake"),
      strategy = ds.flower.strategy.fedavg()
    ),
    "dsflower_model"
  )
})

test_that("recipe validates strategy type", {
  expect_error(
    ds.flower.recipe(
      task = ds.flower.task.classification(),
      model = ds.flower.model.sklearn_logreg(),
      strategy = list(name = "fake")
    ),
    "dsflower_strategy"
  )
})

test_that("recipe validates privacy type", {
  expect_error(
    ds.flower.recipe(
      task = ds.flower.task.classification(),
      model = ds.flower.model.sklearn_logreg(),
      strategy = ds.flower.strategy.fedavg(),
      privacy = list(mode = "fake")
    ),
    "dsflower_privacy"
  )
})

test_that("recipe prints correctly", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.sklearn_logreg(),
    strategy = ds.flower.strategy.fedavg()
  )
  expect_output(print(recipe), "dsflower_recipe")
  expect_output(print(recipe), "classification")
  expect_output(print(recipe), "sklearn_logreg")
  expect_output(print(recipe), "FedAvg")
})
