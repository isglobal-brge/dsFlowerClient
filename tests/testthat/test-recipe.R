# Tests for R/recipe.R — Composable Recipe
# Recipe building is exercised end-to-end by every live ds.flower.fit/submit run.

test_that("recipe builds (DP is server-enforced; no client privacy knob)", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.pytorch_mlp(),
    strategy = ds.flower.strategy.fedadam()
  )
  expect_s3_class(recipe, "dsflower_recipe")
  expect_named(recipe, c(
    "task", "model", "strategy", "num_rounds", "target", "features"))
  expect_null(recipe$privacy)
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

test_that("recipe rejects handcrafted unsupported task specs", {
  unsupported <- structure(list(type = "survival"), class = "dsflower_task")
  expect_error(
    ds.flower.recipe(model = ds.flower.model.pytorch_logreg(),
                     task = unsupported),
    "not supported"
  )
})

test_that("recipe infers and enforces the registered model task", {
  expect_identical(
    ds.flower.recipe(model = "pytorch_huber")$task$type,
    "regression")
  expect_identical(
    ds.flower.recipe(model = "pytorch_quantile")$task$type,
    "regression")
  expect_identical(
    ds.flower.recipe(model = "pytorch_poisson")$task$type,
    "count")
  expect_identical(ds.flower.task("count")$type, "count")
  expect_error(
    ds.flower.recipe(
      model = "pytorch_huber", task = "classification"),
    "incompatible")
  expect_error(
    ds.flower.recipe(
      model = "pytorch_poisson", task = "regression"),
    "incompatible")
  expect_error(ds.flower.recipe(model = "pytorch_logreg", num_rounds = "2"),
               "integer")
})
