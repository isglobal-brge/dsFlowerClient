# Tests for R/recipe.R — Composable Recipe
# (Tests that asserted the removed sklearn_* model constructors were deleted with
# the legacy sklearn track; recipe building is exercised end-to-end by every live
# ds.flower.fit/submit run.)

test_that("recipe builds (DP is server-enforced; no client privacy knob)", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.pytorch_mlp(),
    strategy = ds.flower.strategy.fedadam()
  )
  expect_s3_class(recipe, "dsflower_recipe")
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
  expect_error(
    ds.flower.recipe(model = ds.flower.model.pytorch_logreg(), masks = "mask"),
    "segmentation"
  )
})
