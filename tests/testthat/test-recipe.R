# Tests for R/recipe.R — Composable Recipe
# (Tests that asserted the removed sklearn_* model constructors were deleted with
# the legacy sklearn track; recipe building is exercised end-to-end by every live
# ds.flower.fit/submit run.)

test_that("recipe with custom DP budget", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.pytorch_mlp(),
    strategy = ds.flower.strategy.fedprox(),
    privacy = ds.flower.privacy(epsilon = 0.5)
  )
  expect_equal(recipe$privacy$epsilon, 0.5)
  expect_s3_class(recipe$privacy, "dsflower_privacy")
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
