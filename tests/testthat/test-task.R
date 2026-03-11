# Tests for R/task.R — Task Specs

test_that("classification task has correct type", {
  task <- ds.flower.task.classification()
  expect_s3_class(task, "dsflower_task")
  expect_equal(task$type, "classification")
})

test_that("regression task has correct type", {
  task <- ds.flower.task.regression()
  expect_s3_class(task, "dsflower_task")
  expect_equal(task$type, "regression")
})

test_that("task prints correctly", {
  task <- ds.flower.task.classification()
  expect_output(print(task), "classification")
})
