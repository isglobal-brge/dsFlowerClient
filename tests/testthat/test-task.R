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

test_that("survival task has correct type", {
  task <- ds.flower.task.survival()
  expect_s3_class(task, "dsflower_task")
  expect_equal(task$type, "survival")
})

test_that("segmentation task has correct type", {
  task <- ds.flower.task.segmentation()
  expect_s3_class(task, "dsflower_task")
  expect_equal(task$type, "segmentation")
})

test_that("task prints correctly", {
  task <- ds.flower.task.classification()
  expect_output(print(task), "classification")
})
