# Tests for R/app_builder.R — Flower App Generation

test_that(".template_dependencies returns correct sklearn deps", {
  deps <- dsFlowerClient:::.template_dependencies("sklearn")
  expect_true("scikit-learn>=1.0.0" %in% deps)
  expect_true(any(grepl("flwr", deps)))
  expect_true(any(grepl("numpy", deps)))
})

test_that(".template_dependencies returns correct pytorch deps", {
  deps <- dsFlowerClient:::.template_dependencies("pytorch")
  expect_true("torch>=2.0.0" %in% deps)
  expect_true("opacus>=1.4.0" %in% deps)
  expect_true(any(grepl("flwr", deps)))
})

test_that(".toml_kv formats strings correctly", {
  expect_equal(dsFlowerClient:::.toml_kv("key", "value"), 'key = "value"')
})

test_that(".toml_kv formats numbers correctly", {
  expect_equal(dsFlowerClient:::.toml_kv("key", 42L), "key = 42")
  expect_equal(dsFlowerClient:::.toml_kv("key", 3.14), "key = 3.14")
})

test_that(".toml_kv formats booleans correctly", {
  expect_equal(dsFlowerClient:::.toml_kv("key", TRUE), "key = true")
})

test_that(".build_flower_app creates app directory", {
  skip_if_not(
    nzchar(system.file("flower_templates", "sklearn_logreg",
                        package = "dsFlowerClient")),
    "Package not installed with templates"
  )

  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.sklearn_logreg(),
    strategy = ds.flower.strategy.fedavg(),
    num_rounds = 3L,
    target_column = "target"
  )

  app_dir <- dsFlowerClient:::.build_flower_app(recipe)
  on.exit(unlink(app_dir, recursive = TRUE))

  expect_true(dir.exists(app_dir))
  expect_true(file.exists(file.path(app_dir, "pyproject.toml")))
})

test_that(".write_pyproject_toml generates valid content", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.sklearn_logreg(penalty = "l1", C = 0.5),
    strategy = ds.flower.strategy.fedavg(),
    num_rounds = 10L,
    target_column = "y"
  )

  app_dir <- tempfile("app_test")
  dir.create(app_dir, recursive = TRUE)
  on.exit(unlink(app_dir, recursive = TRUE))

  dsFlowerClient:::.write_pyproject_toml(app_dir, recipe)

  toml_path <- file.path(app_dir, "pyproject.toml")
  expect_true(file.exists(toml_path))

  content <- readLines(toml_path)
  content_str <- paste(content, collapse = "\n")

  expect_true(grepl("num-server-rounds = 10", content_str))
  expect_true(grepl('task-type = "classification"', content_str))
  expect_true(grepl('penalty = "l1"', content_str))
  expect_true(grepl("C = 0.5", content_str))
  expect_true(grepl('strategy = "FedAvg"', content_str))
})

test_that(".write_pyproject_toml emits the DP-always contract (SecAgg case)", {
  recipe <- ds.flower.recipe(
    task = ds.flower.task.classification(),
    model = ds.flower.model.pytorch_resnet18(n_classes = 2L),
    strategy = ds.flower.strategy.fedavg(),
    privacy = ds.flower.privacy(epsilon = 2.0),
    num_rounds = 2L,
    target = "label"
  )
  # Simulate >=3 nodes -> Secure Aggregation (distributed DP).
  recipe$use_secagg <- TRUE
  recipe$strategy$params$min_available_clients <- 3L

  app_dir <- tempfile("app_test")
  dir.create(app_dir, recursive = TRUE)
  on.exit(unlink(app_dir, recursive = TRUE))

  dsFlowerClient:::.write_pyproject_toml(app_dir, recipe)
  content_str <- paste(readLines(file.path(app_dir, "pyproject.toml")),
                       collapse = "\n")

  expect_true(grepl("dp-enabled = true", content_str, fixed = TRUE))
  expect_true(grepl("privacy-epsilon = 2", content_str, fixed = TRUE))
  expect_true(grepl("require-secure-aggregation = true", content_str,
                    fixed = TRUE))
  expect_true(grepl("allow-per-node-metrics = false", content_str,
                    fixed = TRUE))
  expect_true(grepl("fixed-client-sampling = true", content_str,
                    fixed = TRUE))
})
