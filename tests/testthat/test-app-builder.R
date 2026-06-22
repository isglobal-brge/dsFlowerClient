# Tests for R/app_builder.R — Tier-1 harness app generation (dsFlower 2.0)

test_that(".harness_dependencies pins the harness runtime", {
  deps <- dsFlowerClient:::.harness_dependencies()
  expect_true(any(grepl("flwr", deps)))
  expect_true(any(grepl("torch", deps)))
  expect_true(any(grepl("opacus", deps)))
})

test_that(".harness_dependencies adds the vision stack only for image runs", {
  base <- dsFlowerClient:::.harness_dependencies(vision = FALSE)
  expect_false(any(grepl("torchvision", base)))
  vis <- dsFlowerClient:::.harness_dependencies(vision = TRUE)
  expect_true(any(grepl("torchvision", vis)))
  expect_true(any(grepl("monai", vis)))       # 3D path works out of the box
  expect_true(any(grepl("nibabel|SimpleITK|nrrd", vis)))  # medical readers
})

test_that(".harness_vision_backbone maps vision models and rejects segmentation", {
  rn <- ds.flower.recipe(model = ds.flower.model.pytorch_resnet18(), target = "y")
  expect_equal(dsFlowerClient:::.harness_vision_backbone(rn), "resnet18")
  dn <- ds.flower.recipe(model = ds.flower.model.pytorch_densenet121(), target = "y")
  expect_equal(dsFlowerClient:::.harness_vision_backbone(dn), "densenet121")
  # volumetric opt-in selects the true-3D backbone
  v3 <- ds.flower.recipe(
    model = ds.flower.model.pytorch_resnet18(volumetric = TRUE), target = "y")
  expect_equal(dsFlowerClient:::.harness_vision_backbone(v3), "resnet18_3d")
  # tabular models are not vision
  tab <- ds.flower.recipe(model = ds.flower.model.pytorch_logreg(),
                          target = "y", features = c("a", "b"))
  expect_null(dsFlowerClient:::.harness_vision_backbone(tab))
  # segmentation is rejected (linear-probe head is classification only)
  seg <- ds.flower.recipe(model = ds.flower.model.pytorch_unet2d(), target = "y")
  expect_error(dsFlowerClient:::.harness_vision_backbone(seg), "segmentation")
})

test_that(".write_pyproject_toml emits image config for a vision recipe", {
  recipe <- ds.flower.recipe(
    model = ds.flower.model.pytorch_resnet18(n_classes = 3L, image_size = 256L),
    target = "label", num_rounds = 4L)
  app_dir <- withr::local_tempdir()
  dsFlowerClient:::.write_pyproject_toml(app_dir, recipe, n_nodes = 2L)
  toml <- readLines(file.path(app_dir, "pyproject.toml"))
  expect_true(any(grepl('data-kind = "image"', toml, fixed = TRUE)))
  expect_true(any(grepl('backbone = "resnet18"', toml, fixed = TRUE)))
  expect_true(any(grepl("num-classes = 3", toml, fixed = TRUE)))
  expect_true(any(grepl("image-size = 256", toml, fixed = TRUE)))
  expect_false(any(grepl("num-features", toml)))   # irrelevant for image runs
  expect_true(any(grepl("torchvision", toml)))     # vision deps present
})

test_that(".toml_kv formats scalars correctly", {
  expect_equal(dsFlowerClient:::.toml_kv("key", "value"), 'key = "value"')
  expect_equal(dsFlowerClient:::.toml_kv("key", 42L), "key = 42")
  expect_equal(dsFlowerClient:::.toml_kv("key", 3.14), "key = 3.14")
  expect_equal(dsFlowerClient:::.toml_kv("key", TRUE), "key = true")
})

test_that(".harness_model maps recipes to MVP architectures", {
  logreg <- ds.flower.recipe(model = "logreg", target = "y",
                             features = c("a", "b"))
  mlp <- ds.flower.recipe(model = "mlp", target = "y", features = c("a", "b"))
  expect_equal(dsFlowerClient:::.harness_model(logreg), "logreg")
  expect_equal(dsFlowerClient:::.harness_model(mlp), "mlp")
})

test_that(".harness_hyperparams applies defaults and overrides", {
  base <- ds.flower.recipe(model = "logreg", target = "y", features = "a")
  hp <- dsFlowerClient:::.harness_hyperparams(base)
  expect_equal(hp[["batch-size"]], 32L)
  expect_equal(hp[["local-epochs"]], 1L)

  tuned <- ds.flower.recipe(model = "logreg", target = "y", features = "a")
  tuned$model$params$learning_rate <- 0.2
  tuned$model$params$batch_size <- 8L
  hp2 <- dsFlowerClient:::.harness_hyperparams(tuned)
  expect_equal(hp2[["learning-rate"]], 0.2)
  expect_equal(hp2[["batch-size"]], 8L)
})

test_that(".write_pyproject_toml emits harness config, no SecAgg/no pyproject-DP", {
  recipe <- ds.flower.recipe(
    model = "logreg", strategy = "fedavg", target = "y",
    features = c("x1", "x2", "x3"), num_rounds = 4L)

  app_dir <- withr::local_tempdir()
  dsFlowerClient:::.write_pyproject_toml(app_dir, recipe, results_dir = "/tmp/out",
                                         n_nodes = 3L)
  toml <- paste(readLines(file.path(app_dir, "pyproject.toml")), collapse = "\n")

  expect_true(grepl("num-server-rounds = 4", toml, fixed = TRUE))
  expect_true(grepl("num-features = 3", toml, fixed = TRUE))
  expect_true(grepl('model = "logreg"', toml, fixed = TRUE))
  expect_true(grepl("min-train-nodes = 3", toml, fixed = TRUE))
  expect_true(grepl('clientapp = "dsflower_harness.client_app:app"', toml,
                    fixed = TRUE))
  expect_true(grepl('results-dir = "/tmp/out"', toml, fixed = TRUE))
  # DP comes from the node manifest, not pyproject; SecAgg is gone.
  expect_false(grepl("privacy-epsilon", toml, fixed = TRUE))
  expect_false(grepl("secure-aggregation", toml, fixed = TRUE))
})

test_that(".write_pyproject_toml requires explicit features", {
  recipe <- ds.flower.recipe(model = "logreg", target = "y")
  app_dir <- withr::local_tempdir()
  expect_error(
    dsFlowerClient:::.write_pyproject_toml(app_dir, recipe),
    "feature columns")
})

test_that(".build_flower_app copies the bundled harness skeleton", {
  skip_if_not(nzchar(dsFlowerClient:::.harness_skeleton_dir()),
              "harness skeleton not available")
  recipe <- ds.flower.recipe(model = "logreg", strategy = "fedavg",
                             target = "y", features = c("a", "b"),
                             num_rounds = 1L)
  app_dir <- dsFlowerClient:::.build_flower_app(recipe, conns = NULL)
  on.exit(unlink(app_dir, recursive = TRUE))
  expect_true(file.exists(file.path(app_dir, "pyproject.toml")))
  expect_true(file.exists(file.path(app_dir, "dsflower_harness",
                                     "client_app.py")))
})
