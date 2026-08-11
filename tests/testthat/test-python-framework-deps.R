test_that("framework health checks cover every required import", {
  checks <- dsFlowerClient:::.FRAMEWORK_CHECK_MODULE

  expect_setequal(
    checks$pytorch,
    c("torch", "numpy", "pandas", "pyarrow", "opacus", "cryptography")
  )
  expect_setequal(names(checks), c("pytorch", "pytorch_vision"))
  expect_true(all(c("SimpleITK", "monai") %in% checks$pytorch_vision))

  observed <- list()
  local_mocked_bindings(
    .client_python_cmd = function() "python",
    .framework_modules_available = function(python, modules) {
      observed[[length(observed) + 1L]] <<- list(
        python = python, modules = modules)
      TRUE
    },
    .package = "dsFlowerClient"
  )

  for (framework in names(checks)) {
    expect_true(dsFlowerClient:::.ensure_client_framework(framework))
  }
  expect_identical(
    lapply(observed, `[[`, "modules"),
    unname(checks)
  )
})

test_that("client dependency contracts remain limited to PyTorch runtimes", {
  deps <- dsFlowerClient:::.FRAMEWORK_CLIENT_DEPS

  expect_setequal(names(deps), c("pytorch", "pytorch_vision"))
  expect_null(deps$xgboost)
  expect_true("cryptography>=42.0.0" %in% deps$pytorch)
  expect_error(
    dsFlowerClient:::.ensure_client_framework("xgboost"),
    "Unsupported client framework")
})

test_that("prediction routing accepts only declarative PyTorch artifacts", {
  model_dir <- withr::local_tempdir()
  file.create(file.path(model_dir, "model.pt"))
  expect_identical(
    dsFlowerClient:::.resolve_model_for_predict(model_dir)$framework,
    "pytorch")

  for (artifact in c("booster.json", "model.xgb", "model.xgb.json")) {
    retired_dir <- withr::local_tempdir()
    path <- file.path(retired_dir, artifact)
    file.create(path)
    expect_error(
      dsFlowerClient:::.resolve_model_for_predict(retired_dir),
      "Expected model.pt")
    expect_error(
      dsFlowerClient:::.resolve_model_for_predict(path),
      "Unknown model format")
  }
})
