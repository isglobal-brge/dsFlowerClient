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

test_that("framework readiness caches one managed runtime snapshot", {
  cache <- dsFlowerClient:::.client_framework_cache
  dsFlowerClient:::.clear_client_framework_cache()
  withr::defer(dsFlowerClient:::.clear_client_framework_cache())

  python <- "/managed/python-a"
  fingerprint <- "venv-a"
  probes <- list()
  local_mocked_bindings(
    .client_python_cmd = function() python,
    .client_framework_venv_fingerprint = function(...) fingerprint,
    .framework_modules_available = function(python, modules) {
      probes[[length(probes) + 1L]] <<- list(
        python = python, modules = modules)
      TRUE
    },
    .package = "dsFlowerClient"
  )

  for (index in seq_len(20L)) {
    expect_true(dsFlowerClient:::.ensure_client_framework("pytorch"))
  }
  expect_length(probes, 1L)

  expect_true(dsFlowerClient:::.ensure_client_framework("pytorch_vision"))
  python <- "/managed/python-b"
  expect_true(dsFlowerClient:::.ensure_client_framework("pytorch"))
  fingerprint <- "venv-b"
  expect_true(dsFlowerClient:::.ensure_client_framework("pytorch"))

  expect_length(probes, 4L)
  expect_equal(length(ls(cache, all.names = TRUE)), 4L)
})

test_that("unmanaged Python framework health is never cached", {
  cache <- dsFlowerClient:::.client_framework_cache
  dsFlowerClient:::.clear_client_framework_cache()
  withr::defer(dsFlowerClient:::.clear_client_framework_cache())

  probes <- 0L
  local_mocked_bindings(
    .client_python_cmd = function() "/system/python",
    .client_framework_venv_fingerprint = function(...) NULL,
    .framework_modules_available = function(...) {
      probes <<- probes + 1L
      TRUE
    },
    .package = "dsFlowerClient"
  )

  expect_true(dsFlowerClient:::.ensure_client_framework("pytorch"))
  expect_true(dsFlowerClient:::.ensure_client_framework("pytorch"))
  expect_identical(probes, 2L)
  expect_length(ls(cache, all.names = TRUE), 0L)
})

test_that("framework probe failures retry and installation clears cached health", {
  cache <- dsFlowerClient:::.client_framework_cache
  dsFlowerClient:::.clear_client_framework_cache()
  withr::defer(dsFlowerClient:::.clear_client_framework_cache())
  assign("unrelated-ready-runtime", TRUE, envir = cache)

  probes <- 0L
  local_mocked_bindings(
    .client_python_cmd = function() "/managed/python",
    .client_framework_venv_fingerprint = function(...) "venv",
    .framework_modules_available = function(...) {
      probes <<- probes + 1L
      probes > 1L
    },
    .ensure_client_uv = function() {
      expect_length(ls(cache, all.names = TRUE), 0L)
      stop("installation attempted", call. = FALSE)
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    dsFlowerClient:::.ensure_client_framework("pytorch"),
    "installation attempted"
  )
  expect_length(ls(cache, all.names = TRUE), 0L)
  expect_true(dsFlowerClient:::.ensure_client_framework("pytorch"))
  expect_identical(probes, 2L)
  expect_length(ls(cache, all.names = TRUE), 1L)
})

test_that("managed venv recreation invalidates framework health first", {
  cache <- dsFlowerClient:::.client_framework_cache
  root <- withr::local_tempdir()
  dsFlowerClient:::.clear_client_framework_cache()
  withr::defer(dsFlowerClient:::.clear_client_framework_cache())
  assign("ready-runtime", TRUE, envir = cache)

  local_mocked_bindings(
    .client_venv_is_healthy = function() FALSE,
    .client_venv_root = function() root,
    .ensure_client_uv = function() {
      expect_length(ls(cache, all.names = TRUE), 0L)
      stop("provisioning stopped", call. = FALSE)
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    dsFlowerClient:::.ensure_client_venv(),
    "provisioning stopped"
  )
})

test_that("managed venv fingerprint changes with its package inventory", {
  root <- withr::local_tempdir()
  venv <- file.path(root, "venv")
  bindir <- dsFlowerClient:::.venv_bindir(venv)
  site_packages <- if (.Platform$OS.type == "windows") {
    file.path(venv, "Lib", "site-packages")
  } else {
    file.path(venv, "lib", "python3.11", "site-packages")
  }
  dir.create(bindir, recursive = TRUE)
  dir.create(site_packages, recursive = TRUE)
  paths <- file.path(
    bindir,
    vapply(c("python", "flwr", "flower-superlink"),
           dsFlowerClient:::.venv_exe, character(1))
  )
  expect_true(all(file.create(paths)))
  marker <- file.path(venv, ".dsflower_client_ready")
  writeLines("ready", marker, useBytes = TRUE)
  expect_true(file.create(file.path(site_packages, "base.dist-info")))

  local_mocked_bindings(
    .client_venv_path = function() venv,
    .client_venv_marker = function() "ready",
    .package = "dsFlowerClient"
  )

  first <- dsFlowerClient:::.client_framework_venv_fingerprint(paths[[1L]])
  expect_match(first, "^[0-9a-f]{64}$")
  expect_true(file.create(file.path(site_packages, "new.dist-info")))
  second <- dsFlowerClient:::.client_framework_venv_fingerprint(paths[[1L]])
  expect_match(second, "^[0-9a-f]{64}$")
  expect_false(identical(first, second))

  unlink(marker)
  expect_null(
    dsFlowerClient:::.client_framework_venv_fingerprint(paths[[1L]])
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
