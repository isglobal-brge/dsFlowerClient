# Tests for cross-platform venv resolution in R/python_env.R.
# The client provisions a uv venv on the researcher's machine; these guard that
# the binary/path logic is not Unix-only, so a Windows client provisions and
# resolves flwr / flower-superlink the same way (bin/ -> Scripts/, +.exe, ; PATH).

test_that("venv layout helpers resolve Unix paths on this platform", {
  skip_on_os("windows")
  expect_equal(dsFlowerClient:::.venv_bindir("/x"), "/x/bin")
  expect_equal(dsFlowerClient:::.venv_exe("python"), "python")
  expect_equal(dsFlowerClient:::.venv_exe("flower-superlink"), "flower-superlink")
})

test_that("venv layout helpers branch on .Platform$OS.type (not hardcoded)", {
  # Structural guard: a future edit must not re-hardcode bin/ or drop .exe.
  expect_match(paste(deparse(body(dsFlowerClient:::.venv_bindir)), collapse = " "), "OS.type")
  expect_match(paste(deparse(body(dsFlowerClient:::.venv_bindir)), collapse = " "), "Scripts")
  expect_match(paste(deparse(body(dsFlowerClient:::.venv_exe)), collapse = " "), "OS.type")
})

test_that(".ensure_client_uv knows how to fetch uv on Windows", {
  src <- paste(deparse(body(dsFlowerClient:::.ensure_client_uv)), collapse = "\n")
  expect_match(src, "pc-windows-msvc")   # the Windows uv release triple
  expect_match(src, "windows")           # OS switch handles windows (not stop())
  expect_match(src, "unzip")             # Windows ships uv as .zip, not .tar.gz
  expect_false(grepl("releases/latest", src, fixed = TRUE))
})

test_that("uv bootstrap requires an immutable release and archive digest", {
  withr::local_envvar(c(DSFLOWER_UV_VERSION = "", DSFLOWER_UV_SHA256 = ""))
  withr::local_options(list(
    dsflower.client_uv_version = "",
    dsflower.client_uv_sha256 = ""
  ))
  expect_error(dsFlowerClient:::.client_uv_bootstrap_config(),
               "mutable 'latest'.*disabled")

  withr::local_envvar(c(
    DSFLOWER_UV_VERSION = "0.11.14",
    DSFLOWER_UV_SHA256 = strrep("B", 64)
  ))
  expect_equal(
    dsFlowerClient:::.client_uv_bootstrap_config(),
    list(version = "0.11.14", sha256 = strrep("b", 64))
  )

  withr::local_envvar(DSFLOWER_UV_VERSION = "latest")
  expect_error(dsFlowerClient:::.client_uv_bootstrap_config(),
               "valid release tag")
})

test_that("hash-locked requirements bind the client venv marker", {
  lock <- withr::local_tempfile()
  writeLines("example==1.0 --hash=sha256:aaaaaaaa", lock)
  withr::local_envvar(c(
    DSFLOWER_CLIENT_PYTHON_LOCK = lock,
    DSFLOWER_PYTHON_LOCK = "",
    DSFLOWER_PYTHON_VERSION = "",
    DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK = "",
    DSFLOWER_REQUIRE_PYTHON_LOCK = ""
  ))
  withr::local_options(dsflower.client_python_version = "3.11")

  expected <- paste0("python=3.11;lock-sha256:",
                     digest::digest(file = lock, algo = "sha256"))
  expect_equal(dsFlowerClient:::.client_venv_marker(), expected)
})

test_that("strict client provisioning fails closed without a lock", {
  withr::local_envvar(c(
    DSFLOWER_CLIENT_PYTHON_LOCK = "",
    DSFLOWER_PYTHON_LOCK = "",
    DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK = "true",
    DSFLOWER_REQUIRE_PYTHON_LOCK = ""
  ))
  withr::local_options(dsflower.client_python_lock = "")
  expect_error(dsFlowerClient:::.client_python_lock_path(must_exist = TRUE),
               "hash-locked Python environment is required")
  expect_true(is.na(dsFlowerClient:::.client_venv_marker()))
})

test_that(".client_venv_env uses the platform PATH separator", {
  src <- paste(deparse(body(dsFlowerClient:::.client_venv_env)), collapse = "\n")
  expect_match(src, "path\\.sep")         # ; on Windows, : on Unix
})
