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
})

test_that(".client_venv_env uses the platform PATH separator", {
  src <- paste(deparse(body(dsFlowerClient:::.client_venv_env)), collapse = "\n")
  expect_match(src, "path\\.sep")         # ; on Windows, : on Unix
})
