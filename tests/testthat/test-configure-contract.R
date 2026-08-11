test_that("configure skip gate is validated before host Python setup", {
  root <- normalizePath(file.path(testthat::test_path(), "..", ".."),
                        winslash = "/", mustWork = TRUE)
  configure_path <- file.path(root, "configure")
  testthat::skip_if_not(file.exists(configure_path),
                        "source configure script is not installed")
  lines <- readLines(configure_path, warn = FALSE)
  gate <- grep("DSFLOWER_SKIP_PYTHON_SETUP:-", lines, fixed = TRUE)[[1L]]
  venv <- grep("^VENV_ROOT=", lines)[[1L]]
  bootstrap <- grep("^ensure_uv \\|\\|", lines)[[1L]]
  expect_lt(gate, venv)
  expect_lt(gate, bootstrap)

  output <- system2(
    "sh", shQuote(configure_path),
    env = c("DSFLOWER_SKIP_PYTHON_SETUP=1", "HOME="),
    stdout = TRUE, stderr = TRUE)
  expect_match(paste(output, collapse = "\n"), "explicitly skipped")
  expect_false(any(grepl("environment|uv|Venv", output, ignore.case = TRUE)))

  invalid <- suppressWarnings(system2(
    "sh", shQuote(configure_path),
    env = c("DSFLOWER_SKIP_PYTHON_SETUP=maybe", "HOME="),
    stdout = TRUE, stderr = TRUE))
  expect_match(paste(invalid, collapse = "\n"), "must be true or false")
  expect_identical(attr(invalid, "status"), 1L)
})

test_that("a required client Python lock fails closed", {
  root <- normalizePath(file.path(testthat::test_path(), "..", ".."),
                        winslash = "/", mustWork = TRUE)
  configure_path <- file.path(root, "configure")
  testthat::skip_if_not(file.exists(configure_path),
                        "source configure script is not installed")

  missing <- suppressWarnings(system2(
    "sh", shQuote(configure_path),
    env = c(
      "DSFLOWER_SKIP_PYTHON_SETUP=0",
      "DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK=true",
      "DSFLOWER_CLIENT_PYTHON_LOCK=",
      "DSFLOWER_PYTHON_LOCK="
    ), stdout = TRUE, stderr = TRUE))
  expect_identical(attr(missing, "status"), 1L)
  expect_match(paste(missing, collapse = "\n"),
               "CLIENT_REQUIRE_PYTHON_LOCK is enabled")

  unreadable <- suppressWarnings(system2(
    "sh", shQuote(configure_path),
    env = c(
      "DSFLOWER_SKIP_PYTHON_SETUP=0",
      "DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK=true",
      paste0("DSFLOWER_CLIENT_PYTHON_LOCK=", shQuote(root)),
      "DSFLOWER_PYTHON_LOCK="
    ), stdout = TRUE, stderr = TRUE))
  expect_identical(attr(unreadable, "status"), 1L)
  expect_match(paste(unreadable, collapse = "\n"),
               "not a readable regular file")

  invalid <- suppressWarnings(system2(
    "sh", shQuote(configure_path),
    env = c(
      "DSFLOWER_SKIP_PYTHON_SETUP=0",
      "DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK=maybe"
    ), stdout = TRUE, stderr = TRUE))
  expect_identical(attr(invalid, "status"), 1L)
  expect_match(paste(invalid, collapse = "\n"), "must be true or false")
})

.client_fake_uv_lock_fixture <- function(root, mode) {
  dir.create(root, recursive = TRUE)
  root <- normalizePath(root, winslash = "/", mustWork = TRUE)
  bin <- file.path(root, "bin")
  home <- file.path(root, "home")
  venv_root <- file.path(root, "venv-root")
  python <- file.path(root, "fake-python")
  uv <- file.path(bin, "uv")
  log <- file.path(root, "uv.log")
  lock <- file.path(root, "requirements.lock")
  dir.create(bin, recursive = TRUE)
  dir.create(home, recursive = TRUE)
  writeLines(c(
    "#!/bin/sh",
    'case "$1" in',
    "  --version) printf '%s\\n' 'Python 3.11.0' ;;",
    "  -c) exit 0 ;;",
    "  -) printf '%064d\\n' 0 ;;",
    "esac",
    "exit 0"
  ), python)
  writeLines(c(
    "#!/bin/sh",
    'printf \'%s\\n\' "$*" >> "${FAKE_UV_LOG}"',
    'if [ "$1" = python ] && [ "$2" = find ]; then',
    '  printf \'%s\\n\' "${FAKE_PYTHON}"',
    "  exit 0",
    "fi",
    'if [ "$1" = venv ]; then',
    '  if [ "${FAKE_UV_MODE}" = venv-fail ]; then exit 41; fi',
    '  for target in "$@"; do :; done',
    '  mkdir -p "${target}/bin" "${target}/Scripts" || exit 42',
    '  cp "${FAKE_PYTHON}" "${target}/bin/python" || exit 42',
    '  cp "${FAKE_PYTHON}" "${target}/Scripts/python.exe" || exit 42',
    '  chmod 0755 "${target}/bin/python" "${target}/Scripts/python.exe" || exit 42',
    "  exit 0",
    "fi",
    'if [ "$1" = pip ]; then',
    "  printf '%s\\n' 'fake pip failure sentinel'",
    "  exit 43",
    "fi",
    "exit 44"
  ), uv)
  Sys.chmod(c(python, uv), mode = "0755")
  writeLines(
    paste0("flwr==1.31.0 --hash=sha256:", strrep("0", 64L)),
    lock
  )
  file.create(log)
  list(
    home = home, venv_root = venv_root, python = python, bin = bin,
    log = log, lock = lock, mode = mode
  )
}

test_that("required client locks propagate uv venv and pip failures", {
  root <- normalizePath(file.path(testthat::test_path(), "..", ".."),
                        winslash = "/", mustWork = TRUE)
  configure_path <- file.path(root, "configure")
  testthat::skip_if_not(file.exists(configure_path),
                        "source configure script is not installed")
  fixtures <- withr::local_tempdir()

  for (mode in c("venv-fail", "pip-fail")) {
    fixture <- .client_fake_uv_lock_fixture(
      file.path(fixtures, mode), mode
    )
    if (identical(mode, "pip-fail")) {
      stale_venv <- file.path(fixture$venv_root, "venv")
      dir.create(stale_venv, recursive = TRUE)
      writeLines("stale-lock-marker", file.path(
        stale_venv, ".dsflower_client_ready"
      ))
    }
    fixture_env <- c(
      "DSFLOWER_SKIP_PYTHON_SETUP=0",
      paste0("DSFLOWER_CLIENT_PYTHON_LOCK=", fixture$lock),
      "DSFLOWER_PYTHON_LOCK=",
      paste0("DSFLOWER_CLIENT_VENV_ROOT=", fixture$venv_root),
      paste0("HOME=", fixture$home),
      paste0("FAKE_PYTHON=", fixture$python),
      paste0("FAKE_UV_LOG=", fixture$log),
      paste0("FAKE_UV_MODE=", mode),
      paste0("PATH=", fixture$bin, .Platform$path.sep,
             Sys.getenv("PATH"))
    )
    output <- suppressWarnings(system2(
      "sh", shQuote(configure_path),
      env = c(
        "DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK=true",
        fixture_env
      ), stdout = TRUE, stderr = TRUE
    ))
    expect_identical(attr(output, "status"), 1L, info = mode)
    uv_log <- readLines(fixture$log, warn = FALSE)
    expect_true(any(grepl(file.path(fixture$venv_root, "venv"),
                          uv_log, fixed = TRUE)), info = mode)
    if (identical(mode, "pip-fail")) {
      expect_true(any(grepl("--require-hashes", uv_log, fixed = TRUE)))
      expect_match(paste(output, collapse = "\n"),
                   "fake pip failure sentinel")
    }
    expect_false(file.exists(file.path(
      fixture$venv_root, "venv", ".dsflower_client_ready"
    )), info = mode)

    optional <- suppressWarnings(system2(
      "sh", shQuote(configure_path),
      env = c("DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK=false", fixture_env),
      stdout = TRUE, stderr = TRUE
    ))
    expect_null(attr(optional, "status"), info = mode)
  }
})

test_that("configure provisions the Windows venv layout from one contract", {
  root <- normalizePath(file.path(testthat::test_path(), "..", ".."),
                        winslash = "/", mustWork = TRUE)
  testthat::skip_if_not(file.exists(file.path(root, "configure")),
                        "source configure script is not installed")
  configure <- paste(
    readLines(file.path(root, "configure"), warn = FALSE), collapse = "\n")
  configure_win <- readLines(file.path(root, "configure.win"), warn = FALSE)
  expect_true(any(grepl("exec sh ./configure", configure_win, fixed = TRUE)))
  expect_match(configure, "pc-windows-msvc", fixed = TRUE)
  expect_match(configure, "venv_executable", fixed = TRUE)
  expect_false(grepl("VENV_PATH}/bin/(python|flwr|flower-superlink)",
                     configure))
})
