# Module: Client-Side Python Environment Management
#
# Uses uv to create a single venv with the tested Flower CLI on the
# researcher's machine.
# Same pattern as server-side packages (dsFlower, dsImaging):
#   1. Ensure uv is available (download if needed)
#   2. uv creates Python venv (downloads Python if needed)
#   3. Use the venv's flwr CLI + flower-superlink
#
# Zero system dependencies. No pre-existing Python installation required.

.DSFLOWER_CLIENT_PYTHON_DEPS <- c(
  "flwr==1.31.0",
  "optuna==4.8.0"
)

.client_python_lock_required <- function() {
  value <- Sys.getenv("DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK", "")
  if (!nzchar(value)) value <- Sys.getenv("DSFLOWER_REQUIRE_PYTHON_LOCK", "")
  if (nzchar(value)) {
    value <- tolower(value)
    if (value %in% c("1", "true", "yes")) return(TRUE)
    if (value %in% c("0", "false", "no")) return(FALSE)
    stop("DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK must be true or false.", call. = FALSE)
  }
  isTRUE(as.logical(getOption("dsflower.client_require_python_lock", FALSE)))
}

.client_python_lock_path <- function(must_exist = FALSE) {
  path <- Sys.getenv("DSFLOWER_CLIENT_PYTHON_LOCK", "")
  if (!nzchar(path)) path <- Sys.getenv("DSFLOWER_PYTHON_LOCK", "")
  if (!nzchar(path)) {
    path <- as.character(getOption("dsflower.client_python_lock", ""))[1]
  }
  if (!nzchar(path)) {
    if (must_exist && .client_python_lock_required()) {
      stop("A hash-locked Python environment is required, but no client ",
           "Python lock is configured.", call. = FALSE)
    }
    return("")
  }
  if (must_exist && (!file.exists(path) || dir.exists(path) || file.access(path, 4L) != 0L)) {
    stop("Configured dsFlowerClient Python lock is not a readable regular file: ",
         path, call. = FALSE)
  }
  normalizePath(path, winslash = "/", mustWork = FALSE)
}

.client_python_version_spec <- function() {
  version <- Sys.getenv("DSFLOWER_PYTHON_VERSION", "")
  if (!nzchar(version)) {
    version <- as.character(getOption("dsflower.client_python_version", "3.11"))[1]
  }
  if (!grepl("^[0-9]+\\.[0-9]+(\\.[0-9]+)?$", version)) {
    stop("DSFLOWER_PYTHON_VERSION must be major.minor or major.minor.patch.",
         call. = FALSE)
  }
  version
}

.client_venv_marker <- function() {
  python_spec <- paste0("python=", .client_python_version_spec())
  lock <- .client_python_lock_path()
  if (!nzchar(lock) && .client_python_lock_required()) return(NA_character_)
  if (nzchar(lock)) {
    if (!file.exists(lock) || dir.exists(lock) || file.access(lock, 4L) != 0L) {
      return(NA_character_)
    }
    return(paste0(python_spec, ";lock-sha256:",
                  digest::digest(file = lock, algo = "sha256")))
  }
  paste(c(python_spec, sort(.DSFLOWER_CLIENT_PYTHON_DEPS)), collapse = ";")
}

.client_uv_bootstrap_config <- function() {
  version <- Sys.getenv("DSFLOWER_UV_VERSION", "")
  if (!nzchar(version)) {
    version <- as.character(getOption("dsflower.client_uv_version", ""))[1]
  }
  sha256 <- Sys.getenv("DSFLOWER_UV_SHA256", "")
  if (!nzchar(sha256)) {
    sha256 <- as.character(getOption("dsflower.client_uv_sha256", ""))[1]
  }
  if (!nzchar(version) || !nzchar(sha256)) {
    stop("uv is not installed and mutable 'latest' bootstrap is disabled. ",
         "Install uv through the operating system, or configure both ",
         "DSFLOWER_UV_VERSION and DSFLOWER_UV_SHA256 for an audited release.",
         call. = FALSE)
  }
  if (!grepl("^[0-9]+\\.[0-9]+\\.[0-9]+([.-][0-9A-Za-z.-]+)?$", version)) {
    stop("DSFLOWER_UV_VERSION is not a valid release tag.", call. = FALSE)
  }
  if (!grepl("^[0-9A-Fa-f]{64}$", sha256)) {
    stop("DSFLOWER_UV_SHA256 must be 64 hexadecimal characters.", call. = FALSE)
  }
  list(version = version, sha256 = tolower(sha256))
}

.dsflower_client_runtime <- new.env(parent = emptyenv())
# Positive availability only; this environment is never written to disk.
.client_framework_cache <- new.env(parent = emptyenv())

.clear_client_framework_cache <- function() {
  cached <- ls(.client_framework_cache, all.names = TRUE)
  if (length(cached)) rm(list = cached, envir = .client_framework_cache)
  invisible(NULL)
}

#' Get the client venv root directory
#' @keywords internal
.client_venv_root <- function() {
  root <- Sys.getenv("DSFLOWER_CLIENT_VENV_ROOT", "")
  if (!nzchar(root)) root <- getOption("dsflower.client_venv_root", "")
  if (!nzchar(root)) root <- file.path(Sys.getenv("HOME", "~"), ".dsflower")
  root
}

#' Get the client venv path
#' @keywords internal
.client_venv_path <- function() {
  file.path(.client_venv_root(), "venv")
}

# uv/virtualenv put executables in bin/ on Unix and Scripts/ on Windows, and
# Windows executables carry a .exe suffix. These two helpers keep every binary
# lookup cross-platform so a Windows client resolves the venv the same way.

#' Platform venv executable directory (bin on Unix, Scripts on Windows)
#' @param venv_path Character; the venv root path.
#' @keywords internal
.venv_bindir <- function(venv_path) {
  file.path(venv_path, if (.Platform$OS.type == "windows") "Scripts" else "bin")
}

#' Platform executable name (adds .exe on Windows)
#' @param name Character; the base executable name.
#' @keywords internal
.venv_exe <- function(name) {
  if (.Platform$OS.type == "windows") paste0(name, ".exe") else name
}

#' Check if the client venv is healthy
#' @return Logical.
#' @keywords internal
.client_venv_is_healthy <- function() {
  venv_path <- .client_venv_path()
  python <- file.path(.venv_bindir(venv_path), .venv_exe("python"))
  if (!file.exists(python)) return(FALSE)
  marker <- file.path(venv_path, ".dsflower_client_ready")
  if (!file.exists(marker)) return(FALSE)
  recorded <- tryCatch(readLines(marker, warn = FALSE), error = function(e) "")
  if (!identical(paste(recorded, collapse = "\n"), .client_venv_marker())) {
    return(FALSE)
  }
  flwr <- file.path(.venv_bindir(venv_path), .venv_exe("flwr"))
  if (!file.exists(flwr)) return(FALSE)
  superlink <- file.path(.venv_bindir(venv_path), .venv_exe("flower-superlink"))
  if (!file.exists(superlink)) return(FALSE)
  TRUE
}

#' Resolve the flwr CLI binary
#' @return Character; absolute path to flwr.
#' @keywords internal
.client_flwr_cmd <- function() {
  venv_path <- .client_venv_path()
  flwr <- file.path(.venv_bindir(venv_path), .venv_exe("flwr"))
  if (file.exists(flwr)) return(flwr)
  path_flwr <- Sys.which("flwr")
  if (nzchar(path_flwr)) return(path_flwr)
  stop("flwr CLI not found. Install dsFlowerClient with configure support ",
       "or run: pip install 'flwr==1.31.0'", call. = FALSE)
}

#' Resolve the flower-superlink binary
#' @return Character; absolute path to flower-superlink.
#' @keywords internal
.client_superlink_cmd <- function() {
  venv_path <- .client_venv_path()
  superlink <- file.path(.venv_bindir(venv_path), .venv_exe("flower-superlink"))
  if (file.exists(superlink)) return(superlink)
  path_sl <- Sys.which("flower-superlink")
  if (nzchar(path_sl)) return(path_sl)
  stop("flower-superlink not found. Install dsFlowerClient with configure ",
       "support or run: pip install 'flwr==1.31.0'", call. = FALSE)
}

#' Resolve the Python binary from the client venv
#' @return Character; absolute path to python.
#' @keywords internal
.client_python_cmd <- function() {
  venv_path <- .client_venv_path()
  python <- file.path(.venv_bindir(venv_path), .venv_exe("python"))
  if (file.exists(python)) return(python)
  path_py <- Sys.which("python3")
  if (nzchar(path_py)) return(path_py)
  path_py <- Sys.which("python")
  if (nzchar(path_py)) return(path_py)
  stop("Python not found. Install dsFlowerClient with configure support.",
       call. = FALSE)
}

#' Build environment variables for launching venv processes
#' @return Named character vector suitable for processx env parameter.
#' @keywords internal
.client_venv_env <- function(extra = NULL) {
  venv_path <- .client_venv_path()
  venv_bin <- .venv_bindir(venv_path)
  current_path <- Sys.getenv("PATH", "")

  env <- c("current",
    VIRTUAL_ENV = venv_path,
    PATH = paste0(venv_bin, .Platform$path.sep, current_path))

  if (!is.null(extra)) env <- c(env, extra)
  env
}

# --- On-demand framework dependency install ---

.FRAMEWORK_CLIENT_DEPS <- list(
  pytorch = c("torch>=2.0.0,<3.0.0", "numpy>=1.21.0", "pandas>=1.3.0",
              "pyarrow>=10.0.0", "opacus>=1.4.0,<2.0.0",
              "cryptography>=42.0.0"),
  pytorch_vision = c("torch>=2.0.0,<3.0.0", "torchvision>=0.15.0,<1.0.0", "Pillow>=9.0.0",
                     "numpy>=1.21.0", "pandas>=1.3.0", "pyarrow>=10.0.0",
                     "opacus>=1.4.0,<2.0.0", "nibabel>=5.0.0", "pydicom>=2.4.0",
                     "pynrrd>=1.0.0", "SimpleITK>=2.2.0", "monai>=1.3.0",
                     "cryptography>=42.0.0")
)

.FRAMEWORK_CHECK_MODULE <- list(
  pytorch = c("torch", "numpy", "pandas", "pyarrow", "opacus", "cryptography"),
  pytorch_vision = c(
    "torch", "torchvision", "PIL", "numpy", "pandas", "pyarrow", "opacus",
    "nibabel", "pydicom", "nrrd", "SimpleITK", "monai", "cryptography"
  )
)

.framework_modules_available <- function(python, modules) {
  import_code <- paste(paste("import", modules), collapse = "; ")
  rc <- suppressWarnings(
    system2(python, c("-c", shQuote(import_code)),
            stdout = FALSE, stderr = FALSE)
  )
  identical(rc, 0L)
}

.client_framework_venv_fingerprint <- function(python) {
  venv_path <- .client_venv_path()
  expected_python <- file.path(
    .venv_bindir(venv_path), .venv_exe("python"))
  if (!file.exists(python) || !file.exists(expected_python)) return(NULL)

  normalize <- function(path) {
    value <- normalizePath(path, winslash = "/", mustWork = FALSE)
    if (.Platform$OS.type == "windows") tolower(value) else value
  }
  python_path <- normalize(python)
  if (!identical(python_path, normalize(expected_python)) ||
      !.client_venv_is_healthy()) {
    return(NULL)
  }

  marker <- file.path(venv_path, ".dsflower_client_ready")
  marker_value <- tryCatch(
    paste(readLines(marker, warn = FALSE), collapse = "\n"),
    error = function(e) NULL)
  if (is.null(marker_value) ||
      !identical(marker_value, .client_venv_marker())) {
    return(NULL)
  }

  site_packages <- if (.Platform$OS.type == "windows") {
    file.path(venv_path, "Lib", "site-packages")
  } else {
    c(Sys.glob(file.path(venv_path, "lib", "python*", "site-packages")),
      Sys.glob(file.path(venv_path, "lib64", "python*", "site-packages")))
  }
  site_packages <- sort(unique(site_packages[dir.exists(site_packages)]),
                        method = "radix")
  if (!length(site_packages)) return(NULL)
  # Track managed installs, removals and top-level package replacements. Manual
  # in-place edits below an installed package directory are outside the managed
  # venv lifecycle; the actual Python helper/runtime remains the execution check.
  inventory <- tryCatch(
    unlist(lapply(site_packages, function(path) {
      suppressWarnings(list.files(
        path, all.files = TRUE, no.. = TRUE, full.names = TRUE))
    }), use.names = FALSE),
    error = function(e) NULL)
  if (is.null(inventory) || !length(inventory)) return(NULL)
  observed <- sort(unique(c(expected_python, marker, site_packages, inventory)),
                   method = "radix")
  info <- file.info(observed)
  if (anyNA(info$isdir) || anyNA(info$size) || anyNA(info$mtime) ||
      anyNA(info$ctime)) return(NULL)
  observed <- gsub("\\", "/", observed, fixed = TRUE)
  if (.Platform$OS.type == "windows") observed <- tolower(observed)
  python_sha256 <- tryCatch(
    digest::digest(file = expected_python, algo = "sha256"),
    error = function(e) NULL)
  if (is.null(python_sha256)) return(NULL)

  digest::digest(list(
    python = python_path,
    python_sha256 = python_sha256,
    marker = marker_value,
    inventory = data.frame(
      path = observed,
      isdir = unname(info$isdir),
      size = unname(info$size),
      mtime = as.numeric(info$mtime),
      ctime = as.numeric(info$ctime),
      stringsAsFactors = FALSE)
  ), algo = "sha256")
}

#' Ensure ML framework dependencies are installed in the client venv
#'
#' Checks if the framework's Python packages are available.
#' Installs them on-demand if missing. Called automatically before
#' training and prediction.
#'
#' @param framework Character; "pytorch" or "pytorch_vision".
#' @return Invisible TRUE.
#' @keywords internal
.ensure_client_framework <- function(framework) {
  check_mod <- .FRAMEWORK_CHECK_MODULE[[framework]]
  if (is.null(check_mod)) {
    stop("Unsupported client framework: ", framework, ".", call. = FALSE)
  }

  deps <- .FRAMEWORK_CLIENT_DEPS[[framework]]
  if (is.null(deps)) {
    stop("No dependency contract for client framework: ", framework, ".",
         call. = FALSE)
  }

  python <- .client_python_cmd()
  python_path <- normalizePath(python, winslash = "/", mustWork = FALSE)
  if (.Platform$OS.type == "windows") python_path <- tolower(python_path)
  cache_key <- function(fingerprint) digest::digest(list(
    framework = framework,
    python = python_path,
    modules = check_mod,
    dependencies = deps,
    venv = fingerprint
  ), algo = "sha256")
  fingerprint <- .client_framework_venv_fingerprint(python)
  if (!is.null(fingerprint) &&
      exists(cache_key(fingerprint), envir = .client_framework_cache,
             inherits = FALSE)) {
    return(invisible(TRUE))
  }

  if (.framework_modules_available(python, check_mod)) {
    fingerprint <- .client_framework_venv_fingerprint(python)
    if (!is.null(fingerprint)) {
      assign(cache_key(fingerprint), TRUE, envir = .client_framework_cache)
    }
    return(invisible(TRUE))
  }

  # Installing into the shared venv can alter either framework contract.
  .clear_client_framework_cache()
  message("dsFlowerClient: installing ", framework, " dependencies...")
  uv <- .ensure_client_uv()
  lock <- .client_python_lock_path(must_exist = TRUE)
  install_spec <- if (nzchar(lock)) c("--require-hashes", "-r", lock) else deps
  result <- processx::run(
    command = uv,
    args = c("pip", "install", "--python", python, "--quiet", install_spec),
    error_on_status = FALSE,
    timeout = 600
  )
  if (result$status != 0L) {
    stop("Failed to install ", framework, " dependencies:\n",
         result$stderr, call. = FALSE)
  }
  if (!.framework_modules_available(python, check_mod)) {
    stop("Python requirements did not provide all required modules: ",
         paste(check_mod, collapse = ", "), ".", call. = FALSE)
  }
  fingerprint <- .client_framework_venv_fingerprint(python)
  if (!is.null(fingerprint)) {
    assign(cache_key(fingerprint), TRUE, envir = .client_framework_cache)
  }
  message("  ", framework, " ready.")
  invisible(TRUE)
}

#' Ensure the client Python venv exists and is healthy
#'
#' Downloads uv if needed, creates venv with Python 3.11, installs flwr.
#' Idempotent: skips if venv already healthy.
#'
#' @param timeout_secs Numeric; max seconds for install (default 600).
#' @return Invisible TRUE.
#' @keywords internal
.ensure_client_venv <- function(timeout_secs = 600) {
  if (.client_venv_is_healthy()) return(invisible(TRUE))

  .clear_client_framework_cache()

  root <- .client_venv_root()
  dir.create(root, recursive = TRUE, showWarnings = FALSE)

  uv <- .ensure_client_uv()
  venv_path <- .client_venv_path()

  message("dsFlowerClient: creating Python environment...")
  message("  This may take a few minutes on first use.")

  if (dir.exists(venv_path)) unlink(venv_path, recursive = TRUE)

  rc <- system2(uv, c("venv", "--python", .client_python_version_spec(),
                      "--quiet", venv_path),
                stdout = "", stderr = "")
  if (rc != 0L)
    stop("Failed to create venv at ", venv_path, call. = FALSE)

  venv_python <- file.path(.venv_bindir(venv_path), .venv_exe("python"))
  deps <- .DSFLOWER_CLIENT_PYTHON_DEPS
  lock <- .client_python_lock_path(must_exist = TRUE)
  install_spec <- if (nzchar(lock)) c("--require-hashes", "-r", lock) else deps
  if (nzchar(lock)) {
    message("  Installing administrator hash-locked Python requirements")
  } else {
    message("  Installing: ", paste(deps, collapse = ", "))
  }

  result <- processx::run(
    command = uv,
    args = c("pip", "install", "--python", venv_python, "--quiet", install_spec),
    error_on_status = FALSE,
    timeout = timeout_secs
  )

  if (result$status != 0L) {
    unlink(venv_path, recursive = TRUE)
    stop("pip install failed:\n", result$stderr, call. = FALSE)
  }

  flwr <- file.path(.venv_bindir(venv_path), .venv_exe("flwr"))
  superlink <- file.path(.venv_bindir(venv_path), .venv_exe("flower-superlink"))
  if (!file.exists(flwr) || !file.exists(superlink)) {
    unlink(venv_path, recursive = TRUE)
    stop("flwr/flower-superlink not found after install.", call. = FALSE)
  }

  writeLines(.client_venv_marker(),
             file.path(venv_path, ".dsflower_client_ready"), useBytes = TRUE)
  message("  Python environment ready at ", venv_path)
  invisible(TRUE)
}

#' Ensure uv is available (find or download)
#' @return Character; path to uv binary.
#' @keywords internal
.ensure_client_uv <- function() {
  cached <- .dsflower_client_runtime$uv_path
  if (!is.null(cached) && file.exists(cached)) return(cached)

  uv <- Sys.which("uv")  # finds uv / uv.exe on PATH (cross-platform)
  if (nzchar(uv)) { .dsflower_client_runtime$uv_path <- uv; return(uv) }

  home <- Sys.getenv("HOME", "~")
  for (p in c(file.path(home, ".local", "bin", .venv_exe("uv")),
              file.path(home, ".cargo", "bin", .venv_exe("uv")),
              file.path("/usr/local/bin", .venv_exe("uv")))) {
    if (file.exists(p)) { .dsflower_client_runtime$uv_path <- p; return(p) }
  }

  tools_dir <- file.path(.client_venv_root(), ".tools")
  dir.create(tools_dir, recursive = TRUE, showWarnings = FALSE)
  uv_path <- file.path(tools_dir, .venv_exe("uv"))
  if (file.exists(uv_path)) {
    .dsflower_client_runtime$uv_path <- uv_path
    return(uv_path)
  }

  bootstrap <- .client_uv_bootstrap_config()
  message("dsFlowerClient: downloading pinned uv ", bootstrap$version, "...")
  sysname <- tolower(Sys.info()[["sysname"]])
  machine <- tolower(Sys.info()[["machine"]])
  is_win <- identical(sysname, "windows")
  os <- switch(sysname,
    darwin  = "apple-darwin",
    linux   = "unknown-linux-gnu",
    windows = "pc-windows-msvc",
    stop("Unsupported OS: ", sysname,
         ". Install uv: https://docs.astral.sh/uv/", call. = FALSE))
  arch <- switch(machine,
    "x86_64"  = "x86_64", "amd64" = "x86_64", "x86-64" = "x86_64",
    "aarch64" = "aarch64", "arm64" = "aarch64",
    stop("Unsupported arch: ", machine,
         ". Install uv: https://docs.astral.sh/uv/", call. = FALSE))

  # Windows ships uv as a .zip of uv.exe; Unix as a .tar.gz of uv.
  ext <- if (is_win) ".zip" else ".tar.gz"
  triple <- paste(arch, os, sep = "-")
  url <- paste0("https://github.com/astral-sh/uv/releases/download/",
                bootstrap$version, "/uv-", triple, ext)
  tmp <- tempfile(fileext = ext)
  tmp_dir <- tempfile()
  on.exit({ unlink(tmp); unlink(tmp_dir, recursive = TRUE) }, add = TRUE)

  rc <- tryCatch(utils::download.file(url, tmp, mode = "wb", quiet = TRUE),
                  error = function(e) 1L)
  if (!identical(rc, 0L))
    stop("Failed to download pinned uv. Install manually: https://docs.astral.sh/uv/",
         call. = FALSE)

  actual <- digest::digest(file = tmp, algo = "sha256")
  if (!identical(tolower(actual), bootstrap$sha256)) {
    stop("Downloaded uv archive SHA-256 mismatch; refusing to extract it.",
         call. = FALSE)
  }

  dir.create(tmp_dir, showWarnings = FALSE)
  member <- paste0("uv-", triple, "/", if (is_win) "uv.exe" else "uv")
  entries <- if (is_win) utils::unzip(tmp, list = TRUE)$Name else utils::untar(tmp, list = TRUE)
  if (!member %in% entries) stop("uv binary not found in verified archive.", call. = FALSE)
  if (is_win) {
    utils::unzip(tmp, files = member, exdir = tmp_dir)
  } else {
    utils::untar(tmp, files = member, exdir = tmp_dir)
  }
  source <- file.path(tmp_dir, member)
  install_tmp <- tempfile(pattern = ".uv-", tmpdir = tools_dir)
  if (!file.copy(source, install_tmp, overwrite = TRUE)) {
    stop("Could not stage verified uv binary.", call. = FALSE)
  }
  if (.Platform$OS.type != "windows") Sys.chmod(install_tmp, "0755")
  if (!file.rename(install_tmp, uv_path)) {
    unlink(install_tmp)
    stop("Could not atomically install verified uv binary.", call. = FALSE)
  }
  message("dsFlowerClient: uv installed at ", uv_path)
  .dsflower_client_runtime$uv_path <- uv_path
  uv_path
}
