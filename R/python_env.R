# Module: Client-Side Python Environment Management
#
# Uses uv to create a single venv with flwr[app] on the researcher's machine.
# Same pattern as server-side packages (dsFlower, dsRadiomics, dsImaging):
#   1. Ensure uv is available (download if needed)
#   2. uv creates Python venv (downloads Python if needed)
#   3. Use the venv's flwr CLI + flower-superlink
#
# Zero system dependencies. No pre-existing Python installation required.

.DSFLOWER_CLIENT_PYTHON_DEPS <- c("flwr[app]>=1.13.0")

.dsflower_client_runtime <- new.env(parent = emptyenv())

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

#' Check if the client venv is healthy
#' @return Logical.
#' @keywords internal
.client_venv_is_healthy <- function() {
  venv_path <- .client_venv_path()
  python <- file.path(venv_path, "bin", "python")
  if (!file.exists(python)) return(FALSE)
  marker <- file.path(venv_path, ".dsflower_client_ready")
  if (!file.exists(marker)) return(FALSE)
  flwr <- file.path(venv_path, "bin", "flwr")
  if (!file.exists(flwr)) return(FALSE)
  superlink <- file.path(venv_path, "bin", "flower-superlink")
  if (!file.exists(superlink)) return(FALSE)
  TRUE
}

#' Resolve the flwr CLI binary
#' @return Character; absolute path to flwr.
#' @keywords internal
.client_flwr_cmd <- function() {
  venv_path <- .client_venv_path()
  flwr <- file.path(venv_path, "bin", "flwr")
  if (file.exists(flwr)) return(flwr)
  path_flwr <- Sys.which("flwr")
  if (nzchar(path_flwr)) return(path_flwr)
  stop("flwr CLI not found. Install dsFlowerClient with configure support ",
       "or run: pip install 'flwr[app]>=1.13.0'", call. = FALSE)
}

#' Resolve the flower-superlink binary
#' @return Character; absolute path to flower-superlink.
#' @keywords internal
.client_superlink_cmd <- function() {
  venv_path <- .client_venv_path()
  superlink <- file.path(venv_path, "bin", "flower-superlink")
  if (file.exists(superlink)) return(superlink)
  path_sl <- Sys.which("flower-superlink")
  if (nzchar(path_sl)) return(path_sl)
  stop("flower-superlink not found. Install dsFlowerClient with configure ",
       "support or run: pip install 'flwr[app]>=1.13.0'", call. = FALSE)
}

#' Resolve the Python binary from the client venv
#' @return Character; absolute path to python.
#' @keywords internal
.client_python_cmd <- function() {
  venv_path <- .client_venv_path()
  python <- file.path(venv_path, "bin", "python")
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
  venv_bin <- file.path(venv_path, "bin")
  current_path <- Sys.getenv("PATH", "")

  env <- c("current",
    VIRTUAL_ENV = venv_path,
    PATH = paste0(venv_bin, ":", current_path))

  if (!is.null(extra)) env <- c(env, extra)
  env
}

# --- On-demand framework dependency install ---

.FRAMEWORK_CLIENT_DEPS <- list(
  sklearn = c("scikit-learn>=1.0.0", "numpy>=1.21.0", "pandas>=1.3.0",
              "pyarrow>=10.0.0", "joblib"),
  pytorch = c("torch>=2.0.0", "numpy>=1.21.0", "pandas>=1.3.0",
              "pyarrow>=10.0.0"),
  pytorch_vision = c("torch>=2.0.0", "torchvision>=0.15.0", "Pillow>=9.0.0",
                     "numpy>=1.21.0", "pandas>=1.3.0", "pyarrow>=10.0.0"),
  xgboost = c("xgboost>=1.7.0", "numpy>=1.21.0", "pandas>=1.3.0",
              "pyarrow>=10.0.0")
)

.FRAMEWORK_CHECK_MODULE <- list(
  sklearn = "sklearn", pytorch = "torch",
  pytorch_vision = "torchvision", xgboost = "xgboost"
)

#' Ensure ML framework dependencies are installed in the client venv
#'
#' Checks if the framework's Python packages are available.
#' Installs them on-demand if missing. Called automatically before
#' training and prediction.
#'
#' @param framework Character; "sklearn", "pytorch", "pytorch_vision", or "xgboost".
#' @return Invisible TRUE.
#' @keywords internal
.ensure_client_framework <- function(framework) {
  check_mod <- .FRAMEWORK_CHECK_MODULE[[framework]]
  if (is.null(check_mod)) return(invisible(TRUE))

  python <- .client_python_cmd()
  rc <- suppressWarnings(
    system2(python, c("-c", shQuote(paste0("import ", check_mod))),
            stdout = FALSE, stderr = FALSE)
  )
  if (rc == 0L) return(invisible(TRUE))

  deps <- .FRAMEWORK_CLIENT_DEPS[[framework]]
  if (is.null(deps)) return(invisible(TRUE))

  message("dsFlowerClient: installing ", framework, " dependencies...")
  uv <- .ensure_client_uv()
  result <- processx::run(
    command = uv,
    args = c("pip", "install", "--python", python, "--quiet", deps),
    error_on_status = FALSE,
    timeout = 600
  )
  if (result$status != 0L) {
    stop("Failed to install ", framework, " dependencies:\n",
         result$stderr, call. = FALSE)
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

  root <- .client_venv_root()
  dir.create(root, recursive = TRUE, showWarnings = FALSE)

  uv <- .ensure_client_uv()
  venv_path <- .client_venv_path()

  message("dsFlowerClient: creating Python environment...")
  message("  This may take a few minutes on first use.")

  if (dir.exists(venv_path)) unlink(venv_path, recursive = TRUE)

  rc <- system2(uv, c("venv", "--python", "3.11", "--quiet", venv_path),
                stdout = "", stderr = "")
  if (rc != 0L)
    stop("Failed to create venv at ", venv_path, call. = FALSE)

  venv_python <- file.path(venv_path, "bin", "python")
  deps <- .DSFLOWER_CLIENT_PYTHON_DEPS
  message("  Installing: ", paste(deps, collapse = ", "))

  result <- processx::run(
    command = uv,
    args = c("pip", "install", "--python", venv_python, "--quiet", deps),
    error_on_status = FALSE,
    timeout = timeout_secs
  )

  if (result$status != 0L) {
    unlink(venv_path, recursive = TRUE)
    stop("pip install failed:\n", result$stderr, call. = FALSE)
  }

  flwr <- file.path(venv_path, "bin", "flwr")
  superlink <- file.path(venv_path, "bin", "flower-superlink")
  if (!file.exists(flwr) || !file.exists(superlink)) {
    unlink(venv_path, recursive = TRUE)
    stop("flwr/flower-superlink not found after install.", call. = FALSE)
  }

  dep_hash <- digest::digest(paste(sort(deps), collapse = "\n"),
                             algo = "sha256", serialize = FALSE)
  writeLines(dep_hash, file.path(venv_path, ".dsflower_client_ready"))
  message("  Python environment ready at ", venv_path)
  invisible(TRUE)
}

#' Ensure uv is available (find or download)
#' @return Character; path to uv binary.
#' @keywords internal
.ensure_client_uv <- function() {
  cached <- .dsflower_client_runtime$uv_path
  if (!is.null(cached) && file.exists(cached)) return(cached)

  uv <- Sys.which("uv")
  if (nzchar(uv)) { .dsflower_client_runtime$uv_path <- uv; return(uv) }

  home <- Sys.getenv("HOME", "~")
  for (p in c(file.path(home, ".local", "bin", "uv"),
              file.path(home, ".cargo", "bin", "uv"),
              "/usr/local/bin/uv")) {
    if (file.exists(p)) { .dsflower_client_runtime$uv_path <- p; return(p) }
  }

  tools_dir <- file.path(.client_venv_root(), ".tools")
  dir.create(tools_dir, recursive = TRUE, showWarnings = FALSE)
  uv_path <- file.path(tools_dir, "uv")
  if (file.exists(uv_path)) {
    .dsflower_client_runtime$uv_path <- uv_path
    return(uv_path)
  }

  message("dsFlowerClient: downloading uv...")
  sysname <- tolower(Sys.info()[["sysname"]])
  machine <- Sys.info()[["machine"]]
  os <- switch(sysname,
    darwin = "apple-darwin", linux = "unknown-linux-gnu",
    stop("Unsupported OS: ", sysname,
         ". Install uv: https://docs.astral.sh/uv/", call. = FALSE))
  arch <- switch(machine,
    x86_64 = "x86_64", amd64 = "x86_64",
    aarch64 = "aarch64", arm64 = "aarch64",
    stop("Unsupported arch: ", machine, call. = FALSE))

  url <- paste0("https://github.com/astral-sh/uv/releases/latest/download/uv-",
                arch, "-", os, ".tar.gz")
  tmp <- tempfile(fileext = ".tar.gz")
  tmp_dir <- tempfile()
  on.exit({ unlink(tmp); unlink(tmp_dir, recursive = TRUE) }, add = TRUE)

  rc <- tryCatch(utils::download.file(url, tmp, mode = "wb", quiet = TRUE),
                  error = function(e) 1L)
  if (!identical(rc, 0L))
    stop("Failed to download uv. Install manually: https://docs.astral.sh/uv/",
         call. = FALSE)

  dir.create(tmp_dir, showWarnings = FALSE)
  utils::untar(tmp, exdir = tmp_dir)
  bins <- list.files(tmp_dir, pattern = "^uv$", recursive = TRUE,
                     full.names = TRUE)
  if (length(bins) == 0) stop("uv binary not found in archive.", call. = FALSE)

  file.copy(bins[1], uv_path, overwrite = TRUE)
  Sys.chmod(uv_path, "0755")
  message("dsFlowerClient: uv installed at ", uv_path)
  .dsflower_client_runtime$uv_path <- uv_path
  uv_path
}
