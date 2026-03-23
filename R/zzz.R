# Module: Package Hooks
# Session-level lifecycle for dsFlowerClient.

#' Package load hook
#'
#' Checks that the client Python venv is healthy. If not, prints a
#' message guiding the researcher to reinstall or run ensure manually.
#'
#' @param libname Library name.
#' @param pkgname Package name.
#' @keywords internal
.onLoad <- function(libname, pkgname) {
  # Nothing to register (client package, no DataSHIELD methods).
  # Venv check is done in .onAttach to show user-facing messages.
}

#' Package attach hook
#'
#' Verifies the Python venv and prints startup info.
#'
#' @param libname Library name.
#' @param pkgname Package name.
#' @keywords internal
.onAttach <- function(libname, pkgname) {
  if (.client_venv_is_healthy()) {
    packageStartupMessage(
      "dsFlowerClient ", utils::packageVersion(pkgname),
      " -- Python environment OK")
  } else {
    packageStartupMessage(
      "dsFlowerClient ", utils::packageVersion(pkgname),
      "\n  NOTE: Python environment not found at ", .client_venv_path(),
      "\n  Run .ensure_client_venv() or reinstall the package to set up.")
  }
}

#' Package unload hook
#'
#' Automatically stops the SuperLink when the package is unloaded or the
#' R session ends. Prevents orphaned flower-superlink processes.
#'
#' @param libpath Library path.
#' @keywords internal
.onUnload <- function(libpath) {
  tryCatch({
    info <- .dsflower_client_env$.superlink
    if (!is.null(info) && !is.null(info$process) && info$process$is_alive()) {
      info$process$signal(15L)
      info$process$wait(timeout = 3000)
      if (info$process$is_alive()) info$process$kill()
    }
  }, error = function(e) NULL)
}
