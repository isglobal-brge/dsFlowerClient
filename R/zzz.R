# Module: Package Hooks
# Session-level cleanup for dsFlowerClient.

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
