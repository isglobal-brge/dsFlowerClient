# Module: App upload (researcher side) -- build a Flower app into a FAB
# and upload it to the nodes over the DataSHIELD channel (chunked, idempotent),
# then verify + install it by sha256. Upload is a transport/integrity operation;
# it does not authorize execution or establish a DP guarantee. HookApp execution
# is separately governed by the node's sandbox and egress policy. Transport is
# datashield.aggregate with base64 direct-string chunks -- never assign.expr.

#' @keywords internal
.app_enc_b64 <- function(raw) {
  if (length(raw) == 0) return("")
  b <- gsub("[\r\n]", "", jsonlite::base64_enc(raw))
  b <- gsub("\\+", "-", b); b <- gsub("/", "_", b); b <- gsub("=+$", "", b)
  paste0("B64:", b)
}

#' Build a Flower app directory into a FAB (returns the .fab path)
#' @keywords internal
.flwr_build_fab <- function(app_dir) {
  flwr <- .client_flwr_cmd()
  unlink(list.files(app_dir, pattern = "\\.fab$", full.names = TRUE))
  res <- processx::run(flwr, c("build", "--app", app_dir),
                       env = .client_venv_env(), wd = app_dir,
                       error_on_status = FALSE)
  fabs <- list.files(app_dir, pattern = "\\.fab$", full.names = TRUE)
  if (length(fabs) == 0) {
    stop("`flwr build` produced no FAB.\n", res$stdout, res$stderr, call. = FALSE)
  }
  fabs[which.max(file.mtime(fabs))]
}

#' Upload and hash-verify a Flower app archive
#'
#' Builds \code{app_dir} into a FAB, pushes it to every node in \code{conns} in
#' idempotent base64 chunks over DataSHIELD, then installs it (the node verifies
#' the SHA-256 and rejects any mismatch). This function only transports and
#' installs a candidate archive: it does not authorize its execution and does
#' not turn arbitrary code into a per-sample DP computation. HookApps are
#' separately hash-pinned and run only when every node-side execution gate holds.
#'
#' @param conns DSI connections object.
#' @param app_dir Character; path to the Flower app directory to bundle.
#' @param chunk_bytes Integer; bytes per push (default 256 KiB).
#' @param verbose Logical; print progress (default TRUE).
#' @return A \code{dsflower_app} object: list(token, sha256, size).
#' @export
ds.flower.app.upload <- function(conns, app_dir, chunk_bytes = 262144L,
                                 verbose = TRUE) {
  if (!is.numeric(chunk_bytes) || length(chunk_bytes) != 1L || is.na(chunk_bytes) ||
      !is.finite(chunk_bytes) || chunk_bytes <= 0 || chunk_bytes %% 1 != 0) {
    stop("chunk_bytes must be a single positive integer.", call. = FALSE)
  }
  if (!dir.exists(app_dir)) {
    stop("app_dir does not exist: ", app_dir, call. = FALSE)
  }
  fab <- .flwr_build_fab(app_dir)
  n <- file.size(fab)
  if (n == 0) stop("Built FAB is empty.", call. = FALSE)
  raw <- readBin(fab, "raw", n)
  sha <- digest::digest(file = fab, algo = "sha256")
  token <- .new_capability_token("app")
  installed <- FALSE
  on.exit({
    if (!installed) {
      tryCatch(DSI::datashield.aggregate(
        conns, call("flowerAppDeleteDS", token)), error = function(e) NULL)
    }
  }, add = TRUE)

  off <- 0
  while (off < n) {
    hi <- min(off + chunk_bytes, n)
    chunk <- .app_enc_b64(raw[(off + 1):hi])
    DSI::datashield.aggregate(
      conns, call("flowerAppPushDS", token, chunk, off))
    off <- hi
    if (verbose) cat(sprintf("\r  uploading app: %d / %d bytes", off, n))
  }
  if (verbose) cat("\n")

  inst <- DSI::datashield.aggregate(
    conns, call("flowerAppInstallDS", token, sha))
  failed <- names(inst)[!vapply(inst, function(x) isTRUE(x$ok), logical(1))]
  if (length(failed) > 0) {
    stop("App install/verification failed on: ", paste(failed, collapse = ", "),
         ".", call. = FALSE)
  }
  installed <- TRUE
  if (verbose) {
    message("  App verified by sha256 (", substr(sha, 1, 12), ") on ",
            length(inst), " node(s).")
  }
  structure(list(token = token, sha256 = sha, size = n),
            class = "dsflower_app")
}

#' Print a dsflower_app
#' @param x A dsflower_app object.
#' @param ... Ignored.
#' @return Invisibly x.
#' @export
print.dsflower_app <- function(x, ...) {
  cat("dsflower_app (uploaded and hash-verified; execution not authorized)\n")
  cat("  token:  ", x$token, "\n")
  cat("  sha256: ", x$sha256, "\n")
  cat("  size:   ", x$size, "bytes\n")
  invisible(x)
}
