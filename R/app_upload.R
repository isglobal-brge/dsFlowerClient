# Module: App upload (researcher side, Tier-2) -- build a Flower app into a FAB
# and upload it to the nodes over the DataSHIELD channel (chunked, idempotent),
# then verify + install it by sha256. The node runs only the hash-verified app;
# the researcher provisions code but the node never trusts it (it is sandboxed +
# egress-gated at run time). Transport is datashield.aggregate with base64
# direct-string chunks -- never assign.expr.

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

#' Upload a Flower app to the nodes and verify it by sha256 (Tier-2)
#'
#' Builds \code{app_dir} into a FAB, pushes it to every node in \code{conns} in
#' idempotent base64 chunks over DataSHIELD, then installs it (the node verifies
#' the sha256 and rejects any mismatch). Returns a handle carrying the upload
#' token + validated hash; the run then executes only this hash-pinned app.
#'
#' @param conns DSI connections object.
#' @param app_dir Character; path to the Flower app directory to bundle.
#' @param chunk_bytes Integer; bytes per push (default 256 KiB).
#' @param verbose Logical; print progress (default TRUE).
#' @return A \code{dsflower_app} object: list(token, sha256, size).
#' @export
ds.flower.app.upload <- function(conns, app_dir, chunk_bytes = 262144L,
                                 verbose = TRUE) {
  if (!dir.exists(app_dir)) {
    stop("app_dir does not exist: ", app_dir, call. = FALSE)
  }
  fab <- .flwr_build_fab(app_dir)
  n <- file.size(fab)
  if (n == 0) stop("Built FAB is empty.", call. = FALSE)
  raw <- readBin(fab, "raw", n)
  sha <- digest::digest(file = fab, algo = "sha256")
  token <- paste0("app", paste(sample(c(letters, 0:9), 12, replace = TRUE),
                                collapse = ""))

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
  cat("dsflower_app (Tier-2 uploaded, hash-verified)\n")
  cat("  token:  ", x$token, "\n")
  cat("  sha256: ", x$sha256, "\n")
  cat("  size:   ", x$size, "bytes\n")
  invisible(x)
}
