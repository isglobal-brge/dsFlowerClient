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

#' Stream one archive to every node with exact offset acknowledgements
#' @keywords internal
.push_app_archive <- function(conns, path, token, chunk_bytes, progress = NULL) {
  size <- file.size(path)
  if (length(size) != 1L || is.na(size) || size <= 0) {
    stop("The app archive is empty.", call. = FALSE)
  }
  input <- file(path, open = "rb")
  on.exit(close(input), add = TRUE)

  offset <- 0
  while (offset < size) {
    wanted <- as.integer(min(as.numeric(chunk_bytes), size - offset))
    chunk <- readBin(input, "raw", n = wanted)
    if (length(chunk) != wanted) {
      stop("Could not read the complete app archive chunk.", call. = FALSE)
    }
    expected <- offset + length(chunk)
    chunk_hash <- digest::digest(chunk, algo = "sha256", serialize = FALSE)
    expr <- call("flowerAppPushDS", token, .app_enc_b64(chunk), offset)
    .dsi_retry_exact_aggregate(
      conns, expr,
      validate = function(value, node) {
        if (!is.list(value) ||
            !identical(names(value),
                       c("ok", "offset", "size", "bytes", "sha256"))) {
          return(FALSE)
        }
        ack_offset <- suppressWarnings(as.numeric(value$offset))
        ack_size <- suppressWarnings(as.numeric(value$size))
        ack_bytes <- suppressWarnings(as.numeric(value$bytes))
        isTRUE(value$ok) &&
          length(ack_offset) == 1L && is.finite(ack_offset) &&
          ack_offset == offset &&
          length(ack_size) == 1L && is.finite(ack_size) &&
          ack_size == expected &&
          length(ack_bytes) == 1L && is.finite(ack_bytes) &&
          ack_bytes == length(chunk) &&
          identical(as.character(value$sha256), chunk_hash)
      },
      operation = "App upload")
    offset <- expected
    if (is.function(progress)) progress(offset, size)
  }
  invisible(size)
}

#' Verify and install an uploaded app on every node
#' @keywords internal
.install_app_archive <- function(conns, token, sha256, size, operation) {
  expr <- call("flowerAppInstallDS", token, sha256)
  .dsi_retry_exact_aggregate(
    conns, expr,
    validate = function(value, node) {
      if (!is.list(value) ||
          !identical(names(value), c("ok", "sha256", "size", "packages"))) {
        return(FALSE)
      }
      ack_size <- suppressWarnings(as.numeric(value$size))
      isTRUE(value$ok) &&
        identical(as.character(value$sha256), sha256) &&
        length(ack_size) == 1L && is.finite(ack_size) && ack_size == size
    },
    operation = operation)
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
  sha <- digest::digest(file = fab, algo = "sha256")
  token <- .new_capability_token("app")
  installed <- FALSE
  on.exit({
    if (!installed) {
      tryCatch(DSI::datashield.aggregate(
        conns, call("flowerAppDeleteDS", token)), error = function(e) NULL)
    }
  }, add = TRUE)

  .push_app_archive(
    conns, fab, token, chunk_bytes,
    progress = if (verbose) function(done, total) {
      cat(sprintf("\r  uploading app: %d / %d bytes", done, total))
    } else NULL)
  if (verbose) cat("\n")

  inst <- .install_app_archive(
    conns, token, sha, n, "App install/verification")
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
