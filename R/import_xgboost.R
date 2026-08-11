# Safe local admission of one external, data-only XGBoost JSON artifact.

.EXTERNAL_XGBOOST_IMPORT_TIMEOUT_SECONDS <- 300

.external_xgboost_scalar_path <- function(value, name) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(value)) {
    stop(name, " must be one non-empty path.", call. = FALSE)
  }
  path.expand(value)
}

.external_xgboost_input_path <- function(path) {
  path <- .external_xgboost_scalar_path(path, "artifact")
  absolute <- if (.Platform$OS.type == "windows") {
    grepl("^(?:[A-Za-z]:[/\\\\]|[/\\\\]{2})", path, perl = TRUE)
  } else {
    startsWith(path, "/")
  }
  if (absolute) path else file.path(getwd(), path)
}

.external_xgboost_destination <- function(path) {
  path <- .external_xgboost_scalar_path(path, "output_dir")
  leaf <- basename(path)
  if (!nzchar(leaf) || leaf %in% c(".", "..")) {
    stop("output_dir must name one new directory.", call. = FALSE)
  }
  parent <- normalizePath(
    dirname(path), winslash = "/", mustWork = TRUE)
  destination <- file.path(parent, leaf)
  link <- suppressWarnings(Sys.readlink(destination))
  if (file.exists(destination) || dir.exists(destination) ||
      (length(link) == 1L && !is.na(link) && nzchar(link))) {
    stop("output_dir must not already exist.", call. = FALSE)
  }
  list(parent = parent, path = destination, leaf = leaf)
}

.external_xgboost_import_helper <- function() {
  helper <- system.file(
    "python", "import_xgboost_model.py", package = "dsFlowerClient")
  if (!nzchar(helper)) {
    helper <- file.path("inst", "python", "import_xgboost_model.py")
  }
  if (!file.exists(helper)) {
    stop("Bundled external XGBoost importer not found.", call. = FALSE)
  }
  helper
}

.write_external_xgboost_file <- function(path, bytes) {
  if (!is.raw(bytes) || !length(bytes)) {
    stop("External XGBoost import produced empty public metadata.",
         call. = FALSE)
  }
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeBin(bytes, connection)
  close(connection)
  on.exit(NULL, add = FALSE)
  if (.Platform$OS.type != "windows") Sys.chmod(path, "0600")
  invisible(path)
}

.run_external_xgboost_import <- function(
    artifact, request_path, request_sha256, output_dir) {
  processx::run(
    command = .client_python_cmd(),
    args = c(
      "-I", "-S", .external_xgboost_import_helper(),
      "--artifact", artifact,
      "--request", request_path,
      "--request-sha256", request_sha256,
      "--output-dir", output_dir),
    env = .client_venv_env(), error_on_status = FALSE,
    timeout = .EXTERNAL_XGBOOST_IMPORT_TIMEOUT_SECONDS)
}

#' Import a bounded external XGBoost JSON model
#'
#' Converts one external XGBoost 3.4.0 JSON model into the same canonical,
#' data-only ensemble and prediction sidecar used by dsFlower's trusted local
#' predictor and private validation runner. The importer never loads XGBoost,
#' pickle, joblib, callbacks, or executable model payloads.
#'
#' The supplied \code{model} is a public admission profile, not a training
#' request: its tree count, depth, learning rate and leaf bound must admit the
#' external artifact. Feature cuts and bounds must exactly describe every
#' accepted split. The resulting bundle is emitted as
#' \code{external-unverified}; dsFlower does not claim that the imported model
#' was trained with differential privacy. A later \code{ds.flower.validate()}
#' call still releases its pooled metrics through the normal node-DP mechanism.
#' The external origin is bound into the hashed ensemble contract, but this is
#' local format provenance rather than a cryptographic signature against the
#' model-directory owner. External leaf values can contain ordinary non-private
#' statistics, so the bundle records that unnoised statistics may be present.
#'
#' @param artifact Path to one bounded regular XGBoost JSON model file.
#' @param model A valid \code{ds.flower.model.xgboost()} public profile.
#' @param features Ordered public feature names.
#' @param feature_bounds Public \code{list(lower=..., upper=...)} vectors.
#' @param feature_cuts One ordered public cut vector per feature.
#' @param target Public target name used by the model schema.
#' @param target_levels Two ordered public levels for a binary model.
#' @param target_bounds Public \code{list(lower=..., upper=...)} for regression.
#' @param output_dir A new destination directory. It is published only after
#'   the complete bundle has passed the trusted parser and prediction probe.
#' @return The absolute path to the imported model directory.
#' @export
ds.flower.import_xgboost <- function(
    artifact, model, features, feature_bounds, feature_cuts, target,
    target_levels = NULL, target_bounds = NULL, output_dir) {
  if (!inherits(model, "dsflower_model") ||
      !identical(model$name, "xgboost") ||
      !identical(model$track, "native_tree") ||
      !identical(model$framework, "xgboost") || !is.list(model$params)) {
    stop("model must be a valid ds.flower.model.xgboost() profile.",
         call. = FALSE)
  }
  task <- model$params$task
  if (!is.character(task) || length(task) != 1L || is.na(task) ||
      !task %in% c("binary", "regression")) {
    stop("The XGBoost import profile has no supported task.", call. = FALSE)
  }
  public_bounds <- .validate_public_feature_bounds(feature_bounds, features)
  public_target <- .validate_public_target_spec(
    target_levels, target_bounds,
    task_type = if (identical(task, "regression")) {
      "regression"
    } else {
      "classification"
    },
    n_classes = 2L)
  if (identical(task, "binary") && is.null(public_target$levels)) {
    stop("Binary XGBoost import requires two ordered target_levels.",
         call. = FALSE)
  }
  request <- .build_xgboost_request(
    model$params, features,
    public_bounds[c("lower", "upper")], feature_cuts,
    target_name = target, target_levels = public_target$levels,
    target_bounds = public_target$bounds)

  artifact <- .external_xgboost_input_path(artifact)
  destination <- .external_xgboost_destination(output_dir)
  work_dir <- tempfile(
    pattern = paste0(".", destination$leaf, "-import-"),
    tmpdir = destination$parent)
  if (!dir.create(work_dir, mode = "0700", showWarnings = FALSE)) {
    stop("Could not create the private XGBoost import staging directory.",
         call. = FALSE)
  }
  bundle_dir <- tempfile(
    pattern = paste0(".", destination$leaf, "-bundle-"),
    tmpdir = destination$parent)
  if (!dir.create(bundle_dir, mode = "0700", showWarnings = FALSE)) {
    unlink(work_dir, recursive = TRUE, force = TRUE)
    stop("Could not create the private XGBoost bundle staging directory.",
         call. = FALSE)
  }
  installed <- FALSE
  complete <- FALSE
  on.exit({
    if (dir.exists(work_dir)) {
      unlink(work_dir, recursive = TRUE, force = TRUE)
    }
    if (dir.exists(bundle_dir)) {
      unlink(bundle_dir, recursive = TRUE, force = TRUE)
    }
    if (installed && !complete && dir.exists(destination$path)) {
      unlink(destination$path, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)

  request_path <- file.path(work_dir, "request.json")
  .write_external_xgboost_file(
    request_path, charToRaw(enc2utf8(request$json)))
  result <- .run_external_xgboost_import(
    artifact, request_path, request$sha256, bundle_dir)
  manifest_text <- result$stdout %||% ""
  manifest_bytes <- charToRaw(enc2utf8(manifest_text))
  if (!identical(as.integer(result$status), 0L) ||
      !identical(result$stderr %||% "", "") || !length(manifest_bytes) ||
      length(manifest_bytes) > .EXTERNAL_XGBOOST_METADATA_MAX_BYTES ||
      any(as.integer(manifest_bytes) > 127L)) {
    stop("External XGBoost import was rejected.", call. = FALSE)
  }

  metadata_path <- file.path(bundle_dir, "metadata.json")
  metadata_tmp <- file.path(bundle_dir, ".metadata.json.tmp")
  .write_external_xgboost_file(metadata_tmp, manifest_bytes)
  if (!file.rename(metadata_tmp, metadata_path)) {
    stop("Could not finalize external XGBoost metadata.", call. = FALSE)
  }
  staged <- .resolve_validation_contract(bundle_dir, 32L)
  .validate_validation_artifact_preflight(staged)
  if (!file.rename(bundle_dir, destination$path)) {
    stop("Could not publish the external XGBoost model directory.",
         call. = FALSE)
  }
  installed <- TRUE
  final <- .resolve_validation_contract(destination$path, 32L)
  .validate_validation_artifact_preflight(final)
  complete <- TRUE
  normalizePath(destination$path, winslash = "/", mustWork = TRUE)
}
