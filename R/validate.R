# Standalone, node-private validation of a released dsFlower model.  The public
# model is transported as bounded Flower data; exact predictions, labels, counts
# and site metrics remain inside each trusted ClientApp.

.XGBOOST_ENSEMBLE_FILE <- "model.xgboost-ensemble.json"
.XGBOOST_ENSEMBLE_PROFILE_FILE <- "model.xgboost-ensemble.profile.json"
.XGBOOST_ENSEMBLE_FORMAT <- "dsflower-xgboost-ensemble-json-v1"
.XGBOOST_ENSEMBLE_CONTRACT <- "dsflower-xgboost-ensemble-v1"
.XGBOOST_ENSEMBLE_MAX_BYTES <- 64 * 1024^2
.XGBOOST_PREDICTION_PROFILE_CONTRACT <-
  "dsflower-xgboost-prediction-profile-v1"
.XGBOOST_PREDICTION_PROFILE_MAX_BYTES <- 128 * 1024L
.XGBOOST_ENSEMBLE_SANITIZATION <- list(
  profile = "dsflower-xgboost-ensemble-v1",
  privacy_basis = "direct-dp-training-postprocessing",
  contains_raw_records = FALSE,
  contains_unnoised_statistics = FALSE,
  contains_feature_names = FALSE,
  contains_target_name = FALSE,
  contains_training_history = FALSE,
  contains_backend_logs = FALSE,
  contains_paths = FALSE,
  contains_executable_payload = FALSE)

.validation_sha256 <- function(value, name) {
  value <- .validation_atomic(value)
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !grepl("^[0-9a-f]{64}$", value)) {
    stop(name, " must be one lowercase SHA-256 digest.", call. = FALSE)
  }
  value
}

# Validate the public global ensemble before any DSI or private-data access.
.validate_xgboost_ensemble_artifact <- function(meta, model_dir, task) {
  expected_schema_hash <- .validation_sha256(
    meta$public_schema_sha256, "Saved XGBoost public schema SHA-256")
  artifact <- meta$artifact
  if (!is.list(artifact) || !identical(
      sort(names(artifact)), c("file", "format", "sha256", "size_bytes"))) {
    stop("Saved XGBoost artifact metadata has unsupported fields.",
         call. = FALSE)
  }
  file <- .validation_atomic(artifact$file)
  format <- .validation_atomic(artifact$format)
  if (!identical(file, .XGBOOST_ENSEMBLE_FILE) ||
      !identical(format, .XGBOOST_ENSEMBLE_FORMAT)) {
    stop("Saved XGBoost artifact format is not the sanitized ensemble profile.",
         call. = FALSE)
  }
  expected_size <- suppressWarnings(as.numeric(.validation_atomic(
    artifact$size_bytes)))
  if (length(expected_size) != 1L || is.na(expected_size) ||
      !is.finite(expected_size) || expected_size != floor(expected_size) ||
      expected_size < 1 || expected_size > .XGBOOST_ENSEMBLE_MAX_BYTES) {
    stop("Saved XGBoost artifact size is outside its public bound.",
         call. = FALSE)
  }
  expected_hash <- .validation_sha256(
    artifact$sha256, "Saved XGBoost artifact SHA-256")
  sanitization <- meta$sanitization
  if (!is.list(sanitization) ||
      !identical(sort(names(sanitization)),
                 sort(names(.XGBOOST_ENSEMBLE_SANITIZATION))) ||
      !identical(sanitization[sort(names(sanitization))],
                 .XGBOOST_ENSEMBLE_SANITIZATION[
                   sort(names(.XGBOOST_ENSEMBLE_SANITIZATION))])) {
    stop("Saved XGBoost artifact lacks the exact sanitization attestation.",
         call. = FALSE)
  }
  path <- file.path(model_dir, file)
  info <- file.info(path)
  if (!file.exists(path) || is.na(info$isdir) || isTRUE(info$isdir) ||
      is.na(info$size) || info$size != expected_size) {
    stop("Saved XGBoost ensemble artifact is missing or has the wrong size.",
         call. = FALSE)
  }
  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  bytes <- readBin(con, what = "raw", n = as.integer(expected_size))
  if (length(bytes) != expected_size ||
      !identical(digest::digest(bytes, algo = "sha256", serialize = FALSE),
                 expected_hash)) {
    stop("Saved XGBoost ensemble artifact SHA-256 mismatch.", call. = FALSE)
  }
  value <- tryCatch(
    jsonlite::fromJSON(rawToChar(bytes), simplifyVector = FALSE),
    error = function(e) NULL)
  if (!is.list(value) || !identical(
      sort(names(value)),
      c("aggregation", "contract", "engine", "models",
        "public_schema_sha256", "task", "version")) ||
      !identical(value$contract, .XGBOOST_ENSEMBLE_CONTRACT) ||
      !identical(value$engine, "xgboost") ||
      !identical(value$task, task) ||
      !identical(value$aggregation, "mean_prediction") ||
      !(is.integer(value$version) || is.numeric(value$version)) ||
      is.logical(value$version) || length(value$version) != 1L ||
      is.na(value$version) || !is.finite(value$version) ||
      value$version != 1 || !is.list(value$models) || !length(value$models) ||
      !all(vapply(value$models, function(model) {
        is.list(model) && identical(sort(names(model)), c("learner", "version"))
      }, logical(1)))) {
    stop("Saved XGBoost ensemble violates its canonical container contract.",
         call. = FALSE)
  }
  container_schema_hash <- .validation_sha256(
    value$public_schema_sha256,
    "Saved XGBoost ensemble public schema SHA-256")
  if (!identical(container_schema_hash, expected_schema_hash)) {
    stop("Saved XGBoost ensemble public schema SHA-256 mismatch.",
         call. = FALSE)
  }
  list(
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    format = format, sha256 = expected_hash, size_bytes = as.integer(expected_size),
    public_schema_sha256 = expected_schema_hash,
    sanitization = sanitization)
}

# Validate the exact public sidecar emitted atomically with the ensemble. This
# binds the portable files without enabling local prediction or reconstructing
# any node-private backend manifest.
.validate_xgboost_prediction_profile <- function(
    model_dir, request_b64, request_sha256, public_schema_sha256,
    task, artifact) {
  request <- .validate_native_tree_request_wire(request_b64, request_sha256)
  schema_sha256 <- .validation_sha256(
    public_schema_sha256, "Saved XGBoost public schema SHA-256")
  if (!is.character(task) || length(task) != 1L || is.na(task) ||
      !task %in% c("binary", "regression") ||
      !identical(request$value$engine, "xgboost") ||
      !identical(request$value$task, task) ||
      !identical(request$value$public_schema$sha256, schema_sha256)) {
    stop("Saved XGBoost profile inputs disagree with the canonical request.",
         call. = FALSE)
  }
  if (!is.list(artifact)) {
    stop("Saved XGBoost profile has invalid artifact bindings.", call. = FALSE)
  }
  artifact_format <- .validation_atomic(artifact$format)
  artifact_sha256 <- .validation_sha256(
    artifact$sha256, "Saved XGBoost artifact SHA-256")
  artifact_size <- .validation_scalar_integer(
    artifact$size_bytes, "Saved XGBoost artifact size", 1L,
    .XGBOOST_ENSEMBLE_MAX_BYTES)
  if (!identical(artifact_format, .XGBOOST_ENSEMBLE_FORMAT)) {
    stop("Saved XGBoost profile has invalid artifact bindings.", call. = FALSE)
  }

  path <- file.path(model_dir, .XGBOOST_ENSEMBLE_PROFILE_FILE)
  info <- file.info(path)
  if (!file.exists(path) || is.na(info$isdir) || isTRUE(info$isdir) ||
      is.na(info$size) || info$size < 1 ||
      info$size > .XGBOOST_PREDICTION_PROFILE_MAX_BYTES) {
    stop("Saved XGBoost prediction profile is missing or outside its byte bound.",
         call. = FALSE)
  }
  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  bytes <- readBin(con, what = "raw", n = as.integer(info$size))
  if (length(bytes) != info$size || any(as.integer(bytes) > 127L)) {
    stop("Saved XGBoost prediction profile is not canonical ASCII JSON.",
         call. = FALSE)
  }
  value <- tryCatch(
    jsonlite::fromJSON(rawToChar(bytes), simplifyVector = FALSE),
    error = function(e) NULL)
  expected_fields <- c(
    "artifact", "contract", "native_tree_request_b64",
    "native_tree_request_sha256", "public_schema_sha256", "task", "version")
  artifact_fields <- c("format", "sha256", "size_bytes")
  if (!is.list(value) || is.null(names(value)) || anyDuplicated(names(value)) ||
      !identical(sort(names(value)), sort(expected_fields)) ||
      !is.list(value$artifact) || is.null(names(value$artifact)) ||
      anyDuplicated(names(value$artifact)) ||
      !identical(sort(names(value$artifact)), sort(artifact_fields)) ||
      !identical(value$contract, .XGBOOST_PREDICTION_PROFILE_CONTRACT) ||
      !is.integer(value$version) || !identical(value$version, 1L)) {
    stop("Saved XGBoost prediction profile violates its exact contract.",
         call. = FALSE)
  }
  profile_request <- .validate_native_tree_request_wire(
    value$native_tree_request_b64, value$native_tree_request_sha256)
  if (!identical(profile_request$b64, request$b64) ||
      !identical(profile_request$sha256, request$sha256) ||
      !identical(value$public_schema_sha256, schema_sha256) ||
      !identical(value$task, task) ||
      !identical(value$artifact$format, artifact_format) ||
      !identical(value$artifact$sha256, artifact_sha256) ||
      !identical(value$artifact$size_bytes, artifact_size)) {
    stop("Saved XGBoost prediction profile bindings do not match the ensemble.",
         call. = FALSE)
  }
  canonical <- list(
    artifact = list(
      format = artifact_format,
      sha256 = artifact_sha256,
      size_bytes = artifact_size),
    contract = .XGBOOST_PREDICTION_PROFILE_CONTRACT,
    native_tree_request_b64 = request$b64,
    native_tree_request_sha256 = request$sha256,
    public_schema_sha256 = schema_sha256,
    task = task,
    version = 1L)
  if (!identical(bytes, .native_tree_json(canonical))) {
    stop("Saved XGBoost prediction profile is not canonically encoded.",
         call. = FALSE)
  }
  list(
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    size_bytes = as.integer(info$size),
    sha256 = digest::digest(bytes, algo = "sha256", serialize = FALSE))
}

.validation_scalar_integer <- function(value, name, lower, upper) {
  value <- unlist(value, use.names = FALSE)
  if (!is.numeric(value) || is.logical(value)) {
    stop(name, " must be one integer in [", lower, ", ", upper, "].",
         call. = FALSE)
  }
  number <- as.numeric(value)
  if (length(number) != 1L || is.na(number) || !is.finite(number) ||
      number != floor(number) || number < lower || number > upper) {
    stop(name, " must be one integer in [", lower, ", ", upper, "].",
         call. = FALSE)
  }
  as.integer(number)
}

.validation_atomic <- function(value) {
  if (is.null(value)) return(NULL)
  unlist(value, recursive = TRUE, use.names = FALSE)
}

.validation_public_bounds <- function(meta, prefix) {
  lower <- suppressWarnings(as.numeric(.validation_atomic(
    meta[[paste0(prefix, "_lower")]])))
  upper <- suppressWarnings(as.numeric(.validation_atomic(
    meta[[paste0(prefix, "_upper")]])))
  if (!length(lower) && !length(upper)) return(NULL)
  if (!length(lower) || length(lower) != length(upper) ||
      any(!is.finite(lower)) || any(!is.finite(upper)) ||
      any(lower >= upper) || any(abs(lower) > 1e6) ||
      any(abs(upper) > 1e6)) {
    stop("Saved model has an invalid public ", prefix, " bound contract.",
         call. = FALSE)
  }
  list(lower = lower, upper = upper)
}

.validation_target_bounds <- function(meta) {
  bounds <- meta$target_bounds
  if (is.null(bounds)) return(NULL)
  if (!is.list(bounds)) {
    stop("Saved model has an invalid public target bound contract.",
         call. = FALSE)
  }
  lower <- suppressWarnings(as.numeric(.validation_atomic(bounds$lower)))
  upper <- suppressWarnings(as.numeric(.validation_atomic(bounds$upper)))
  if (length(lower) != 1L || length(upper) != 1L ||
      !is.finite(lower) || !is.finite(upper) || lower >= upper ||
      abs(lower) > 1e6 || abs(upper) > 1e6) {
    stop("Saved model has an invalid public target bound contract.",
         call. = FALSE)
  }
  list(lower = lower, upper = upper)
}

.validation_model_dir <- function(model) {
  path <- if (inherits(model, "dsflower_run")) {
    model$output_dir
  } else if (is.character(model) && length(model) == 1L && !is.na(model)) {
    model
  } else {
    NULL
  }
  if (is.null(path) || !nzchar(path) || !dir.exists(path)) {
    stop("'model' must be a dsflower_run or its saved output directory.",
         call. = FALSE)
  }
  normalizePath(path, winslash = "/", mustWork = TRUE)
}

.resolve_validation_contract <- function(model, bins) {
  model_dir <- .validation_model_dir(model)
  metadata_path <- file.path(model_dir, "metadata.json")
  if (!file.exists(metadata_path)) {
    stop("Private validation requires the model's metadata.json contract.",
         call. = FALSE)
  }
  meta <- tryCatch(
    jsonlite::fromJSON(metadata_path, simplifyVector = FALSE),
    error = function(e) NULL)
  if (!is.list(meta)) {
    stop("Saved model metadata is unreadable.", call. = FALSE)
  }
  track <- tolower(as.character(.validation_atomic(meta$track %||% "")))
  if (length(track) != 1L || !track %in% c("neural", "native_tree")) {
    stop("Private validation supports declarative neural artifacts or sanitized ",
         "native-tree artifacts.", call. = FALSE)
  }
  data_kind <- .validation_atomic(meta$data_kind)
  if (!is.character(data_kind) || length(data_kind) != 1L ||
      is.na(data_kind) || !data_kind %in% c("tabular", "image")) {
    stop("Saved model has an invalid data_kind contract.", call. = FALSE)
  }
  if (identical(data_kind, "image")) {
    stop("ds.flower.validate() currently supports tabular artifacts only; ",
         "image models require a dedicated image validation protocol.",
         call. = FALSE)
  }
  features <- as.character(.validation_atomic(meta$features))
  if (!length(features) || anyNA(features) || any(!nzchar(features)) ||
      anyDuplicated(features)) {
    stop("Private validation requires the saved tabular feature contract.",
         call. = FALSE)
  }
  feature_bounds <- .validation_public_bounds(meta, "feature")
  if (!is.null(feature_bounds) &&
      length(feature_bounds$lower) != length(features)) {
    stop("Saved public feature bounds do not match the feature contract.",
         call. = FALSE)
  }
  target_bounds <- .validation_target_bounds(meta)
  bins <- .validation_scalar_integer(bins, "bins", 4L, 512L)

  if (identical(track, "native_tree")) {
    expected_fields <- c(
      "artifact", "data_kind", "engine", "feature_lower", "feature_upper",
      "features", "native_tree_request_b64", "native_tree_request_sha256",
      "public_schema_sha256", "sanitization", "target_bounds",
      "target_levels", "task", "track")
    if (is.null(names(meta)) || anyDuplicated(names(meta)) ||
        !identical(sort(names(meta)), expected_fields)) {
      stop("Saved XGBoost metadata has unsupported or missing fields.",
           call. = FALSE)
    }
    engine <- tolower(as.character(.validation_atomic(meta$engine %||% "")))
    task <- tolower(as.character(.validation_atomic(meta$task %||% "")))
    if (length(engine) != 1L || !identical(engine, "xgboost") ||
        length(task) != 1L || !task %in% c("binary", "regression")) {
      stop("Saved native-tree validation supports XGBoost binary or regression only.",
           call. = FALSE)
    }
    if (is.null(feature_bounds)) {
      stop("Saved XGBoost validation requires exact public feature bounds.",
           call. = FALSE)
    }
    public_levels <- .validation_atomic(meta$target_levels)
    if (identical(task, "binary")) {
      if (length(public_levels) != 2L || anyNA(public_levels) ||
          anyDuplicated(public_levels) || !is.null(target_bounds)) {
        stop("Saved binary XGBoost requires two public target levels and no ",
             "regression target bounds.", call. = FALSE)
      }
    } else if (!is.null(public_levels) || is.null(target_bounds)) {
      stop("Saved regression XGBoost requires public target bounds and no ",
           "classification levels.", call. = FALSE)
    }
    request_sha256 <- .validation_sha256(
      meta$native_tree_request_sha256,
      "Saved native-tree request SHA-256")
    request <- .validate_native_tree_request_wire(
      .validation_atomic(meta$native_tree_request_b64), request_sha256)
    request_schema <- request$value$public_schema
    request_features <- as.character(unlist(
      request_schema$features, use.names = FALSE))
    if (!identical(request$value$engine, engine) ||
        !identical(request$value$task, task) ||
        !identical(request_features, features) ||
        !identical(as.numeric(request_schema$lower), feature_bounds$lower) ||
        !identical(as.numeric(request_schema$upper), feature_bounds$upper) ||
        !identical(request_schema$sha256,
                   .validation_atomic(meta$public_schema_sha256))) {
      stop("Saved native-tree request differs from its public model metadata.",
           call. = FALSE)
    }
    request_target <- request_schema$target
    if (identical(task, "binary")) {
      level_type <- request_target$levels[[1L]]$type
      template <- switch(
        level_type, string = character(1), boolean = logical(1),
        number = numeric(1), NULL)
      request_levels <- if (is.null(template)) NULL else vapply(
        request_target$levels, `[[`, template, "value")
      if (is.null(request_levels) ||
          !identical(unname(public_levels), unname(request_levels))) {
        stop("Saved native-tree target levels differ from its canonical request.",
             call. = FALSE)
      }
    } else if (!identical(
        c(target_bounds$lower, target_bounds$upper),
        c(as.numeric(request_target$lower),
          as.numeric(request_target$upper)))) {
      stop("Saved native-tree target bounds differ from its canonical request.",
           call. = FALSE)
    }
    artifact <- .validate_xgboost_ensemble_artifact(meta, model_dir, task)
    profile <- .validate_xgboost_prediction_profile(
      model_dir, request$b64, request_sha256,
      artifact$public_schema_sha256, task, meta$artifact)
    return(list(
      model_dir = model_dir, artifact = artifact$path,
      artifact_format = artifact$format, artifact_sha256 = artifact$sha256,
      artifact_size_bytes = artifact$size_bytes,
      sanitization = artifact$sanitization,
      native_tree_request_b64 = request$b64,
      native_tree_request_sha256 = request_sha256,
      public_schema_sha256 = artifact$public_schema_sha256,
      prediction_profile = profile,
      engine = engine, track = track, task = task, bins = bins,
      features = features, feature_bounds = feature_bounds,
      target_bounds = target_bounds, target_levels = public_levels,
      model_spec = NULL,
      loss_name = if (identical(task, "binary")) "bce_logits" else "mse",
      n_classes = 2L, n_labels = 2L, data_kind = data_kind))
  }

  params <- meta$model_params
  if (!is.list(params)) params <- list()
  public_levels <- .validation_atomic(meta$target_levels)
  class_default <- if (length(public_levels) >= 2L) length(public_levels) else 2L
  n_classes <- .validation_scalar_integer(
    params[["n_classes"]] %||% params[["num_classes"]] %||% class_default,
    "Saved num_classes", 2L, 1024L)
  n_labels <- .validation_scalar_integer(
    params[["num_labels"]] %||% 2L, "Saved num_labels", 2L, 1024L)
  spec <- meta$model_spec
  loss <- tolower(as.character(.validation_atomic(meta$loss_name %||% "")))
  if (!is.list(spec) || !length(spec) || length(loss) != 1L || !nzchar(loss)) {
    stop("Neural private validation needs the saved declarative spec and loss.",
         call. = FALSE)
  }
  task <- switch(loss,
    bce_logits = "binary",
    cross_entropy = if (n_classes > 2L) "multiclass" else "binary",
    hinge = if (n_classes > 2L) "multiclass" else "binary",
    ordinal = "ordinal", multilabel_bce = "multilabel",
    mse = "regression", huber = "regression", quantile = "regression",
    gamma_nll = "regression",
    poisson_nll = "count", negbin_nll = "count", NULL)
  if (is.null(task)) {
    stop("Saved neural loss has no trusted validation semantics: ", loss, ".",
         call. = FALSE)
  }
  if (identical(loss, "bce_logits") && n_classes != 2L) {
    stop("Saved bce_logits model is not a valid multiclass artifact; use cross_entropy.",
         call. = FALSE)
  }
  if (identical(task, "multilabel") && n_classes != 2L) {
    stop("Saved multilabel model must use binary target levels.",
         call. = FALSE)
  }
  artifact <- file.path(model_dir, "model.pt")
  if (task %in% c("regression", "count") && is.null(target_bounds)) {
    stop("Numeric private validation requires the saved public target_bounds.",
         call. = FALSE)
  }
  info <- file.info(artifact)
  if (!file.exists(artifact) || is.na(info$isdir) || isTRUE(info$isdir) ||
      is.na(info$size) || info$size <= 0 || info$size > 1024^3) {
    stop("Saved model is missing its bounded native public artifact.",
         call. = FALSE)
  }
  list(
    model_dir = model_dir, artifact = normalizePath(
      artifact, winslash = "/", mustWork = TRUE),
    track = track, task = task, bins = bins, features = features,
    feature_bounds = feature_bounds, target_bounds = target_bounds,
    target_levels = public_levels,
    model_spec = spec, loss_name = loss,
    n_classes = n_classes, n_labels = n_labels, data_kind = data_kind)
}

.validation_model_path_b64 <- function(path) {
  gsub("[\r\n]", "", jsonlite::base64_enc(charToRaw(enc2utf8(path))))
}

.validate_validation_artifact_preflight <- function(contract) {
  if (identical(contract$track, "native_tree")) {
    # .resolve_validation_contract already parsed, bounded and hash-checked the
    # singular canonical ensemble without contacting a data node.
    return(invisible(TRUE))
  }
  .ensure_client_framework("pytorch")
  script <- system.file(
    "python", "validate_model_artifact.py", package = "dsFlowerClient")
  if (!nzchar(script)) {
    script <- file.path("inst", "python", "validate_model_artifact.py")
  }
  if (!file.exists(script)) {
    stop("Bundled validation artifact preflight not found.", call. = FALSE)
  }
  config <- list(
    "validation-model-track" = contract$track,
    "validation-model-path-b64" = .validation_model_path_b64(contract$artifact),
    "num-features" = length(contract$features),
    "num-classes" = contract$n_classes,
    "num-labels" = contract$n_labels,
    "loss-name" = contract$loss_name)
  config[["model-spec-b64"]] <- .spec_to_b64(contract$model_spec)
  path <- tempfile(fileext = ".json")
  on.exit(unlink(path), add = TRUE)
  jsonlite::write_json(
    config, path, auto_unbox = TRUE, null = "null", digits = NA)
  result <- processx::run(
    command = .client_python_cmd(), args = c(script, path),
    env = .client_venv_env(), error_on_status = FALSE, timeout = 60)
  if (result$status != 0L) {
    detail <- trimws(result$stderr %||% "")
    if (!nzchar(detail)) detail <- "the trusted decoder rejected it"
    stop("Saved model artifact preflight failed: ", detail, call. = FALSE)
  }
  invisible(TRUE)
}

.validation_contract_sha256 <- function(run_config, feature_columns,
                                        target_column, privacy_unit) {
  bounds <- run_config[["feature-bounds"]] %||% NULL
  target_bounds <- run_config[["target-bounds"]] %||% NULL
  levels <- run_config[["target-levels"]] %||% NULL
  if (is.list(levels) && !is.null(levels$type) && !is.null(levels$values)) {
    level_type <- as.character(levels$type)
    level_values <- unlist(levels$values, use.names = FALSE)
  } else if (is.null(levels)) {
    level_type <- NULL
    level_values <- NULL
  } else {
    level_values <- unlist(levels, use.names = FALSE)
    level_type <- if (is.character(level_values)) "character" else
      if (is.logical(level_values)) "logical" else "numeric"
  }
  payload <- list(
    schema = 1L,
    privacy_unit = as.character(privacy_unit),
    patient_id_canonicalization = if (identical(privacy_unit, "patient"))
      "trim-utf8-v2" else NULL,
    features = as.character(feature_columns),
    targets = as.character(target_column),
    feature_lower = if (is.null(bounds)) NULL else as.numeric(bounds$lower),
    feature_upper = if (is.null(bounds)) NULL else as.numeric(bounds$upper),
    target_level_type = level_type,
    target_levels = level_values,
    target_lower = if (is.null(target_bounds)) NULL else
      as.numeric(target_bounds$lower),
    target_upper = if (is.null(target_bounds)) NULL else
      as.numeric(target_bounds$upper),
    model_track = run_config[["validation-model-track"]],
    task = run_config[["validation-task"]],
    bins = as.integer(run_config[["validation-bins"]]),
    loss = run_config[["loss-name"]],
    num_features = as.integer(run_config[["num-features"]]),
    num_classes = as.integer(run_config[["num-classes"]]),
    num_labels = as.integer(run_config[["num-labels"]]),
    model_spec_b64 = run_config[["model-spec-b64"]] %||% NULL)
  if (identical(run_config[["validation-model-track"]], "native_tree")) {
    payload$native_tree_request_b64 <-
      run_config[["validation-native-tree-request-b64"]]
    payload$native_tree_request_sha256 <-
      run_config[["validation-native-tree-request-sha256"]]
    payload$artifact_format <- run_config[["validation-artifact-format"]]
    payload$artifact_sha256 <- run_config[["validation-artifact-sha256"]]
    payload$artifact_size_bytes <-
      as.integer(run_config[["validation-artifact-size-bytes"]])
    payload$profile_sha256 <- run_config[["validation-profile-sha256"]]
    payload$profile_size_bytes <-
      as.integer(run_config[["validation-profile-size-bytes"]])
    payload$public_schema_sha256 <-
      run_config[["validation-public-schema-sha256"]]
  }
  canonical <- as.character(jsonlite::toJSON(
    payload, auto_unbox = TRUE, null = "null", na = "null",
    digits = NA, always_decimal = TRUE, pretty = FALSE))
  digest::digest(charToRaw(enc2utf8(canonical)), algo = "sha256",
                 serialize = FALSE)
}

.validation_common_privacy_unit <- function(capabilities) {
  if (!is.list(capabilities) || !length(capabilities)) {
    stop("Validation could not verify the federation privacy unit.",
         call. = FALSE)
  }
  units <- vapply(capabilities, function(capability) {
    value <- if (is.list(capability)) capability$privacy_unit else NULL
    value <- tolower(as.character(.validation_atomic(value)))
    if (length(value) == 1L && !is.na(value) &&
        value %in% c("row", "patient")) value else ""
  }, character(1))
  if (any(!nzchar(units)) || length(unique(units)) != 1L) {
    stop("All validation nodes must declare the same row/patient privacy unit.",
         call. = FALSE)
  }
  units[[1L]]
}

.read_private_validation_result <- function(results_dir) {
  path <- file.path(results_dir, "validation.json")
  if (!file.exists(path)) return(NULL)
  value <- tryCatch(jsonlite::fromJSON(path, simplifyVector = FALSE),
                    error = function(e) NULL)
  allowed <- c("pooled_only", "privacy", "task", "n_nodes", "available",
               "metrics")
  if (!is.list(value) || !identical(value$pooled_only, TRUE) ||
      !identical(value$privacy, "node-dp-pooled-postprocessing") ||
      is.null(names(value)) || anyDuplicated(names(value)) ||
      any(!names(value) %in% allowed) ||
      !(identical(value$available, TRUE) ||
        identical(value$available, FALSE)) ||
      length(value$task) != 1L || !value$task %in% c(
        "binary", "multiclass", "ordinal", "multilabel",
        "regression", "count") ||
      length(value$n_nodes) != 1L || !is.numeric(value$n_nodes) ||
      !is.finite(value$n_nodes) || value$n_nodes < 1 ||
      value$n_nodes != floor(value$n_nodes)) {
    stop("Validation output failed the pooled-only privacy contract.",
         call. = FALSE)
  }
  if ((isTRUE(value$available) &&
       (!("metrics" %in% names(value)) || !is.list(value$metrics))) ||
      (!isTRUE(value$available) && "metrics" %in% names(value))) {
    stop("Validation output failed the pooled-only privacy contract.",
         call. = FALSE)
  }
  value
}

#' Differentially-private federated model validation
#'
#' Evaluates a released tabular declarative neural model or sanitized native
#' XGBoost ensemble on the dataset assigned for this call inside each data node.
#' The native request, ensemble and prediction-profile sidecar are pinned into
#' the ephemeral execution contract and every node re-sanitizes the ensemble
#' before opening its data. Vision artifacts fail explicitly because this
#' validator does not reconstruct image loaders or backbones. Reusing the
#' training dataset is resubstitution validation; assigning an independent
#' dataset is external
#' validation. Each protected row/patient contributes one bounded sufficient-statistic
#' vector, the node releases it once through the server-owned Gaussian mechanism,
#' and only pooled post-processed metrics are returned. Exact predictions,
#' labels, counts and per-node metrics never leave the node.
#' All nodes must declare the same row- or patient-level estimand. Privacy is
#' guaranteed per node; if one person occurs in multiple nodes, those node
#' releases compose for that person and deployments should account for the
#' overlap or ensure cohorts are disjoint.
#' If not every expected node returns the fixed private release, the result
#' has \code{available=FALSE} and no metrics rather than exact, per-node or
#' zero-filled substitutes. This is an operational availability result, not a
#' historical query denial.
#'
#' @param conns DSI connections.
#' @param model A successful \code{dsflower_run} or saved model directory.
#' @param target Target column name(s); multilabel validation requires one per
#'   saved label.
#' @param data Optional server-side data symbol.
#' @param resource Optional Opal resource name.
#' @param symbol Optional server-side handle symbol.
#' @param bins Public number of probability bins in \code{[4,512]}.
#' @param torch_backend Node torch backend selection for neural artifacts. Native
#'   XGBoost validation does not provision Torch.
#' @param verbose Show Flower output.
#' @param silent Suppress progress messages.
#' @param allow_insecure_http Exact connection names allowed to use HTTP.
#' @return A \code{dsflower_validation}. Its \code{available} field is false
#'   and \code{metrics} is null when the complete pooled release was not
#'   available; no exact or zero-filled substitute metrics are returned.
#' @export
ds.flower.validate <- function(conns, model, target, data = NULL,
                               resource = NULL, symbol = NULL, bins = 32L,
                               torch_backend = "auto", verbose = FALSE,
                               silent = FALSE,
                               allow_insecure_http = getOption(
                                 "dsflower.dsi_allow_insecure_http", character())) {
  torch_backend <- .validate_torch_backend(torch_backend)
  contract <- .resolve_validation_contract(model, bins)
  .validate_validation_artifact_preflight(contract)
  expected_targets <- if (identical(contract$task, "multilabel")) {
    contract$n_labels
  } else 1L
  if (!is.character(target) || length(target) != expected_targets || anyNA(target) ||
      any(!nzchar(target)) || anyDuplicated(target)) {
    stop("'target' must contain exactly ", expected_targets,
         " unique, non-empty column name(s).", call. = FALSE)
  }
  target <- enc2utf8(target)
  .require_flwr_cli()
  suppressWarnings(.validate_dsi_transport_security(
    conns, allow_insecure_http = allow_insecure_http))
  if (is.null(data) && is.null(resource) && is.null(symbol)) symbol <- "D"

  old_opt <- options(dsflower.silent = isTRUE(silent))
  on.exit(options(old_opt), add = TRUE)
  flower <- ds.flower.connect(
    conns, data = data, resource = resource, symbol = symbol)
  conns <- flower$conns
  hsym <- flower$symbol
  on.exit({
    tryCatch(ds.flower.link.down(conns), error = function(e) NULL)
    tryCatch(ds.flower.nodes.cleanup(conns, hsym), error = function(e) NULL)
    tryCatch(ds.flower.disconnect(flower), error = function(e) NULL)
  }, add = TRUE)
  capabilities <- .assert_runner_compatibility(conns)
  privacy_unit <- .validation_common_privacy_unit(capabilities)

  task_type <- if (contract$task %in% c("regression", "count")) {
    contract$task
  } else "classification"
  prepare <- list(
    "dp-track" = "validation",
    "validation-model-track" = contract$track,
    "validation-task" = contract$task,
    "validation-bins" = contract$bins,
    "task-type" = task_type,
    "loss-name" = contract$loss_name,
    "num-server-rounds" = 1L,
    "num-features" = length(contract$features),
    "num-classes" = contract$n_classes,
    "num-labels" = contract$n_labels)
  if (identical(contract$track, "native_tree")) {
    prepare[["validation-native-tree-request-b64"]] <-
      contract$native_tree_request_b64
    prepare[["validation-native-tree-request-sha256"]] <-
      contract$native_tree_request_sha256
    prepare[["validation-artifact-format"]] <- contract$artifact_format
    prepare[["validation-artifact-sha256"]] <- contract$artifact_sha256
    prepare[["validation-artifact-size-bytes"]] <-
      contract$artifact_size_bytes
    prepare[["validation-profile-sha256"]] <-
      contract$prediction_profile$sha256
    prepare[["validation-profile-size-bytes"]] <-
      contract$prediction_profile$size_bytes
    prepare[["validation-public-schema-sha256"]] <-
      contract$public_schema_sha256
  } else {
    prepare[["model-spec-b64"]] <- .spec_to_b64(contract$model_spec)
  }
  if (!is.null(contract$feature_bounds)) {
    prepare[["feature-bounds"]] <- contract$feature_bounds
  }
  if (!is.null(contract$target_bounds)) {
    prepare[["target-bounds"]] <- contract$target_bounds
  }
  if (!is.null(contract$target_levels) && length(contract$target_levels)) {
    prepare[["target-levels"]] <- contract$target_levels
  }
  contract_sha256 <- .validation_contract_sha256(
    prepare, contract$features, target, privacy_unit)
  prepare[["validation-contract-sha256"]] <- contract_sha256
  ds.flower.nodes.prepare(
    conns, hsym, target_column = target,
    feature_columns = contract$features, run_config = prepare)

  results_dir <- tempfile(
    pattern = "validation_", tmpdir = file.path(tempdir(), "dsflower_results"))
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  config <- c(
    .toml_kv("dp-track", "validation"),
    .toml_kv("validation-model-track", contract$track),
    .toml_kv("validation-task", contract$task),
    .toml_kv("validation-contract-sha256", contract_sha256),
    paste0("validation-bins = ", contract$bins),
    paste0("num-features = ", length(contract$features)),
    paste0("num-classes = ", contract$n_classes),
    paste0("num-labels = ", contract$n_labels),
    "num-server-rounds = 1",
    paste0("min-train-nodes = ", length(conns)),
    .toml_kv("loss-name", contract$loss_name),
    .toml_kv("results-dir", results_dir),
    .toml_kv("validation-model-path-b64",
             .validation_model_path_b64(contract$artifact)))
  if (identical(contract$track, "native_tree")) {
    config <- c(config,
      .toml_kv("validation-profile-path-b64",
               .validation_model_path_b64(
                 contract$prediction_profile$path)),
      .toml_kv("validation-native-tree-request-b64",
               contract$native_tree_request_b64),
      .toml_kv("validation-native-tree-request-sha256",
               contract$native_tree_request_sha256),
      .toml_kv("validation-artifact-format", contract$artifact_format),
      .toml_kv("validation-artifact-sha256", contract$artifact_sha256),
      paste0("validation-artifact-size-bytes = ",
             contract$artifact_size_bytes),
      .toml_kv("validation-profile-sha256",
               contract$prediction_profile$sha256),
      paste0("validation-profile-size-bytes = ",
             contract$prediction_profile$size_bytes),
      .toml_kv("validation-public-schema-sha256",
               contract$public_schema_sha256))
  } else {
    config <- c(config,
      .toml_kv("model-spec-b64", .spec_to_b64(contract$model_spec)))
  }
  if (!is.null(contract$target_bounds)) {
    config <- c(config,
      paste0("validation-target-lower = ", contract$target_bounds$lower),
      paste0("validation-target-upper = ", contract$target_bounds$upper))
  }
  app_dir <- .build_submission_app(
    list(pkg_dir = NULL,
         track = if (identical(contract$track, "native_tree")) {
           "native_tree_validation"
         } else {
           "validation"
         }),
    config, results_dir, vision = FALSE)
  if (!identical(contract$track, "native_tree")) {
    .ensure_client_framework("pytorch")
  }
  ds.flower.link.up(conns, allow_insecure_http = allow_insecure_http)
  ds.flower.nodes.ensure(
    conns, hsym,
    torch_backend = if (identical(contract$track, "native_tree")) {
      NULL
    } else {
      torch_backend
    })

  superlink <- .dsflower_client_env$.superlink
  result <- .run_flwr_with_artifact_watchdog(
    command = .client_flwr_cmd(),
    args = c("run", app_dir, "dsflower", "--stream"),
    env = .client_venv_env(extra = c(
      FLWR_HOME = superlink$flwr_home, PYTHONUNBUFFERED = "1")),
    results_dir = results_dir, num_rounds = 1L, expect_artifacts = TRUE)
  stdout <- gsub("\033\\[[0-9;]*m", "", result$stdout)
  stderr <- gsub("\033\\[[0-9;]*m", "", result$stderr)
  if (isTRUE(verbose) && nzchar(stdout)) message(stdout)
  if (!identical(as.integer(result$status), 0L)) {
    stop("Federated private validation failed (status ", result$status, ").",
         call. = FALSE)
  }
  released <- .read_private_validation_result(results_dir)
  if (is.null(released)) {
    stop("Federated private validation produced no pooled DP result.",
         call. = FALSE)
  }
  structure(list(
    available = released$available, metrics = released$metrics,
    task = released$task,
    privacy = released$privacy, pooled_only = TRUE,
    n_nodes = released$n_nodes, bins = contract$bins,
    model_dir = contract$model_dir, results_dir = results_dir,
    stdout = stdout, stderr = stderr),
    class = "dsflower_validation")
}

#' Print a private dsFlower validation result
#' @param x A \code{dsflower_validation} object.
#' @param ... Ignored.
#' @export
print.dsflower_validation <- function(x, ...) {
  cat("Federated private model validation\n")
  cat("  Task:    ", x$task, "\n")
  cat("  Sites:   ", x$n_nodes, "\n")
  cat("  Privacy: node-DP, pooled metrics only\n")
  if (!isTRUE(x$available)) {
    cat("  Available: no complete private metric release\n")
    return(invisible(x))
  }
  scalar <- x$metrics[vapply(x$metrics, function(value) {
    is.null(value) || (is.atomic(value) && length(value) == 1L)
  }, logical(1))]
  if (length(scalar)) {
    cat("  Metrics:\n")
    for (name in names(scalar)) {
      value <- scalar[[name]]
      cat("    ", name, ": ",
          if (is.null(value)) "NA" else format(value, digits = 5), "\n",
          sep = "")
    }
  }
  invisible(x)
}
