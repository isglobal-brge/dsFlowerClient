# Module: Prediction
# Apply trained federated models to new data via trusted data-only predictors.

.XGBOOST_LOCAL_PREDICTION_CONTRACT <-
  "dsflower-xgboost-local-prediction-v1"
.XGBOOST_LOCAL_MAX_ROWS <- 5000000L
.XGBOOST_LOCAL_MAX_INPUT_BYTES <- 256 * 1024^2
.XGBOOST_LOCAL_MAX_OUTPUT_BYTES <- 128 * 1024^2

#' Predict with a federated model
#'
#' Uses a saved declarative PyTorch state dictionary or a sanitized native
#' XGBoost ensemble to generate tabular predictions. XGBoost runs through the
#' bundled standard-library-only predictor: the artifact, canonical request and
#' public sidecar are revalidated before execution, and no XGBoost library,
#' pickle or executable model payload is loaded. Vision artifacts are not
#' accepted by this tabular predictor.
#'
#' @param model A \code{dsflower_run} object, a saved model list (from
#'   \code{ds.flower.load_model}), or a path to a model directory.
#' @param newdata A data.frame or matrix with feature columns.
#' @param type Character; \code{"response"} returns a predicted class for
#'   classification models and a continuous response for regression/count
#'   models. \code{"prob"} returns probabilities for classification models.
#' @return A numeric vector, integer class vector, or probability matrix for
#'   multiclass, ordinal, and multilabel models.
#' @export
ds.flower.predict <- function(model, newdata, type = c("response", "prob")) {
  type <- match.arg(type)

  # Resolve model directory and framework
  info <- .resolve_model_for_predict(model)
  model_file <- info$model_file
  framework <- info$framework
  contract <- info$contract %||% list()
  if (!is.character(contract$data_kind) || length(contract$data_kind) != 1L ||
      is.na(contract$data_kind) ||
      !contract$data_kind %in% c("tabular", "image")) {
    stop("Saved model metadata must declare data_kind as 'tabular' or 'image'.",
         call. = FALSE)
  }
  if (identical(contract$data_kind, "image")) {
    stop("ds.flower.predict() accepts tabular artifacts only; image models ",
         "require an image/backbone inference pipeline.", call. = FALSE)
  }
  if (identical(framework, "xgboost")) {
    return(.predict_xgboost_local(info, newdata, type))
  }
  if (!is.list(contract$model_spec) || !length(contract$model_spec) ||
      !is.character(contract$loss_name) || length(contract$loss_name) != 1L ||
      is.na(contract$loss_name) || !nzchar(contract$loss_name)) {
    stop("Local prediction requires saved declarative 'model_spec' and ",
         "'loss_name' metadata; Tier-2 and other no-spec artifacts must use ",
         "federated validation.", call. = FALSE)
  }

  # Ensure only the dependencies required by this artifact kind are installed.
  .ensure_client_framework(framework)

  # Align newdata to the TRAINING feature order/names: the model consumes columns
  # positionally, so a reordered or partial newdata would otherwise predict silently wrong.
  newdata <- as.data.frame(newdata)
  if (!is.null(info$features)) {
    miss <- setdiff(info$features, names(newdata))
    if (length(miss)) {
      stop("newdata is missing training feature column(s): ",
           paste(miss, collapse = ", "), call. = FALSE)
    }
    newdata <- newdata[, info$features, drop = FALSE]
  }

  # Write data to temp CSV
  tmp_data <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_data), add = TRUE)
  utils::write.csv(newdata, tmp_data, row.names = FALSE)

  # Find predict helper script
  helper <- system.file("python", "predict_helper.py",
                        package = "dsFlowerClient")
  if (!nzchar(helper)) {
    stop("predict_helper.py not found in dsFlowerClient.", call. = FALSE)
  }

  # Run Python predict
  python <- .client_python_cmd()
  spec_b64 <- .spec_to_b64(contract$model_spec)
  result <- processx::run(
    command = python,
    args = c(helper,
             "--model", model_file,
             "--data", tmp_data,
             "--type", type,
             "--framework", framework,
             "--spec-b64", spec_b64,
             "--loss-name", contract$loss_name,
             "--num-classes", as.character(contract$num_classes %||% 2L),
             "--num-labels", as.character(contract$num_labels %||% 2L),
             # Repeat the node's public clip + affine transform when configured.
             if (identical(framework, "pytorch") && !is.null(info$bounds))
               c("--bounds-b64", .spec_to_b64(info$bounds))),
    env = .client_venv_env(),
    error_on_status = FALSE
  )

  if (result$status != 0L) {
    stop("Prediction failed:\n", result$stderr, call. = FALSE)
  }

  jsonlite::fromJSON(result$stdout)
}

.xgboost_prediction_frame <- function(newdata, features) {
  if (!(is.data.frame(newdata) || is.matrix(newdata))) {
    stop("XGBoost newdata must be a data.frame or matrix.", call. = FALSE)
  }
  frame <- as.data.frame(
    newdata, stringsAsFactors = FALSE, check.names = FALSE)
  columns <- names(frame)
  if (is.null(columns) || anyNA(columns) || any(!nzchar(columns)) ||
      anyDuplicated(columns)) {
    stop("XGBoost newdata must have unique, non-empty feature names.",
         call. = FALSE)
  }
  missing <- setdiff(features, columns)
  extra <- setdiff(columns, features)
  if (length(missing) || length(extra) || length(columns) != length(features)) {
    details <- c(
      if (length(missing)) paste0("missing: ", paste(missing, collapse = ", ")),
      if (length(extra)) paste0("extra: ", paste(extra, collapse = ", ")))
    stop("XGBoost newdata columns must match the saved feature contract exactly",
         if (length(details)) paste0(" (", paste(details, collapse = "; "), ")"),
         ".", call. = FALSE)
  }
  if (nrow(frame) > .XGBOOST_LOCAL_MAX_ROWS) {
    stop("XGBoost newdata exceeds the local prediction row ceiling.",
         call. = FALSE)
  }
  frame <- frame[, features, drop = FALSE]
  valid <- vapply(frame, function(column) {
    is.numeric(column) || is.logical(column)
  }, logical(1))
  if (!all(valid)) {
    stop("XGBoost newdata features must be numeric or logical.",
         call. = FALSE)
  }
  frame[] <- lapply(frame, as.numeric)
  frame
}

.xgboost_predict_helper <- function() {
  helper <- system.file(
    "python", "xgboost_predict_helper.py", package = "dsFlowerClient")
  if (!nzchar(helper)) {
    helper <- file.path("inst", "python", "xgboost_predict_helper.py")
  }
  if (!file.exists(helper) || dir.exists(helper)) {
    stop("Bundled XGBoost prediction helper is unavailable.", call. = FALSE)
  }
  normalizePath(helper, winslash = "/", mustWork = TRUE)
}

.read_xgboost_prediction_output <- function(path, expected_rows, task, type) {
  info <- file.info(path)
  if (!file.exists(path) || is.na(info$isdir) || isTRUE(info$isdir) ||
      is.na(info$size) || info$size < 1 ||
      info$size > .XGBOOST_LOCAL_MAX_OUTPUT_BYTES) {
    stop("Local XGBoost predictor produced no bounded data-only output.",
         call. = FALSE)
  }
  bytes <- readBin(path, what = "raw", n = as.integer(info$size))
  if (length(bytes) != info$size || any(as.integer(bytes) > 127L)) {
    stop("Local XGBoost predictor output is malformed.", call. = FALSE)
  }
  value <- tryCatch(
    jsonlite::fromJSON(rawToChar(bytes), simplifyVector = TRUE),
    error = function(e) NULL)
  expected_fields <- c("contract", "predictions", "task", "type", "version")
  if (!is.list(value) || is.null(names(value)) || anyDuplicated(names(value)) ||
      !identical(sort(names(value)), expected_fields) ||
      !identical(value$contract, .XGBOOST_LOCAL_PREDICTION_CONTRACT) ||
      !is.integer(value$version) || !identical(value$version, 1L) ||
      !identical(value$task, task) || !identical(value$type, type) ||
      !(is.numeric(value$predictions) ||
        (is.list(value$predictions) && !length(value$predictions))) ||
      length(value$predictions) != expected_rows) {
    stop("Local XGBoost predictor output violates its data-only contract.",
         call. = FALSE)
  }
  predictions <- suppressWarnings(as.numeric(value$predictions))
  if (length(predictions) != expected_rows || any(!is.finite(predictions))) {
    stop("Local XGBoost predictor returned invalid predictions.",
         call. = FALSE)
  }
  if (identical(task, "binary") &&
      any(predictions < 0 | predictions > 1)) {
    stop("Local XGBoost predictor returned invalid probabilities.",
         call. = FALSE)
  }
  predictions
}

.run_xgboost_local_predict <- function(native_contract, frame, type) {
  data_path <- tempfile(fileext = ".csv")
  output_path <- tempfile(fileext = ".json")
  on.exit(unlink(c(data_path, output_path, paste0(output_path, ".tmp"))),
          add = TRUE)
  utils::write.csv(
    frame, data_path, row.names = FALSE, na = "NA",
    fileEncoding = "UTF-8")
  Sys.chmod(data_path, "0600")
  input_info <- file.info(data_path)
  if (!file.exists(data_path) || is.na(input_info$isdir) ||
      isTRUE(input_info$isdir) || is.na(input_info$size) ||
      input_info$size > .XGBOOST_LOCAL_MAX_INPUT_BYTES) {
    stop("XGBoost newdata exceeds the local transport byte ceiling.",
         call. = FALSE)
  }
  profile <- file.path(
    native_contract$model_dir, .XGBOOST_ENSEMBLE_PROFILE_FILE)
  result <- processx::run(
    command = .client_python_cmd(),
    args = c(
      "-I", "-S", .xgboost_predict_helper(),
      "--model", native_contract$artifact,
      "--profile", profile,
      "--data", data_path,
      "--output", output_path,
      "--type", type,
      "--expected-rows", as.character(nrow(frame))),
    env = .client_venv_env(), error_on_status = FALSE, timeout = 900)
  if (!identical(as.integer(result$status), 0L) ||
      !identical(result$stdout, "ok")) {
    stop("Local XGBoost prediction rejected the saved bundle or input.",
         call. = FALSE)
  }
  .read_xgboost_prediction_output(
    output_path, nrow(frame), native_contract$task, type)
}

.predict_xgboost_local <- function(info, newdata, type) {
  native <- info$native_contract
  if (!is.list(native) || !identical(native$engine, "xgboost") ||
      !native$task %in% c("binary", "regression") ||
      !is.character(native$features) || !length(native$features)) {
    stop("Saved XGBoost prediction contract is unavailable.", call. = FALSE)
  }
  if (identical(native$task, "regression") && identical(type, "prob")) {
    stop("type = 'prob' is unavailable for XGBoost regression.",
         call. = FALSE)
  }
  frame <- .xgboost_prediction_frame(newdata, native$features)
  predictions <- .run_xgboost_local_predict(native, frame, type)
  if (identical(native$task, "binary") && identical(type, "response")) {
    levels <- native$target_levels
    if (length(levels) != 2L || anyNA(levels) || anyDuplicated(levels)) {
      stop("Saved XGBoost target levels are invalid.", call. = FALSE)
    }
    return(unname(levels[ifelse(predictions >= 0.5, 2L, 1L)]))
  }
  unname(predictions)
}

#' Resolve model info for prediction
#'
#' Finds the native model file and determines the framework.
#'
#' @param model dsflower_run, list, or path.
#' @return List with model_file and framework.
#' @keywords internal
.resolve_model_for_predict <- function(model) {
  # Determine model directory
  model_dir <- NULL
  if (inherits(model, "dsflower_run")) {
    model_dir <- model$output_dir
  } else if (is.character(model) && length(model) == 1) {
    if (dir.exists(model)) {
      model_dir <- model
    } else if (file.exists(model)) {
      # Direct file path
      framework <- if (identical(basename(model), .XGBOOST_ENSEMBLE_FILE)) {
        "xgboost"
      } else {
        ext <- tolower(tools::file_ext(model))
        switch(ext, pt = "pytorch",
               stop("Unknown model format: ", ext, call. = FALSE))
      }
      if (identical(framework, "xgboost")) {
        native <- .resolve_validation_contract(dirname(model), 32L)
        return(list(
          model_file = native$artifact, framework = framework,
          features = native$features, bounds = native$feature_bounds,
          contract = list(data_kind = native$data_kind),
          native_contract = native))
      }
      return(list(model_file = model, framework = framework,
                  bounds = .read_meta_bounds(dirname(model)),
                  contract = .read_meta_model_contract(dirname(model))))
    }
  } else if (is.list(model)) {
    # Loaded model list -- check for source directory
    if (!is.null(model$source_dir)) model_dir <- model$source_dir
    if (is.null(model_dir)) {
      # Need model_dir to find the file
      stop("Cannot predict from in-memory model list without a model directory. ",
           "Pass the output_dir path instead.", call. = FALSE)
    }
  }

  if (is.null(model_dir) || !dir.exists(model_dir)) {
    stop("Cannot resolve model directory for prediction.", call. = FALSE)
  }

  feats <- .read_meta_features(model_dir)   # training feature order, to align newdata
  bnd <- .read_meta_bounds(model_dir)        # public clipped-affine bounds (or NULL)
  contract <- .read_meta_model_contract(model_dir)

  candidates <- list(
    list(file = .XGBOOST_ENSEMBLE_FILE, framework = "xgboost"),
    list(file = "model.pt", framework = "pytorch"))

  for (c in candidates) {
    path <- file.path(model_dir, c$file)
    if (file.exists(path)) {
      if (identical(c$framework, "xgboost")) {
        native <- .resolve_validation_contract(model_dir, 32L)
        return(list(
          model_file = native$artifact, framework = c$framework,
          features = native$features, bounds = native$feature_bounds,
          contract = list(data_kind = native$data_kind),
          native_contract = native))
      }
      return(list(model_file = path, framework = c$framework,
                  features = feats, bounds = bnd, contract = contract))
    }
  }

  stop("No native model file found in ", model_dir,
       ". Expected model.pt or ", .XGBOOST_ENSEMBLE_FILE, ".",
       call. = FALSE)
}

#' Read the training feature names (in order) from a model dir's metadata.json
#' @keywords internal
.read_meta_features <- function(model_dir) {
  if (is.null(model_dir) || !nzchar(model_dir)) return(NULL)
  meta_path <- file.path(model_dir, "metadata.json")
  if (!file.exists(meta_path)) return(NULL)
  meta <- tryCatch(jsonlite::fromJSON(meta_path), error = function(e) NULL)
  f <- meta$features
  if (is.null(f) || length(f) == 0L) NULL else as.character(f)
}

#' Read the public, data-only model contract required for exact reconstruction
#' @keywords internal
.read_meta_model_contract <- function(model_dir) {
  empty <- list(model_spec = NULL, loss_name = NULL,
                num_classes = 2L, num_labels = 2L, data_kind = NULL)
  if (is.null(model_dir) || !nzchar(model_dir)) return(empty)
  meta_path <- file.path(model_dir, "metadata.json")
  if (!file.exists(meta_path)) return(empty)
  meta <- tryCatch(
    jsonlite::fromJSON(meta_path, simplifyVector = FALSE),
    error = function(e) NULL)
  if (!is.list(meta)) return(empty)
  params <- meta$model_params
  if (!is.list(params)) params <- list()
  scalar_int <- function(value, fallback) {
    value <- suppressWarnings(as.numeric(unlist(value, use.names = FALSE)))
    if (length(value) != 1L || !is.finite(value) || value < 1 ||
        value != floor(value) || value > 1024) return(as.integer(fallback))
    as.integer(value)
  }
  n_classes <- scalar_int(
    params$n_classes %||% params$num_classes %||% length(meta$target_levels %||% list()),
    2L)
  n_labels <- scalar_int(params$num_labels, 2L)
  loss <- meta$loss_name
  if (is.null(loss) || length(loss) != 1L || is.na(loss) || !nzchar(loss)) loss <- NULL
  data_kind <- meta$data_kind
  if (!is.character(data_kind) || length(data_kind) != 1L ||
      is.na(data_kind) || !data_kind %in% c("tabular", "image")) {
    data_kind <- NULL
  }
  list(model_spec = if (is.list(meta$model_spec)) meta$model_spec else NULL,
       loss_name = loss, num_classes = n_classes, num_labels = n_labels,
       data_kind = data_kind)
}

#' Read public feature bounds from a model directory's metadata.json
#'
#' Returns list(lower, upper) only when both vectors are finite, aligned, and
#' strictly ordered. Models without public bounds return NULL.
#' @keywords internal
.read_meta_bounds <- function(model_dir) {
  if (is.null(model_dir) || !nzchar(model_dir)) return(NULL)
  meta_path <- file.path(model_dir, "metadata.json")
  if (!file.exists(meta_path)) return(NULL)
  meta <- tryCatch(jsonlite::fromJSON(meta_path), error = function(e) NULL)
  lower <- meta$feature_lower; upper <- meta$feature_upper
  if (is.null(lower) || is.null(upper) || length(lower) == 0L ||
      length(lower) != length(upper)) return(NULL)
  lower <- as.numeric(lower); upper <- as.numeric(upper)
  if (any(!is.finite(lower)) || any(!is.finite(upper)) || any(lower >= upper))
    return(NULL)
  list(lower = lower, upper = upper)
}
