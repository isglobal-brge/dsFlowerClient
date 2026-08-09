# Module: Prediction
# Apply trained federated models to new data via Python (native format).

#' Predict with a federated model
#'
#' Uses the saved declarative PyTorch state dictionary or dsFlower DP-GBDT JSON
#' artifact to generate tabular predictions via Python. Vision artifacts are not
#' accepted by this tabular predictor. The appropriate framework dependencies
#' are installed on-demand in the client venv if not already present.
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
  template <- info$template %||% ""
  contract <- info$contract %||% list()
  if (identical(contract$data_kind %||% "tabular", "image")) {
    stop("ds.flower.predict() accepts tabular artifacts only; image models ",
         "require an image/backbone inference pipeline.", call. = FALSE)
  }

  # Ensure framework deps are installed
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
  spec_b64 <- if (identical(framework, "pytorch") &&
                  is.list(contract$model_spec) && length(contract$model_spec)) {
    .spec_to_b64(contract$model_spec)
  } else NULL
  result <- processx::run(
    command = python,
    args = c(helper,
             "--model", model_file,
             "--data", tmp_data,
             "--type", type,
             "--framework", framework,
             if (nzchar(template)) c("--template", template),
             if (!is.null(spec_b64)) c("--spec-b64", spec_b64),
             if (!is.null(spec_b64) && nzchar(contract$loss_name %||% ""))
               c("--loss-name", as.character(contract$loss_name)),
             if (!is.null(spec_b64))
               c("--num-classes", as.character(contract$num_classes %||% 2L)),
             if (!is.null(spec_b64))
               c("--num-labels", as.character(contract$num_labels %||% 2L)),
             if (identical(framework, "xgboost") && !is.null(info$bounds))
               c("--tree-bounds-b64", .spec_to_b64(info$bounds)),
             # Repeat the node's public clip + affine transform for new models.
             # Legacy models carry only mean/SD and retain their old prediction path.
             if (identical(framework, "pytorch") && !is.null(info$bounds))
               c("--bounds-b64", .spec_to_b64(info$bounds))
             else if (identical(framework, "pytorch") && !is.null(info$norm))
               c("--norm-b64", .spec_to_b64(list(means = info$norm$means,
                                                  sds = info$norm$sds)))),
    env = .client_venv_env(),
    error_on_status = FALSE
  )

  if (result$status != 0L) {
    stop("Prediction failed:\n", result$stderr, call. = FALSE)
  }

  jsonlite::fromJSON(result$stdout)
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
      ext <- tolower(tools::file_ext(model))
      framework <- switch(ext,
        pt = "pytorch", xgb = "xgboost", json = "xgboost",
        stop("Unknown model format: ", ext, call. = FALSE))
      tmpl <- .read_template_meta(dirname(model))
      return(list(model_file = model, framework = framework, template = tmpl,
                  bounds = .read_meta_bounds(dirname(model)),
                  norm = .read_meta_norm(dirname(model)),
                  contract = .read_meta_model_contract(dirname(model))))
    }
  } else if (is.list(model)) {
    # Loaded model list -- check for source directory
    if (!is.null(model$source_dir)) model_dir <- model$source_dir
    if (is.null(model_dir) && !is.null(model$template)) {
      # Try to find by framework
      fw <- if (grepl("xgboost", model$template)) "xgboost" else "pytorch"
      # Need model_dir to find the file
      stop("Cannot predict from in-memory model list without a model directory. ",
           "Pass the output_dir path instead.", call. = FALSE)
    }
  }

  if (is.null(model_dir) || !dir.exists(model_dir)) {
    stop("Cannot resolve model directory for prediction.", call. = FALSE)
  }

  # Template (from metadata.json) drives modality-aware output semantics in the
  # Python predictor (regression vs classification vs survival vs multilabel ...).
  tmpl <- .read_template_meta(model_dir)
  feats <- .read_meta_features(model_dir)   # training feature order, to align newdata
  bnd <- .read_meta_bounds(model_dir)        # public clipped-affine bounds (or NULL)
  nrm <- .read_meta_norm(model_dir)          # legacy standardization stats (or NULL)
  contract <- .read_meta_model_contract(model_dir)

  # Find native model file (priority: pt > xgb.json > booster.json > xgb). The trees runner
  # writes the XGBoost model as booster.json.
  candidates <- list(
    list(file = "model.pt", framework = "pytorch"),
    list(file = "model.xgb.json", framework = "xgboost"),
    list(file = "booster.json", framework = "xgboost"),
    list(file = "model.xgb", framework = "xgboost")
  )

  for (c in candidates) {
    path <- file.path(model_dir, c$file)
    if (file.exists(path)) {
      return(list(model_file = path, framework = c$framework, template = tmpl,
                  features = feats, bounds = bnd, norm = nrm,
                  contract = contract))
    }
  }

  # Fallback: route global_model.json by the template's framework (metadata).
  json_path <- file.path(model_dir, "global_model.json")
  if (file.exists(json_path)) {
    if (grepl("xgboost", tmpl)) {
      return(list(model_file = json_path, framework = "xgboost", template = tmpl,
                  features = feats, bounds = bnd, norm = nrm,
                  contract = contract))
    }
    return(list(model_file = json_path, framework = "pytorch", template = tmpl,
                features = feats, bounds = bnd, norm = nrm,
                contract = contract))
  }

  stop("No native model file found in ", model_dir,
       ". Expected model.pt or model.xgb.json.", call. = FALSE)
}

#' Read the template name from a model directory's metadata.json
#' @keywords internal
.read_template_meta <- function(model_dir) {
  if (is.null(model_dir) || !nzchar(model_dir)) return("")
  meta_path <- file.path(model_dir, "metadata.json")
  if (!file.exists(meta_path)) return("")
  meta <- tryCatch(jsonlite::fromJSON(meta_path), error = function(e) NULL)
  meta$template %||% meta$model %||% ""
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
                num_classes = 2L, num_labels = 2L, data_kind = "tabular")
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
    template <- as.character(meta$template %||% meta$model %||% "")
    framework <- as.character(meta$framework %||% "")
    data_kind <- if (identical(framework, "pytorch_vision") ||
                         template %in% c("pytorch_resnet18", "pytorch_densenet121")) {
      "image"
    } else {
      "tabular"
    }
  }
  list(model_spec = if (is.list(meta$model_spec)) meta$model_spec else NULL,
       loss_name = loss, num_classes = n_classes, num_labels = n_labels,
       data_kind = data_kind)
}

#' Read public feature bounds from a model directory's metadata.json
#'
#' Returns list(lower, upper) only when both vectors are finite, aligned, and
#' strictly ordered. Missing fields identify a legacy/raw model and return NULL.
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

#' Read the GLOBAL feature standardization stats from a model dir's metadata.json
#'
#' Returns list(means, sds) (numeric, in training feature order) iff the model was
#' trained with standardization, else NULL (model trained on raw features). Used to
#' scale newdata at prediction EXACTLY as the training features were scaled.
#' @keywords internal
.read_meta_norm <- function(model_dir) {
  if (is.null(model_dir) || !nzchar(model_dir)) return(NULL)
  meta_path <- file.path(model_dir, "metadata.json")
  if (!file.exists(meta_path)) return(NULL)
  meta <- tryCatch(jsonlite::fromJSON(meta_path), error = function(e) NULL)
  mu <- meta$feature_means; sd <- meta$feature_sds
  if (is.null(mu) || is.null(sd) || length(mu) == 0L || length(mu) != length(sd))
    return(NULL)
  list(means = as.numeric(mu), sds = as.numeric(sd))
}
