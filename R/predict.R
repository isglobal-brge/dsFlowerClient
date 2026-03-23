# Module: Prediction
# Apply trained federated models to new data via Python (native format).

#' Predict with a federated model
#'
#' Uses the saved model in native format (joblib/pt/xgb) to generate
#' predictions via Python. The appropriate framework dependencies are
#' installed on-demand in the client venv if not already present.
#'
#' @param model A \code{dsflower_run} object, a saved model list (from
#'   \code{ds.flower.load_model}), or a path to a model directory.
#' @param newdata A data.frame or matrix with feature columns.
#' @param type Character; \code{"response"} for predicted class (default),
#'   \code{"prob"} for probabilities.
#' @return A numeric vector of predictions.
#' @export
ds.flower.predict <- function(model, newdata, type = c("response", "prob")) {
  type <- match.arg(type)

  # Resolve model directory and framework
  info <- .resolve_model_for_predict(model)
  model_file <- info$model_file
  framework <- info$framework

  # Ensure framework deps are installed
  .ensure_client_framework(framework)

  # Write data to temp CSV
  tmp_data <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_data), add = TRUE)
  utils::write.csv(as.data.frame(newdata), tmp_data, row.names = FALSE)

  # Find predict helper script
  helper <- system.file("python", "predict_helper.py",
                        package = "dsFlowerClient")
  if (!nzchar(helper)) {
    stop("predict_helper.py not found in dsFlowerClient.", call. = FALSE)
  }

  # Run Python predict
  python <- .client_python_cmd()
  result <- processx::run(
    command = python,
    args = c(helper,
             "--model", model_file,
             "--data", tmp_data,
             "--type", type,
             "--framework", framework),
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
        joblib = "sklearn", pt = "pytorch",
        xgb = "xgboost", json = "xgboost",
        stop("Unknown model format: ", ext, call. = FALSE))
      return(list(model_file = model, framework = framework))
    }
  } else if (is.list(model)) {
    # Loaded model list -- check for source directory
    if (!is.null(model$source_dir)) model_dir <- model$source_dir
    if (is.null(model_dir) && !is.null(model$template)) {
      # Try to find by framework
      fw <- if (grepl("sklearn", model$template)) "sklearn"
            else if (grepl("xgboost", model$template)) "xgboost"
            else "pytorch"
      # Need model_dir to find the file
      stop("Cannot predict from in-memory model list without a model directory. ",
           "Pass the output_dir path instead.", call. = FALSE)
    }
  }

  if (is.null(model_dir) || !dir.exists(model_dir)) {
    stop("Cannot resolve model directory for prediction.", call. = FALSE)
  }

  # Find native model file (priority: joblib > pt > xgb.json > xgb)
  candidates <- list(
    list(file = "model.joblib", framework = "sklearn"),
    list(file = "model.pt", framework = "pytorch"),
    list(file = "model.xgb.json", framework = "xgboost"),
    list(file = "model.xgb", framework = "xgboost")
  )

  for (c in candidates) {
    path <- file.path(model_dir, c$file)
    if (file.exists(path)) {
      return(list(model_file = path, framework = c$framework))
    }
  }

  # Fallback: try global_model.json with pytorch
  json_path <- file.path(model_dir, "global_model.json")
  if (file.exists(json_path)) {
    # Detect framework from metadata
    meta_path <- file.path(model_dir, "metadata.json")
    if (file.exists(meta_path)) {
      meta <- jsonlite::fromJSON(meta_path)
      if (grepl("sklearn", meta$template %||% "")) {
        return(list(model_file = json_path, framework = "sklearn"))
      }
    }
    return(list(model_file = json_path, framework = "pytorch"))
  }

  stop("No native model file found in ", model_dir,
       ". Expected model.joblib, model.pt, or model.xgb.json.", call. = FALSE)
}
