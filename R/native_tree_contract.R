# Module: Native tree manifest contract
# Internal analyst-to-server request ABI only. The node validates and enriches
# it with node-owned privacy and data-scope state for the separate trusted
# Python ABI (including binary -> binary_classification). It is not a public
# model registration or a declaration that a native backend is available.

.NATIVE_TREE_CONTRACT <- "dsflower-native-tree-request-v1"
.NATIVE_TREE_ENGINES <- c(
  "catboost", "extra_trees", "lightgbm", "random_forest", "xgboost")
.NATIVE_TREE_PURE_ENGINES <- c(
  "catboost", "extra_trees", "lightgbm", "random_forest")
.NATIVE_TREE_RELEASE_SPECS <- list(
  xgboost = list(
    artifact_contract = "dsflower-xgboost-ensemble-v1",
    artifact_format = "dsflower-xgboost-ensemble-json-v1",
    artifact_file = "model.xgboost-ensemble.json",
    profile_contract = "dsflower-xgboost-prediction-profile-v1",
    profile_file = "model.xgboost-ensemble.profile.json",
    profile_version = 1L),
  extra_trees = list(
    artifact_contract = "dsflower-forest-ensemble-v1",
    artifact_format = "dsflower-forest-ensemble-json-v1",
    artifact_file = "model.extra-trees-ensemble.json",
    profile_contract = "dsflower-native-tree-prediction-profile-v2",
    profile_file = "model.extra-trees-ensemble.profile.json",
    profile_version = 2L),
  random_forest = list(
    artifact_contract = "dsflower-forest-ensemble-v1",
    artifact_format = "dsflower-forest-ensemble-json-v1",
    artifact_file = "model.random-forest-ensemble.json",
    profile_contract = "dsflower-native-tree-prediction-profile-v2",
    profile_file = "model.random-forest-ensemble.profile.json",
    profile_version = 2L),
  lightgbm = list(
    artifact_contract = "dsflower-lightgbm-safe-ensemble-v1",
    artifact_format = "dsflower-lightgbm-ensemble-json-v1",
    artifact_file = "model.lightgbm-ensemble.json",
    profile_contract = "dsflower-native-tree-prediction-profile-v2",
    profile_file = "model.lightgbm-ensemble.profile.json",
    profile_version = 2L),
  catboost = list(
    artifact_contract = "dsflower-catboost-safe-ensemble-v1",
    artifact_format = "dsflower-catboost-ensemble-json-v1",
    artifact_file = "model.catboost-ensemble.json",
    profile_contract = "dsflower-native-tree-prediction-profile-v2",
    profile_file = "model.catboost-ensemble.profile.json",
    profile_version = 2L))
.NATIVE_TREE_MODES <- "native-tight"
.NATIVE_TREE_TASKS <- c("binary", "regression")
.NATIVE_TREE_PARAMETER_TYPES <- c(
  "boolean", "integer", "number", "string",
  "boolean_array", "integer_array", "number_array", "string_array")
.NATIVE_TREE_TIGHT_FORBIDDEN <- c(
  "custom_eval", "custom_metric", "early_stop", "early_stop_rounds",
  "early_stopping", "early_stopping_round", "early_stopping_rounds", "eval",
  "eval_fn", "eval_metric", "eval_set", "evals", "evaluation", "feval",
  "fobj", "loss_function", "metric", "metrics", "obj", "objective",
  "objective_fn", "random_seed", "random_state", "seed", "tree_method",
  "use_best_model")
.NATIVE_TREE_RESOURCE_DEFAULTS <- list(
  max_features = 4096L,
  max_trees = 4096L,
  max_depth = 32L,
  max_bins = 4096L,
  max_threads = 32L,
  memory_mb = 32768L,
  timeout_seconds = 21600L)
.NATIVE_TREE_RESOURCE_LIMITS <- list(
  max_features = 8192L,
  max_trees = 10000L,
  max_depth = 32L,
  max_bins = 65536L,
  max_threads = 64L,
  memory_mb = 65536L,
  timeout_seconds = 21600L)
.NATIVE_TREE_MAX_FLOAT_ABS <- 1e12
.NATIVE_TREE_MAX_TOTAL_CUTS <- 16384L
.NATIVE_TREE_MANIFEST_MAX_BYTES <- 65536L
.NATIVE_TREE_XGBOOST_MAX_DEPTH <- 30L
.NATIVE_TREE_XGBOOST_REQUIRED_PARAMETERS <- c(
  "learning_rate", "max_delta_step", "max_depth", "min_child_weight",
  "min_split_loss", "num_boost_round", "reg_alpha", "reg_lambda")
.NATIVE_TREE_XGBOOST_OPTIONAL_PARAMETERS <- character()
.NATIVE_TREE_XGBOOST_PARAMETER_TYPES <- c(
  learning_rate = "number",
  max_delta_step = "number",
  max_depth = "integer",
  min_child_weight = "number",
  min_split_loss = "number",
  num_boost_round = "integer",
  reg_alpha = "number",
  reg_lambda = "number")
.NATIVE_TREE_EXTRA_TREES_MAX_DEPTH <- 12L
.NATIVE_TREE_EXTRA_TREES_MAX_TREES <- 512L
.NATIVE_TREE_EXTRA_TREES_REQUIRED_PARAMETERS <- c(
  "max_depth", "n_estimators")
.NATIVE_TREE_EXTRA_TREES_PARAMETER_TYPES <- c(
  max_depth = "integer", n_estimators = "integer")
.NATIVE_TREE_RANDOM_FOREST_MAX_DEPTH <- 12L
.NATIVE_TREE_RANDOM_FOREST_MAX_TREES <- 512L
.NATIVE_TREE_RANDOM_FOREST_REQUIRED_PARAMETERS <- c(
  "max_depth", "max_features", "n_estimators")
.NATIVE_TREE_RANDOM_FOREST_PARAMETER_TYPES <- c(
  max_depth = "integer", max_features = "integer", n_estimators = "integer")
.NATIVE_TREE_LIGHTGBM_MAX_DEPTH <- 32L
.NATIVE_TREE_LIGHTGBM_MAX_LEAVES <- 256L
.NATIVE_TREE_LIGHTGBM_REQUIRED_PARAMETERS <- c(
  "lambda_l1", "lambda_l2", "learning_rate", "max_delta_step",
  "max_depth", "min_data_in_leaf", "min_gain_to_split", "num_iterations",
  "num_leaves")
.NATIVE_TREE_LIGHTGBM_PARAMETER_TYPES <- c(
  lambda_l1 = "number", lambda_l2 = "number", learning_rate = "number",
  max_delta_step = "number", max_depth = "integer",
  min_data_in_leaf = "integer", min_gain_to_split = "number",
  num_iterations = "integer", num_leaves = "integer")
.NATIVE_TREE_CATBOOST_MAX_DEPTH <- 16L
.NATIVE_TREE_CATBOOST_REQUIRED_PARAMETERS <- c(
  "depth", "iterations", "l2_leaf_reg", "learning_rate", "max_delta_step")
.NATIVE_TREE_CATBOOST_PARAMETER_TYPES <- c(
  depth = "integer", iterations = "integer", l2_leaf_reg = "number",
  learning_rate = "number", max_delta_step = "number")

.native_tree_release_spec <- function(engine) {
  engine <- as.character(engine)
  if (length(engine) != 1L || is.na(engine) ||
      !engine %in% names(.NATIVE_TREE_RELEASE_SPECS)) {
    stop("This release has no executable adapter for the requested tree engine.",
         call. = FALSE)
  }
  .NATIVE_TREE_RELEASE_SPECS[[engine]]
}

#' Canonical JSON bytes for the native-tree cross-runtime ABI
#' @keywords internal
.native_tree_json <- function(value) {
  json <- as.character(jsonlite::toJSON(
    value, auto_unbox = TRUE, null = "null", na = "null",
    digits = NA, always_decimal = TRUE, pretty = FALSE))
  # jsonlite protects embedded HTML by escaping a solidus after "<".  The
  # cross-runtime ABI uses Python's JSON canonical form, where that optional
  # escape is absent.
  json <- gsub("<\\\\/", "</", json)
  charToRaw(enc2utf8(json))
}

#' Validate one canonical scalar string
#' @keywords internal
.native_tree_string <- function(value, name, pattern = NULL,
                                max_bytes = 256L) {
  if (is.list(value)) value <- unlist(value, recursive = FALSE, use.names = FALSE)
  if (!is.character(value) || length(value) != 1L || is.na(value)) {
    stop(name, " must be one non-missing string.", call. = FALSE)
  }
  value <- enc2utf8(value)
  if (!nzchar(value) ||
      is.na(iconv(value, from = "UTF-8", to = "UTF-8", sub = NA_character_)) ||
      nchar(value, type = "bytes") > max_bytes ||
      grepl("[[:cntrl:]]", value, perl = TRUE) ||
      (!is.null(pattern) && !grepl(pattern, value, perl = TRUE))) {
    stop(name, " is not a canonical safe string.", call. = FALSE)
  }
  value
}

#' Validate one numeric vector and normalise negative zero
#' @keywords internal
.native_tree_number_vector <- function(value, name, integer = FALSE) {
  if (inherits(value, "AsIs")) value <- unclass(value)
  if (is.list(value)) {
    valid <- vapply(value, function(x) {
      (is.integer(x) || is.numeric(x)) && length(x) == 1L &&
        !is.object(x) && !is.na(x) && is.finite(x)
    }, logical(1))
    if (!all(valid)) stop(name, " must contain only finite numbers.", call. = FALSE)
    value <- unlist(value, use.names = FALSE)
  }
  if (!(is.integer(value) || is.numeric(value)) || !length(value) ||
      is.object(value) || anyNA(value) || any(!is.finite(value))) {
    stop(name, " must be a non-empty finite numeric vector.", call. = FALSE)
  }
  if (any(abs(value) > .NATIVE_TREE_MAX_FLOAT_ABS)) {
    stop(name, " exceeds the supported numeric range.", call. = FALSE)
  }
  if (isTRUE(integer) && any(value != floor(value))) {
    stop(name, " must contain only integers.", call. = FALSE)
  }
  if (isTRUE(integer) && any(abs(value) > .Machine$integer.max)) {
    stop(name, " exceeds the supported integer range.", call. = FALSE)
  }
  value[value == 0] <- 0
  if (isTRUE(integer)) as.integer(value) else as.numeric(value)
}

#' Round finite public values through the cross-runtime float32 ABI
#' @keywords internal
.native_tree_float32 <- function(value, name) {
  bytes <- writeBin(as.numeric(value), raw(), size = 4L, endian = "little")
  result <- readBin(
    bytes, what = numeric(), n = length(value), size = 4L, endian = "little")
  if (length(result) != length(value) || any(!is.finite(result))) {
    stop(name, " must remain finite as float32.", call. = FALSE)
  }
  result[result == 0] <- 0
  result
}

# Canonicalise one type-preserving public classification label.
.native_tree_target_level <- function(record) {
  if (!is.list(record) ||
      !identical(sort(names(record)), c("type", "value"))) {
    stop("Each target level must contain exactly type and value.",
         call. = FALSE)
  }
  type <- .native_tree_string(
    record$type, "target level type", "^(string|boolean|number)$", 16L)
  value <- record$value
  if (is.list(value)) {
    value <- unlist(value, recursive = FALSE, use.names = FALSE)
  }
  if (length(value) != 1L || is.object(value) || anyNA(value)) {
    stop("Each target level must contain one plain non-missing scalar.",
         call. = FALSE)
  }
  if (identical(type, "string")) {
    if (!is.character(value)) {
      stop("string target levels must be character.", call. = FALSE)
    }
    value <- .native_tree_string(
      value, "string target level", max_bytes = 512L)
  } else if (identical(type, "boolean")) {
    if (!is.logical(value)) {
      stop("boolean target levels must be logical.", call. = FALSE)
    }
    value <- as.logical(value)
  } else {
    value <- .native_tree_number_vector(value, "number target level")
    if (length(value) != 1L) {
      stop("number target levels must be scalar.", call. = FALSE)
    }
    value <- value[[1L]]
  }
  list(type = type, value = unname(value))
}

#' Canonicalise the public target schema and typed binary labels
#' @keywords internal
.native_tree_target_schema <- function(target, task, features) {
  if (!is.list(target) || !identical(
      sort(names(target)), c("kind", "levels", "lower", "name", "upper"))) {
    stop("target must contain exactly name, kind, levels, lower and upper.",
         call. = FALSE)
  }
  name <- .native_tree_string(target$name, "target name",
                              "^[^/\\\\]+$", 256L)
  if (name %in% features) {
    stop("target name must differ from every feature name.", call. = FALSE)
  }
  kind <- .native_tree_string(target$kind, "target kind", "^[a-z]+$", 16L)
  lower <- .native_tree_number_vector(target$lower, "target$lower")
  upper <- .native_tree_number_vector(target$upper, "target$upper")
  if (length(lower) != 1L || length(upper) != 1L) {
    stop("target lower and upper must be scalar.", call. = FALSE)
  }
  lower <- lower[[1L]]
  upper <- upper[[1L]]
  if (identical(task, "binary")) {
    if (!identical(kind, "binary") || lower != 0 || upper != 1) {
      stop("binary task requires target kind binary with public bounds 0 and 1.",
           call. = FALSE)
    }
    if (!is.list(target$levels) || length(target$levels) != 2L) {
      stop("binary task requires exactly two ordered tagged target levels.",
           call. = FALSE)
    }
    levels <- lapply(target$levels, .native_tree_target_level)
    identities <- vapply(
      levels, function(value) rawToChar(.native_tree_json(value)), character(1))
    if (anyDuplicated(identities)) {
      stop("binary target levels must be distinct and ordered.", call. = FALSE)
    }
  } else if (!identical(kind, "continuous") || lower >= upper) {
    stop("regression task requires continuous target with lower < upper.",
         call. = FALSE)
  } else {
    if (!is.null(target$levels)) {
      stop("regression target levels must be null.", call. = FALSE)
    }
    levels <- NULL
  }
  list(name = name, kind = kind, levels = levels,
       lower = lower, upper = upper)
}

#' Canonicalise and hash the public feature and target schema
#' @keywords internal
.native_tree_public_schema <- function(features, bounds, cuts, target, task,
                                       schema_sha256) {
  if (inherits(features, "AsIs")) features <- unclass(features)
  if (is.list(features)) {
    valid <- vapply(features, function(x) {
      is.character(x) && length(x) == 1L && !is.na(x)
    }, logical(1))
    if (!all(valid)) stop("features must contain only strings.", call. = FALSE)
    features <- unlist(features, use.names = FALSE)
  }
  if (!is.character(features) || !length(features) || length(features) > 8192L ||
      anyNA(features) || anyDuplicated(features)) {
    stop("features must be a non-empty unique character vector.", call. = FALSE)
  }
  features <- vapply(features, .native_tree_string, character(1),
                     name = "feature name", pattern = "^[^/\\\\]+$",
                     max_bytes = 256L)
  if (!is.list(bounds) || !identical(sort(names(bounds)), c("lower", "upper"))) {
    stop("bounds must contain exactly lower and upper.", call. = FALSE)
  }
  lower <- .native_tree_number_vector(bounds$lower, "bounds$lower")
  upper <- .native_tree_number_vector(bounds$upper, "bounds$upper")
  if (length(lower) != length(features) || length(upper) != length(features) ||
      any(lower >= upper)) {
    stop("bounds must match features and satisfy lower < upper.", call. = FALSE)
  }

  canonical_cuts <- NULL
  if (!is.null(cuts)) {
    if (!is.list(cuts) || length(cuts) != length(features)) {
      stop("cuts must be NULL or one numeric vector per feature.", call. = FALSE)
    }
    canonical_cuts <- vector("list", length(cuts))
    for (i in seq_along(cuts)) {
      if (length(cuts[[i]]) > .NATIVE_TREE_MAX_TOTAL_CUTS) {
        stop("public cuts exceed the 16384-cut contract cap.", call. = FALSE)
      }
      one <- .native_tree_number_vector(cuts[[i]], "feature cuts")
      if (any(diff(one) <= 0) || any(one <= lower[[i]]) ||
          any(one >= upper[[i]])) {
        stop("feature cuts must be strictly increasing inside public bounds.",
             call. = FALSE)
      }
      canonical_cuts[[i]] <- I(one)
    }
    if (sum(vapply(canonical_cuts, length, integer(1))) >
        .NATIVE_TREE_MAX_TOTAL_CUTS) {
      stop("public cuts exceed the 16384-cut contract cap.", call. = FALSE)
    }
  }
  canonical_target <- .native_tree_target_schema(target, task, features)
  core <- list(
    version = 1L,
    features = I(unname(features)),
    lower = I(lower),
    upper = I(upper),
    cuts = canonical_cuts,
    target = canonical_target)
  actual <- digest::digest(.native_tree_json(core), algo = "sha256",
                           serialize = FALSE)
  if (!is.null(schema_sha256) &&
      (!is.character(schema_sha256) || length(schema_sha256) != 1L ||
       is.na(schema_sha256) || !identical(schema_sha256, actual))) {
    stop("public schema SHA-256 does not match its canonical content.",
         call. = FALSE)
  }
  c(core, list(sha256 = actual))
}

#' Whether a parameter name could alter privacy or execute external code
#' @keywords internal
.native_tree_reserved_parameter <- function(name) {
  name %in% c(
    "accountant", "adjacency", "allow_writing_files", "callbacks", "clip_norm",
    "clipping_norm", "config", "custom_objective", "data", "delta",
    "dependencies", "dependency", "device", "devices", "dp", "epsilon",
    "forcedsplits_filename", "gpu_device_id", "input_model", "logging_level",
    "machine_list_file", "machines", "max_rows_per_unit", "model_file", "n_jobs", "noise",
    "noise_multiplier", "noise_root", "noise_seed", "nthread", "num_threads",
    "output_model", "patient_column", "privacy", "requirements", "requirement",
    "sensitivity", "snapshot_file", "task_type", "thread_count", "train_dir",
    "unit", "unit_canonicalization", "updater", "verbose", "verbosity") ||
    startsWith(name, "contribution_") || startsWith(name, "dp_") ||
    startsWith(name, "privacy_") || startsWith(name, "unit_") ||
    grepl("(^|_)custom_objective($|_)", name, perl = TRUE) ||
    grepl("(^|_)(callback|code|host|log|machine|network|plugin|port|script|socket)($|_)",
          name, perl = TRUE) ||
    grepl("(_dir|_file|_filename|_path)$", name, perl = TRUE)
}

#' Canonicalise one explicitly typed public parameter record
#' @keywords internal
.native_tree_parameter_record <- function(record) {
  if (!is.list(record) ||
      !identical(sort(names(record)), c("name", "type", "value"))) {
    stop("Each parameter record must contain exactly name, type and value.",
         call. = FALSE)
  }
  name <- .native_tree_string(
    record$name, "parameter name", "^[a-z][a-z0-9_]{0,63}$", 64L)
  if (.native_tree_reserved_parameter(name)) {
    stop("Parameter '", name, "' is reserved by the server.", call. = FALSE)
  }
  type <- .native_tree_string(record$type, "parameter type",
                              "^[a-z_]+$", 32L)
  if (!type %in% .NATIVE_TREE_PARAMETER_TYPES) {
    stop("Unsupported public parameter type.", call. = FALSE)
  }
  value <- record$value
  array <- endsWith(type, "_array")
  scalar_type <- sub("_array$", "", type)
  if (inherits(value, "AsIs")) value <- unclass(value)
  if (isTRUE(array)) {
    if (length(value) > 256L) {
      stop("Array parameters must contain between 1 and 256 scalars.",
           call. = FALSE)
    }
    if (is.list(value)) value <- unlist(value, recursive = FALSE, use.names = FALSE)
    if (!is.atomic(value) || !length(value) || length(value) > 256L) {
      stop("Array parameters must contain between 1 and 256 scalars.",
           call. = FALSE)
    }
  } else if (is.list(value)) {
    value <- unlist(value, recursive = FALSE, use.names = FALSE)
  }
  expected_length <- if (isTRUE(array)) NULL else 1L
  if (!is.null(expected_length) && length(value) != expected_length) {
    stop("Scalar parameter records must contain one value.", call. = FALSE)
  }
  if (is.object(value) || anyNA(value)) {
    stop("Parameter values must be plain non-missing scalars.", call. = FALSE)
  }
  if (identical(scalar_type, "boolean")) {
    if (!is.logical(value)) stop("boolean parameters must be logical.", call. = FALSE)
    value <- as.logical(value)
  } else if (identical(scalar_type, "integer")) {
    value <- .native_tree_number_vector(value, "integer parameter", integer = TRUE)
  } else if (identical(scalar_type, "number")) {
    value <- .native_tree_number_vector(value, "number parameter")
  } else {
    if (!is.character(value)) stop("string parameters must be character.", call. = FALSE)
    value <- vapply(value, .native_tree_string, character(1),
                    name = "string parameter", max_bytes = 512L)
  }
  if (!isTRUE(array)) value <- value[[1L]]
  list(name = name, type = type,
       value = if (isTRUE(array)) I(unname(value)) else unname(value))
}

#' Canonicalise the typed public parameter array
#' @keywords internal
.native_tree_parameters <- function(parameters, mode) {
  if (!is.list(parameters)) {
    stop("parameters must be an array of typed records.", call. = FALSE)
  }
  if (length(parameters) > 128L) {
    stop("parameters exceeds the 128-parameter contract cap.", call. = FALSE)
  }
  out <- lapply(parameters, .native_tree_parameter_record)
  names <- vapply(out, `[[`, character(1), "name")
  if (anyDuplicated(names)) stop("Parameter names must be unique.", call. = FALSE)
  out <- out[order(names, method = "radix")]
  names <- sort(names, method = "radix")
  forbidden <- intersect(names, .NATIVE_TREE_TIGHT_FORBIDDEN)
  if (length(forbidden)) {
    stop("native-tight forbids callbacks, objectives, evaluation and early stopping: ",
         paste(forbidden, collapse = ", "), ".", call. = FALSE)
  }
  unname(out)
}

#' Enforce the initial native-tight XGBoost parameter profile
#' @keywords internal
.native_tree_xgboost_parameters <- function(parameters, schema, mode) {
  if (!identical(mode, "native-tight")) {
    stop("XGBoost request v1 supports native-tight mode only.", call. = FALSE)
  }
  by_name <- stats::setNames(
    parameters, vapply(parameters, `[[`, character(1), "name"))
  actual <- names(by_name)
  allowed <- c(.NATIVE_TREE_XGBOOST_REQUIRED_PARAMETERS,
               .NATIVE_TREE_XGBOOST_OPTIONAL_PARAMETERS)
  unknown <- setdiff(actual, allowed)
  missing <- setdiff(.NATIVE_TREE_XGBOOST_REQUIRED_PARAMETERS, actual)
  if (length(unknown)) {
    stop("Unsupported XGBoost parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  if (length(missing)) {
    stop("Missing required XGBoost parameter(s): ",
         paste(missing, collapse = ", "), ".", call. = FALSE)
  }
  for (name in actual) {
    if (!identical(by_name[[name]]$type,
                   unname(.NATIVE_TREE_XGBOOST_PARAMETER_TYPES[[name]]))) {
      stop("XGBoost parameter '", name, "' has the wrong declared type.",
           call. = FALSE)
    }
  }
  bounded <- function(name, lower, upper, lower_open = FALSE) {
    value <- by_name[[name]]$value
    lower_ok <- if (isTRUE(lower_open)) value > lower else value >= lower
    if (length(value) != 1L || !lower_ok || value > upper) {
      stop("XGBoost parameter '", name, "' is outside its supported range.",
           call. = FALSE)
    }
  }
  bounded("learning_rate", 0, 1, lower_open = TRUE)
  bounded("max_delta_step", 0, .NATIVE_TREE_MAX_FLOAT_ABS, lower_open = TRUE)
  bounded("max_depth", 1, .NATIVE_TREE_XGBOOST_MAX_DEPTH)
  bounded("min_child_weight", 0, .NATIVE_TREE_MAX_FLOAT_ABS)
  bounded("min_split_loss", 0, .NATIVE_TREE_MAX_FLOAT_ABS)
  bounded("num_boost_round", 1, .NATIVE_TREE_RESOURCE_LIMITS$max_trees)
  bounded("reg_alpha", 0, .NATIVE_TREE_MAX_FLOAT_ABS)
  bounded("reg_lambda", 0, .NATIVE_TREE_MAX_FLOAT_ABS, lower_open = TRUE)
  for (name in c("learning_rate", "max_delta_step", "reg_lambda")) {
    if (.native_tree_float32(by_name[[name]]$value, name) <= 0) {
      stop("XGBoost parameter '", name,
           "' must remain positive as float32.", call. = FALSE)
    }
  }
  if (!is.null(schema$lower) && !is.null(schema$upper) &&
      !is.null(schema$cuts)) {
    lower <- .native_tree_float32(schema$lower, "public feature lower bounds")
    upper <- .native_tree_float32(schema$upper, "public feature upper bounds")
    cuts <- lapply(schema$cuts, .native_tree_float32,
                   name = "public feature cuts")
    valid <- lower < upper
    for (i in seq_along(cuts)) {
      valid[[i]] <- valid[[i]] && all(diff(cuts[[i]]) > 0) &&
        all(cuts[[i]] > lower[[i]]) && all(cuts[[i]] < upper[[i]])
    }
    if (!all(valid)) {
      stop("XGBoost public cuts and bounds must remain strict as float32.",
           call. = FALSE)
    }
    target <- .native_tree_float32(
      c(schema$target$lower, schema$target$upper), "public target bounds")
    if (target[[1L]] >= target[[2L]]) {
      stop("XGBoost public target bounds must remain strict as float32.",
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

#' Enforce the data-independent ExtraTrees parameter profile
#' @keywords internal
.native_tree_extra_trees_parameters <- function(parameters, schema, mode) {
  if (!identical(mode, "native-tight")) {
    stop("ExtraTrees request v1 supports native-tight mode only.",
         call. = FALSE)
  }
  by_name <- stats::setNames(
    parameters, vapply(parameters, `[[`, character(1), "name"))
  actual <- names(by_name)
  unknown <- setdiff(actual, .NATIVE_TREE_EXTRA_TREES_REQUIRED_PARAMETERS)
  missing <- setdiff(.NATIVE_TREE_EXTRA_TREES_REQUIRED_PARAMETERS, actual)
  if (length(unknown)) {
    stop("Unsupported ExtraTrees parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  if (length(missing)) {
    stop("Missing required ExtraTrees parameter(s): ",
         paste(missing, collapse = ", "), ".", call. = FALSE)
  }
  for (name in actual) {
    if (!identical(by_name[[name]]$type,
                   unname(.NATIVE_TREE_EXTRA_TREES_PARAMETER_TYPES[[name]]))) {
      stop("ExtraTrees parameter '", name, "' has the wrong declared type.",
           call. = FALSE)
    }
  }
  depth <- by_name$max_depth$value
  trees <- by_name$n_estimators$value
  if (length(depth) != 1L || depth < 1L ||
      depth > .NATIVE_TREE_EXTRA_TREES_MAX_DEPTH) {
    stop("ExtraTrees parameter 'max_depth' is outside its supported range.",
         call. = FALSE)
  }
  if (length(trees) != 1L || trees < 1L ||
      trees > .NATIVE_TREE_EXTRA_TREES_MAX_TREES) {
    stop("ExtraTrees parameter 'n_estimators' is outside its supported range.",
         call. = FALSE)
  }
  if (!is.null(schema$lower) && !is.null(schema$upper) &&
      !is.null(schema$cuts)) {
    lower <- .native_tree_float32(schema$lower, "public feature lower bounds")
    upper <- .native_tree_float32(schema$upper, "public feature upper bounds")
    cuts <- lapply(schema$cuts, .native_tree_float32,
                   name = "public feature cuts")
    valid <- lower < upper
    for (i in seq_along(cuts)) {
      valid[[i]] <- valid[[i]] && all(diff(cuts[[i]]) > 0) &&
        all(cuts[[i]] > lower[[i]]) && all(cuts[[i]] < upper[[i]])
    }
    if (!all(valid)) {
      stop("ExtraTrees public cuts and bounds must remain strict as float32.",
           call. = FALSE)
    }
    target <- .native_tree_float32(
      c(schema$target$lower, schema$target$upper), "public target bounds")
    if (target[[1L]] >= target[[2L]]) {
      stop("ExtraTrees public target bounds must remain strict as float32.",
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

#' Enforce the adaptive private Random Forest parameter profile
#' @keywords internal
.native_tree_random_forest_parameters <- function(parameters, schema, mode) {
  if (!identical(mode, "native-tight")) {
    stop("Random Forest request v1 supports native-tight mode only.",
         call. = FALSE)
  }
  by_name <- stats::setNames(
    parameters, vapply(parameters, `[[`, character(1), "name"))
  actual <- names(by_name)
  unknown <- setdiff(actual, .NATIVE_TREE_RANDOM_FOREST_REQUIRED_PARAMETERS)
  missing <- setdiff(.NATIVE_TREE_RANDOM_FOREST_REQUIRED_PARAMETERS, actual)
  if (length(unknown)) {
    stop("Unsupported Random Forest parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  if (length(missing)) {
    stop("Missing required Random Forest parameter(s): ",
         paste(missing, collapse = ", "), ".", call. = FALSE)
  }
  for (name in actual) {
    if (!identical(by_name[[name]]$type,
                   unname(.NATIVE_TREE_RANDOM_FOREST_PARAMETER_TYPES[[name]]))) {
      stop("Random Forest parameter '", name,
           "' has the wrong declared type.", call. = FALSE)
    }
  }
  depth <- by_name$max_depth$value
  trees <- by_name$n_estimators$value
  max_features <- by_name$max_features$value
  feature_count <- length(schema$features)
  if (length(depth) != 1L || depth < 1L ||
      depth > .NATIVE_TREE_RANDOM_FOREST_MAX_DEPTH) {
    stop("Random Forest parameter 'max_depth' is outside its supported range.",
         call. = FALSE)
  }
  if (length(trees) != 1L || trees < 1L ||
      trees > .NATIVE_TREE_RANDOM_FOREST_MAX_TREES) {
    stop("Random Forest parameter 'n_estimators' is outside its supported range.",
         call. = FALSE)
  }
  if (length(max_features) != 1L || max_features < 1L ||
      feature_count < 1L || max_features > feature_count) {
    stop("Random Forest parameter 'max_features' must be within the public feature count.",
         call. = FALSE)
  }
  if (!is.null(schema$lower) && !is.null(schema$upper) &&
      !is.null(schema$cuts)) {
    lower <- .native_tree_float32(schema$lower, "public feature lower bounds")
    upper <- .native_tree_float32(schema$upper, "public feature upper bounds")
    cuts <- lapply(schema$cuts, .native_tree_float32,
                   name = "public feature cuts")
    valid <- lower < upper
    for (i in seq_along(cuts)) {
      valid[[i]] <- valid[[i]] && all(diff(cuts[[i]]) > 0) &&
        all(cuts[[i]] > lower[[i]]) && all(cuts[[i]] < upper[[i]])
    }
    target <- .native_tree_float32(
      c(schema$target$lower, schema$target$upper), "public target bounds")
    if (!all(valid) || target[[1L]] >= target[[2L]]) {
      stop("Random Forest public cuts and bounds must remain strict as float32.",
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

#' Enforce the dsFlower asymmetric public-bin boosting profile
#' @keywords internal
.native_tree_lightgbm_parameters <- function(parameters, schema, mode) {
  if (!identical(mode, "native-tight")) {
    stop("LightGBM-style request v1 supports native-tight mode only.",
         call. = FALSE)
  }
  by_name <- stats::setNames(
    parameters, vapply(parameters, `[[`, character(1), "name"))
  actual <- names(by_name)
  unknown <- setdiff(actual, .NATIVE_TREE_LIGHTGBM_REQUIRED_PARAMETERS)
  missing <- setdiff(.NATIVE_TREE_LIGHTGBM_REQUIRED_PARAMETERS, actual)
  if (length(unknown)) {
    stop("Unsupported LightGBM-style parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  if (length(missing)) {
    stop("Missing required LightGBM-style parameter(s): ",
         paste(missing, collapse = ", "), ".", call. = FALSE)
  }
  for (name in actual) {
    if (!identical(by_name[[name]]$type,
                   unname(.NATIVE_TREE_LIGHTGBM_PARAMETER_TYPES[[name]]))) {
      stop("LightGBM-style parameter '", name,
           "' has the wrong declared type.", call. = FALSE)
    }
  }
  bounded <- function(name, lower, upper, lower_open = FALSE) {
    value <- by_name[[name]]$value
    lower_ok <- if (isTRUE(lower_open)) value > lower else value >= lower
    if (length(value) != 1L || !lower_ok || value > upper) {
      stop("LightGBM-style parameter '", name,
           "' is outside its supported range.", call. = FALSE)
    }
  }
  bounded("learning_rate", 0, 1, TRUE)
  bounded("max_delta_step", 0, .NATIVE_TREE_MAX_FLOAT_ABS, TRUE)
  bounded("max_depth", 1, .NATIVE_TREE_LIGHTGBM_MAX_DEPTH)
  bounded("min_data_in_leaf", 1, .Machine$integer.max)
  bounded("min_gain_to_split", 0, .NATIVE_TREE_MAX_FLOAT_ABS)
  bounded("num_iterations", 1, .NATIVE_TREE_RESOURCE_LIMITS$max_trees)
  bounded("lambda_l1", 0, .NATIVE_TREE_MAX_FLOAT_ABS)
  bounded("lambda_l2", 0, .NATIVE_TREE_MAX_FLOAT_ABS, TRUE)
  max_leaves <- min(.NATIVE_TREE_LIGHTGBM_MAX_LEAVES,
                    2^(min(by_name$max_depth$value, 8L)))
  bounded("num_leaves", 2, max_leaves)
  invisible(TRUE)
}

#' Enforce the dsFlower numeric oblivious public-bin boosting profile
#' @keywords internal
.native_tree_catboost_parameters <- function(parameters, schema, mode) {
  if (!identical(mode, "native-tight")) {
    stop("CatBoost-style request v1 supports native-tight mode only.",
         call. = FALSE)
  }
  by_name <- stats::setNames(
    parameters, vapply(parameters, `[[`, character(1), "name"))
  actual <- names(by_name)
  unknown <- setdiff(actual, .NATIVE_TREE_CATBOOST_REQUIRED_PARAMETERS)
  missing <- setdiff(.NATIVE_TREE_CATBOOST_REQUIRED_PARAMETERS, actual)
  if (length(unknown)) {
    stop("Unsupported CatBoost-style parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  if (length(missing)) {
    stop("Missing required CatBoost-style parameter(s): ",
         paste(missing, collapse = ", "), ".", call. = FALSE)
  }
  for (name in actual) {
    if (!identical(by_name[[name]]$type,
                   unname(.NATIVE_TREE_CATBOOST_PARAMETER_TYPES[[name]]))) {
      stop("CatBoost-style parameter '", name,
           "' has the wrong declared type.", call. = FALSE)
    }
  }
  bounded <- function(name, lower, upper, lower_open = FALSE) {
    value <- by_name[[name]]$value
    lower_ok <- if (isTRUE(lower_open)) value > lower else value >= lower
    if (length(value) != 1L || !lower_ok || value > upper) {
      stop("CatBoost-style parameter '", name,
           "' is outside its supported range.", call. = FALSE)
    }
  }
  bounded("depth", 1, .NATIVE_TREE_CATBOOST_MAX_DEPTH)
  bounded("iterations", 1, .NATIVE_TREE_RESOURCE_LIMITS$max_trees)
  bounded("l2_leaf_reg", 0, .NATIVE_TREE_MAX_FLOAT_ABS, TRUE)
  bounded("learning_rate", 0, 1, TRUE)
  bounded("max_delta_step", 0, .NATIVE_TREE_MAX_FLOAT_ABS, TRUE)
  lower <- .native_tree_float32(schema$lower, "public feature lower bounds")
  upper <- .native_tree_float32(schema$upper, "public feature upper bounds")
  cuts <- lapply(schema$cuts, .native_tree_float32,
                 name = "public feature cuts")
  valid <- lower < upper
  for (i in seq_along(cuts)) {
    valid[[i]] <- valid[[i]] && all(diff(cuts[[i]]) > 0) &&
      all(cuts[[i]] > lower[[i]]) && all(cuts[[i]] < upper[[i]])
  }
  if (!all(valid)) {
    stop("CatBoost-style public cuts and bounds must remain strict as float32.",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Canonicalise hard execution ceilings
#' @keywords internal
.native_tree_resources <- function(resources) {
  if (!is.list(resources) ||
      !identical(sort(names(resources)), sort(names(.NATIVE_TREE_RESOURCE_DEFAULTS)))) {
    stop("resources must contain every canonical resource cap exactly once.",
         call. = FALSE)
  }
  out <- .NATIVE_TREE_RESOURCE_DEFAULTS
  for (name in names(out)) {
    value <- .native_tree_number_vector(resources[[name]],
                                        paste0("resources$", name), TRUE)
    if (length(value) != 1L || value < 1L ||
        value > .NATIVE_TREE_RESOURCE_LIMITS[[name]]) {
      stop("resources$", name, " exceeds its hard server limit.", call. = FALSE)
    }
    out[[name]] <- value[[1L]]
  }
  out
}

#' Enforce common parameter-to-resource ceilings
#' @keywords internal
.native_tree_parameter_resource_check <- function(parameters, resources) {
  aliases <- list(
    max_trees = c("iterations", "n_estimators", "num_boost_round",
                  "num_iterations", "num_trees"),
    max_depth = c("depth", "max_depth"),
    max_bins = c("border_count", "max_bin", "max_bins"),
    max_threads = c("n_jobs", "nthread", "num_threads", "thread_count"))
  by_name <- stats::setNames(parameters,
                             vapply(parameters, `[[`, character(1), "name"))
  for (cap in names(aliases)) {
    for (name in intersect(names(by_name), aliases[[cap]])) {
      record <- by_name[[name]]
      if (record$type %in% c("integer", "number") &&
          record$value > resources[[cap]]) {
        stop("Parameter '", name, "' exceeds resources$", cap, ".",
             call. = FALSE)
      }
    }
  }
  invisible(TRUE)
}

#' Validate and canonicalise one native-tree manifest
#' @keywords internal
.canonical_native_tree_manifest <- function(manifest) {
  expected <- c(
    "contract", "engine", "mode", "parameters", "public_schema",
    "resources", "task")
  if (!is.list(manifest) || !identical(sort(names(manifest)), expected)) {
    stop("Native-tree manifest fields do not match contract v1.", call. = FALSE)
  }
  contract <- .native_tree_string(manifest$contract, "contract")
  if (!identical(contract, .NATIVE_TREE_CONTRACT)) {
    stop("Unsupported native-tree contract.", call. = FALSE)
  }
  engine <- .native_tree_string(manifest$engine, "engine", "^[a-z_]+$")
  mode <- .native_tree_string(manifest$mode, "mode", "^[a-z-]+$")
  task <- .native_tree_string(manifest$task, "task", "^[a-z]+$")
  if (!engine %in% .NATIVE_TREE_ENGINES) stop("Unsupported tree engine.", call. = FALSE)
  if (!mode %in% .NATIVE_TREE_MODES) stop("Unsupported tree mode.", call. = FALSE)
  if (!task %in% .NATIVE_TREE_TASKS) stop("Unsupported tree task.", call. = FALSE)
  schema <- manifest$public_schema
  schema_version <- if (is.list(schema)) schema$version else NULL
  if (!is.list(schema) ||
      !identical(sort(names(schema)),
                 c("cuts", "features", "lower", "sha256", "target", "upper",
                   "version")) ||
      !(is.integer(schema_version) || is.numeric(schema_version)) ||
      is.logical(schema_version) || length(schema_version) != 1L ||
      is.na(schema_version) || !is.finite(schema_version) ||
      schema_version != 1) {
    stop("Invalid public feature schema.", call. = FALSE)
  }
  schema <- .native_tree_public_schema(
    schema$features, list(lower = schema$lower, upper = schema$upper),
    schema$cuts, schema$target, task, schema$sha256)
  if (is.null(schema$cuts)) {
    stop("native-tight requires public cuts for every feature.", call. = FALSE)
  }
  parameters <- .native_tree_parameters(manifest$parameters, mode)
  resources <- .native_tree_resources(manifest$resources)
  if (length(schema$features) > resources$max_features) {
    stop("Feature count exceeds resources$max_features.", call. = FALSE)
  }
  if (!is.null(schema$cuts) &&
      max(vapply(schema$cuts, length, integer(1)) + 1L) > resources$max_bins) {
    stop("Public cut count exceeds resources$max_bins.", call. = FALSE)
  }
  if (identical(engine, "xgboost")) {
    .native_tree_xgboost_parameters(parameters, schema, mode)
  } else if (identical(engine, "extra_trees")) {
    .native_tree_extra_trees_parameters(parameters, schema, mode)
  } else if (identical(engine, "random_forest")) {
    .native_tree_random_forest_parameters(parameters, schema, mode)
  } else if (identical(engine, "lightgbm")) {
    .native_tree_lightgbm_parameters(parameters, schema, mode)
  } else if (identical(engine, "catboost")) {
    .native_tree_catboost_parameters(parameters, schema, mode)
  }
  .native_tree_parameter_resource_check(parameters, resources)
  list(
    contract = contract,
    engine = engine,
    mode = mode,
    task = task,
    public_schema = schema,
    parameters = parameters,
    resources = resources)
}

#' Infer the explicit wire type of one public parameter
#' @keywords internal
.native_tree_typed_parameter <- function(name, value) {
  if (is.list(value) &&
      identical(sort(names(value)), c("type", "value"))) {
    return(.native_tree_parameter_record(list(
      name = name, type = value$type, value = value$value)))
  }
  if (is.object(value) || is.factor(value) || is.matrix(value) ||
      is.array(value) || is.list(value) || is.raw(value) ||
      is.complex(value) || !is.atomic(value) || !length(value) ||
      length(value) > 256L || anyNA(value)) {
    stop("Public tree parameters must be finite plain atomic values.",
         call. = FALSE)
  }
  base <- if (is.logical(value)) {
    "boolean"
  } else if (is.integer(value)) {
    "integer"
  } else if (is.numeric(value)) {
    if (any(!is.finite(value))) {
      stop("Public tree numeric parameters must be finite.", call. = FALSE)
    }
    "number"
  } else if (is.character(value)) {
    "string"
  } else {
    stop("Unsupported public tree parameter type.", call. = FALSE)
  }
  type <- if (length(value) == 1L) base else paste0(base, "_array")
  .native_tree_parameter_record(list(name = name, type = type, value = value))
}

#' Build a canonical typed public parameter schema
#' @keywords internal
.build_native_tree_parameters <- function(parameters = list()) {
  if (!is.list(parameters) ||
      (length(parameters) && (is.null(names(parameters)) ||
        any(!nzchar(names(parameters))) || anyDuplicated(names(parameters))))) {
    stop("parameters must be a uniquely named list.", call. = FALSE)
  }
  records <- lapply(seq_along(parameters), function(i) {
    .native_tree_typed_parameter(names(parameters)[[i]], parameters[[i]])
  })
  records[order(vapply(records, `[[`, character(1), "name"), method = "radix")]
}

#' Build and pin a native-tree manifest without exposing a public model yet
#' @keywords internal
.build_native_tree_manifest <- function(engine, mode, task, features, bounds,
                                        target, cuts = NULL, parameters = list(),
                                        resources = list(),
                                        schema_sha256 = NULL) {
  if (!is.list(resources) ||
      (length(resources) && (is.null(names(resources)) ||
        any(!nzchar(names(resources))) || anyDuplicated(names(resources)) ||
        length(setdiff(names(resources),
                       names(.NATIVE_TREE_RESOURCE_DEFAULTS)))))) {
    stop("resources contains an unknown or duplicate resource cap.", call. = FALSE)
  }
  caps <- .NATIVE_TREE_RESOURCE_DEFAULTS
  for (name in names(resources)) caps[[name]] <- resources[[name]]
  schema <- .native_tree_public_schema(
    features, bounds, cuts, target, task, schema_sha256)
  manifest <- list(
    contract = .NATIVE_TREE_CONTRACT,
    engine = engine,
    mode = mode,
    task = task,
    public_schema = schema,
    parameters = .build_native_tree_parameters(parameters),
    resources = caps)
  value <- .canonical_native_tree_manifest(manifest)
  bytes <- .native_tree_json(value)
  if (length(bytes) > .NATIVE_TREE_MANIFEST_MAX_BYTES) {
    stop("Canonical native-tree manifest exceeds 65536 bytes.", call. = FALSE)
  }
  list(
    value = value,
    json = rawToChar(bytes),
    b64 = gsub("[\r\n]", "", jsonlite::base64_enc(bytes)),
    sha256 = digest::digest(bytes, algo = "sha256", serialize = FALSE))
}

#' Decode and re-pin exact canonical native-tree request bytes
#' @keywords internal
.validate_native_tree_request_wire <- function(request_b64, request_sha256) {
  if (!is.character(request_b64) || length(request_b64) != 1L ||
      is.na(request_b64) || !nzchar(request_b64) ||
      nchar(request_b64, type = "bytes") >
        4L * ceiling(.NATIVE_TREE_MANIFEST_MAX_BYTES / 3L)) {
    stop("native-tree request must be one bounded canonical base64 string.",
         call. = FALSE)
  }
  if (!is.character(request_sha256) || length(request_sha256) != 1L ||
      is.na(request_sha256) ||
      !grepl("^[0-9a-f]{64}$", request_sha256)) {
    stop("native-tree request SHA-256 must be one lowercase digest.",
         call. = FALSE)
  }
  decoded <- tryCatch(jsonlite::base64_dec(request_b64),
                      error = function(e) NULL)
  canonical_b64 <- if (is.null(decoded)) NULL else
    gsub("[\r\n]", "", jsonlite::base64_enc(decoded))
  if (is.null(decoded) || !length(decoded) ||
      length(decoded) > .NATIVE_TREE_MANIFEST_MAX_BYTES ||
      !identical(canonical_b64, request_b64)) {
    stop("native-tree request base64 is not canonical and bounded.",
         call. = FALSE)
  }
  json <- tryCatch(rawToChar(decoded), error = function(e) NULL)
  parsed <- tryCatch(
    jsonlite::fromJSON(json, simplifyVector = FALSE),
    error = function(e) NULL)
  if (is.null(json) || !is.list(parsed) ||
      is.na(iconv(json, from = "UTF-8", to = "UTF-8", sub = NA_character_))) {
    stop("native-tree request must decode to valid UTF-8 JSON.",
         call. = FALSE)
  }
  value <- .canonical_native_tree_manifest(parsed)
  bytes <- .native_tree_json(value)
  actual <- digest::digest(bytes, algo = "sha256", serialize = FALSE)
  if (!identical(bytes, decoded) || !identical(actual, request_sha256)) {
    stop("native-tree request bytes or SHA-256 do not match the canonical request.",
         call. = FALSE)
  }
  list(value = value, json = json, b64 = request_b64, sha256 = actual)
}

#' Build the exact typed XGBoost parameter array for request v1
#' @keywords internal
.native_tree_xgboost_parameter_values <- function(params) {
  if (!is.list(params) || is.null(names(params)) || anyNA(names(params)) ||
      any(!nzchar(names(params))) || anyDuplicated(names(params))) {
    stop("XGBoost params must be a uniquely named list.", call. = FALSE)
  }
  names <- c(.NATIVE_TREE_XGBOOST_REQUIRED_PARAMETERS,
             .NATIVE_TREE_XGBOOST_OPTIONAL_PARAMETERS)
  unknown <- setdiff(names(params), c("task", names))
  if (length(unknown)) {
    stop("Unsupported XGBoost parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  present <- intersect(names, names(params))
  values <- stats::setNames(vector("list", length(present)), present)
  for (name in present) {
    type <- unname(.NATIVE_TREE_XGBOOST_PARAMETER_TYPES[[name]])
    values[[name]] <- list(type = type, value = params[[name]])
  }
  values
}

#' Validate XGBoost modelling parameters before a public schema is available
#' @keywords internal
.validate_xgboost_model_params <- function(params) {
  values <- .native_tree_xgboost_parameter_values(params)
  records <- .build_native_tree_parameters(values)
  canonical <- .native_tree_parameters(records, "native-tight")
  .native_tree_xgboost_parameters(
    canonical, list(features = character()), "native-tight")
  invisible(params)
}

# Tag the public binary vocabulary without coercing its scalar type.
.native_tree_tag_target_levels <- function(levels, model = "XGBoost") {
  if (is.factor(levels)) levels <- as.character(levels)
  if (!is.atomic(levels) || !is.null(dim(levels)) || length(levels) != 2L ||
      anyNA(levels) || anyDuplicated(levels)) {
    stop("Binary ", model,
         " requires exactly two distinct ordered public target_levels.",
         call. = FALSE)
  }
  type <- if (is.character(levels)) {
    "string"
  } else if (is.logical(levels)) {
    "boolean"
  } else if (is.numeric(levels)) {
    if (any(!is.finite(levels))) {
      stop("Numeric target_levels must be finite.", call. = FALSE)
    }
    "number"
  } else {
    stop("target_levels must be character, logical, or numeric.",
         call. = FALSE)
  }
  lapply(seq_along(levels), function(index) {
    .native_tree_target_level(list(type = type, value = levels[[index]]))
  })
}

#' Build the exact typed ExtraTrees parameter array for request v1
#' @keywords internal
.native_tree_extra_trees_parameter_values <- function(params) {
  if (!is.list(params) || is.null(names(params)) || anyNA(names(params)) ||
      any(!nzchar(names(params))) || anyDuplicated(names(params))) {
    stop("ExtraTrees params must be a uniquely named list.", call. = FALSE)
  }
  unknown <- setdiff(
    names(params), c("task", .NATIVE_TREE_EXTRA_TREES_REQUIRED_PARAMETERS))
  if (length(unknown)) {
    stop("Unsupported ExtraTrees parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  present <- intersect(
    .NATIVE_TREE_EXTRA_TREES_REQUIRED_PARAMETERS, names(params))
  values <- stats::setNames(vector("list", length(present)), present)
  for (name in present) {
    values[[name]] <- list(
      type = unname(.NATIVE_TREE_EXTRA_TREES_PARAMETER_TYPES[[name]]),
      value = params[[name]])
  }
  values
}

.native_tree_random_forest_parameter_values <- function(params) {
  if (!is.list(params) || is.null(names(params)) || anyNA(names(params)) ||
      any(!nzchar(names(params))) || anyDuplicated(names(params))) {
    stop("Random Forest params must be a uniquely named list.", call. = FALSE)
  }
  unknown <- setdiff(
    names(params), c("task", .NATIVE_TREE_RANDOM_FOREST_REQUIRED_PARAMETERS))
  if (length(unknown)) {
    stop("Unsupported Random Forest parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  present <- intersect(
    .NATIVE_TREE_RANDOM_FOREST_REQUIRED_PARAMETERS, names(params))
  values <- stats::setNames(vector("list", length(present)), present)
  for (name in present) {
    values[[name]] <- list(
      type = unname(.NATIVE_TREE_RANDOM_FOREST_PARAMETER_TYPES[[name]]),
      value = params[[name]])
  }
  values
}

#' Validate ExtraTrees modelling parameters before public schema is available
#' @keywords internal
.validate_extra_trees_model_params <- function(params) {
  values <- .native_tree_extra_trees_parameter_values(params)
  records <- .build_native_tree_parameters(values)
  canonical <- .native_tree_parameters(records, "native-tight")
  .native_tree_extra_trees_parameters(
    canonical, list(features = character()), "native-tight")
  invisible(params)
}

.validate_random_forest_model_params <- function(params) {
  max_features <- params$max_features
  automatic <- is.null(max_features) ||
    (is.character(max_features) && length(max_features) == 1L &&
     !is.na(max_features) && identical(tolower(max_features), "auto"))
  if (!automatic && (!(is.integer(max_features) || is.numeric(max_features)) ||
      is.logical(max_features) || length(max_features) != 1L ||
      is.na(max_features) || !is.finite(max_features) ||
      max_features != floor(max_features) || max_features < 1L)) {
    stop("Random Forest max_features must be NULL, 'auto', or a positive integer.",
         call. = FALSE)
  }
  resolved <- params
  resolved$max_features <- if (automatic) 1L else as.integer(max_features)
  records <- .build_native_tree_parameters(
    .native_tree_random_forest_parameter_values(resolved))
  canonical <- .native_tree_parameters(records, "native-tight")
  # Placeholder public names validate the typed/ranged profile; the real public
  # feature count is rechecked when the request resolves mtry.
  .native_tree_random_forest_parameters(
    canonical,
    list(features = paste0("x", seq_len(resolved$max_features))),
    "native-tight")
  invisible(params)
}

#' Build the exact typed dsFlower boosting-style parameter array for request v1
#' @keywords internal
.native_tree_boosting_parameter_values <- function(params, engine) {
  profiles <- list(
    lightgbm = list(
      label = "LightGBM-style",
      required = .NATIVE_TREE_LIGHTGBM_REQUIRED_PARAMETERS,
      types = .NATIVE_TREE_LIGHTGBM_PARAMETER_TYPES),
    catboost = list(
      label = "CatBoost-style",
      required = .NATIVE_TREE_CATBOOST_REQUIRED_PARAMETERS,
      types = .NATIVE_TREE_CATBOOST_PARAMETER_TYPES))
  profile <- profiles[[engine]]
  if (is.null(profile)) {
    stop("Unsupported dsFlower boosting-style engine.", call. = FALSE)
  }
  if (!is.list(params) || is.null(names(params)) || anyNA(names(params)) ||
      any(!nzchar(names(params))) || anyDuplicated(names(params))) {
    stop(profile$label, " params must be a uniquely named list.",
         call. = FALSE)
  }
  unknown <- setdiff(names(params), c("task", profile$required))
  if (length(unknown)) {
    stop("Unsupported ", profile$label, " parameter(s): ",
         paste(unknown, collapse = ", "), ".", call. = FALSE)
  }
  present <- intersect(profile$required, names(params))
  values <- stats::setNames(vector("list", length(present)), present)
  for (name in present) {
    values[[name]] <- list(
      type = unname(profile$types[[name]]), value = params[[name]])
  }
  values
}

.native_tree_lightgbm_parameter_values <- function(params) {
  .native_tree_boosting_parameter_values(params, "lightgbm")
}

.native_tree_catboost_parameter_values <- function(params) {
  .native_tree_boosting_parameter_values(params, "catboost")
}

#' Validate dsFlower boosting-style parameters before the public schema exists
#' @keywords internal
.validate_boosting_style_model_params <- function(params, engine) {
  values <- .native_tree_boosting_parameter_values(params, engine)
  records <- .build_native_tree_parameters(values)
  canonical <- .native_tree_parameters(records, "native-tight")
  if (identical(engine, "lightgbm")) {
    .native_tree_lightgbm_parameters(
      canonical, list(features = character()), "native-tight")
  } else {
    # CatBoost's float32 public-cut check needs a real schema, so validate the
    # exact typed/ranged parameter profile here and repeat the full check when
    # the request is built.
    by_name <- stats::setNames(
      canonical, vapply(canonical, `[[`, character(1), "name"))
    bounded <- function(name, lower, upper, lower_open = FALSE) {
      value <- by_name[[name]]$value
      lower_ok <- if (isTRUE(lower_open)) value > lower else value >= lower
      if (length(value) != 1L || !lower_ok || value > upper) {
        stop("CatBoost-style parameter '", name,
             "' is outside its supported range.", call. = FALSE)
      }
    }
    bounded("depth", 1, .NATIVE_TREE_CATBOOST_MAX_DEPTH)
    bounded("iterations", 1, .NATIVE_TREE_RESOURCE_LIMITS$max_trees)
    bounded("l2_leaf_reg", 0, .NATIVE_TREE_MAX_FLOAT_ABS, TRUE)
    bounded("learning_rate", 0, 1, TRUE)
    bounded("max_delta_step", 0, .NATIVE_TREE_MAX_FLOAT_ABS, TRUE)
  }
  invisible(params)
}

.validate_lightgbm_model_params <- function(params) {
  .validate_boosting_style_model_params(params, "lightgbm")
}

.validate_catboost_model_params <- function(params) {
  .validate_boosting_style_model_params(params, "catboost")
}

#' Build one canonical analyst-facing ExtraTrees request
#' @keywords internal
.build_extra_trees_request <- function(
    params, features, feature_bounds, feature_cuts, target_name,
    target_levels = NULL, target_bounds = NULL, schema_sha256 = NULL) {
  if (!is.list(params) || !is.character(params$task) ||
      length(params$task) != 1L || is.na(params$task) ||
      !params$task %in% .NATIVE_TREE_TASKS) {
    stop("ExtraTrees params must contain task='binary' or task='regression'.",
         call. = FALSE)
  }
  .validate_extra_trees_model_params(params)
  task <- params$task
  target <- if (identical(task, "binary")) {
    if (!is.null(target_bounds)) {
      stop("Binary ExtraTrees fixes encoded target bounds to 0 and 1.",
           call. = FALSE)
    }
    list(name = target_name, kind = "binary",
         levels = .native_tree_tag_target_levels(
           target_levels, "ExtraTrees"), lower = 0, upper = 1)
  } else {
    if (!is.null(target_levels)) {
      stop("Regression ExtraTrees does not accept target_levels.",
           call. = FALSE)
    }
    if (!is.list(target_bounds) ||
        !identical(sort(names(target_bounds)), c("lower", "upper"))) {
      stop("Regression ExtraTrees requires target_bounds with lower and upper.",
           call. = FALSE)
    }
    list(name = target_name, kind = "continuous", levels = NULL,
         lower = target_bounds$lower, upper = target_bounds$upper)
  }
  cut_counts <- if (is.list(feature_cuts)) {
    vapply(feature_cuts, length, integer(1))
  } else {
    integer()
  }
  max_bins <- if (length(cut_counts)) max(cut_counts + 1L) else 2L
  resources <- .NATIVE_TREE_RESOURCE_DEFAULTS
  resources$max_features <- length(features)
  resources$max_trees <- as.integer(params$n_estimators)
  resources$max_depth <- as.integer(params$max_depth)
  resources$max_bins <- as.integer(max_bins)
  .build_native_tree_manifest(
    engine = "extra_trees", mode = "native-tight", task = task,
    features = features, bounds = feature_bounds, cuts = feature_cuts,
    target = target,
    parameters = .native_tree_extra_trees_parameter_values(params),
    resources = resources, schema_sha256 = schema_sha256)
}

.build_random_forest_request <- function(
    params, features, feature_bounds, feature_cuts, target_name,
    target_levels = NULL, target_bounds = NULL, schema_sha256 = NULL) {
  if (!is.list(params) || !is.character(params$task) ||
      length(params$task) != 1L || is.na(params$task) ||
      !params$task %in% .NATIVE_TREE_TASKS) {
    stop("Random Forest params must contain task='binary' or task='regression'.",
         call. = FALSE)
  }
  .validate_random_forest_model_params(params)
  task <- params$task
  p <- length(features)
  if (p < 1L) stop("Random Forest requires at least one public feature.",
                   call. = FALSE)
  max_features <- params$max_features
  automatic <- is.null(max_features) ||
    (is.character(max_features) && length(max_features) == 1L &&
     !is.na(max_features) && identical(tolower(max_features), "auto"))
  if (automatic) {
    max_features <- if (identical(task, "binary")) {
      ceiling(sqrt(p))
    } else {
      max(1L, ceiling(p / 3))
    }
  }
  max_features <- as.integer(max_features)
  if (max_features > p) {
    stop("Random Forest max_features cannot exceed the public feature count.",
         call. = FALSE)
  }
  wire_params <- params
  wire_params$max_features <- max_features
  target <- if (identical(task, "binary")) {
    if (!is.null(target_bounds)) {
      stop("Binary Random Forest fixes encoded target bounds to 0 and 1.",
           call. = FALSE)
    }
    list(name = target_name, kind = "binary",
         levels = .native_tree_tag_target_levels(
           target_levels, "Random Forest"), lower = 0, upper = 1)
  } else {
    if (!is.null(target_levels)) {
      stop("Regression Random Forest does not accept target_levels.",
           call. = FALSE)
    }
    if (!is.list(target_bounds) ||
        !identical(sort(names(target_bounds)), c("lower", "upper"))) {
      stop("Regression Random Forest requires target_bounds with lower and upper.",
           call. = FALSE)
    }
    list(name = target_name, kind = "continuous", levels = NULL,
         lower = target_bounds$lower, upper = target_bounds$upper)
  }
  cut_counts <- if (is.list(feature_cuts)) {
    vapply(feature_cuts, length, integer(1))
  } else integer()
  max_bins <- if (length(cut_counts)) max(cut_counts + 1L) else 2L
  resources <- .NATIVE_TREE_RESOURCE_DEFAULTS
  resources$max_features <- p
  resources$max_trees <- as.integer(wire_params$n_estimators)
  resources$max_depth <- as.integer(wire_params$max_depth)
  resources$max_bins <- as.integer(max_bins)
  .build_native_tree_manifest(
    engine = "random_forest", mode = "native-tight", task = task,
    features = features, bounds = feature_bounds, cuts = feature_cuts,
    target = target,
    parameters = .native_tree_random_forest_parameter_values(wire_params),
    resources = resources, schema_sha256 = schema_sha256)
}

#' Build one canonical analyst-facing dsFlower boosting-style request
#' @keywords internal
.build_boosting_style_request <- function(
    params, engine, features, feature_bounds, feature_cuts, target_name,
    target_levels = NULL, target_bounds = NULL, schema_sha256 = NULL) {
  label <- if (identical(engine, "lightgbm")) {
    "LightGBM-style"
  } else if (identical(engine, "catboost")) {
    "CatBoost-style"
  } else {
    stop("Unsupported dsFlower boosting-style engine.", call. = FALSE)
  }
  if (!is.list(params) || !is.character(params$task) ||
      length(params$task) != 1L || is.na(params$task) ||
      !params$task %in% .NATIVE_TREE_TASKS) {
    stop(label, " params must contain task='binary' or task='regression'.",
         call. = FALSE)
  }
  .validate_boosting_style_model_params(params, engine)
  task <- params$task
  target <- if (identical(task, "binary")) {
    if (!is.null(target_bounds)) {
      stop("Binary ", label, " fixes encoded target bounds to 0 and 1.",
           call. = FALSE)
    }
    list(name = target_name, kind = "binary",
         levels = .native_tree_tag_target_levels(target_levels, label),
         lower = 0, upper = 1)
  } else {
    if (!is.null(target_levels)) {
      stop("Regression ", label, " does not accept target_levels.",
           call. = FALSE)
    }
    if (!is.list(target_bounds) ||
        !identical(sort(names(target_bounds)), c("lower", "upper"))) {
      stop("Regression ", label,
           " requires target_bounds with lower and upper.", call. = FALSE)
    }
    list(name = target_name, kind = "continuous", levels = NULL,
         lower = target_bounds$lower, upper = target_bounds$upper)
  }
  cut_counts <- if (is.list(feature_cuts)) {
    vapply(feature_cuts, length, integer(1))
  } else integer()
  max_bins <- if (length(cut_counts)) max(cut_counts + 1L) else 2L
  resources <- .NATIVE_TREE_RESOURCE_DEFAULTS
  resources$max_features <- length(features)
  if (identical(engine, "lightgbm")) {
    resources$max_trees <- as.integer(params$num_iterations)
    resources$max_depth <- as.integer(params$max_depth)
    values <- .native_tree_lightgbm_parameter_values(params)
  } else {
    resources$max_trees <- as.integer(params$iterations)
    resources$max_depth <- as.integer(params$depth)
    values <- .native_tree_catboost_parameter_values(params)
  }
  resources$max_bins <- as.integer(max_bins)
  .build_native_tree_manifest(
    engine = engine, mode = "native-tight", task = task,
    features = features, bounds = feature_bounds, cuts = feature_cuts,
    target = target, parameters = values, resources = resources,
    schema_sha256 = schema_sha256)
}

.build_lightgbm_request <- function(
    params, features, feature_bounds, feature_cuts, target_name,
    target_levels = NULL, target_bounds = NULL, schema_sha256 = NULL) {
  .build_boosting_style_request(
    params, "lightgbm", features, feature_bounds, feature_cuts, target_name,
    target_levels, target_bounds, schema_sha256)
}

.build_catboost_request <- function(
    params, features, feature_bounds, feature_cuts, target_name,
    target_levels = NULL, target_bounds = NULL, schema_sha256 = NULL) {
  .build_boosting_style_request(
    params, "catboost", features, feature_bounds, feature_cuts, target_name,
    target_levels, target_bounds, schema_sha256)
}

#' Build one canonical analyst-facing XGBoost request
#' @keywords internal
.build_xgboost_request <- function(params, features, feature_bounds,
                                   feature_cuts, target_name, target_levels = NULL,
                                   target_bounds = NULL,
                                   schema_sha256 = NULL) {
  if (!is.list(params) || !is.character(params$task) ||
      length(params$task) != 1L || is.na(params$task) ||
      !params$task %in% .NATIVE_TREE_TASKS) {
    stop("XGBoost params must contain task='binary' or task='regression'.",
         call. = FALSE)
  }
  task <- params$task
  target <- if (identical(task, "binary")) {
    if (!is.null(target_bounds)) {
      stop("Binary XGBoost fixes encoded target bounds to 0 and 1.",
           call. = FALSE)
    }
    list(name = target_name, kind = "binary",
         levels = .native_tree_tag_target_levels(target_levels),
         lower = 0, upper = 1)
  } else {
    if (!is.null(target_levels)) {
      stop("Regression XGBoost does not accept target_levels.",
           call. = FALSE)
    }
    if (!is.list(target_bounds) ||
        !identical(sort(names(target_bounds)), c("lower", "upper"))) {
      stop("Regression XGBoost requires target_bounds with lower and upper.",
           call. = FALSE)
    }
    list(name = target_name, kind = "continuous", levels = NULL,
         lower = target_bounds$lower, upper = target_bounds$upper)
  }
  cut_counts <- if (is.list(feature_cuts)) {
    vapply(feature_cuts, length, integer(1))
  } else {
    integer()
  }
  max_bins <- if (length(cut_counts)) max(cut_counts + 1L) else 2L
  resources <- .NATIVE_TREE_RESOURCE_DEFAULTS
  resources$max_features <- length(features)
  resources$max_trees <- as.integer(params$num_boost_round)
  resources$max_depth <- as.integer(params$max_depth)
  resources$max_bins <- as.integer(max_bins)
  .build_native_tree_manifest(
    engine = "xgboost", mode = "native-tight", task = task,
    features = features, bounds = feature_bounds, cuts = feature_cuts,
    target = target,
    parameters = .native_tree_xgboost_parameter_values(params),
    resources = resources, schema_sha256 = schema_sha256)
}

#' Build the canonical request for one implemented native-tree engine
#' @keywords internal
.build_native_tree_request <- function(
    engine, params, features, feature_bounds, feature_cuts, target_name,
    target_levels = NULL, target_bounds = NULL, schema_sha256 = NULL) {
  builder <- switch(engine,
    xgboost = .build_xgboost_request,
    extra_trees = .build_extra_trees_request,
    random_forest = .build_random_forest_request,
    lightgbm = .build_lightgbm_request,
    catboost = .build_catboost_request,
    stop("Native-tree engine has no implemented request builder.",
         call. = FALSE))
  builder(
    params, features, feature_bounds, feature_cuts, target_name,
    target_levels, target_bounds, schema_sha256)
}
