# Module: Native tree manifest contract
# Internal analyst-to-server request ABI only. The node validates and enriches
# it with node-owned privacy and data-scope state for the separate trusted
# Python ABI (including binary -> binary_classification). It is not a public
# model registration or a declaration that a native backend is available.

.NATIVE_TREE_CONTRACT <- "dsflower-native-tree-request-v1"
.NATIVE_TREE_ENGINES <- c(
  "catboost", "lightgbm", "random_forest", "xgboost")
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
.NATIVE_TREE_XGBOOST_AVAILABLE <- FALSE
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

#' Canonical JSON bytes for the native-tree cross-runtime ABI
#' @keywords internal
.native_tree_json <- function(value) {
  json <- as.character(jsonlite::toJSON(
    value, auto_unbox = TRUE, null = "null", na = "null",
    digits = NA, always_decimal = TRUE, pretty = FALSE))
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
.native_tree_tag_target_levels <- function(levels) {
  if (is.factor(levels)) levels <- as.character(levels)
  if (!is.atomic(levels) || !is.null(dim(levels)) || length(levels) != 2L ||
      anyNA(levels) || anyDuplicated(levels)) {
    stop("Binary XGBoost requires exactly two distinct ordered public target_levels.",
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
