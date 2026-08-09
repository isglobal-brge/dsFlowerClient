# Module: Atomic resampling contracts

.HOLDOUT_DENOMINATOR <- 1000000L
.HOLDOUT_VALIDATION_BINS <- 32L

.normalize_holdout <- function(value) {
  if (is.null(value)) return(NULL)
  if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
      is.na(value) || !is.finite(value)) {
    stop("'holdout' must be one finite numeric fraction.", call. = FALSE)
  }
  if (value <= 0 || value >= 1) {
    stop("'holdout' must be strictly between zero and one.", call. = FALSE)
  }
  scaled <- as.numeric(value) * .HOLDOUT_DENOMINATOR
  millionths <- round(scaled)
  if (abs(scaled - millionths) > 1e-8) {
    stop("'holdout' supports at most six decimal places.", call. = FALSE)
  }
  list(test_millionths = as.integer(millionths))
}
.holdout_contract <- function(spec, privacy_unit) {
  if (!is.list(spec) || !identical(names(spec), "test_millionths")) {
    stop("Invalid holdout specification.", call. = FALSE)
  }
  numerator <- suppressWarnings(as.numeric(spec$test_millionths))
  if (length(numerator) != 1L || !is.finite(numerator) ||
      numerator != floor(numerator) || numerator < 1L ||
      numerator >= .HOLDOUT_DENOMINATOR) {
    stop("Invalid holdout test fraction.", call. = FALSE)
  }
  unit <- tolower(as.character(unlist(privacy_unit, use.names = FALSE)))
  if (length(unit) != 1L || is.na(unit) || !unit %in% c("row", "patient")) {
    stop("Holdout needs one common row/patient privacy unit.", call. = FALSE)
  }
  payload <- list(
    assignment = "hmac-sha256-threshold-v1",
    method = "holdout",
    privacy_unit = unit,
    test_denominator = .HOLDOUT_DENOMINATOR,
    test_numerator = as.integer(numerator),
    unit_canonicalization = if (identical(unit, "patient"))
      "trim-utf8-v2" else "row-ordinal-v1",
    version = "dsflower-resampling-v1"
  )
  wire <- jsonlite::toJSON(
    payload, auto_unbox = TRUE, null = "null", digits = NA,
    pretty = FALSE)
  c(payload, list(sha256 = digest::digest(
    charToRaw(enc2utf8(wire)), algo = "sha256", serialize = FALSE)))
}

.holdout_config <- function(contract) {
  list(
    "resampling-version" = contract$version,
    "resampling-method" = contract$method,
    "resampling-assignment" = contract$assignment,
    "resampling-test-numerator" = contract$test_numerator,
    "resampling-test-denominator" = contract$test_denominator,
    "resampling-privacy-unit" = contract$privacy_unit,
    "resampling-unit-canonicalization" = contract$unit_canonicalization,
    "resampling-contract-sha256" = contract$sha256,
    "holdout-validation-bins" = .HOLDOUT_VALIDATION_BINS
  )
}

.assert_holdout_supported <- function(sub, data_kind) {
  if (!is.list(sub) || !identical(sub$track %||% NULL, "neural")) {
    stop("Atomic holdout is currently implemented only for neural models; ",
         "this backend is not advertised as supported.", call. = FALSE)
  }
  if (!identical(data_kind, "tabular")) {
    stop("Atomic neural holdout currently supports tabular data only.",
         call. = FALSE)
  }
  invisible(TRUE)
}
