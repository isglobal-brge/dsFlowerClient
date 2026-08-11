# Module: pooled differentially-private binary association

.ASSOCIATION_CONTRACT <- "dsflower-binary-association-3x3/v1"
.ASSOCIATION_RESULT_CONTRACT <- "dsflower-binary-association-result/v1"
.ASSOCIATION_MECHANISM <- "binary-association-joint-gaussian/v1"
.ASSOCIATION_EXECUTION_PROFILE <- "dsflower-binary-association-execution/v1"
.ASSOCIATION_MAX_LEVEL_BYTES <- 4096L
.ASSOCIATION_MAX_RESULT_BYTES <- 65536L
.ASSOCIATION_MAX_NODES <- 65536L

.association_scalar_text <- function(value, name) {
  if (!is.character(value) || length(value) != 1L || is.na(value)) {
    stop(name, " must be one non-missing character value.", call. = FALSE)
  }
  value <- tryCatch(
    iconv(value, from = "", to = "UTF-8", sub = NA_character_),
    error = function(e) NA_character_)
  if (is.na(value) || !nzchar(value) ||
      nchar(value, type = "bytes") > .ASSOCIATION_MAX_LEVEL_BYTES) {
    stop(name, " must be non-empty valid UTF-8 of at most 4096 bytes.",
         call. = FALSE)
  }
  enc2utf8(value)
}

.association_column <- function(value, name) {
  .association_scalar_text(value, name)
}

.association_level_kind <- function(value) {
  if (is.character(value)) return("string")
  if (is.logical(value)) return("boolean")
  if ((is.integer(value) || is.double(value)) && !is.object(value)) {
    return("number")
  }
  "unsupported"
}

.association_level_spec <- function(value, name) {
  if (is.factor(value)) value <- as.character(value)

  claimed <- NULL
  if (is.list(value) && !is.null(names(value))) {
    if (!identical(sort(names(value), method = "radix"),
                   c("type", "values"))) {
      stop(name, " must be a two-level vector or typed level object.",
           call. = FALSE)
    }
    claimed <- value[["type"]]
    if (!is.character(claimed) || length(claimed) != 1L ||
        is.na(claimed) || !claimed %in% c("string", "boolean", "number")) {
      stop(name, " has an invalid public level type.", call. = FALSE)
    }
    raw_values <- value[["values"]]
    if (!is.list(raw_values)) raw_values <- as.list(raw_values)
    if (length(raw_values) != 2L ||
        any(vapply(raw_values, length, integer(1)) != 1L)) {
      stop(name, " must contain exactly two public levels.", call. = FALSE)
    }
    kinds <- vapply(raw_values, .association_level_kind, character(1))
    if (any(kinds != claimed)) {
      stop(name, " values disagree with their public level type.",
           call. = FALSE)
    }
    value <- unlist(raw_values, use.names = FALSE)
  } else if (is.list(value)) {
    stop(name, " must be a two-level vector or typed level object.",
         call. = FALSE)
  }

  if (!is.atomic(value) || length(value) != 2L || anyNA(value)) {
    stop(name, " must contain exactly two non-missing public levels.",
         call. = FALSE)
  }
  kind <- .association_level_kind(value)
  if (identical(kind, "unsupported") ||
      (!is.null(claimed) && !identical(kind, claimed))) {
    stop(name, " levels must share one string, boolean, or numeric type.",
         call. = FALSE)
  }

  if (identical(kind, "string")) {
    value <- vapply(seq_along(value), function(index) {
      .association_scalar_text(value[[index]], paste0(name, "[", index, "]"))
    }, character(1))
  } else if (identical(kind, "number")) {
    value <- as.numeric(value)
    if (any(!is.finite(value))) {
      stop(name, " numeric levels must be finite.", call. = FALSE)
    }
    value[value == 0] <- 0
  } else {
    value <- as.logical(value)
  }
  if (anyDuplicated(value)) {
    stop(name, " reference and positive levels must be distinct.",
         call. = FALSE)
  }
  list(type = kind, values = unname(value))
}

.association_privacy_unit <- function(value) {
  value <- tolower(as.character(unlist(value, use.names = FALSE)))
  if (length(value) != 1L || is.na(value) ||
      !value %in% c("row", "patient")) {
    stop("association privacy unit must be exactly row or patient.",
         call. = FALSE)
  }
  value
}

.association_unit_semantics <- function(privacy_unit) {
  if (identical(.association_privacy_unit(privacy_unit), "row")) {
    "row-one-hot/v1"
  } else {
    "patient-ever-positive/v1"
  }
}

.association_contract_payload <- function(
    outcome_column, exposure_column, outcome_levels, exposure_levels,
    privacy_unit) {
  outcome_column <- .association_column(outcome_column, "outcome column")
  exposure_column <- .association_column(exposure_column, "exposure column")
  if (identical(outcome_column, exposure_column)) {
    stop("Association outcome and exposure columns must be distinct.",
         call. = FALSE)
  }
  unit <- .association_privacy_unit(privacy_unit)
  list(
    contract = .ASSOCIATION_CONTRACT,
    schema = 1L,
    exposure = list(
      column = exposure_column,
      levels = .association_level_spec(
        exposure_levels, "association-exposure-levels")),
    outcome = list(
      column = outcome_column,
      levels = .association_level_spec(
        outcome_levels, "association-outcome-levels")),
    order = "exposure-major/outcome-minor",
    privacy_unit = unit,
    shape = c(3L, 3L),
    unit_semantics = .association_unit_semantics(unit),
    unknown = "all-other-values/v1"
  )
}

.association_canonical_json <- function(value) {
  canonical <- as.character(jsonlite::toJSON(
    value, auto_unbox = TRUE, null = "null", na = "null", digits = NA,
    always_decimal = TRUE, pretty = FALSE))
  raw <- charToRaw(enc2utf8(canonical))
  if (length(raw) > .ASSOCIATION_MAX_RESULT_BYTES) {
    stop("Canonical association contract exceeds 65536 bytes.",
         call. = FALSE)
  }
  canonical
}

.association_contract_sha256 <- function(
    outcome_column, exposure_column, outcome_levels, exposure_levels,
    privacy_unit) {
  payload <- .association_contract_payload(
    outcome_column, exposure_column, outcome_levels, exposure_levels,
    privacy_unit)
  digest::digest(
    charToRaw(enc2utf8(.association_canonical_json(payload))),
    algo = "sha256", serialize = FALSE)
}

.association_sha256 <- function(value, name) {
  value <- as.character(unlist(value, use.names = FALSE))
  if (length(value) != 1L || is.na(value) ||
      !identical(value, tolower(value)) ||
      !grepl("^[0-9a-f]{64}$", value)) {
    stop(name, " must be one lowercase SHA-256 digest.", call. = FALSE)
  }
  value
}

.association_node_count <- function(value) {
  value <- suppressWarnings(as.numeric(unlist(value, use.names = FALSE)))
  if (length(value) != 1L || !is.finite(value) || value != floor(value) ||
      value < 1L || value > .ASSOCIATION_MAX_NODES) {
    stop("association-n-nodes must be an integer in [1, 65536].",
         call. = FALSE)
  }
  as.integer(value)
}

.association_job_payload <- function(
    association_contract_sha256, runner_abi, runner_sha256, n_nodes) {
  contract_sha <- .association_sha256(
    association_contract_sha256, "association-contract-sha256")
  abi <- suppressWarnings(as.numeric(unlist(runner_abi, use.names = FALSE)))
  if (length(abi) != 1L || !is.finite(abi) || abi != 3) {
    stop("Association jobs require runner_abi=3.", call. = FALSE)
  }
  list(
    schema = 1L,
    "association-contract-sha256" = contract_sha,
    runner_abi = 3L,
    runner_sha256 = .association_sha256(
      runner_sha256, "runner_sha256"),
    n_nodes = .association_node_count(n_nodes),
    dp_track = "association"
  )
}

.association_job_sha256 <- function(
    association_contract_sha256, runner_abi, runner_sha256, n_nodes) {
  payload <- .association_job_payload(
    association_contract_sha256, runner_abi, runner_sha256, n_nodes)
  digest::digest(
    charToRaw(enc2utf8(.association_canonical_json(payload))),
    algo = "sha256", serialize = FALSE)
}

.association_common_privacy_unit <- function(capabilities) {
  if (!is.list(capabilities) || !length(capabilities)) {
    stop("Association could not verify the federation privacy unit.",
         call. = FALSE)
  }
  units <- vapply(capabilities, function(capability) {
    value <- if (is.list(capability)) capability[["privacy_unit"]] else NULL
    value <- tolower(as.character(unlist(value, use.names = FALSE)))
    if (length(value) == 1L && !is.na(value) &&
        value %in% c("row", "patient")) value else ""
  }, character(1))
  if (any(!nzchar(units)) || length(unique(units)) != 1L) {
    stop("All association nodes must declare the same row/patient privacy unit.",
         call. = FALSE)
  }
  units[[1L]]
}

.assert_association_capability <- function(conns) {
  n_nodes <- .association_node_count(length(conns))
  expected_hash <- .compute_local_runner_hash()
  raw_caps <- tryCatch(
    DSI::datashield.aggregate(
      conns, expr = call("flowerGetCapabilitiesDS", "none", "runtime")),
    error = function(e) {
      stop("Could not verify the trusted association runtime: ",
           conditionMessage(e), call. = FALSE)
    })
  capabilities <- .validate_runner_compatibility(
    raw_caps, conns, expected_hash)
  unavailable <- character()
  for (index in seq_along(capabilities)) {
    association <- if (is.list(capabilities[[index]])) {
      capabilities[[index]][["association"]]
    } else NULL
    shape <- if (is.list(association)) {
      suppressWarnings(as.integer(unlist(
        association[["shape"]], use.names = FALSE)))
    } else integer()
    ready <- is.list(association) &&
      identical(as.character(association[["contract"]] %||% ""),
                .ASSOCIATION_CONTRACT) &&
      identical(as.character(association[["result_contract"]] %||% ""),
                .ASSOCIATION_RESULT_CONTRACT) &&
      identical(as.character(association[["mechanism"]] %||% ""),
                .ASSOCIATION_MECHANISM) &&
      identical(as.character(association[["execution_profile"]] %||% ""),
                .ASSOCIATION_EXECUTION_PROFILE) &&
      identical(as.character(association[["order"]] %||% ""),
                "exposure-major/outcome-minor") &&
      identical(shape, c(3L, 3L)) &&
      identical(as.character(unlist(
        association[["privacy_units"]], use.names = FALSE)),
        c("row", "patient")) &&
      isTRUE(association[["pooled_only"]]) &&
      identical(as.character(
        association[["availability_semantics"]] %||% ""),
        "fresh-executable-node-probe") &&
      isTRUE(association[["probed"]]) &&
      isTRUE(association[["available"]])
    if (!ready) unavailable <- c(unavailable, names(capabilities)[[index]])
  }
  if (length(unavailable)) {
    stop("Verified trusted association runtime is unavailable on: ",
         paste(unavailable, collapse = ", "), ".", call. = FALSE)
  }
  list(
    capabilities = capabilities,
    privacy_unit = .association_common_privacy_unit(capabilities),
    runner_abi = 3L,
    runner_sha256 = expected_hash,
    n_nodes = n_nodes)
}

.association_result_scalar <- function(value, name, positive = FALSE) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      (isTRUE(positive) && value <= 0)) {
    stop("Association result has an invalid ", name, ".", call. = FALSE)
  }
  as.numeric(value)
}

.association_result_table <- function(value) {
  if (!is.list(value) || length(value) != 3L) {
    stop("Association result has an invalid 3x3 table.", call. = FALSE)
  }
  rows <- lapply(value, function(row) {
    cells <- unlist(row, use.names = FALSE)
    if (!is.numeric(cells) || length(cells) != 3L ||
        any(!is.finite(cells)) || any(cells < 0)) {
      stop("Association result has an invalid 3x3 table.", call. = FALSE)
    }
    as.numeric(cells)
  })
  do.call(rbind, rows)
}

.association_result_measures <- function(value) {
  expected <- c(
    "odds_ratio", "prevalence_difference", "prevalence_exposed",
    "prevalence_ratio", "prevalence_unexposed")
  if (!is.list(value) || is.null(names(value)) ||
      anyDuplicated(names(value)) || !setequal(names(value), expected)) {
    stop("Association result has invalid descriptive measures.",
         call. = FALSE)
  }
  for (name in expected) {
    item <- value[[name]]
    if (!is.null(item) &&
        (!is.numeric(item) || length(item) != 1L || !is.finite(item))) {
      stop("Association result has invalid descriptive measures.",
           call. = FALSE)
    }
    if (!is.null(item)) {
      valid_domain <- if (name %in% c(
          "prevalence_exposed", "prevalence_unexposed")) {
        item >= 0 && item <= 1
      } else if (identical(name, "prevalence_difference")) {
        item >= -1 && item <= 1
      } else {
        item >= 0
      }
      if (!isTRUE(valid_domain)) {
        stop("Association result has invalid descriptive measures.",
             call. = FALSE)
      }
    }
  }
  value[expected]
}

.association_commit_marker <- function(results_dir, available) {
  path <- file.path(results_dir, "history.json")
  info <- file.info(path)
  if (!file.exists(path) || is.na(info$isdir) || isTRUE(info$isdir) ||
      is.na(info$size) || info$size < 1 || info$size > 4096L) return(FALSE)
  history <- tryCatch(
    jsonlite::fromJSON(path, simplifyVector = FALSE),
    error = function(e) NULL)
  if (!is.list(history) || length(history) != 1L ||
      !is.list(history[[1L]]) || is.null(names(history[[1L]])) ||
      anyDuplicated(names(history[[1L]])) ||
      !setequal(names(history[[1L]]), c("available", "round"))) {
    return(FALSE)
  }
  round <- history[[1L]][["round"]]
  valid_round <- (is.integer(round) || is.double(round)) &&
    !is.object(round) && !is.logical(round) && length(round) == 1L &&
    is.finite(round) && round == 1
  isTRUE(valid_round) &&
    identical(history[[1L]][["available"]], available)
}

.read_association_result <- function(
    results_dir, association_contract_sha256, association_job_sha256,
    n_nodes, privacy_unit) {
  path <- file.path(results_dir, "association.json")
  if (!file.exists(path)) return(NULL)
  info <- file.info(path)
  if (is.na(info$isdir) || isTRUE(info$isdir) || is.na(info$size) ||
      info$size < 1 || info$size > .ASSOCIATION_MAX_RESULT_BYTES) {
    stop("Association output failed its bounded pooled-only contract.",
         call. = FALSE)
  }
  value <- tryCatch(
    jsonlite::fromJSON(path, simplifyVector = FALSE),
    error = function(e) NULL)
  base_fields <- c(
    "available", "association_contract", "association_contract_sha256",
    "association_job_sha256", "contract", "n_nodes", "pooled_only",
    "privacy", "schema", "unit_semantics")
  if (!is.list(value) || is.null(names(value)) || anyDuplicated(names(value)) ||
      !(identical(value[["available"]], TRUE) ||
        identical(value[["available"]], FALSE))) {
    stop("Association output failed its bounded pooled-only contract.",
         call. = FALSE)
  }
  expected_fields <- if (isTRUE(value[["available"]])) {
    c(base_fields, "measures", "noise_sd_pooled", "table_dp")
  } else base_fields
  expected_contract_sha <- .association_sha256(
    association_contract_sha256, "association-contract-sha256")
  expected_job_sha <- .association_sha256(
    association_job_sha256, "association-job-sha256")
  expected_nodes <- .association_node_count(n_nodes)
  expected_semantics <- .association_unit_semantics(privacy_unit)
  privacy <- value[["privacy"]]
  privacy_fields <- c("adjacency", "mechanism", "scope", "sticky")
  valid_privacy <- is.list(privacy) && !is.null(names(privacy)) &&
    !anyDuplicated(names(privacy)) && setequal(names(privacy), privacy_fields) &&
    identical(as.character(privacy[["adjacency"]] %||% ""), "replace-one") &&
    identical(as.character(privacy[["mechanism"]] %||% ""),
              .ASSOCIATION_MECHANISM) &&
    identical(as.character(privacy[["scope"]] %||% ""),
              "per-job-node-dp") &&
    identical(privacy[["sticky"]], TRUE)
  actual_nodes <- suppressWarnings(as.numeric(unlist(
    value[["n_nodes"]], use.names = FALSE)))
  raw_nodes <- value[["n_nodes"]]
  raw_schema <- value[["schema"]]
  valid_nodes_type <- (is.integer(raw_nodes) || is.double(raw_nodes)) &&
    !is.object(raw_nodes) && !is.logical(raw_nodes)
  valid_schema <- (is.integer(raw_schema) || is.double(raw_schema)) &&
    !is.object(raw_schema) && !is.logical(raw_schema) &&
    length(raw_schema) == 1L && is.finite(raw_schema) && raw_schema == 1
  if (!setequal(names(value), expected_fields) ||
      !identical(value[["association_contract"]], .ASSOCIATION_CONTRACT) ||
      !identical(value[["association_contract_sha256"]],
                 expected_contract_sha) ||
      !identical(value[["association_job_sha256"]], expected_job_sha) ||
      !identical(value[["contract"]], .ASSOCIATION_RESULT_CONTRACT) ||
      !valid_nodes_type || length(actual_nodes) != 1L ||
      !is.finite(actual_nodes) ||
      actual_nodes != expected_nodes ||
      !identical(value[["pooled_only"]], TRUE) ||
      !valid_schema ||
      !identical(value[["unit_semantics"]], expected_semantics) ||
      !valid_privacy ||
      !.association_commit_marker(results_dir, value[["available"]])) {
    stop("Association output failed its bounded pooled-only contract.",
         call. = FALSE)
  }
  if (isTRUE(value[["available"]])) {
    value[["table_dp"]] <- .association_result_table(value[["table_dp"]])
    value[["measures"]] <- .association_result_measures(value[["measures"]])
    value[["noise_sd_pooled"]] <- .association_result_scalar(
      value[["noise_sd_pooled"]], "pooled noise scale", positive = TRUE)
  }
  value
}

#' Differentially-private pooled binary association
#'
#' Builds one bounded 3x3 exposure/outcome table at each node, applies one
#' server-owned sticky Gaussian release, and returns only the pooled table and
#' descriptive prevalence measures. Values other than the two ordered public
#' levels are retained in the \code{unknown} row or column. No exact or per-node
#' counts leave a data node.
#'
#' This is a descriptive association, not a causal or adjusted effect. Under
#' patient privacy, each axis means "ever positive" across all rows for that
#' patient; exposure and outcome need not occur at the same visit.
#'
#' @param conns DSI connections.
#' @param outcome One outcome column name.
#' @param exposure One exposure column name.
#' @param outcome_levels Ordered public \code{c(reference, positive)} levels.
#' @param exposure_levels Ordered public \code{c(reference, positive)} levels.
#' @param data Optional server-side data source.
#' @param resource Optional Opal resource name.
#' @param symbol Optional assigned server-side symbol.
#' @param verbose Show Flower output.
#' @param silent Suppress progress messages.
#' @param allow_insecure_http Exact connection names allowed to use HTTP.
#' @return A \code{dsflower_association}. If a complete release is unavailable,
#'   \code{available} is false and table/measures/noise are null.
#' @export
ds.flower.associate <- function(
    conns, outcome, exposure, outcome_levels, exposure_levels,
    data = NULL, resource = NULL, symbol = NULL,
    verbose = FALSE, silent = FALSE,
    allow_insecure_http = getOption(
      "dsflower.dsi_allow_insecure_http", character())) {
  outcome <- .association_column(outcome, "outcome")
  exposure <- .association_column(exposure, "exposure")
  if (identical(outcome, exposure)) {
    stop("Association outcome and exposure columns must be distinct.",
         call. = FALSE)
  }
  outcome_spec <- .association_level_spec(outcome_levels, "outcome_levels")
  exposure_spec <- .association_level_spec(
    exposure_levels, "exposure_levels")
  suppressWarnings(.validate_dsi_transport_security(
    conns, allow_insecure_http = allow_insecure_http))
  capability <- .assert_association_capability(conns)
  privacy_unit <- capability$privacy_unit
  contract_sha <- .association_contract_sha256(
    outcome, exposure, outcome_spec, exposure_spec, privacy_unit)
  job_sha <- .association_job_sha256(
    contract_sha, capability$runner_abi, capability$runner_sha256,
    capability$n_nodes)
  .require_flwr_cli()
  if (is.null(data) && is.null(resource) && is.null(symbol)) symbol <- "D"

  old_opt <- options(dsflower.silent = isTRUE(silent))
  on.exit(options(old_opt), add = TRUE)
  flower <- ds.flower.connect(
    conns, data = data, resource = resource, symbol = symbol)
  conns <- flower$conns
  handle_symbol <- flower$symbol
  on.exit({
    tryCatch(ds.flower.link.down(conns), error = function(e) NULL)
    tryCatch(ds.flower.nodes.cleanup(conns, handle_symbol),
             error = function(e) NULL)
    tryCatch(ds.flower.disconnect(flower), error = function(e) NULL)
  }, add = TRUE)

  prepare <- list(
    "dp-track" = "association",
    "data_type" = "tabular",
    "num-server-rounds" = 1L,
    "association-outcome-levels" = outcome_spec,
    "association-exposure-levels" = exposure_spec,
    "association-contract-sha256" = contract_sha,
    "association-n-nodes" = capability$n_nodes,
    "association-job-sha256" = job_sha)
  ds.flower.nodes.prepare(
    conns, handle_symbol, target_column = outcome,
    feature_columns = exposure, run_config = prepare)

  results_root <- file.path(tempdir(), "dsflower_results")
  dir.create(results_root, recursive = TRUE, showWarnings = FALSE)
  results_dir <- tempfile(pattern = "association_", tmpdir = results_root)
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  semantics <- .association_unit_semantics(privacy_unit)
  config <- c(
    .toml_kv("dp-track", "association"),
    .toml_kv("association-contract", .ASSOCIATION_CONTRACT),
    .toml_kv("association-contract-sha256", contract_sha),
    .toml_kv("association-job-sha256", job_sha),
    paste0("association-n-nodes = ", capability$n_nodes),
    .toml_kv("association-privacy-unit", privacy_unit),
    .toml_kv("association-unit-semantics", semantics),
    "num-server-rounds = 1",
    paste0("min-train-nodes = ", capability$n_nodes),
    "round-timeout = 3600.0",
    .toml_kv("results-dir", results_dir))
  app_dir <- .build_submission_app(
    list(pkg_dir = NULL, track = "association"),
    config, results_dir, vision = FALSE)
  ds.flower.link.up(conns, allow_insecure_http = allow_insecure_http)
  ds.flower.nodes.ensure(conns, handle_symbol, torch_backend = NULL)

  superlink <- .dsflower_client_env$.superlink
  run <- .run_flwr_with_artifact_watchdog(
    command = .client_flwr_cmd(),
    args = c("run", app_dir, "dsflower", "--stream"),
    env = .client_venv_env(extra = c(
      FLWR_HOME = superlink$flwr_home, PYTHONUNBUFFERED = "1")),
    results_dir = results_dir, num_rounds = 1L, expect_artifacts = FALSE)
  stdout <- gsub("\033\\[[0-9;]*m", "", run$stdout)
  stderr <- gsub("\033\\[[0-9;]*m", "", run$stderr)
  if (isTRUE(verbose) && nzchar(stdout)) message(stdout)
  if (!identical(as.integer(run$status), 0L)) {
    stop("Federated private association failed (status ", run$status, ").",
         call. = FALSE)
  }
  released <- .read_association_result(
    results_dir, contract_sha, job_sha, capability$n_nodes, privacy_unit)
  if (is.null(released)) {
    stop("Federated private association produced no pooled DP result.",
         call. = FALSE)
  }
  table <- released[["table_dp"]] %||% NULL
  if (!is.null(table)) {
    dimnames(table) <- list(
      exposure = c("reference", "positive", "unknown"),
      outcome = c("reference", "positive", "unknown"))
  }
  recipe <- list(
    contract = .ASSOCIATION_CONTRACT,
    association_contract_sha256 = contract_sha,
    association_job_sha256 = job_sha,
    runner_abi = capability$runner_abi,
    runner_sha256 = capability$runner_sha256,
    n_nodes = capability$n_nodes,
    dp_track = "association",
    outcome = list(column = outcome, levels = outcome_spec),
    exposure = list(column = exposure, levels = exposure_spec),
    privacy_unit = privacy_unit,
    unit_semantics = semantics)
  structure(list(
    available = released[["available"]],
    table_dp = table,
    measures = released[["measures"]] %||% NULL,
    noise_sd_pooled = released[["noise_sd_pooled"]] %||% NULL,
    n_nodes = released[["n_nodes"]],
    pooled_only = TRUE,
    privacy = released[["privacy"]],
    recipe = recipe,
    results_dir = results_dir,
    stdout = stdout,
    stderr = stderr),
    class = "dsflower_association")
}

#' Print a pooled private association
#' @param x A \code{dsflower_association}.
#' @param ... Ignored.
#' @export
print.dsflower_association <- function(x, ...) {
  cat("Federated private binary association\n")
  cat("  Sites:   ", x$n_nodes, "\n")
  cat("  Unit:    ", x$recipe$privacy_unit, "\n")
  cat("  Privacy: node-DP, pooled result only\n")
  if (!isTRUE(x$available)) {
    cat("  Available: no complete private association release\n")
    return(invisible(x))
  }
  print(x$table_dp)
  cat("  Measures:\n")
  for (name in names(x$measures)) {
    value <- x$measures[[name]]
    cat("    ", name, ": ",
        if (is.null(value)) "NA" else format(value, digits = 5), "\n",
        sep = "")
  }
  invisible(x)
}
