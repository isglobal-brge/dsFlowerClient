# Module: Hook federated run (researcher side). Upload an untrusted training
# module and request execution through the node-side HookApp gate.
# The researcher provides a Python package exposing:
#   initial_arrays(cfg, input_dim) -> list[np.ndarray]
#   local_update(global_arrays, X, y, cfg) -> list[np.ndarray]
# Arbitrary code cannot receive generic per-sample DP-SGD guarantees. The node
# executes the module only when the custodian enabled HookApps and attested the
# required filesystem/network sandbox and minimum-duration timing envelope.
# Otherwise the module is not executed and the run is reported unavailable;
# the incoming public model is not accepted as a trained release. Archive
# validation, scanning and hash pinning are additional integrity controls.

.HOOK_APP_PARAMS_MAX_DEPTH <- 8L
.HOOK_APP_PARAMS_MAX_ITEMS <- 2048L
.HOOK_APP_PARAMS_MAX_BYTES <- 65536L
.HOOK_APP_PARAMS_MAX_KEY_BYTES <- 128L
.HOOK_APP_PARAMS_MAX_STRING_BYTES <- 4096L

#' @keywords internal
.hook_app_reserved_key <- function(key) {
  normalized <- gsub("([a-z0-9])([A-Z])", "\\1_\\2", key, perl = TRUE)
  normalized <- gsub("[-.]", "_", tolower(normalized))
  startsWith(normalized, "privacy") || startsWith(normalized, "dp") ||
    normalized %in% c(
      "privacy", "dp", "epsilon", "delta", "clipping_norm",
      "user_module", "app_params", "app_params_b64", "app_params_sha256",
      "round", "round_index", "server_round", "num_rounds",
      "num_server_rounds", "task", "task_type", "num_classes",
      "runtime_profile", "backend", "requirements", "requirement",
      "dependencies", "dependency", "pip", "pythonpath", "python_path"
    ) ||
    grepl(
      "(^|_)(path|dir|directory|file|filename|secret|token|password|credential|requirements?|dependencies?)($|_)",
      normalized, perl = TRUE
    ) ||
    grepl(
      "(^|_)(privacy|dp|epsilon|delta|noise|sensitivity|accountant|clip|clipping)($|_)",
      normalized, perl = TRUE
    )
}

#' @keywords internal
.validate_hook_app_key <- function(key) {
  if (!is.character(key) || length(key) != 1L || is.na(key)) {
    stop("app_params object keys must be non-missing UTF-8 strings.",
         call. = FALSE)
  }
  key <- enc2utf8(key)
  if (!nzchar(key) || is.na(iconv(key, from = "UTF-8", to = "UTF-8",
                                  sub = NA_character_)) ||
      nchar(key, type = "bytes") > .HOOK_APP_PARAMS_MAX_KEY_BYTES ||
      grepl("[[:cntrl:]/\\\\]", key, perl = TRUE)) {
    stop("app_params object keys must be safe UTF-8 strings of at most 128 bytes.",
         call. = FALSE)
  }
  if (.hook_app_reserved_key(key)) {
    stop("app_params contains a reserved or path/security-related key: ", key,
         call. = FALSE)
  }
  key
}

#' @keywords internal
.canonical_hook_app_value <- function(value, depth, state, top = FALSE) {
  if (depth > .HOOK_APP_PARAMS_MAX_DEPTH) {
    stop("app_params exceeds the maximum nesting depth (8).", call. = FALSE)
  }
  if (is.object(value) || is.factor(value) || is.data.frame(value) ||
      is.matrix(value) ||
      is.array(value) || is.raw(value) || is.complex(value) ||
      inherits(value, c("Date", "POSIXt"))) {
    stop("app_params accepts only JSON-like null, scalar, array and object values.",
         call. = FALSE)
  }

  if (is.logical(value) || is.integer(value) || is.numeric(value) ||
      is.character(value)) {
    if (length(value) != 1L) {
      values <- as.list(value)
      names(values) <- names(value)
      return(.canonical_hook_app_value(values, depth, state, top = top))
    }
  }

  state$items <- state$items + 1L
  if (state$items > .HOOK_APP_PARAMS_MAX_ITEMS) {
    stop("app_params exceeds the maximum item count (2048).", call. = FALSE)
  }

  if (is.null(value)) return(NULL)
  if (is.logical(value) || is.integer(value) || is.numeric(value) ||
      is.character(value)) {
    if (is.na(value)) {
      stop("app_params scalar values cannot be missing.", call. = FALSE)
    }
    if (is.numeric(value) && !is.finite(value)) {
      stop("app_params numeric values must be finite.", call. = FALSE)
    }
    if (is.character(value)) {
      value <- enc2utf8(value)
      if (is.na(iconv(value, from = "UTF-8", to = "UTF-8",
                      sub = NA_character_)) ||
          nchar(value, type = "bytes") > .HOOK_APP_PARAMS_MAX_STRING_BYTES ||
          grepl("[[:cntrl:]]", value, perl = TRUE) ||
          grepl("(^~[/\\\\])|[/\\\\]|^[A-Za-z]:", value, perl = TRUE)) {
        stop("app_params strings must be safe UTF-8 non-path values of at most 4096 bytes.",
             call. = FALSE)
      }
    }
    return(value)
  }
  if (!is.list(value)) {
    stop("app_params accepts only JSON-like null, scalar, array and object values.",
         call. = FALSE)
  }

  object_names <- names(value)
  is_object <- isTRUE(top) ||
    (!is.null(object_names) && length(object_names) == length(value) &&
       all(nzchar(object_names)))
  if (!is_object && !is.null(object_names) && any(nzchar(object_names))) {
    stop("app_params containers cannot mix named and unnamed elements.",
         call. = FALSE)
  }
  if (is_object) {
    if (is.null(object_names)) object_names <- rep.int("", length(value))
    if (length(value) && (any(!nzchar(object_names)) || anyDuplicated(object_names))) {
      stop("app_params objects require unique, non-empty keys.", call. = FALSE)
    }
    safe_names <- vapply(object_names, .validate_hook_app_key, character(1))
    order_index <- order(safe_names, method = "radix")
    out <- structure(vector("list", length(value)),
                     names = unname(safe_names[order_index]))
    for (i in seq_along(order_index)) {
      out[i] <- list(.canonical_hook_app_value(
        value[[order_index[[i]]]], depth + 1L, state, top = FALSE))
    }
    return(out)
  }

  out <- vector("list", length(value))
  for (i in seq_along(value)) {
    out[i] <- list(.canonical_hook_app_value(
      value[[i]], depth + 1L, state, top = FALSE))
  }
  out
}

#' Canonicalise public HookApp parameters for the cross-runtime contract
#' @keywords internal
.canonical_hook_app_params <- function(app_params = list()) {
  if (!is.list(app_params)) {
    stop("app_params must be a named JSON-like object.", call. = FALSE)
  }
  state <- new.env(parent = emptyenv())
  state$items <- 0L
  canonical <- .canonical_hook_app_value(
    app_params, depth = 0L, state = state, top = TRUE)
  json <- as.character(jsonlite::toJSON(
    canonical, auto_unbox = TRUE, null = "null", na = "null",
    digits = NA, always_decimal = TRUE, pretty = FALSE))
  bytes <- charToRaw(enc2utf8(json))
  if (length(bytes) > .HOOK_APP_PARAMS_MAX_BYTES) {
    stop("app_params canonical JSON exceeds 65536 bytes.", call. = FALSE)
  }
  b64 <- gsub("[\r\n]", "", jsonlite::base64_enc(bytes))
  list(
    value = canonical,
    json = json,
    b64 = b64,
    sha256 = digest::digest(bytes, algo = "sha256", serialize = FALSE)
  )
}

#' @keywords internal
.tier2_skeleton_dir <- function() {
  p <- system.file("flower_app", "dsflower_runner", package = "dsFlowerClient")
  if (nzchar(p) && dir.exists(p)) return(p)
  alt <- file.path("inst", "flower_app", "dsflower_runner")
  if (dir.exists(alt)) return(normalizePath(alt))
  stop("Bundled canonical runner (dsflower_runner) not found.", call. = FALSE)
}

#' Validate a Python top-level package name
#' @keywords internal
.validate_user_module_name <- function(name) {
  if (!is.character(name) || length(name) != 1L || is.na(name) ||
      !grepl("^[A-Za-z_][A-Za-z0-9_]*$", name)) {
    stop("The hook package directory must be a valid Python module name ",
         "([A-Za-z_][A-Za-z0-9_]*).", call. = FALSE)
  }
  name
}

#' Upload the researcher's training package to the nodes (zip + chunked push +
#' verify + exfiltration scan + install). Returns the upload token + package name.
#' @keywords internal
.upload_user_module <- function(conns, user_pkg_dir, chunk_bytes = 262144L) {
  chunk_bytes <- .dsi_raw_chunk_bytes(chunk_bytes)
  user_pkg_dir <- normalizePath(user_pkg_dir, mustWork = TRUE)
  pkg_name <- .validate_user_module_name(basename(user_pkg_dir))
  if (!file.exists(file.path(user_pkg_dir, "__init__.py"))) {
    stop("The hook package must contain __init__.py.", call. = FALSE)
  }
  parent <- dirname(user_pkg_dir)
  zipfile <- tempfile(fileext = ".zip")
  on.exit(unlink(zipfile), add = TRUE)
  wd <- getwd()
  setwd(parent)
  on.exit(setwd(wd), add = TRUE)
  utils::zip(zipfile, files = pkg_name, flags = "-q -r")
  setwd(wd)

  n <- file.size(zipfile)
  if (length(n) != 1L || is.na(n) || n <= 0) {
    stop("The hook package archive is empty.", call. = FALSE)
  }
  sha <- digest::digest(file = zipfile, algo = "sha256")
  token <- .new_capability_token("usr")
  installed <- FALSE
  on.exit({
    if (!installed) {
      tryCatch(DSI::datashield.aggregate(
        conns, call("flowerAppDeleteDS", token)), error = function(e) NULL)
    }
  }, add = TRUE)
  .push_app_archive(conns, zipfile, token, chunk_bytes)
  .install_app_archive(
    conns, token, sha, n, "Hook module install/scan")
  installed <- TRUE
  list(token = token, sha256 = sha, package = pkg_name)
}

#' Pin one verified HookApp package with an explicit per-node ACK
#' @keywords internal
.pin_user_module <- function(conns, handle_symbol, upload) {
  expr <- call("flowerTier2PinDS", handle_symbol, upload$token)
  .dsi_retry_exact_aggregate(
    conns, expr,
    validate = function(value, node) {
      if (!is.list(value) ||
          !identical(names(value), c("ok", "pinned", "user_module"))) {
        return(FALSE)
      }
      pinned <- as.character(unlist(value$pinned, use.names = FALSE))
      isTRUE(value$ok) &&
        identical(as.character(value$user_module), upload$package) &&
        length(pinned) == 2L &&
        setequal(pinned, c("dsflower_runner", upload$package))
    },
    operation = "Hook module pin")
}

#' Fail before upload/staging when a node publicly reports that arbitrary Hook
#' execution is administratively disabled. Runtime sandbox self-tests still run
#' node-side and fail closed; this preflight removes the common silent no-op UX.
#' @keywords internal
.assert_hook_execution_configured <- function(capabilities, conns) {
  if (!is.list(capabilities) || length(capabilities) != length(conns)) {
    stop("Could not verify HookApp execution readiness on every node.",
         call. = FALSE)
  }
  node_names <- names(conns) %||% as.character(seq_along(conns))
  failed <- character()
  for (i in seq_along(capabilities)) {
    cap <- capabilities[[i]]
    if (!is.list(cap) || !isTRUE(cap$hook_execution_configured)) {
      missing <- c(
        if (!is.list(cap) || !isTRUE(cap$hook_enabled)) "enabled",
        if (!is.list(cap) || !isTRUE(cap$hook_sandbox_attested)) "sandbox",
        if (!is.list(cap) ||
            !isTRUE(cap$hook_resource_isolation_attested)) "resources",
        if (!is.list(cap) ||
            !isTRUE(cap$hook_time_envelope_configured)) "timing")
      failed <- c(failed, paste0(node_names[[i]], " [",
                                  paste(missing, collapse = ","), "]"))
    }
  }
  if (length(failed)) {
    stop("HookApp execution is not configured on: ",
         paste(failed, collapse = "; "),
         ". Use a declarative built-in app, or ask each node administrator to ",
         "enable and attest the Hook sandbox/resource/timing policy.",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Build the Hook runner app dir (bundled dsflower_runner + the user package +
#' pyproject). The user package is included so the researcher-side ServerApp can
#' call initial_arrays(); on the node it is still hash-verified by the integrity
#' hook against the node-computed upload hash (so it cannot be tampered with).
#' @keywords internal
.build_tier2_app <- function(user_module, user_pkg_dir, n_features, results_dir,
                             n_nodes, num_rounds, task, num_classes,
                             app_params_b64, app_params_sha256) {
  user_module <- .validate_user_module_name(user_module)
  app_dir <- file.path(tempdir(), "dsflower_tier2_app", "dsflower-tier2")
  if (dir.exists(app_dir)) unlink(app_dir, recursive = TRUE)
  dir.create(app_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(.tier2_skeleton_dir(), app_dir, recursive = TRUE)
  file.copy(normalizePath(user_pkg_dir), app_dir, recursive = TRUE)
  unlink(file.path(app_dir, basename(user_pkg_dir), "__pycache__"),
         recursive = TRUE)

  config_lines <- c(
    paste0("num-server-rounds = ", as.integer(num_rounds)),
    paste0("num-features = ", as.integer(n_features)),
    paste0("num-classes = ", as.integer(num_classes)),
    .toml_kv("task-type", task),
    .toml_kv("user-module", user_module),
    .toml_kv("app-params-b64", app_params_b64),
    .toml_kv("app-params-sha256", app_params_sha256),
    paste0("min-train-nodes = ", as.integer(n_nodes)),
    .toml_kv("results-dir", results_dir)
  )
  toml <- paste0(
    '[build-system]\nrequires = ["hatchling"]\n',
    'build-backend = "hatchling.build"\n\n',
    '[project]\nname = "dsflower-tier2"\nversion = "1.0.0"\n',
    'description = "dsFlower Tier-2 trusted runner"\nlicense = "MIT"\n',
    'dependencies = [', paste0('"', .harness_dependencies(), '"',
                               collapse = ", "), ']\n\n',
    '[tool.hatch.build.targets.wheel]\npackages = ["dsflower_runner", "',
    user_module, '"]\n\n',
    '[tool.flwr.app]\npublisher = "dsflower"\n\n',
    '[tool.flwr.app.components]\n',
    'serverapp = "dsflower_runner.server_app:app"\n',
    'clientapp = "dsflower_runner.client_app:app"\n\n',
    '[tool.flwr.app.config]\n', paste(config_lines, collapse = "\n"), '\n'
  )
  writeLines(toml, file.path(app_dir, "pyproject.toml"))
  app_dir
}

#' Request a HookApp run through the node-side egress policy
#'
#' The package exposes \code{initial_arrays()} and \code{local_update()} but is
#' never trusted by the node. When the custodian-enabled sandbox and timing gates
#' are available, the node clips the complete update and applies its configured
#' Gaussian output mechanism (optionally over disjoint blocks). Publicly absent
#' gates cause a client preflight error before upload. If a gate disappears after
#' that preflight, the node refuses to open private data and marks the unchanged
#' update unavailable; the coordinator does not report it as a trained model.
#' Hash verification and static scanning do not provide
#' \code{nn.Module}/per-sample DP-SGD granularity for arbitrary code. The timing
#' envelope is defense in depth, not a formal constant-time guarantee.
#'
#' @param conns DSI connections object.
#' @param user_app_dir Character; path to the researcher's training package dir
#'   (a folder with __init__.py exposing initial_arrays + local_update).
#' @param target Character; target column name.
#' @param features Character vector; feature column names.
#' @param symbol Character; server-side data handle symbol (default "D").
#' @param num_rounds Integer; federated rounds (default 1).
#' @param task Character; supervised task type used by node disclosure checks.
#' @param verbose Logical.
#' @param target_levels Optional ordered public label vocabulary for
#'   classification HookApps. Missing or unknown values map to public code zero.
#' @param target_bounds Required public \code{list(lower=..., upper=...)} for
#'   regression/count HookApps.
#' @param allow_insecure_http Character vector of exact connection names allowed
#'   to use plaintext HTTP. Empty by default. This exception does not provide
#'   transport security; use it only behind an independently trusted network.
#' @param app_params Named JSON-like list of public HookApp hyperparameters.
#'   Values may contain bounded nested objects/arrays and finite scalars. Privacy,
#'   runtime, dependency, credential and filesystem-path fields are reserved. The
#'   canonical value is hash-pinned and supplied to both app hooks.
#' @return A \code{dsflower_run} object. A public readiness failure is rejected
#'   before upload. If no private release is available, the completed object has
#'   \code{available = FALSE} and contains no fallback model artifact.
#' @export
ds.flower.hook.run <- function(conns, user_app_dir, target, features,
                               symbol = "D", num_rounds = 1L,
                               task = c("classification", "regression", "count"),
                               verbose = TRUE, target_levels = NULL,
                               target_bounds = NULL,
                               allow_insecure_http = getOption(
                                 "dsflower.dsi_allow_insecure_http", character()),
                               app_params = list()) {
  task <- match.arg(task)
  public_app_params <- .canonical_hook_app_params(app_params)
  n_target_classes <- if (is.null(target_levels)) 2L else length(target_levels)
  public_target <- .validate_public_target_spec(
    target_levels, target_bounds, task_type = task,
    n_classes = n_target_classes)
  if (!is.numeric(num_rounds) || is.logical(num_rounds)) {
    stop("num_rounds must be a single positive integer no greater than 500.",
         call. = FALSE)
  }
  rounds_value <- as.numeric(num_rounds)
  if (length(rounds_value) != 1L || is.na(rounds_value) || !is.finite(rounds_value) ||
      rounds_value < 1 || rounds_value %% 1 != 0 ||
      rounds_value > 500) {
    stop("num_rounds must be a single positive integer no greater than 500.",
         call. = FALSE)
  }
  num_rounds <- as.integer(rounds_value)
  if (!is.character(target) || length(target) != 1L || is.na(target) || !nzchar(target)) {
    stop("Hook runs require one non-empty target column.", call. = FALSE)
  }
  if (!is.character(features) || length(features) < 1L || anyNA(features) ||
      any(!nzchar(features)) || anyDuplicated(features)) {
    stop("Hook runs require unique, non-empty character feature columns.",
         call. = FALSE)
  }
  features <- enc2utf8(features)
  .require_flwr_cli()
  # Reject an accidental downgrade before handle creation, staging or upload;
  # link.up revalidates and emits the explicit-HTTP warning at tunnel startup.
  suppressWarnings(.validate_dsi_transport_security(
    conns, allow_insecure_http = allow_insecure_http))
  n_clients <- length(conns)
  n_features <- length(features)

  # Initialise the server-side Flower handle (flowerInitDS) from the assigned
  # data symbol; all run ops use this handle symbol.
  flower <- ds.flower.connect(conns, symbol = symbol)
  conns <- flower$conns
  hsym <- flower$symbol
  up <- NULL
  on.exit({
    tryCatch(ds.flower.link.down(conns), error = function(e) NULL)
    tryCatch(ds.flower.nodes.cleanup(conns, hsym), error = function(e) NULL)
    if (!is.null(up)) {
      tryCatch(DSI::datashield.aggregate(
        conns, call("flowerAppDeleteDS", up$token)), error = function(e) NULL)
    }
    tryCatch(ds.flower.disconnect(flower), error = function(e) NULL)
  }, add = TRUE)
  capabilities <- .assert_runner_compatibility(conns, hsym)
  .assert_hook_execution_configured(capabilities, conns)

  up <- .upload_user_module(conns, user_app_dir)
  if (verbose) message("  Hook module '", up$package, "' uploaded + scanned.")

  prepare_config <- list(
    "task-type" = task, "dp-track" = "egress",
    "num-server-rounds" = as.integer(num_rounds),
    "num-features" = as.integer(n_features),
    "num-classes" = as.integer(n_target_classes),
    "app-params-b64" = public_app_params$b64)
  if (!is.null(public_target$levels)) {
    prepare_config[["target-levels"]] <- public_target$levels
  }
  if (!is.null(public_target$bounds)) {
    prepare_config[["target-bounds"]] <- public_target$bounds
  }
  ds.flower.nodes.prepare(
    conns, hsym, target_column = target, feature_columns = features,
    run_config = prepare_config)

  pin <- .pin_user_module(conns, hsym, up)
  if (verbose) {
    message("  Pinned: ", paste(unique(unlist(lapply(pin, `[[`, "pinned"))),
                                 collapse = ", "), ".")
  }

  results_dir <- file.path(tempdir(), "dsflower_results",
                           format(Sys.time(), "%Y%m%d_%H%M%S"))
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  app_dir <- .build_tier2_app(
    up$package, user_app_dir, n_features, results_dir,
    n_clients, num_rounds, task, n_target_classes,
    public_app_params$b64, public_app_params$sha256)
  .ensure_client_framework("pytorch")

  # link.up owns the local SuperLink + DSI tunnel lifecycle; on.exit reverses it.
  ds.flower.link.up(conns, allow_insecure_http = allow_insecure_http)
  recipe <- structure(list(
    model = list(name = "tier2", template = "tier2", framework = "pytorch",
                 track = "egress"),
    strategy = list(name = "FedAvg", params = list()),
    num_rounds = as.integer(num_rounds),
    features = features, target_levels = public_target$levels,
    target_bounds = public_target$bounds,
    model_params = list(
      app_params = public_app_params$value,
      app_params_sha256 = public_app_params$sha256),
    data_kind = "tabular", evaluation_only = FALSE),
    class = "dsflower_recipe")

  ds.flower.nodes.ensure(conns, hsym)
  ds.flower.run.start(recipe, conns, app_dir = app_dir,
                      results_dir = results_dir, symbol = hsym,
                      verbose = verbose)
}

#' @rdname ds.flower.hook.run
#' @export
ds.flower.tier2.run <- function(conns, user_app_dir, target, features,
                                symbol = "D", num_rounds = 1L,
                                verbose = TRUE,
                                allow_insecure_http = getOption(
                                  "dsflower.dsi_allow_insecure_http", character())) {
  .Deprecated("ds.flower.hook.run")
  ds.flower.hook.run(
    conns = conns, user_app_dir = user_app_dir, target = target,
    features = features, symbol = symbol, num_rounds = num_rounds,
    task = "classification", verbose = verbose,
    allow_insecure_http = allow_insecure_http)
}
