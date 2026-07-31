# Module: Hook federated run (researcher side). Upload an untrusted training
# module and request execution through the node-side HookApp gate.
# The researcher provides a Python package exposing:
#   initial_arrays(cfg, input_dim) -> list[np.ndarray]
#   local_update(global_arrays, X, y, cfg) -> list[np.ndarray]
# Arbitrary code cannot receive generic per-sample DP-SGD guarantees. The node
# executes the module only when the custodian enabled HookApps and attested the
# required filesystem/network sandbox and timing envelope. Otherwise the module
# is not executed and the incoming public model is returned unchanged. Archive
# validation, scanning and hash pinning are additional integrity controls.

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
  if (!is.numeric(chunk_bytes) || length(chunk_bytes) != 1L || is.na(chunk_bytes) ||
      !is.finite(chunk_bytes) ||
      chunk_bytes <= 0 || chunk_bytes %% 1 != 0) {
    stop("chunk_bytes must be a single positive integer.", call. = FALSE)
  }
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
  raw <- readBin(zipfile, "raw", n)
  sha <- digest::digest(file = zipfile, algo = "sha256")
  token <- .new_capability_token("usr")
  installed <- FALSE
  on.exit({
    if (!installed) {
      tryCatch(DSI::datashield.aggregate(
        conns, call("flowerAppDeleteDS", token)), error = function(e) NULL)
    }
  }, add = TRUE)
  off <- 0
  while (off < n) {
    hi <- min(off + chunk_bytes, n)
    DSI::datashield.aggregate(
      conns, call("flowerAppPushDS", token, .app_enc_b64(raw[(off + 1):hi]), off))
    off <- hi
  }
  inst <- DSI::datashield.aggregate(
    conns, call("flowerAppInstallDS", token, sha))
  failed <- names(inst)[!vapply(inst, function(x) isTRUE(x$ok), logical(1))]
  if (length(failed) > 0) {
    stop("Hook module install/scan failed on: ",
         paste(failed, collapse = ", "), ".", call. = FALSE)
  }
  installed <- TRUE
  list(token = token, sha256 = sha, package = pkg_name)
}

#' Build the Hook runner app dir (bundled dsflower_runner + the user package +
#' pyproject). The user package is included so the researcher-side ServerApp can
#' call initial_arrays(); on the node it is still hash-verified by the integrity
#' hook against the node-computed upload hash (so it cannot be tampered with).
#' @keywords internal
.build_tier2_app <- function(user_module, user_pkg_dir, n_features, results_dir,
                             n_nodes, num_rounds) {
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
    .toml_kv("user-module", user_module),
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
#' Gaussian output mechanism (optionally over disjoint blocks). If any gate is
#' absent, the uploaded code is not executed and the operation returns the
#' incoming public model unchanged. Hash verification and static scanning do not
#' provide \code{nn.Module}/per-sample DP-SGD granularity for arbitrary code.
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
#' @param target_levels Optional ordered, exhaustive public label vocabulary for
#'   classification HookApps.
#' @param target_bounds Required public \code{list(lower=..., upper=...)} for
#'   regression/count HookApps.
#' @return A \code{dsflower_run} object. A successful Flower run may represent a
#'   deliberate unchanged-model no-op when the node disallows HookApp execution.
#' @export
ds.flower.hook.run <- function(conns, user_app_dir, target, features,
                               symbol = "D", num_rounds = 1L,
                               task = c("classification", "regression", "count"),
                               verbose = TRUE, target_levels = NULL,
                               target_bounds = NULL) {
  task <- match.arg(task)
  n_target_classes <- if (is.null(target_levels)) 2L else length(target_levels)
  public_target <- .validate_public_target_spec(
    target_levels, target_bounds, task_type = task,
    n_classes = n_target_classes)
  rounds_value <- suppressWarnings(as.numeric(num_rounds))
  if (length(rounds_value) != 1L || is.na(rounds_value) || !is.finite(rounds_value) ||
      rounds_value < 1 || rounds_value %% 1 != 0 ||
      rounds_value > .Machine$integer.max) {
    stop("num_rounds must be a single positive integer.", call. = FALSE)
  }
  num_rounds <- as.integer(rounds_value)
  if (!is.character(target) || length(target) != 1L || is.na(target) || !nzchar(target)) {
    stop("Hook runs require one non-empty target column.", call. = FALSE)
  }
  features <- as.character(features)
  if (length(features) < 1L || anyNA(features) || any(!nzchar(features))) {
    stop("Hook runs require explicit, non-empty feature columns.", call. = FALSE)
  }
  .require_flwr_cli()
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
  .assert_runner_compatibility(conns, hsym)

  up <- .upload_user_module(conns, user_app_dir)
  if (verbose) message("  Hook module '", up$package, "' uploaded + scanned.")

  prepare_config <- list(
    "task-type" = task, "dp-track" = "egress",
    "num-server-rounds" = as.integer(num_rounds),
    "num-features" = as.integer(n_features),
    "num-classes" = as.integer(n_target_classes))
  if (!is.null(public_target$levels)) {
    prepare_config[["target-levels"]] <- public_target$levels
  }
  if (!is.null(public_target$bounds)) {
    prepare_config[["target-bounds"]] <- public_target$bounds
  }
  ds.flower.nodes.prepare(
    conns, hsym, target_column = target, feature_columns = features,
    run_config = prepare_config)

  pin <- DSI::datashield.aggregate(
    conns, call("flowerTier2PinDS", hsym, up$token))
  if (verbose) {
    message("  Pinned: ", paste(unique(unlist(lapply(pin, `[[`, "pinned"))),
                                 collapse = ", "), ".")
  }

  results_dir <- file.path(tempdir(), "dsflower_results",
                           format(Sys.time(), "%Y%m%d_%H%M%S"))
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  app_dir <- .build_tier2_app(up$package, user_app_dir, n_features, results_dir,
                              n_clients, num_rounds)
  .ensure_client_framework("pytorch")

  # link.up owns the local SuperLink + DSI tunnel lifecycle; on.exit reverses it.
  ds.flower.link.up(conns)
  recipe <- structure(list(
    model = list(name = "tier2", template = "tier2", framework = "pytorch",
                 track = "egress"),
    strategy = list(name = "FedAvg", params = list()),
    num_rounds = as.integer(num_rounds),
    features = features, target_levels = public_target$levels,
    target_bounds = public_target$bounds, evaluation_only = FALSE),
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
                                verbose = TRUE) {
  .Deprecated("ds.flower.hook.run")
  ds.flower.hook.run(
    conns = conns, user_app_dir = user_app_dir, target = target,
    features = features, symbol = symbol, num_rounds = num_rounds,
    task = "classification", verbose = verbose)
}
