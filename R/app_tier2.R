# Module: Tier-2 federated run (researcher side). Upload an arbitrary training
# module, run it under the trusted node-side runner's output-perturbation DP gate.
# The researcher provides a Python package exposing:
#   initial_arrays(cfg, input_dim) -> list[np.ndarray]
#   local_update(global_arrays, X, y, cfg) -> list[np.ndarray]
# It trains however it likes; the trusted runner clips + noises the released
# update, so DP holds for ANY code. The app is sha256-verified + exfiltration
# -scanned + hash-pinned on the node before it can run.

#' @keywords internal
.tier2_skeleton_dir <- function() {
  p <- system.file("flower_app", "dsflower_tier2", package = "dsFlowerClient")
  if (nzchar(p) && dir.exists(p)) return(p)
  alt <- file.path("inst", "flower_app", "dsflower_tier2")
  if (dir.exists(alt)) return(normalizePath(alt))
  stop("Bundled Tier-2 runner (dsflower_tier2) not found.", call. = FALSE)
}

#' Upload the researcher's training package to the nodes (zip + chunked push +
#' verify + exfiltration scan + install). Returns the upload token + package name.
#' @keywords internal
.upload_user_module <- function(conns, user_pkg_dir, chunk_bytes = 262144L) {
  user_pkg_dir <- normalizePath(user_pkg_dir, mustWork = TRUE)
  pkg_name <- basename(user_pkg_dir)
  parent <- dirname(user_pkg_dir)
  zipfile <- tempfile(fileext = ".zip")
  wd <- getwd(); setwd(parent)
  utils::zip(zipfile, files = pkg_name, flags = "-q -r")
  setwd(wd)

  n <- file.size(zipfile)
  raw <- readBin(zipfile, "raw", n)
  sha <- digest::digest(file = zipfile, algo = "sha256")
  token <- paste0("usr", paste(sample(c(letters, 0:9), 12, replace = TRUE),
                                collapse = ""))
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
    stop("Tier-2 module install/scan failed on: ",
         paste(failed, collapse = ", "), ".", call. = FALSE)
  }
  list(token = token, sha256 = sha, package = pkg_name)
}

#' Build the Tier-2 runner app dir (bundled dsflower_tier2 + the user package +
#' pyproject). The user package is included so the researcher-side ServerApp can
#' call initial_arrays(); on the node it is still hash-verified by the integrity
#' hook against the node-computed upload hash (so it cannot be tampered with).
#' @keywords internal
.build_tier2_app <- function(user_module, user_pkg_dir, n_features, results_dir,
                             n_nodes, num_rounds) {
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
    '[tool.hatch.build.targets.wheel]\npackages = ["dsflower_tier2", "',
    user_module, '"]\n\n',
    '[tool.flwr.app]\npublisher = "dsflower"\n\n',
    '[tool.flwr.app.components]\n',
    'serverapp = "dsflower_tier2.server_app:app"\n',
    'clientapp = "dsflower_tier2.client_app:app"\n\n',
    '[tool.flwr.app.config]\n', paste(config_lines, collapse = "\n"), '\n'
  )
  writeLines(toml, file.path(app_dir, "pyproject.toml"))
  app_dir
}

#' Run a Tier-2 federated job: upload arbitrary training code, run it DP-gated
#'
#' @param conns DSI connections object.
#' @param user_app_dir Character; path to the researcher's training package dir
#'   (a folder with __init__.py exposing initial_arrays + local_update).
#' @param target Character; target column name.
#' @param features Character vector; feature column names.
#' @param symbol Character; server-side data handle symbol (default "D").
#' @param num_rounds Integer; federated rounds (default 1).
#' @param verbose Logical.
#' @return A \code{dsflower_run} object.
#' @export
ds.flower.tier2.run <- function(conns, user_app_dir, target, features,
                                symbol = "D", num_rounds = 1L,
                                verbose = TRUE) {
  .require_flwr_cli()
  if (length(features) < 1) {
    stop("Tier-2 requires explicit feature columns.", call. = FALSE)
  }
  n_clients <- length(conns)
  n_features <- length(features)

  # Initialise the server-side Flower handle (flowerInitDS) from the assigned
  # data symbol; all run ops use this handle symbol.
  flower <- ds.flower.connect(conns, symbol = symbol)
  conns <- flower$conns
  hsym <- flower$symbol

  up <- .upload_user_module(conns, user_app_dir)
  if (verbose) message("  Tier-2 module '", up$package, "' uploaded + scanned.")

  ds.flower.nodes.prepare(
    conns, hsym, target_column = target, feature_columns = features,
    run_config = list("task-type" = "classification"))

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

  ds.flower.link.up(conns)   # establish the DSI tunnel: self-contained (the neural path does
                             # this inside submit; nodes.ensure needs the tunnel present)
  started_superlink <- FALSE
  if (!isTRUE(ds.flower.superlink.status()$running)) {
    ds.flower.superlink.start()
    started_superlink <- TRUE
  }
  recipe <- structure(list(
    model = list(name = "tier2", template = "tier2", framework = "pytorch",
                 track = "egress"),
    strategy = list(name = "FedAvg", params = list()),
    num_rounds = as.integer(num_rounds),
    features = features, evaluation_only = FALSE),
    class = "dsflower_recipe")

  run <- tryCatch(
    {
      ds.flower.nodes.ensure(conns, hsym)
      ds.flower.run.start(recipe, conns, app_dir = app_dir,
                          results_dir = results_dir, symbol = hsym,
                          verbose = verbose)
    },
    error = function(e) {
      tryCatch(ds.flower.nodes.cleanup(conns, hsym), error = function(e) NULL)
      if (started_superlink) tryCatch(ds.flower.superlink.stop(), error = function(e) NULL)
      stop(e)
    }
  )

  tryCatch(ds.flower.nodes.cleanup(conns, hsym), error = function(e) NULL)
  if (started_superlink) tryCatch(ds.flower.superlink.stop(), error = function(e) NULL)
  tryCatch(DSI::datashield.aggregate(conns, call("flowerAppDeleteDS", up$token)),
           error = function(e) NULL)
  tryCatch(ds.flower.disconnect(flower), error = function(e) NULL)
  run
}
